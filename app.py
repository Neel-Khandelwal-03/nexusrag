"""Chainlit entry point for NexusRAG.

Run with:  chainlit run app.py

The chat UI over the self-correcting RAG agent:

* password login (``AUTH_USERNAME`` / ``AUTH_PASSWORD``) and chat history that survives
  restarts (SQLite data layer), with thumbs up/down feedback on every answer;
* drag-and-drop uploads with per-file progress, and web pages added by URL;
* a settings panel (knowledge base, document filter, top-k, retrieval toggles, answer
  style) and three chat profiles: Q&A, Summarize and Compare;
* every agent node shown as a timed step, cited passages in side panels (PDFs open at
  the cited page), suggested follow-up questions and a ``/stats`` command.

Uploads and messages are rate-limited per user. Logs never include document text,
questions, answers or secrets (see ``nexusrag.log``).
"""

from __future__ import annotations

import asyncio
import functools
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ParamSpec, cast

import chainlit as cl
from chainlit.types import CommandDict, ThreadDict
from chainlit.utils import utc_now
from pydantic import ValidationError

from nexusrag.config import get_settings
from nexusrag.generation.follow_ups import FollowUpSuggester
from nexusrag.ingestion.loaders import SUPPORTED_EXTENSIONS
from nexusrag.ingestion.pipeline import IngestionPipeline, IngestResult, ProgressEvent
from nexusrag.llm.gemini_client import GeminiError
from nexusrag.log import configure_logging, get_logger
from nexusrag.models import AgentStep, Answer, ChatTurn, Citation, Route, SourceType
from nexusrag.service import RAGService
from nexusrag.store.registry import InvalidCollectionName, validate_collection_name
from nexusrag.ui.auth import check_credentials, ensure_auth_secret, password_configured
from nexusrag.ui.data_layer import build_data_layer
from nexusrag.ui.panel import settings_widgets
from nexusrag.ui.render import (
    cache_note,
    citation_element_name,
    closest_matches_footer,
    match_element_name,
    pdf_element_name,
    progress_markdown,
    setup_error_markdown,
    source_panel_markdown,
    sources_footer,
    starter_prompts,
    stats_markdown,
    step_markdown,
    step_title,
    step_type,
    welcome_markdown,
)
from nexusrag.ui.state import (
    W_NEW_COLLECTION,
    RateLimiter,
    UISettings,
    history_from_thread,
    uploaded_doc_ids,
)

settings = get_settings()
configure_logging(settings)
log = get_logger("nexusrag.app")

# Chainlit reads the token-signing secret after importing this module, so setting it
# here works. Raises in staging/production when it's missing.
if ensure_auth_secret(settings, os.environ):
    log.warning("auth.secret_generated")
if not password_configured(settings):
    log.warning("auth.no_password", deployed=settings.is_deployed)

#: Chat profile -> forced agent route (None lets the router decide).
PROFILE_MODES: dict[str, Route | None] = {
    "Q&A": None,
    "Summarize": Route.SUMMARIZE,
    "Compare": Route.COMPARE,
}
PROFILE_STARTERS = {"Q&A": "qa", "Summarize": "summarize", "Compare": "compare"}
PROFILE_HINTS = {
    "Summarize": "**Summarize mode**: name a document and get a structured summary of it.",
    "Compare": "**Compare mode**: name two to four documents and what to compare them on.",
}
COMMANDS: list[CommandDict] = [
    {
        "id": "stats",
        "icon": "database",
        "description": "Show knowledge base statistics",
        "button": False,
        "persistent": False,
        "selected": False,
    },
    {
        "id": "clear-cache",
        "icon": "eraser",
        "description": "Forget cached answers for this knowledge base",
        "button": False,
        "persistent": False,
        "selected": False,
    },
    {
        "id": "url",
        "icon": "globe",
        "description": "Index a web page: type its URL after the command",
        "button": False,
        "persistent": False,
        "selected": False,
    },
]
#: PDF previews per answer (each one sends the whole file to the browser).
MAX_PDF_PREVIEWS = 4

message_limiter = RateLimiter(settings.rate_limit_messages_per_minute, 60)
upload_limiter = RateLimiter(settings.rate_limit_uploads_per_hour, 3600)

_service: RAGService | None = None
_pipeline: IngestionPipeline | None = None
_suggester: FollowUpSuggester | None = None
_warm_up_task: asyncio.Task[None] | None = None


def get_service() -> RAGService:
    """Process-wide service, created on first use (so a missing key shows in the chat)."""
    global _service
    if _service is None:
        _service = RAGService.create(settings)
    return _service


def get_pipeline(service: RAGService) -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(settings, service.gemini, service.stores)
    return _pipeline


def get_suggester(service: RAGService) -> FollowUpSuggester:
    global _suggester
    if _suggester is None:
        _suggester = FollowUpSuggester(service.gemini)
    return _suggester


async def service_or_error() -> RAGService | None:
    try:
        return get_service()
    except GeminiError as exc:
        await cl.Message(content=setup_error_markdown(exc.user_message)).send()
        return None


# --------------------------------------------------------------------------- session state


def current_ui() -> UISettings:
    raw = cl.user_session.get("ui")
    if isinstance(raw, dict):
        try:
            return UISettings.model_validate(raw)
        except ValidationError:
            pass
    return UISettings.defaults(settings)


def save_ui(ui: UISettings) -> None:
    # Stored as a plain dict: the session is persisted as JSON for resumed chats.
    cl.user_session.set("ui", ui.model_dump())


def user_key() -> str:
    user = cl.user_session.get("user")
    identifier = getattr(user, "identifier", None)
    return str(identifier or cl.user_session.get("id") or "anonymous")


def current_profile() -> str:
    profile = cl.user_session.get("chat_profile")
    return profile if profile in PROFILE_MODES else "Q&A"


async def send_panel(service: RAGService, ui: UISettings) -> None:
    registry = service.stores.registry
    collections = [c.name for c in registry.list_collections()] + [settings.default_collection]
    filenames = [d.filename for d in registry.list_documents(ui.collection)]
    await cl.ChatSettings(settings_widgets(ui, collections, filenames)).send()


async def clear_follow_ups() -> None:
    """Remove the previous answer's follow-up buttons (they're stale after a new question)."""
    for action in cl.user_session.get("follow_ups") or []:
        if isinstance(action, cl.Action):
            await action.remove()
    cl.user_session.set("follow_ups", [])


async def name_thread_once(name: str) -> None:
    """Name the chat after its first question.

    Chainlit names a thread after its first interaction, which for a button click is
    the action name ("ask"). Called after the action's work is done, so it can't race
    Chainlit's own naming.
    """
    if cl.user_session.get("thread_named"):
        return
    cl.user_session.set("thread_named", True)
    await cl.context.emitter.init_thread(name[:80])


P = ParamSpec("P")


def friendly_errors(func: Callable[P, Awaitable[None]]) -> Callable[P, Awaitable[None]]:
    """Log unexpected errors and show a short apology instead of the raw exception."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            await func(*args, **kwargs)
        except Exception:
            log.exception("app.callback_failed", callback=func.__name__)
            await cl.Message(content="⚠️ Something went wrong. Please try again.").send()

    return wrapper


# --------------------------------------------------------------------------- lifecycle


@cl.data_layer
def data_layer() -> Any:
    return build_data_layer(settings)


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    if check_credentials(settings, username, password):
        return cl.User(identifier=settings.auth_username, metadata={"provider": "credentials"})
    log.warning("auth.failed")
    return None


@cl.on_app_startup
async def on_app_startup() -> None:
    """Load the cross-encoder in the background, so the first question isn't slow."""
    global _warm_up_task
    try:
        service = get_service()
    except GeminiError:
        return  # the chat explains the missing key
    _warm_up_task = asyncio.create_task(asyncio.to_thread(service.warm_up))


@cl.set_chat_profiles
async def chat_profiles(user: cl.User | None, language: str | None = None) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name="Q&A",
            markdown_description="Ask questions about your documents; every answer cites "
            "the exact passages.",
            default=True,
        ),
        cl.ChatProfile(
            name="Summarize",
            markdown_description="Get a structured summary of a whole document.",
        ),
        cl.ChatProfile(
            name="Compare",
            markdown_description="Compare two to four documents side by side.",
        ),
    ]


async def send_welcome(service: RAGService, ui: UISettings) -> None:
    documents = service.stores.registry.list_documents(ui.collection)
    profile = current_profile()
    content = welcome_markdown(ui.collection, documents)
    if hint := PROFILE_HINTS.get(profile):
        content += f"\n\n{hint}"
    actions = [cl.Action(name="add_url", payload={}, label="Add a web page", icon="globe")]
    actions += [
        cl.Action(name="ask", payload={"question": message}, label=label, icon="message-circle")
        for label, message in starter_prompts(PROFILE_STARTERS[profile], documents)
    ]
    await cl.Message(content=content, actions=actions).send()


@cl.on_chat_start
@friendly_errors
async def on_chat_start() -> None:
    service = await service_or_error()
    if service is None:
        return
    ui = UISettings.defaults(settings)
    save_ui(ui)
    cl.user_session.set("history", [])
    await cl.context.emitter.set_commands(COMMANDS)
    await send_panel(service, ui)
    await send_welcome(service, ui)


@cl.on_chat_resume
@friendly_errors
async def on_chat_resume(thread: ThreadDict) -> None:
    service = await service_or_error()
    if service is None:
        return
    ui = current_ui()  # restored from the thread's saved session
    save_ui(ui)
    cl.user_session.set("thread_named", True)
    steps = cast(list[dict[str, Any]], thread.get("steps") or [])
    cl.user_session.set("history", history_from_thread(steps, settings.history_turns))
    await cl.context.emitter.set_commands(COMMANDS)
    await send_panel(service, ui)


@cl.on_settings_update
@friendly_errors
async def on_settings_update(values: dict[str, Any]) -> None:
    service = await service_or_error()
    if service is None:
        return
    old = current_ui()
    try:
        ui = old.updated(values)
    except ValidationError:
        await cl.Message(content="⚠️ Those settings aren't valid; nothing was changed.").send()
        return

    new_name = str(values.get(W_NEW_COLLECTION) or "").strip()
    if new_name:
        try:
            name = validate_collection_name(new_name)
        except InvalidCollectionName as exc:
            await cl.Message(content=f"⚠️ {exc}").send()
        else:
            service.stores.registry.ensure_collection(name)
            ui = ui.model_copy(update={"collection": name})

    switched = ui.collection != old.collection
    if switched:
        ui = ui.model_copy(update={"documents": []})  # the filter belonged to the old KB
        cl.user_session.set("recent_uploads", [])
    save_ui(ui)
    if switched or new_name:
        # Refresh the panel: the new KB's documents, and clear the "new KB" field.
        await send_panel(service, ui)
    if switched:
        count = len(service.stores.registry.list_documents(ui.collection))
        await cl.Message(
            content=f"📚 Switched to knowledge base **{ui.collection}** ({count} documents)."
        ).send()
    log.info("ui.settings_updated", collection=ui.collection, switched=switched)


# --------------------------------------------------------------------------- messages


@cl.on_message
@friendly_errors
async def on_message(message: cl.Message) -> None:
    service = await service_or_error()
    if service is None:
        return
    cl.user_session.set("thread_named", True)  # Chainlit names it after this message
    text = message.content.strip()
    command = message.command or ""
    if command == "stats" or text.lower() == "/stats":
        await show_stats(service)
        return
    if command == "clear-cache" or text.lower() == "/clear-cache":
        await clear_cache(service)
        return
    if command == "url" or text.lower().startswith("/url"):
        await ingest_url(service, text.removeprefix("/url").strip())
        return

    files = [e for e in message.elements or [] if getattr(e, "path", None)]
    if files:
        await ingest_uploads(service, files)
    if text:
        await answer_question(service, text)


@cl.action_callback("ask")
@friendly_errors
async def on_ask_action(action: cl.Action) -> None:
    """Starter and follow-up buttons: ask the question as if the user had typed it."""
    question = str(action.payload.get("question") or "").strip()
    service = await service_or_error()
    if service is None or not question:
        return
    await cl.Message(content=question, type="user_message", author=user_key()).send()
    # Chainlit offers thumbs up/down only on answers inside a run step, which it opens
    # for typed messages but not for button clicks: open the same kind of run here.
    async with cl.Step(name="on_message", type="run"):
        await answer_question(service, question)
    await name_thread_once(question)


@cl.action_callback("add_url")
@friendly_errors
async def on_add_url_action(action: cl.Action) -> None:
    reply = await cl.AskUserMessage(
        content="Paste the address of the web page to index (http or https):", timeout=180
    ).send()
    service = await service_or_error()
    if service is None or not reply:
        return
    url = str(reply.get("output") or "").strip()
    await ingest_url(service, url)
    await name_thread_once(f"Add web page: {url}")


def _took(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


class StepTracker:
    """Shows the agent's nodes as timed Chainlit steps: running at start, filled in at the end.

    They're nested under one "Pipeline" step, created before the answer starts streaming,
    so nodes that run after the answer (the groundedness check) still appear above it.
    """

    def __init__(self) -> None:
        self._parent: cl.Step | None = None
        self._started = 0.0
        self._open: cl.Step | None = None
        self._node = ""

    async def _ensure_parent(self) -> cl.Step:
        if self._parent is None:
            # Messages nest under Chainlit's current run step (e.g. "on_message")
            # automatically; manually sent steps don't, so attach it explicitly to keep
            # it next to the answer.
            run = cl.context.current_step
            self._parent = cl.Step(
                name="Pipeline",
                type="tool",
                parent_id=run.id if run else None,
                show_input=False,
            )
            self._parent.start = utc_now()
            self._started = time.perf_counter()
            await self._parent.send()
        return self._parent

    async def _new(self, node: str) -> cl.Step:
        parent = await self._ensure_parent()
        step = cl.Step(
            name=step_title(node),
            type=cast(Any, step_type(node)),
            parent_id=parent.id,
            show_input=False,
        )
        step.start = utc_now()
        return step

    async def start(self, node: str) -> None:
        self._open, self._node = await self._new(node), node
        await self._open.send()

    async def finish(self, agent_step: AgentStep) -> None:
        step, sent = self._open, True
        if step is None or self._node != agent_step.node:
            step, sent = await self._new(agent_step.node), False
        step.name = f"{step_title(agent_step.node)} · {_took(agent_step.ms)}"
        step.output = step_markdown(agent_step)
        step.end = utc_now()
        await (step.update() if sent else step.send())
        self._open = None

    async def done(self, *, failed: bool = False) -> None:
        if self._open is not None:
            self._open.is_error = True
            self._open.output = "Failed"
            self._open.end = utc_now()
            await self._open.update()
            self._open = None
        if self._parent is not None:
            took = _took((time.perf_counter() - self._started) * 1000)
            self._parent.name = f"Pipeline · {took}" + (" (failed)" if failed else "")
            self._parent.is_error = failed
            self._parent.end = utc_now()
            await self._parent.update()


def pdf_previews(citations: list[Citation], collection: str) -> list[tuple[int, Any]]:
    """``(citation index, cl.Pdf)`` for cited PDFs whose original file was kept."""
    previews: list[tuple[int, Any]] = []
    for c in citations:
        if c.source_type != SourceType.PDF or len(previews) >= MAX_PDF_PREVIEWS:
            continue
        path = settings.source_file_path(collection, c.doc_id)
        if path.is_file():
            pdf = cl.Pdf(
                name=pdf_element_name(c), path=str(path), page=c.page_start or 1, display="side"
            )
            previews.append((c.index, pdf))
    return previews


def compose_reply(answer: Answer, collection: str) -> tuple[str, list[Any]]:
    """The final message text (answer + footers) and its side-panel elements."""
    content = answer.text
    elements: list[Any] = []
    if answer.citations:
        previews = pdf_previews(answer.citations, collection)
        content += "\n\n" + sources_footer(answer.citations, with_pdf=[i for i, _ in previews])
        elements += [
            cl.Text(name=citation_element_name(c), content=source_panel_markdown(c), display="side")
            for c in answer.citations
        ]
        elements += [pdf for _, pdf in previews]
    if answer.cached:
        content += "\n\n" + cache_note(answer.cache_similarity)
    if answer.closest_matches:
        content += "\n\n" + closest_matches_footer(answer.closest_matches)
        elements += [
            cl.Text(name=match_element_name(m), content=source_panel_markdown(m), display="side")
            for m in answer.closest_matches
        ]
    return content, elements


async def answer_question(service: RAGService, question: str) -> None:
    key = user_key()
    if not message_limiter.allow(key):
        wait = message_limiter.retry_after(key)
        await cl.Message(
            content=f"⏳ You're sending messages quickly. Try again in {wait:.0f} s."
        ).send()
        return
    await clear_follow_ups()

    ui = current_ui()
    history: list[ChatTurn] = [
        t for t in cl.user_session.get("history") or [] if isinstance(t, ChatTurn)
    ]
    records = service.stores.registry.list_documents(ui.collection)
    reply = cl.Message(content="")
    steps = StepTracker()

    async def reset_reply() -> None:
        # The agent is regenerating (model failure or failed groundedness check): clear
        # the draft the user already saw.
        reply.content = ""
        await reply.update()

    try:
        result = await service.ask(
            question,
            collection=ui.collection,
            history=history,
            filters=ui.filters({r.filename: r.doc_id for r in records}),
            options=ui.retrieval_options(settings),
            style=ui.style,
            mode=PROFILE_MODES[current_profile()],
            self_correct=ui.self_correct,
            recent_doc_ids=cl.user_session.get("recent_uploads") or [],
            use_cache=ui.cache,
            on_token=reply.stream_token,
            on_reset=reset_reply,
            on_step=steps.finish,
            on_node_start=steps.start,
        )
    except GeminiError as exc:
        await steps.done(failed=True)
        reply.content = f"⚠️ {exc.user_message}"
        await reply.send()
        return
    except Exception:
        # Never show a stack trace in the UI; the full error is in the logs.
        log.exception("app.unhandled_error")
        await steps.done(failed=True)
        reply.content = "⚠️ Something went wrong while answering. Please try again."
        await reply.send()
        return

    await steps.done()
    answer = result.answer
    history += [
        ChatTurn(role="user", content=question),
        ChatTurn(role="assistant", content=answer.text),
    ]
    # HISTORY_TURNS counts messages; note history[-0:] would keep everything.
    keep = settings.history_turns
    cl.user_session.set("history", history[-keep:] if keep else [])

    reply.content, reply.elements = compose_reply(answer, ui.collection)
    await reply.send()

    if settings.enable_follow_ups:
        suggestions = await get_suggester(service).suggest(question, answer)
        if suggestions:
            reply.actions = [
                cl.Action(name="ask", payload={"question": q}, label=q, icon="corner-down-right")
                for q in suggestions
            ]
            await reply.update()
            cl.user_session.set("follow_ups", reply.actions)


async def clear_cache(service: RAGService) -> None:
    collection = current_ui().collection
    removed = await asyncio.to_thread(service.clear_cache, collection)
    await cl.Message(
        content=f"🧹 Cleared {removed} cached answer{'s' if removed != 1 else ''} "
        f"for knowledge base **{collection}**."
    ).send()


async def show_stats(service: RAGService) -> None:
    stats = await asyncio.to_thread(service.stats, current_ui().collection)
    await cl.Message(content=stats_markdown(stats)).send()


# --------------------------------------------------------------------------- ingestion


def _progress_updater(status: cl.Message) -> Any:
    async def update(event: ProgressEvent) -> None:
        status.content = progress_markdown(event)
        await status.update()

    return update


def _upload_problem(name: str, path: Path) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return f"unsupported file type ({suffix or 'none'}); supported: {supported}"
    if path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
        return f"larger than the {settings.max_upload_mb} MB limit"
    return None


async def _after_ingest(service: RAGService, results: list[IngestResult]) -> None:
    if doc_ids := uploaded_doc_ids(results):
        # "What is this file about?" now refers to these documents.
        cl.user_session.set("recent_uploads", doc_ids)
    if any(r.status in ("indexed", "updated") for r in results):
        await send_panel(service, current_ui())  # new documents appear in the filter


async def ingest_uploads(service: RAGService, files: list[Any]) -> None:
    key = user_key()
    if len(files) > settings.max_upload_files:
        await cl.Message(
            content=f"⚠️ Upload at most {settings.max_upload_files} files at a time "
            f"(got {len(files)})."
        ).send()
        return
    if not upload_limiter.allow(key, cost=len(files)):
        wait = upload_limiter.retry_after(key) / 60
        await cl.Message(
            content=f"⏳ Upload limit reached ({settings.rate_limit_uploads_per_hour} files "
            f"per hour). Try again in about {max(1, round(wait))} min."
        ).send()
        return

    collection = current_ui().collection
    pipeline = get_pipeline(service)
    results: list[IngestResult] = []
    for element in files:
        name, path = str(element.name), Path(element.path)
        status = cl.Message(content=f"⏳ **{name}**: queued…")
        await status.send()
        if problem := _upload_problem(name, path):
            status.content = f"❌ **{name}**: {problem}"
            await status.update()
            continue
        results.append(
            await pipeline.ingest_file(
                path, collection=collection, filename=name, progress=_progress_updater(status)
            )
        )
    await _after_ingest(service, results)


async def ingest_url(service: RAGService, url: str) -> None:
    if not url.startswith(("http://", "https://")):
        await cl.Message(content="⚠️ Give a full web address starting with https://").send()
        return
    key = user_key()
    if not upload_limiter.allow(key):
        wait = upload_limiter.retry_after(key) / 60
        await cl.Message(
            content=f"⏳ Upload limit reached. Try again in about {max(1, round(wait))} min."
        ).send()
        return
    status = cl.Message(content=f"⏳ **{url}**: fetching…")
    await status.send()
    result = await get_pipeline(service).ingest_url(
        url, collection=current_ui().collection, progress=_progress_updater(status)
    )
    await _after_ingest(service, [result])
