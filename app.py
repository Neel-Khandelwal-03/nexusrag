"""Chainlit entry point for NexusRAG.

Run with:  chainlit run app.py

A minimal chat over the self-correcting agent. It streams grounded answers, shows each
cited source (or the closest matches, when refusing) in a side panel, and lists the
agent's steps. Uploads, settings, profiles, rich pipeline steps, authentication and
persistent history arrive in phase 6.
"""

from __future__ import annotations

from typing import Any

import chainlit as cl

from nexusrag.config import get_settings
from nexusrag.llm.gemini_client import GeminiError
from nexusrag.log import configure_logging, get_logger
from nexusrag.models import AgentStep, ChatTurn
from nexusrag.service import RAGService
from nexusrag.ui.render import (
    citation_element_name,
    closest_matches_footer,
    match_element_name,
    setup_error_markdown,
    source_panel_markdown,
    sources_footer,
    welcome_markdown,
)

settings = get_settings()
configure_logging(settings)
log = get_logger("nexusrag.app")

_service: RAGService | None = None


def get_service() -> RAGService:
    """Process-wide service, created on first use (so a missing key shows in the chat)."""
    global _service
    if _service is None:
        _service = RAGService.create(settings)
    return _service


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        service = get_service()
    except GeminiError as exc:
        await cl.Message(content=setup_error_markdown(exc.user_message)).send()
        return
    collection = settings.default_collection
    documents = service.stores.registry.list_documents(collection)
    await cl.Message(content=welcome_markdown(collection, documents)).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        service = get_service()
    except GeminiError as exc:
        await cl.Message(content=setup_error_markdown(exc.user_message)).send()
        return

    # Previous turns let the retriever rewrite follow-ups ("and its warranty?") into
    # standalone questions. Only answer text is kept, without the sources footer.
    history: list[ChatTurn] = cl.user_session.get("history") or []
    reply = cl.Message(content="")

    async def reset_reply() -> None:
        # The agent is regenerating (model failure or failed groundedness check): clear
        # the draft the user already saw.
        reply.content = ""
        await reply.update()

    async def show_step(step: AgentStep) -> None:
        # Minimal pipeline trace; phase 6 replaces this with timed, nested steps.
        async with cl.Step(name=step.node, type="tool") as ui_step:
            ui_step.output = f"{step.label} ({step.ms:.0f} ms)"

    try:
        result = await service.ask(
            message.content,
            history=history,
            on_token=reply.stream_token,
            on_reset=reset_reply,
            on_step=show_step,
        )
    except GeminiError as exc:
        reply.content = f"⚠️ {exc.user_message}"
        await reply.send()
        return
    except Exception:
        # Never show a stack trace in the UI; the full error is in the logs.
        log.exception("app.unhandled_error")
        reply.content = "⚠️ Something went wrong while answering. Please try again."
        await reply.send()
        return

    answer = result.answer
    history += [
        ChatTurn(role="user", content=message.content),
        ChatTurn(role="assistant", content=answer.text),
    ]
    # HISTORY_TURNS counts messages; note history[-0:] would keep everything.
    keep = settings.history_turns
    cl.user_session.set("history", history[-keep:] if keep else [])

    reply.content = answer.text
    # Chainlit types `elements` with a TypeVar, so annotate the list explicitly.
    elements: list[Any] = []
    if answer.citations:
        reply.content += "\n\n" + sources_footer(answer.citations)
        elements += [
            cl.Text(
                name=citation_element_name(citation),
                content=source_panel_markdown(citation),
                display="side",
            )
            for citation in answer.citations
        ]
    if answer.closest_matches:
        reply.content += "\n\n" + closest_matches_footer(answer.closest_matches)
        elements += [
            cl.Text(
                name=match_element_name(match),
                content=source_panel_markdown(match),
                display="side",
            )
            for match in answer.closest_matches
        ]
    reply.elements = elements
    await reply.send()
