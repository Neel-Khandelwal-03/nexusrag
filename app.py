"""Chainlit entry point for NexusRAG.

Run with:  chainlit run app.py

Phase 3 scope: a minimal chat that streams grounded answers and shows each cited
source in a side panel. Uploads, settings, profiles, visible pipeline steps,
authentication and persistent history arrive in phase 6.
"""

from __future__ import annotations

from typing import Any

import chainlit as cl

from nexusrag.config import get_settings
from nexusrag.llm.gemini_client import GeminiError
from nexusrag.log import configure_logging, get_logger
from nexusrag.models import ChatTurn
from nexusrag.service import RAGService
from nexusrag.ui.render import (
    citation_element_name,
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
    try:
        result = await service.ask(message.content, history=history, on_token=reply.stream_token)
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
    if answer.citations:
        reply.content += "\n\n" + sources_footer(answer.citations)
        # Chainlit types `elements` with a TypeVar, so annotate the list explicitly.
        elements: list[Any] = [
            cl.Text(
                name=citation_element_name(citation),
                content=source_panel_markdown(citation),
                display="side",
            )
            for citation in answer.citations
        ]
        reply.elements = elements
    await reply.send()
