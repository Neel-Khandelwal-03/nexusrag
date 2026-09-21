"""Grounded answer generation with streaming and inline citations.

The prompt gives the model numbered ``<passage>`` blocks, each labelled with its
source, and rules: answer only from the passages, cite every factual sentence, admit
gaps, and treat passage content as untrusted data (a basic prompt-injection guard).
Tokens are streamed to the caller as they arrive. Citations are resolved once the
full answer is known.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from nexusrag.generation.citations import build_citations, strip_invalid_markers
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.llm.prompts import (
    ANSWER_STYLES,
    ANSWER_SYSTEM,
    ANSWER_USER,
    PASSAGE_TEMPLATE,
    REFUSAL_MESSAGE,
    render,
)
from nexusrag.models import Citation, ContextPassage

AnswerStyle = Literal["concise", "detailed"]
TokenCallback = Callable[[str], Awaitable[None]]
ResetCallback = Callable[[], Awaitable[None]]

_QUOTES = "\"'“”‘’"


@dataclass
class GeneratedAnswer:
    """The final answer text and the citations it uses."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    #: True if the text differs from what was streamed (e.g. invalid citations removed).
    edited: bool = False


def passage_source(passage: ContextPassage) -> str:
    """Label shown to the model for a passage: file, pages and section."""
    p = passage.parent
    parts = [p.filename]
    if p.page_start is not None:
        pages = (
            f"{p.page_start}"
            if p.page_end in (None, p.page_start)
            else f"{p.page_start}-{p.page_end}"
        )
        parts.append(f"p. {pages}")
    if p.section_path:
        parts.append(p.section_path)
    return " | ".join(parts).replace('"', "'")


def format_passages(passages: Sequence[ContextPassage]) -> str:
    return "\n\n".join(
        render(PASSAGE_TEMPLATE, index=p.index, source=passage_source(p), text=p.parent.text)
        for p in passages
    )


def build_answer_prompt(
    question: str, passages: Sequence[ContextPassage], style: AnswerStyle = "detailed"
) -> str:
    """User prompt with the numbered context passages and the question."""
    return render(
        ANSWER_USER,
        passages=format_passages(passages),
        style=ANSWER_STYLES[style],
        question=question.strip(),
    )


def answer_system_instruction() -> str:
    return render(ANSWER_SYSTEM, refusal=REFUSAL_MESSAGE)


def is_refusal(text: str) -> bool:
    """Whether the answer opens with the refusal phrase (quotes and case ignored)."""

    def normalise(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().strip(_QUOTES).replace("’", "'")).lower()

    return normalise(text).startswith(normalise(REFUSAL_MESSAGE).rstrip("."))


class AnswerGenerator:
    """Streams a grounded, cited answer from the main model."""

    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def generate(
        self,
        question: str,
        passages: Sequence[ContextPassage],
        *,
        style: AnswerStyle = "detailed",
        on_token: TokenCallback | None = None,
        on_reset: ResetCallback | None = None,
    ) -> GeneratedAnswer:
        """Answer ``question`` from ``passages``, streaming tokens to ``on_token``.

        If the model fails mid-answer, generation restarts once on the fallback model.
        ``on_reset`` is awaited first so the caller can clear what was already displayed.
        Without a reset hook, a streaming caller can't un-show text, so no restart happens.
        """
        if not passages:
            # Nothing retrieved: refuse without spending an LLM call.
            if on_token is not None:
                await on_token(REFUSAL_MESSAGE)
            return GeneratedAnswer(text=REFUSAL_MESSAGE, refused=True)

        parts: list[str] = []

        async def restart() -> None:
            parts.clear()
            if on_reset is not None:
                await on_reset()

        can_restart = on_token is None or on_reset is not None
        async for delta in self.gemini.stream(
            build_answer_prompt(question, passages, style),
            role="main",
            system_instruction=answer_system_instruction(),
            stage="answer",
            on_restart=restart if can_restart else None,
        ):
            parts.append(delta)
            if on_token is not None:
                await on_token(delta)

        raw = "".join(parts).strip()
        text = strip_invalid_markers(raw, {p.index for p in passages})
        return GeneratedAnswer(
            text=text,
            citations=build_citations(text, passages),
            refused=is_refusal(text),
            edited=text != raw,
        )
