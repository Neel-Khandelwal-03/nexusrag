"""Suggested follow-up questions, shown as clickable buttons under an answer.

One fast-model call with structured output. Suggestions are only offered after grounded
answers from the documents (not after refusals or chitchat), and a failure just means no
buttons.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.llm.prompts import FOLLOW_UP_PROMPT, render
from nexusrag.log import get_logger
from nexusrag.models import Answer, Route
from nexusrag.utils.tokens import truncate_to_tokens

log = get_logger(__name__)

#: Routes whose answers can have meaningful follow-ups.
FOLLOW_UP_ROUTES = frozenset({Route.DOC_QA, Route.SUMMARIZE, Route.COMPARE})
MAX_SUGGESTION_CHARS = 120


class _FollowUps(BaseModel):
    questions: list[str] = Field(default_factory=list)


def should_suggest(answer: Answer) -> bool:
    """Only after a real, cited answer from the documents."""
    return answer.route in FOLLOW_UP_ROUTES and not answer.refused and bool(answer.citations)


def clean_suggestions(questions: Sequence[str], original: str, n: int) -> list[str]:
    """Deduplicate, drop the original question and over-long suggestions, cap at ``n``."""
    seen = {original.strip().lower().rstrip("?")}
    out: list[str] = []
    for q in questions:
        text = " ".join(q.split()).strip().strip('"')
        key = text.lower().rstrip("?")
        if text and key not in seen and len(text) <= MAX_SUGGESTION_CHARS:
            seen.add(key)
            out.append(text if text.endswith("?") else f"{text}?")
    return out[:n]


class FollowUpSuggester:
    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def suggest(self, question: str, answer: Answer, n: int = 3) -> list[str]:
        if not should_suggest(answer):
            return []
        documents = ", ".join(dict.fromkeys(c.filename for c in answer.citations))
        prompt = render(
            FOLLOW_UP_PROMPT,
            n=n,
            documents=documents,
            question=question,
            answer=truncate_to_tokens(answer.text, 800),
        )
        try:
            result = await self.gemini.generate_structured(
                prompt, _FollowUps, role="fast", stage="follow_ups"
            )
        except GeminiError as exc:
            log.warning("follow_ups.failed", error_type=type(exc).__name__)
            return []
        return clean_suggestions(result.questions, question, n)
