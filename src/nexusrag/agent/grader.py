"""Self-correction checks, both run with the fast model and structured output.

* **Relevance** (after retrieval): can these passages answer the question? If not, what's
  missing and what should we search for instead? This drives the rewrite-and-retry loop.
* **Groundedness** (after generation): is every factual claim in the answer supported by
  the passages? This drives the regenerate-or-refuse step.

Both checks *fail open*. If the grader call itself fails, the pipeline carries on as if
the check passed, because a grader outage shouldn't stop users getting grounded answers.
The answer prompt and citation mapping still enforce grounding on their own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from nexusrag.generation.answer import format_passages
from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.llm.prompts import GROUNDEDNESS_PROMPT, RELEVANCE_PROMPT, render
from nexusrag.log import get_logger
from nexusrag.models import ContextPassage
from nexusrag.utils.tokens import truncate_to_tokens

log = get_logger(__name__)

#: Per-passage cap when grading relevance: enough to judge, cheap to send.
RELEVANCE_PASSAGE_TOKENS = 400


class _RelevanceOutput(BaseModel):
    sufficient: bool
    missing: str = ""
    better_query: str = ""


class _GroundednessOutput(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


@dataclass
class RelevanceVerdict:
    sufficient: bool
    missing: str = ""
    better_query: str = ""
    #: False when the grader failed and the verdict is the fail-open default.
    checked: bool = True


@dataclass
class GroundednessVerdict:
    grounded: bool
    unsupported_claims: list[str] = field(default_factory=list)
    checked: bool = True


def _trimmed(passages: Sequence[ContextPassage], max_tokens: int) -> list[ContextPassage]:
    return [
        p.model_copy(
            update={
                "parent": p.parent.model_copy(
                    update={"text": truncate_to_tokens(p.parent.text, max_tokens)}
                )
            }
        )
        for p in passages
    ]


class RelevanceGrader:
    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def grade(self, question: str, passages: Sequence[ContextPassage]) -> RelevanceVerdict:
        """Judge whether ``passages`` can answer ``question``; suggest a better query if not."""
        context = (
            format_passages(_trimmed(passages, RELEVANCE_PASSAGE_TOKENS))
            if passages
            else "(no passages were found)"
        )
        try:
            out = await self.gemini.generate_structured(
                render(RELEVANCE_PROMPT, question=question, passages=context),
                _RelevanceOutput,
                role="fast",
                stage="grade_relevance",
            )
        except GeminiError as exc:
            log.warning("agent.relevance_grader_failed", error_type=type(exc).__name__)
            return RelevanceVerdict(sufficient=True, checked=False)
        verdict = RelevanceVerdict(
            sufficient=out.sufficient,
            missing=out.missing.strip(),
            better_query=out.better_query.strip(),
        )
        log.info(
            "agent.relevance", sufficient=verdict.sufficient, has_rewrite=bool(verdict.better_query)
        )
        return verdict


class GroundednessGrader:
    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def grade(self, answer: str, passages: Sequence[ContextPassage]) -> GroundednessVerdict:
        """Check every factual claim in ``answer`` against ``passages``."""
        try:
            out = await self.gemini.generate_structured(
                render(GROUNDEDNESS_PROMPT, passages=format_passages(passages), answer=answer),
                _GroundednessOutput,
                role="fast",
                stage="grade_groundedness",
            )
        except GeminiError as exc:
            log.warning("agent.groundedness_grader_failed", error_type=type(exc).__name__)
            return GroundednessVerdict(grounded=True, checked=False)
        claims = [c.strip() for c in out.unsupported_claims if c.strip()]
        # A "not grounded" verdict with no claim to point at gives the regeneration
        # prompt nothing to fix, so treat it as grounded.
        grounded = out.grounded or not claims
        log.info("agent.groundedness", grounded=grounded, unsupported=len(claims))
        return GroundednessVerdict(grounded=grounded, unsupported_claims=[] if grounded else claims)
