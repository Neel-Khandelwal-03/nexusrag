"""Query transformation: condense follow-ups, expand into variants, optionally HyDE.

* **Condensation.** "What about its warranty?" can't be searched on its own. With chat
  history, the fast model rewrites it into a standalone question ("What is the
  warranty on the Aurora X1?"). With no history this step is skipped and costs nothing.
* **Multi-query.** A single phrasing can miss passages that use different words.
  Paraphrases ("battery endurance", "flight time per charge") widen recall, and RRF
  merges their results.
* **HyDE** (Hypothetical Document Embeddings). The fast model drafts a passage that
  *would* answer the question. Its embedding sits near real answer passages, which
  helps when questions and documents are phrased very differently. It's off by
  default because it adds an LLM call to every query.

Every transform degrades gracefully: if the fast model fails, the pipeline carries on
with the original question rather than failing the request.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.llm.prompts import CONDENSE_PROMPT, HYDE_PROMPT, MULTI_QUERY_PROMPT, render
from nexusrag.log import get_logger
from nexusrag.models import ChatTurn
from nexusrag.retrieval.hybrid import SearchQuery

log = get_logger(__name__)

_HISTORY_CHARS = 1200  # per turn; long answers are truncated in the condensation prompt


class CondensedQuestion(BaseModel):
    standalone_question: str = Field(description="The rewritten, self-contained question")


class QueryVariants(BaseModel):
    queries: list[str] = Field(description="Alternative search queries")


@dataclass
class QueryPlan:
    """The queries retrieval will run, and how each was derived."""

    original: str
    standalone: str
    variants: list[str] = field(default_factory=list)
    hyde: str | None = None

    @property
    def queries(self) -> list[SearchQuery]:
        """All queries to search with; the standalone question always comes first."""
        kind = "standalone" if self.standalone != self.original else "original"
        out = [SearchQuery(self.standalone, kind)]  # type: ignore[arg-type]
        out += [SearchQuery(v, "variant") for v in self.variants]
        if self.hyde:
            out.append(SearchQuery(self.hyde, "hyde"))
        return out


def format_history(history: Sequence[ChatTurn]) -> str:
    lines = []
    for turn in history:
        content = turn.content.strip()
        if len(content) > _HISTORY_CHARS:
            content = content[:_HISTORY_CHARS] + " …"
        lines.append(f"{turn.role.capitalize()}: {content}")
    return "\n".join(lines)


def _clean_variants(variants: Sequence[str], question: str, n: int) -> list[str]:
    seen = {question.strip().lower()}
    out: list[str] = []
    for variant in variants:
        text = variant.strip().strip('"')
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:n]


class QueryTransformer:
    """Builds a :class:`QueryPlan` for a question using the fast model."""

    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def condense(self, question: str, history: Sequence[ChatTurn]) -> str:
        """Standalone version of ``question`` given ``history`` (unchanged if no history)."""
        if not history:
            return question
        try:
            result = await self.gemini.generate_structured(
                render(CONDENSE_PROMPT, history=format_history(history), question=question),
                CondensedQuestion,
                role="fast",
                stage="condense",
            )
        except GeminiError as exc:
            log.warning("query.condense_failed", error_type=type(exc).__name__)
            return question
        return result.standalone_question.strip() or question

    async def expand(self, question: str, n: int) -> list[str]:
        """Up to ``n`` paraphrases of ``question`` (excluding the question itself)."""
        if n <= 0:
            return []
        try:
            result = await self.gemini.generate_structured(
                render(MULTI_QUERY_PROMPT, n=n, question=question),
                QueryVariants,
                role="fast",
                stage="multi_query",
            )
        except GeminiError as exc:
            log.warning("query.multi_query_failed", error_type=type(exc).__name__)
            return []
        return _clean_variants(result.queries, question, n)

    async def hypothetical_document(self, question: str) -> str | None:
        """A made-up answer passage for HyDE, or None if generation fails."""
        try:
            result = await self.gemini.generate(
                render(HYDE_PROMPT, question=question),
                role="fast",
                stage="hyde",
                max_output_tokens=1024,
            )
        except GeminiError as exc:
            log.warning("query.hyde_failed", error_type=type(exc).__name__)
            return None
        return result.text.strip() or None

    async def plan(
        self,
        question: str,
        history: Sequence[ChatTurn] = (),
        *,
        multi_query: bool = False,
        num_variants: int = 3,
        hyde: bool = False,
    ) -> QueryPlan:
        """Condense first (variants must build on the standalone question), then expand."""
        standalone = await self.condense(question, history)
        variants, passage = await asyncio.gather(
            self.expand(standalone, num_variants) if multi_query else _none_list(),
            self.hypothetical_document(standalone) if hyde else _none(),
        )
        plan = QueryPlan(original=question, standalone=standalone, variants=variants, hyde=passage)
        log.info(
            "query.plan",
            condensed=standalone != question,
            variants=len(variants),
            hyde=passage is not None,
        )
        return plan


async def _none_list() -> list[str]:
    return []


async def _none() -> None:
    return None
