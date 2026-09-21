"""Responders for the non-Q&A routes: chitchat, document summaries and comparisons.

* **Chitchat** is a short streamed reply from the fast model, with no retrieval.
* **Summaries** read one document's parent sections in order. A small document is
  summarised in a single call ("stuff"). A large one uses map-reduce: the fast model
  condenses batches of sections into cited bullet points in parallel (map), then the
  main model streams one summary from those partials (reduce). Passage markers survive
  both steps, so citations still point at real sections.
* **Comparisons** receive passages retrieved separately per document, numbered globally
  and grouped by document, and produce a side-by-side table.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from nexusrag.config import Settings
from nexusrag.generation.answer import (
    AnswerStyle,
    GeneratedAnswer,
    ResetCallback,
    TokenCallback,
    finalize,
    format_passages,
    stream_text,
)
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.llm.prompts import (
    ANSWER_STYLES,
    CHITCHAT_SYSTEM,
    COMPARE_SYSTEM,
    COMPARE_USER,
    SUMMARY_MAP_PROMPT,
    SUMMARY_REDUCE_USER,
    SUMMARY_SYSTEM,
    SUMMARY_USER,
    render,
)
from nexusrag.log import get_logger
from nexusrag.models import ChatTurn, ContextPassage, ParentSection
from nexusrag.retrieval.query_transform import format_history
from nexusrag.store.registry import DocumentRecord

log = get_logger(__name__)

_MAP_CONCURRENCY = 4


async def chitchat_reply(
    gemini: GeminiClient,
    message: str,
    *,
    history: Sequence[ChatTurn],
    titles: Sequence[str],
    on_token: TokenCallback | None = None,
    on_reset: ResetCallback | None = None,
) -> GeneratedAnswer:
    """Brief conversational reply; never answers factual questions from general knowledge."""
    system = render(CHITCHAT_SYSTEM, titles=", ".join(titles[:5]) or "(no documents yet)")
    context = format_history(history[-4:])
    prompt = f"{context}\nUser: {message}" if context else message
    text = await stream_text(
        gemini,
        prompt,
        system_instruction=system,
        role="fast",
        stage="chitchat",
        on_token=on_token,
        on_reset=on_reset,
    )
    return GeneratedAnswer(text=text)


def batch_passages(
    passages: Sequence[ContextPassage], max_tokens: int
) -> list[list[ContextPassage]]:
    """Consecutive groups within ``max_tokens`` each (an oversized passage stands alone)."""
    groups: list[list[ContextPassage]] = []
    current: list[ContextPassage] = []
    size = 0
    for passage in passages:
        tokens = passage.parent.token_count
        if current and size + tokens > max_tokens:
            groups.append(current)
            current, size = [], 0
        current.append(passage)
        size += tokens
    if current:
        groups.append(current)
    return groups


@dataclass
class SummaryResult:
    answer: GeneratedAnswer
    passages: list[ContextPassage]
    #: True when map-reduce was used (the final text is built from partial summaries).
    map_reduce: bool


class DocumentSummarizer:
    def __init__(self, gemini: GeminiClient, settings: Settings) -> None:
        self.gemini = gemini
        self.settings = settings

    async def summarize(
        self,
        question: str,
        document: DocumentRecord,
        parents: Sequence[ParentSection],
        *,
        style: AnswerStyle = "detailed",
        on_token: TokenCallback | None = None,
        on_reset: ResetCallback | None = None,
        extra_instructions: str | None = None,
    ) -> SummaryResult:
        """Summarise one document from its sections, in reading order."""
        passages = [ContextPassage(index=i, parent=p) for i, p in enumerate(parents, start=1)]
        total = sum(p.token_count for p in parents)
        map_reduce = total > self.settings.summary_stuff_tokens
        if not map_reduce:
            prompt = render(
                SUMMARY_USER,
                title=document.title,
                filename=document.filename,
                passages=format_passages(passages),
                question=question,
                style=ANSWER_STYLES[style],
            )
        else:
            groups = batch_passages(passages, self.settings.summary_map_batch_tokens)
            partials = await self._map(document, groups)
            prompt = render(
                SUMMARY_REDUCE_USER,
                title=document.title,
                filename=document.filename,
                partials="\n\n".join(f"Part {i}:\n{text}" for i, text in enumerate(partials, 1)),
                question=question,
                style=ANSWER_STYLES[style],
            )
        log.info("agent.summarize", sections=len(parents), tokens=total, map_reduce=map_reduce)
        raw = await stream_text(
            self.gemini,
            prompt,
            system_instruction=SUMMARY_SYSTEM + (extra_instructions or ""),
            stage="summarize.reduce" if map_reduce else "summarize",
            on_token=on_token,
            on_reset=on_reset,
        )
        return SummaryResult(
            answer=finalize(raw, passages), passages=passages, map_reduce=map_reduce
        )

    async def _map(
        self, document: DocumentRecord, groups: Sequence[Sequence[ContextPassage]]
    ) -> list[str]:
        slots = asyncio.Semaphore(_MAP_CONCURRENCY)

        async def one(group: Sequence[ContextPassage]) -> str:
            async with slots:
                result = await self.gemini.generate(
                    render(
                        SUMMARY_MAP_PROMPT, title=document.title, passages=format_passages(group)
                    ),
                    role="fast",
                    stage="summarize.map",
                )
                return result.text.strip()

        return list(await asyncio.gather(*(one(g) for g in groups)))


@dataclass
class ComparisonGroup:
    document: DocumentRecord
    passages: list[ContextPassage]


def format_comparison_context(groups: Sequence[ComparisonGroup]) -> str:
    blocks = []
    for group in groups:
        doc = group.document
        body = format_passages(group.passages) if group.passages else "(no relevant passages found)"
        title = doc.title.replace('"', "'")
        blocks.append(f'<document title="{title}" file="{doc.filename}">\n{body}\n</document>')
    return "\n\n".join(blocks)


class DocumentComparer:
    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def compare(
        self,
        question: str,
        groups: Sequence[ComparisonGroup],
        *,
        style: AnswerStyle = "detailed",
        on_token: TokenCallback | None = None,
        on_reset: ResetCallback | None = None,
        extra_instructions: str | None = None,
    ) -> GeneratedAnswer:
        """Side-by-side comparison from per-document passages (globally numbered)."""
        prompt = render(
            COMPARE_USER,
            documents=format_comparison_context(groups),
            question=question,
            style=ANSWER_STYLES[style],
        )
        raw = await stream_text(
            self.gemini,
            prompt,
            system_instruction=COMPARE_SYSTEM + (extra_instructions or ""),
            stage="compare",
            on_token=on_token,
            on_reset=on_reset,
        )
        return finalize(raw, [p for group in groups for p in group.passages])
