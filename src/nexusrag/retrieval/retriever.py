"""The retrieval pipeline: query -> ranked child chunks -> parent-section context.

Phase 3 implements the baseline: dense vector search, then parent expansion. Later
stages (query transformation, BM25 + RRF fusion, reranking) plug in before parent
expansion without changing the result shape.

Parent expansion ("retrieve small, read big"): children are precise search units but
too thin to answer from, so each retrieved child is swapped for its parent section.
Parents are deduplicated (several children often share one), kept in the rank order
of their best child, and added until the context token budget is used up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from nexusrag.config import Settings
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.log import get_logger, timed
from nexusrag.models import (
    ContextPassage,
    ParentSection,
    RetrievedChunk,
    SearchFilters,
    StageScores,
    StageTiming,
)
from nexusrag.store import Stores
from nexusrag.store.vector_store import build_where
from nexusrag.utils.tokens import count_tokens, truncate_to_tokens

log = get_logger(__name__)

ParentFetcher = Callable[[list[str]], list[ParentSection]]


@dataclass
class RetrievalResult:
    """Everything retrieval produced, including per-stage data for the transparency UI."""

    query: str
    chunks: list[RetrievedChunk]
    passages: list[ContextPassage]
    timings: list[StageTiming] = field(default_factory=list)


def where_for(filters: SearchFilters | None) -> dict[str, object] | None:
    """Chroma ``where`` clause for UI filters."""
    if filters is None or filters.is_empty:
        return None
    return build_where(
        doc_ids=filters.doc_ids,
        filenames=filters.filenames,
        source_types=[t.value for t in filters.source_types] if filters.source_types else None,
    )


def _fallback_parent(children: Sequence[RetrievedChunk]) -> ParentSection:
    """Stand-in when a parent row is missing (e.g. mid-reindex): use the children's text."""
    first = children[0].chunk
    text = "\n\n".join(rc.chunk.text for rc in children)
    meta = first.metadata
    return ParentSection(
        parent_id=meta.parent_id,
        doc_id=meta.doc_id,
        collection=meta.collection,
        filename=meta.filename,
        source_type=meta.source_type,
        title=meta.title,
        section_path=meta.section_path,
        page_start=meta.page_start,
        page_end=meta.page_end,
        text=text,
        token_count=count_tokens(text),
    )


def expand_to_parents(
    chunks: Sequence[RetrievedChunk], fetch_parents: ParentFetcher, token_budget: int
) -> list[ContextPassage]:
    """Swap ranked children for their parents: deduplicated, rank-ordered, within budget.

    A parent that doesn't fit the remaining budget is skipped (a smaller, lower-ranked one
    may still fit). If even the top parent exceeds the budget on its own, it's truncated
    rather than dropped, so the best evidence always reaches the model.
    """
    order: list[str] = []
    grouped: dict[str, list[RetrievedChunk]] = {}
    for rc in chunks:
        parent_id = rc.chunk.metadata.parent_id
        if parent_id not in grouped:
            order.append(parent_id)
            grouped[parent_id] = []
        grouped[parent_id].append(rc)

    found = {p.parent_id: p for p in fetch_parents(order)} if order else {}
    passages: list[ContextPassage] = []
    used = 0
    for parent_id in order:
        parent = found.get(parent_id) or _fallback_parent(grouped[parent_id])
        if used + parent.token_count > token_budget:
            if passages:
                continue
            text = truncate_to_tokens(parent.text, token_budget)
            parent = parent.model_copy(update={"text": text, "token_count": count_tokens(text)})
        passages.append(
            ContextPassage(index=len(passages) + 1, parent=parent, chunks=grouped[parent_id])
        )
        used += parent.token_count
    return passages


class Retriever:
    """Retrieves context for a question from one knowledge base."""

    def __init__(self, settings: Settings, gemini: GeminiClient, stores: Stores) -> None:
        self.settings = settings
        self.gemini = gemini
        self.stores = stores

    async def retrieve(
        self,
        query: str,
        *,
        collection: str,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> RetrievalResult:
        """Dense search for ``query``, then expand the hits into parent passages."""
        k = top_k or self.settings.top_k
        timings: list[StageTiming] = []

        with timed() as t:
            vector = await self.gemini.embed_query(query)
        timings.append(StageTiming(stage="embed_query", ms=t.ms))

        with timed() as t:
            hits = await asyncio.to_thread(
                self.stores.vectors.query, collection, vector, k, where_for(filters)
            )
        timings.append(StageTiming(stage="dense_search", ms=t.ms))
        chunks = [
            RetrievedChunk(
                chunk=hit.chunk,
                scores=StageScores(dense_rank=hit.rank, dense_score=hit.score),
                matched_queries=[query],
            )
            for hit in hits
        ]

        with timed() as t:
            passages = await asyncio.to_thread(
                expand_to_parents,
                chunks,
                lambda ids: self.stores.parents.get_many(collection, ids),
                self.settings.context_token_budget,
            )
        timings.append(StageTiming(stage="parent_expansion", ms=t.ms))

        log.info(
            "retrieval.done",
            collection=collection,
            chunks=len(chunks),
            passages=len(passages),
            context_tokens=sum(p.parent.token_count for p in passages),
            **{f"{s.stage}_ms": round(s.ms, 1) for s in timings},
        )
        return RetrievalResult(query=query, chunks=chunks, passages=passages, timings=timings)
