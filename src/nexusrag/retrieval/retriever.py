"""The retrieval pipeline, end to end.

    question (+ chat history)
      -> query plan       condense follow-ups; optional multi-query variants and HyDE
      -> hybrid search    dense + BM25 for every query, fused with Reciprocal Rank Fusion
      -> rerank           cross-encoder rescoring of the top candidates, with a score threshold
      -> parent expansion swap child chunks for their parent sections, within a token budget

Each stage can be switched off per request (:class:`RetrievalOptions`), which is how
the evaluation suite compares "dense only" against "hybrid + rerank + rewriting".

Parent expansion ("retrieve small, read big"): children are precise search units but
too thin to answer from, so each retrieved child is swapped for its parent section.
Parents are deduplicated (several children often share one), kept in the rank order
of their best child, and added until the context token budget is used up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from nexusrag.config import Settings
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.log import get_logger, timed
from nexusrag.models import (
    ChatTurn,
    ContextPassage,
    ParentSection,
    RetrievedChunk,
    SearchFilters,
    StageTiming,
)
from nexusrag.retrieval.hybrid import HybridSearcher, RankedList, where_for
from nexusrag.retrieval.query_transform import QueryPlan, QueryTransformer
from nexusrag.retrieval.reranker import Reranker, build_reranker, rerank
from nexusrag.store import Stores
from nexusrag.utils.tokens import count_tokens, truncate_to_tokens

__all__ = [
    "RetrievalOptions",
    "RetrievalResult",
    "Retriever",
    "expand_to_parents",
    "where_for",
]

log = get_logger(__name__)

ParentFetcher = Callable[[list[str]], list[ParentSection]]


@dataclass(frozen=True)
class RetrievalOptions:
    """Per-request switches for each retrieval stage (defaults come from settings)."""

    top_k: int = 5
    hybrid: bool = True
    multi_query: bool = True
    num_variants: int = 3
    hyde: bool = False
    rerank: bool = True
    #: Overrides ``RERANK_THRESHOLD`` (None keeps the setting; 0 reorders without dropping).
    rerank_threshold: float | None = None

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> RetrievalOptions:
        base = cls(
            top_k=settings.top_k,
            hybrid=settings.enable_hybrid,
            multi_query=settings.enable_multi_query,
            num_variants=settings.num_query_variants,
            hyde=settings.enable_hyde,
            rerank=settings.enable_rerank,
        )
        return replace(base, **overrides)


@dataclass
class RetrievalResult:
    """Everything retrieval produced, including per-stage data for the transparency UI."""

    plan: QueryPlan
    #: Fused (RRF) candidates before reranking, best first.
    candidates: list[RetrievedChunk]
    #: Final ranked chunks after reranking/threshold (at most top_k).
    chunks: list[RetrievedChunk]
    passages: list[ContextPassage]
    rankings: list[RankedList] = field(default_factory=list)
    #: Name of the reranker that produced the final order, or None if reranking was skipped.
    reranker: str | None = None
    timings: list[StageTiming] = field(default_factory=list)

    @property
    def query(self) -> str:
        """The standalone question retrieval actually searched for."""
        return self.plan.standalone


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

    def __init__(
        self,
        settings: Settings,
        gemini: GeminiClient,
        stores: Stores,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.stores = stores
        self.transformer = QueryTransformer(gemini)
        self.searcher = HybridSearcher(settings, gemini, stores)
        self.reranker = reranker or build_reranker(settings, gemini)

    def warm_up(self) -> None:
        """Load slow models (the local cross-encoder) now rather than on the first query.

        Blocking: call it from a worker thread at app startup.
        """
        warm = getattr(self.reranker, "warm_up", None)
        if warm is not None and self.settings.enable_rerank:
            warm()

    async def retrieve(
        self,
        question: str,
        *,
        collection: str,
        history: Sequence[ChatTurn] = (),
        filters: SearchFilters | None = None,
        options: RetrievalOptions | None = None,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline for ``question``."""
        s = self.settings
        opts = options or RetrievalOptions.from_settings(s)
        timings: list[StageTiming] = []

        with timed() as t:
            plan = await self.transformer.plan(
                question,
                history[-s.history_turns :] if s.history_turns else (),
                multi_query=opts.multi_query,
                num_variants=opts.num_variants,
                hyde=opts.hyde,
            )
        timings.append(StageTiming(stage="query_transform", ms=t.ms))

        with timed() as t:
            hybrid = await self.searcher.search(
                plan.queries, collection=collection, filters=filters, use_bm25=opts.hybrid
            )
        timings.append(
            StageTiming(stage="hybrid_search" if opts.hybrid else "dense_search", ms=t.ms)
        )

        chunks = hybrid.candidates[: opts.top_k]
        reranker_name: str | None = None
        if opts.rerank and hybrid.candidates:
            with timed() as t:
                try:
                    chunks = await rerank(
                        self.reranker,
                        plan.standalone,
                        hybrid.candidates,
                        top_k=opts.top_k,
                        threshold=s.rerank_threshold
                        if opts.rerank_threshold is None
                        else opts.rerank_threshold,
                        max_candidates=s.rerank_candidates,
                        fusion_weight=s.rerank_fusion_weight,
                        rrf_k=s.rrf_k,
                    )
                    reranker_name = self.reranker.name
                except Exception as exc:
                    # Reranking improves ordering but isn't essential: keep the RRF order.
                    log.warning("retrieval.rerank_failed", error_type=type(exc).__name__)
            timings.append(StageTiming(stage="rerank", ms=t.ms))

        with timed() as t:
            passages = await asyncio.to_thread(
                expand_to_parents,
                chunks,
                lambda ids: self.stores.parents.get_many(collection, ids),
                s.context_token_budget,
            )
        timings.append(StageTiming(stage="parent_expansion", ms=t.ms))

        log.info(
            "retrieval.done",
            collection=collection,
            queries=len(plan.queries),
            candidates=len(hybrid.candidates),
            chunks=len(chunks),
            passages=len(passages),
            reranker=reranker_name,
            context_tokens=sum(p.parent.token_count for p in passages),
            **{f"{st.stage}_ms": round(st.ms, 1) for st in timings},
        )
        return RetrievalResult(
            plan=plan,
            candidates=hybrid.candidates,
            chunks=chunks,
            passages=passages,
            rankings=hybrid.rankings,
            reranker=reranker_name,
            timings=timings,
        )
