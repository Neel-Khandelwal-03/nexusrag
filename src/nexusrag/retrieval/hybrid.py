"""Hybrid retrieval: dense vectors + BM25 keywords, merged with Reciprocal Rank Fusion.

Why hybrid? Dense embeddings capture meaning ("how long can it fly" ~ "flight time")
but are weak on exact tokens such as product codes ("CH-400"), part numbers and rare
names. BM25 is the reverse. Running both and fusing the rankings recovers passages
either method alone would miss.

Why RRF? The two retrievers produce scores on incomparable scales (cosine similarity
vs unbounded BM25). RRF ignores raw scores and combines *ranks*:

    rrf(d) = sum over rankings r of  w_r / (k + rank_r(d))

With k = 60 (the value from Cormack et al., 2009), a document ranked highly by several
lists beats one ranked first by a single list, and no score normalisation is needed.
Every query variant (original, rewrites, HyDE) contributes its own lists, so this
also performs the multi-query fusion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from nexusrag.config import Settings
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.log import get_logger
from nexusrag.models import Chunk, RetrievedChunk, SearchFilters, StageScores
from nexusrag.store import Stores
from nexusrag.store.vector_store import build_where

log = get_logger(__name__)

QueryKind = Literal["original", "standalone", "variant", "hyde"]


def where_for(filters: SearchFilters | None) -> dict[str, object] | None:
    """Chroma ``where`` clause for UI filters (None = no restriction)."""
    if filters is None or filters.is_empty:
        return None
    return build_where(
        doc_ids=filters.doc_ids,
        filenames=filters.filenames,
        source_types=[t.value for t in filters.source_types] if filters.source_types else None,
    )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists with RRF. Returns ``(id, score)`` sorted best first.

    Ties are broken by the best rank a document reached in any list, then by first
    appearance, so the output is deterministic.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for list_index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[list_index]
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(
        scores.items(), key=lambda item: (-item[1], best_rank[item[0]], first_seen[item[0]])
    )


@dataclass(frozen=True)
class SearchQuery:
    """One query to run: the user's question or a transformed variant of it."""

    text: str
    kind: QueryKind


@dataclass
class RankedList:
    """One retriever's ranking for one query (kept for the transparency UI)."""

    query: SearchQuery
    retriever: Literal["dense", "bm25"]
    chunk_ids: list[str]
    scores: list[float]


@dataclass
class HybridResult:
    """Fused candidates plus the individual rankings that produced them."""

    candidates: list[RetrievedChunk]
    rankings: list[RankedList] = field(default_factory=list)


class HybridSearcher:
    """Runs dense and BM25 search for every query and fuses the results."""

    def __init__(self, settings: Settings, gemini: GeminiClient, stores: Stores) -> None:
        self.settings = settings
        self.gemini = gemini
        self.stores = stores

    async def search(
        self,
        queries: Sequence[SearchQuery],
        *,
        collection: str,
        filters: SearchFilters | None = None,
        use_bm25: bool = True,
    ) -> HybridResult:
        """Search with every query and return RRF-fused candidates (best first)."""
        s = self.settings
        vectors = await self._embed(queries)
        where = where_for(filters)

        dense_tasks = [
            asyncio.to_thread(self.stores.vectors.query, collection, vector, s.dense_k, where)
            for vector in vectors
        ]
        dense_results = await asyncio.gather(*dense_tasks)

        rankings: list[RankedList] = []
        chunks: dict[str, Chunk] = {}
        stage: dict[str, StageScores] = {}
        matched: dict[str, list[str]] = {}

        def note(chunk_id: str, query: SearchQuery) -> None:
            matched.setdefault(chunk_id, [])
            if query.text not in matched[chunk_id]:
                matched[chunk_id].append(query.text)

        for query, hits in zip(queries, dense_results, strict=True):
            rankings.append(
                RankedList(
                    query, "dense", [h.chunk.chunk_id for h in hits], [h.score for h in hits]
                )
            )
            for hit in hits:
                cid = hit.chunk.chunk_id
                chunks[cid] = hit.chunk
                scores = stage.setdefault(cid, StageScores())
                if scores.dense_rank is None or hit.rank < scores.dense_rank:
                    scores.dense_rank, scores.dense_score = hit.rank, hit.score
                note(cid, query)

        if use_bm25:
            # HyDE text is invented; it helps dense matching but would inject made-up
            # keywords into BM25, so keyword search uses only real queries.
            keyword_queries = [q for q in queries if q.kind != "hyde"]
            bm25_results = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self.stores.bm25.search,
                        collection,
                        q.text,
                        s.bm25_k,
                        doc_ids=filters.doc_ids if filters else None,
                        filenames=filters.filenames if filters else None,
                        source_types=[t.value for t in filters.source_types]
                        if filters and filters.source_types
                        else None,
                    )
                    for q in keyword_queries
                )
            )
            for query, bm25_hits in zip(keyword_queries, bm25_results, strict=True):
                rankings.append(
                    RankedList(
                        query, "bm25", [h.chunk_id for h in bm25_hits], [h.score for h in bm25_hits]
                    )
                )
                for bm25_hit in bm25_hits:
                    scores = stage.setdefault(bm25_hit.chunk_id, StageScores())
                    if scores.bm25_rank is None or bm25_hit.rank < scores.bm25_rank:
                        scores.bm25_rank, scores.bm25_score = bm25_hit.rank, bm25_hit.score
                    note(bm25_hit.chunk_id, query)

            missing = [cid for cid in stage if cid not in chunks]
            if missing:
                chunks.update(
                    await asyncio.to_thread(self.stores.vectors.get_chunks, collection, missing)
                )

        fused = reciprocal_rank_fusion([r.chunk_ids for r in rankings], k=s.rrf_k)
        candidates = []
        for chunk_id, rrf_score in fused:
            chunk = chunks.get(chunk_id)
            if chunk is None:  # in BM25 but not in Chroma (mid-reindex); skip
                continue
            scores = stage[chunk_id]
            scores.rrf_score = rrf_score
            candidates.append(
                RetrievedChunk(chunk=chunk, scores=scores, matched_queries=matched[chunk_id])
            )
        log.info(
            "retrieval.hybrid",
            collection=collection,
            queries=len(queries),
            rankings=len(rankings),
            candidates=len(candidates),
            bm25=use_bm25,
        )
        return HybridResult(candidates=candidates, rankings=rankings)

    async def _embed(self, queries: Sequence[SearchQuery]) -> list[list[float]]:
        """Embed queries in one batch; HyDE passages are embedded as *documents*.

        A hypothetical answer is document-like text, so it's compared document-to-document
        (``title: none | text: ...``) instead of with the query prefix.
        """
        question_idx = [i for i, q in enumerate(queries) if q.kind != "hyde"]
        hyde_idx = [i for i, q in enumerate(queries) if q.kind == "hyde"]
        question_vecs, hyde_vecs = await asyncio.gather(
            self.gemini.embed_queries([queries[i].text for i in question_idx]),
            self.gemini.embed_documents([queries[i].text for i in hyde_idx]),
        )
        vectors: list[list[float]] = [[] for _ in queries]
        for i, vec in zip(question_idx, question_vecs, strict=True):
            vectors[i] = vec
        for i, vec in zip(hyde_idx, hyde_vecs, strict=True):
            vectors[i] = vec
        return vectors
