"""Reranking: rescore fused candidates with a model that reads query and passage together.

Embeddings and BM25 score the query and each passage *independently*, which is fast
but coarse. A cross-encoder reads the pair jointly and judges relevance much more
accurately, but it's too slow to run over a whole corpus. So it runs on the top few
dozen fused candidates only. Its ranking is then fused with the first-stage ranking to
pick the final top-k, and its score threshold drops irrelevant passages (see
:func:`rerank`).

Both backends read :func:`~nexusrag.ingestion.chunker.rerank_text`, with Markdown
tables rewritten as ``Header: value`` lines, which rerankers score far more reliably.

Backends:

* ``cross_encoder``: local ``BAAI/bge-reranker-base`` via sentence-transformers,
  with sigmoid scores in [0, 1]. The model is loaded lazily on first use; install the
  ``rerank`` extra to get it.
* ``gemini``: the fast Gemini model rates passages 0-10 (normalised to [0, 1]). Used
  when configured, or automatically when the local model can't be loaded (package
  missing, no network to download weights, not enough memory).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from nexusrag.config import Settings
from nexusrag.ingestion.chunker import rerank_text
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.llm.prompts import RERANK_PROMPT, render
from nexusrag.log import get_logger
from nexusrag.models import RetrievedChunk
from nexusrag.utils.tokens import truncate_to_tokens

log = get_logger(__name__)

ModelLoader = Callable[[str, int], Any]


class Reranker(Protocol):
    """Scores candidates for a query; higher is more relevant, roughly in [0, 1]."""

    @property
    def name(self) -> str: ...

    async def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]: ...


def _load_cross_encoder(model_name: str, max_length: int) -> Any:
    # Imported lazily: torch is heavy and optional (see the `rerank` extra).
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=max_length, activation_fn=torch.nn.Sigmoid())


class CrossEncoderReranker:
    """Local cross-encoder, loaded once per process on first use (thread-safe)."""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        batch_size: int = 16,
        loader: ModelLoader | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._loader = loader or _load_cross_encoder
        self._model: Any = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self.load()

    def load(self) -> None:
        """Load the model now (e.g. at startup) instead of on the first query."""
        with self._lock:
            if self._model is None:
                log.info("rerank.loading_model", model=self.model_name)
                started = time.perf_counter()
                self._model = self._loader(self.model_name, self.max_length)
                log.info(
                    "rerank.model_loaded",
                    model=self.model_name,
                    ms=round((time.perf_counter() - started) * 1000),
                )

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.load()
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    async def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        pairs = [(query, rerank_text(rc.chunk)) for rc in candidates]
        return await asyncio.to_thread(self._predict, pairs)


class _PassageScore(BaseModel):
    id: int
    score: float = Field(ge=0, le=10)


class _RerankScores(BaseModel):
    scores: list[_PassageScore]


class GeminiReranker:
    """LLM-as-reranker: one fast-model call rates every candidate 0-10."""

    name = "gemini"

    def __init__(self, gemini: GeminiClient, *, max_passage_tokens: int = 200) -> None:
        self.gemini = gemini
        self.max_passage_tokens = max_passage_tokens

    async def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        passages = "\n\n".join(
            f'<passage id="{i}">\n'
            f"{truncate_to_tokens(rerank_text(rc.chunk), self.max_passage_tokens)}\n"
            "</passage>"
            for i, rc in enumerate(candidates, start=1)
        )
        result = await self.gemini.generate_structured(
            render(RERANK_PROMPT, query=query, passages=passages),
            _RerankScores,
            role="fast",
            stage="rerank",
        )
        by_id = {s.id: s.score for s in result.scores}
        return [by_id.get(i, 0.0) / 10.0 for i in range(1, len(candidates) + 1)]


class FallbackReranker:
    """Uses ``primary`` until it fails once, then switches to ``fallback`` for good.

    A cross-encoder that can't load (missing package, failed download) fails the same
    way every time, so we switch once instead of paying for the failure on every query.
    """

    def __init__(self, primary: Reranker, fallback: Reranker) -> None:
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False

    @property
    def name(self) -> str:
        return self.fallback.name if self._use_fallback else self.primary.name

    def warm_up(self) -> None:
        """Load the primary model ahead of the first query; switch to the fallback on failure."""
        load = getattr(self.primary, "load", None)
        if load is None or self._use_fallback:
            return
        try:
            load()
        except Exception as exc:
            self._use_fallback = True
            log.warning(
                "rerank.fallback",
                primary=self.primary.name,
                fallback=self.fallback.name,
                error_type=type(exc).__name__,
                detail=str(exc)[:200],
                during="warm_up",
            )

    async def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        if not self._use_fallback:
            try:
                return await self.primary.score(query, candidates)
            except Exception as exc:
                self._use_fallback = True
                log.warning(
                    "rerank.fallback",
                    primary=self.primary.name,
                    fallback=self.fallback.name,
                    error_type=type(exc).__name__,
                    detail=str(exc)[:200],
                )
        return await self.fallback.score(query, candidates)


def build_reranker(settings: Settings, gemini: GeminiClient) -> Reranker:
    """Reranker for the configured backend (cross-encoder with Gemini fallback by default)."""
    gemini_reranker = GeminiReranker(gemini)
    if settings.reranker_backend == "gemini":
        return gemini_reranker
    cross_encoder = CrossEncoderReranker(
        settings.reranker_model, max_length=settings.reranker_max_length
    )
    return FallbackReranker(cross_encoder, gemini_reranker)


async def rerank(
    reranker: Reranker,
    query: str,
    candidates: Sequence[RetrievedChunk],
    *,
    top_k: int,
    threshold: float,
    max_candidates: int,
    fusion_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Rescore the top ``max_candidates``; keep the best ``top_k`` scoring >= ``threshold``.

    The final order fuses two rankings with RRF: the first-stage (hybrid) order and the
    reranker's order, with ``fusion_weight`` for the first stage (0 = pure reranker order).
    Measured on the sample corpus, pure reranking buried the one table holding the answer
    to "Aurora X1 flight time". BM25, dense search and fusion all ranked it #1, but the
    cross-encoder scored it 0.12 and put it 6th, outside the top 5. Fusing the rankings
    kept it (3rd) and raised the right passage's rank on other questions too, so a single
    noisy reranker score can no longer overrule agreement between both retrievers.

    The threshold still applies to the reranker's own score, so irrelevant passages are
    dropped. The result may be empty, which lets the pipeline say "not in your documents"
    instead of answering from noise. Returns copies with ``scores.rerank_score`` set.
    """
    from nexusrag.retrieval.hybrid import reciprocal_rank_fusion  # local: avoids an import cycle

    pool = list(candidates[:max_candidates])
    if not pool:
        return []
    scores = await reranker.score(query, pool)
    rescored = {
        rc.chunk.chunk_id: rc.model_copy(
            update={"scores": rc.scores.model_copy(update={"rerank_score": score})}
        )
        for rc, score in zip(pool, scores, strict=True)
    }
    first_stage = [rc.chunk.chunk_id for rc in pool]
    by_reranker = sorted(
        first_stage, key=lambda cid: rescored[cid].scores.rerank_score or 0.0, reverse=True
    )
    if fusion_weight > 0:
        order = [
            cid
            for cid, _ in reciprocal_rank_fusion(
                [first_stage, by_reranker], k=rrf_k, weights=[fusion_weight, 1.0]
            )
        ]
    else:
        order = by_reranker
    kept = [
        rescored[cid] for cid in order if (rescored[cid].scores.rerank_score or 0.0) >= threshold
    ]
    return kept[:top_k]
