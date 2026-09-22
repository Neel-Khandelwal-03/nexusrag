"""Application service: one object that answers questions against a knowledge base.

The UI, the ``ask`` CLI and the evaluation suite all go through :class:`RAGService`,
so they exercise the same pipeline: the self-correcting agent (routing, retrieval with
relevance-driven retries, grounded generation with a groundedness check, and the
summarize/compare/chitchat routes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from nexusrag.agent.graph import (
    AgentGraph,
    AgentRequest,
    AgentResult,
    NodeStartCallback,
    StepCallback,
)
from nexusrag.config import Settings, get_settings
from nexusrag.generation.answer import (
    AnswerGenerator,
    AnswerStyle,
    ResetCallback,
    TokenCallback,
)
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.llm.usage import process_usage, track_usage
from nexusrag.log import get_logger, request_context, timed
from nexusrag.models import (
    Answer,
    ChatTurn,
    ContextPassage,
    Route,
    SearchFilters,
    StageTiming,
    UsageStats,
)
from nexusrag.retrieval.reranker import Reranker
from nexusrag.retrieval.retriever import RetrievalOptions, RetrievalResult, Retriever
from nexusrag.store import Stores
from nexusrag.store.registry import DocumentRecord, validate_collection_name

log = get_logger(__name__)


@dataclass
class KnowledgeBaseStats:
    """What ``/stats`` reports."""

    collection: str
    documents: list[DocumentRecord]
    parents: int
    chunks: int
    vectors: int
    collections: list[str]
    usage: UsageStats
    #: Filled in once the semantic cache exists (phase 7).
    cache_hits: int | None = None
    cache_lookups: int | None = None
    #: Answers stored in the semantic cache for this knowledge base (None: cache off).
    cache_entries: int | None = None


@dataclass
class AskResult:
    """An answer plus the trace behind it (for the UI and evaluation)."""

    answer: Answer
    #: Every retrieval the agent ran (retries, one per document for comparisons).
    retrievals: list[RetrievalResult] = field(default_factory=list)
    #: The context passages the final answer was generated from.
    passages: list[ContextPassage] = field(default_factory=list)
    agent: AgentResult | None = None

    @property
    def retrieval(self) -> RetrievalResult | None:
        """The most recent retrieval, or None for routes that don't retrieve (chitchat)."""
        return self.retrievals[-1] if self.retrievals else None


class RAGService:
    """The self-correcting RAG agent over the configured stores."""

    def __init__(
        self,
        settings: Settings,
        gemini: GeminiClient,
        stores: Stores,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.gemini = gemini
        self.stores = stores
        self.retriever = Retriever(settings, gemini, stores, reranker=reranker)
        self.generator = AnswerGenerator(gemini)
        self.agent = AgentGraph(settings, gemini, stores, self.retriever, self.generator)

    @classmethod
    def create(cls, settings: Settings | None = None) -> RAGService:
        """Build the service from settings (opens storage, creates the Gemini client)."""
        settings = settings or get_settings()
        return cls(settings, GeminiClient(settings), Stores.open(settings))

    async def ask(
        self,
        question: str,
        *,
        collection: str | None = None,
        history: Sequence[ChatTurn] = (),
        filters: SearchFilters | None = None,
        options: RetrievalOptions | None = None,
        style: AnswerStyle = "detailed",
        mode: Route | None = None,
        self_correct: bool | None = None,
        recent_doc_ids: Sequence[str] = (),
        use_cache: bool | None = None,
        on_token: TokenCallback | None = None,
        on_reset: ResetCallback | None = None,
        on_step: StepCallback | None = None,
        on_node_start: NodeStartCallback | None = None,
        request_id: str | None = None,
    ) -> AskResult:
        """Answer ``question``, streaming tokens to ``on_token`` and steps to ``on_step``.

        ``mode`` forces the summarize/compare routes (chat profiles). ``self_correct``
        overrides ``ENABLE_SELF_CORRECTION`` for this request (used by the evaluation).
        ``recent_doc_ids`` are documents the user just uploaded, so "what is this file
        about?" resolves to them. ``use_cache`` overrides ``ENABLE_SEMANTIC_CACHE`` (the
        evaluation turns it off so every question is really answered).
        """
        kb = validate_collection_name(collection or self.settings.default_collection)
        request = AgentRequest(
            question=question,
            collection=kb,
            history=history,
            filters=filters,
            options=options,
            style=style,
            mode=mode,
            self_correct=self.settings.enable_self_correction
            if self_correct is None
            else self_correct,
            recent_doc_ids=recent_doc_ids,
            use_cache=self.settings.enable_semantic_cache if use_cache is None else use_cache,
        )
        with (
            request_context(request_id, collection=kb) as rid,
            track_usage() as usage,
            timed() as total,
        ):
            result = await self.agent.run(
                request,
                on_token=on_token,
                on_reset=on_reset,
                on_step=on_step,
                on_node_start=on_node_start,
            )
            totals = usage.totals()
            log.info(
                "rag.answer",
                route=result.route.value,
                steps=len(result.steps),
                retrievals=len(result.retrievals),
                passages=len(result.passages),
                cited=len(result.citations),
                refused=result.refused,
                grounded=result.grounded,
                cached=result.cached,
                total_ms=round(total.ms, 1),
                llm_calls=totals.calls,
                prompt_tokens=totals.prompt_tokens,
                output_tokens=totals.output_tokens,
                cost_usd=round(totals.cost_usd, 6),
            )
        hit = result.cache_lookup.hit if result.cache_lookup is not None else None
        answer = Answer(
            text=result.text,
            route=result.route,
            citations=result.citations,
            grounded=result.grounded,
            refused=result.refused,
            closest_matches=result.closest_matches,
            cached=hit is not None,
            cache_similarity=hit.similarity if hit is not None else None,
            usage=usage.totals(),
            timings=[StageTiming(stage=step.node, ms=step.ms) for step in result.steps],
            steps=result.steps,
            request_id=rid,
        )
        return AskResult(
            answer=answer, retrievals=result.retrievals, passages=result.passages, agent=result
        )

    def warm_up(self) -> None:
        """Load slow models ahead of the first question (blocking; run in a thread)."""
        self.retriever.warm_up()

    def stats(self, collection: str | None = None) -> KnowledgeBaseStats:
        """Counts for the ``/stats`` command."""
        kb = validate_collection_name(collection or self.settings.default_collection)
        documents = self.stores.registry.list_documents(kb)
        cache = self.stores.cache
        counters = cache.counters(kb)
        cache_on = self.settings.enable_semantic_cache
        return KnowledgeBaseStats(
            collection=kb,
            documents=documents,
            parents=self.stores.parents.count(kb),
            chunks=sum(d.num_chunks for d in documents),
            vectors=self.stores.vectors.count(kb),
            collections=[c.name for c in self.stores.registry.list_collections()],
            usage=process_usage(),
            cache_hits=counters.hits if cache_on else None,
            cache_lookups=counters.lookups if cache_on else None,
            cache_entries=cache.count(kb) if cache_on else None,
        )

    def clear_cache(self, collection: str | None = None) -> int:
        """Drop the knowledge base's cached answers; returns how many were removed."""
        kb = validate_collection_name(collection or self.settings.default_collection)
        return self.stores.cache.invalidate(kb)

    def close(self) -> None:
        self.stores.close()
