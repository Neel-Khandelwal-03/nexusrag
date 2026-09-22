"""The self-correcting agent: an explicit, hand-written state machine.

Every node is an async method that updates :class:`AgentState` and returns the next node.
The whole control flow therefore sits in one readable place, every step is recorded for
the transparency UI, and there's no framework magic::

    ROUTE ─ chitchat ──────► CHITCHAT ──────────────────────────────────────► DONE
      │ ─ out_of_scope ────► OUT_OF_SCOPE ──────────────────────────────────► DONE
      │ ─ summarize ───────► SUMMARIZE ─► CHECK_GROUNDEDNESS ─┐
      │ ─ compare ─────────► COMPARE ───► CHECK_GROUNDEDNESS ─┤
      └ doc_qa ─► RETRIEVE ─► GRADE_RELEVANCE                 │
                     ▲             │  insufficient and         │
                     └─────────────┘  retries left: rewrite    │
                                   │ sufficient / out of tries │
                                   ▼                           │
                                GENERATE ── refused ─► DONE    │ (+ closest matches)
                                   ▼                           │
                            CHECK_GROUNDEDNESS ◄───────────────┘
                              │ grounded ──────────────────────────────────► DONE
                              │ unsupported, first time
                              ▼
                            REGENERATE (stricter prompt) ─► CHECK_GROUNDEDNESS
                              │ still unsupported
                              ▼
                            REFUSE_UNSUPPORTED ─────────────────────────────► DONE

With the semantic cache on, document routes pass through CHECK_CACHE first. A hit (a
near-identical earlier question, same knowledge base version and settings) goes straight
to DONE with the stored answer. A miss continues to the route's node, and the final
answer is stored if it is cited and grounded.

Self-correction (relevance grading, groundedness checking) can be switched off per
request, which is how the evaluation suite measures what it adds.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from nexusrag.agent.grader import (
    GroundednessGrader,
    GroundednessVerdict,
    RelevanceGrader,
    RelevanceVerdict,
)
from nexusrag.agent.router import RouteDecision, Router
from nexusrag.config import Settings
from nexusrag.generation.answer import (
    AnswerGenerator,
    AnswerStyle,
    GeneratedAnswer,
    ResetCallback,
    TokenCallback,
)
from nexusrag.generation.citations import closest_matches
from nexusrag.generation.modes import (
    ComparisonGroup,
    DocumentComparer,
    DocumentSummarizer,
    chitchat_reply,
)
from nexusrag.llm.gemini_client import GeminiClient, GeminiError, query_embedding_memo
from nexusrag.llm.prompts import (
    ASK_WHICH_DOCUMENT,
    OUT_OF_SCOPE_MESSAGE,
    REFUSAL_MESSAGE,
    STRICT_ANSWER_ADDENDUM,
    render,
)
from nexusrag.log import get_logger, timed
from nexusrag.models import (
    AgentStep,
    Answer,
    ChatTurn,
    Citation,
    ContextPassage,
    RetrievedChunk,
    Route,
    SearchFilters,
)
from nexusrag.retrieval.retriever import (
    RetrievalOptions,
    RetrievalResult,
    Retriever,
    expand_to_parents,
)
from nexusrag.store import Stores
from nexusrag.store.registry import DocumentRecord
from nexusrag.store.semantic_cache import CacheKey, CacheLookup

log = get_logger(__name__)

StepCallback = Callable[[AgentStep], Awaitable[None]]
NodeStartCallback = Callable[[str], Awaitable[None]]

#: Rows of the ranking tables attached to retrieval steps (keeps step payloads small).
TRACE_ROWS = 8

#: Hard stop for runaway loops; the longest legitimate path is about 10 steps.
MAX_STEPS = 20

#: Routes whose answers come from the documents, and so can be cached.
CACHEABLE_ROUTES = frozenset({Route.DOC_QA, Route.SUMMARIZE, Route.COMPARE})
#: Bump when prompts or the stored answer format change, so old entries stop matching.
CACHE_FORMAT = 1

UNSUPPORTED_MESSAGE = (
    f"{REFUSAL_MESSAGE} The passages I found don't support a reliable answer, so I'm showing "
    "the closest matches instead of guessing."
)


class Node(StrEnum):
    ROUTE = "route"
    CHECK_CACHE = "check_cache"
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    RETRIEVE = "retrieve"
    GRADE_RELEVANCE = "grade_relevance"
    GENERATE = "generate"
    CHECK_GROUNDEDNESS = "check_groundedness"
    REGENERATE = "regenerate"
    REFUSE_UNSUPPORTED = "refuse_unsupported"
    DONE = "done"


@dataclass
class AgentRequest:
    """Everything the agent needs to answer one message."""

    question: str
    collection: str
    history: Sequence[ChatTurn] = ()
    filters: SearchFilters | None = None
    options: RetrievalOptions | None = None
    style: AnswerStyle = "detailed"
    #: Force summarize/compare (chat profiles). Greetings still get a chitchat reply.
    mode: Route | None = None
    self_correct: bool = True
    #: Documents the user just uploaded: what "this file" or "this project" refers to.
    recent_doc_ids: Sequence[str] = ()
    #: Look up and store answers in the semantic cache.
    use_cache: bool = False


@dataclass
class _Callbacks:
    on_token: TokenCallback | None = None
    on_reset: ResetCallback | None = None


@dataclass
class AgentState:
    """Mutable working memory for one run."""

    request: AgentRequest
    documents: list[DocumentRecord]
    callbacks: _Callbacks
    decision: RouteDecision | None = None
    query: str = ""
    tried_queries: list[str] = field(default_factory=list)
    retrievals: list[RetrievalResult] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    passages: list[ContextPassage] = field(default_factory=list)
    comparison: list[ComparisonGroup] = field(default_factory=list)
    target_document: DocumentRecord | None = None
    summary_map_reduce: bool = False
    relevance: list[RelevanceVerdict] = field(default_factory=list)
    generated: GeneratedAnswer | None = None
    groundedness: list[GroundednessVerdict] = field(default_factory=list)
    regenerated: bool = False
    grounded: bool | None = None
    closest: list[Citation] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    #: Where to go after CHECK_CACHE on a miss.
    after_cache: Node | None = None
    cache_key: CacheKey | None = None
    cache_lookup: CacheLookup | None = None

    @property
    def route(self) -> Route:
        return self.decision.route if self.decision else Route.DOC_QA

    @property
    def standalone(self) -> str:
        return self.decision.standalone_question if self.decision else self.request.question


@dataclass
class AgentResult:
    """Outcome of a run: the answer plus everything that led to it."""

    text: str
    route: Route
    citations: list[Citation]
    refused: bool
    grounded: bool | None
    closest_matches: list[Citation]
    passages: list[ContextPassage]
    retrievals: list[RetrievalResult]
    steps: list[AgentStep]
    decision: RouteDecision | None
    cache_lookup: CacheLookup | None = None

    @property
    def cached(self) -> bool:
        return self.cache_lookup is not None and self.cache_lookup.hit is not None


def cache_fingerprint(
    settings: Settings,
    request: AgentRequest,
    decision: RouteDecision,
    options: RetrievalOptions,
    filters: SearchFilters | None,
) -> str:
    """Hash of everything besides the question that shapes an answer.

    Two questions share a cached answer only if these match: the route (and, for
    summaries and comparisons, which documents), the search scope actually used, answer
    style, retrieval switches, self-correction and the models.
    """
    payload = {
        "format": CACHE_FORMAT,
        "route": decision.route.value,
        # For doc_qa, router targets only matter through `filters` (the search scope).
        "targets": [] if decision.route == Route.DOC_QA else sorted(decision.doc_ids),
        "style": request.style,
        "filters": None
        if filters is None or filters.is_empty
        else {
            "doc_ids": sorted(filters.doc_ids or []),
            "filenames": sorted(filters.filenames or []),
            "source_types": sorted(t.value for t in filters.source_types or []),
        },
        "options": asdict(options),
        "self_correct": request.self_correct,
        "models": [settings.generation_model, settings.embedding_model, settings.embedding_dim],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def merge_round_robin(
    first: Sequence[RetrievedChunk], second: Sequence[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    """Interleave two ranked lists (deduplicated) so both attempts are represented.

    After a relevance retry, the first retrieval usually holds part of the answer and the
    retry holds the missing part. Reranking the union against the original question would
    tend to recreate the miss, so we alternate instead.
    """
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for i in range(max(len(first), len(second))):
        for source in (first, second):
            if i < len(source) and source[i].chunk.chunk_id not in seen:
                seen.add(source[i].chunk.chunk_id)
                merged.append(source[i])
    return merged[:limit]


def _trace_row(rc: RetrievedChunk) -> dict[str, Any]:
    """Compact, JSON-safe view of a retrieved chunk for the pipeline UI."""
    m, s = rc.chunk.metadata, rc.scores
    where = " · ".join(
        part
        for part in (
            m.filename,
            f"p. {m.page_start}" if m.page_start is not None else "",
            m.section_path,
        )
        if part
    )
    return {
        "location": where,
        "dense_rank": s.dense_rank,
        "bm25_rank": s.bm25_rank,
        "rrf": round(s.rrf_score, 4) if s.rrf_score is not None else None,
        "rerank": round(s.rerank_score, 3) if s.rerank_score is not None else None,
    }


def _document_listing(documents: Sequence[DocumentRecord]) -> str:
    return "\n".join(f"- {d.title} (`{d.filename}`)" for d in documents[:20])


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


class AgentGraph:
    """Runs the state machine above for each message."""

    def __init__(
        self,
        settings: Settings,
        gemini: GeminiClient,
        stores: Stores,
        retriever: Retriever,
        generator: AnswerGenerator,
    ) -> None:
        self.settings = settings
        self.gemini = gemini
        self.stores = stores
        self.retriever = retriever
        self.generator = generator
        self.router = Router(gemini)
        self.relevance_grader = RelevanceGrader(gemini)
        self.groundedness_grader = GroundednessGrader(gemini)
        self.summarizer = DocumentSummarizer(gemini, settings)
        self.comparer = DocumentComparer(gemini)
        self._nodes: dict[
            Node, Callable[[AgentState], Awaitable[tuple[Node, str, dict[str, Any]]]]
        ] = {
            Node.ROUTE: self._route,
            Node.CHECK_CACHE: self._check_cache,
            Node.CHITCHAT: self._chitchat,
            Node.OUT_OF_SCOPE: self._out_of_scope,
            Node.SUMMARIZE: self._summarize,
            Node.COMPARE: self._compare,
            Node.RETRIEVE: self._retrieve,
            Node.GRADE_RELEVANCE: self._grade_relevance,
            Node.GENERATE: self._generate,
            Node.CHECK_GROUNDEDNESS: self._check_groundedness,
            Node.REGENERATE: self._regenerate,
            Node.REFUSE_UNSUPPORTED: self._refuse_unsupported,
        }

    # ------------------------------------------------------------------ driver

    async def run(
        self,
        request: AgentRequest,
        *,
        on_token: TokenCallback | None = None,
        on_reset: ResetCallback | None = None,
        on_step: StepCallback | None = None,
        on_node_start: NodeStartCallback | None = None,
    ) -> AgentResult:
        """Run the state machine. ``on_node_start`` fires before each node and ``on_step``
        after it, so a UI can show a step as running and then fill in its result."""
        documents = await asyncio.to_thread(self.stores.registry.list_documents, request.collection)
        state = AgentState(
            request=request, documents=documents, callbacks=_Callbacks(on_token, on_reset)
        )
        node = Node.ROUTE
        # The cache check and retrieval embed the same question: embed it once.
        with query_embedding_memo():
            for _ in range(MAX_STEPS):
                if node is Node.DONE:
                    break
                if on_node_start is not None:
                    await on_node_start(node.value)
                with timed() as t:
                    next_node, label, detail = await self._nodes[node](state)
                step = AgentStep(node=node.value, label=label, ms=t.ms, detail=detail)
                state.steps.append(step)
                log.info("agent.step", node=node.value, next=next_node.value, ms=round(t.ms, 1))
                if on_step is not None:
                    await on_step(step)
                node = next_node
            else:
                raise RuntimeError(f"agent exceeded {MAX_STEPS} steps")
        result = self._result(state)
        await self._store_in_cache(state, result)
        return result

    def _result(self, state: AgentState) -> AgentResult:
        generated = state.generated or GeneratedAnswer(text="")
        return AgentResult(
            text=generated.text,
            route=state.route,
            citations=generated.citations,
            refused=generated.refused,
            grounded=state.grounded,
            closest_matches=state.closest,
            passages=state.passages,
            retrievals=state.retrievals,
            steps=state.steps,
            decision=state.decision,
            cache_lookup=state.cache_lookup,
        )

    async def _store_in_cache(self, state: AgentState, result: AgentResult) -> None:
        """Keep a fresh answer for reuse if it is cited, grounded and not a refusal."""
        key = state.cache_key
        if key is None or result.cached:
            return
        if (
            result.route not in CACHEABLE_ROUTES
            or result.refused
            or not result.citations
            or result.grounded is False
        ):
            return
        answer = Answer(
            text=result.text,
            route=result.route,
            citations=result.citations,
            grounded=result.grounded,
        )
        try:
            await asyncio.to_thread(self.stores.cache.put, key, answer)
        except sqlite3.Error as exc:  # the cache is an optimisation: never fail the turn
            log.warning("cache.store_failed", error_type=type(exc).__name__)

    # ------------------------------------------------------------------ helpers

    async def _emit(self, state: AgentState, text: str) -> None:
        """Send a complete, non-streamed message through the token callback."""
        if state.callbacks.on_token is not None:
            await state.callbacks.on_token(text)

    async def _reset(self, state: AgentState) -> None:
        if state.callbacks.on_reset is not None:
            await state.callbacks.on_reset()

    def _options(self, state: AgentState) -> RetrievalOptions:
        return state.request.options or RetrievalOptions.from_settings(self.settings)

    def _documents_by_id(self, state: AgentState, doc_ids: Sequence[str]) -> list[DocumentRecord]:
        by_id = {d.doc_id: d for d in state.documents}
        return [by_id[d] for d in doc_ids if d in by_id]

    def _target_documents(self, state: AgentState) -> list[DocumentRecord]:
        """Documents a summarize/compare request refers to: router, then UI filter."""
        assert state.decision is not None
        ids = state.decision.doc_ids or list(
            (state.request.filters.doc_ids if state.request.filters else None) or []
        )
        return self._documents_by_id(state, ids)

    def _retrieval_filters(self, state: AgentState) -> SearchFilters | None:
        """The UI's document filter, else just-uploaded documents a question points at.

        Router targets outside the fresh uploads don't restrict the search: a question
        naming a product ("the Aurora X1's price") may be answered by another document
        (a sales report), and hybrid search finds the named one anyway.
        """
        request, decision = state.request, state.decision
        if request.filters is not None and not request.filters.is_empty:
            return request.filters
        if decision is not None and decision.route == Route.DOC_QA and decision.doc_ids:
            recent = set(request.recent_doc_ids)
            if recent and set(decision.doc_ids) <= recent:
                return SearchFilters(doc_ids=decision.doc_ids)
        return request.filters

    def _set_closest(self, state: AgentState) -> None:
        candidates = [rc for r in state.retrievals for rc in r.candidates]
        state.closest = closest_matches(candidates, limit=3)

    # ------------------------------------------------------------------ nodes

    async def _route(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        request = state.request
        decision = await self.router.route(
            request.question,
            documents=state.documents,
            history=request.history,
            recent_doc_ids=request.recent_doc_ids,
        )
        forced = (
            request.mode in (Route.SUMMARIZE, Route.COMPARE) and decision.route != Route.CHITCHAT
        )
        if forced and request.mode is not None:
            decision.route = request.mode
        state.decision = decision
        state.query = decision.standalone_question
        next_node = {
            Route.CHITCHAT: Node.CHITCHAT,
            Route.OUT_OF_SCOPE: Node.OUT_OF_SCOPE,
            Route.SUMMARIZE: Node.SUMMARIZE,
            Route.COMPARE: Node.COMPARE,
            Route.DOC_QA: Node.RETRIEVE,
        }[decision.route]
        if request.use_cache and decision.route in CACHEABLE_ROUTES:
            state.after_cache, next_node = next_node, Node.CHECK_CACHE
        targets = [d.filename for d in self._documents_by_id(state, decision.doc_ids)]
        reason = f" ({decision.reason})" if decision.reason else ""
        return (
            next_node,
            f"Route: {decision.route.value}{reason}",
            {
                "route": decision.route.value,
                "standalone_question": decision.standalone_question,
                "documents": targets,
                "forced_by_mode": forced,
                "from_model": decision.from_model,
            },
        )

    async def _check_cache(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        request, decision, after = state.request, state.decision, state.after_cache
        assert decision is not None
        assert after is not None
        threshold = self.settings.cache_similarity_threshold
        try:
            embedding = await self.gemini.embed_query(state.standalone)
        except GeminiError as exc:
            log.warning("cache.embed_failed", error_type=type(exc).__name__)
            return after, "Semantic cache: skipped (embedding failed)", {"hit": False}
        version = await asyncio.to_thread(self.stores.registry.version, request.collection)
        key = CacheKey(
            collection=request.collection,
            version=version,
            fingerprint=cache_fingerprint(
                self.settings,
                request,
                decision,
                self._options(state),
                self._retrieval_filters(state),
            ),
            question=state.standalone,
            embedding=embedding,
        )
        lookup = await asyncio.to_thread(self.stores.cache.lookup, key, threshold)
        state.cache_key, state.cache_lookup = key, lookup
        detail: dict[str, Any] = {
            "hit": lookup.hit is not None,
            "similarity": round(lookup.best_similarity, 4),
            "threshold": threshold,
            "candidates": lookup.candidates,
            "blocked_by_key_terms": lookup.blocked_by_key_terms,
        }
        if lookup.hit is None:
            if lookup.blocked_by_key_terms:
                why = " (a similar question asked about different numbers or codes)"
            elif lookup.candidates:
                why = f" (closest {lookup.best_similarity:.3f}, needs {threshold:.2f})"
            else:
                why = ""
            return after, f"Semantic cache: miss{why}", detail
        hit = lookup.hit
        state.generated = GeneratedAnswer(text=hit.answer.text, citations=hit.answer.citations)
        state.grounded = hit.answer.grounded
        await self._emit(state, hit.answer.text)
        detail |= {
            "cached_question": hit.question,
            "cached_at": hit.created_at.isoformat(),
            "hits": hit.hits,
        }
        return Node.DONE, f"Semantic cache: hit (similarity {hit.similarity:.3f})", detail

    async def _chitchat(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        state.generated = await chitchat_reply(
            self.gemini,
            state.request.question,
            history=state.request.history,
            titles=[d.title for d in state.documents],
            on_token=state.callbacks.on_token,
            on_reset=state.callbacks.on_reset,
        )
        return Node.DONE, "Replied conversationally (no retrieval)", {}

    async def _out_of_scope(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        titles = ", ".join(d.title for d in state.documents[:3]) or "none yet"
        text = render(OUT_OF_SCOPE_MESSAGE, count=len(state.documents), titles=titles)
        await self._emit(state, text)
        state.generated = GeneratedAnswer(text=text, refused=True)
        return Node.DONE, "Out of scope: declined politely", {}

    async def _ask_for_documents(
        self, state: AgentState, what: str, verb: str
    ) -> tuple[Node, str, dict[str, Any]]:
        text = render(
            ASK_WHICH_DOCUMENT, what=what, verb=verb, listing=_document_listing(state.documents)
        )
        await self._emit(state, text)
        state.generated = GeneratedAnswer(text=text)
        return Node.DONE, f"Asked which {what} to {verb}", {}

    async def _summarize(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        targets = self._target_documents(state)
        if not targets and len(state.documents) == 1:
            targets = state.documents
        if not targets:
            return await self._ask_for_documents(state, "document", "summarize")
        document = targets[0]
        parents = await asyncio.to_thread(
            self.stores.parents.for_document, state.request.collection, document.doc_id
        )
        result = await self.summarizer.summarize(
            state.standalone,
            document,
            parents,
            style=state.request.style,
            on_token=state.callbacks.on_token,
            on_reset=state.callbacks.on_reset,
        )
        state.target_document = document
        state.passages = result.passages
        state.generated = result.answer
        state.summary_map_reduce = result.map_reduce
        # A map-reduce summary is built from partial summaries, not the raw passages, so the
        # claim-by-claim check would be both expensive and unfair; it's skipped there.
        check = state.request.self_correct and not result.map_reduce
        method = "map-reduce" if result.map_reduce else "single pass"
        return (
            Node.CHECK_GROUNDEDNESS if check else Node.DONE,
            f"Summarised {document.filename} ({len(parents)} sections, {method})",
            {
                "document": document.filename,
                "sections": len(parents),
                "map_reduce": result.map_reduce,
            },
        )

    async def _compare(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        targets = self._target_documents(state)
        if len(targets) < 2 and len(state.documents) == 2:
            targets = state.documents
        if len(targets) < 2:
            return await self._ask_for_documents(state, "documents", "compare")
        targets = targets[: self.settings.max_compare_documents]
        # Each search is scoped to one document the user named, so the reranker only
        # reorders: its threshold, tuned for single questions, drops sections that a
        # multi-attribute comparison ("flight time and warranty") needs.
        options = replace(
            self._options(state), top_k=self.settings.compare_top_k_per_doc, rerank_threshold=0.0
        )
        results = await asyncio.gather(
            *(
                self.retriever.retrieve(
                    state.standalone,
                    collection=state.request.collection,
                    filters=SearchFilters(doc_ids=[doc.doc_id]),
                    options=options,
                )
                for doc in targets
            )
        )
        # Renumber passages globally so [n] is unique across documents.
        groups: list[ComparisonGroup] = []
        counter = 0
        for doc, result in zip(targets, results, strict=True):
            renumbered = []
            for passage in result.passages:
                counter += 1
                renumbered.append(passage.model_copy(update={"index": counter}))
            groups.append(ComparisonGroup(document=doc, passages=renumbered))
        state.retrievals.extend(results)
        state.comparison = groups
        state.passages = [p for g in groups for p in g.passages]
        state.generated = await self.comparer.compare(
            state.standalone,
            groups,
            style=state.request.style,
            on_token=state.callbacks.on_token,
            on_reset=state.callbacks.on_reset,
        )
        counts = ", ".join(f"{g.document.filename}: {len(g.passages)}" for g in groups)
        return (
            Node.CHECK_GROUNDEDNESS if state.request.self_correct else Node.DONE,
            f"Compared {len(groups)} documents (passages per document: {counts})",
            {"documents": [g.document.filename for g in groups]},
        )

    async def _retrieve(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        request = state.request
        options = self._options(state)
        attempt = len(state.tried_queries)
        state.tried_queries.append(state.query)
        # The router already condensed the question; only let retrieval condense again when
        # routing fell back (no model output) and there is history to resolve.
        history = () if (state.decision and state.decision.from_model) else request.history
        filters = self._retrieval_filters(state)
        result = await self.retriever.retrieve(
            state.query,
            collection=request.collection,
            history=history,
            filters=filters,
            options=options,
        )
        scoped = filters.doc_ids if filters is not None and filters.doc_ids else []
        state.retrievals.append(result)
        if attempt == 0:
            state.chunks = list(result.chunks)
        else:
            state.chunks = merge_round_robin(state.chunks, result.chunks, limit=options.top_k * 2)
        state.passages = await asyncio.to_thread(
            expand_to_parents,
            state.chunks,
            lambda ids: self.stores.parents.get_many(request.collection, ids),
            self.settings.context_token_budget,
        )
        next_node = Node.GRADE_RELEVANCE if request.self_correct else Node.GENERATE
        prefix = f"Retry {attempt}: " if attempt else ""
        return (
            next_node,
            (
                f"{prefix}Retrieved {len(result.chunks)} chunks from {len(result.candidates)} "
                f"candidates → {len(state.passages)} passages"
            ),
            {
                "query": state.query,
                "scope": [d.filename for d in self._documents_by_id(state, scoped)],
                "variants": result.plan.variants,
                "reranker": result.reranker,
                "candidates": [_trace_row(rc) for rc in result.candidates[:TRACE_ROWS]],
                "kept": [_trace_row(rc) for rc in result.chunks],
                "timings": {t.stage: round(t.ms, 1) for t in result.timings},
                "passages": [p.to_citation().location for p in state.passages],
            },
        )

    async def _grade_relevance(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        verdict = await self.relevance_grader.grade(state.standalone, state.passages)
        state.relevance.append(verdict)
        detail = {
            "sufficient": verdict.sufficient,
            "missing": verdict.missing,
            "checked": verdict.checked,
        }
        if verdict.sufficient:
            return Node.GENERATE, "Relevance: passages can answer the question", detail
        retries_used = len(state.tried_queries) - 1
        better = verdict.better_query
        fresh = better and better.lower() not in {q.lower() for q in state.tried_queries}
        # If a rewritten query also found nothing that passes the relevance threshold, the
        # documents almost certainly don't cover this; another rewrite only adds latency.
        last_empty = not state.retrievals[-1].chunks
        if retries_used > 0 and last_empty:
            return Node.GENERATE, "Relevance: the rewrite found nothing either; stopping", detail
        if retries_used < self.settings.max_retrieval_retries and fresh:
            state.query = better
            return (
                Node.RETRIEVE,
                f"Relevance: missing {verdict.missing or 'details'} → searching “{better}”",
                {
                    **detail,
                    "better_query": better,
                },
            )
        return (
            Node.GENERATE,
            "Relevance: still incomplete after retries; answering with what was found",
            detail,
        )

    async def _generate(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        state.generated = await self.generator.generate(
            state.standalone,
            state.passages,
            style=state.request.style,
            on_token=state.callbacks.on_token,
            on_reset=state.callbacks.on_reset,
        )
        if state.generated.refused:
            self._set_closest(state)
            return Node.DONE, "Answer: not found in the documents (showing closest matches)", {}
        next_node = Node.CHECK_GROUNDEDNESS if state.request.self_correct else Node.DONE
        return next_node, f"Generated answer with {len(state.generated.citations)} citations", {}

    async def _check_groundedness(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        generated = state.generated
        assert generated is not None
        if generated.refused:
            self._set_closest(state)
            return Node.DONE, "Groundedness: skipped (answer is a refusal)", {}
        cited = {c.index for c in generated.citations}
        evidence = [p for p in state.passages if p.index in cited] or state.passages
        verdict = await self.groundedness_grader.grade(generated.text, evidence)
        state.groundedness.append(verdict)
        detail = {
            "grounded": verdict.grounded,
            "unsupported_claims": verdict.unsupported_claims,
            "checked": verdict.checked,
        }
        if verdict.grounded:
            state.grounded = True if verdict.checked else None
            label = (
                "Groundedness: every claim is supported"
                if verdict.checked
                else "Groundedness: check unavailable"
            )
            return Node.DONE, label, detail
        if not state.regenerated:
            return (
                Node.REGENERATE,
                f"Groundedness: {len(verdict.unsupported_claims)} unsupported claim(s) "
                "→ regenerating",
                detail,
            )
        return Node.REFUSE_UNSUPPORTED, "Groundedness: still unsupported after regeneration", detail

    async def _regenerate(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        state.regenerated = True
        claims = state.groundedness[-1].unsupported_claims
        extra = render(STRICT_ANSWER_ADDENDUM, claims=_bullets(claims), refusal=REFUSAL_MESSAGE)
        await self._reset(state)  # clear the unsupported answer the user already saw
        cb = state.callbacks
        if state.route == Route.COMPARE:
            state.generated = await self.comparer.compare(
                state.standalone,
                state.comparison,
                style=state.request.style,
                on_token=cb.on_token,
                on_reset=cb.on_reset,
                extra_instructions=extra,
            )
        elif state.route == Route.SUMMARIZE and state.target_document is not None:
            parents = [p.parent for p in state.passages]
            result = await self.summarizer.summarize(
                state.standalone,
                state.target_document,
                parents,
                style=state.request.style,
                on_token=cb.on_token,
                on_reset=cb.on_reset,
                extra_instructions=extra,
            )
            state.generated = result.answer
        else:
            state.generated = await self.generator.generate(
                state.standalone,
                state.passages,
                style=state.request.style,
                on_token=cb.on_token,
                on_reset=cb.on_reset,
                extra_instructions=extra,
            )
        return Node.CHECK_GROUNDEDNESS, "Regenerated with stricter instructions", {"claims": claims}

    async def _refuse_unsupported(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        await self._reset(state)
        await self._emit(state, UNSUPPORTED_MESSAGE)
        state.generated = GeneratedAnswer(text=UNSUPPORTED_MESSAGE, refused=True)
        state.grounded = False
        self._set_closest(state)
        return Node.DONE, "Declined: the documents don't support a reliable answer", {}
