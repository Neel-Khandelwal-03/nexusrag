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

Self-correction (relevance grading, groundedness checking) can be switched off per
request, which is how the evaluation suite measures what it adds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
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
from nexusrag.llm.gemini_client import GeminiClient
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

log = get_logger(__name__)

StepCallback = Callable[[AgentStep], Awaitable[None]]

#: Hard stop for runaway loops; the longest legitimate path is about 10 steps.
MAX_STEPS = 20

UNSUPPORTED_MESSAGE = (
    f"{REFUSAL_MESSAGE} The passages I found don't support a reliable answer, so I'm showing "
    "the closest matches instead of guessing."
)


class Node(StrEnum):
    ROUTE = "route"
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
    ) -> AgentResult:
        documents = await asyncio.to_thread(self.stores.registry.list_documents, request.collection)
        state = AgentState(
            request=request, documents=documents, callbacks=_Callbacks(on_token, on_reset)
        )
        node = Node.ROUTE
        for _ in range(MAX_STEPS):
            if node is Node.DONE:
                break
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
        return self._result(state)

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
        )

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

    def _set_closest(self, state: AgentState) -> None:
        candidates = [rc for r in state.retrievals for rc in r.candidates]
        state.closest = closest_matches(candidates, limit=3)

    # ------------------------------------------------------------------ nodes

    async def _route(self, state: AgentState) -> tuple[Node, str, dict[str, Any]]:
        request = state.request
        decision = await self.router.route(
            request.question, documents=state.documents, history=request.history
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
        options = replace(self._options(state), top_k=self.settings.compare_top_k_per_doc)
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
        result = await self.retriever.retrieve(
            state.query,
            collection=request.collection,
            history=history,
            filters=request.filters,
            options=options,
        )
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
                "variants": result.plan.variants,
                "reranker": result.reranker,
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
