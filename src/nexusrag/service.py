"""Application service: one object that answers questions against a knowledge base.

The UI, the ``ask`` CLI and the evaluation suite all go through :class:`RAGService`,
so they exercise the same pipeline. Phase 3 runs retrieve -> generate. Later phases
put query transformation, reranking and the self-correcting agent behind the same
interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexusrag.config import Settings, get_settings
from nexusrag.generation.answer import AnswerGenerator, AnswerStyle, TokenCallback
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.llm.usage import track_usage
from nexusrag.log import get_logger, request_context, timed
from nexusrag.models import Answer, Route, SearchFilters, StageTiming
from nexusrag.retrieval.retriever import RetrievalResult, Retriever
from nexusrag.store import Stores
from nexusrag.store.registry import validate_collection_name

log = get_logger(__name__)


@dataclass
class AskResult:
    """An answer plus the retrieval trace behind it (for the UI and evaluation)."""

    answer: Answer
    retrieval: RetrievalResult


class RAGService:
    """Retrieval + grounded generation over the configured stores."""

    def __init__(self, settings: Settings, gemini: GeminiClient, stores: Stores) -> None:
        self.settings = settings
        self.gemini = gemini
        self.stores = stores
        self.retriever = Retriever(settings, gemini, stores)
        self.generator = AnswerGenerator(gemini)

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
        filters: SearchFilters | None = None,
        style: AnswerStyle = "detailed",
        on_token: TokenCallback | None = None,
        request_id: str | None = None,
    ) -> AskResult:
        """Answer ``question`` from the documents, streaming tokens to ``on_token``."""
        kb = validate_collection_name(collection or self.settings.default_collection)
        with (
            request_context(request_id, collection=kb, route=Route.DOC_QA.value) as rid,
            track_usage() as usage,
            timed() as total,
        ):
            retrieval = await self.retriever.retrieve(question, collection=kb, filters=filters)
            with timed() as gen:
                generated = await self.generator.generate(
                    question, retrieval.passages, style=style, on_token=on_token
                )
            totals = usage.totals()
            log.info(
                "rag.answer",
                chunks=len(retrieval.chunks),
                passages=len(retrieval.passages),
                cited=len(generated.citations),
                refused=generated.refused,
                generate_ms=round(gen.ms, 1),
                total_ms=round(total.ms, 1),
                prompt_tokens=totals.prompt_tokens,
                output_tokens=totals.output_tokens,
                cost_usd=round(totals.cost_usd, 6),
            )
        answer = Answer(
            text=generated.text,
            route=Route.DOC_QA,
            citations=generated.citations,
            refused=generated.refused,
            usage=usage.totals(),
            timings=[*retrieval.timings, StageTiming(stage="generate", ms=gen.ms)],
            request_id=rid,
        )
        return AskResult(answer=answer, retrieval=retrieval)

    def close(self) -> None:
        self.stores.close()
