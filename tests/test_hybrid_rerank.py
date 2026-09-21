"""Hybrid search, reranking and the configurable retrieval pipeline (fake Gemini + fake models)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nexusrag.config import Settings
from nexusrag.ingestion.pipeline import IngestionPipeline
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.models import ChatTurn, RetrievedChunk, SearchFilters
from nexusrag.retrieval.hybrid import HybridSearcher, SearchQuery
from nexusrag.retrieval.reranker import (
    CrossEncoderReranker,
    FallbackReranker,
    GeminiReranker,
    build_reranker,
    rerank,
)
from nexusrag.retrieval.retriever import RetrievalOptions, Retriever
from nexusrag.service import RAGService
from nexusrag.store import Stores
from tests.fakes import (
    FakeGenAI,
    hash_embedder,
    keyword_embedder,
    make_response,
    router_response,
)

DOCS = {
    "aurora.md": """# Aurora Spec

## Battery

A full charge takes 75 minutes with the CH-400 charger. Batteries are hot-swappable.

## Propellers

Inspect the propellers for cracks before every flight.
""",
    "policy.md": """# Remote Work Policy

## Stipends

Employees receive a home office stipend of 600 USD after probation.
""",
    "faq.txt": """SUPPORT FAQ

Q: How do I update the firmware? A: Use the Skylark Pilot app and keep the battery above 50%.
""",
}


class FakeCrossEncoder:
    """Scores a pair high when a keyword appears in both the query and the passage."""

    def __init__(self, keywords: Sequence[str]) -> None:
        self.keywords = [k.lower() for k in keywords]
        self.calls: list[int] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls.append(len(pairs))
        return [
            0.95 if any(k in q.lower() and k in text.lower() for k in self.keywords) else 0.01
            for q, text in pairs
        ]


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(storage_dir=tmp_path / "storage", top_k=3, dense_k=10, bm25_k=10)


@pytest.fixture
def stores(settings: Settings) -> Iterator[Stores]:
    s = Stores.open(settings)
    yield s
    s.close()


async def index(settings: Settings, gemini: GeminiClient, stores: Stores, tmp_path: Path) -> None:
    pipe = IngestionPipeline(settings, gemini, stores)
    for name, text in DOCS.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert (await pipe.ingest_file(path, collection="default")).status == "indexed"


def texts(chunks: Sequence[RetrievedChunk]) -> list[str]:
    return [rc.chunk.text for rc in chunks]


# --------------------------------------------------------------------------- hybrid search


async def test_bm25_rescues_exact_codes_that_dense_misses(
    settings: Settings, fake_genai: FakeGenAI, stores: Stores, tmp_path: Path
) -> None:
    # Random vectors: dense ranking is noise, so only BM25 can find the exact code.
    fake_genai.models.embed_fn = hash_embedder()
    gemini = GeminiClient(settings, client=fake_genai)
    await index(settings, gemini, stores, tmp_path)
    searcher = HybridSearcher(settings, gemini, stores)
    query = [SearchQuery("CH-400 charger", "original")]

    hybrid = await searcher.search(query, collection="default", use_bm25=True)
    top = hybrid.candidates[0]
    assert "CH-400" in top.chunk.text
    assert top.scores.bm25_rank == 1
    assert top.scores.rrf_score is not None
    assert top.matched_queries == ["CH-400 charger"]
    assert {r.retriever for r in hybrid.rankings} == {"dense", "bm25"}

    dense_only = await searcher.search(query, collection="default", use_bm25=False)
    assert {r.retriever for r in dense_only.rankings} == {"dense"}
    assert all(rc.scores.bm25_rank is None for rc in dense_only.candidates)


async def test_hyde_is_dense_only_and_filters_apply(
    settings: Settings, fake_genai: FakeGenAI, stores: Stores, tmp_path: Path
) -> None:
    fake_genai.models.embed_fn = keyword_embedder()
    gemini = GeminiClient(settings, client=fake_genai)
    await index(settings, gemini, stores, tmp_path)
    searcher = HybridSearcher(settings, gemini, stores)
    queries = [
        SearchQuery("stipend", "original"),
        SearchQuery("Employees get a stipend of 600 USD.", "hyde"),
    ]
    result = await searcher.search(
        queries, collection="default", filters=SearchFilters(filenames=["policy.md"])
    )
    assert [(r.retriever, r.query.kind) for r in result.rankings] == [
        ("dense", "original"),
        ("dense", "hyde"),
        ("bm25", "original"),
    ]
    assert {rc.chunk.metadata.filename for rc in result.candidates} == {"policy.md"}
    # HyDE text is embedded as a document, not with the query prefix.
    embedded = [c["contents"] for c in fake_genai.models.calls_to("embed_content")][-1]
    assert embedded[0].parts[0].text.startswith("title: none | text: Employees")


# --------------------------------------------------------------------------- reranking


def candidates_from(chunk_texts: Sequence[str]) -> list[RetrievedChunk]:
    from datetime import UTC, datetime

    from nexusrag.models import Chunk, ChunkMetadata, SourceType, StageScores

    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                text=text,
                token_count=5,
                metadata=ChunkMetadata(
                    doc_id="d",
                    collection="kb",
                    filename="f.md",
                    source_type=SourceType.MARKDOWN,
                    title="T",
                    chunk_index=i,
                    parent_id=f"p{i}",
                    content_hash=str(i),
                    ingested_at=datetime(2026, 9, 21, tzinfo=UTC),
                ),
            ),
            scores=StageScores(rrf_score=1 / (60 + i)),
        )
        for i, text in enumerate(chunk_texts)
    ]


async def test_cross_encoder_reorders_and_applies_threshold() -> None:
    model = FakeCrossEncoder(["charger"])
    reranker = CrossEncoderReranker("fake-model", loader=lambda name, max_len: model)
    pool = candidates_from(["propellers", "stipend", "the CH-400 charger", "firmware"])
    kept = await rerank(reranker, "charger?", pool, top_k=3, threshold=0.05, max_candidates=10)
    assert texts(kept) == ["the CH-400 charger"]
    assert kept[0].scores.rerank_score == pytest.approx(0.95)
    assert kept[0].scores.rrf_score == pool[2].scores.rrf_score  # earlier scores preserved
    assert pool[2].scores.rerank_score is None  # inputs not mutated


async def test_rerankers_read_linearised_tables() -> None:
    seen: list[str] = []

    class Recorder(FakeCrossEncoder):
        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
            seen.extend(text for _, text in pairs)
            return [0.5] * len(pairs)

    reranker = CrossEncoderReranker("fake-model", loader=lambda name, max_len: Recorder([]))
    table = "| Metric | Value |\n|---|---|\n| Range | 12 km |"
    await reranker.score("q", candidates_from([table]))
    assert seen == ["Range: 12 km"]


async def test_first_stage_consensus_is_not_buried_by_one_reranker_score() -> None:
    """RRF of hybrid order + reranker order keeps a #1 hybrid hit the reranker under-scores."""
    pool = candidates_from(["target", "b", "c", "d", "e", "f"])  # hybrid order: target first
    scores = {"target": 0.2, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6, "f": 0.5}

    class Fixed(FakeCrossEncoder):
        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
            return [scores[text] for _, text in pairs]

    reranker = CrossEncoderReranker("fake-model", loader=lambda name, max_len: Fixed([]))
    fused = await rerank(reranker, "q", pool, top_k=3, threshold=0.1, max_candidates=10)
    pure = await rerank(
        reranker, "q", pool, top_k=3, threshold=0.1, max_candidates=10, fusion_weight=0.0
    )
    assert "target" in texts(fused)
    assert texts(pure) == ["b", "c", "d"]


async def test_only_top_candidates_are_rescored() -> None:
    model = FakeCrossEncoder(["x"])
    reranker = CrossEncoderReranker("fake-model", loader=lambda name, max_len: model)
    pool = candidates_from([f"text {i}" for i in range(50)])
    await rerank(reranker, "q", pool, top_k=5, threshold=0.0, max_candidates=30)
    assert model.calls == [30]


async def test_model_loads_once() -> None:
    loads: list[str] = []

    def loader(name: str, max_len: int) -> FakeCrossEncoder:
        loads.append(name)
        return FakeCrossEncoder(["a"])

    reranker = CrossEncoderReranker("fake-model", loader=loader)
    for _ in range(3):
        await reranker.score("q", candidates_from(["a"]))
    assert loads == ["fake-model"]


async def test_gemini_reranker_parses_and_normalises(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(
        make_response(json.dumps({"scores": [{"id": 2, "score": 9}, {"id": 1, "score": 3}]}))
    )
    scores = await GeminiReranker(gemini).score("q", candidates_from(["a", "b", "c"]))
    assert scores == [pytest.approx(0.3), pytest.approx(0.9), 0.0]  # missing id 3 -> 0
    assert fake_genai.models.calls[0]["model"] == "gemini-3.5-flash-lite"


async def test_fallback_switches_once_when_local_model_fails(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    attempts: list[int] = []

    def broken_loader(name: str, max_len: int) -> Any:
        attempts.append(1)
        raise ImportError("No module named 'sentence_transformers'")

    fake_genai.models.generate_fn = lambda prompt: make_response(
        json.dumps({"scores": [{"id": 1, "score": 8}]})
    )
    reranker = FallbackReranker(
        CrossEncoderReranker("fake-model", loader=broken_loader), GeminiReranker(gemini)
    )
    assert reranker.name == "cross_encoder"
    for _ in range(2):
        assert await reranker.score("q", candidates_from(["a"])) == [pytest.approx(0.8)]
    assert reranker.name == "gemini"
    assert attempts == [1]  # the broken model isn't retried on every query


def test_fallback_warm_up_loads_the_primary_or_switches(gemini: GeminiClient) -> None:
    loaded: list[str] = []

    def loader(name: str, max_len: int) -> Any:
        loaded.append(name)
        return object()

    ok = FallbackReranker(CrossEncoderReranker("fake-model", loader=loader), GeminiReranker(gemini))
    ok.warm_up()
    ok.warm_up()
    assert loaded == ["fake-model"]
    assert ok.name == "cross_encoder"

    def broken(name: str, max_len: int) -> Any:
        loaded.append("broken")
        raise OSError("download failed")

    bad = FallbackReranker(
        CrossEncoderReranker("fake-model", loader=broken), GeminiReranker(gemini)
    )
    bad.warm_up()
    bad.warm_up()
    assert bad.name == "gemini"
    assert loaded.count("broken") == 1  # not retried

    FallbackReranker(GeminiReranker(gemini), GeminiReranker(gemini)).warm_up()  # nothing to load


def test_build_reranker_backends(
    make_settings: Callable[..., Settings], gemini: GeminiClient
) -> None:
    assert build_reranker(make_settings(reranker_backend="gemini"), gemini).name == "gemini"
    default = build_reranker(make_settings(), gemini)
    assert isinstance(default, FallbackReranker)
    assert default.name == "cross_encoder"


# --------------------------------------------------------------------------- full pipeline


@pytest.fixture
async def full_service(
    settings: Settings, fake_genai: FakeGenAI, stores: Stores, tmp_path: Path
) -> RAGService:
    fake_genai.models.embed_fn = keyword_embedder()
    gemini = GeminiClient(settings, client=fake_genai)
    await index(settings, gemini, stores, tmp_path)
    reranker = CrossEncoderReranker(
        "fake-model", loader=lambda name, max_len: FakeCrossEncoder(["charg"])
    )
    return RAGService(settings, gemini, stores, reranker=reranker)


async def test_full_pipeline_with_all_stages(
    full_service: RAGService, fake_genai: FakeGenAI
) -> None:
    def route(prompt: str) -> object:
        if "You route messages" in prompt:  # the router condenses the follow-up
            return router_response(standalone="How long does charging the battery take?")
        if "standalone search question" in prompt:
            return make_response(
                json.dumps({"standalone_question": "How long does charging the battery take?"})
            )
        if "alternative search queries" in prompt:
            return make_response(
                json.dumps({"queries": ["battery charge time", "CH-400 charger duration"]})
            )
        return make_response("A full charge takes 75 minutes.")  # HyDE passage

    fake_genai.models.generate_fn = route
    fake_genai.models.stream_queue.append([make_response("It takes 75 minutes [1].")])
    options = RetrievalOptions(
        top_k=3, hybrid=True, multi_query=True, num_variants=2, hyde=True, rerank=True
    )
    history = [ChatTurn(role="user", content="Tell me about the Aurora battery")]

    result = await full_service.ask("How long to charge it?", history=history, options=options)
    retrieval = result.retrieval
    assert retrieval.query == "How long does charging the battery take?"
    # The router already condensed the follow-up, so retrieval searches it as-is.
    assert [q.kind for q in retrieval.plan.queries] == ["original", "variant", "variant", "hyde"]
    assert retrieval.reranker == "cross_encoder"
    assert [t.stage for t in retrieval.timings] == [
        "query_transform",
        "hybrid_search",
        "rerank",
        "parent_expansion",
    ]
    assert len(retrieval.candidates) > len(retrieval.chunks)
    assert all("charg" in rc.chunk.text.lower() for rc in retrieval.chunks)
    assert all(rc.scores.rerank_score == pytest.approx(0.95) for rc in retrieval.chunks)
    # The answer model receives the standalone question, not the ambiguous follow-up.
    stream_call = fake_genai.models.calls_to("generate_content_stream")[0]
    assert "Question: How long does charging the battery take?" in stream_call["contents"]
    assert result.answer.citations[0].index == 1


async def test_threshold_can_empty_the_context_and_trigger_refusal(
    full_service: RAGService, fake_genai: FakeGenAI
) -> None:
    options = RetrievalOptions(top_k=3, multi_query=False, rerank=True)
    result = await full_service.ask("What is the CEO's favourite colour?", options=options)
    assert result.retrieval.candidates  # something was found...
    assert result.retrieval.chunks == []  # ...but nothing passed the relevance threshold
    assert result.answer.refused
    assert fake_genai.models.calls_to("generate_content_stream") == []


async def test_rerank_failure_keeps_fused_order(
    settings: Settings, fake_genai: FakeGenAI, stores: Stores, tmp_path: Path
) -> None:
    fake_genai.models.embed_fn = keyword_embedder()
    gemini = GeminiClient(settings, client=fake_genai)
    await index(settings, gemini, stores, tmp_path)

    class Exploding:
        name = "exploding"

        async def score(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
            raise RuntimeError("model crashed")

    retriever = Retriever(settings, gemini, stores, reranker=Exploding())
    result = await retriever.retrieve(
        "battery charger",
        collection="default",
        options=RetrievalOptions(top_k=2, multi_query=False, rerank=True),
    )
    assert result.reranker is None
    assert result.chunks == result.candidates[:2]


async def test_rerank_threshold_can_be_relaxed_per_request(
    settings: Settings, fake_genai: FakeGenAI, stores: Stores, tmp_path: Path
) -> None:
    fake_genai.models.embed_fn = keyword_embedder()
    gemini = GeminiClient(settings, client=fake_genai)
    await index(settings, gemini, stores, tmp_path)
    reranker = CrossEncoderReranker(
        "fake-model", loader=lambda name, max_len: FakeCrossEncoder(["charg"])
    )
    retriever = Retriever(settings, gemini, stores, reranker=reranker)
    options = RetrievalOptions(top_k=3, multi_query=False, rerank=True)

    strict = await retriever.retrieve("battery charging", collection="default", options=options)
    assert len(strict.chunks) == 1  # only the charging passage clears the threshold

    relaxed = await retriever.retrieve(
        "battery charging",
        collection="default",
        options=replace(options, rerank_threshold=0.0),
    )
    assert len(relaxed.chunks) == 3  # reordered, nothing dropped
    assert strict.chunks[0].chunk.chunk_id in {rc.chunk.chunk_id for rc in relaxed.chunks}


def test_options_from_settings(make_settings: Callable[..., Settings]) -> None:
    opts = RetrievalOptions.from_settings(make_settings(top_k=7, enable_hyde=True), rerank=False)
    assert (opts.top_k, opts.hyde, opts.rerank) == (7, True, False)
