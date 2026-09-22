"""Dense retrieval, parent expansion and the end-to-end RAG service (fake Gemini)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nexusrag.config import Settings
from nexusrag.ingestion.pipeline import IngestionPipeline
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.models import (
    Chunk,
    ChunkMetadata,
    ParentSection,
    RetrievedChunk,
    Route,
    SearchFilters,
    SourceType,
)
from nexusrag.retrieval.reranker import CrossEncoderReranker
from nexusrag.retrieval.retriever import expand_to_parents, where_for
from nexusrag.service import RAGService
from nexusrag.store import Stores
from nexusrag.ui.render import source_panel_markdown, sources_footer, welcome_markdown
from tests.fakes import FakeGenAI, keyword_embedder, make_response

SPEC = """# Aurora Spec

## Battery

The battery lasts 46 minutes. Charging the battery takes 75 minutes with the CH-400 charger.

## Warranty

The warranty covers manufacturing defects for 12 months. Crash damage is excluded.
"""

POLICY = """# Remote Work Policy

## Stipends

Employees receive a home office stipend of 600 USD and an internet allowance of 50 USD monthly.
"""


def rc(chunk_id: str, parent_id: str, text: str = "child") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            token_count=1,
            metadata=ChunkMetadata(
                doc_id="d",
                collection="kb",
                filename="f.md",
                source_type=SourceType.MARKDOWN,
                title="T",
                section_path="S",
                chunk_index=0,
                parent_id=parent_id,
                content_hash=chunk_id,
                ingested_at=datetime(2026, 9, 21, tzinfo=UTC),
            ),
        )
    )


def parent(parent_id: str, tokens: int, text: str | None = None) -> ParentSection:
    return ParentSection(
        parent_id=parent_id,
        doc_id="d",
        collection="kb",
        filename="f.md",
        source_type=SourceType.MARKDOWN,
        title="T",
        text=text or f"text of {parent_id}",
        token_count=tokens,
    )


# --------------------------------------------------------------------------- parent expansion


def test_expand_groups_children_and_keeps_rank_order() -> None:
    parents = {"pA": parent("pA", 100), "pB": parent("pB", 100)}
    chunks = [rc("c1", "pB"), rc("c2", "pA"), rc("c3", "pB")]
    passages = expand_to_parents(chunks, lambda ids: [parents[i] for i in ids], 1000)
    assert [(p.index, p.parent.parent_id) for p in passages] == [(1, "pB"), (2, "pA")]
    assert [c.chunk.chunk_id for c in passages[0].chunks] == ["c1", "c3"]


def test_expand_respects_budget_but_keeps_smaller_later_parents() -> None:
    parents = {"p1": parent("p1", 600), "p2": parent("p2", 600), "p3": parent("p3", 300)}
    chunks = [rc("c1", "p1"), rc("c2", "p2"), rc("c3", "p3")]
    passages = expand_to_parents(chunks, lambda ids: [parents[i] for i in ids], 1000)
    assert [p.parent.parent_id for p in passages] == ["p1", "p3"]
    assert [p.index for p in passages] == [1, 2]


def test_expand_truncates_an_oversized_top_parent() -> None:
    big = parent("p1", 5000, text="word " * 5000)
    passages = expand_to_parents([rc("c1", "p1")], lambda ids: [big], 100)
    assert len(passages) == 1
    assert passages[0].parent.token_count <= 100


def test_expand_falls_back_to_child_text_when_parent_missing() -> None:
    passages = expand_to_parents([rc("c1", "gone", "orphan child text")], lambda ids: [], 1000)
    assert passages[0].parent.text == "orphan child text"


def test_where_for_filters() -> None:
    assert where_for(None) is None
    assert where_for(SearchFilters()) is None
    assert where_for(SearchFilters(source_types=[SourceType.PDF])) == {
        "source_type": {"$in": ["pdf"]}
    }


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def service(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI, tmp_path: Path
) -> Iterator[RAGService]:
    settings = make_settings(storage_dir=tmp_path / "storage", top_k=2)
    fake_genai.models.embed_fn = keyword_embedder()
    stores = Stores.open(settings)
    svc = RAGService(settings, GeminiClient(settings, client=fake_genai), stores)
    yield svc
    svc.close()


async def ingest(svc: RAGService, tmp_path: Path) -> None:
    pipe = IngestionPipeline(svc.settings, svc.gemini, svc.stores)
    for name, text in (("aurora.md", SPEC), ("policy.md", POLICY)):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert (await pipe.ingest_file(path, collection="default")).status == "indexed"


async def test_dense_retrieval_ranks_relevant_section_first(
    service: RAGService, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    result = await service.retriever.retrieve(
        "how long does charging the battery take", collection="default"
    )
    top = result.chunks[0]
    assert top.chunk.metadata.section_path == "Battery"
    assert top.scores.dense_rank == 1
    assert top.scores.dense_score is not None
    assert result.passages[0].parent.parent_id == top.chunk.metadata.parent_id
    assert [t.stage for t in result.timings] == [
        "query_transform",
        "hybrid_search",
        "parent_expansion",
    ]


async def test_filters_restrict_retrieval(service: RAGService, tmp_path: Path) -> None:
    await ingest(service, tmp_path)
    result = await service.retriever.retrieve(
        "battery", collection="default", filters=SearchFilters(filenames=["policy.md"])
    )
    assert {c.chunk.metadata.filename for c in result.chunks} == {"policy.md"}


async def test_service_answers_with_citations_and_usage(
    service: RAGService, fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    fake_genai.models.stream_queue.append(
        [make_response("Charging takes 75 minutes [1].", prompt_tokens=400, output_tokens=9)]
    )
    tokens: list[str] = []

    async def on_token(token: str) -> None:
        tokens.append(token)

    result = await service.ask("How long does charging take?", on_token=on_token)
    answer = result.answer
    assert tokens == ["Charging takes 75 minutes [1]."]
    assert answer.route == Route.DOC_QA
    assert [c.index for c in answer.citations] == [1]
    assert answer.citations[0].filename == "aurora.md"
    assert answer.citations[0].highlights  # the matched child chunk is attached
    assert answer.usage.prompt_tokens == 400
    assert answer.usage.embedding_tokens > 0
    assert answer.request_id
    assert answer.timings[-1].stage == "generate"


async def test_service_refuses_on_empty_knowledge_base(
    service: RAGService, fake_genai: FakeGenAI
) -> None:
    result = await service.ask("Anything at all?")
    assert result.answer.refused
    assert fake_genai.models.calls_to("generate_content_stream") == []


async def test_render_helpers(service: RAGService, fake_genai: FakeGenAI, tmp_path: Path) -> None:
    await ingest(service, tmp_path)
    fake_genai.models.stream_queue.append([make_response("Stipend is 600 USD [1].")])
    answer = (await service.ask("What is the home office stipend?")).answer
    citation = answer.citations[0]
    footer = sources_footer(answer.citations)
    assert footer.startswith("**Sources**")
    assert f"[1] {citation.location}" in footer
    panel = source_panel_markdown(citation)
    assert "**Matched excerpt**" in panel
    assert citation.text in panel
    docs = service.stores.registry.list_documents("default")
    assert "2 documents" in welcome_markdown("default", docs)
    assert "empty" in welcome_markdown("default", [])


async def test_stats_counts_the_knowledge_base(service: RAGService, tmp_path: Path) -> None:
    empty = service.stats()
    assert (empty.documents, empty.parents, empty.chunks, empty.vectors) == ([], 0, 0, 0)
    await ingest(service, tmp_path)
    stats = service.stats("default")
    assert sorted(d.filename for d in stats.documents) == ["aurora.md", "policy.md"]
    assert stats.chunks == sum(d.num_chunks for d in stats.documents) == stats.vectors
    assert stats.parents >= 2
    assert "default" in stats.collections
    assert stats.cache_lookups is None  # the semantic cache arrives in phase 7


def test_warm_up_loads_the_reranker_only_when_enabled(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    loaded: list[bool] = []

    def loader(name: str, max_len: int) -> object:
        loaded.append(True)
        return object()

    for enabled in (False, True):
        settings = make_settings(storage_dir=tmp_path / f"storage-{enabled}", enable_rerank=enabled)
        reranker = CrossEncoderReranker("fake-model", loader=loader)
        svc = RAGService(
            settings,
            GeminiClient(settings, client=fake_genai),
            Stores.open(settings),
            reranker=reranker,
        )
        svc.warm_up()
        svc.warm_up()
        svc.close()
    assert loaded == [True]


async def test_stats_and_clear_cache_with_the_cache_on(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    settings = make_settings(storage_dir=tmp_path / "storage", top_k=2, enable_semantic_cache=True)
    fake_genai.models.embed_fn = keyword_embedder()
    svc = RAGService(settings, GeminiClient(settings, client=fake_genai), Stores.open(settings))
    await ingest(svc, tmp_path)
    fake_genai.models.stream_queue.append([make_response("Stipend is 600 USD [1].")])
    first = await svc.ask("What is the home office stipend?")
    second = await svc.ask("What is the home office stipend?")
    assert (first.answer.cached, second.answer.cached) == (False, True)
    stats = svc.stats()
    assert (stats.cache_hits, stats.cache_lookups, stats.cache_entries) == (1, 2, 1)
    assert svc.clear_cache() == 1
    assert svc.stats().cache_entries == 0
    fake_genai.models.stream_queue.append([make_response("Stipend is 600 USD [1].")])
    third = await svc.ask("What is the home office stipend?", use_cache=False)
    assert "check_cache" not in [s.node for s in third.answer.steps]
    svc.close()
