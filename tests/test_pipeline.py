"""Incremental indexing, dedup and stale-chunk cleanup, end to end with a fake Gemini."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from nexusrag import ingest
from nexusrag.config import Settings
from nexusrag.ingestion import pipeline as pipeline_module
from nexusrag.ingestion.pipeline import IngestionPipeline, ProgressEvent
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.store import Stores
from tests.fakes import FakeGenAI, hash_embedder

DOC_V1 = """# Drone Guide

## Battery

The battery lasts 46 minutes and charges in 75 minutes with the CH-400 charger.

## Propellers

Inspect propellers for cracks before every flight and replace them every 50 hours.
"""


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(storage_dir=tmp_path / "storage", data_dir=tmp_path / "docs")


@pytest.fixture
def fake(fake_genai: FakeGenAI) -> FakeGenAI:
    fake_genai.models.embed_fn = hash_embedder()
    return fake_genai


@pytest.fixture
def pipe(settings: Settings, fake: FakeGenAI) -> IngestionPipeline:
    stores = Stores.open(settings)
    yield IngestionPipeline(settings, GeminiClient(settings, client=fake), stores)  # type: ignore[misc]
    stores.close()


def embedded_count(fake: FakeGenAI) -> int:
    return sum(len(c["contents"]) for c in fake.models.calls_to("embed_content"))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_index_then_skip_unchanged(
    pipe: IngestionPipeline, fake: FakeGenAI, tmp_path: Path
) -> None:
    path = write(tmp_path / "guide.md", DOC_V1)
    events: list[ProgressEvent] = []

    first = await pipe.ingest_file(path, collection="kb", progress=events.append)
    assert first.status == "indexed"
    assert first.num_chunks >= 2
    assert [e.stage for e in events] == ["parsing", "chunking", "embedding", "storing", "done"]

    s = pipe.stores
    assert s.vectors.count("kb") == first.num_chunks
    assert s.bm25.count("kb") == first.num_chunks
    assert s.parents.count("kb") == first.num_parents
    assert s.registry.version("kb") == 1

    calls_before = embedded_count(fake)
    second = await pipe.ingest_file(path, collection="kb")
    assert second.status == "skipped"
    assert embedded_count(fake) == calls_before  # no embedding for unchanged files
    assert s.registry.version("kb") == 1


async def test_changed_file_replaces_stale_chunks_and_reuses_vectors(
    pipe: IngestionPipeline, fake: FakeGenAI, tmp_path: Path
) -> None:
    path = write(tmp_path / "guide.md", DOC_V1)
    first = await pipe.ingest_file(path, collection="kb")
    embedded_v1 = embedded_count(fake)

    v2 = DOC_V1.replace(
        "Inspect propellers for cracks before every flight and replace them every 50 hours.",
        "Swap propellers every 80 hours using the QuickLock tool.",
    )
    write(path, v2)
    second = await pipe.ingest_file(path, collection="kb")

    assert second.status == "updated"
    assert second.doc_id == first.doc_id
    assert second.reused_embeddings >= 1  # the unchanged battery section kept its vector
    assert embedded_count(fake) - embedded_v1 == second.num_chunks - second.reused_embeddings

    s = pipe.stores
    assert s.vectors.count("kb") == second.num_chunks
    assert s.bm25.count("kb") == second.num_chunks
    assert s.bm25.search("kb", "cracks", k=5) == []  # stale text is gone from BM25
    assert [h.chunk_id for h in s.bm25.search("kb", "QuickLock", k=5)]
    assert s.registry.version("kb") == 2


async def test_force_reindex_reuses_all_vectors(
    pipe: IngestionPipeline, fake: FakeGenAI, tmp_path: Path
) -> None:
    path = write(tmp_path / "guide.md", DOC_V1)
    await pipe.ingest_file(path, collection="kb")
    before = embedded_count(fake)
    result = await pipe.ingest_file(path, collection="kb", force=True)
    assert result.status == "updated"
    assert result.reused_embeddings == result.num_chunks
    assert embedded_count(fake) == before


async def test_directory_ingest_with_prune(pipe: IngestionPipeline, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    write(docs / "guide.md", DOC_V1)
    write(docs / "sub" / "faq.txt", "SUPPORT\n\nQ: Hours? A: Weekdays 07:00 to 19:00.")
    write(docs / ".hidden" / "skip.md", "# Hidden\n\nShould not be indexed.")
    write(docs / "notes.xyz", "unsupported, ignored by directory scans")

    results = await pipe.ingest_directory(docs, collection="kb")
    assert sorted((r.filename, r.status) for r in results) == [
        ("guide.md", "indexed"),
        ("sub/faq.txt", "indexed"),
    ]

    (docs / "sub" / "faq.txt").unlink()
    results = await pipe.ingest_directory(docs, collection="kb", prune=True)
    assert sorted((r.filename, r.status) for r in results) == [
        ("guide.md", "skipped"),
        ("sub/faq.txt", "deleted"),
    ]
    remaining = pipe.stores.registry.list_documents("kb")
    assert [r.filename for r in remaining] == ["guide.md"]
    assert pipe.stores.vectors.count("kb") == remaining[0].num_chunks


async def test_failures_are_reported_not_raised(pipe: IngestionPipeline, tmp_path: Path) -> None:
    bad = write(tmp_path / "empty.md", "   ")
    events: list[ProgressEvent] = []
    result = await pipe.ingest_file(bad, collection="kb", progress=events.append)
    assert result.status == "failed"
    assert "No extractable text" in (result.error or "")
    assert events[-1].stage == "failed"

    unsupported = write(tmp_path / "x.exe", "MZ")
    assert (await pipe.ingest_file(unsupported, collection="kb")).status == "failed"

    bad_name = await pipe.ingest_file(write(tmp_path / "ok.md", DOC_V1), collection="../evil")
    assert bad_name.status == "failed"
    assert "Invalid knowledge base name" in (bad_name.error or "")


async def test_delete_document(pipe: IngestionPipeline, tmp_path: Path) -> None:
    result = await pipe.ingest_file(write(tmp_path / "guide.md", DOC_V1), collection="kb")
    assert await pipe.delete_document("kb", result.doc_id or "")
    s = pipe.stores
    assert (s.vectors.count("kb"), s.bm25.count("kb"), s.parents.count("kb")) == (0, 0, 0)
    assert not await pipe.delete_document("kb", result.doc_id or "")


def write_pdf(path: Path) -> Path:
    import pymupdf

    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Drone Guide", fontsize=20)
        page.insert_text((72, 110), "The battery lasts 46 minutes and charges in 75 minutes.")
        doc.save(path)
    return path


async def test_pdfs_are_kept_for_previews_and_removed_on_delete(
    pipe: IngestionPipeline, settings: Settings, tmp_path: Path
) -> None:
    pdf = await pipe.ingest_file(write_pdf(tmp_path / "guide.pdf"), collection="kb")
    assert pdf.status == "indexed"
    assert pdf.doc_id
    kept = settings.source_file_path("kb", pdf.doc_id)
    assert kept.read_bytes() == (tmp_path / "guide.pdf").read_bytes()

    # A PDF indexed before copies were kept gets one on its next (skipped) ingest.
    kept.unlink()
    again = await pipe.ingest_file(tmp_path / "guide.pdf", collection="kb")
    assert again.status == "skipped"
    assert kept.is_file()

    md = await pipe.ingest_file(write(tmp_path / "notes.md", DOC_V1), collection="kb")
    assert md.doc_id
    assert not settings.source_file_path("kb", md.doc_id).exists()

    assert await pipe.delete_document("kb", pdf.doc_id)
    assert not kept.exists()


async def test_collections_are_isolated(pipe: IngestionPipeline, tmp_path: Path) -> None:
    path = write(tmp_path / "guide.md", DOC_V1)
    await pipe.ingest_file(path, collection="alpha")
    assert (await pipe.ingest_file(path, collection="beta")).status == "indexed"
    assert pipe.stores.bm25.search("gamma-missing", "battery", k=3) == []
    assert pipe.stores.bm25.search("beta", "battery", k=3)


async def test_ingest_url(pipe: IngestionPipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    html = (
        "<html><head><title>Pairing Guide</title></head><body><article><h1>Pairing Guide</h1><p>"
        + "Hold the pairing button for five seconds until the light blinks. " * 10
        + "</p></article></body></html>"
    )

    async def fake_fetch(url: str, **kwargs: object) -> tuple[str, str]:
        return html, url

    monkeypatch.setattr(pipeline_module, "fetch_url", fake_fetch)
    result = await pipe.ingest_url("https://example.com/pairing", collection="kb")
    assert result.status == "indexed"
    assert result.title == "Pairing Guide"
    [record] = pipe.stores.registry.list_documents("kb")
    assert record.source == "https://example.com/pairing"


def test_cli_ingest_and_list(
    settings: Settings,
    fake: FakeGenAI,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    write(tmp_path / "docs" / "guide.md", DOC_V1)
    monkeypatch.setattr(ingest, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest, "GeminiClient", lambda s: GeminiClient(s, client=fake))

    assert ingest.main([str(tmp_path / "docs"), "--collection", "kb"]) == 0
    out = capsys.readouterr().out
    assert "[indexed] guide.md" in out
    assert "1 indexed" in out

    assert ingest.main(["--list", "--collection", "kb"]) == 0
    assert "guide.md" in capsys.readouterr().out
    assert ingest.main(["--list-collections"]) == 0
    assert "kb" in capsys.readouterr().out
