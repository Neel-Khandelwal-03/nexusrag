from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nexusrag.models import Chunk, ChunkMetadata, ParentSection, SourceType
from nexusrag.store.bm25_index import BM25Index, tokenize
from nexusrag.store.parent_store import ParentStore
from nexusrag.store.registry import (
    DocumentRecord,
    InvalidCollectionName,
    Registry,
    validate_collection_name,
)
from nexusrag.store.sqlite_db import Database
from nexusrag.store.vector_store import EmbeddingMismatchError, VectorStore, build_where

NOW = datetime(2026, 9, 21, tzinfo=UTC)


def chunk(cid: str, text: str, doc_id: str = "d1", filename: str = "a.md") -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        token_count=len(text.split()),
        metadata=ChunkMetadata(
            doc_id=doc_id,
            collection="kb",
            filename=filename,
            source_type=SourceType.MARKDOWN,
            title="T",
            section_path="S",
            page_start=1,
            chunk_index=0,
            parent_id=f"{doc_id}-p0000",
            content_hash=cid,
            ingested_at=NOW,
        ),
    )


def parent(pid: str, doc_id: str = "d1", text: str = "body") -> ParentSection:
    return ParentSection(
        parent_id=pid,
        doc_id=doc_id,
        collection="kb",
        filename="a.md",
        source_type=SourceType.MARKDOWN,
        title="T",
        section_path="S",
        text=text,
        token_count=1,
    )


@pytest.fixture
def db() -> Database:
    return Database(":memory:")


# --------------------------------------------------------------------------- sqlite / registry


@pytest.mark.parametrize(
    ("raw", "expected"), [("Default", "default"), ("my kb", "my-kb"), ("a", "a")]
)
def test_collection_names(raw: str, expected: str) -> None:
    assert validate_collection_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "-bad", "bad-", "x" * 51, "../etc", "semi;colon"])
def test_invalid_collection_names(raw: str) -> None:
    with pytest.raises(InvalidCollectionName):
        validate_collection_name(raw)


def test_nested_transaction_rolls_back_everything(db: Database) -> None:
    registry = Registry(db)

    def failing_unit_of_work() -> None:
        with db.transaction():
            registry.ensure_collection("kb")
            with db.transaction():
                registry.bump_version("kb")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        failing_unit_of_work()
    assert registry.list_collections() == []


def test_registry_documents_and_versions(db: Database) -> None:
    registry = Registry(db)
    registry.ensure_collection("kb")
    assert registry.version("kb") == 0
    record = DocumentRecord(
        collection="kb",
        doc_id="d1",
        source="a.md",
        filename="a.md",
        source_type=SourceType.MARKDOWN,
        title="A",
        content_hash="h1",
        num_parents=2,
        num_chunks=5,
        ingested_at=NOW,
    )
    registry.upsert_document(record)
    registry.upsert_document(record.model_copy(update={"content_hash": "h2"}))
    assert registry.get_document("kb", "d1").content_hash == "h2"  # type: ignore[union-attr]
    assert registry.bump_version("kb") == 1
    [info] = registry.list_collections()
    assert (info.name, info.num_documents, info.num_chunks, info.version) == ("kb", 1, 5, 1)
    assert registry.delete_document("kb", "d1")
    assert registry.get_document("kb", "d1") is None


def test_parent_store_order_and_delete(db: Database) -> None:
    store = ParentStore(db)
    store.add_many([parent("d1-p0000", text="first"), parent("d1-p0001", text="second")])
    store.add_many([parent("d2-p0000", doc_id="d2")])
    assert [p.text for p in store.get_many("kb", ["d1-p0001", "missing", "d1-p0000"])] == [
        "second",
        "first",
    ]
    assert [p.parent_id for p in store.for_document("kb", "d1")] == ["d1-p0000", "d1-p0001"]
    assert store.delete_document("kb", "d1") == 2
    assert store.count("kb") == 1


# --------------------------------------------------------------------------- bm25


def test_tokenize() -> None:
    assert tokenize("The Café's X1 battery—is CHARGED!") == [
        "cafe",
        "s",
        "x1",
        "battery",
        "charged",
    ]


def test_bm25_ranks_and_filters(db: Database) -> None:
    Registry(db).ensure_collection("kb")
    index = BM25Index(db)
    chunks = [
        chunk("c1", "battery charge time is 75 minutes", doc_id="d1", filename="a.md"),
        chunk("c2", "propeller inspection before every flight", doc_id="d1", filename="a.md"),
        chunk("c3", "battery storage temperature guidance", doc_id="d2", filename="b.md"),
    ]
    index.add("kb", chunks, [c.text for c in chunks])
    hits = index.search("kb", "battery charge", k=5)
    assert [h.chunk_id for h in hits][:2] == ["c1", "c3"]
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.score > 0 for h in hits)
    assert [h.chunk_id for h in index.search("kb", "battery", k=5, filenames={"b.md"})] == ["c3"]
    assert index.search("kb", "the and of", k=5) == []  # only stopwords
    assert index.search("kb", "zeppelin", k=5) == []


def test_bm25_small_corpus_still_scores(db: Database) -> None:
    # Robertson IDF would be 0 here (term in 1 of 2 docs); Lucene IDF stays positive.
    index = BM25Index(db)
    index.add(
        "kb", [chunk("c1", "alpha beta"), chunk("c2", "gamma delta")], ["alpha beta", "gamma delta"]
    )
    assert [h.chunk_id for h in index.search("kb", "alpha", k=3)] == ["c1"]


def test_bm25_delete_and_cross_instance_invalidation(db: Database) -> None:
    registry = Registry(db)
    registry.ensure_collection("kb")
    writer, reader = BM25Index(db), BM25Index(db)
    writer.add("kb", [chunk("c1", "lidar module")], ["lidar module"])
    registry.bump_version("kb")
    assert [h.chunk_id for h in reader.search("kb", "lidar", k=3)] == ["c1"]

    writer.delete_document("kb", "d1")
    registry.bump_version("kb")  # what the pipeline does in the same transaction
    assert reader.search("kb", "lidar", k=3) == []
    assert writer.count("kb") == 0


# --------------------------------------------------------------------------- vectors


@pytest.fixture
def vectors(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedding_model="test-embed", embedding_dim=4)


def test_build_where() -> None:
    assert build_where() is None
    assert build_where(doc_ids={"b", "a"}) == {"doc_id": {"$in": ["a", "b"]}}
    assert build_where(doc_ids={"a"}, source_types={"pdf"}) == {
        "$and": [{"doc_id": {"$in": ["a"]}}, {"source_type": {"$in": ["pdf"]}}]
    }


def test_vector_store_roundtrip(vectors: VectorStore) -> None:
    chunks = [chunk("c1", "one", doc_id="d1"), chunk("c2", "two", doc_id="d2")]
    vectors.upsert("kb", chunks, [[1, 0, 0, 0], [0, 1, 0, 0]], ["h1", "h2"])
    assert vectors.count("kb") == 2

    hits = vectors.query("kb", [1, 0, 0, 0], k=2)
    assert [h.chunk.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].chunk.metadata == chunks[0].metadata
    assert hits[0].chunk.token_count == 1

    filtered = vectors.query("kb", [1, 0, 0, 0], k=2, where=build_where(doc_ids={"d2"}))
    assert [h.chunk.chunk_id for h in filtered] == ["c2"]
    assert set(vectors.get_chunks("kb", ["c2", "nope"])) == {"c2"}
    assert vectors.reusable_embeddings("kb", "d1") == {"h1": [1.0, 0.0, 0.0, 0.0]}

    vectors.delete_document("kb", "d1")
    assert vectors.count("kb") == 1


def test_missing_collection_is_empty(vectors: VectorStore) -> None:
    assert vectors.query("nothing", [1, 0, 0, 0], k=3) == []
    assert vectors.count("nothing") == 0
    vectors.delete_document("nothing", "d1")  # no-op, no error


def test_embedding_mismatch_is_refused(tmp_path: Path) -> None:
    first = VectorStore(tmp_path / "chroma", embedding_model="test-embed", embedding_dim=4)
    first.upsert("kb", [chunk("c1", "one")], [[1, 0, 0, 0]], ["h1"])
    other = VectorStore(tmp_path / "chroma", embedding_model="test-embed", embedding_dim=8)
    with pytest.raises(EmbeddingMismatchError, match="Re-index"):
        other.count("kb")
