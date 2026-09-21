"""ChromaDB wrapper for child-chunk vectors.

One Chroma collection per knowledge base (``nx_<name>``), using cosine distance. We
always supply our own Gemini embeddings (``embedding_function=None``), so Chroma
never downloads its default ONNX model. Each collection records the embedding model
and dimension it was built with, and opening it with a different configuration is
refused: mixing vector spaces would silently return garbage.
"""

from __future__ import annotations

from collections.abc import Collection as AbstractCollection
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from nexusrag.models import Chunk, ChunkMetadata

COLLECTION_PREFIX = "nx_"


class EmbeddingMismatchError(RuntimeError):
    """The collection was indexed with a different embedding model or dimension."""


@dataclass(frozen=True)
class DenseHit:
    """A vector search result."""

    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]
    rank: int  # 1-based


def build_where(
    *,
    doc_ids: AbstractCollection[str] | None = None,
    filenames: AbstractCollection[str] | None = None,
    source_types: AbstractCollection[str] | None = None,
) -> dict[str, Any] | None:
    """Translate metadata filters into a Chroma ``where`` clause (None = no filter)."""
    clauses: list[dict[str, Any]] = []
    for field, values in (
        ("doc_id", doc_ids),
        ("filename", filenames),
        ("source_type", source_types),
    ):
        if values is not None:
            clauses.append({field: {"$in": sorted(values)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _chunk_from_record(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        token_count=int(metadata.get("token_count", 0)),
        metadata=ChunkMetadata.from_chroma(metadata),
    )


class VectorStore:
    """Dense retrieval over child chunks."""

    def __init__(
        self,
        path: Path | str,
        *,
        embedding_model: str,
        embedding_dim: int,
        client: ClientAPI | None = None,
    ) -> None:
        if client is None:
            Path(path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
        self._client = client
        self._model = embedding_model
        self._dim = embedding_dim
        self._cache: dict[str, Collection] = {}

    @staticmethod
    def collection_name(knowledge_base: str) -> str:
        return f"{COLLECTION_PREFIX}{knowledge_base}"

    def _existing_names(self) -> set[str]:
        return {c.name if hasattr(c, "name") else str(c) for c in self._client.list_collections()}

    def _collection(self, knowledge_base: str, *, create: bool) -> Collection | None:
        if knowledge_base in self._cache:
            return self._cache[knowledge_base]
        name = self.collection_name(knowledge_base)
        if not create and name not in self._existing_names():
            return None
        collection = self._client.get_or_create_collection(
            name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self._model, "embedding_dim": self._dim},
        )
        meta = collection.metadata or {}
        if (
            meta.get("embedding_model", self._model) != self._model
            or int(meta.get("embedding_dim", self._dim)) != self._dim
        ):
            raise EmbeddingMismatchError(
                f"Knowledge base {knowledge_base!r} was indexed with {meta.get('embedding_model')} "
                f"({meta.get('embedding_dim')}-d) but the current config uses {self._model} "
                f"({self._dim}-d). Re-index it or restore the original embedding settings."
            )
        self._cache[knowledge_base] = collection
        return collection

    # ------------------------------------------------------------------ writes

    def upsert(
        self,
        knowledge_base: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        embed_hashes: Sequence[str],
    ) -> None:
        """Store chunks with their vectors. ``embed_hashes`` identify the exact embedding input."""
        if not (len(chunks) == len(embeddings) == len(embed_hashes)):
            raise ValueError("chunks, embeddings and embed_hashes must have the same length")
        collection = self._collection(knowledge_base, create=True)
        assert collection is not None
        batch = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch):
            part = slice(start, start + batch)
            vectors: list[Sequence[float]] = [list(v) for v in embeddings[part]]
            collection.upsert(
                ids=[c.chunk_id for c in chunks[part]],
                documents=[c.text for c in chunks[part]],
                embeddings=vectors,
                metadatas=[
                    {**c.metadata.to_chroma(), "token_count": c.token_count, "embed_hash": h}
                    for c, h in zip(chunks[part], embed_hashes[part], strict=True)
                ],
            )

    def delete_document(self, knowledge_base: str, doc_id: str) -> None:
        collection = self._collection(knowledge_base, create=False)
        if collection is not None:
            collection.delete(where={"doc_id": doc_id})

    def delete_collection(self, knowledge_base: str) -> None:
        name = self.collection_name(knowledge_base)
        self._cache.pop(knowledge_base, None)
        if name in self._existing_names():
            self._client.delete_collection(name)

    # ------------------------------------------------------------------ reads

    def reusable_embeddings(self, knowledge_base: str, doc_id: str) -> dict[str, list[float]]:
        """Existing vectors of a document keyed by embed hash, to skip re-embedding on update."""
        collection = self._collection(knowledge_base, create=False)
        if collection is None:
            return {}
        result = collection.get(where={"doc_id": doc_id}, include=["embeddings", "metadatas"])
        embeddings = result.get("embeddings")
        metadatas = result.get("metadatas") or []
        if embeddings is None:
            return {}
        return {
            str(meta["embed_hash"]): [float(x) for x in vector]
            for meta, vector in zip(metadatas, embeddings, strict=True)
            if meta and "embed_hash" in meta
        }

    def query(
        self,
        knowledge_base: str,
        embedding: Sequence[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[DenseHit]:
        """Top-``k`` chunks by cosine similarity."""
        collection = self._collection(knowledge_base, create=False)
        if collection is None or collection.count() == 0:
            return []
        queries: list[Sequence[float]] = [list(embedding)]
        result = collection.query(
            query_embeddings=queries,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            DenseHit(
                chunk=_chunk_from_record(cid, doc or "", dict(meta or {})),
                # Chroma's cosine *distance* is 1 - cosine similarity.
                score=1.0 - float(dist),
                rank=rank,
            )
            for rank, (cid, doc, meta, dist) in enumerate(
                zip(ids, documents, metadatas, distances, strict=True), start=1
            )
        ]

    def get_chunks(self, knowledge_base: str, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        """Fetch chunks by ID (used to hydrate BM25 hits)."""
        collection = self._collection(knowledge_base, create=False)
        if collection is None or not chunk_ids:
            return {}
        result = collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return {
            cid: _chunk_from_record(cid, doc or "", dict(meta or {}))
            for cid, doc, meta in zip(result["ids"], documents, metadatas, strict=True)
        }

    def count(self, knowledge_base: str) -> int:
        collection = self._collection(knowledge_base, create=False)
        return 0 if collection is None else collection.count()
