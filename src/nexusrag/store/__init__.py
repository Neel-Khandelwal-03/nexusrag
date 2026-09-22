"""Persistence: ChromaDB for child vectors; SQLite for the registry, parents, BM25 and cache."""

from __future__ import annotations

from dataclasses import dataclass

from nexusrag.config import Settings
from nexusrag.store.bm25_index import BM25Index
from nexusrag.store.parent_store import ParentStore
from nexusrag.store.registry import Registry
from nexusrag.store.semantic_cache import SemanticCache
from nexusrag.store.sqlite_db import Database
from nexusrag.store.vector_store import VectorStore


@dataclass
class Stores:
    """Every store the pipeline needs, opened against the same storage directory."""

    db: Database
    vectors: VectorStore
    bm25: BM25Index
    parents: ParentStore
    registry: Registry
    cache: SemanticCache

    @classmethod
    def open(cls, settings: Settings) -> Stores:
        db = Database(settings.sqlite_path)
        return cls(
            db=db,
            vectors=VectorStore(
                settings.chroma_dir,
                embedding_model=settings.embedding_model,
                embedding_dim=settings.embedding_dim,
            ),
            bm25=BM25Index(db),
            parents=ParentStore(db),
            registry=Registry(db),
            cache=SemanticCache(db, max_entries=settings.cache_max_entries),
        )

    def close(self) -> None:
        self.db.close()


__all__ = [
    "BM25Index",
    "Database",
    "ParentStore",
    "Registry",
    "SemanticCache",
    "Stores",
    "VectorStore",
]
