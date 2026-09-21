"""Registry of knowledge bases (collections) and the documents indexed in each.

The registry is the source of truth for incremental indexing: a document whose
``content_hash`` matches its registry row is already fully indexed and can be
skipped. Each collection also has a monotonically increasing ``version``, bumped on
every change, which caches (BM25 snapshots, the semantic cache) use for invalidation.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime

from pydantic import BaseModel

from nexusrag.models import SourceType, utcnow
from nexusrag.store.sqlite_db import Database

_COLLECTION_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,48}[a-z0-9])?$")


class InvalidCollectionName(ValueError):
    """Raised for knowledge base names that can't be used as storage identifiers."""


def validate_collection_name(name: str) -> str:
    """Normalise and validate a knowledge base name (lowercase, 1-50 chars of a-z 0-9 _ -)."""
    normalized = name.strip().lower().replace(" ", "-")
    if not _COLLECTION_RE.match(normalized):
        raise InvalidCollectionName(
            f"Invalid knowledge base name {name!r}: use 1-50 lowercase letters, digits, "
            "'-' or '_', starting and ending with a letter or digit."
        )
    return normalized


class DocumentRecord(BaseModel):
    """One indexed document as stored in the registry."""

    collection: str
    doc_id: str
    source: str
    filename: str
    source_type: SourceType
    title: str
    content_hash: str
    num_parents: int
    num_chunks: int
    ingested_at: datetime


class CollectionInfo(BaseModel):
    """Summary of one knowledge base."""

    name: str
    version: int
    num_documents: int
    num_chunks: int
    updated_at: datetime


def _record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord.model_validate(dict(row))


class Registry:
    """CRUD over the ``collections`` and ``documents`` tables."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------ collections

    def ensure_collection(self, name: str) -> str:
        """Create the collection row if missing; returns the validated name."""
        name = validate_collection_name(name)
        now = utcnow().isoformat()
        self._db.execute(
            "INSERT OR IGNORE INTO collections (name, version, created_at, updated_at) "
            "VALUES (?, 0, ?, ?)",
            (name, now, now),
        )
        return name

    def bump_version(self, name: str) -> int:
        """Mark the collection as changed; returns the new version."""
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE collections SET version = version + 1, updated_at = ? WHERE name = ?",
                (utcnow().isoformat(), name),
            )
            row = conn.execute("SELECT version FROM collections WHERE name = ?", (name,)).fetchone()
        return int(row["version"]) if row else 0

    def version(self, name: str) -> int:
        """Current version of a collection (0 if it doesn't exist)."""
        rows = self._db.query("SELECT version FROM collections WHERE name = ?", (name,))
        return int(rows[0]["version"]) if rows else 0

    def list_collections(self) -> list[CollectionInfo]:
        rows = self._db.query(
            """
            SELECT c.name, c.version, c.updated_at,
                   COUNT(d.doc_id) AS num_documents,
                   COALESCE(SUM(d.num_chunks), 0) AS num_chunks
            FROM collections c LEFT JOIN documents d ON d.collection = c.name
            GROUP BY c.name ORDER BY c.name
            """
        )
        return [CollectionInfo.model_validate(dict(row)) for row in rows]

    def delete_collection(self, name: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM documents WHERE collection = ?", (name,))
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))

    # ------------------------------------------------------------------ documents

    def get_document(self, collection: str, doc_id: str) -> DocumentRecord | None:
        rows = self._db.query(
            "SELECT * FROM documents WHERE collection = ? AND doc_id = ?", (collection, doc_id)
        )
        return _record(rows[0]) if rows else None

    def list_documents(self, collection: str) -> list[DocumentRecord]:
        rows = self._db.query(
            "SELECT * FROM documents WHERE collection = ? ORDER BY filename", (collection,)
        )
        return [_record(row) for row in rows]

    def upsert_document(self, record: DocumentRecord) -> None:
        self._db.execute(
            """
            INSERT INTO documents (collection, doc_id, source, filename, source_type, title,
                                   content_hash, num_parents, num_chunks, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (collection, doc_id) DO UPDATE SET
                source = excluded.source, filename = excluded.filename,
                source_type = excluded.source_type, title = excluded.title,
                content_hash = excluded.content_hash, num_parents = excluded.num_parents,
                num_chunks = excluded.num_chunks, ingested_at = excluded.ingested_at
            """,
            (
                record.collection,
                record.doc_id,
                record.source,
                record.filename,
                record.source_type.value,
                record.title,
                record.content_hash,
                record.num_parents,
                record.num_chunks,
                record.ingested_at.isoformat(),
            ),
        )

    def delete_document(self, collection: str, doc_id: str) -> bool:
        return (
            self._db.execute(
                "DELETE FROM documents WHERE collection = ? AND doc_id = ?", (collection, doc_id)
            )
            > 0
        )
