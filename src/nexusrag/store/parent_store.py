"""Storage for parent sections, the full-context units sent to the LLM.

Child chunks in Chroma/BM25 carry a ``parent_id``. After retrieval, children are
swapped for their parents via :meth:`ParentStore.get_many`. Whole-document
operations (summaries, comparisons) read parents in document order via
:meth:`ParentStore.for_document`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from nexusrag.models import ParentSection
from nexusrag.store.sqlite_db import Database

_COLUMNS = (
    "parent_id, doc_id, collection, filename, source_type, title, section_path, "
    "page_start, page_end, text, token_count"
)


def _parent(row: sqlite3.Row) -> ParentSection:
    data = dict(row)
    data.pop("position", None)
    return ParentSection.model_validate(data)


class ParentStore:
    """CRUD over the ``parents`` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add_many(self, parents: Sequence[ParentSection]) -> None:
        """Insert or replace parents; list order is stored as document position."""
        self._db.executemany(
            """
            INSERT OR REPLACE INTO parents (collection, parent_id, doc_id, position, filename,
                source_type, title, section_path, page_start, page_end, text, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    p.collection,
                    p.parent_id,
                    p.doc_id,
                    position,
                    p.filename,
                    p.source_type.value,
                    p.title,
                    p.section_path,
                    p.page_start,
                    p.page_end,
                    p.text,
                    p.token_count,
                )
                for position, p in enumerate(parents)
            ],
        )

    def get_many(self, collection: str, parent_ids: Sequence[str]) -> list[ParentSection]:
        """Fetch parents by ID, preserving the order of ``parent_ids`` (missing IDs are skipped)."""
        if not parent_ids:
            return []
        placeholders = ",".join("?" * len(parent_ids))
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM parents "
            f"WHERE collection = ? AND parent_id IN ({placeholders})",
            (collection, *parent_ids),
        )
        by_id = {row["parent_id"]: _parent(row) for row in rows}
        return [by_id[pid] for pid in parent_ids if pid in by_id]

    def for_document(self, collection: str, doc_id: str) -> list[ParentSection]:
        """All parents of a document in reading order."""
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM parents WHERE collection = ? AND doc_id = ? ORDER BY position",
            (collection, doc_id),
        )
        return [_parent(row) for row in rows]

    def delete_document(self, collection: str, doc_id: str) -> int:
        return self._db.execute(
            "DELETE FROM parents WHERE collection = ? AND doc_id = ?", (collection, doc_id)
        )

    def delete_collection(self, collection: str) -> None:
        self._db.execute("DELETE FROM parents WHERE collection = ?", (collection,))

    def count(self, collection: str) -> int:
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM parents WHERE collection = ?", (collection,)
        )
        return int(rows[0]["n"])
