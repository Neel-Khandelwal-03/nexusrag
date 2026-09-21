"""Shared SQLite database for the registry, parent sections and BM25 token lists.

Keeping these three in one file means a document can be replaced in a single
transaction: old parents and BM25 rows are removed, new ones are inserted, and the
registry row (the "this version is fully indexed" marker) is updated atomically.
Only ChromaDB lives outside that transaction; see ``ingestion/pipeline.py``.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    name        TEXT PRIMARY KEY,
    version     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    collection   TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    source       TEXT NOT NULL,
    filename     TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    title        TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    num_parents  INTEGER NOT NULL,
    num_chunks   INTEGER NOT NULL,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (collection, doc_id)
);

CREATE TABLE IF NOT EXISTS parents (
    collection   TEXT NOT NULL,
    parent_id    TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    position     INTEGER NOT NULL,
    filename     TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    title        TEXT NOT NULL,
    section_path TEXT NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    PRIMARY KEY (collection, parent_id)
);
CREATE INDEX IF NOT EXISTS parents_by_doc ON parents (collection, doc_id, position);

CREATE TABLE IF NOT EXISTS bm25_chunks (
    collection   TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    filename     TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    tokens       TEXT NOT NULL,
    PRIMARY KEY (collection, chunk_id)
);
CREATE INDEX IF NOT EXISTS bm25_by_doc ON bm25_chunks (collection, doc_id);
"""


class Database:
    """Thread-safe wrapper around one SQLite connection.

    Chainlit runs blocking store calls in worker threads, so a single connection is
    shared behind a re-entrant lock. :meth:`transaction` nests: only the outermost
    block issues ``BEGIN``/``COMMIT``, so store methods can be composed into one
    atomic unit by the ingestion pipeline.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit, with explicit BEGIN in transaction().
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        with self._lock:
            if self.path != ":memory:":
                # WAL lets the app keep reading while an ingest (maybe another process) writes.
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run the block atomically (re-entrant: nested blocks join the outer transaction)."""
        with self._lock:
            outermost = self._depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self._conn
            except BaseException:
                self._depth -= 1
                if outermost:
                    self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outermost:
                self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a write statement; returns the number of affected rows."""
        with self.transaction() as conn:
            return conn.execute(sql, params).rowcount

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        with self.transaction() as conn:
            conn.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
