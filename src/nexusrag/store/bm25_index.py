"""Keyword search with BM25 (``rank-bm25``), persisted in SQLite and updated incrementally.

``rank_bm25`` needs the whole corpus up front to compute IDF, so an in-place update
isn't possible. Instead:

* each chunk's *token list* is persisted as a row in ``bm25_chunks``, so adding or
  removing a document only touches that document's rows;
* the in-memory BM25 model is rebuilt lazily, on the first search after a change.
  Staleness is detected by comparing the collection's registry version, so changes
  made by another process (e.g. the ingest CLI) are picked up too.

Rebuilding is O(corpus) but cheap: tokenised rows load straight from SQLite.
"""

from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from nexusrag.models import Chunk
from nexusrag.store.sqlite_db import Database

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# A compact English stopword list. BM25's IDF already down-weights frequent terms;
# dropping these just keeps the index smaller and the scores less noisy.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours
    yourself yourselves
    """.split()  # noqa: SIM905 (a word list reads better than 150 quoted lines)
)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split on non-word characters and drop stopwords."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [tok for tok in _TOKEN_RE.findall(folded) if tok not in STOPWORDS]


class _LuceneBM25(BM25Okapi):
    """BM25Okapi with Lucene's non-negative IDF: ``log(1 + (N - n + 0.5) / (n + 0.5))``.

    The classic Robertson IDF used by ``BM25Okapi`` is zero or negative for any term
    that appears in half the corpus or more. On small knowledge bases, where a
    product name can easily appear in half the chunks, that silently zeroes out
    perfectly good matches.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


@dataclass(frozen=True)
class BM25Hit:
    """A keyword search result."""

    chunk_id: str
    score: float
    rank: int  # 1-based


@dataclass
class _Snapshot:
    version: int
    chunk_ids: list[str]
    doc_ids: list[str]
    filenames: list[str]
    source_types: list[str]
    model: _LuceneBM25 | None


class BM25Index:
    """Per-collection BM25 index backed by the ``bm25_chunks`` table."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._snapshots: dict[str, _Snapshot] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ writes

    def add(self, collection: str, chunks: Sequence[Chunk], texts: Sequence[str]) -> None:
        """Index chunks. ``texts`` is what gets tokenised (chunk text plus context)."""
        if len(chunks) != len(texts):
            raise ValueError("chunks and texts must have the same length")
        self._db.executemany(
            "INSERT OR REPLACE INTO bm25_chunks "
            "(collection, chunk_id, doc_id, filename, source_type, tokens) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    collection,
                    chunk.chunk_id,
                    chunk.metadata.doc_id,
                    chunk.metadata.filename,
                    chunk.metadata.source_type.value,
                    json.dumps(tokenize(text)),
                )
                for chunk, text in zip(chunks, texts, strict=True)
            ],
        )
        self._invalidate(collection)

    def delete_document(self, collection: str, doc_id: str) -> int:
        removed = self._db.execute(
            "DELETE FROM bm25_chunks WHERE collection = ? AND doc_id = ?", (collection, doc_id)
        )
        self._invalidate(collection)
        return removed

    def delete_collection(self, collection: str) -> None:
        self._db.execute("DELETE FROM bm25_chunks WHERE collection = ?", (collection,))
        self._invalidate(collection)

    def count(self, collection: str) -> int:
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM bm25_chunks WHERE collection = ?", (collection,)
        )
        return int(rows[0]["n"])

    # ------------------------------------------------------------------ search

    def search(
        self,
        collection: str,
        query: str,
        k: int,
        *,
        doc_ids: Collection[str] | None = None,
        filenames: Collection[str] | None = None,
        source_types: Collection[str] | None = None,
    ) -> list[BM25Hit]:
        """Top-``k`` chunks by BM25 score, optionally restricted by metadata filters."""
        terms = tokenize(query)
        snapshot = self._snapshot(collection)
        if not terms or snapshot.model is None:
            return []
        scores = snapshot.model.get_scores(terms)
        candidates = [
            i
            for i, score in enumerate(scores)
            if score > 0
            and (doc_ids is None or snapshot.doc_ids[i] in doc_ids)
            and (filenames is None or snapshot.filenames[i] in filenames)
            and (source_types is None or snapshot.source_types[i] in source_types)
        ]
        candidates.sort(key=lambda i: scores[i], reverse=True)
        return [
            BM25Hit(chunk_id=snapshot.chunk_ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(candidates[:k], start=1)
        ]

    # ------------------------------------------------------------------ snapshot cache

    def _current_version(self, collection: str) -> int:
        rows = self._db.query("SELECT version FROM collections WHERE name = ?", (collection,))
        return int(rows[0]["version"]) if rows else 0

    def _invalidate(self, collection: str) -> None:
        with self._lock:
            self._snapshots.pop(collection, None)

    def _snapshot(self, collection: str) -> _Snapshot:
        version = self._current_version(collection)
        with self._lock:
            cached = self._snapshots.get(collection)
            if cached is not None and cached.version == version:
                return cached
        rows = self._db.query(
            "SELECT chunk_id, doc_id, filename, source_type, tokens FROM bm25_chunks "
            "WHERE collection = ? ORDER BY chunk_id",
            (collection,),
        )
        corpus = [json.loads(row["tokens"]) for row in rows]
        # BM25Okapi divides by the average document length, so an all-empty corpus can't
        # be scored; treat it as "nothing to search".
        has_terms = any(corpus)
        snapshot = _Snapshot(
            version=version,
            chunk_ids=[row["chunk_id"] for row in rows],
            doc_ids=[row["doc_id"] for row in rows],
            filenames=[row["filename"] for row in rows],
            source_types=[row["source_type"] for row in rows],
            model=_LuceneBM25(corpus) if has_terms else None,
        )
        with self._lock:
            self._snapshots[collection] = snapshot
        return snapshot
