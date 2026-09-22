"""Semantic cache: reuse the answer to a near-identical earlier question.

Questions are embedded after routing, in the router's standalone form, so a follow-up
such as "and its warranty?" is compared as the full question it stands for. A cached
answer is reused only when all of these hold:

* **Same knowledge base, unchanged.** Each entry records the knowledge base version.
  Ingestion bumps the version and deletes the knowledge base's entries in the same
  transaction, so answers never outlive the documents they cite.
* **Same settings fingerprint:** route, answer style, document filters, retrieval
  switches, self-correction and models. A concise answer is never served for a detailed
  request, nor one retrieved without reranking for a request with it.
* **Cosine similarity at least the threshold** (``CACHE_SIMILARITY_THRESHOLD``).
* **Same key terms**: numbers and codes such as Q2, 2026, X1 or CH-400. On
  gemini-embedding-2, "revenue in Q2 2026" vs "revenue in Q1 2026" scored 0.947, higher
  than some true paraphrases, so embeddings alone can't be trusted with them.

Entries live in the shared SQLite database. Lookups scan one knowledge base's entries
for one fingerprint with numpy, which is instant at the sizes kept (``CACHE_MAX_ENTRIES``
per knowledge base, least recently used evicted first).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from nexusrag.log import get_logger
from nexusrag.models import Answer, utcnow
from nexusrag.store.sqlite_db import Database

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
#: Answer fields kept in the cache; everything else is per-request (usage, timings...).
_STORED_FIELDS = {"text", "route", "citations", "grounded"}


def key_terms(text: str) -> frozenset[str]:
    """Tokens that contain a digit (numbers, years, quarters, model codes), lowercased."""
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if any(c.isdigit() for c in t))


@dataclass(frozen=True)
class CacheKey:
    """Where an answer is looked up and stored."""

    collection: str
    version: int
    fingerprint: str
    question: str
    embedding: Sequence[float]


@dataclass(frozen=True)
class CachedAnswer:
    entry_id: int
    question: str
    answer: Answer
    similarity: float
    created_at: datetime
    hits: int


@dataclass(frozen=True)
class CacheLookup:
    """Result of a lookup, with enough detail to explain a miss in the UI."""

    hit: CachedAnswer | None
    #: Similarity of the hit, or of the closest entry on a miss (0 when there are none).
    best_similarity: float = 0.0
    #: Entries compared (same knowledge base version and fingerprint).
    candidates: int = 0
    #: On a miss: the closest entry cleared the threshold but asked about different
    #: numbers or codes, so the key-term guard refused it.
    blocked_by_key_terms: bool = False


@dataclass
class CacheCounters:
    lookups: int = 0
    hits: int = 0


@dataclass
class _Row:
    entry_id: int
    question: str
    vector: np.ndarray
    answer_json: str
    created_at: str
    hits: int
    similarity: float = 0.0


class SemanticCache:
    """Stores answers in the ``semantic_cache`` table and finds near-duplicate questions."""

    def __init__(self, db: Database, *, max_entries: int = 1000) -> None:
        self._db = db
        self.max_entries = max_entries
        self._counters: dict[str, CacheCounters] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lookup

    def lookup(self, key: CacheKey, threshold: float) -> CacheLookup:
        """Best matching answer for ``key``, if one clears the threshold and key terms."""
        rows = self._candidates(key)
        query = np.asarray(key.embedding, dtype=np.float32)
        terms = key_terms(key.question)
        scored = [r for r in rows if r.vector.shape == query.shape]
        for row in scored:
            row.similarity = float(np.dot(row.vector, query))
        scored.sort(key=lambda r: -r.similarity)
        hit_row = next(
            (r for r in scored if r.similarity >= threshold and key_terms(r.question) == terms),
            None,
        )
        hit = self._record_hit(hit_row) if hit_row is not None else None
        best = scored[0].similarity if scored else 0.0
        # Only worth mentioning when the guard changed the outcome.
        other_terms = (
            hit is None
            and best >= threshold
            and key_terms(scored[0].question) != terms  # scored is non-empty here
        )
        with self._lock:
            counters = self._counters.setdefault(key.collection, CacheCounters())
            counters.lookups += 1
            counters.hits += hit is not None
        log.info(
            "cache.lookup",
            collection=key.collection,
            hit=hit is not None,
            candidates=len(rows),
            similarity=round(hit.similarity if hit else best, 4),
            blocked_by_key_terms=other_terms,
        )
        return CacheLookup(
            hit=hit,
            best_similarity=hit.similarity if hit else best,
            candidates=len(rows),
            blocked_by_key_terms=other_terms,
        )

    def _candidates(self, key: CacheKey) -> list[_Row]:
        rows = self._db.query(
            "SELECT id, question, embedding, answer_json, created_at, hits FROM semantic_cache "
            "WHERE collection = ? AND kb_version = ? AND fingerprint = ?",
            (key.collection, key.version, key.fingerprint),
        )
        return [
            _Row(
                entry_id=int(r["id"]),
                question=r["question"],
                vector=np.frombuffer(r["embedding"], dtype=np.float32),
                answer_json=r["answer_json"],
                created_at=r["created_at"],
                hits=int(r["hits"]),
            )
            for r in rows
        ]

    def _record_hit(self, row: _Row) -> CachedAnswer | None:
        try:
            answer = Answer.model_validate_json(row.answer_json)
        except ValueError:
            # Written by an incompatible older version: drop it rather than fail the turn.
            self._db.execute("DELETE FROM semantic_cache WHERE id = ?", (row.entry_id,))
            return None
        self._db.execute(
            "UPDATE semantic_cache SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
            (utcnow().isoformat(), row.entry_id),
        )
        return CachedAnswer(
            entry_id=row.entry_id,
            question=row.question,
            answer=answer.model_copy(update={"cached": True, "cache_similarity": row.similarity}),
            similarity=row.similarity,
            created_at=datetime.fromisoformat(row.created_at),
            hits=row.hits + 1,
        )

    # ------------------------------------------------------------------ writes

    def put(self, key: CacheKey, answer: Answer) -> int:
        """Store ``answer`` for ``key``; evicts stale versions and least recently used."""
        vector = np.asarray(key.embedding, dtype=np.float32)
        with self._db.transaction() as conn:
            # Entries from older versions can never match again.
            conn.execute(
                "DELETE FROM semantic_cache WHERE collection = ? AND kb_version <> ?",
                (key.collection, key.version),
            )
            cursor = conn.execute(
                "INSERT INTO semantic_cache (collection, kb_version, fingerprint, question, "
                "embedding, answer_json, created_at, hits) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    key.collection,
                    key.version,
                    key.fingerprint,
                    key.question,
                    vector.tobytes(),
                    answer.model_dump_json(include=_STORED_FIELDS),
                    utcnow().isoformat(),
                ),
            )
            conn.execute(
                "DELETE FROM semantic_cache WHERE collection = ? AND id NOT IN ("
                "  SELECT id FROM semantic_cache WHERE collection = ?"
                "  ORDER BY COALESCE(last_hit_at, created_at) DESC, id DESC LIMIT ?)",
                (key.collection, key.collection, self.max_entries),
            )
        entry_id = int(cursor.lastrowid or 0)
        log.info("cache.stored", collection=key.collection, entry_id=entry_id)
        return entry_id

    def invalidate(self, collection: str) -> int:
        """Delete every entry for ``collection`` (called when its documents change)."""
        removed = self._db.execute("DELETE FROM semantic_cache WHERE collection = ?", (collection,))
        if removed:
            log.info("cache.invalidated", collection=collection, entries=removed)
        return removed

    # ------------------------------------------------------------------ stats

    def count(self, collection: str) -> int:
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM semantic_cache WHERE collection = ?", (collection,)
        )
        return int(rows[0]["n"])

    def counters(self, collection: str) -> CacheCounters:
        """Lookups and hits for ``collection`` since the process started."""
        with self._lock:
            c = self._counters.get(collection, CacheCounters())
            return CacheCounters(lookups=c.lookups, hits=c.hits)
