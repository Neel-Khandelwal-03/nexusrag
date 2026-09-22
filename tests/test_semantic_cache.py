"""The semantic cache store: matching rules, invalidation, eviction and counters."""

from __future__ import annotations

import numpy as np
import pytest

from nexusrag.models import Answer, Citation, Route, SourceType
from nexusrag.store.semantic_cache import CacheKey, SemanticCache, key_terms
from nexusrag.store.sqlite_db import Database

DIM = 16


def unit(*weights: float) -> list[float]:
    """A unit vector from the first few coordinates (the rest are zero)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(weights)] = weights
    return [float(x) for x in v / np.linalg.norm(v)]


def at_similarity(cos: float) -> list[float]:
    """A unit vector whose cosine with ``unit(1)`` is exactly ``cos``."""
    return unit(cos, float(np.sqrt(1 - cos**2)))


def key(
    question: str = "How long does the Aurora X1 battery last?",
    embedding: list[float] | None = None,
    *,
    collection: str = "kb",
    version: int = 1,
    fingerprint: str = "fp",
) -> CacheKey:
    return CacheKey(
        collection=collection,
        version=version,
        fingerprint=fingerprint,
        question=question,
        embedding=embedding or unit(1),
    )


def answer(text: str = "It lasts 46 minutes [1].") -> Answer:
    citation = Citation(
        index=1,
        parent_id="p1",
        doc_id="d1",
        filename="aurora.pdf",
        source_type=SourceType.PDF,
        title="Aurora",
        page_start=2,
        text="Maximum flight time is 46 minutes.",
    )
    return Answer(text=text, route=Route.DOC_QA, citations=[citation], grounded=True)


@pytest.fixture
def cache() -> SemanticCache:
    return SemanticCache(Database(":memory:"), max_entries=3)


def test_key_terms_are_numbers_and_codes() -> None:
    assert key_terms("Revenue in Q2 2026 for the X1 with the CH-400?") == {
        "q2",
        "2026",
        "x1",
        "ch-400",
    }
    assert key_terms("how long does the aurora x1 battery last") == {"x1"}
    assert key_terms("What is the return policy?") == frozenset()


def test_paraphrase_above_threshold_hits(cache: SemanticCache) -> None:
    cache.put(key(), answer())
    lookup = cache.lookup(
        key("What is the battery life of the Aurora X1?", at_similarity(0.97)), 0.96
    )
    assert lookup.hit is not None
    assert lookup.hit.similarity == pytest.approx(0.97, abs=1e-4)
    assert lookup.hit.question == "How long does the Aurora X1 battery last?"
    restored = lookup.hit.answer
    assert restored.cached
    assert restored.cache_similarity == pytest.approx(0.97, abs=1e-4)
    assert restored.text == "It lasts 46 minutes [1]."
    assert restored.citations[0].page_start == 2  # the source panels survive the round trip
    assert restored.usage.calls == 0  # per-request data isn't stored


def test_below_threshold_misses_but_reports_closest(cache: SemanticCache) -> None:
    cache.put(key(), answer())
    lookup = cache.lookup(
        key("How long does the Aurora X1 take to charge?", at_similarity(0.94)), 0.96
    )
    assert lookup.hit is None
    assert lookup.candidates == 1
    assert lookup.best_similarity == pytest.approx(0.94, abs=1e-4)
    assert not lookup.blocked_by_key_terms


def test_different_numbers_never_hit_even_when_embeddings_agree(cache: SemanticCache) -> None:
    cache.put(key("What was revenue in Q2 2026?"), answer("Revenue was 48M [1]."))
    lookup = cache.lookup(key("What was revenue in Q1 2026?", at_similarity(0.99)), 0.96)
    assert lookup.hit is None
    assert lookup.blocked_by_key_terms
    assert lookup.best_similarity == pytest.approx(0.99, abs=1e-4)
    # Below the threshold the guard changes nothing, so it isn't blamed.
    far = cache.lookup(key("What was revenue in Q1 2026?", at_similarity(0.90)), 0.96)
    assert not far.blocked_by_key_terms


def test_matching_is_scoped_to_collection_version_and_fingerprint(cache: SemanticCache) -> None:
    cache.put(key(), answer())
    assert cache.lookup(key(), 0.96).hit is not None
    assert cache.lookup(key(collection="other"), 0.96).hit is None
    assert cache.lookup(key(version=2), 0.96).hit is None  # documents changed since
    assert cache.lookup(key(fingerprint="concise"), 0.96).hit is None  # other settings


def test_best_compatible_entry_wins(cache: SemanticCache) -> None:
    cache.put(key("Battery life of the Aurora X1?", at_similarity(0.97)), answer("A [1]."))
    cache.put(key("How long does the Aurora X1 battery last?"), answer("B [1]."))
    hit = cache.lookup(key(), 0.96).hit
    assert hit is not None
    assert hit.answer.text == "B [1]."
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)


def test_hits_are_counted_and_recorded(cache: SemanticCache) -> None:
    cache.put(key(), answer())
    cache.lookup(key(), 0.96)
    second = cache.lookup(key(), 0.96)
    cache.lookup(key("Something else entirely?", unit(0, 1)), 0.96)
    assert second.hit is not None
    assert second.hit.hits == 2
    counters = cache.counters("kb")
    assert (counters.lookups, counters.hits) == (3, 2)
    assert cache.counters("other").lookups == 0


def test_invalidate_and_stale_versions(cache: SemanticCache) -> None:
    cache.put(key(), answer())
    cache.put(key(collection="other"), answer())
    assert cache.invalidate("kb") == 1
    assert cache.count("kb") == 0
    assert cache.count("other") == 1
    cache.put(key(version=1), answer())
    cache.put(key("A newer question about the X1?", version=2), answer())
    assert cache.count("kb") == 1  # storing at version 2 drops version 1 entries


def test_least_recently_used_entries_are_evicted(cache: SemanticCache) -> None:
    ids = [cache.put(key(f"Question number {i}?", unit(1, i)), answer()) for i in range(3)]
    cache.lookup(key("Question number 0?", unit(1, 0)), 0.96)  # touch the oldest
    cache.put(key("Question number 3?", unit(1, 3)), answer())
    rows = cache._db.query("SELECT id FROM semantic_cache ORDER BY id")
    assert [r["id"] for r in rows] == [ids[0], ids[2], ids[2] + 1]


def test_vectors_of_another_dimension_are_ignored(cache: SemanticCache) -> None:
    cache.put(key(), answer())
    lookup = cache.lookup(key(embedding=[1.0, 0.0, 0.0]), 0.96)
    assert lookup.hit is None
    assert lookup.best_similarity == 0.0


def test_unreadable_entries_are_dropped(cache: SemanticCache) -> None:
    cache.put(key(), answer())
    cache._db.execute("UPDATE semantic_cache SET answer_json = '{\"nope\": 1}'")
    assert cache.lookup(key(), 0.96).hit is None
    assert cache.count("kb") == 0
