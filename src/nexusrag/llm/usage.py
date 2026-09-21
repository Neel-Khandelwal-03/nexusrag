"""Token and cost accounting for Gemini calls.

Every call made through :class:`~nexusrag.llm.gemini_client.GeminiClient` produces a
:class:`UsageRecord`. Records are added to:

* the tracker for the current request, if a :func:`track_usage` block is active. It is
  stored in a ``ContextVar``, so concurrent asyncio tasks spawned inside the block
  report into the same tracker;
* a process-wide tracker (totals only), surfaced later by the ``/stats`` command.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from nexusrag.config import ModelPrice
from nexusrag.models import UsageStats


@dataclass(frozen=True)
class UsageRecord:
    """Usage of a single Gemini API call."""

    stage: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class UsageTracker:
    """Accumulates usage. Keeps individual records only when ``keep_records`` is True."""

    def __init__(self, keep_records: bool = True) -> None:
        self._keep_records = keep_records
        self._lock = threading.Lock()
        self._totals = UsageStats()
        self.records: list[UsageRecord] = []

    def add(self, record: UsageRecord) -> None:
        with self._lock:
            t = self._totals
            t.calls += 1
            t.prompt_tokens += record.prompt_tokens
            t.output_tokens += record.output_tokens
            t.thoughts_tokens += record.thoughts_tokens
            t.embedding_tokens += record.embedding_tokens
            t.cost_usd += record.cost_usd
            if self._keep_records:
                self.records.append(record)

    def totals(self) -> UsageStats:
        """Snapshot of the aggregated usage."""
        with self._lock:
            return self._totals.model_copy()


_current: ContextVar[UsageTracker | None] = ContextVar("nexusrag_usage", default=None)
_process_totals = UsageTracker(keep_records=False)


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    """Collect usage of every Gemini call made inside the block."""
    tracker = UsageTracker()
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def record_usage(record: UsageRecord) -> None:
    """Add a record to the active request tracker (if any) and the process totals."""
    tracker = _current.get()
    if tracker is not None:
        tracker.add(record)
    _process_totals.add(record)


def process_usage() -> UsageStats:
    """Usage totals since the process started."""
    return _process_totals.totals()


def estimate_cost(
    prices: Mapping[str, ModelPrice], model: str, *, input_tokens: int, output_tokens: int
) -> float:
    """Approximate USD cost of a call. Unknown models cost 0 (logged, not fatal)."""
    price = prices.get(model)
    if price is None:
        return 0.0
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000
