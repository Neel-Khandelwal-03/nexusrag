from __future__ import annotations

import asyncio

import pytest

from nexusrag.config import DEFAULT_PRICES
from nexusrag.llm.usage import (
    UsageRecord,
    estimate_cost,
    process_usage,
    record_usage,
    track_usage,
)


def test_estimate_cost() -> None:
    cost = estimate_cost(
        DEFAULT_PRICES, "gemini-3.8-flash", input_tokens=2_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(2 * 0.75 + 3.75)


def test_unknown_model_costs_zero() -> None:
    assert estimate_cost(DEFAULT_PRICES, "mystery", input_tokens=10, output_tokens=10) == 0.0


async def test_tracker_collects_from_concurrent_tasks() -> None:
    async def call(i: int) -> None:
        await asyncio.sleep(0)
        record_usage(UsageRecord(stage=f"s{i}", model="m", prompt_tokens=i, cost_usd=0.5))

    before = process_usage().calls
    with track_usage() as usage:
        await asyncio.gather(*(call(i) for i in range(1, 5)))
    totals = usage.totals()
    assert totals.calls == 4
    assert totals.prompt_tokens == 10
    assert totals.cost_usd == pytest.approx(2.0)
    assert len(usage.records) == 4
    assert process_usage().calls == before + 4


def test_records_outside_block_only_hit_process_totals() -> None:
    with track_usage() as usage:
        pass
    record_usage(UsageRecord(stage="late", model="m"))
    assert usage.totals().calls == 0
