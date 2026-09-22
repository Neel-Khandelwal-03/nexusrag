"""Evaluation metrics, implemented by hand (with an LLM judge where a metric needs one).

Retrieval, for answerable questions with known source sections. "Passages" are the parent
sections given to the answer model, in rank order:

* **hit@k**: 1 if a source section is among the first k passages.
* **MRR**: 1 / rank of the first source section (0 if absent), averaged.
* **context precision** (judge): average precision of the passages judged relevant to the
  reference answer, so relevant passages ranked first score higher (as in RAGAS).
* **context recall** (judge): share of the reference answer's statements that the
  passages support.

Generation:

* **faithfulness** (judge): share of the answer's factual claims the passages support.
  Answers without claims (plain refusals) are skipped.
* **answer relevance** (judge): 1-5 rating of how directly the answer addresses the
  question, scaled to 0-1. Answerable questions only, so a wrong refusal scores low.
* **correct refusals**: unanswerable questions the system declined.
* **false refusals**: answerable questions it declined.

Operations: latency p50/p95 and cost per query. Only the system's own model calls count;
the judge's calls are excluded.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from pydantic import BaseModel, Field

from nexusrag.generation.answer import format_passages
from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.llm.prompts import (
    EVAL_ANSWER_JUDGE_PROMPT,
    EVAL_CONTEXT_JUDGE_PROMPT,
    REFUSAL_MESSAGE,
    render,
)
from nexusrag.models import Answer, ContextPassage

# --------------------------------------------------------------------------- retrieval


def hit_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    """1.0 if any of the first ``k`` items is in ``gold``."""
    return 1.0 if any(item in gold for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Collection[str]) -> float:
    """1 / (1-based rank of the first item in ``gold``), or 0."""
    for rank, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def average_precision(relevant: Sequence[bool]) -> float:
    """Mean of precision@i over the positions i that are relevant (0 if none are)."""
    hits, total = 0, 0.0
    for i, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def share(flags: Sequence[bool]) -> float | None:
    """Fraction of true values; None for an empty list (nothing to judge)."""
    return sum(flags) / len(flags) if flags else None


def is_refusal(answer: Answer) -> bool:
    return answer.refused or answer.text.strip().startswith(REFUSAL_MESSAGE)


# --------------------------------------------------------------------------- statistics


def mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


# --------------------------------------------------------------------------- judges


class _PassageVerdict(BaseModel):
    id: int
    relevant: bool


class _StatementVerdict(BaseModel):
    statement: str
    supported: bool


class _ContextVerdict(BaseModel):
    passages: list[_PassageVerdict] = Field(default_factory=list)
    statements: list[_StatementVerdict] = Field(default_factory=list)


class _ClaimVerdict(BaseModel):
    claim: str
    supported: bool


class _AnswerVerdict(BaseModel):
    claims: list[_ClaimVerdict] = Field(default_factory=list)
    relevance: int = Field(ge=1, le=5)


@dataclass
class ContextScores:
    precision: float | None = None
    recall: float | None = None
    error: str | None = None


@dataclass
class AnswerScores:
    faithfulness: float | None = None
    relevance: float | None = None
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None


class Judge:
    """LLM-as-judge for the metrics that need one (one structured call each)."""

    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def context(
        self, question: str, reference: str, passages: Sequence[ContextPassage]
    ) -> ContextScores:
        if not passages:
            return ContextScores(precision=0.0, recall=0.0)
        try:
            verdict = await self.gemini.generate_structured(
                render(
                    EVAL_CONTEXT_JUDGE_PROMPT,
                    question=question,
                    reference=reference,
                    passages=format_passages(passages),
                ),
                _ContextVerdict,
                role="fast",
                stage="judge.context",
            )
        except GeminiError as exc:
            return ContextScores(error=type(exc).__name__)
        by_id = {v.id: v.relevant for v in verdict.passages}
        relevant = [by_id.get(p.index, False) for p in passages]
        return ContextScores(
            precision=average_precision(relevant),
            recall=share([s.supported for s in verdict.statements]),
        )

    async def answer(
        self, question: str, answer: str, passages: Sequence[ContextPassage]
    ) -> AnswerScores:
        try:
            verdict = await self.gemini.generate_structured(
                render(
                    EVAL_ANSWER_JUDGE_PROMPT,
                    question=question,
                    passages=format_passages(passages) if passages else "(no passages)",
                    answer=answer,
                ),
                _AnswerVerdict,
                role="fast",
                stage="judge.answer",
            )
        except GeminiError as exc:
            return AnswerScores(error=type(exc).__name__)
        return AnswerScores(
            faithfulness=share([c.supported for c in verdict.claims]),
            relevance=(verdict.relevance - 1) / 4,
            unsupported=[c.claim for c in verdict.claims if not c.supported],
        )


# --------------------------------------------------------------------------- results


class QuestionResult(BaseModel):
    """One question answered under one configuration, with its scores."""

    config: str
    item_id: str
    kind: str
    answerable: bool
    route: str | None = None
    refused: bool = False
    answer: str = ""
    #: Parent section IDs given to the answer model, in rank order.
    passages: list[str] = Field(default_factory=list)
    hit: float | None = None
    reciprocal_rank: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0
    #: Retries and model fallbacks while answering (not judging). Non-zero means the
    #: latency includes time spent waiting out rate limits. None: not recorded.
    retries: int | None = None
    fallbacks: int | None = None
    error: str | None = None
    judge_error: str | None = None


@dataclass
class ConfigSummary:
    config: str
    questions: int
    errors: int
    hit_at_k: float | None
    mrr: float | None
    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevance: float | None
    correct_refusals: float | None
    false_refusals: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    cost_per_query_usd: float | None
    #: Questions slowed by rate limiting (retries or model fallbacks).
    throttled: int = 0
    #: Latency over the questions that weren't throttled.
    clean_latency_p50_ms: float | None = None
    clean_latency_p95_ms: float | None = None


def summarize(config: str, results: Sequence[QuestionResult]) -> ConfigSummary:
    ok = [r for r in results if r.error is None]
    answerable = [r for r in ok if r.answerable]
    unanswerable = [r for r in ok if not r.answerable]
    latencies = [r.latency_ms for r in ok]
    throttled = [r for r in ok if (r.retries or 0) + (r.fallbacks or 0) > 0]
    clean = [r.latency_ms for r in ok if r.retries is not None and r not in throttled]
    return ConfigSummary(
        config=config,
        questions=len(results),
        errors=len(results) - len(ok),
        hit_at_k=mean([r.hit for r in answerable]),
        mrr=mean([r.reciprocal_rank for r in answerable]),
        context_precision=mean([r.context_precision for r in answerable]),
        context_recall=mean([r.context_recall for r in answerable]),
        faithfulness=mean([r.faithfulness for r in ok]),
        answer_relevance=mean([r.answer_relevance for r in answerable]),
        correct_refusals=share([r.refused for r in unanswerable]),
        false_refusals=share([r.refused for r in answerable]),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        cost_per_query_usd=mean([r.cost_usd for r in ok]),
        throttled=len(throttled),
        clean_latency_p50_ms=percentile(clean, 50),
        clean_latency_p95_ms=percentile(clean, 95),
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_table(summaries: Sequence[ConfigSummary], labels: dict[str, str], k: int) -> str:
    """The comparison table for the report and README."""
    header = (
        f"| Configuration | Hit@{k} | MRR | Context precision | Context recall | Faithfulness "
        "| Answer relevance | Correct refusals | False refusals | Latency p50 / p95 "
        "| Cost / query |"
    )
    lines = [header, "|" + "---|" * 11]
    for s in summaries:
        # Prefer latency without rate-limit waits when it was recorded.
        p50, p95 = s.clean_latency_p50_ms, s.clean_latency_p95_ms
        if p50 is None or p95 is None:
            p50, p95 = s.latency_p50_ms, s.latency_p95_ms
        latency = "—" if p50 is None or p95 is None else f"{p50 / 1000:.1f} s / {p95 / 1000:.1f} s"
        cost = "—" if s.cost_per_query_usd is None else f"${s.cost_per_query_usd:.4f}"
        lines.append(
            f"| {labels.get(s.config, s.config)} | {_pct(s.hit_at_k)} | {_num(s.mrr)} "
            f"| {_pct(s.context_precision)} | {_pct(s.context_recall)} | {_pct(s.faithfulness)} "
            f"| {_pct(s.answer_relevance)} | {_pct(s.correct_refusals)} "
            f"| {_pct(s.false_refusals)} | {latency} | {cost} |"
        )
    return "\n".join(lines)


def write_summary_csv(summaries: Sequence[ConfigSummary], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[f.name for f in fields(ConfigSummary)], lineterminator="\n"
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))


def write_results_csv(results: Sequence[QuestionResult], path: Path) -> None:
    columns = [name for name in QuestionResult.model_fields if name != "passages"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for r in results:
            row = r.model_dump(include=set(columns))
            row["unsupported_claims"] = " | ".join(r.unsupported_claims)
            writer.writerow(row)
