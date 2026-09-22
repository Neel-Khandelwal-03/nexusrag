"""Compare pipeline configurations on the evaluation dataset.

    python -m eval.run_eval                                  # the four configurations
    python -m eval.run_eval --configs dense,full --limit 10  # a quick subset
    python -m eval.run_eval --resume eval/reports/<run>      # continue an interrupted run

Every question goes through the real agent (router included) with the semantic cache
off. Results are appended to ``<run>/results.jsonl`` as they arrive, so a run interrupted
by rate limits can be resumed without paying for finished questions again: a per-minute
limit waits and retries, while a used-up daily quota (the free tier allows 20 requests a
day per Flash model) stops the run so it can be resumed after the reset. At the end the
run folder gets ``results.csv`` (one row per question and configuration), ``summary.csv``
and ``summary.md``, and ``summary.md`` is copied to ``eval/reports/latest.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eval.dataset import DEFAULT_PATH, EvalItem, load_dataset, resolve_gold
from eval.metrics import (
    ConfigSummary,
    Judge,
    QuestionResult,
    hit_at_k,
    is_refusal,
    markdown_table,
    mean,
    reciprocal_rank,
    summarize,
    write_results_csv,
    write_summary_csv,
)
from nexusrag.config import Settings, get_settings
from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.log import configure_logging
from nexusrag.retrieval.retriever import RetrievalOptions
from nexusrag.service import RAGService

REPORTS_DIR = Path(__file__).with_name("reports")
QUOTA_EXHAUSTED = "GeminiQuotaExhaustedError"
RATE_LIMITED = "GeminiRateLimitError"


@dataclass(frozen=True)
class EvalConfig:
    label: str
    hybrid: bool
    rerank: bool
    multi_query: bool
    self_correct: bool
    hyde: bool = False

    def options(self, settings: Settings) -> RetrievalOptions:
        return RetrievalOptions.from_settings(
            settings,
            hybrid=self.hybrid,
            rerank=self.rerank,
            multi_query=self.multi_query,
            hyde=self.hyde,
        )


CONFIGS: dict[str, EvalConfig] = {
    "dense": EvalConfig("Dense only (baseline)", False, False, False, False),
    "hybrid": EvalConfig("Hybrid (dense + BM25 + RRF)", True, False, False, False),
    "hybrid_rerank": EvalConfig("Hybrid + reranking", True, True, False, False),
    "full": EvalConfig(
        "Full (hybrid + reranking + query rewriting + self-correction)", True, True, True, True
    ),
    # Not in the default comparison: HyDE on top of the full pipeline.
    "full_hyde": EvalConfig("Full + HyDE", True, True, True, True, hyde=True),
}
DEFAULT_CONFIGS = ("dense", "hybrid", "hybrid_rerank", "full")


async def evaluate_item(
    service: RAGService,
    judge: Judge | None,
    item: EvalItem,
    name: str,
    config: EvalConfig,
    *,
    collection: str,
    k: int,
) -> QuestionResult:
    """Answer one question under one configuration and score it."""
    base = QuestionResult(config=name, item_id=item.id, kind=item.kind, answerable=item.answerable)
    started = time.perf_counter()
    try:
        result = await service.ask(
            item.question,
            collection=collection,
            options=config.options(service.settings),
            self_correct=config.self_correct,
            use_cache=False,
        )
    except GeminiError as exc:
        return base.model_copy(update={"error": type(exc).__name__})
    latency_ms = (time.perf_counter() - started) * 1000
    answer, passages = result.answer, result.passages
    ranked = [p.parent.parent_id for p in passages]
    refused = is_refusal(answer)
    scores = base.model_copy(
        update={
            "route": answer.route.value,
            "refused": refused,
            "answer": answer.text,
            "passages": ranked,
            "latency_ms": latency_ms,
            "cost_usd": answer.usage.cost_usd,
            "llm_calls": answer.usage.calls,
        }
    )
    gold = set(item.source_parent_ids)
    if item.answerable and gold:
        scores.hit = hit_at_k(ranked, gold, k)
        scores.reciprocal_rank = reciprocal_rank(ranked, gold)
    if judge is None:
        return scores
    errors = []
    if item.answerable:
        context = await judge.context(item.question, item.answer, passages)
        scores.context_precision, scores.context_recall = context.precision, context.recall
        errors.append(context.error)
    if item.answerable or not refused:
        verdict = await judge.answer(item.question, answer.text, passages)
        # A refusal has no claims worth checking; its relevance still counts (it failed).
        scores.faithfulness = None if refused else verdict.faithfulness
        scores.answer_relevance = verdict.relevance if item.answerable else None
        scores.unsupported_claims = verdict.unsupported
        errors.append(verdict.error)
    scores.judge_error = next((e for e in errors if e), None)
    return scores


def load_checkpoint(path: Path) -> dict[tuple[str, str], QuestionResult]:
    """Latest result per (configuration, question) from a results.jsonl file."""
    done: dict[tuple[str, str], QuestionResult] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = QuestionResult.model_validate_json(line)
                done[(r.config, r.item_id)] = r
    return done


async def run(
    service: RAGService,
    judge: Judge | None,
    items: Sequence[EvalItem],
    config_names: Sequence[str],
    run_dir: Path,
    *,
    collection: str,
    k: int,
    delay_s: float = 0.0,
    rate_limit_wait_s: float = 60.0,
) -> dict[str, list[QuestionResult]]:
    """Evaluate every item under every configuration, skipping finished ones.

    Stops early (keeping everything done so far) when a daily quota runs out.
    """
    checkpoint = run_dir / "results.jsonl"
    done = load_checkpoint(checkpoint)
    if any(CONFIGS[name].rerank for name in config_names):
        # Load the cross-encoder first, so the first question's latency is honest.
        await asyncio.to_thread(service.warm_up)
    stopped = False
    with checkpoint.open("a", encoding="utf-8") as out:
        for name in config_names:
            if stopped:
                break
            config = CONFIGS[name]
            for n, item in enumerate(items, start=1):
                if _finished(done.get((name, item.id)), judged=judge is not None):
                    continue
                for attempt in range(3):
                    result = await evaluate_item(
                        service, judge, item, name, config, collection=collection, k=k
                    )
                    if RATE_LIMITED not in (result.error, result.judge_error) or attempt == 2:
                        break
                    print(f"  rate-limited; waiting {rate_limit_wait_s:.0f} s", flush=True)
                    await asyncio.sleep(rate_limit_wait_s)
                done[(name, item.id)] = result
                out.write(result.model_dump_json() + "\n")
                out.flush()
                status = result.error or (
                    "refused" if result.refused else f"hit={_fmt(result.hit)}"
                )
                print(
                    f"[{name}] {n}/{len(items)} {item.id}: {status} "
                    f"({result.latency_ms / 1000:.1f} s)",
                    flush=True,
                )
                if QUOTA_EXHAUSTED in (result.error, result.judge_error):
                    print(
                        f"\nA daily model quota is used up. Resume after it resets with:\n"
                        f"  python -m eval.run_eval --resume {run_dir}",
                        flush=True,
                    )
                    stopped = True
                    break
                if delay_s:
                    await asyncio.sleep(delay_s)
    grouped: dict[str, list[QuestionResult]] = defaultdict(list)
    wanted = {item.id for item in items}
    for (name, item_id), result in done.items():
        if name in config_names and item_id in wanted:
            grouped[name].append(result)
    return grouped


def _finished(result: QuestionResult | None, *, judged: bool) -> bool:
    """Done unless it failed, or (when judging) its judge scores are missing."""
    if result is None or result.error is not None:
        return False
    return not (judged and result.judge_error is not None)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def report_markdown(
    summaries: Sequence[ConfigSummary],
    results: dict[str, list[QuestionResult]],
    items: Sequence[EvalItem],
    *,
    k: int,
    settings: Settings,
    judge_model: str | None,
) -> str:
    kinds = Counter(item.kind for item in items)
    answerable = sum(1 for item in items if item.answerable)
    labels = {name: CONFIGS[name].label for name in CONFIGS}
    lines = [
        "# NexusRAG evaluation report",
        "",
        f"*{datetime.now():%Y-%m-%d %H:%M}* · {len(items)} questions ({answerable} answerable, "
        f"{len(items) - answerable} unanswerable) · generation `{settings.generation_model}`, "
        f"fast `{settings.fast_model}`, embeddings `{settings.embedding_model}` · judge "
        f"`{judge_model or 'none'}` · k = {k}",
        "",
        markdown_table(summaries, labels, k),
        "",
        "Retrieval metrics are computed on the parent sections given to the answer model. "
        "Faithfulness skips refusals, answer relevance counts answerable questions only, and "
        "cost covers the system's own model calls (not the judge's).",
        "",
        "## Hit rate by question type",
        "",
        "| Configuration | "
        + " | ".join(f"{kind} ({n})" for kind, n in sorted(kinds.items()))
        + " |",
        "|---|" + "---|" * len(kinds),
    ]
    for s in summaries:
        cells = []
        for kind in sorted(kinds):
            rows = [r for r in results.get(s.config, []) if r.kind == kind and r.error is None]
            value = (
                mean([0.0 if r.refused else 1.0 for r in rows])
                if kind == "unanswerable"
                else mean([r.hit for r in rows])
            )
            cells.append("—" if value is None else f"{value:.0%}")
        lines.append(f"| {labels.get(s.config, s.config)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "(For unanswerable questions the cell shows how often the system *answered*: lower "
        "is better.)",
    ]
    last = summaries[-1].config if summaries else None
    misses = [r for r in results.get(last or "", []) if _is_miss(r)]
    if misses:
        by_id = {item.id: item for item in items}
        lines += ["", f"## Where `{labels.get(last or '', last)}` went wrong", ""]
        for r in misses:
            item = by_id[r.item_id]
            lines.append(f"- **{r.item_id}** ({r.kind}) {item.question}: {_miss_reason(r)}")
    return "\n".join(lines) + "\n"


def _is_miss(r: QuestionResult) -> bool:
    if r.error:
        return True
    if not r.answerable:
        return not r.refused
    return r.refused or r.hit == 0.0 or bool(r.unsupported_claims)


def _miss_reason(r: QuestionResult) -> str:
    if r.error:
        return f"error ({r.error})"
    if not r.answerable:
        return "answered a question the documents don't cover"
    reasons = []
    if r.hit == 0.0:
        reasons.append("source section not retrieved")
    if r.refused:
        reasons.append("refused although the documents answer it")
    if r.unsupported_claims:
        reasons.append("unsupported claim: " + "; ".join(r.unsupported_claims[:2]))
    return ", ".join(reasons)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--k", type=int, default=None, help="hit@k cut-off (default: TOP_K)")
    parser.add_argument("--judge-model", default=None, help="default: GENERATION_MODEL")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judged metrics")
    parser.add_argument("--resume", type=Path, default=None, help="run folder to continue")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        parser.error(f"unknown configurations {unknown}; choose from {sorted(CONFIGS)}")

    settings = get_settings().model_copy(update={"log_level": "WARNING"})
    configure_logging(settings)
    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    collection = args.collection or settings.default_collection
    k = args.k or settings.top_k

    service = RAGService.create(settings)
    for warning in resolve_gold(items, service.stores, collection):
        print(f"warning: {warning}", file=sys.stderr)
    judge_model = None if args.no_judge else (args.judge_model or settings.generation_model)
    judge = (
        None
        if judge_model is None
        else Judge(
            GeminiClient(
                settings.model_copy(
                    update={
                        "fast_model": judge_model,
                        "fast_fallback_model": settings.generation_fallback_model,
                    }
                )
            )
        )
    )

    run_dir = args.resume or REPORTS_DIR / datetime.now().strftime("%Y-%m-%dT%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(items)} questions × {len(names)} configurations -> {run_dir}")
    try:
        results = asyncio.run(
            run(
                service,
                judge,
                items,
                names,
                run_dir,
                collection=collection,
                k=k,
                delay_s=args.delay,
            )
        )
    finally:
        service.close()

    summaries = [summarize(name, results.get(name, [])) for name in names]
    report = report_markdown(
        summaries, results, items, k=k, settings=settings, judge_model=judge_model
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    write_summary_csv(summaries, run_dir / "summary.csv")
    write_results_csv([r for name in names for r in results.get(name, [])], run_dir / "results.csv")
    shutil.copyfile(run_dir / "summary.md", REPORTS_DIR / "latest.md")
    print()
    print(markdown_table(summaries, {n: CONFIGS[n].label for n in CONFIGS}, k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
