"""The evaluation suite: metrics, judges, dataset handling and the configuration runner."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from eval.dataset import EvalItem, load_dataset, normalize, resolve_gold, save_dataset
from eval.generate_dataset import finalize, heading, pick_sections, valid_quotes
from eval.metrics import (
    Judge,
    QuestionResult,
    average_precision,
    hit_at_k,
    is_refusal,
    markdown_table,
    mean,
    percentile,
    reciprocal_rank,
    share,
    summarize,
)
from eval.run_eval import CONFIGS, load_checkpoint, report_markdown, run
from nexusrag.config import Settings
from nexusrag.ingestion.pipeline import IngestionPipeline
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.llm.prompts import REFUSAL_MESSAGE
from nexusrag.models import Answer, ContextPassage, ParentSection, Route, SourceType
from nexusrag.service import RAGService
from nexusrag.store import Stores
from tests.fakes import FakeGenAI, api_error, json_response, keyword_embedder, make_response

# --------------------------------------------------------------------------- metrics


def test_hit_and_reciprocal_rank() -> None:
    ranked = ["p3", "p1", "p7"]
    assert hit_at_k(ranked, {"p1"}, k=2) == 1.0
    assert hit_at_k(ranked, {"p7"}, k=2) == 0.0
    assert reciprocal_rank(ranked, {"p1", "p7"}) == 0.5
    assert reciprocal_rank(ranked, {"p9"}) == 0.0
    assert reciprocal_rank([], {"p1"}) == 0.0


def test_average_precision_rewards_relevant_passages_first() -> None:
    assert average_precision([True, True, False]) == 1.0
    assert average_precision([False, True]) == 0.5
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0


def test_statistics() -> None:
    assert share([True, False, True, True]) == 0.75
    assert share([]) is None
    assert mean([0.5, None, 1.0]) == 0.75
    assert mean([None]) is None
    assert percentile([100, 200, 300, 400], 50) == 250
    assert percentile([100, 200, 300, 400], 95) == pytest.approx(385)
    assert percentile([], 50) is None


def test_refusal_detection() -> None:
    assert is_refusal(Answer(text="x", route=Route.OUT_OF_SCOPE, refused=True))
    assert is_refusal(
        Answer(text=f"{REFUSAL_MESSAGE} The passages cover pricing.", route=Route.DOC_QA)
    )
    assert not is_refusal(Answer(text="It lasts 46 minutes [1].", route=Route.DOC_QA))


def result(**fields: Any) -> QuestionResult:
    base: dict[str, Any] = {"config": "dense", "item_id": "q1", "kind": "fact", "answerable": True}
    base.update(fields)
    return QuestionResult(**base)


def test_summarize_separates_answerable_and_unanswerable() -> None:
    rows = [
        result(item_id="q1", hit=1.0, reciprocal_rank=1.0, faithfulness=1.0,
               answer_relevance=1.0, latency_ms=1000, cost_usd=0.002),
        result(item_id="q2", hit=0.0, reciprocal_rank=0.0, refused=True,
               answer_relevance=0.0, latency_ms=3000, cost_usd=0.004),
        result(item_id="q3", answerable=False, kind="unanswerable", refused=True,
               latency_ms=2000, cost_usd=0.003),
        result(item_id="q4", error="GeminiRateLimitError"),
    ]  # fmt: skip
    s = summarize("dense", rows)
    assert (s.questions, s.errors) == (4, 1)
    assert (s.hit_at_k, s.mrr) == (0.5, 0.5)
    assert s.faithfulness == 1.0  # the refusal has no claims to check
    assert s.answer_relevance == 0.5
    assert (s.correct_refusals, s.false_refusals) == (1.0, 0.5)
    assert s.latency_p50_ms == 2000
    assert s.cost_per_query_usd == pytest.approx(0.003)
    table = markdown_table([s], {"dense": "Dense only"}, k=5)
    assert "| Dense only | 50% | 0.50 |" in table
    assert "| 2.0 s / 2.9 s | $0.0030 |" in table
    assert "Hit@5" in table


def test_latency_excludes_rate_limited_questions() -> None:
    rows = [
        result(item_id="q1", latency_ms=1000, retries=0, fallbacks=0),
        result(item_id="q2", latency_ms=2000, retries=0, fallbacks=0),
        result(item_id="q3", latency_ms=60000, retries=3, fallbacks=1),
    ]
    s = summarize("full", rows)
    assert s.throttled == 1
    assert s.clean_latency_p50_ms == 1500
    assert s.latency_p95_ms == pytest.approx(54200)
    assert "| 1.5 s / 1.9 s |" in markdown_table([s], {}, k=5)  # p95 of 1 s and 2 s


# --------------------------------------------------------------------------- judges


def passage(index: int, text: str) -> ContextPassage:
    return ContextPassage(
        index=index,
        parent=ParentSection(
            parent_id=f"p{index}",
            doc_id="d",
            collection="kb",
            filename="f.md",
            source_type=SourceType.MARKDOWN,
            title="T",
            text=text,
            token_count=5,
        ),
    )


async def test_context_judge(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(
        json_response(
            passages=[{"id": 2, "relevant": True}, {"id": 1, "relevant": False}],
            statements=[
                {"statement": "Charging takes 75 minutes", "supported": True},
                {"statement": "It uses the CH-400", "supported": False},
            ],
        )
    )
    scores = await Judge(gemini).context(
        "How long to charge?", "75 minutes with the CH-400.", [passage(1, "a"), passage(2, "b")]
    )
    assert scores.precision == 0.5  # the only relevant passage is ranked second
    assert scores.recall == 0.5
    empty = await Judge(gemini).context("q", "r", [])
    assert (empty.precision, empty.recall) == (0.0, 0.0)


async def test_answer_judge_and_failures(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.extend(
        [
            json_response(
                claims=[
                    {"claim": "It lasts 46 minutes", "supported": True},
                    {"claim": "It weighs 3 kg", "supported": False},
                ],
                relevance=4,
            ),
            api_error(400, "bad request"),
        ]
    )
    judge = Judge(gemini)
    scores = await judge.answer("Battery life?", "46 minutes, weighs 3 kg.", [passage(1, "x")])
    assert (scores.faithfulness, scores.relevance) == (0.5, 0.75)
    assert scores.unsupported == ["It weighs 3 kg"]
    failed = await judge.answer("q", "a", [])
    assert failed.error == "GeminiError"
    assert failed.faithfulness is None


# --------------------------------------------------------------------------- dataset


def test_dataset_round_trip_and_validation(tmp_path: Path) -> None:
    items = [
        EvalItem(id="q001", question="How long?", answer="46 min", source_parent_ids=["p1"]),
        EvalItem(id="q002", question="CEO salary?", answerable=False, kind="unanswerable"),
    ]
    path = tmp_path / "dataset.jsonl"
    save_dataset(items, path)
    assert load_dataset(path) == items
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate ids"):
        load_dataset(path)


def test_normalize_matches_table_rows_and_curly_quotes() -> None:
    table = "| Metric | Value |\n|---|---|\n| **Maximum flight time** | 46 minutes |"
    assert normalize("Maximum flight time 46 minutes") in normalize(table)
    assert normalize("the pilot’s  licence") == "the pilot's licence"


def test_generator_helpers() -> None:
    def parent(i: int, tokens: int, text: str = "t") -> ParentSection:
        return ParentSection(
            parent_id=f"p{i}", doc_id="d", collection="kb", filename="f.md",
            source_type=SourceType.MARKDOWN, title="T", text=text, token_count=tokens,
        )  # fmt: skip

    parents = [parent(i, 10 if i == 1 else 100) for i in range(10)]
    picked = pick_sections(parents, 3)
    assert [p.parent_id for p in picked] == ["p0", "p4", "p7"]  # p1 is too short
    assert heading("2 Hardware > 2.3 Battery System") == "battery system"
    assert heading("3 Performance") == "performance"
    section = parent(0, 100, "The battery lasts **46 minutes** at 25 °C.")
    assert valid_quotes(["battery lasts 46 minutes", "lasts 50 minutes", " "], [section]) == [
        "battery lasts 46 minutes"
    ]
    items = finalize(
        [EvalItem(id="", question="Battery life?"), EvalItem(id="", question="battery  life?"),
         EvalItem(id="", question="Range?")]
    )  # fmt: skip
    assert [(i.id, i.question) for i in items] == [("q001", "Battery life?"), ("q002", "Range?")]


# --------------------------------------------------------------------------- runner

DOCS = {
    "aurora.md": "# Aurora Spec\n\n## Battery\n\nA full charge takes 75 minutes with the "
    "CH-400 charger.\n\n## Warranty\n\nThe warranty lasts 12 months.\n",
    "policy.md": "# Remote Work Policy\n\n## Stipends\n\nEmployees receive a home office "
    "stipend of 600 USD.\n",
}


@pytest.fixture
def service(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI, tmp_path: Path
) -> Iterator[RAGService]:
    settings = make_settings(storage_dir=tmp_path / "storage", top_k=3)
    fake_genai.models.embed_fn = keyword_embedder()
    svc = RAGService(settings, GeminiClient(settings, client=fake_genai), Stores.open(settings))
    yield svc
    svc.close()


async def ingest(svc: RAGService, tmp_path: Path) -> None:
    pipe = IngestionPipeline(svc.settings, svc.gemini, svc.stores)
    for name, text in DOCS.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert (await pipe.ingest_file(path, collection="default")).status == "indexed"


def dataset(svc: RAGService) -> list[EvalItem]:
    aurora = next(
        d for d in svc.stores.registry.list_documents("default") if d.filename == "aurora.md"
    )
    battery = next(
        p for p in svc.stores.parents.for_document("default", aurora.doc_id)
        if "Battery" in p.section_path
    )  # fmt: skip
    return [
        EvalItem(id="q001", question="How long does charging the battery take?",
                 answer="75 minutes with the CH-400 charger.", documents=["aurora.md"],
                 source_parent_ids=[battery.parent_id],
                 evidence=["A full charge takes 75 minutes"]),
        EvalItem(id="q002", question="What is the CEO's salary?", answerable=False,
                 kind="unanswerable"),
    ]  # fmt: skip


def scripted(prompt: str) -> Any:
    if "You route messages" in prompt:
        question = prompt.rsplit("Latest message:", 1)[1].strip()
        return json_response(route="doc_qa", documents=[], standalone_question=question, reason="")
    if "evaluating the retrieval step" in prompt:
        return json_response(
            passages=[{"id": 1, "relevant": True}],
            statements=[{"statement": "75 minutes", "supported": True}],
        )
    if "evaluating an answer" in prompt:
        return json_response(claims=[{"claim": "75 minutes", "supported": True}], relevance=5)
    raise AssertionError(f"unscripted prompt: {prompt[:60]}")


async def test_run_scores_configs_and_resumes(
    service: RAGService, fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    items = dataset(service)
    fake_genai.models.generate_fn = scripted
    for _ in range(2):  # two configurations
        fake_genai.models.stream_queue.append([make_response("It takes 75 minutes [1].")])
        fake_genai.models.stream_queue.append([make_response(REFUSAL_MESSAGE)])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    judge = Judge(service.gemini)
    results = await run(
        service, judge, items, ["dense", "hybrid"], run_dir, collection="default", k=3
    )
    dense = {r.item_id: r for r in results["dense"]}
    assert dense["q001"].hit == 1.0
    assert dense["q001"].context_precision == 1.0
    assert dense["q001"].faithfulness == 1.0
    assert dense["q001"].answer_relevance == 1.0
    assert dense["q002"].refused
    assert dense["q002"].faithfulness is None  # refusals aren't judged for claims
    assert (dense["q001"].retries, dense["q001"].fallbacks) == (0, 0)
    assert all(r.error is None for rs in results.values() for r in rs)

    # Resuming does no new work: every (configuration, question) is already done.
    calls = len(fake_genai.models.calls)
    again = await run(
        service, judge, items, ["dense", "hybrid"], run_dir, collection="default", k=3
    )
    assert len(fake_genai.models.calls) == calls
    assert {k: len(v) for k, v in again.items()} == {"dense": 2, "hybrid": 2}
    assert len(load_checkpoint(run_dir / "results.jsonl")) == 4

    summaries = [summarize(name, again[name]) for name in ("dense", "hybrid")]
    report = report_markdown(
        summaries, again, items, k=3, settings=service.settings, judge_model="judge-model"
    )
    assert "| Dense only (baseline) | 100% |" in report
    assert "unanswerable (1)" in report
    assert "went wrong" not in report  # nothing missed


async def test_errors_are_recorded_and_retried(
    service: RAGService, fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    items = dataset(service)[:1]
    fake_genai.models.generate_fn = scripted
    fake_genai.models.stream_queue.append([api_error(400, "bad request")])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = await run(service, None, items, ["dense"], run_dir, collection="default", k=3)
    assert first["dense"][0].error == "GeminiError"
    fake_genai.models.stream_queue.append([make_response("It takes 75 minutes [1].")])
    second = await run(service, None, items, ["dense"], run_dir, collection="default", k=3)
    assert second["dense"][0].error is None  # retried on resume
    assert second["dense"][0].context_precision is None  # no judge


async def test_daily_quota_stops_the_run_and_resume_finishes_it(
    service: RAGService, fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    items = dataset(service)
    fake_genai.models.generate_fn = scripted
    daily = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    fake_genai.models.stream_queue.append(
        [api_error(429, "limit: 20, model: gemini-3.8-flash", quota_id=daily)]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stopped = await run(service, None, items, ["dense", "hybrid"], run_dir,
                        collection="default", k=3)  # fmt: skip
    assert [r.error for r in stopped["dense"]] == ["GeminiQuotaExhaustedError"]
    assert "hybrid" not in stopped  # nothing else was attempted

    for _ in range(2):
        fake_genai.models.stream_queue.append([make_response("It takes 75 minutes [1].")])
        fake_genai.models.stream_queue.append([make_response(REFUSAL_MESSAGE)])
    resumed = await run(service, None, items, ["dense", "hybrid"], run_dir,
                        collection="default", k=3)  # fmt: skip
    assert {k: [r.error for r in v] for k, v in resumed.items()} == {
        "dense": [None, None],
        "hybrid": [None, None],
    }


async def test_per_minute_limits_wait_and_retry(
    service: RAGService, fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    items = dataset(service)[:1]
    fake_genai.models.generate_fn = scripted
    per_minute = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    fake_genai.models.stream_queue.extend([[api_error(429, quota_id=per_minute)]] * 3)
    fake_genai.models.stream_queue.append([make_response("It takes 75 minutes [1].")])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = await run(service, None, items, ["dense"], run_dir, collection="default", k=3,
                        rate_limit_wait_s=0)  # fmt: skip
    assert results["dense"][0].error is None
    assert results["dense"][0].hit == 1.0


async def test_answers_from_a_fallback_model_are_redone(
    service: RAGService, fake_genai: FakeGenAI, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    items = dataset(service)[:1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    old = QuestionResult(config="dense", item_id="q001", kind="fact", answerable=True,
                         fallbacks=1, answer="from the backup model")  # fmt: skip
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        print(old.model_dump_json(), file=f)
    fake_genai.models.generate_fn = scripted
    fake_genai.models.stream_queue.append([make_response("It takes 75 minutes [1].")])

    kept = await run(service, None, items, ["dense"], run_dir, collection="default", k=3,
                     allow_fallback=True)  # fmt: skip
    assert kept["dense"][0].answer == "from the backup model"
    redone = await run(service, None, items, ["dense"], run_dir, collection="default", k=3)
    assert redone["dense"][0].answer == "It takes 75 minutes [1]."
    assert redone["dense"][0].fallbacks == 0


async def test_resolve_gold_refinds_sections_from_evidence(
    service: RAGService, tmp_path: Path
) -> None:
    await ingest(service, tmp_path)
    item = dataset(service)[0]
    expected = item.source_parent_ids
    stale = item.model_copy(update={"source_parent_ids": ["gone"]})
    lost = EvalItem(id="q9", question="?", documents=["aurora.md"], evidence=["not in the doc"])
    warnings = resolve_gold([stale, lost], service.stores, "default")
    assert stale.source_parent_ids == expected
    assert warnings == ["q9: source sections not found; retrieval metrics skipped"]


def test_configs_cover_the_comparison() -> None:
    dense, full = CONFIGS["dense"], CONFIGS["full"]
    assert not any([dense.hybrid, dense.rerank, dense.multi_query, dense.self_correct])
    assert all([full.hybrid, full.rerank, full.multi_query, full.self_correct])
    assert json.dumps(sorted(CONFIGS)) == json.dumps(
        ["dense", "full", "full_hyde", "hybrid", "hybrid_rerank"]
    )
