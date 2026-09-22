"""The self-correcting agent end to end: every route and every correction path (fake Gemini)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from nexusrag.agent.graph import merge_round_robin
from nexusrag.config import Settings
from nexusrag.ingestion.pipeline import IngestionPipeline
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.llm.prompts import REFUSAL_MESSAGE
from nexusrag.models import AgentStep, Route, SearchFilters
from nexusrag.retrieval.retriever import RetrievalOptions
from nexusrag.service import RAGService
from nexusrag.store import Stores
from tests.fakes import (
    FakeGenAI,
    api_error,
    json_response,
    keyword_embedder,
    make_response,
    router_response,
)

DOCS = {
    # Catalog numbers follow filename order: 1 = aurora.md, 2 = borealis.md, 3 = policy.md
    "aurora.md": """# Aurora Spec

## Battery

A full charge takes 75 minutes with the CH-400 charger.

## Performance

Maximum flight time is 46 minutes without payload.

## Propellers

Inspect the propellers for cracks before every flight.
""",
    "borealis.md": """# Borealis Sheet

## Performance

Maximum flight time is 95 minutes. Wind resistance is 10 m/s.

## Warranty

The warranty lasts 24 months.
""",
    "policy.md": """# Remote Work Policy

## Stipends

Employees receive a home office stipend of 600 USD after probation.
""",
}


class Script:
    """Routes fake generate_content calls by prompt type; each queue pops in order."""

    def __init__(self) -> None:
        self.queues: dict[str, deque[Any]] = {}
        self.prompts: dict[str, list[str]] = {}

    MARKERS: ClassVar[dict[str, str]] = {
        "route": "You route messages",
        "relevance": "You check whether retrieved passages",
        "grounded": "strict fact-checker",
        "map": "Summarise these sections",
        "variants": "alternative search queries",
    }

    def add(self, kind: str, *responses: Any) -> None:
        self.queues.setdefault(kind, deque()).extend(responses)

    def __call__(self, prompt: str) -> Any:
        for kind, marker in self.MARKERS.items():
            if marker in prompt:
                self.prompts.setdefault(kind, []).append(prompt)
                queue = self.queues.get(kind)
                if not queue:
                    raise AssertionError(f"unexpected {kind} call")
                return queue.popleft() if len(queue) > 1 else queue[0]
        raise AssertionError(f"unscripted prompt: {prompt[:80]}")

    def count(self, kind: str) -> int:
        return len(self.prompts.get(kind, []))


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(
        storage_dir=tmp_path / "storage",
        top_k=3,
        enable_self_correction=True,
        max_retrieval_retries=2,
    )


@pytest.fixture
def script(fake_genai: FakeGenAI) -> Script:
    s = Script()
    fake_genai.models.generate_fn = s
    fake_genai.models.embed_fn = keyword_embedder()
    return s


@pytest.fixture
async def service(
    settings: Settings, fake_genai: FakeGenAI, script: Script, tmp_path: Path
) -> Iterator[RAGService]:
    stores = Stores.open(settings)
    svc = RAGService(settings, GeminiClient(settings, client=fake_genai), stores)
    pipe = IngestionPipeline(settings, svc.gemini, stores)
    for name, text in DOCS.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert (await pipe.ingest_file(path, collection="default")).status == "indexed"
    yield svc  # type: ignore[misc]
    svc.close()


def stream(fake: FakeGenAI, *texts: str) -> None:
    for text in texts:
        fake.models.stream_queue.append([make_response(text)])


def nodes(steps: list[AgentStep]) -> list[str]:
    return [s.node for s in steps]


class UI:
    """Collects what a UI would show."""

    def __init__(self) -> None:
        self.text = ""
        self.resets = 0
        self.steps: list[AgentStep] = []
        self.started: list[str] = []

    async def token(self, t: str) -> None:
        self.text += t

    async def reset(self) -> None:
        self.text = ""
        self.resets += 1

    async def step(self, s: AgentStep) -> None:
        assert self.started[-1] == s.node  # every step was announced before it ran
        self.steps.append(s)

    async def start(self, node: str) -> None:
        self.started.append(node)

    def kwargs(self) -> dict[str, Any]:
        return {
            "on_token": self.token,
            "on_reset": self.reset,
            "on_step": self.step,
            "on_node_start": self.start,
        }


# --------------------------------------------------------------------------- simple routes


async def test_greeting_skips_router_and_retrieval(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    stream(fake_genai, "Hi! Ask me anything about your documents.")
    ui = UI()
    result = await service.ask("hi!", **ui.kwargs())
    assert result.answer.route == Route.CHITCHAT
    assert nodes(result.answer.steps) == ["route", "chitchat"]
    assert ui.text == "Hi! Ask me anything about your documents."
    assert script.count("route") == 0
    assert result.retrievals == []
    assert (
        fake_genai.models.calls_to("generate_content_stream")[0]["model"] == "gemini-3.5-flash-lite"
    )


async def test_out_of_scope_is_declined(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("out_of_scope", "Write me a poem about cats"))
    ui = UI()
    result = await service.ask("Write me a poem about cats", **ui.kwargs())
    assert result.answer.route == Route.OUT_OF_SCOPE
    assert result.answer.refused
    assert "outside what I can help with" in ui.text
    assert "3 documents" in ui.text
    assert fake_genai.models.calls_to("generate_content_stream") == []


# ----------------------------------------------------------------- doc_qa + self-correction


async def test_doc_qa_happy_path(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "How long does charging the battery take?"))
    script.add("relevance", json_response(sufficient=True))
    script.add("grounded", json_response(grounded=True, unsupported_claims=[]))
    stream(fake_genai, "A full charge takes 75 minutes [1].")
    ui = UI()
    result = await service.ask("how long to charge?", **ui.kwargs())
    answer = result.answer
    assert nodes(answer.steps) == [
        "route",
        "retrieve",
        "grade_relevance",
        "generate",
        "check_groundedness",
    ]
    assert [s.node for s in ui.steps] == nodes(answer.steps)  # streamed to the UI as they happen
    assert ui.started == nodes(answer.steps)
    # The retrieve step carries the rankings the UI shows as tables.
    detail = answer.steps[1].detail
    assert set(detail["candidates"][0]) == {"location", "dense_rank", "bm25_rank", "rrf", "rerank"}
    assert detail["kept"][0]["location"].startswith("aurora.md")
    assert detail["timings"]
    assert all(isinstance(v, float) for v in detail["timings"].values())
    assert answer.grounded is True
    assert answer.citations[0].index == 1
    assert ui.text == "A full charge takes 75 minutes [1]."
    # The answer model got the router's standalone question.
    prompt = fake_genai.models.calls_to("generate_content_stream")[0]["contents"]
    assert "Question: How long does charging the battery take?" in prompt


async def test_relevance_retry_rewrites_query_and_merges_context(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "How long does the Aurora battery last?"))
    script.add(
        "relevance",
        json_response(
            sufficient=False, missing="flight duration", better_query="maximum flight time"
        ),
        json_response(sufficient=True),
    )
    script.add("grounded", json_response(grounded=True))
    stream(fake_genai, "It flies for 46 minutes [1].")
    result = await service.ask("How long does the Aurora battery last?")
    assert nodes(result.answer.steps) == [
        "route",
        "retrieve",
        "grade_relevance",
        "retrieve",
        "grade_relevance",
        "generate",
        "check_groundedness",
    ]
    assert [r.query for r in result.retrievals] == [
        "How long does the Aurora battery last?",
        "maximum flight time",
    ]
    context = " ".join(p.parent.text for p in result.passages)
    assert "46 minutes" in context  # found by the retry
    assert (
        "Maximum flight time" in result.answer.steps[2].label
        or result.answer.steps[2].detail["better_query"]
    )


async def test_retries_are_capped(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "What is the price of the Aurora?"))
    script.add(
        "relevance",
        json_response(sufficient=False, missing="price", better_query="Aurora price list"),
        json_response(sufficient=False, missing="price", better_query="Aurora cost USD"),
        json_response(sufficient=False, missing="price", better_query="Aurora kit pricing"),
    )
    stream(fake_genai, REFUSAL_MESSAGE)
    result = await service.ask("What is the price of the Aurora?")
    assert nodes(result.answer.steps).count("retrieve") == 3  # 1 + MAX_RETRIEVAL_RETRIES
    assert result.answer.refused
    assert result.answer.closest_matches  # refusal shows what *is* in the documents
    assert "still incomplete" in result.answer.steps[-2].label


async def test_stops_retrying_when_the_rewrite_finds_nothing(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "What is the CEO's salary?"))
    script.add(
        "relevance",
        json_response(sufficient=False, missing="salary", better_query="CEO compensation"),
        json_response(sufficient=False, missing="salary", better_query="executive pay"),
    )
    # A filter on a non-existent document makes every retrieval come back empty.
    result = await service.ask("What is the CEO's salary?", filters=SearchFilters(doc_ids=["nope"]))
    assert nodes(result.answer.steps).count("retrieve") == 2  # not 3: the rewrite was empty too
    assert "found nothing either" in result.answer.steps[-2].label
    assert result.answer.refused
    assert fake_genai.models.calls_to("generate_content_stream") == []  # refused without a call


async def test_repeated_rewrite_does_not_loop(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "battery"))
    script.add("relevance", json_response(sufficient=False, missing="x", better_query="Battery"))
    stream(fake_genai, "Answer [1].")
    script.add("grounded", json_response(grounded=True))
    result = await service.ask("battery")
    assert nodes(result.answer.steps).count("retrieve") == 1


async def test_unsupported_answer_is_regenerated(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "How long does charging take?"))
    script.add("relevance", json_response(sufficient=True))
    script.add(
        "grounded",
        json_response(grounded=False, unsupported_claims=["charges in 20 minutes"]),
        json_response(grounded=True),
    )
    stream(fake_genai, "It charges in 20 minutes [1].", "A full charge takes 75 minutes [1].")
    ui = UI()
    result = await service.ask("How long does charging take?", **ui.kwargs())
    assert nodes(result.answer.steps)[-3:] == [
        "check_groundedness",
        "regenerate",
        "check_groundedness",
    ]
    assert ui.resets == 1
    assert ui.text == result.answer.text == "A full charge takes 75 minutes [1]."
    assert result.answer.grounded is True
    retry_system = fake_genai.models.calls_to("generate_content_stream")[1][
        "config"
    ].system_instruction
    assert "charges in 20 minutes" in retry_system


async def test_still_unsupported_after_regeneration_refuses(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "How long does charging take?"))
    script.add("relevance", json_response(sufficient=True))
    script.add("grounded", json_response(grounded=False, unsupported_claims=["made-up claim"]))
    stream(fake_genai, "Made up [1].", "Still made up [1].")
    ui = UI()
    result = await service.ask("How long does charging take?", **ui.kwargs())
    answer = result.answer
    assert nodes(answer.steps)[-1] == "refuse_unsupported"
    assert answer.text.startswith(REFUSAL_MESSAGE)
    assert ui.text == answer.text
    assert (answer.refused, answer.grounded) == (True, False)
    assert answer.closest_matches
    assert ui.resets == 2  # before regenerating, and before refusing


async def test_generator_refusal_skips_groundedness(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "What is the CEO's salary?"))
    script.add("relevance", json_response(sufficient=True))
    stream(fake_genai, REFUSAL_MESSAGE)
    result = await service.ask("What is the CEO's salary?")
    assert result.answer.refused
    assert script.count("grounded") == 0
    assert 1 <= len(result.answer.closest_matches) <= 3


async def test_self_correction_can_be_switched_off(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "How long does charging take?"))
    stream(fake_genai, "75 minutes [1].")
    result = await service.ask("How long does charging take?", self_correct=False)
    assert nodes(result.answer.steps) == ["route", "retrieve", "generate"]
    assert script.count("relevance") == script.count("grounded") == 0
    assert result.answer.grounded is None


# --------------------------------------------------------------------------- summarize


async def test_summarize_single_pass(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add(
        "route", router_response("summarize_document", "Summarise the Aurora spec", documents=[1])
    )
    script.add("grounded", json_response(grounded=True))
    stream(fake_genai, "The Aurora charges in 75 minutes [1] and flies 46 minutes [2].")
    result = await service.ask("Summarise the Aurora spec")
    answer = result.answer
    assert answer.route == Route.SUMMARIZE
    assert nodes(answer.steps) == ["route", "summarize", "check_groundedness"]
    assert {c.filename for c in answer.citations} == {"aurora.md"}
    assert [c.section_path for c in answer.citations] == ["Battery", "Performance"]
    prompt = fake_genai.models.calls_to("generate_content_stream")[0]["contents"]
    assert "Document: Aurora Spec (aurora.md)" in prompt


async def test_summarize_map_reduce_for_long_documents(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI, script: Script, tmp_path: Path
) -> None:
    settings = make_settings(
        storage_dir=tmp_path / "big",
        summary_stuff_tokens=1000,
        summary_map_batch_tokens=500,
        enable_self_correction=True,
    )
    stores = Stores.open(settings)
    svc = RAGService(settings, GeminiClient(settings, client=fake_genai), stores)
    sections = "\n\n".join(
        f"## Section {i}\n\n" + f"Section {i} describes procedure number {i} in detail. " * 25
        for i in range(8)
    )
    path = tmp_path / "manual.md"
    path.write_text(f"# Manual\n\n{sections}", encoding="utf-8")
    await IngestionPipeline(settings, svc.gemini, stores).ingest_file(path, collection="default")

    script.add(
        "route", router_response("summarize_document", "Summarise the manual", documents=[1])
    )
    script.add("map", make_response("- Procedure details [1]"))
    stream(fake_genai, "The manual covers procedures [1].")
    result = await svc.ask("Summarise the manual")
    assert script.count("map") >= 2
    assert "map-reduce" in result.answer.steps[1].label
    assert nodes(result.answer.steps) == ["route", "summarize"]  # no claim check on map-reduce
    assert result.answer.citations[0].filename == "manual.md"
    svc.close()


async def test_summarize_asks_which_document(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("summarize_document", "Summarise the document"))
    ui = UI()
    result = await service.ask("Summarise the document", **ui.kwargs())
    assert "Which document would you like me to summarize?" in ui.text
    assert "aurora.md" in ui.text
    assert fake_genai.models.calls_to("generate_content_stream") == []
    assert nodes(result.answer.steps) == ["route", "summarize"]


async def test_summarize_uses_the_ui_document_filter(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("summarize_document", "Summarise this"))
    script.add("grounded", json_response(grounded=True))
    stream(fake_genai, "Policy summary [1].")
    doc_id = next(
        d.doc_id
        for d in service.stores.registry.list_documents("default")
        if d.filename == "policy.md"
    )
    result = await service.ask("Summarise this", filters=SearchFilters(doc_ids=[doc_id]))
    assert result.answer.citations[0].filename == "policy.md"


# --------------------------------------------------------------------------- compare


async def test_compare_retrieves_each_document_separately(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add(
        "route",
        router_response(
            "compare_documents", "Compare Aurora and Borealis flight time", documents=[1, 2]
        ),
    )
    script.add("grounded", json_response(grounded=True))
    stream(
        fake_genai,
        "| | Aurora | Borealis |\n|---|---|---|\n| Flight time | 46 min [1] | 95 min [4] |",
    )
    result = await service.ask("Compare Aurora and Borealis flight time")
    answer = result.answer
    assert answer.route == Route.COMPARE
    assert len(result.retrievals) == 2
    per_doc = [{rc.chunk.metadata.filename for rc in r.chunks} for r in result.retrievals]
    assert per_doc == [{"aurora.md"}, {"borealis.md"}]
    assert [p.index for p in result.passages] == list(range(1, len(result.passages) + 1))
    prompt = fake_genai.models.calls_to("generate_content_stream")[0]["contents"]
    assert '<document title="Aurora Spec" file="aurora.md">' in prompt
    assert '<document title="Borealis Sheet" file="borealis.md">' in prompt
    assert {c.filename for c in answer.citations} <= {"aurora.md", "borealis.md"}


async def test_question_about_a_specific_document_searches_only_that_document(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    policy = next(d for d in service.stores.registry.list_documents("default")
                  if d.filename == "policy.md")  # fmt: skip
    script.add("route", router_response("doc_qa", "What is the policy about?", documents=[1]))
    script.add("relevance", json_response(sufficient=True))
    script.add("grounded", json_response(grounded=True, unsupported_claims=[]))
    stream(fake_genai, "It covers a home office stipend [1].")
    result = await service.ask("what is this file about?", recent_doc_ids=[policy.doc_id])
    assert "1. Remote Work Policy (policy.md) [just uploaded]" in script.prompts["route"][0]
    retrieve = result.answer.steps[1]
    assert retrieve.detail["scope"] == ["policy.md"]
    assert {rc.chunk.metadata.filename for rc in result.retrievals[0].chunks} == {"policy.md"}


async def test_named_documents_outside_fresh_uploads_do_not_restrict_the_search(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    # "The Aurora X1's price" may live in a sales report, not the spec the router names.
    script.add("route", router_response("doc_qa", "How long is the flight time?", documents=[1]))
    script.add("relevance", json_response(sufficient=True))
    script.add("grounded", json_response(grounded=True, unsupported_claims=[]))
    stream(fake_genai, "46 minutes [1].")
    result = await service.ask("How long is the Aurora flight time?")
    assert result.answer.steps[1].detail["scope"] == []


async def test_ui_document_filter_wins_over_the_router(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    docs = {d.filename: d.doc_id for d in service.stores.registry.list_documents("default")}
    script.add("route", router_response("doc_qa", "How long is the flight time?", documents=[3]))
    script.add("relevance", json_response(sufficient=True))
    script.add("grounded", json_response(grounded=True, unsupported_claims=[]))
    stream(fake_genai, "46 minutes [1].")
    result = await service.ask(
        "How long is the flight time?", filters=SearchFilters(doc_ids=[docs["aurora.md"]])
    )
    assert result.answer.steps[1].detail["scope"] == ["aurora.md"]


async def test_compare_does_not_drop_passages_below_the_rerank_threshold(
    service: RAGService, fake_genai: FakeGenAI, script: Script, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[RetrievalOptions] = []
    original = service.retriever.retrieve

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["options"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(service.retriever, "retrieve", spy)
    script.add(
        "route",
        router_response("compare_documents", "Compare flight time and warranty", documents=[1, 2]),
    )
    script.add("grounded", json_response(grounded=True))
    stream(fake_genai, "Aurora: 46 min [1]. Borealis: 95 min [2].")
    await service.ask("Compare flight time and warranty")
    assert [o.rerank_threshold for o in seen] == [0.0, 0.0]
    assert {o.top_k for o in seen} == {service.settings.compare_top_k_per_doc}


async def test_compare_needs_two_documents(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("compare_documents", "Compare the Aurora", documents=[1]))
    ui = UI()
    await service.ask("Compare the Aurora", **ui.kwargs())
    assert "Which documents would you like me to compare?" in ui.text


async def test_mode_forces_route_but_not_for_greetings(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "Tell me about the Aurora", documents=[1]))
    script.add("grounded", json_response(grounded=True))
    stream(fake_genai, "Aurora summary [1].", "Hello!")
    forced = await service.ask("Tell me about the Aurora", mode=Route.SUMMARIZE)
    assert forced.answer.route == Route.SUMMARIZE
    assert forced.answer.steps[0].detail["forced_by_mode"] is True
    greeting = await service.ask("hello", mode=Route.SUMMARIZE)
    assert greeting.answer.route == Route.CHITCHAT


# --------------------------------------------------------------------------- helpers


def test_merge_round_robin() -> None:
    from tests.test_hybrid_rerank import candidates_from

    first = candidates_from(["a", "b", "c"])
    second = candidates_from(["x", "y"])
    second = [
        rc.model_copy(update={"chunk": rc.chunk.model_copy(update={"chunk_id": f"s{i}"})})
        for i, rc in enumerate(second)
    ]
    merged = merge_round_robin(first, [*second, first[0]], limit=4)
    assert [rc.chunk.chunk_id for rc in merged] == ["c0", "s0", "c1", "s1"]


# --------------------------------------------------------------------------- semantic cache


def answerable(
    script: Script, fake: FakeGenAI, text: str = "A full charge takes 75 minutes [1]."
) -> None:
    script.add("route", router_response("doc_qa", "How long does charging the battery take?"))
    script.add("relevance", json_response(sufficient=True))
    script.add("grounded", json_response(grounded=True, unsupported_claims=[]))
    stream(fake, text)


def embed_calls(fake: FakeGenAI) -> int:
    return len(fake.models.calls_to("embed_content"))


async def test_repeated_question_is_served_from_the_cache(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    answerable(script, fake_genai)
    before = embed_calls(fake_genai)
    first = await service.ask("how long to charge?", use_cache=True)
    assert not first.answer.cached
    assert nodes(first.answer.steps)[:3] == ["route", "check_cache", "retrieve"]
    assert first.answer.steps[1].detail["hit"] is False
    # The cache check and retrieval share one embedding of the question.
    assert embed_calls(fake_genai) - before == 1

    ui = UI()
    second = await service.ask("how long does it take to charge?", use_cache=True, **ui.kwargs())
    answer = second.answer
    assert answer.cached
    assert nodes(answer.steps) == ["route", "check_cache"]
    assert answer.cache_similarity == pytest.approx(1.0, abs=1e-5)
    assert ui.text == "A full charge takes 75 minutes [1]."
    assert answer.citations == first.answer.citations
    assert answer.grounded is True
    assert second.retrievals == []
    assert len(fake_genai.models.calls_to("generate_content_stream")) == 1  # generated once
    detail = answer.steps[1].detail
    assert detail["cached_question"] == "How long does charging the battery take?"
    assert detail["hits"] == 1
    stats = service.stats()
    assert service.stores.cache.count("default") == 1
    assert (stats.cache_hits, stats.cache_lookups) == (None, None)  # cache off in settings


async def test_other_settings_do_not_share_cached_answers(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    answerable(script, fake_genai)
    await service.ask("how long to charge?", use_cache=True)
    stream(fake_genai, "75 minutes [1].")
    concise = await service.ask("how long to charge?", use_cache=True, style="concise")
    assert not concise.answer.cached
    assert concise.answer.steps[1].detail["candidates"] == 0  # different fingerprint


async def test_ingestion_invalidates_cached_answers(
    service: RAGService, fake_genai: FakeGenAI, script: Script, tmp_path: Path
) -> None:
    answerable(script, fake_genai)
    await service.ask("how long to charge?", use_cache=True)
    assert service.stores.cache.count("default") == 1
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\n## Charging\n\nFast charging takes 40 minutes.\n", encoding="utf-8")
    pipe = IngestionPipeline(service.settings, service.gemini, service.stores)
    assert (await pipe.ingest_file(path, collection="default")).status == "indexed"
    assert service.stores.cache.count("default") == 0

    stream(fake_genai, "A full charge takes 75 minutes [1].")
    again = await service.ask("how long to charge?", use_cache=True)
    assert not again.answer.cached


async def test_refusals_and_chitchat_are_not_cached(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    script.add("route", router_response("doc_qa", "What is the CEO's salary?"))
    script.add("relevance", json_response(sufficient=True))
    stream(fake_genai, REFUSAL_MESSAGE)
    refused = await service.ask("What is the CEO's salary?", use_cache=True)
    assert refused.answer.refused
    assert service.stores.cache.count("default") == 0

    stream(fake_genai, "Hello! Ask me about your documents.")
    greeting = await service.ask("hi", use_cache=True)
    assert "check_cache" not in nodes(greeting.answer.steps)


async def test_cache_is_skipped_when_embedding_fails(
    service: RAGService, fake_genai: FakeGenAI, script: Script
) -> None:
    answerable(script, fake_genai)
    embed = fake_genai.models.embed_fn
    failures = [api_error(400, "embedding failed")]

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if failures:
            raise failures.pop()
        return embed(*args, **kwargs)

    fake_genai.models.embed_fn = flaky
    result = await service.ask("how long to charge?", use_cache=True)
    assert result.answer.steps[1].label == "Semantic cache: skipped (embedding failed)"
    assert result.answer.citations  # answered normally
    assert service.stores.cache.count("default") == 0  # nothing to key it by
