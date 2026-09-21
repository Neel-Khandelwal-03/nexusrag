from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nexusrag.agent.grader import GroundednessGrader, RelevanceGrader
from nexusrag.agent.router import MAX_CATALOG_ENTRIES, Router, format_catalog, is_greeting
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.models import ChatTurn, ContextPassage, ParentSection, Route, SourceType
from nexusrag.store.registry import DocumentRecord
from tests.fakes import FakeGenAI, api_error, json_response, make_response, router_response


def doc(i: int) -> DocumentRecord:
    return DocumentRecord(
        collection="kb",
        doc_id=f"d{i}",
        source=f"f{i}.md",
        filename=f"f{i}.md",
        source_type=SourceType.MARKDOWN,
        title=f"Doc {i}",
        content_hash="h",
        num_parents=1,
        num_chunks=1,
        ingested_at=datetime(2026, 9, 21, tzinfo=UTC),
    )


DOCS = [doc(1), doc(2), doc(3)]


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
            token_count=len(text.split()),
        ),
    )


# --------------------------------------------------------------------------- router


@pytest.mark.parametrize("text", ["hi", "Hello!", "thanks :)", "Thank you.", "good morning", "ok"])
def test_greetings(text: str) -> None:
    assert is_greeting(text)


@pytest.mark.parametrize("text", ["hi, what is the warranty?", "thanks, and the range?", "okay so"])
def test_not_greetings(text: str) -> None:
    assert not is_greeting(text)


async def test_greeting_needs_no_model_call(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    decision = await Router(gemini).route("Hi there!", documents=DOCS)
    assert decision.route == Route.CHITCHAT
    assert not decision.from_model
    assert fake_genai.models.calls == []


async def test_router_maps_catalog_numbers_to_doc_ids(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(
        router_response("compare_documents", "Compare doc 1 and doc 3", documents=[3, 1, 9, 3])
    )
    history = [ChatTurn(role="user", content="Tell me about doc 1")]
    decision = await Router(gemini).route(
        "compare it with the third one", documents=DOCS, history=history
    )
    assert decision.route == Route.COMPARE
    assert decision.doc_ids == ["d3", "d1"]  # out-of-range 9 dropped, duplicates removed
    assert decision.standalone_question == "Compare doc 1 and doc 3"
    prompt = fake_genai.models.calls[0]["contents"]
    assert "3. Doc 3 (f3.md)" in prompt
    assert "User: Tell me about doc 1" in prompt
    assert fake_genai.models.calls[0]["model"] == "gemini-3.5-flash-lite"


async def test_router_empty_standalone_falls_back_to_message(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(router_response("doc_qa", "   "))
    decision = await Router(gemini).route("What is the range?", documents=DOCS)
    assert decision.standalone_question == "What is the range?"


async def test_router_failure_defaults_to_doc_qa(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(api_error(400, "bad"))
    decision = await Router(gemini).route("What is the range?", documents=DOCS)
    assert decision.route == Route.DOC_QA
    assert not decision.from_model


def test_catalog_formatting() -> None:
    assert format_catalog([]) == "(the knowledge base is empty)"
    big = [doc(i) for i in range(MAX_CATALOG_ENTRIES + 5)]
    text = format_catalog(big)
    assert text.count("\n") == MAX_CATALOG_ENTRIES
    assert text.endswith("... and 5 more")


# --------------------------------------------------------------------------- graders


async def test_relevance_verdicts(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.extend(
        [
            json_response(sufficient=True, missing="", better_query=""),
            json_response(
                sufficient=False, missing="flight time", better_query="maximum flight time"
            ),
        ]
    )
    grader = RelevanceGrader(gemini)
    ok = await grader.grade("q", [passage(1, "text")])
    assert ok.sufficient
    assert ok.checked
    miss = await grader.grade("q", [passage(1, "text")])
    assert (miss.sufficient, miss.missing, miss.better_query) == (
        False,
        "flight time",
        "maximum flight time",
    )


async def test_relevance_prompt_truncates_long_passages(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(json_response(sufficient=True))
    await RelevanceGrader(gemini).grade("q", [passage(1, "word " * 3000)])
    prompt = fake_genai.models.calls[0]["contents"]
    assert prompt.count("word") < 600


async def test_relevance_without_passages(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(json_response(sufficient=False, better_query="x"))
    await RelevanceGrader(gemini).grade("q", [])
    assert "(no passages were found)" in fake_genai.models.calls[0]["contents"]


async def test_graders_fail_open(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.extend([api_error(400, "x"), make_response("not json")])
    relevance = await RelevanceGrader(gemini).grade("q", [passage(1, "t")])
    assert relevance.sufficient
    assert not relevance.checked
    grounded = await GroundednessGrader(gemini).grade("answer", [passage(1, "t")])
    assert grounded.grounded
    assert not grounded.checked


async def test_groundedness_verdicts(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.extend(
        [
            json_response(grounded=True, unsupported_claims=[]),
            json_response(grounded=False, unsupported_claims=["flies 90 minutes", " "]),
            json_response(grounded=False, unsupported_claims=[]),
        ]
    )
    grader = GroundednessGrader(gemini)
    assert (await grader.grade("a", [passage(1, "t")])).grounded
    bad = await grader.grade("a", [passage(1, "t")])
    assert (bad.grounded, bad.unsupported_claims) == (False, ["flies 90 minutes"])
    # "Not grounded" without any claim to fix is treated as grounded.
    assert (await grader.grade("a", [passage(1, "t")])).grounded
