"""Suggested follow-up questions: when they're offered, cleaning, and failure handling."""

from __future__ import annotations

from nexusrag.generation.follow_ups import FollowUpSuggester, clean_suggestions, should_suggest
from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.models import Answer, Citation, Route, SourceType
from tests.fakes import FakeGenAI, api_error, json_response


def answer(route: Route = Route.DOC_QA, *, refused: bool = False, cited: bool = True) -> Answer:
    citations = [
        Citation(
            index=1,
            parent_id="p1",
            doc_id="d1",
            filename="aurora.pdf",
            source_type=SourceType.PDF,
            title="Aurora",
            text="The battery lasts 46 minutes.",
        )
    ]
    return Answer(
        text="The battery lasts 46 minutes [1].",
        route=route,
        refused=refused,
        citations=citations if cited else [],
    )


def test_only_cited_document_answers_get_follow_ups() -> None:
    assert should_suggest(answer())
    assert should_suggest(answer(Route.COMPARE))
    assert not should_suggest(answer(refused=True))
    assert not should_suggest(answer(cited=False))
    assert not should_suggest(answer(Route.CHITCHAT))
    assert not should_suggest(answer(Route.OUT_OF_SCOPE))


def test_clean_suggestions() -> None:
    raw = [
        "How long does the battery last",  # the original question, reworded case
        '  "What is the   charging time?" ',
        "what is the charging time",  # duplicate
        "",
        "x" * 200,  # too long for a button
        "Does the warranty cover crashes?",
        "Is there a spare battery?",
    ]
    assert clean_suggestions(raw, "How long does the battery last?", n=2) == [
        "What is the charging time?",
        "Does the warranty cover crashes?",
    ]


async def test_suggest_uses_the_fast_model(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(
        json_response(questions=["What is the charging time?", "How heavy is it"])
    )
    suggestions = await FollowUpSuggester(gemini).suggest("Battery life?", answer())
    assert suggestions == ["What is the charging time?", "How heavy is it?"]
    call = fake_genai.models.calls_to("generate_content")[0]
    assert call["model"] == gemini.settings.fast_model
    assert "aurora.pdf" in call["contents"]


async def test_suggest_skips_refusals_and_survives_errors(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    suggester = FollowUpSuggester(gemini)
    assert await suggester.suggest("Salary?", answer(refused=True)) == []
    assert fake_genai.models.calls_to("generate_content") == []

    fake_genai.models.generate_queue.extend([api_error(400, "bad request")])
    assert await suggester.suggest("Battery life?", answer()) == []
