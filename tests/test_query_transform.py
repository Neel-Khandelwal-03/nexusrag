from __future__ import annotations

import json

from nexusrag.llm.gemini_client import GeminiClient
from nexusrag.models import ChatTurn
from nexusrag.retrieval.query_transform import QueryTransformer, format_history
from tests.fakes import FakeGenAI, api_error, make_response

HISTORY = [
    ChatTurn(role="user", content="Tell me about the Aurora X1."),
    ChatTurn(role="assistant", content="The Aurora X1 is a field inspection drone [1]."),
]


def structured(payload: dict[str, object]) -> object:
    return make_response(json.dumps(payload))


async def test_no_history_skips_condensation(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    plan = await QueryTransformer(gemini).plan("What is the flight time?")
    assert plan.standalone == "What is the flight time?"
    assert [(q.text, q.kind) for q in plan.queries] == [("What is the flight time?", "original")]
    assert fake_genai.models.calls == []


async def test_condenses_follow_up_with_history(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(
        structured({"standalone_question": "What is the warranty on the Aurora X1?"})
    )
    plan = await QueryTransformer(gemini).plan("And its warranty?", HISTORY)
    assert plan.standalone == "What is the warranty on the Aurora X1?"
    assert plan.queries[0].kind == "standalone"
    call = fake_genai.models.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"  # the cheap model
    assert "User: Tell me about the Aurora X1." in call["contents"]
    assert "Latest message: And its warranty?" in call["contents"]


async def test_multi_query_and_hyde(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    def route(prompt: str) -> object:
        if "alternative search queries" in prompt:
            return structured(
                {
                    "queries": [
                        "flight endurance",
                        "What is the flight time?",
                        "flight endurance ",
                        "battery life per charge",
                        "minutes of flight",
                        "extra one",
                    ]
                }
            )
        return make_response("The Aurora X1 flies for 46 minutes on one battery.")

    fake_genai.models.generate_fn = route
    plan = await QueryTransformer(gemini).plan(
        "What is the flight time?", multi_query=True, num_variants=3, hyde=True
    )
    # Duplicates and the original question are dropped, then capped at n.
    assert plan.variants == ["flight endurance", "battery life per charge", "minutes of flight"]
    assert plan.hyde == "The Aurora X1 flies for 46 minutes on one battery."
    assert [q.kind for q in plan.queries] == ["original", "variant", "variant", "variant", "hyde"]


async def test_failures_fall_back_to_the_original_question(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_fn = lambda prompt: api_error(400, "bad request")
    plan = await QueryTransformer(gemini).plan(
        "And its warranty?", HISTORY, multi_query=True, hyde=True
    )
    assert plan.standalone == "And its warranty?"
    assert plan.variants == []
    assert plan.hyde is None


def test_format_history_truncates_long_turns() -> None:
    text = format_history([ChatTurn(role="assistant", content="x" * 5000)])
    assert text.startswith("Assistant: ")
    assert len(text) < 1300
