from __future__ import annotations

from collections.abc import Callable

import httpx
import numpy as np
import pytest
from google.genai import types
from pydantic import BaseModel

from nexusrag.config import Settings
from nexusrag.llm.gemini_client import (
    GeminiBlockedError,
    GeminiClient,
    GeminiConfigError,
    GeminiError,
    GeminiQuotaExhaustedError,
    GeminiRateLimitError,
    StructuredOutputError,
    format_embedding_input,
    is_retryable,
    l2_normalize,
    parse_structured,
)
from nexusrag.llm.usage import track_usage
from tests.fakes import FakeGenAI, api_error, embedded_texts, hash_embedder, make_response


class Verdict(BaseModel):
    relevant: bool
    reason: str


# --------------------------------------------------------------------------- construction


def test_missing_key_raises_config_error(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY"):
        GeminiClient(make_settings(gemini_api_key=None))


def test_model_per_role(gemini: GeminiClient) -> None:
    assert gemini.model_for("main") == "gemini-3.8-flash"
    assert gemini.model_for("fast") == "gemini-3.5-flash-lite"


def test_config_omits_temperature_by_default(gemini: GeminiClient) -> None:
    config = gemini.build_config("main")
    assert config.temperature is None
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    # Fast role leaves thinking at the model default.
    assert gemini.build_config("fast").thinking_config is None


def test_config_sends_temperature_when_configured(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI
) -> None:
    client = GeminiClient(make_settings(fast_temperature=0.2), client=fake_genai)
    assert client.build_config("fast").temperature == 0.2


# --------------------------------------------------------------------------- generation


async def test_generate_returns_text_and_records_cost(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(
        make_response(
            "pong",
            thought="hidden reasoning",
            prompt_tokens=1_000_000,
            output_tokens=0,
            thoughts_tokens=0,
        )
    )
    with track_usage() as usage:
        result = await gemini.generate("ping", role="fast", stage="test")
    assert result.text == "pong"  # thought parts are excluded
    assert result.model == "gemini-3.5-flash-lite"
    totals = usage.totals()
    assert totals.calls == 1
    assert totals.prompt_tokens == 1_000_000
    assert totals.cost_usd == pytest.approx(0.30)  # 1M input tokens at $0.30/M


async def test_thinking_tokens_billed_as_output(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(
        make_response("x", prompt_tokens=0, output_tokens=500_000, thoughts_tokens=500_000)
    )
    with track_usage() as usage:
        await gemini.generate("q", role="main")
    assert usage.totals().cost_usd == pytest.approx(3.75)  # 1M output-rate tokens on 3.8 Flash


async def test_retries_rate_limit_then_succeeds(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.extend([api_error(429), api_error(503), make_response("ok")])
    result = await gemini.generate("q")
    assert result.text == "ok"
    assert len(fake_genai.models.calls_to("generate_content")) == 3


async def test_gives_up_after_max_attempts(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.extend([api_error(429)] * 3)
    with pytest.raises(GeminiRateLimitError) as info:
        await gemini.generate("q")
    assert "rate-limiting" in info.value.user_message
    assert len(fake_genai.models.calls_to("generate_content")) == 3  # llm_max_attempts


async def test_bad_request_is_not_retried(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(api_error(400, "invalid schema"))
    with pytest.raises(GeminiError):
        await gemini.generate("q")
    assert len(fake_genai.models.calls_to("generate_content")) == 1


@pytest.mark.parametrize(
    ("code", "message"), [(400, "API key not valid. Please pass a valid API key."), (403, "no")]
)
async def test_auth_errors_become_config_errors(
    gemini: GeminiClient, fake_genai: FakeGenAI, code: int, message: str
) -> None:
    fake_genai.models.generate_queue.append(api_error(code, message))
    with pytest.raises(GeminiConfigError):
        await gemini.generate("q")


async def test_unknown_model_is_config_error(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(api_error(404, "model not found"))
    with pytest.raises(GeminiConfigError) as info:
        await gemini.generate("q")
    assert "model names" in info.value.user_message


async def test_blocked_prompt(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(make_response(block_reason="SAFETY"))
    with pytest.raises(GeminiBlockedError):
        await gemini.generate("q")


def test_is_retryable_network_errors() -> None:
    assert is_retryable(httpx.ReadTimeout("slow"))
    assert is_retryable(TimeoutError())
    assert not is_retryable(ValueError())


# --------------------------------------------------------------------------- structured output


async def test_generate_structured_sends_schema_and_parses(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(make_response('{"relevant": true, "reason": "match"}'))
    verdict = await gemini.generate_structured("grade", Verdict, stage="grade")
    assert verdict == Verdict(relevant=True, reason="match")
    config = fake_genai.models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["properties"].keys() == {"relevant", "reason"}


def test_parse_structured_tolerates_code_fence() -> None:
    text = '```json\n{"relevant": false, "reason": "off-topic"}\n```'
    assert parse_structured(text, Verdict).relevant is False


async def test_structured_output_mismatch(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(make_response('{"relevant": "maybe"}'))
    with pytest.raises(StructuredOutputError):
        await gemini.generate_structured("grade", Verdict)


# --------------------------------------------------------------------------- streaming


async def test_stream_yields_visible_text_and_records_usage(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.stream_queue.append(
        [
            make_response("Hello", thought="planning", with_usage=False),
            make_response(", world", with_usage=False),
            make_response("!", prompt_tokens=20, output_tokens=3, thoughts_tokens=7),
        ]
    )
    with track_usage() as usage:
        tokens = [t async for t in gemini.stream("hi")]
    assert tokens == ["Hello", ", world", "!"]
    totals = usage.totals()
    assert (totals.calls, totals.prompt_tokens, totals.thoughts_tokens) == (1, 20, 7)
    assert fake_genai.models.calls[0]["model"] == "gemini-3.8-flash"


async def test_stream_retries_before_first_token(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.stream_queue.extend([api_error(503), [api_error(429)], [make_response("ok")]])
    tokens = [t async for t in gemini.stream("hi")]
    assert tokens == ["ok"]
    assert len(fake_genai.models.calls_to("generate_content_stream")) == 3


async def test_stream_does_not_retry_after_text_was_sent(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.stream_queue.append([make_response("partial"), api_error(503)])
    received: list[str] = []

    async def consume() -> None:
        async for token in gemini.stream("hi"):
            received.append(token)

    with pytest.raises(GeminiError):
        await consume()
    assert received == ["partial"]
    assert len(fake_genai.models.calls_to("generate_content_stream")) == 1


# --------------------------------------------------------------------------- embeddings


def test_format_embedding_input() -> None:
    assert format_embedding_input("what is bm25", "query") == (
        "task: search result | query: what is bm25"
    )
    assert format_embedding_input("body", "document", title="Guide") == "title: Guide | text: body"
    assert format_embedding_input("body", "document") == "title: none | text: body"
    assert format_embedding_input("body", "document", use_task_type=True) == "body"


def test_l2_normalize() -> None:
    assert np.linalg.norm(l2_normalize([3.0, 4.0])) == pytest.approx(1.0)
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


async def test_embed_documents_uses_prefixes_for_embedding_2(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    vectors = await gemini.embed_documents(["alpha", "beta"], titles=["Doc A", None])
    call = fake_genai.models.calls_to("embed_content")[0]
    assert embedded_texts(call) == ["title: Doc A | text: alpha", "title: none | text: beta"]
    assert call["config"].task_type is None
    assert call["config"].output_dimensionality == 128
    assert len(vectors) == 2
    assert all(np.linalg.norm(v) == pytest.approx(1.0) for v in vectors)


async def test_embed_query_prefix(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    vector = await gemini.embed_query("what is rrf")
    call = fake_genai.models.calls_to("embed_content")[0]
    assert embedded_texts(call) == ["task: search result | query: what is rrf"]
    assert len(vector) == 128


async def test_legacy_model_uses_task_type(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI
) -> None:
    client = GeminiClient(make_settings(embedding_model="gemini-embedding-001"), client=fake_genai)
    await client.embed_documents(["alpha"])
    await client.embed_query("q")
    doc_call, query_call = fake_genai.models.calls_to("embed_content")
    assert embedded_texts(doc_call) == ["alpha"]
    assert doc_call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert query_call["config"].task_type == "RETRIEVAL_QUERY"


async def test_embeddings_are_batched_in_order(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI
) -> None:
    client = GeminiClient(make_settings(embedding_batch_size=2), client=fake_genai)
    texts = [f"t{i}" for i in range(5)]
    vectors = await client.embed_documents(texts)
    calls = fake_genai.models.calls_to("embed_content")
    assert [len(c["contents"]) for c in calls] == [2, 2, 1]
    # One-hot fake: position within batch -> hot index; order must be preserved.
    assert [int(np.argmax(v)) for v in vectors] == [0, 1, 0, 1, 0]


async def test_oversized_batch_is_split(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    default_embed = fake_genai.models.embed_fn

    def limited(model, contents, config):  # type: ignore[no-untyped-def]
        if len(contents) > 2:
            raise api_error(400, "Request payload size exceeds the limit")
        return default_embed(model, contents, config)

    fake_genai.models.embed_fn = limited
    vectors = await gemini.embed_documents([f"t{i}" for i in range(5)])
    assert len(vectors) == 5
    succeeded = [c for c in fake_genai.models.calls_to("embed_content") if len(c["contents"]) <= 2]
    assert [t for c in succeeded for t in embedded_texts(c)] == [
        f"title: none | text: t{i}" for i in range(5)
    ]


async def test_embedding_dimension_mismatch(
    make_settings: Callable[..., Settings], fake_genai: FakeGenAI
) -> None:
    def wrong_size(model, contents, config):  # type: ignore[no-untyped-def]
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[1.0] * 4) for _ in contents]
        )

    fake_genai.models.embed_fn = wrong_size
    client = GeminiClient(make_settings(), client=fake_genai)
    with pytest.raises(GeminiConfigError, match="size mismatch"):
        await client.embed_query("q")


async def test_embed_records_estimated_tokens(gemini: GeminiClient) -> None:
    with track_usage() as usage:
        await gemini.embed_documents(["some text to embed"])
    totals = usage.totals()
    assert totals.embedding_tokens > 0
    assert totals.cost_usd > 0


async def test_embed_empty_input_makes_no_call(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    assert await gemini.embed_documents([]) == []
    assert fake_genai.models.calls == []


# --------------------------------------------------------------------------- model fallback


@pytest.fixture
def with_fallback(make_settings: Callable[..., Settings], fake_genai: FakeGenAI) -> GeminiClient:
    settings = make_settings(
        generation_fallback_model="gemini-3.7-flash", fast_fallback_model="gemini-3.1-flash-lite"
    )
    return GeminiClient(settings, client=fake_genai)


def test_models_for_role(with_fallback: GeminiClient, gemini: GeminiClient) -> None:
    assert with_fallback.models_for("main") == ["gemini-3.8-flash", "gemini-3.7-flash"]
    assert with_fallback.models_for("fast") == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    assert gemini.models_for("main") == ["gemini-3.8-flash"]  # fallback disabled


async def test_overloaded_primary_falls_back(
    with_fallback: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.extend([api_error(503)] * 3 + [make_response("ok")])
    with track_usage() as usage:
        result = await with_fallback.generate("q", role="fast")
    assert result.text == "ok"
    assert result.model == "gemini-3.1-flash-lite"
    models = [c["model"] for c in fake_genai.models.calls_to("generate_content")]
    assert models == ["gemini-3.5-flash-lite"] * 3 + ["gemini-3.1-flash-lite"]
    assert usage.records[0].model == "gemini-3.1-flash-lite"  # cost uses the model that answered


async def test_missing_model_falls_back(with_fallback: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.extend(
        [api_error(404, "model not found"), make_response("ok")]
    )
    assert (await with_fallback.generate("q", role="main")).model == "gemini-3.7-flash"


async def test_bad_request_does_not_fall_back(
    with_fallback: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.append(api_error(400, "invalid argument"))
    with pytest.raises(GeminiError):
        await with_fallback.generate("q")
    assert len(fake_genai.models.calls) == 1


async def test_stream_retries_when_error_follows_empty_chunks(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    # Overload errors can arrive as a stream event after empty/thought-only chunks.
    fake_genai.models.stream_queue.extend(
        [
            [make_response(thought="planning", with_usage=False), api_error(503)],
            [make_response("ok")],
        ]
    )
    tokens = [t async for t in gemini.stream("hi")]
    assert tokens == ["ok"]
    assert len(fake_genai.models.calls_to("generate_content_stream")) == 2


async def test_stream_falls_back_before_first_token(
    with_fallback: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.stream_queue.extend([api_error(503)] * 3 + [[make_response("hello")]])
    tokens = [t async for t in with_fallback.stream("hi")]
    assert tokens == ["hello"]
    models = [c["model"] for c in fake_genai.models.calls_to("generate_content_stream")]
    assert models[-1] == "gemini-3.7-flash"


async def test_mid_stream_failure_restarts_on_fallback(
    with_fallback: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.stream_queue.extend(
        [[make_response("partial"), api_error(503)], [make_response("fresh answer")]]
    )
    restarts: list[int] = []

    async def on_restart() -> None:
        restarts.append(1)

    with track_usage() as usage:
        tokens = [t async for t in with_fallback.stream("hi", on_restart=on_restart)]
    assert tokens == ["partial", "fresh answer"]  # the caller discards "partial" in on_restart
    assert restarts == [1]
    models = [c["model"] for c in fake_genai.models.calls_to("generate_content_stream")]
    assert models == ["gemini-3.8-flash", "gemini-3.7-flash"]
    assert [r.model for r in usage.records] == ["gemini-3.8-flash", "gemini-3.7-flash"]


async def test_mid_stream_restart_happens_once_on_same_model_without_fallback(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.stream_queue.extend(
        [[make_response("a"), api_error(503)], [make_response("b"), api_error(503)]]
    )

    async def on_restart() -> None:
        return None

    async def consume() -> list[str]:
        return [t async for t in gemini.stream("hi", on_restart=on_restart)]

    with pytest.raises(GeminiError):
        await consume()
    models = [c["model"] for c in fake_genai.models.calls_to("generate_content_stream")]
    assert models == ["gemini-3.8-flash", "gemini-3.8-flash"]


async def test_fallback_exhausted_raises_friendly_error(
    with_fallback: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.extend([api_error(429)] * 6)
    with pytest.raises(GeminiRateLimitError):
        await with_fallback.generate("q")
    assert len(fake_genai.models.calls) == 6  # 3 attempts on each model


async def test_query_embedding_memo_reuses_vectors(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    from nexusrag.llm.gemini_client import query_embedding_memo

    fake_genai.models.embed_fn = hash_embedder()
    with query_embedding_memo():
        first = await gemini.embed_query("battery life")
        both = await gemini.embed_queries(["battery life", "charge time", "battery life"])
    assert both[0] == first
    assert both[2] == first
    calls = fake_genai.models.calls_to("embed_content")
    assert [len(c["contents"]) for c in calls] == [1, 1]  # only "charge time" was new
    await gemini.embed_query("battery life")  # outside the block: embedded again
    assert len(fake_genai.models.calls_to("embed_content")) == 3


DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
QUOTA_MESSAGE = "Quota exceeded for metric: free_tier_requests, limit: 20, model: gemini-3.8-flash"


async def test_daily_quota_is_not_retried(gemini: GeminiClient, fake_genai: FakeGenAI) -> None:
    fake_genai.models.generate_queue.append(api_error(429, QUOTA_MESSAGE, quota_id=DAILY))
    with pytest.raises(GeminiQuotaExhaustedError) as info:
        await gemini.generate("q", role="main")
    assert len(fake_genai.models.calls_to("generate_content")) == 1  # retrying can't help
    assert "`gemini-3.8-flash`" in info.value.user_message
    assert "midnight Pacific" in info.value.user_message
    assert isinstance(info.value, GeminiRateLimitError)
    assert not is_retryable(api_error(429, quota_id=DAILY))


async def test_per_minute_quota_is_still_retried(
    gemini: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.extend(
        [api_error(429, "slow down", quota_id=PER_MINUTE), make_response("ok")]
    )
    assert (await gemini.generate("q")).text == "ok"


async def test_daily_quota_falls_back_to_the_other_model(
    with_fallback: GeminiClient, fake_genai: FakeGenAI
) -> None:
    fake_genai.models.generate_queue.extend(
        [api_error(429, QUOTA_MESSAGE, quota_id=DAILY), make_response("ok")]
    )
    result = await with_fallback.generate("q", role="main")
    assert result.text == "ok"
    models = [c["model"] for c in fake_genai.models.calls_to("generate_content")]
    assert models == ["gemini-3.8-flash", "gemini-3.7-flash"]


async def test_resilience_counters(with_fallback: GeminiClient, fake_genai: FakeGenAI) -> None:
    from nexusrag.llm.gemini_client import resilience_stats

    before = resilience_stats()
    fake_genai.models.generate_queue.extend(
        [api_error(503), api_error(503), api_error(503), make_response("ok")]
    )
    await with_fallback.generate("q", role="main")  # 2 retries, then a fallback
    after = resilience_stats()
    assert (after.retries - before.retries, after.fallbacks - before.fallbacks) == (2, 1)


def test_daily_quota_is_recognised_in_streamed_errors() -> None:
    """Streams report the error JSON as an escaped string inside details["message"]."""
    import json

    from google.genai import errors as genai_errors

    from nexusrag.llm.gemini_client import is_daily_quota, translate_error

    payload = {
        "error": {
            "code": 429,
            "message": "Quota exceeded, limit: 500, model: gemini-3.5-flash-lite",
            "details": [{"violations": [{"quotaId": DAILY, "quotaValue": "500"}]}],
        }
    }
    exc = genai_errors.ClientError(429, {"message": json.dumps(payload, indent=2)})
    assert is_daily_quota(exc)
    assert not is_retryable(exc)
    error = translate_error(exc, "answer")
    assert isinstance(error, GeminiQuotaExhaustedError)
    assert "`gemini-3.5-flash-lite`" in error.user_message
