from __future__ import annotations

import time

import structlog

from nexusrag.log import RedactProcessor, request_context, scrub, timed


def _run(processor: RedactProcessor, **event: object) -> dict[str, object]:
    return dict(processor(None, "info", dict(event)))


def test_scrub_masks_credentials_in_free_text() -> None:
    line = (
        "calling with key=AIzaSyA1234567890abcdefghijklmnopqrstu and hf_abcdefghijklmnopqrstuvwxyz"
    )
    cleaned = scrub(line)
    assert "AIza" not in cleaned
    assert "hf_abc" not in cleaned
    assert cleaned.count("***") == 2


def test_secret_keys_are_masked() -> None:
    out = _run(RedactProcessor(), event="x", api_key="k", hf_token="t", auth_password="p")
    assert out["api_key"] == out["hf_token"] == out["auth_password"] == "***"


def test_token_counts_are_not_mistaken_for_secrets() -> None:
    out = _run(RedactProcessor(), event="llm.call", prompt_tokens=10, output_tokens=5)
    assert out["prompt_tokens"] == 10
    assert out["output_tokens"] == 5


def test_content_redacted_by_default() -> None:
    out = _run(RedactProcessor(), event="retrieve", query="secret plans", chunks_found=3)
    assert out["query"] == "<redacted 12 chars>"
    assert out["chunks_found"] == 3


def test_content_kept_when_enabled() -> None:
    out = _run(RedactProcessor(log_content=True), event="retrieve", query="secret plans")
    assert out["query"] == "secret plans"


def test_event_message_is_scrubbed() -> None:
    out = _run(RedactProcessor(), event="failed with AIzaSyA1234567890abcdefghijklmnopqrstu")
    assert "AIza" not in str(out["event"])


def test_request_context_binds_request_id() -> None:
    with request_context(route="doc_qa") as rid:
        bound = structlog.contextvars.get_contextvars()
        assert bound["request_id"] == rid
        assert bound["route"] == "doc_qa"
    assert "request_id" not in structlog.contextvars.get_contextvars()


def test_timed_freezes_after_block() -> None:
    with timed() as t:
        time.sleep(0.01)
    frozen = t.ms
    assert frozen >= 10
    time.sleep(0.01)
    assert t.ms == frozen
