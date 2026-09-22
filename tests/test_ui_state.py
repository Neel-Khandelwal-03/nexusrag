"""Settings-panel state, rate limiting, history restoration and authentication (no Chainlit)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import SecretStr, ValidationError

from nexusrag.config import Settings
from nexusrag.models import ChatTurn
from nexusrag.ui.auth import (
    AUTH_SECRET_ENV,
    check_credentials,
    ensure_auth_secret,
    password_configured,
)
from nexusrag.ui.state import (
    CLOSEST_MARKER,
    SOURCES_MARKER,
    W_CACHE,
    W_COLLECTION,
    W_DOCUMENTS,
    W_RERANK,
    W_STYLE,
    W_TOP_K,
    RateLimiter,
    UISettings,
    history_from_thread,
    strip_footers,
    uploaded_doc_ids,
)

# --------------------------------------------------------------------------- UISettings


def test_defaults_mirror_settings(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(top_k=7, enable_hyde=True, default_collection="kb")
    ui = UISettings.defaults(settings)
    assert (ui.collection, ui.top_k, ui.hyde, ui.rerank) == ("kb", 7, True, False)
    assert ui.style == "detailed"
    assert ui.documents == []


def test_updated_coerces_widget_values(settings: Settings) -> None:
    ui = UISettings.defaults(settings)
    new = ui.updated({W_TOP_K: 7.0, W_RERANK: True, W_STYLE: "concise", W_DOCUMENTS: ["a.pdf", ""]})
    assert (new.top_k, new.rerank, new.style, new.documents) == (7, True, "concise", ["a.pdf"])
    assert ui.top_k == settings.top_k  # the original is unchanged


def test_cache_switch(make_settings: Callable[..., Settings]) -> None:
    ui = UISettings.defaults(make_settings(enable_semantic_cache=True))
    assert ui.cache is True
    assert ui.updated({W_CACHE: False}).cache is False


def test_updated_ignores_missing_values_and_clears_empty_filter(settings: Settings) -> None:
    ui = UISettings.defaults(settings).updated({W_DOCUMENTS: ["a.pdf"]})
    assert ui.updated({W_COLLECTION: None}).collection == settings.default_collection
    assert ui.updated({}).documents == ["a.pdf"]
    assert ui.updated({W_DOCUMENTS: None}).documents == []  # an emptied multiselect


def test_updated_rejects_invalid_values(settings: Settings) -> None:
    ui = UISettings.defaults(settings)
    with pytest.raises(ValidationError):
        ui.updated({W_STYLE: "rambling"})
    with pytest.raises(ValidationError):
        ui.updated({W_TOP_K: 99})


def test_retrieval_options_and_filters(settings: Settings) -> None:
    ui = UISettings.defaults(settings).updated({W_TOP_K: 3, W_DOCUMENTS: ["a.pdf", "gone.md"]})
    options = ui.retrieval_options(settings)
    assert options.top_k == 3
    assert options.rerank is False
    filters = ui.filters({"a.pdf": "doc-a", "b.md": "doc-b"})
    assert filters is not None
    assert filters.doc_ids == ["doc-a"]
    assert UISettings.defaults(settings).filters({"a.pdf": "doc-a"}) is None
    assert ui.updated({W_DOCUMENTS: ["gone.md"]}).filters({"a.pdf": "doc-a"}) is None


# --------------------------------------------------------------------------- RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_sliding_window() -> None:
    clock = Clock()
    limiter = RateLimiter(limit=3, window_s=60, clock=clock)
    assert all(limiter.allow("u") for _ in range(3))
    assert not limiter.allow("u")
    assert limiter.allow("other")  # limits are per key
    assert limiter.retry_after("u") == pytest.approx(60)
    clock.now += 30
    assert limiter.retry_after("u") == pytest.approx(30)
    clock.now += 30.5
    assert limiter.retry_after("u") == 0
    assert limiter.allow("u")


def test_rate_limiter_cost_is_all_or_nothing() -> None:
    limiter = RateLimiter(limit=5, window_s=3600, clock=Clock())
    assert limiter.allow("u", cost=4)
    assert not limiter.allow("u", cost=2)  # refused without recording anything
    assert limiter.allow("u", cost=1)
    assert not limiter.allow("u")


def test_uploaded_doc_ids() -> None:
    from nexusrag.ingestion.pipeline import IngestResult

    results = [
        IngestResult("a.pdf", "indexed", doc_id="a"),
        IngestResult("b.md", "skipped", doc_id="b"),  # already indexed: still "this file"
        IngestResult("c.txt", "failed", error="empty"),
        IngestResult("d.md", "updated", doc_id="d"),
    ]
    assert uploaded_doc_ids(results) == ["a", "b", "d"]


# --------------------------------------------------------------------------- history


def test_strip_footers() -> None:
    answer = "Charging takes 75 minutes [1]."
    assert strip_footers(f"{answer}{SOURCES_MARKER}\n- [1] spec.pdf") == answer
    assert strip_footers(f"Not found.{CLOSEST_MARKER}\n- match 1") == "Not found."
    assert strip_footers(answer) == answer


def test_history_from_thread_rebuilds_turns_in_order() -> None:
    steps = [
        {"type": "assistant_message", "output": "Welcome!", "createdAt": "2026-01-01T00:00:00"},
        {"type": "assistant_message", "output": "75 min [1].\n\n**Sources**\n- [1] x",
         "createdAt": "2026-01-01T00:00:03"},
        {"type": "user_message", "output": "How long to charge?",
         "createdAt": "2026-01-01T00:00:01"},
        {"type": "tool", "output": "Retrieved 3 passages", "createdAt": "2026-01-01T00:00:02"},
        {"type": "assistant_message", "output": "nested", "parentId": "s1",
         "createdAt": "2026-01-01T00:00:04"},
        {"type": "user_message", "output": "  ", "createdAt": "2026-01-01T00:00:05"},
    ]  # fmt: skip
    turns = history_from_thread(steps, limit=6)
    assert turns == [
        ChatTurn(role="assistant", content="Welcome!"),
        ChatTurn(role="user", content="How long to charge?"),
        ChatTurn(role="assistant", content="75 min [1]."),
    ]
    assert history_from_thread(steps, limit=2) == turns[-2:]
    assert history_from_thread(steps, limit=0) == []


# --------------------------------------------------------------------------- auth


def test_credentials_with_password(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(auth_username="neel", auth_password=SecretStr("s3cret"))
    assert password_configured(settings)
    assert check_credentials(settings, "neel", "s3cret")
    assert check_credentials(settings, " neel ", "s3cret")  # stray whitespace in the name
    assert not check_credentials(settings, "neel", "wrong")
    assert not check_credentials(settings, "admin", "s3cret")
    assert not check_credentials(settings, "neel", "")


def test_no_password_allows_local_use_only(make_settings: Callable[..., Settings]) -> None:
    local = make_settings(environment="development")
    assert not password_configured(local)
    assert check_credentials(local, "admin", "anything")
    assert not check_credentials(local, "someone-else", "anything")
    for environment in ("staging", "production"):
        deployed = make_settings(environment=environment, auth_password=SecretStr(""))
        assert not check_credentials(deployed, "admin", "")


def test_auth_secret(make_settings: Callable[..., Settings]) -> None:
    environ = {AUTH_SECRET_ENV: "already-set"}
    assert ensure_auth_secret(make_settings(environment="production"), environ) is None
    assert environ[AUTH_SECRET_ENV] == "already-set"

    local: dict[str, str] = {}
    warning = ensure_auth_secret(make_settings(environment="development"), local)
    assert warning is not None
    assert "temporary" in warning
    assert len(local[AUTH_SECRET_ENV]) >= 32
    assert local[AUTH_SECRET_ENV] not in warning  # the secret itself is never logged

    with pytest.raises(RuntimeError, match=AUTH_SECRET_ENV):
        ensure_auth_secret(make_settings(environment="staging"), {})
