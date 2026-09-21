"""Shared fixtures. No test may reach the real Gemini API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import SecretStr

from nexusrag.config import Settings, get_settings
from nexusrag.llm.gemini_client import GeminiClient
from tests.fakes import FakeGenAI

# Environment variables that would leak a developer's local config into tests.
_ISOLATED_ENV = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GENERATION_MODEL",
    "FAST_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "LOG_CONTENT",
    "STORAGE_DIR",
    "DATA_DIR",
    "DEFAULT_COLLECTION",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block real SDK clients and strip developer env vars for every test."""

    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Tests must not construct a real google.genai.Client")

    monkeypatch.setattr("nexusrag.llm.gemini_client.genai.Client", _forbidden)
    for name in _ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Build Settings that ignore any local .env file, with fast retries."""

    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "environment": "test",
            "gemini_api_key": SecretStr("test-key"),
            "llm_retry_initial_wait_s": 0,
            "llm_retry_max_wait_s": 0,
            "llm_max_attempts": 3,
            "embedding_dim": 128,
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)  # type: ignore[call-arg]

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def fake_genai() -> FakeGenAI:
    return FakeGenAI()


@pytest.fixture
def gemini(settings: Settings, fake_genai: FakeGenAI) -> GeminiClient:
    return GeminiClient(settings, client=fake_genai)
