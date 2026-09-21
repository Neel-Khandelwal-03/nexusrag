"""Shared fixtures. No test may reach the real Gemini API."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import structlog
from pydantic import SecretStr

from nexusrag.config import Settings, get_settings
from nexusrag.llm.gemini_client import GeminiClient
from tests.fakes import FakeGenAI

# Importing chainlit loads ./.env into os.environ; point it at a file that doesn't exist.
os.environ["CHAINLIT_ENV_FILE"] = ".env.not-loaded-in-tests"

# Environment variables that would leak a developer's local config into tests.
_ISOLATED_ENV = (
    *(name.upper() for name in Settings.model_fields),
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "CHAINLIT_AUTH_SECRET",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block real SDK clients and model downloads, strip developer env vars, reset logging."""

    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Tests must not construct a real google.genai.Client")

    monkeypatch.setattr("nexusrag.llm.gemini_client.genai.Client", _forbidden)

    def _no_model_download(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Tests must not load the real cross-encoder (inject a fake reranker)")

    monkeypatch.setattr("nexusrag.retrieval.reranker._load_cross_encoder", _no_model_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for name in _ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    # CLI tests call configure_logging(); don't let that configuration leak into later tests.
    structlog.reset_defaults()


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
            # Stages that need extra LLM calls or a local model are opted into per test.
            "enable_multi_query": False,
            "enable_rerank": False,
            # Fallback models are opted into by the tests that exercise them.
            "generation_fallback_model": None,
            "fast_fallback_model": None,
            # The agent's graders are opted into by the tests that exercise them.
            "enable_self_correction": False,
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
