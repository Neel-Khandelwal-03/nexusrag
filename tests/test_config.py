from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from nexusrag.config import DEFAULT_PRICES, Settings


def test_defaults_come_from_settings(make_settings: Callable[..., Settings]) -> None:
    s = make_settings()
    assert s.generation_model == "gemini-3.8-flash"
    assert s.fast_model == "gemini-3.5-flash-lite"
    assert s.embedding_model == "gemini-embedding-2"
    assert s.generation_temperature is None  # Gemini 3: leave temperature at the model default
    assert s.llm_prices == DEFAULT_PRICES


def test_env_overrides_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATION_MODEL", "gemini-custom-flash")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.generation_model == "gemini-custom-flash"
    assert s.embedding_dim == 1536


def test_google_api_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    assert Settings(_env_file=None).require_gemini_key() == "abc123"  # type: ignore[call-arg]


def test_missing_key_gives_actionable_error() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        s.require_gemini_key()


def test_secrets_never_rendered(make_settings: Callable[..., Settings]) -> None:
    s = make_settings(gemini_api_key="AIzaSuperSecretValue", auth_password="hunter2")
    rendered = repr(s) + str(s) + s.model_dump_json()
    assert "AIzaSuperSecretValue" not in rendered
    assert "hunter2" not in rendered


@pytest.mark.parametrize("raw", ["", "none", "None", "default", "null"])
def test_none_strings_mean_model_default(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("FAST_THINKING_LEVEL", raw)
    monkeypatch.setenv("GENERATION_TEMPERATURE", raw)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.fast_thinking_level is None
    assert s.generation_temperature is None


def test_thinking_level_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATION_THINKING_LEVEL", "HIGH")
    assert Settings(_env_file=None).generation_thinking_level == "high"  # type: ignore[call-arg]


def test_empty_env_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "")
    assert Settings(_env_file=None).top_k == 5  # type: ignore[call-arg]


@pytest.mark.parametrize("dim", [64, 4096])
def test_embedding_dim_bounds(make_settings: Callable[..., Settings], dim: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(embedding_dim=dim)


def test_overlap_must_be_smaller_than_child(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValidationError, match="CHILD_CHUNK_OVERLAP"):
        make_settings(child_chunk_tokens=100, child_chunk_overlap=100)


def test_child_must_be_smaller_than_parent(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValidationError, match="PARENT_CHUNK_TOKENS"):
        make_settings(child_chunk_tokens=800, parent_chunk_tokens=600)


def test_prices_override_from_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PRICES", '{"my-model": {"input_per_m": 1, "output_per_m": 2}}')
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_prices["my-model"].output_per_m == 2


def test_derived_paths(make_settings: Callable[..., Settings]) -> None:
    s = make_settings(storage_dir=Path("/tmp/nx"))
    assert s.chroma_dir == Path("/tmp/nx/chroma")
    assert s.sqlite_path == Path("/tmp/nx/nexusrag.db")
    assert s.bm25_dir == Path("/tmp/nx/bm25")
    assert s.chat_db_path == Path("/tmp/nx/chainlit.db")
    assert not s.is_deployed
    assert make_settings(environment="production").is_deployed
