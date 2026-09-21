"""Application settings loaded from environment variables and an optional `.env` file.

Every tunable (model names, chunk sizes, retrieval depths, feature toggles, limits)
lives here so behaviour can change per environment without code edits. Field names
map to upper-case environment variables, e.g. ``generation_model`` -> ``GENERATION_MODEL``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ThinkingLevel = Literal["minimal", "low", "medium", "high"]
Environment = Literal["development", "test", "staging", "production"]


class ModelPrice(BaseModel):
    """USD price per one million tokens for a model."""

    input_per_m: float = Field(ge=0)
    output_per_m: float = Field(ge=0)


# Paid-tier standard text prices (USD per 1M tokens) from
# https://ai.google.dev/gemini-api/docs/pricing, checked 2026-09. These are *data* for
# approximate cost logging only; which model is used always comes from settings.
# Override the whole table with LLM_PRICES='{"model-id": {"input_per_m": .., "output_per_m": ..}}'.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gemini-3.8-flash": ModelPrice(input_per_m=0.75, output_per_m=3.75),
    "gemini-3.7-flash": ModelPrice(input_per_m=0.75, output_per_m=3.75),
    "gemini-3.6-flash": ModelPrice(input_per_m=0.75, output_per_m=3.75),
    "gemini-3.5-flash-lite": ModelPrice(input_per_m=0.30, output_per_m=2.50),
    "gemini-3.1-flash-lite": ModelPrice(input_per_m=0.25, output_per_m=1.50),
    "gemini-2.5-flash": ModelPrice(input_per_m=0.30, output_per_m=2.50),
    "gemini-2.5-flash-lite": ModelPrice(input_per_m=0.10, output_per_m=0.40),
    "gemini-embedding-2": ModelPrice(input_per_m=0.20, output_per_m=0.0),
    "gemini-embedding-001": ModelPrice(input_per_m=0.15, output_per_m=0.0),
}

# Strings accepted in env vars to mean "unset / use the model's own default".
_NONE_STRINGS = {"", "none", "null", "default"}


class Settings(BaseSettings):
    """All runtime configuration for NexusRAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        # utf-8-sig: Windows editors may add a BOM, which would otherwise rename the first key.
        env_file_encoding="utf-8-sig",
        extra="ignore",
        case_sensitive=False,
        # `FOO=` in .env means "use the default", not "empty string".
        env_ignore_empty=True,
    )

    # ------------------------------------------------------------------ runtime
    app_name: str = "NexusRAG"
    environment: Environment = "development"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    # Prompts, queries and document text are redacted from logs unless this is on.
    log_content: bool = False

    # ------------------------------------------------------------------ gemini
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    # Main answer model and a cheaper/faster model for rewriting, routing and grading.
    generation_model: str = "gemini-3.8-flash"
    fast_model: str = "gemini-3.5-flash-lite"
    # Used when the primary model is still overloaded (429/5xx) after retries, or is gone
    # (404). "none" disables fallback for that role.
    generation_fallback_model: str | None = "gemini-3.6-flash"
    fast_fallback_model: str | None = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = Field(default=768, ge=128, le=3072)
    # gemini-embedding-2 marks queries/documents with text prefixes; legacy models use
    # the `task_type` parameter. "auto" picks based on the model name.
    embedding_input_mode: Literal["auto", "prefix", "task_type"] = "auto"
    embedding_batch_size: int = Field(default=64, ge=1, le=250)
    embedding_concurrency: int = Field(default=4, ge=1, le=32)
    # Thinking depth per role; None leaves the model default. gemini-3.8-flash rejects "minimal".
    generation_thinking_level: ThinkingLevel | None = "low"
    fast_thinking_level: ThinkingLevel | None = None
    # Gemini 3 models are tuned for the default temperature (1.0) and Google advises against
    # lowering it, so temperature is only sent when explicitly configured.
    generation_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    fast_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192, ge=256)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    # Attempts per model; kept low for interactive latency because a fallback model exists.
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    llm_retry_initial_wait_s: float = Field(default=1.0, ge=0)
    llm_retry_max_wait_s: float = Field(default=8.0, ge=0)
    llm_prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICES))

    # ------------------------------------------------------------------ storage
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")
    default_collection: str = "default"

    # ------------------------------------------------------------------ chunking
    # Children are what we embed and search; parents are what the LLM reads.
    child_chunk_tokens: int = Field(default=256, ge=32)
    child_chunk_overlap: int = Field(default=40, ge=0)
    parent_chunk_tokens: int = Field(default=1200, ge=128)

    # ------------------------------------------------------------------ retrieval
    top_k: int = Field(default=5, ge=1, le=50)
    dense_k: int = Field(default=20, ge=1, le=200)
    bm25_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    enable_hybrid: bool = True
    enable_multi_query: bool = True
    num_query_variants: int = Field(default=3, ge=1, le=8)
    enable_hyde: bool = False
    enable_rerank: bool = True
    reranker_backend: Literal["cross_encoder", "gemini"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_max_length: int = Field(default=512, ge=64, le=1024)
    # Only the top-N fused candidates are rescored: cross-encoders are accurate but slow.
    rerank_candidates: int = Field(default=20, ge=1, le=200)
    # Calibrated on the sample corpus: weakest real evidence ~0.23, unanswerable questions <= 0.07.
    rerank_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    # Weight of the first-stage (hybrid) ranking when fusing it with the reranker order
    # (RRF). 0 = pure reranker order. See retrieval/reranker.py for the evidence.
    rerank_fusion_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    context_token_budget: int = Field(default=6000, ge=500)
    # Previous chat messages (user + assistant) used to condense follow-up questions.
    history_turns: int = Field(default=6, ge=0, le=50)

    # ------------------------------------------------------------------ agent
    enable_self_correction: bool = True
    max_retrieval_retries: int = Field(default=2, ge=0, le=5)
    # Summaries: documents up to this many tokens are summarised in one call; larger ones
    # use map-reduce over batches of this size.
    summary_stuff_tokens: int = Field(default=12000, ge=1000)
    summary_map_batch_tokens: int = Field(default=6000, ge=500)
    # Comparisons: chunks retrieved per document, and the most documents compared at once.
    compare_top_k_per_doc: int = Field(default=4, ge=1, le=20)
    max_compare_documents: int = Field(default=4, ge=2, le=8)

    # ------------------------------------------------------------------ cache
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = Field(default=0.95, gt=0.0, le=1.0)

    # ------------------------------------------------------------------ UI / security
    # Suggest follow-up questions (one fast-model call) after grounded answers.
    enable_follow_ups: bool = True
    auth_username: str = "admin"
    auth_password: SecretStr | None = None
    max_upload_files: int = Field(default=10, ge=1)
    max_upload_mb: int = Field(default=20, ge=1)
    rate_limit_messages_per_minute: int = Field(default=20, ge=1)
    rate_limit_uploads_per_hour: int = Field(default=30, ge=1)

    # ------------------------------------------------------------------ validators
    @field_validator(
        "generation_fallback_model",
        "fast_fallback_model",
        "generation_thinking_level",
        "fast_thinking_level",
        "generation_temperature",
        "fast_temperature",
        mode="before",
    )
    @classmethod
    def _none_strings(cls, value: object) -> object:
        if isinstance(value, str):
            if value.strip().lower() in _NONE_STRINGS:
                return None
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def _check_chunk_sizes(self) -> Settings:
        if self.child_chunk_overlap >= self.child_chunk_tokens:
            raise ValueError("CHILD_CHUNK_OVERLAP must be smaller than CHILD_CHUNK_TOKENS")
        if self.child_chunk_tokens >= self.parent_chunk_tokens:
            raise ValueError("CHILD_CHUNK_TOKENS must be smaller than PARENT_CHUNK_TOKENS")
        return self

    # ------------------------------------------------------------------ derived paths
    @property
    def chroma_dir(self) -> Path:
        """Persistent ChromaDB directory (child chunk vectors)."""
        return self.storage_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        """SQLite file for the document registry, parent sections and BM25 token lists."""
        return self.storage_dir / "nexusrag.db"

    @property
    def files_dir(self) -> Path:
        """Copies of ingested PDFs, used to preview cited pages in the UI."""
        return self.storage_dir / "files"

    def source_file_path(self, collection: str, doc_id: str) -> Path:
        """Where the preview copy of a document's PDF is kept."""
        return self.files_dir / collection / f"{doc_id}.pdf"

    @property
    def chat_db_path(self) -> Path:
        """SQLite file backing Chainlit's chat history data layer."""
        return self.storage_dir / "chainlit.db"

    @property
    def is_deployed(self) -> bool:
        """True for staging and production, where secrets must be present."""
        return self.environment in ("staging", "production")

    # ------------------------------------------------------------------ helpers
    def require_gemini_key(self) -> str:
        """Return the Gemini API key or raise a clear error if it's missing."""
        if self.gemini_api_key is None or not self.gemini_api_key.get_secret_value().strip():
            raise ValueError(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
                "and add it to your .env file (see .env.example)."
            )
        return self.gemini_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached; call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
