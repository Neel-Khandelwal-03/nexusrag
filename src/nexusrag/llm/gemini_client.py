"""Typed async wrapper around the official ``google-genai`` SDK.

Responsibilities:

* pick the model per *role*: ``main`` for user-facing answers, ``fast`` for cheap
  rewriting, routing and grading calls. Both names come from settings;
* retry transient failures (429, 5xx, timeouts) with exponential backoff and full jitter;
* structured JSON-schema output, parsed into Pydantic models;
* token streaming for the answer generator;
* batched, L2-normalised embeddings with the right query/document formatting;
* token and approximate cost accounting (plus a structured log line) for every call;
* translating SDK errors into :class:`GeminiError` with a UI-safe ``user_message``.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, TypeAlias, TypeVar

import httpx
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from nexusrag.config import Settings, get_settings
from nexusrag.llm.usage import UsageRecord, estimate_cost, record_usage
from nexusrag.log import get_logger
from nexusrag.utils.tokens import count_tokens

log = get_logger(__name__)

ModelRole = Literal["main", "fast"]
EmbedKind = Literal["query", "document"]
Prompt: TypeAlias = str | Sequence[types.Content]
T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# gemini-embedding-2 has no `task_type` parameter. Asymmetric retrieval is signalled by
# formatting the text itself (https://ai.google.dev/gemini-api/docs/embeddings).
QUERY_TEMPLATE = "task: search result | query: {text}"
DOCUMENT_TEMPLATE = "title: {title} | text: {text}"
# Legacy models (gemini-embedding-001) use task types instead of prefixes.
TASK_TYPES: dict[EmbedKind, str] = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


# --------------------------------------------------------------------------- errors


class GeminiError(RuntimeError):
    """Base error. ``user_message`` is safe to show in the UI; details stay in logs."""

    default_message = "The AI service failed to respond. Please try again."

    def __init__(self, detail: str, user_message: str | None = None) -> None:
        super().__init__(detail)
        self.user_message = user_message or self.default_message


class GeminiConfigError(GeminiError):
    """Missing/invalid API key or unknown model name."""

    default_message = "The AI service is not configured correctly (API key or model name)."


class GeminiRateLimitError(GeminiError):
    """Quota exhausted even after retries."""

    default_message = "The AI service is rate-limiting requests. Please wait a moment and retry."


class GeminiBlockedError(GeminiError):
    """The provider's safety filters blocked the prompt."""

    default_message = "The request was blocked by the AI provider's safety filters."


class StructuredOutputError(GeminiError):
    """The model's JSON didn't match the requested schema."""

    default_message = "The AI service returned an unexpected response format."


def is_retryable(exc: BaseException) -> bool:
    """True for transient failures worth retrying (rate limits, server errors, timeouts)."""
    if isinstance(exc, genai_errors.APIError):
        return exc.code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError | TimeoutError)


def translate_error(exc: BaseException, stage: str) -> GeminiError:
    """Map SDK/network exceptions to :class:`GeminiError` subclasses."""
    if isinstance(exc, GeminiError):
        return exc
    if isinstance(exc, genai_errors.APIError):
        detail = f"{stage}: Gemini API error {exc.code} {exc.status}: {exc.message}"
        message = (exc.message or "").lower()
        if exc.code == 429:
            return GeminiRateLimitError(detail)
        # An invalid key comes back as 400 INVALID_ARGUMENT, not 401.
        if exc.code in (401, 403) or (exc.code == 400 and "api key" in message):
            return GeminiConfigError(detail)
        if exc.code == 404:
            return GeminiConfigError(
                detail, "A configured Gemini model was not found. Check the model names in config."
            )
        return GeminiError(detail)
    if is_retryable(exc):
        return GeminiError(
            f"{stage}: network error {type(exc).__name__}",
            "The AI service timed out. Please try again.",
        )
    return GeminiError(f"{stage}: unexpected {type(exc).__name__}: {exc}")


@contextmanager
def _translated(stage: str) -> Iterator[None]:
    try:
        yield
    except GeminiError:
        raise
    except Exception as exc:
        err = translate_error(exc, stage)
        log.error("llm.error", stage=stage, error_type=type(exc).__name__, detail=str(err))
        raise err from exc


# --------------------------------------------------------------------------- pure helpers


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Scale ``vector`` to unit length so dot product equals cosine similarity."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return [float(x) for x in (arr / norm if norm > 0 else arr)]


def format_embedding_input(
    text: str, kind: EmbedKind, *, title: str | None = None, use_task_type: bool = False
) -> str:
    """Format text for embedding: prefixes for gemini-embedding-2, raw text for task_type models."""
    if use_task_type:
        return text
    if kind == "query":
        return QUERY_TEMPLATE.format(text=text)
    return DOCUMENT_TEMPLATE.format(title=title or "none", text=text)


def parse_structured(text: str, schema: type[T]) -> T:
    """Validate model JSON against ``schema``, tolerating a surrounding Markdown code fence."""
    match = _FENCE_RE.match(text)
    payload = match.group(1) if match else text.strip()
    try:
        return schema.model_validate_json(payload)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"Could not parse {schema.__name__}: {exc.error_count()} validation error(s)"
        ) from exc


def response_text(response: types.GenerateContentResponse) -> str:
    """Concatenate the visible (non-thought) text parts of the first candidate."""
    if not response.candidates or response.candidates[0].content is None:
        return ""
    parts = response.candidates[0].content.parts or []
    return "".join(part.text for part in parts if part.text and not part.thought)


def _raise_if_blocked(response: types.GenerateContentResponse, stage: str) -> None:
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        raise GeminiBlockedError(f"{stage}: prompt blocked ({feedback.block_reason})")


def _as_contents(prompt: Prompt) -> Any:
    return prompt if isinstance(prompt, str) else list(prompt)


# --------------------------------------------------------------------------- client


@dataclass(frozen=True)
class GenerationResult:
    """Text output plus the usage record of the call that produced it."""

    text: str
    model: str
    usage: UsageRecord


class GeminiClient:
    """Async Gemini access used by every pipeline stage.

    Pass ``client`` to inject a fake SDK client in tests. Otherwise a real
    ``google.genai.Client`` is built from settings.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        if client is None:
            try:
                api_key = settings.require_gemini_key()
            except ValueError as exc:
                raise GeminiConfigError(str(exc)) from exc
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000)),
            )
        self._client = client
        self._embed_slots = asyncio.Semaphore(settings.embedding_concurrency)

    @property
    def settings(self) -> Settings:
        return self._settings

    def model_for(self, role: ModelRole) -> str:
        """Model ID for a role, from settings."""
        return self._settings.generation_model if role == "main" else self._settings.fast_model

    def build_config(
        self,
        role: ModelRole,
        *,
        system_instruction: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> types.GenerateContentConfig:
        """Generation config for a role, honouring thinking/temperature settings."""
        s = self._settings
        level = s.generation_thinking_level if role == "main" else s.fast_thinking_level
        temperature = s.generation_temperature if role == "main" else s.fast_temperature
        kwargs: dict[str, Any] = {"max_output_tokens": max_output_tokens or s.max_output_tokens}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            kwargs["temperature"] = temperature
        if level is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(level.upper())
            )
        if json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = json_schema
        return types.GenerateContentConfig(**kwargs)

    # ------------------------------------------------------------------ generation

    async def generate(
        self,
        prompt: Prompt,
        *,
        role: ModelRole = "fast",
        system_instruction: str | None = None,
        stage: str = "generate",
        max_output_tokens: int | None = None,
    ) -> GenerationResult:
        """Single-shot text generation."""
        return await self._generate(
            prompt,
            role=role,
            system_instruction=system_instruction,
            stage=stage,
            max_output_tokens=max_output_tokens,
            json_schema=None,
        )

    async def generate_structured(
        self,
        prompt: Prompt,
        schema: type[T],
        *,
        role: ModelRole = "fast",
        system_instruction: str | None = None,
        stage: str = "structured",
        max_output_tokens: int | None = None,
    ) -> T:
        """Generate JSON constrained to ``schema`` and return the parsed model."""
        result = await self._generate(
            prompt,
            role=role,
            system_instruction=system_instruction,
            stage=stage,
            max_output_tokens=max_output_tokens,
            json_schema=schema.model_json_schema(),
        )
        try:
            return parse_structured(result.text, schema)
        except StructuredOutputError as exc:
            log.warning("llm.structured_parse_failed", stage=stage, detail=str(exc))
            raise

    async def stream(
        self,
        prompt: Prompt,
        *,
        role: ModelRole = "main",
        system_instruction: str | None = None,
        stage: str = "answer",
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream visible text deltas. Usage is recorded when the stream ends or is closed."""
        model = self.model_for(role)
        config = self.build_config(
            role, system_instruction=system_instruction, max_output_tokens=max_output_tokens
        )
        contents = _as_contents(prompt)
        started = time.perf_counter()

        async def open_stream() -> tuple[AsyncIterator[types.GenerateContentResponse], Any]:
            iterator = await self._client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
            # Pull the first chunk inside the retry scope: quota and connection errors surface
            # here, and retrying is only safe before any text has reached the user.
            first = await anext(iterator, None)
            return iterator, first

        usage_md: types.GenerateContentResponseUsageMetadata | None = None
        opened = False
        try:
            with _translated(stage):
                iterator, chunk = await self._with_retry(open_stream, stage=stage, model=model)
                opened = True
                if chunk is not None:
                    _raise_if_blocked(chunk, stage)
                while chunk is not None:
                    if chunk.usage_metadata is not None:
                        usage_md = chunk.usage_metadata
                    text = response_text(chunk)
                    if text:
                        yield text
                    chunk = await anext(iterator, None)
        finally:
            # Also runs when the consumer stops early, so partial streams are still accounted.
            if opened:
                self._record_generation(stage, model, usage_md, started)

    async def _generate(
        self,
        prompt: Prompt,
        *,
        role: ModelRole,
        system_instruction: str | None,
        stage: str,
        max_output_tokens: int | None,
        json_schema: dict[str, Any] | None,
    ) -> GenerationResult:
        model = self.model_for(role)
        config = self.build_config(
            role,
            system_instruction=system_instruction,
            json_schema=json_schema,
            max_output_tokens=max_output_tokens,
        )
        contents = _as_contents(prompt)
        started = time.perf_counter()
        with _translated(stage):
            response = await self._with_retry(
                lambda: self._client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                ),
                stage=stage,
                model=model,
            )
            _raise_if_blocked(response, stage)
        usage = self._record_generation(stage, model, response.usage_metadata, started)
        return GenerationResult(text=response_text(response), model=model, usage=usage)

    # ------------------------------------------------------------------ embeddings

    def uses_task_type(self) -> bool:
        """Whether the configured embedding model takes ``task_type`` (vs text prefixes)."""
        mode = self._settings.embedding_input_mode
        if mode == "auto":
            model = self._settings.embedding_model
            return "embedding-001" in model or model.startswith("text-embedding")
        return mode == "task_type"

    async def embed_documents(
        self, texts: Sequence[str], titles: Sequence[str | None] | None = None
    ) -> list[list[float]]:
        """Embed document chunks (optionally with their document titles)."""
        if titles is not None and len(titles) != len(texts):
            raise ValueError("titles must be the same length as texts")
        use_task_type = self.uses_task_type()
        inputs = [
            format_embedding_input(
                text,
                "document",
                title=titles[i] if titles is not None else None,
                use_task_type=use_task_type,
            )
            for i, text in enumerate(texts)
        ]
        task_type = TASK_TYPES["document"] if use_task_type else None
        return await self._embed(inputs, task_type=task_type, stage="embed.documents")

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed search queries (the original question plus any rewrites)."""
        use_task_type = self.uses_task_type()
        inputs = [format_embedding_input(t, "query", use_task_type=use_task_type) for t in texts]
        task_type = TASK_TYPES["query"] if use_task_type else None
        return await self._embed(inputs, task_type=task_type, stage="embed.queries")

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return (await self.embed_queries([text]))[0]

    async def _embed(
        self, inputs: list[str], *, task_type: str | None, stage: str
    ) -> list[list[float]]:
        if not inputs:
            return []
        size = self._settings.embedding_batch_size
        batches = [inputs[i : i + size] for i in range(0, len(inputs), size)]
        started = time.perf_counter()
        with _translated(stage):
            results = await asyncio.gather(
                *(self._embed_batch(batch, task_type, stage) for batch in batches)
            )
        # The Gemini API doesn't report embedding token counts, so estimate them for costing.
        tokens = sum(count_tokens(text) for text in inputs)
        self._record_embedding(stage, tokens, len(inputs), started)
        return [vector for batch in results for vector in batch]

    async def _embed_batch(
        self, batch: list[str], task_type: str | None, stage: str
    ) -> list[list[float]]:
        async with self._embed_slots:
            try:
                return await self._with_retry(
                    lambda: self._embed_call(batch, task_type),
                    stage=stage,
                    model=self._settings.embedding_model,
                )
            except genai_errors.ClientError as exc:
                oversized = exc.code == 400 and "api key" not in (exc.message or "").lower()
                if not oversized or len(batch) == 1:
                    raise
        # The request exceeded a size limit: split it and embed each half. This runs after
        # the semaphore is released so the halves can't deadlock waiting on our slot.
        log.warning("llm.embed_batch_split", stage=stage, batch_size=len(batch))
        mid = len(batch) // 2
        left = await self._embed_batch(batch[:mid], task_type, stage)
        right = await self._embed_batch(batch[mid:], task_type, stage)
        return left + right

    async def _embed_call(self, batch: list[str], task_type: str | None) -> list[list[float]]:
        # One Content per text gives one embedding per text. Passing a list of plain strings to
        # gemini-embedding-2 would aggregate them into a single vector.
        contents = [types.Content(parts=[types.Part(text=text)]) for text in batch]
        config = types.EmbedContentConfig(
            output_dimensionality=self._settings.embedding_dim, task_type=task_type
        )
        response = await self._client.aio.models.embed_content(
            model=self._settings.embedding_model, contents=contents, config=config
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise GeminiError(f"Expected {len(batch)} embeddings, got {len(embeddings)}")
        vectors = [e.values or [] for e in embeddings]
        if any(len(v) != self._settings.embedding_dim for v in vectors):
            raise GeminiConfigError(
                f"Embedding size mismatch: expected {self._settings.embedding_dim}, "
                f"got {sorted({len(v) for v in vectors})}"
            )
        # gemini-embedding-2 normalises truncated vectors itself; legacy models don't. Normalising
        # again is cheap and idempotent, and guarantees cosine == dot product downstream.
        return [l2_normalize(v) for v in vectors]

    # ------------------------------------------------------------------ plumbing

    async def _with_retry(self, call: Callable[[], Awaitable[R]], *, stage: str, model: str) -> R:
        s = self._settings

        def before_sleep(state: RetryCallState) -> None:
            exc = state.outcome.exception() if state.outcome else None
            log.warning(
                "llm.retry",
                stage=stage,
                model=model,
                attempt=state.attempt_number,
                error_type=type(exc).__name__ if exc else None,
                status=getattr(exc, "code", None),
            )

        retrying = AsyncRetrying(
            stop=stop_after_attempt(s.llm_max_attempts),
            # Full jitter: wait a random time in [0, min(max, initial * 2^attempt)].
            wait=wait_random_exponential(
                multiplier=s.llm_retry_initial_wait_s, max=s.llm_retry_max_wait_s
            ),
            retry=retry_if_exception(is_retryable),
            before_sleep=before_sleep,
            reraise=True,
        )

        # tenacity only awaits *coroutine functions*; a lambda returning a coroutine would be
        # treated as sync and its coroutine returned un-awaited (and never retried).
        async def attempt() -> R:
            return await call()

        result: R = await retrying(attempt)
        return result

    def _record_generation(
        self,
        stage: str,
        model: str,
        usage_md: types.GenerateContentResponseUsageMetadata | None,
        started: float,
    ) -> UsageRecord:
        prompt_tokens = (usage_md.prompt_token_count or 0) if usage_md else 0
        output_tokens = (usage_md.candidates_token_count or 0) if usage_md else 0
        thoughts_tokens = (usage_md.thoughts_token_count or 0) if usage_md else 0
        # Thinking tokens are billed at the output rate.
        cost = estimate_cost(
            self._settings.llm_prices,
            model,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens + thoughts_tokens,
        )
        record = UsageRecord(
            stage=stage,
            model=model,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thoughts_tokens=thoughts_tokens,
            cost_usd=cost,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        record_usage(record)
        log.info(
            "llm.call",
            stage=stage,
            model=model,
            latency_ms=round(record.latency_ms, 1),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thoughts_tokens=thoughts_tokens,
            cost_usd=round(cost, 6),
        )
        return record

    def _record_embedding(self, stage: str, tokens: int, count: int, started: float) -> None:
        model = self._settings.embedding_model
        cost = estimate_cost(self._settings.llm_prices, model, input_tokens=tokens, output_tokens=0)
        record = UsageRecord(
            stage=stage,
            model=model,
            embedding_tokens=tokens,
            cost_usd=cost,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        record_usage(record)
        log.info(
            "llm.embed",
            stage=stage,
            model=model,
            inputs=count,
            latency_ms=round(record.latency_ms, 1),
            embedding_tokens_est=tokens,
            cost_usd=round(cost, 6),
        )


@lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient:
    """Process-wide client built from :func:`~nexusrag.config.get_settings`."""
    return GeminiClient(get_settings())
