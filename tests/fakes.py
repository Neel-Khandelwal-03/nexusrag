"""In-memory stand-ins for the google-genai SDK client (no network, fully scripted)."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from google.genai import errors as genai_errors
from google.genai import types


def api_error(code: int, message: str = "boom", status: str = "ERROR") -> genai_errors.APIError:
    """Build the SDK exception the real client raises for an HTTP status code."""
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


def make_response(
    text: str = "",
    *,
    thought: str | None = None,
    prompt_tokens: int | None = 10,
    output_tokens: int | None = 5,
    thoughts_tokens: int | None = 0,
    with_usage: bool = True,
    block_reason: str | None = None,
) -> types.GenerateContentResponse:
    """A GenerateContentResponse with optional thought part, usage and block reason."""
    parts: list[types.Part] = []
    if thought:
        parts.append(types.Part(text=thought, thought=True))
    if text:
        parts.append(types.Part(text=text))
    usage = (
        types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thoughts_tokens,
        )
        if with_usage
        else None
    )
    feedback = (
        types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason(block_reason))
        if block_reason
        else None
    )
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
        if parts
        else [],
        usage_metadata=usage,
        prompt_feedback=feedback,
    )


EmbedFn = Callable[[str, list[types.Content], types.EmbedContentConfig], types.EmbedContentResponse]


def one_hot_embedder(scale: float = 3.0) -> EmbedFn:
    """Embed the i-th text of each request as a (deliberately unnormalised) one-hot vector."""

    def _embed(
        model: str, contents: list[types.Content], config: types.EmbedContentConfig
    ) -> types.EmbedContentResponse:
        dim = config.output_dimensionality or 8
        embeddings = []
        for i, _ in enumerate(contents):
            values = [0.0] * dim
            values[i % dim] = scale
            embeddings.append(types.ContentEmbedding(values=values))
        return types.EmbedContentResponse(embeddings=embeddings)

    return _embed


def keyword_embedder() -> EmbedFn:
    """Bag-of-words vectors (hashing trick): texts sharing words get high cosine similarity.

    Lets retrieval tests assert *which* chunk ranks first without a real model.
    """
    import re

    ignore = {"task", "search", "result", "query", "title", "none", "text", "the", "a", "is", "of"}

    def _embed(
        model: str, contents: list[types.Content], config: types.EmbedContentConfig
    ) -> types.EmbedContentResponse:
        dim = config.output_dimensionality or 8
        embeddings = []
        for content in contents:
            values = [0.0] * dim
            text = (content.parts[0].text if content.parts else "") or ""
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word not in ignore:
                    digest = hashlib.sha256(word.encode()).digest()
                    values[int.from_bytes(digest[:4], "little") % dim] += 1.0
            if not any(values):
                values[0] = 1.0
            embeddings.append(types.ContentEmbedding(values=values))
        return types.EmbedContentResponse(embeddings=embeddings)

    return _embed


def hash_embedder() -> EmbedFn:
    """Deterministic pseudo-random vector per text: same text -> same vector, others ~orthogonal."""

    def _embed(
        model: str, contents: list[types.Content], config: types.EmbedContentConfig
    ) -> types.EmbedContentResponse:
        dim = config.output_dimensionality or 8
        embeddings = []
        for content in contents:
            text = content.parts[0].text if content.parts else ""
            seed = int.from_bytes(hashlib.sha256((text or "").encode()).digest()[:8], "little")
            values = np.random.default_rng(seed).standard_normal(dim).tolist()
            embeddings.append(types.ContentEmbedding(values=values))
        return types.EmbedContentResponse(embeddings=embeddings)

    return _embed


class FakeModels:
    """Scripted replacement for ``client.aio.models``."""

    def __init__(self) -> None:
        self.generate_queue: deque[Any] = deque()
        self.stream_queue: deque[Any] = deque()
        self.embed_fn: EmbedFn = one_hot_embedder()
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        self.calls.append(
            {"method": "generate_content", "model": model, "contents": contents, "config": config}
        )
        item = self.generate_queue.popleft()
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[no-any-return]

    async def generate_content_stream(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> AsyncIterator[types.GenerateContentResponse]:
        self.calls.append(
            {
                "method": "generate_content_stream",
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        item = self.stream_queue.popleft()
        if isinstance(item, BaseException):
            raise item
        chunks: Sequence[Any] = item

        async def _iterate() -> AsyncIterator[types.GenerateContentResponse]:
            for chunk in chunks:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return _iterate()

    async def embed_content(
        self, *, model: str, contents: list[types.Content], config: types.EmbedContentConfig
    ) -> types.EmbedContentResponse:
        self.calls.append(
            {"method": "embed_content", "model": model, "contents": contents, "config": config}
        )
        return self.embed_fn(model, contents, config)

    def calls_to(self, method: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method]


class FakeGenAI:
    """Mimics ``google.genai.Client`` closely enough for :class:`GeminiClient`."""

    def __init__(self) -> None:
        self.models = FakeModels()
        self.aio = SimpleNamespace(models=self.models)


def embedded_texts(call: dict[str, Any]) -> list[str]:
    """Texts sent in an ``embed_content`` call."""
    return [content.parts[0].text for content in call["contents"]]
