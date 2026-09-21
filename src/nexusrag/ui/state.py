"""Per-session UI state: settings-panel values, rate limiting and history restoration.

Pure Python (no Chainlit imports), so it's unit-tested directly.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from nexusrag.config import Settings
from nexusrag.models import ChatTurn, SearchFilters
from nexusrag.retrieval.retriever import RetrievalOptions

# Widget ids in the chat settings panel.
W_COLLECTION = "collection"
W_NEW_COLLECTION = "new_collection"
W_DOCUMENTS = "documents"
W_TOP_K = "top_k"
W_HYBRID = "hybrid"
W_MULTI_QUERY = "multi_query"
W_HYDE = "hyde"
W_RERANK = "rerank"
W_SELF_CORRECT = "self_correct"
W_STYLE = "style"

#: Marker that separates an answer from the sources footer in stored messages.
SOURCES_MARKER = "\n\n**Sources**"
CLOSEST_MARKER = "\n\n**Closest matches in your documents**"


class UISettings(BaseModel):
    """What the user picked in the settings panel."""

    collection: str
    top_k: int = Field(ge=1, le=20)
    hybrid: bool
    multi_query: bool
    hyde: bool
    rerank: bool
    self_correct: bool
    style: Literal["concise", "detailed"] = "detailed"
    #: Filenames to restrict retrieval to; empty means the whole knowledge base.
    documents: list[str] = Field(default_factory=list)

    @classmethod
    def defaults(cls, settings: Settings) -> UISettings:
        return cls(
            collection=settings.default_collection,
            top_k=settings.top_k,
            hybrid=settings.enable_hybrid,
            multi_query=settings.enable_multi_query,
            hyde=settings.enable_hyde,
            rerank=settings.enable_rerank,
            self_correct=settings.enable_self_correction,
        )

    def updated(self, values: Mapping[str, Any]) -> UISettings:
        """Apply raw widget values (sliders send floats, empty multiselects send None)."""
        data = self.model_dump()
        mapping = {
            W_COLLECTION: "collection",
            W_TOP_K: "top_k",
            W_HYBRID: "hybrid",
            W_MULTI_QUERY: "multi_query",
            W_HYDE: "hyde",
            W_RERANK: "rerank",
            W_SELF_CORRECT: "self_correct",
            W_STYLE: "style",
        }
        for widget, field in mapping.items():
            if values.get(widget) is not None:
                data[field] = values[widget]
        data["top_k"] = round(float(data["top_k"]))
        documents = values.get(W_DOCUMENTS, data["documents"])
        data["documents"] = [d for d in (documents or []) if d]
        return UISettings.model_validate(data)

    def retrieval_options(self, settings: Settings) -> RetrievalOptions:
        return RetrievalOptions.from_settings(
            settings,
            top_k=self.top_k,
            hybrid=self.hybrid,
            multi_query=self.multi_query,
            hyde=self.hyde,
            rerank=self.rerank,
        )

    def filters(self, doc_ids_by_filename: Mapping[str, str]) -> SearchFilters | None:
        """Search filters for the selected documents (unknown names are ignored)."""
        ids = [doc_ids_by_filename[f] for f in self.documents if f in doc_ids_by_filename]
        return SearchFilters(doc_ids=ids) if ids else None


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` units per ``window_s`` seconds per key."""

    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._events: dict[str, deque[float]] = {}

    def _prune(self, key: str) -> deque[float]:
        events = self._events.setdefault(key, deque())
        cutoff = self._clock() - self.window_s
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    def allow(self, key: str, cost: int = 1) -> bool:
        """Record ``cost`` units for ``key`` if they fit in the window; otherwise refuse."""
        events = self._prune(key)
        if len(events) + cost > self.limit:
            return False
        now = self._clock()
        events.extend([now] * cost)
        return True

    def retry_after(self, key: str) -> float:
        """Seconds until at least one more unit fits."""
        events = self._prune(key)
        if len(events) < self.limit:
            return 0.0
        return max(0.0, events[0] + self.window_s - self._clock())


def strip_footers(text: str) -> str:
    """Answer text without the sources / closest-matches footers the UI appends."""
    for marker in (SOURCES_MARKER, CLOSEST_MARKER):
        text = text.split(marker, 1)[0]
    return text.strip()


def history_from_thread(steps: Sequence[Mapping[str, Any]], limit: int) -> list[ChatTurn]:
    """Rebuild chat history from a persisted Chainlit thread (for resumed chats)."""
    turns: list[ChatTurn] = []
    ordered = sorted(steps, key=lambda s: str(s.get("createdAt") or ""))
    for step in ordered:
        kind, text = step.get("type"), str(step.get("output") or "").strip()
        if not text:
            continue
        if kind == "user_message":
            turns.append(ChatTurn(role="user", content=text))
        elif kind == "assistant_message" and not step.get("parentId"):
            turns.append(ChatTurn(role="assistant", content=strip_footers(text)))
    return turns[-limit:] if limit else []
