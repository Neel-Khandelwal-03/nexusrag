"""Token counting for chunk sizing and context budgets.

Gemini's tokenizer isn't available offline, so we use tiktoken's ``cl100k_base`` as a
close proxy (usually within ~10-20% for English prose, which is plenty for sizing
chunks and budgets). If the encoding can't be loaded, for example with no network on
first run, we fall back to the common ~4 characters per token heuristic.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from nexusrag.log import get_logger

CHARS_PER_TOKEN = 4

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _encoding() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # network/offline or missing package
        log.warning("tokens.fallback_heuristic", error=type(exc).__name__)
        return None


def count_tokens(text: str) -> int:
    """Approximate number of LLM tokens in ``text``."""
    if not text:
        return 0
    enc = _encoding()
    if enc is None:
        return math.ceil(len(text) / CHARS_PER_TOKEN)
    return len(enc.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return the longest prefix of ``text`` that fits in ``max_tokens``."""
    if max_tokens <= 0 or not text:
        return ""
    enc = _encoding()
    if enc is None:
        return text[: max_tokens * CHARS_PER_TOKEN]
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return str(enc.decode(ids[:max_tokens]))


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Cut ``text`` into consecutive windows of at most ``max_tokens`` (last-resort splitting)."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not text:
        return []
    enc = _encoding()
    if enc is None:
        size = max_tokens * CHARS_PER_TOKEN
        return [text[i : i + size] for i in range(0, len(text), size)]
    ids = enc.encode(text, disallowed_special=())
    return [str(enc.decode(ids[i : i + max_tokens])) for i in range(0, len(ids), max_tokens)]
