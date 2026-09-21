"""Structured logging with request-scoped context and redaction.

Every log line is a structlog event (JSON in deployed environments, pretty console
locally). Two guarantees are enforced centrally rather than trusted to call sites:

* **No secrets.** Values under secret-looking keys are masked, and anything shaped like
  an API key or token is scrubbed wherever it appears in a string.
* **No document contents.** Fields that may carry prompts, user questions or document
  text are replaced with a length summary unless ``LOG_CONTENT=true``.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import structlog
from structlog.contextvars import bound_contextvars, merge_contextvars
from structlog.typing import FilteringBoundLogger

from nexusrag.config import Settings

_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google API keys
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),  # Hugging Face tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
)
_SECRET_KEYS = frozenset({"api_key", "password", "secret", "token", "authorization", "credentials"})
_SECRET_SUFFIXES = ("_key", "_secret", "_password", "_token")

#: Event fields that may contain document text or user input.
CONTENT_KEYS = frozenset(
    {
        "text",
        "content",
        "contents",
        "prompt",
        "query",
        "queries",
        "question",
        "answer",
        "chunk",
        "document",
        "context",
    }
)

# Chatty third-party loggers that would otherwise flood INFO output.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "google_genai",
    "chromadb.telemetry",
    "sentence_transformers",
)


def scrub(value: str) -> str:
    """Mask substrings that look like credentials."""
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("***", value)
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SECRET_KEYS or lowered.endswith(_SECRET_SUFFIXES)


def _describe(value: Any) -> str:
    if isinstance(value, str):
        return f"<redacted {len(value)} chars>"
    if isinstance(value, list | tuple | set | dict):
        return f"<redacted {len(value)} items>"
    return "<redacted>"


class RedactProcessor:
    """structlog processor that removes secrets and (optionally) content from events."""

    def __init__(self, log_content: bool = False) -> None:
        self.log_content = log_content

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        for key, value in list(event_dict.items()):
            if _is_secret_key(key):
                event_dict[key] = "***"
            elif not self.log_content and key.lower() in CONTENT_KEYS:
                event_dict[key] = _describe(value)
            elif isinstance(value, str):
                event_dict[key] = scrub(value)
        return event_dict


class _ScrubFilter(logging.Filter):
    """Applies :func:`scrub` to stdlib log records emitted by third-party libraries."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub(record.getMessage())
        record.args = None
        return True


class _CurrentStdoutHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stdout`` is *now*, not the stream at configuration time.

    Test runners and some servers swap ``sys.stdout``; a handler holding the old stream
    would write to a closed file.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


class _CurrentStdoutLogger(structlog.PrintLogger):
    """structlog ``PrintLogger`` that resolves ``sys.stdout`` on every write (see above)."""

    def msg(self, message: str) -> None:
        self._file = sys.stdout
        super().msg(message)

    log = debug = info = warn = warning = error = critical = exception = fatal = failure = msg


def configure_logging(settings: Settings) -> None:
    """Configure structlog and the stdlib root logger. Safe to call more than once."""
    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)

    handler = _CurrentStdoutHandler()
    handler.addFilter(_ScrubFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))

    processors: list[Any] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if settings.log_format == "json":
        # Render tracebacks to a string *before* redaction so they get scrubbed too.
        processors.append(structlog.processors.format_exc_info)
        processors.append(RedactProcessor(settings.log_content))
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(RedactProcessor(settings.log_content))
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=lambda *args: _CurrentStdoutLogger(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger. Works before :func:`configure_logging` is called."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))


def new_request_id() -> str:
    """Short random ID used to correlate every log line of one user request."""
    return uuid.uuid4().hex[:12]


@contextmanager
def request_context(request_id: str | None = None, **fields: Any) -> Iterator[str]:
    """Bind ``request_id`` (and any extra fields) to all log lines inside the block."""
    rid = request_id or new_request_id()
    with bound_contextvars(request_id=rid, **fields):
        yield rid


@dataclass
class Timer:
    """Wall-clock timer; ``ms`` is live while running and frozen once stopped."""

    started: float = field(default_factory=time.perf_counter)
    stopped: float | None = None

    @property
    def ms(self) -> float:
        end = self.stopped if self.stopped is not None else time.perf_counter()
        return (end - self.started) * 1000.0


@contextmanager
def timed() -> Iterator[Timer]:
    """Measure the duration of a block: ``with timed() as t: ...; t.ms``."""
    timer = Timer()
    try:
        yield timer
    finally:
        timer.stopped = time.perf_counter()
