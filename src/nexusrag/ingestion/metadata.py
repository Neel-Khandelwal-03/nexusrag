"""Identity, hashing and text-normalisation helpers used during ingestion.

IDs are deterministic so re-ingesting the same source always maps to the same
document, parent and chunk IDs:

* ``doc_id``   = hash of the normalised source key (relative path, filename or URL)
* ``parent_id`` = ``{doc_id}-p{index}``
* ``chunk_id``  = ``{doc_id}-c{index}``
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from nexusrag.models import DocumentElement, ElementKind

_READ_BLOCK = 1 << 20
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")
_INLINE_SPACE_RE = re.compile(r"[ \t  -   　]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def short_hash(text: str, length: int = 16) -> str:
    """Truncated SHA-256 (64 bits by default): plenty to tell chunks apart within a corpus."""
    return sha256_text(text)[:length]


def file_sha256(path: Path) -> str:
    """Hash a file's bytes without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def normalize_source_key(key: str) -> str:
    """Canonical form of a source key: forward slashes, no URL fragment, trimmed."""
    key = key.strip().replace("\\", "/")
    return key.split("#", 1)[0] if key.startswith(("http://", "https://")) else key


def make_doc_id(source_key: str) -> str:
    return short_hash(normalize_source_key(source_key))


def make_parent_id(doc_id: str, index: int) -> str:
    return f"{doc_id}-p{index:04d}"


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}-c{index:05d}"


def clean_text(text: str, *, preserve_whitespace: bool = False) -> str:
    """NFC-normalise and strip control characters.

    Unless ``preserve_whitespace`` is set (code blocks), runs of spaces are collapsed,
    lines are trimmed and 3+ consecutive newlines become one blank line.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    if preserve_whitespace:
        return text.strip("\n")
    lines = (_INLINE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def title_from_filename(filename: str) -> str:
    """``q2-2026_report.pdf`` -> ``Q2 2026 Report``."""
    stem = PurePosixPath(filename.replace("\\", "/")).stem
    words = re.sub(r"[_\-]+", " ", stem).strip()
    return words.title() if words.islower() else words or filename


def infer_title(elements: Sequence[DocumentElement], fallback: str) -> str:
    """First level-1 heading, else the first heading of any level, else ``fallback``."""
    headings = [e for e in elements if e.kind == ElementKind.HEADING]
    for heading in headings:
        if heading.level == 1:
            return heading.text[:200]
    return headings[0].text[:200] if headings else fallback
