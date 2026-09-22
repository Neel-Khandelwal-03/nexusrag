"""The evaluation dataset: one JSON object per line in ``eval/dataset.jsonl``.

Each answerable item records where its answer lives: the parent sections (what the answer
model reads) and the child chunks (what search retrieves) that contain the evidence
quotes. Chunk and parent IDs are derived from document IDs and positions, so they stay
stable while the documents and chunking settings do. If they change, :func:`resolve_gold`
finds the sections again from the evidence quotes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from nexusrag.store import Stores

Kind = Literal["fact", "numeric", "table", "comparison", "unanswerable"]

DEFAULT_PATH = Path(__file__).with_name("dataset.jsonl")


class EvalItem(BaseModel):
    """One test question with its reference answer and ground-truth sources."""

    id: str
    question: str
    #: Reference answer; empty for unanswerable questions.
    answer: str = ""
    answerable: bool = True
    kind: Kind = "fact"
    #: Filenames of the documents that contain the answer.
    documents: list[str] = Field(default_factory=list)
    source_parent_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    #: Exact quotes from the documents that prove the answer.
    evidence: list[str] = Field(default_factory=list)
    #: Set once a person has checked the item.
    reviewed: bool = False
    notes: str = ""


def load_dataset(path: Path = DEFAULT_PATH) -> list[EvalItem]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                items.append(EvalItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: invalid item: {exc}") from exc
    ids = [item.id for item in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate ids {sorted(duplicates)}")
    return items


def save_dataset(items: Sequence[EvalItem], path: Path = DEFAULT_PATH) -> None:
    lines = [json.dumps(item.model_dump(), ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_MARKDOWN_RE = re.compile(r"[|*_`#>]+")


def normalize(text: str) -> str:
    """Text reduced for matching evidence quotes.

    Lowercase, straight quotes, collapsed whitespace, and no Markdown punctuation, so a
    quote of a table row ("Maximum flight time 46 minutes") matches "| Maximum flight time
    | 46 minutes |".
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = _MARKDOWN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def resolve_gold(items: Sequence[EvalItem], stores: Stores, collection: str) -> list[str]:
    """Re-find source sections whose IDs no longer exist, using the evidence quotes.

    Updates the items in place and returns warnings for items that couldn't be resolved.
    """
    warnings: list[str] = []
    docs = {d.filename: d for d in stores.registry.list_documents(collection)}
    for item in items:
        if not item.answerable:
            continue
        found = stores.parents.get_many(collection, item.source_parent_ids)
        if item.source_parent_ids and len(found) == len(item.source_parent_ids):
            continue
        parent_ids: list[str] = []
        for filename in item.documents:
            doc = docs.get(filename)
            if doc is None:
                continue
            for parent in stores.parents.for_document(collection, doc.doc_id):
                text = normalize(parent.text)
                if any(normalize(quote) in text for quote in item.evidence):
                    parent_ids.append(parent.parent_id)
        if parent_ids:
            item.source_parent_ids = parent_ids
        else:
            warnings.append(f"{item.id}: source sections not found; retrieval metrics skipped")
    return warnings
