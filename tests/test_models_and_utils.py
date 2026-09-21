from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nexusrag.llm.prompts import render, template_fields
from nexusrag.models import ChunkMetadata, Citation, SourceType
from nexusrag.utils import tokens


def _metadata(**overrides: object) -> ChunkMetadata:
    values: dict[str, object] = {
        "doc_id": "d1",
        "collection": "default",
        "filename": "report.pdf",
        "source_type": SourceType.PDF,
        "title": "Report",
        "section_path": "2 Methods > 2.1 Data",
        "page_start": 3,
        "page_end": None,
        "chunk_index": 0,
        "parent_id": "p1",
        "content_hash": "abc",
        "ingested_at": datetime(2026, 9, 21, tzinfo=UTC),
    }
    values.update(overrides)
    return ChunkMetadata.model_validate(values)


def test_chroma_metadata_is_flat_and_non_null() -> None:
    flat = _metadata().to_chroma()
    assert "page_end" not in flat  # Chroma rejects None values
    assert flat["source_type"] == "pdf"
    assert isinstance(flat["ingested_at"], str)
    assert all(isinstance(v, str | int | float | bool) for v in flat.values())


def test_chroma_metadata_round_trip() -> None:
    original = _metadata()
    assert ChunkMetadata.from_chroma(original.to_chroma()) == original


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (3, 4, "report.pdf · p. 3-4 · Intro"),
        (3, 3, "report.pdf · p. 3 · Intro"),
        (None, None, "report.pdf · Intro"),
    ],
)
def test_citation_location(start: int | None, end: int | None, expected: str) -> None:
    citation = Citation(
        index=1,
        chunk_id="c",
        parent_id="p",
        doc_id="d",
        filename="report.pdf",
        source_type=SourceType.PDF,
        title="Report",
        section_path="Intro",
        page_start=start,
        page_end=end,
        text="...",
    )
    assert citation.location == expected


def test_render_fills_template() -> None:
    assert render("Q: {question}", question="why?") == "Q: why?"
    assert template_fields("{a} and {b}") == {"a", "b"}


def test_render_rejects_missing_or_extra_vars() -> None:
    with pytest.raises(KeyError, match="missing"):
        render("{a} {b}", a=1)
    with pytest.raises(KeyError, match="unexpected"):
        render("{a}", a=1, c=2)


def test_render_does_not_interpret_braces_in_values() -> None:
    assert render("ctx: {context}", context="{not_a_var}") == "ctx: {not_a_var}"


def test_count_tokens_basic() -> None:
    assert tokens.count_tokens("") == 0
    assert tokens.count_tokens("hello world, this is a sentence") > 3


def test_truncate_to_tokens() -> None:
    text = "word " * 200
    cut = tokens.truncate_to_tokens(text, 50)
    assert tokens.count_tokens(cut) <= 50
    assert text.startswith(cut)
    assert tokens.truncate_to_tokens("short", 50) == "short"


def test_token_fallback_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokens, "_encoding", lambda: None)
    assert tokens.count_tokens("abcdefgh") == 2
    assert tokens.truncate_to_tokens("abcdefghij", 2) == "abcdefgh"
