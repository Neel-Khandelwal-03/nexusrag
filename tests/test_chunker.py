from __future__ import annotations

from itertools import pairwise

import pytest

from nexusrag.ingestion.chunker import (
    ChunkingConfig,
    chunk_document,
    contextual_text,
    keyword_text,
    linearize_tables,
    split_table,
    split_text,
    tail_text,
)
from nexusrag.models import Document, DocumentElement, ElementKind, SourceType
from nexusrag.utils.tokens import count_tokens

CFG = ChunkingConfig(child_tokens=64, child_overlap=12, parent_tokens=400)


def h(text: str, level: int, page: int | None = None) -> DocumentElement:
    return DocumentElement(kind=ElementKind.HEADING, text=text, level=level, page=page)


def p(text: str, page: int | None = None) -> DocumentElement:
    return DocumentElement(kind=ElementKind.PARAGRAPH, text=text, page=page)


def table(rows: int, cols: int = 3) -> DocumentElement:
    header = "| " + " | ".join(f"Col {c}" for c in range(cols)) + " |"
    sep = "|" + "|".join(["---"] * cols) + "|"
    body = [
        "| " + " | ".join(f"row {r} value {c} alpha" for c in range(cols)) + " |"
        for r in range(rows)
    ]
    return DocumentElement(kind=ElementKind.TABLE, text="\n".join([header, sep, *body]))


def sentences(n: int, topic: str = "battery") -> str:
    return " ".join(
        f"Sentence {i} explains the {topic} behaviour in some detail for testing." for i in range(n)
    )


def make_doc(*elements: DocumentElement, title: str = "Spec") -> Document:
    return Document(
        doc_id="doc1",
        source="spec.md",
        filename="spec.md",
        source_type=SourceType.MARKDOWN,
        title=title,
        content_hash="h",
        elements=list(elements),
    )


# --------------------------------------------------------------------------- splitting helpers


def test_split_text_respects_budget_and_keeps_order() -> None:
    text = sentences(40)
    pieces = split_text(text, 50)
    assert len(pieces) > 1
    assert all(count_tokens(piece) <= 50 for piece in pieces)
    assert " ".join(pieces).split() == text.split()


def test_split_text_falls_back_to_tokens_for_unbreakable_text() -> None:
    blob = "x" * 2000  # one "word" far over budget
    pieces = split_text(blob, 20)
    assert all(count_tokens(piece) <= 20 for piece in pieces)
    assert "".join(pieces) == blob


def test_tail_text_is_sentence_aligned() -> None:
    text = "First sentence is long enough. Second one. Third one here."
    tail = tail_text(text, 8)
    assert tail.endswith("Third one here.")
    assert text.endswith(tail)
    assert not tail.startswith("sentence")


def test_split_table_repeats_header() -> None:
    element = table(rows=60)
    parts = split_table(element.text, 200)
    header = element.text.split("\n")[:2]
    assert len(parts) > 1
    for part in parts:
        assert part.split("\n")[:2] == header
        assert count_tokens(part) <= 200
    # Every data row appears exactly once across the parts.
    rows = [line for part in parts for line in part.split("\n")[2:]]
    assert rows == element.text.split("\n")[2:]


# --------------------------------------------------------------------------- sections & parents


def test_section_paths_and_title_heading_skipped() -> None:
    doc = make_doc(
        h("Spec", 1),
        p("Intro text about the product."),
        h("2 Hardware", 2),
        h("2.1 Battery", 3),
        p("The battery lasts long."),
    )
    result = chunk_document(doc, "kb", CFG)
    paths = [c.metadata.section_path for c in result.chunks]
    assert paths == ["", "2 Hardware > 2.1 Battery"]  # the lone H1 is the title, not a section


def test_every_chunk_points_at_an_existing_parent() -> None:
    doc = make_doc(h("A", 2), p(sentences(30)), h("B", 2), p(sentences(30, "motor")))
    result = chunk_document(doc, "kb", CFG)
    parent_ids = {parent.parent_id for parent in result.parents}
    assert all(chunk.metadata.parent_id in parent_ids for chunk in result.chunks)
    assert [c.chunk_id for c in result.chunks] == [
        f"doc1-c{i:05d}" for i in range(len(result.chunks))
    ]
    assert [c.metadata.chunk_index for c in result.chunks] == list(range(len(result.chunks)))


def test_budgets_respected() -> None:
    doc = make_doc(h("Long", 2), p(sentences(120)), h("Other", 2), p(sentences(5, "motor")))
    result = chunk_document(doc, "kb", CFG)
    assert all(c.token_count <= CFG.child_tokens for c in result.chunks)
    assert all(parent.token_count <= CFG.parent_tokens + 10 for parent in result.parents)
    assert len(result.parents) >= 3  # the long section was split into several parents


def test_children_overlap() -> None:
    doc = make_doc(h("Long", 2), p(sentences(30)))
    chunks = chunk_document(doc, "kb", CFG).chunks
    assert len(chunks) >= 3
    for prev, nxt in pairwise(chunks):
        # The next child starts with a sentence-aligned tail of the previous one...
        overlap = nxt.text[: nxt.text.index("testing.") + len("testing.")]
        assert prev.text.endswith(overlap)
        # ...joined with a space, not a fake paragraph break.
        assert "\n" not in nxt.text


def test_small_sibling_sections_merge_into_one_parent() -> None:
    doc = make_doc(
        h("3 Work arrangements", 2),
        h("3.1 Hybrid", 3),
        p("Two office days per week."),
        h("3.2 Remote", 3),
        p("Needs approval from the VP."),
        h("4 Equipment", 2),
        p("A stipend of 600 USD is paid once."),
    )
    result = chunk_document(doc, "kb", CFG)
    assert [parent.section_path for parent in result.parents] == [
        "3 Work arrangements",
        "4 Equipment",
    ]
    merged = result.parents[0]
    assert "### 3.1 Hybrid" in merged.text
    assert "### 3.2 Remote" in merged.text
    # Children keep their precise section for citations.
    assert [c.metadata.section_path for c in result.chunks] == [
        "3 Work arrangements > 3.1 Hybrid",
        "3 Work arrangements > 3.2 Remote",
        "4 Equipment",
    ]


def test_tables_are_never_split_when_they_fit_a_parent() -> None:
    element = table(rows=12)  # bigger than a child, smaller than a parent
    assert CFG.child_tokens < count_tokens(element.text) < CFG.parent_tokens
    doc = make_doc(h("Specs", 2), p("See the table."), element, p("Values are nominal."))
    result = chunk_document(doc, "kb", CFG)
    holders = [c for c in result.chunks if "| Col 0 |" in c.text]
    assert len(holders) == 1
    assert element.text in holders[0].text
    assert sum(c.text.count("row 5 value 0") for c in result.chunks) == 1


def test_oversized_table_split_between_rows_with_header() -> None:
    element = table(rows=80)
    assert count_tokens(element.text) > CFG.parent_tokens
    result = chunk_document(make_doc(h("Big", 2), element), "kb", CFG)
    table_chunks = [c for c in result.chunks if c.text.startswith("| Col 0 |")]
    assert len(table_chunks) > 1
    assert all(c.text.split("\n")[1] == "|---|---|---|" for c in table_chunks)


def test_page_ranges() -> None:
    doc = make_doc(h("Pages", 2, page=2), p(sentences(3), page=2), p(sentences(3, "motor"), page=3))
    result = chunk_document(doc, "kb", CFG)
    assert result.parents[0].page_start == 2
    assert result.parents[0].page_end == 3
    assert result.chunks[0].metadata.page_start == 2


def test_duplicate_chunks_are_dropped() -> None:
    boilerplate = "This document is confidential and for internal use only."
    doc = make_doc(
        h("A", 2), p(boilerplate), h("B", 2), p(boilerplate), h("C", 2), p("Unique content here.")
    )
    texts = [c.text for c in chunk_document(doc, "kb", CFG).chunks]
    assert sum(boilerplate in t for t in texts) == 1


def test_contextual_and_keyword_text() -> None:
    doc = make_doc(
        h("2 Hardware", 2), h("2.1 Battery", 3), p("Lasts 46 minutes."), title="Aurora X1"
    )
    chunk = chunk_document(doc, "kb", CFG).chunks[0]
    assert contextual_text(chunk) == "2 Hardware > 2.1 Battery\nLasts 46 minutes."
    assert keyword_text(chunk).startswith("Aurora X1\n2 Hardware")


def test_empty_document_yields_nothing() -> None:
    result = chunk_document(make_doc(h("Only a heading", 2)), "kb", CFG)
    assert result.parents == []
    assert result.chunks == []


@pytest.mark.parametrize("child", [32, 128, 256])
def test_chunking_is_deterministic(child: int) -> None:
    config = ChunkingConfig(child_tokens=child, child_overlap=8, parent_tokens=600)
    doc = make_doc(h("A", 2), p(sentences(50)), table(rows=8), h("B", 2), p(sentences(20, "motor")))
    first = chunk_document(doc, "kb", config)
    second = chunk_document(doc, "kb", config)
    assert [c.model_dump() for c in first.chunks] == [c.model_dump() for c in second.chunks]


def test_linearize_two_column_table() -> None:
    text = (
        "3 Performance\n| Metric | Value |\n|---|---|\n"
        "| Wind resistance | 10 m/s |\n| Range | 18 km |"
    )
    assert linearize_tables(text) == "3 Performance\nWind resistance: 10 m/s\nRange: 18 km"


def test_linearize_wide_table_and_escaped_pipes() -> None:
    text = (
        "| Product | Units | Revenue |\n|:---|---:|---|\n| Aurora X1 | 1,420 | 20.6 |\n"
        "| A \\| B | | 7.9 |\nAfter the table."
    )
    assert linearize_tables(text).split("\n") == [
        "Product: Aurora X1; Units: 1,420; Revenue: 20.6",
        "Product: A | B; Revenue: 7.9",
        "After the table.",
    ]


def test_linearize_leaves_prose_alone() -> None:
    text = "Plain text with a | pipe in the middle."
    assert linearize_tables(text) == text
