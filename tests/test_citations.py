from __future__ import annotations

import pytest

from nexusrag.generation.citations import (
    build_citations,
    cited_indices,
    parse_marker,
    strip_invalid_markers,
)
from nexusrag.models import ContextPassage, ParentSection, SourceType


def passage(index: int, filename: str = "spec.pdf") -> ContextPassage:
    return ContextPassage(
        index=index,
        parent=ParentSection(
            parent_id=f"p{index}",
            doc_id="d",
            collection="kb",
            filename=filename,
            source_type=SourceType.PDF,
            title="Spec",
            section_path=f"Section {index}",
            page_start=index,
            page_end=index,
            text=f"passage {index} text",
            token_count=3,
        ),
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [("1", [1]), ("1, 2", [1, 2]), ("2-4", [2, 3, 4]), ("2–3", [2, 3]), ("1,3-4", [1, 3, 4])],
)
def test_parse_marker(body: str, expected: list[int]) -> None:
    assert parse_marker(body) == expected


def test_cited_indices_order_and_dedup() -> None:
    text = "Battery lasts 46 min [2]. Charging takes 75 min [1][2]. Range is 12 km [3, 1]."
    assert cited_indices(text) == [2, 1, 3]


def test_years_and_links_are_not_citations() -> None:
    assert cited_indices("Founded in [2026] and see [docs](http://x).") == []


def test_strip_invalid_markers() -> None:
    text = "Fact one [1]. Invented [7]. Mixed [2, 9]. End."
    assert strip_invalid_markers(text, {1, 2}) == "Fact one [1]. Invented. Mixed [2]. End."


def test_strip_keeps_text_untouched_without_markers() -> None:
    text = "Q : spacing before a colon stays as written ."
    assert strip_invalid_markers(text, {1}) == text


def test_build_citations_maps_to_passages() -> None:
    passages = [passage(1), passage(2, "policy.md"), passage(3)]
    citations = build_citations("Stipend is 600 USD [2]. Battery [1]. Bogus [9].", passages)
    assert [c.index for c in citations] == [2, 1]
    assert citations[0].filename == "policy.md"
    assert citations[0].location == "policy.md · p. 2 · Section 2"
    assert citations[0].text == "passage 2 text"
