"""Map inline ``[n]`` citation markers in an answer to their source passages.

The model cites passage numbers from its context. After generation we:

* find which passages were cited, in order of first appearance;
* remove markers pointing at passages that don't exist (hallucinated citations);
* build :class:`~nexusrag.models.Citation` objects for the cited passages only.

Markers are kept with their original numbers rather than renumbered: the answer has
already been streamed to the user, and rewriting "[3]" to "[1]" afterwards would make
the text visibly change.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from nexusrag.models import Citation, ContextPassage

# [1]  [1, 2]  [1,2,3]  [2-4]  [2–4]; up to 3 digits so years like [2026] aren't matched.
MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]")
_MAX_RANGE = 20


def parse_marker(body: str) -> list[int]:
    """Numbers referenced by one marker body, e.g. ``"1, 3-4"`` -> ``[1, 3, 4]``."""
    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", body.strip()):
        bounds = re.split(r"\s*[\-–]\s*", part)
        if len(bounds) == 2:
            start, end = int(bounds[0]), int(bounds[1])
            if start <= end and end - start <= _MAX_RANGE:
                numbers.extend(range(start, end + 1))
            else:
                numbers.extend([start, end])
        elif part:
            numbers.append(int(part))
    return numbers


def cited_indices(text: str) -> list[int]:
    """Distinct passage numbers cited in ``text``, in order of first appearance."""
    seen: list[int] = []
    for match in MARKER_RE.finditer(text):
        for number in parse_marker(match.group(1)):
            if number not in seen:
                seen.append(number)
    return seen


def strip_invalid_markers(text: str, valid: Collection[int]) -> str:
    """Drop references to non-existent passages; normalise the rest to ``[a][b]`` form."""

    def fix(match: re.Match[str]) -> str:
        numbers = [n for n in parse_marker(match.group(1)) if n in valid]
        # A sentinel marks fully-invalid markers so we can also drop the space before them.
        return "".join(f"[{n}]" for n in dict.fromkeys(numbers)) or "\x00"

    return re.sub(r"[ \t]*\x00", "", MARKER_RE.sub(fix, text))


def build_citations(text: str, passages: Sequence[ContextPassage]) -> list[Citation]:
    """Citations for the passages actually cited in ``text``, in order of first mention."""
    by_index = {p.index: p for p in passages}
    return [by_index[n].to_citation() for n in cited_indices(text) if n in by_index]
