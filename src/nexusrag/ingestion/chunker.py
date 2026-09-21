"""Structure-aware parent/child chunking.

Pipeline (all pure functions):

1. **Sections.** Walk the document's elements, tracking the heading stack, so every
   body element knows its section path, e.g. ``2 Hardware > 2.3 Battery``.
2. **Parents** (~``PARENT_CHUNK_TOKENS``). Whole sections are the unit. Small
   neighbouring sections under the same top-level heading are merged so the LLM
   gets enough context. Long sections are split at element boundaries, and only an
   over-long paragraph falls back to recursive splitting. Parents are what the LLM
   reads.
3. **Children** (~``CHILD_CHUNK_TOKENS`` with ``CHILD_CHUNK_OVERLAP``). Each parent
   is cut into small pieces for embedding and search. Children never cross parent or
   section boundaries, so every child has one exact section path to cite.

Why this shape: small chunks embed sharply (one topic per vector) and rank well,
but they're too thin to answer from. Retrieving small and reading big gets
precise retrieval *and* enough context.

Tables are atomic. A table is never split across chunks unless it is larger than a
whole parent, and then only between rows, with the header row repeated, so no chunk
ever holds a partial row or a headerless fragment. Code blocks are also kept whole
when they fit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from nexusrag.config import Settings
from nexusrag.ingestion.metadata import make_chunk_id, make_parent_id, short_hash
from nexusrag.models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentElement,
    ElementKind,
    ParentSection,
)
from nexusrag.utils.tokens import count_tokens, split_by_tokens

SECTION_SEPARATOR = " > "

# Split preference for over-long text: paragraphs, then lines, then sentences, then words.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class ChunkingConfig:
    """Token budgets for chunking (see ``Settings`` for the env vars)."""

    child_tokens: int = 256
    child_overlap: int = 40
    parent_tokens: int = 1200

    @property
    def min_section_tokens(self) -> int:
        """Sections smaller than this are merged with neighbours under the same top heading."""
        return max(60, self.parent_tokens // 8)

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkingConfig:
        return cls(
            child_tokens=settings.child_chunk_tokens,
            child_overlap=settings.child_chunk_overlap,
            parent_tokens=settings.parent_chunk_tokens,
        )


@dataclass
class ChunkedDocument:
    """Output of :func:`chunk_document`."""

    parents: list[ParentSection]
    chunks: list[Chunk]


# --------------------------------------------------------------------------- recursive splitting


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s.strip()]


def _split_on(text: str, level: int) -> tuple[list[str], str]:
    if level == 0:
        return text.split("\n\n"), "\n\n"
    if level == 1:
        return text.split("\n"), "\n"
    if level == 2:
        return split_sentences(text), " "
    return text.split(" "), " "


def split_text(text: str, max_tokens: int, _level: int = 0) -> list[str]:
    """Split ``text`` into pieces of at most ``max_tokens``.

    Tries paragraph, line, sentence and word boundaries in that order, recursing into
    any piece that is still too long. Adjacent small pieces are re-merged greedily, so
    the output is as few pieces as the budget allows. A single "word" longer than the
    budget (e.g. a URL) is cut on token boundaries as a last resort.
    """
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    if _level > 3:
        return split_by_tokens(text, max_tokens)
    parts, joiner = _split_on(text, _level)
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 1:
        return split_text(text, max_tokens, _level + 1)

    pieces: list[str] = []
    for part in parts:
        pieces.extend(split_text(part, max_tokens, _level + 1))

    merged: list[str] = []
    for piece in pieces:
        if merged and count_tokens(merged[-1] + joiner + piece) <= max_tokens:
            merged[-1] = merged[-1] + joiner + piece
        else:
            merged.append(piece)
    return merged


def tail_text(text: str, max_tokens: int) -> str:
    """The last ``max_tokens`` of ``text``, aligned to sentence (else word) boundaries."""
    if max_tokens <= 0 or not text:
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(text.replace("\n", " "))):
        tokens = count_tokens(sentence)
        if total + tokens > max_tokens:
            break
        tail.insert(0, sentence)
        total += tokens
    if tail:
        return " ".join(tail)
    words: list[str] = []
    for word in reversed(text.split()):
        if count_tokens(" ".join([word, *words])) > max_tokens:
            break
        words.insert(0, word)
    return " ".join(words)


def split_table(markdown: str, max_tokens: int) -> list[str]:
    """Split a Markdown table between rows, repeating the header in every part."""
    lines = markdown.split("\n")
    if len(lines) <= 3 or count_tokens(markdown) <= max_tokens:
        return [markdown]
    header, rows = lines[:2], lines[2:]
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and count_tokens("\n".join([*header, *current, row])) > max_tokens:
            parts.append("\n".join([*header, *current]))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join([*header, *current]))
    return parts


# --------------------------------------------------------------------------- sections


@dataclass
class _Block:
    kind: ElementKind
    text: str
    page: int | None
    tokens: int

    @property
    def atomic(self) -> bool:
        return self.kind == ElementKind.TABLE


@dataclass
class _Section:
    path: str
    top: str  # top-level heading this section lives under (merge boundary)
    heading: str | None
    level: int
    blocks: list[_Block] = field(default_factory=list)

    @property
    def heading_line(self) -> str | None:
        return f"{'#' * min(self.level, 6)} {self.heading}" if self.heading else None

    @property
    def tokens(self) -> int:
        heading = count_tokens(self.heading_line) if self.heading_line else 0
        return heading + sum(block.tokens for block in self.blocks)


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _title_heading(doc: Document) -> DocumentElement | None:
    """A lone H1 that opens the document is its title, not a section, so it's kept out of paths."""
    headings = [e for e in doc.elements if e.kind == ElementKind.HEADING]
    h1s = [h for h in headings if h.level == 1]
    if len(h1s) == 1 and headings[0] is h1s[0]:
        return h1s[0]
    if h1s and _normalize(h1s[0].text) == _normalize(doc.title) and headings[0] is h1s[0]:
        return h1s[0]
    return None


def _make_blocks(element: DocumentElement, parent_tokens: int) -> list[_Block]:
    tokens = count_tokens(element.text)
    if element.kind == ElementKind.TABLE and tokens > parent_tokens:
        return [
            _Block(ElementKind.TABLE, part, element.page, count_tokens(part))
            for part in split_table(element.text, parent_tokens)
        ]
    return [_Block(element.kind, element.text, element.page, tokens)]


def build_sections(doc: Document, config: ChunkingConfig) -> list[_Section]:
    """Group body elements under their heading path. Empty sections are dropped."""
    title_heading = _title_heading(doc)
    stack: list[tuple[int, str]] = []
    sections = [_Section(path="", top="", heading=None, level=1)]
    for element in doc.elements:
        if element.kind == ElementKind.HEADING:
            if element is title_heading:
                continue
            level = element.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, element.text))
            sections.append(
                _Section(
                    path=SECTION_SEPARATOR.join(text for _, text in stack),
                    top=stack[0][1],
                    heading=element.text,
                    level=level,
                )
            )
        elif element.text.strip():
            sections[-1].blocks.extend(_make_blocks(element, config.parent_tokens))
    return [s for s in sections if s.blocks]


# --------------------------------------------------------------------------- parents


@dataclass
class _ParentDraft:
    sections: list[_Section] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sections)

    @property
    def top(self) -> str:
        return self.sections[0].top if self.sections else ""


def _split_section(section: _Section, config: ChunkingConfig) -> list[_Section]:
    """Split an over-long section into parts that fit a parent, at element boundaries."""
    heading_tokens = count_tokens(section.heading_line) if section.heading_line else 0
    budget = max(config.child_tokens, config.parent_tokens - heading_tokens)
    blocks: list[_Block] = []
    for block in section.blocks:
        if block.tokens > budget and not block.atomic:
            blocks.extend(
                _Block(block.kind, part, block.page, count_tokens(part))
                for part in split_text(block.text, budget)
            )
        else:
            blocks.append(block)

    parts: list[_Section] = []
    current: list[_Block] = []
    size = 0
    for block in blocks:
        if current and size + block.tokens > budget:
            parts.append(
                _Section(section.path, section.top, section.heading, section.level, current)
            )
            current, size = [], 0
        current.append(block)
        size += block.tokens
    if current:
        parts.append(_Section(section.path, section.top, section.heading, section.level, current))
    return parts


def plan_parents(sections: Sequence[_Section], config: ChunkingConfig) -> list[_ParentDraft]:
    """Pack sections into parents: split big ones, merge small neighbours."""
    drafts: list[_ParentDraft] = []
    current = _ParentDraft()

    def flush() -> None:
        nonlocal current
        if current.sections:
            drafts.append(current)
        current = _ParentDraft()

    for section in sections:
        if section.tokens > config.parent_tokens:
            flush()
            drafts.extend(_ParentDraft([part]) for part in _split_section(section, config))
            continue
        small = section.tokens < config.min_section_tokens or (
            bool(current.sections) and current.tokens < config.min_section_tokens
        )
        fits = current.tokens + section.tokens <= config.parent_tokens
        if current.sections and small and fits and section.top == current.top:
            current.sections.append(section)
        else:
            flush()
            current.sections.append(section)
    flush()
    return drafts


def _common_path(paths: Sequence[str]) -> str:
    split = [p.split(SECTION_SEPARATOR) if p else [] for p in paths]
    common: list[str] = []
    for parts in zip(*split, strict=False):
        if all(part == parts[0] for part in parts):
            common.append(parts[0])
        else:
            break
    return SECTION_SEPARATOR.join(common)


def _page_range(pages: Sequence[int | None]) -> tuple[int | None, int | None]:
    known = [p for p in pages if p is not None]
    return (min(known), max(known)) if known else (None, None)


def _parent_text(draft: _ParentDraft) -> str:
    parts: list[str] = []
    for section in draft.sections:
        if section.heading_line:
            parts.append(section.heading_line)
        parts.extend(block.text for block in section.blocks)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- children


@dataclass
class _Piece:
    text: str
    tokens: int
    page: int | None
    atomic: bool
    block: int  # index of the source block; pieces of one block are joined with a space


def _child_pieces(section: _Section, config: ChunkingConfig) -> list[_Piece]:
    # Leave room for the overlap carried in from the previous child, so long paragraphs
    # still overlap instead of each piece filling the whole budget.
    budget = max(16, config.child_tokens - config.child_overlap)
    pieces: list[_Piece] = []
    for index, block in enumerate(section.blocks):
        if block.atomic or block.tokens <= config.child_tokens:
            pieces.append(_Piece(block.text, block.tokens, block.page, block.atomic, index))
        else:
            pieces.extend(
                _Piece(part, count_tokens(part), block.page, False, index)
                for part in split_text(block.text, budget)
            )
    return pieces


def _join_pieces(pieces: Sequence[_Piece]) -> str:
    """Join pieces, keeping paragraph breaks only between different source blocks."""
    out = pieces[0].text
    for prev, piece in pairwise(pieces):
        out += (" " if piece.block == prev.block else "\n\n") + piece.text
    return out


def pack_children(
    pieces: Sequence[_Piece], config: ChunkingConfig
) -> list[tuple[str, list[int | None]]]:
    """Greedily pack pieces into children with sentence-aligned overlap.

    Returns ``(text, pages)`` per child. Tables never donate overlap text, and an
    oversized table becomes a child on its own rather than being cut.
    """
    children: list[tuple[str, list[int | None]]] = []
    current: list[_Piece] = []
    size = 0
    carry: _Piece | None = None

    def flush(with_overlap: bool) -> None:
        nonlocal current, size, carry
        if not current:
            return
        text = _join_pieces(current)
        children.append((text, [p.page for p in current]))
        carry = None
        last = current[-1]
        if with_overlap and not last.atomic and config.child_overlap > 0:
            overlap = tail_text(last.text, config.child_overlap)
            if overlap and overlap != text:
                carry = _Piece(overlap, count_tokens(overlap), last.page, False, last.block)
        current, size = [], 0

    for piece in pieces:
        if current and size + piece.tokens > config.child_tokens:
            flush(with_overlap=True)
        if not current and carry is not None:
            if not piece.atomic and carry.tokens + piece.tokens <= config.child_tokens:
                current, size = [carry], carry.tokens
            carry = None
        current.append(piece)
        size += piece.tokens
    flush(with_overlap=False)
    return children


# --------------------------------------------------------------------------- entry point


def contextual_text(chunk: Chunk) -> str:
    """Text used for embedding and BM25: the chunk prefixed with its section path.

    The section path supplies context a small chunk lacks ("Battery > Charging" makes
    "takes 90 minutes" findable). The document title goes into the embedding's
    ``title:`` field and into :func:`keyword_text` for BM25.
    """
    path = chunk.metadata.section_path
    return f"{path}\n{chunk.text}" if path else chunk.text


def keyword_text(chunk: Chunk) -> str:
    """Text tokenised for BM25: title + section path + chunk."""
    return f"{chunk.metadata.title}\n{contextual_text(chunk)}"


_TABLE_ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{3,}:?$")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")


def linearize_tables(text: str) -> str:
    """Rewrite Markdown tables as ``Header: value`` lines; other text is unchanged.

    Cross-encoders are trained on prose. Measured on the sample corpus, pipes and dashes
    made them under-score table chunks badly: the "3 Performance" table scored 0.28 for
    a wind-resistance question, but 0.81 once linearised. Two-column tables become
    ``Key: value``; wider tables become ``Col: value; Col: value`` per row.
    """
    out: list[str] = []
    header: list[str] | None = None
    for line in text.split("\n"):
        match = _TABLE_ROW_RE.match(line)
        if match is None:
            header = None
            out.append(line)
            continue
        cells = [c.strip().replace("\\|", "|") for c in _UNESCAPED_PIPE_RE.split(match.group(1))]
        if all(_TABLE_SEP_CELL_RE.match(c) for c in cells if c) and any(cells):
            continue  # |---|---|
        if header is None:
            header = cells
            continue
        if len(header) == 2 and len(cells) >= 2:
            out.append(f"{cells[0]}: {cells[1]}")
        else:
            out.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells, strict=False) if c))
    return "\n".join(out)


def rerank_text(chunk: Chunk) -> str:
    """Text a reranker reads: section path + chunk, with tables linearised."""
    return linearize_tables(contextual_text(chunk))


def chunk_document(doc: Document, collection: str, config: ChunkingConfig) -> ChunkedDocument:
    """Turn a parsed document into parent sections and child chunks."""
    parents: list[ParentSection] = []
    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()

    for p_index, draft in enumerate(plan_parents(build_sections(doc, config), config)):
        parent_id = make_parent_id(doc.doc_id, p_index)
        parent_text = _parent_text(draft)
        start, end = _page_range([b.page for s in draft.sections for b in s.blocks])
        parents.append(
            ParentSection(
                parent_id=parent_id,
                doc_id=doc.doc_id,
                collection=collection,
                filename=doc.filename,
                source_type=doc.source_type,
                title=doc.title,
                section_path=_common_path([s.path for s in draft.sections]),
                page_start=start,
                page_end=end,
                text=parent_text,
                token_count=count_tokens(parent_text),
            )
        )
        for section in draft.sections:
            for text, pages in pack_children(_child_pieces(section, config), config):
                content_hash = short_hash(text)
                if content_hash in seen_hashes:
                    continue  # exact duplicate within this document (boilerplate, repeated notes)
                seen_hashes.add(content_hash)
                c_start, c_end = _page_range(pages)
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc.doc_id, len(chunks)),
                        text=text,
                        token_count=count_tokens(text),
                        metadata=ChunkMetadata(
                            doc_id=doc.doc_id,
                            collection=collection,
                            filename=doc.filename,
                            source_type=doc.source_type,
                            title=doc.title,
                            section_path=section.path,
                            page_start=c_start,
                            page_end=c_end,
                            chunk_index=len(chunks),
                            parent_id=parent_id,
                            content_hash=content_hash,
                            ingested_at=doc.ingested_at,
                        ),
                    )
                )
    return ChunkedDocument(parents=parents, chunks=chunks)
