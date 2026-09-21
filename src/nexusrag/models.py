"""Core data models shared by ingestion, retrieval, generation and evaluation.

The flow of objects through the pipeline:

    Document (list of DocumentElement)
        -> ParentSection (~1,000-1,500 tokens, sent to the LLM)
            -> Chunk (~200-300 tokens, embedded and searched; points at its parent)
                -> RetrievedChunk (chunk + scores from every retrieval stage)
                    -> Citation ([n] marker -> exact source) -> Answer
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


class SourceType(StrEnum):
    """Where a document came from; drives loader choice and UI filters."""

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TEXT = "text"
    URL = "url"


class ElementKind(StrEnum):
    """Structural element types produced by loaders and consumed by the chunker."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"


class Route(StrEnum):
    """Intent classes the router assigns to each user message."""

    CHITCHAT = "chitchat"
    DOC_QA = "doc_qa"
    SUMMARIZE = "summarize_document"
    COMPARE = "compare_documents"
    OUT_OF_SCOPE = "out_of_scope"


# --------------------------------------------------------------------------- ingestion


class DocumentElement(BaseModel):
    """One structural unit of a document (heading, paragraph, table, ...)."""

    kind: ElementKind
    text: str
    #: Heading depth 1-6 for headings; None otherwise.
    level: int | None = Field(default=None, ge=1, le=6)
    #: 1-based page number (PDF only).
    page: int | None = Field(default=None, ge=1)


class Document(BaseModel):
    """A parsed source document, before chunking."""

    doc_id: str
    source: str  # file path or URL
    filename: str
    source_type: SourceType
    title: str
    content_hash: str
    elements: list[DocumentElement]
    ingested_at: datetime = Field(default_factory=utcnow)

    @property
    def text(self) -> str:
        """Concatenated text of all elements (for hashing and debugging)."""
        return "\n\n".join(element.text for element in self.elements)


ChromaScalar = str | int | float | bool


class ChunkMetadata(BaseModel):
    """Metadata stored alongside every child chunk (in Chroma and BM25)."""

    doc_id: str
    collection: str
    filename: str
    source_type: SourceType
    title: str
    section_path: str = ""  # e.g. "2 Methods > 2.1 Data"
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int = Field(ge=0)
    parent_id: str
    content_hash: str
    ingested_at: datetime

    def to_chroma(self) -> dict[str, ChromaScalar]:
        """Flatten to Chroma-compatible metadata (scalars only, no ``None`` values)."""
        data = self.model_dump(mode="json", exclude_none=True)
        return {key: value for key, value in data.items() if isinstance(value, ChromaScalar)}

    @classmethod
    def from_chroma(cls, metadata: dict[str, ChromaScalar]) -> ChunkMetadata:
        """Inverse of :meth:`to_chroma`."""
        return cls.model_validate(metadata)


class Chunk(BaseModel):
    """A child chunk: the unit that is embedded and searched."""

    chunk_id: str
    text: str
    token_count: int = Field(ge=0)
    metadata: ChunkMetadata


class ParentSection(BaseModel):
    """A full section: the unit of context actually sent to the LLM."""

    parent_id: str
    doc_id: str
    collection: str
    filename: str
    source_type: SourceType
    title: str
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    text: str
    token_count: int = Field(ge=0)


# --------------------------------------------------------------------------- retrieval


class StageScores(BaseModel):
    """Scores and ranks a chunk received at each retrieval stage (None = not run / not found)."""

    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with provenance for the transparency UI."""

    chunk: Chunk
    scores: StageScores = Field(default_factory=StageScores)
    #: Which query variants (original, rewrites, HyDE) retrieved this chunk.
    matched_queries: list[str] = Field(default_factory=list)


class SearchFilters(BaseModel):
    """Metadata restrictions applied to every retrieval stage (None = unrestricted)."""

    doc_ids: list[str] | None = None
    filenames: list[str] | None = None
    source_types: list[SourceType] | None = None

    @property
    def is_empty(self) -> bool:
        return self.doc_ids is None and self.filenames is None and self.source_types is None


class ContextPassage(BaseModel):
    """A numbered unit of context shown to the LLM.

    It is a parent section, plus the retrieved child chunks inside it (the evidence
    that pulled this section into the context).
    """

    index: int = Field(ge=1)
    parent: ParentSection
    chunks: list[RetrievedChunk] = Field(default_factory=list)

    def to_citation(self) -> Citation:
        p = self.parent
        return Citation(
            index=self.index,
            parent_id=p.parent_id,
            doc_id=p.doc_id,
            filename=p.filename,
            source_type=p.source_type,
            title=p.title,
            section_path=p.section_path,
            page_start=p.page_start,
            page_end=p.page_end,
            text=p.text,
            chunk_ids=[rc.chunk.chunk_id for rc in self.chunks],
            highlights=[rc.chunk.text for rc in self.chunks],
        )


# --------------------------------------------------------------------------- generation


class Citation(BaseModel):
    """Maps an inline ``[n]`` marker to its source.

    ``text`` is the full passage the model read (the parent section). ``chunk_ids`` and
    ``highlights`` are the specific retrieved chunks within it, shown highlighted in
    the source panel.
    """

    index: int = Field(ge=1)
    parent_id: str
    doc_id: str
    filename: str
    source_type: SourceType
    title: str
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    text: str
    chunk_ids: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)

    @property
    def location(self) -> str:
        """Human-readable location, e.g. ``report.pdf · p. 3-4 · 2 Methods > 2.1 Data``."""
        parts = [self.filename]
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"p. {self.page_start}-{self.page_end}")
            else:
                parts.append(f"p. {self.page_start}")
        if self.section_path:
            parts.append(self.section_path)
        return " · ".join(parts)


class ChatTurn(BaseModel):
    """One message of conversation history."""

    role: Literal["user", "assistant"]
    content: str


class UsageStats(BaseModel):
    """Aggregated LLM usage for a request (or the whole process)."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0


class StageTiming(BaseModel):
    """Latency of one pipeline stage."""

    stage: str
    ms: float


class Answer(BaseModel):
    """Final result of one chat turn, as rendered by the UI and scored by the eval suite."""

    text: str
    route: Route
    citations: list[Citation] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    #: Result of the groundedness check; None if the check didn't run.
    grounded: bool | None = None
    #: True when the answer says the documents don't cover the question.
    refused: bool = False
    cached: bool = False
    usage: UsageStats = Field(default_factory=UsageStats)
    timings: list[StageTiming] = Field(default_factory=list)
    request_id: str | None = None
