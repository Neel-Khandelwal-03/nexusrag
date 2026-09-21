"""Markdown for pipeline steps, upload progress, /stats, starters and PDF source links."""

from __future__ import annotations

from datetime import UTC, datetime

from nexusrag.ingestion.pipeline import ProgressEvent
from nexusrag.models import AgentStep, Citation, SourceType, UsageStats
from nexusrag.service import KnowledgeBaseStats
from nexusrag.store.registry import DocumentRecord
from nexusrag.ui.render import (
    pdf_element_name,
    progress_markdown,
    sources_footer,
    starter_prompts,
    stats_markdown,
    step_markdown,
    step_title,
    step_type,
)


def citation(index: int, filename: str = "spec.pdf", page: int | None = 3) -> Citation:
    return Citation(
        index=index,
        parent_id=f"p{index}",
        doc_id=f"d{index}",
        filename=filename,
        source_type=SourceType.PDF if filename.endswith(".pdf") else SourceType.MARKDOWN,
        title="Spec",
        section_path="Battery",
        page_start=page,
        text="The battery lasts 46 minutes.",
    )


def record(filename: str, title: str = "Doc", chunks: int = 5) -> DocumentRecord:
    return DocumentRecord(
        collection="default",
        doc_id=filename,
        source=filename,
        filename=filename,
        source_type=SourceType.PDF,
        title=title,
        content_hash="h",
        num_parents=2,
        num_chunks=chunks,
        ingested_at=datetime(2026, 9, 1, 12, 30, tzinfo=UTC),
    )


# --------------------------------------------------------------------------- citations


def test_pdf_links_in_sources_footer() -> None:
    cites = [citation(1), citation(2, "notes.md", page=None)]
    assert pdf_element_name(cites[0]) == "PDF page 3 [1]"
    assert pdf_element_name(citation(3, page=None)) == "PDF page 1 [3]"
    footer = sources_footer(cites, with_pdf=[1])
    assert "- [1] spec.pdf · p. 3 · Battery · PDF page 3 [1]" in footer
    assert "- [2] notes.md · Battery" in footer
    assert "PDF" not in footer.splitlines()[-1]


# --------------------------------------------------------------------------- steps


def test_step_titles_and_types() -> None:
    assert step_title("grade_relevance") == "Check relevance"
    assert step_title("some_new_node") == "Some new node"
    assert step_type("retrieve") == "retrieval"
    assert step_type("route") == "llm"
    assert step_type("chitchat") == "tool"


def test_route_step() -> None:
    step = AgentStep(
        node="route",
        label="Route: doc_qa",
        detail={
            "route": "doc_qa",
            "standalone_question": "How long does the Aurora battery last?",
            "documents": ["aurora.pdf"],
            "forced_by_mode": True,
        },
    )
    text = step_markdown(step)
    assert text.startswith("**Route: doc_qa**")
    assert "`doc_qa`" in text
    assert "*How long does the Aurora battery last?*" in text
    assert "`aurora.pdf`" in text
    assert "chat profile" in text


def test_retrieve_step_shows_hybrid_and_reranked_tables() -> None:
    row = {"location": "spec.pdf | p. 3", "dense_rank": 1, "bm25_rank": None, "rrf": 0.0164}
    step = AgentStep(
        node="retrieve",
        label="Retrieved 1 passage",
        detail={
            "query": "battery life",
            "scope": ["spec.pdf"],
            "variants": ["how long does the battery last"],
            "timings": {"dense": 120.4, "bm25": 3.2},
            "candidates": [row],
            "kept": [{**row, "rerank": 0.91}],
            "reranker": "cross_encoder",
        },
    )
    text = step_markdown(step)
    assert "- Search query: *battery life*" in text
    assert "- Searched only: `spec.pdf`" in text
    assert "variant: *how long does the battery last*" in text
    assert "dense 120 ms, bm25 3 ms" in text
    assert "**Hybrid candidates**" in text
    assert "(reranker: cross_encoder)" in text
    assert "| 1 | spec.pdf / p. 3 | 1 | — | 0.0164 |" in text  # pipes can't break the table
    assert "| 1 | spec.pdf / p. 3 | 1 | — | 0.0164 | 0.91 |" in text


def test_grader_steps() -> None:
    relevance = AgentStep(
        node="grade_relevance",
        label="Relevance: insufficient",
        detail={"sufficient": False, "missing": "charge time", "better_query": "charging time"},
    )
    text = step_markdown(relevance)
    assert "- Sufficient: no" in text
    assert "- Missing: charge time" in text
    assert "*charging time*" in text
    grounded = AgentStep(
        node="check_groundedness",
        label="Groundedness: unsupported",
        detail={"unsupported_claims": ["It lasts 3 hours."]},
    )
    assert "  - It lasts 3 hours." in step_markdown(grounded)
    summary = AgentStep(node="summarize", label="Summarised", detail={"documents": "spec.pdf"})
    assert "`spec.pdf`" in step_markdown(summary)
    assert step_markdown(AgentStep(node="chitchat", label="Replied")) == "**Replied**"


# --------------------------------------------------------------------------- progress


def test_progress_lines() -> None:
    assert progress_markdown(ProgressEvent("a.pdf", "parsing")) == "⏳ **a.pdf**: parsing…"
    embedding = ProgressEvent("a.pdf", "embedding", {"chunks": 12, "parents": 4})
    assert "embedding 12 chunks (4 sections)" in progress_markdown(embedding)
    done = ProgressEvent("a.pdf", "done", {"status": "indexed", "parents": 4, "chunks": 12})
    assert progress_markdown(done) == "✅ **a.pdf**: indexed, 4 sections, 12 chunks"
    updated = ProgressEvent(
        "a.pdf", "done", {"status": "updated", "parents": 4, "chunks": 12, "reused": 9}
    )
    assert "updated, 4 sections, 12 chunks, 9 vectors reused" in progress_markdown(updated)
    assert "already indexed" in progress_markdown(ProgressEvent("a.pdf", "skipped"))
    failed = ProgressEvent("a.pdf", "failed", {"error": "The PDF has no text layer."})
    assert progress_markdown(failed) == "❌ **a.pdf**: The PDF has no text layer."


# --------------------------------------------------------------------------- stats


def stats(**overrides: object) -> KnowledgeBaseStats:
    values: dict[str, object] = {
        "collection": "default",
        "documents": [record("spec.pdf", "Aurora Spec", 30), record("faq.txt", "FAQ", 12)],
        "parents": 20,
        "chunks": 42,
        "vectors": 42,
        "collections": ["default", "contracts"],
        "usage": UsageStats(calls=3, prompt_tokens=1200, output_tokens=300, cost_usd=0.0012),
    }
    values.update(overrides)
    return KnowledgeBaseStats(**values)  # type: ignore[arg-type]


def test_stats_report() -> None:
    text = stats_markdown(stats())
    assert "### Knowledge base `default`" in text
    assert "| 2 | 20 | 42 | 42 |" in text
    assert "| Aurora Spec (`spec.pdf`) | pdf | 30 | 2026-09-01 12:30 UTC |" in text
    assert "3 calls, 1,200 prompt + 300 output" in text
    assert "est. $0.0012" in text
    assert "`default`, `contracts`" in text
    assert "not enabled yet" in text


def test_stats_cache_hit_rate_and_empty_kb() -> None:
    assert "3/4 hits (75%)" in stats_markdown(stats(cache_hits=3, cache_lookups=4))
    assert "no lookups yet" in stats_markdown(stats(cache_hits=0, cache_lookups=0))
    empty = stats_markdown(stats(documents=[], parents=0, chunks=0, vectors=0))
    assert "| Document |" not in empty


# --------------------------------------------------------------------------- starters


def test_curated_starters_need_their_documents() -> None:
    docs = [record("aurora-x1-spec.pdf"), record("borealis-s2-product-sheet.docx")]
    qa = starter_prompts("qa", docs)
    assert [label for label, _ in qa] == ["Aurora battery"]
    assert [label for label, _ in starter_prompts("compare", docs)] == ["Aurora vs Borealis"]
    assert starter_prompts("summarize", docs)[0][1] == "Summarize the Borealis S2 product sheet"


def test_generic_starters_for_other_documents() -> None:
    docs = [record("a.pdf", "Handbook"), record("b.pdf", "Contract")]
    assert starter_prompts("qa", docs) == [
        ("What's in here?", "What topics do these documents cover?")
    ]
    assert starter_prompts("summarize", docs)[0] == ("Handbook", "Summarize Handbook")
    assert starter_prompts("compare", docs) == [
        ("Compare two documents", "Compare Handbook and Contract")
    ]
    assert starter_prompts("compare", docs[:1]) == []
    assert starter_prompts("qa", []) == []
