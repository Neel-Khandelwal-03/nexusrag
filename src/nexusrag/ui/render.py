"""Markdown rendering for chat messages, source panels, pipeline steps and stats.

Kept free of Chainlit imports so the formatting can be unit-tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from nexusrag.models import AgentStep, Citation
from nexusrag.store.registry import DocumentRecord

if TYPE_CHECKING:
    from nexusrag.ingestion.pipeline import ProgressEvent
    from nexusrag.service import KnowledgeBaseStats

# --------------------------------------------------------------------------- citations


def citation_element_name(citation: Citation) -> str:
    """Name of the side-panel element for a citation.

    Chainlit turns every occurrence of an element's name in the message into a link,
    so naming it ``[n]`` makes the inline citation markers clickable.
    """
    return f"[{citation.index}]"


def pdf_element_name(citation: Citation) -> str:
    """Name of the PDF preview element for a citation (also appears in the footer)."""
    page = citation.page_start or 1
    return f"PDF page {page} [{citation.index}]"


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


def source_panel_markdown(citation: Citation) -> str:
    """Side-panel content: where the passage comes from, what matched, the full section."""
    lines = [f"**{citation.title}**", f"*{citation.location}*", ""]
    if citation.highlights:
        lines.append(
            "**Matched excerpt**" if len(citation.highlights) == 1 else "**Matched excerpts**"
        )
        for highlight in citation.highlights:
            lines += [_quote(highlight), ""]
    lines += ["**Full section sent to the model**", "", citation.text]
    return "\n".join(lines)


def sources_footer(citations: Sequence[Citation], with_pdf: Sequence[int] = ()) -> str:
    """Compact list of the cited sources, appended below the answer.

    ``with_pdf`` lists citation indexes that also have a PDF preview element; its name is
    added to the line so Chainlit renders it as a link.
    """
    rows = []
    for c in citations:
        pdf = f" · {pdf_element_name(c)}" if c.index in with_pdf else ""
        rows.append(f"- {citation_element_name(c)} {c.location}{pdf}")
    return "**Sources**\n" + "\n".join(rows)


def match_element_name(citation: Citation) -> str:
    """Side-panel name for a "closest match" (distinct from answer citations)."""
    return f"match {citation.index}"


def closest_matches_footer(matches: Sequence[Citation]) -> str:
    """Shown under a refusal: what the knowledge base *does* contain near the question."""
    rows = [f"- {match_element_name(m)}: {m.location}" for m in matches]
    return "**Closest matches in your documents**\n" + "\n".join(rows)


# --------------------------------------------------------------------------- pipeline steps

_STEP_TITLES = {
    "route": "Route the request",
    "retrieve": "Retrieve passages",
    "grade_relevance": "Check relevance",
    "generate": "Write the answer",
    "check_groundedness": "Check groundedness",
    "regenerate": "Regenerate (stricter)",
    "refuse_unsupported": "Decline",
    "summarize": "Summarise document",
    "compare": "Compare documents",
    "chitchat": "Reply",
    "out_of_scope": "Out of scope",
}
_STEP_TYPES = {
    "retrieve": "retrieval",
    "route": "llm",
    "grade_relevance": "llm",
    "check_groundedness": "llm",
    "generate": "llm",
    "regenerate": "llm",
}


def step_title(node: str) -> str:
    return _STEP_TITLES.get(node, node.replace("_", " ").capitalize())


def step_type(node: str) -> str:
    """Chainlit step type (drives the icon)."""
    return _STEP_TYPES.get(node, "tool")


def _cell(value: Any) -> str:
    return "—" if value is None else str(value)


def _ranking_table(rows: Sequence[Mapping[str, Any]], with_rerank: bool) -> str:
    header = "| # | Source | Dense | BM25 | RRF |" + (" Rerank |" if with_rerank else "")
    sep = "|---|---|---|---|---|" + ("---|" if with_rerank else "")
    lines = [header, sep]
    for i, row in enumerate(rows, start=1):
        cells = [
            str(i),
            str(row.get("location", "")).replace("|", "/"),
            _cell(row.get("dense_rank")),
            _cell(row.get("bm25_rank")),
            _cell(row.get("rrf")),
        ]
        if with_rerank:
            cells.append(_cell(row.get("rerank")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def step_markdown(step: AgentStep) -> str:
    """Body of a pipeline step in the UI: the summary line plus node-specific detail."""
    d = step.detail
    parts = [f"**{step.label}**"]
    if step.node == "route":
        parts.append(f"- Route: `{d.get('route', '')}`")
        if d.get("standalone_question"):
            parts.append(f"- Standalone question: *{d['standalone_question']}*")
        if d.get("documents"):
            parts.append("- Documents: " + ", ".join(f"`{x}`" for x in d["documents"]))
        if d.get("forced_by_mode"):
            parts.append("- Forced by the selected chat profile")
    elif step.node == "retrieve":
        parts.append(f"- Search query: *{d.get('query', '')}*")
        if d.get("scope"):
            parts.append("- Searched only: " + ", ".join(f"`{x}`" for x in d["scope"]))
        for variant in d.get("variants") or []:
            parts.append(f"  - variant: *{variant}*")
        timings = d.get("timings") or {}
        if timings:
            parts.append("- Stages: " + ", ".join(f"{k} {v:.0f} ms" for k, v in timings.items()))
        candidates = d.get("candidates") or []
        if candidates:
            parts += ["", "**Hybrid candidates** (dense + BM25, fused with RRF)", ""]
            parts.append(_ranking_table(candidates, with_rerank=False))
        kept = d.get("kept") or []
        if kept:
            reranker = d.get("reranker") or "none"
            parts += ["", f"**Kept for the answer** (reranker: {reranker})", ""]
            parts.append(_ranking_table(kept, with_rerank=True))
    elif step.node == "grade_relevance":
        parts.append(f"- Sufficient: {'yes' if d.get('sufficient') else 'no'}")
        if d.get("missing"):
            parts.append(f"- Missing: {d['missing']}")
        if d.get("better_query"):
            parts.append(f"- Rewritten query: *{d['better_query']}*")
    elif step.node == "check_groundedness":
        claims = d.get("unsupported_claims") or []
        if claims:
            parts.append("- Unsupported claims:")
            parts += [f"  - {c}" for c in claims]
    elif step.node in ("summarize", "compare") and d.get("documents"):
        documents = d["documents"]
        documents = [documents] if isinstance(documents, str) else documents
        parts.append("- Documents: " + ", ".join(f"`{x}`" for x in documents))
    return "\n".join(parts)


def steps_markdown(steps: Sequence[AgentStep]) -> str:
    """Compact pipeline trace, one line per agent step."""
    return "\n".join(f"{i}. {s.label} ({s.ms:.0f} ms)" for i, s in enumerate(steps, start=1))


# --------------------------------------------------------------------------- ingestion


def progress_markdown(event: ProgressEvent) -> str:
    """One line of per-file upload progress."""
    name = f"**{event.filename}**"
    d = event.detail
    stage = event.stage
    if stage in ("parsing", "chunking", "storing"):
        return f"⏳ {name}: {stage}…"
    if stage == "embedding":
        return (
            f"⏳ {name}: embedding {d.get('chunks', '?')} chunks "
            f"({d.get('parents', '?')} sections)…"
        )
    if stage == "done":
        verb = "updated" if d.get("status") == "updated" else "indexed"
        reused = f", {d['reused']} vectors reused" if d.get("reused") else ""
        return (
            f"✅ {name}: {verb}, {d.get('parents', 0)} sections, "
            f"{d.get('chunks', 0)} chunks{reused}"
        )
    if stage == "skipped":
        return f"⏭️ {name}: already indexed (unchanged)"
    return f"❌ {name}: {d.get('error', 'failed')}"


# --------------------------------------------------------------------------- stats & welcome


def stats_markdown(stats: KnowledgeBaseStats) -> str:
    """The /stats report."""
    u = stats.usage
    if stats.cache_lookups:
        rate = (stats.cache_hits or 0) / stats.cache_lookups
        cache = f"{stats.cache_hits}/{stats.cache_lookups} hits ({rate:.0%})"
    else:
        cache = "no lookups yet" if stats.cache_lookups == 0 else "not enabled yet"
    lines = [
        f"### Knowledge base `{stats.collection}`",
        "",
        "| Documents | Sections | Chunks | Vectors |",
        "|---|---|---|---|",
        f"| {len(stats.documents)} | {stats.parents} | {stats.chunks} | {stats.vectors} |",
        "",
    ]
    if stats.documents:
        lines += ["| Document | Type | Chunks | Indexed |", "|---|---|---|---|"]
        for d in stats.documents:
            lines.append(
                f"| {d.title} (`{d.filename}`) | {d.source_type.value} | {d.num_chunks} "
                f"| {d.ingested_at:%Y-%m-%d %H:%M} UTC |"
            )
        lines.append("")
    lines += [
        f"**Semantic cache:** {cache}",
        f"**Model usage since start:** {u.calls} calls, {u.prompt_tokens:,} prompt + "
        f"{u.output_tokens:,} output + {u.thoughts_tokens:,} thinking tokens, "
        f"~{u.embedding_tokens:,} embedding tokens, est. ${u.cost_usd:.4f}",
        f"**Knowledge bases:** {', '.join(f'`{c}`' for c in stats.collections) or 'none'}",
    ]
    return "\n".join(lines)


def welcome_markdown(collection: str, documents: Sequence[DocumentRecord]) -> str:
    """First message of a chat: what's indexed and how to start."""
    if not documents:
        return (
            f"👋 Welcome to **NexusRAG**. The knowledge base **{collection}** is empty.\n\n"
            "Drag files into the chat (PDF, DOCX, Markdown or TXT), add a web page with the "
            "button below, or index a folder with `python -m nexusrag.ingest data/`."
        )
    listing = "\n".join(f"- {d.title} (`{d.filename}`)" for d in documents[:12])
    more = f"\n- …and {len(documents) - 12} more" if len(documents) > 12 else ""
    chunks = sum(d.num_chunks for d in documents)
    return (
        f"👋 Welcome to **NexusRAG**. Knowledge base **{collection}**: "
        f"{len(documents)} documents, {chunks} chunks.\n\n{listing}{more}\n\n"
        "Ask a question and every answer cites its sources: click a `[n]` marker to see "
        "the exact passage. Drag in more files, add a web page, switch mode (Q&A, "
        "Summarize, Compare) at the top, or type `/stats`."
    )


def setup_error_markdown(message: str) -> str:
    return (
        f"⚠️ {message}\n\nAdd `GEMINI_API_KEY` to your `.env` file (see `.env.example`) "
        "and restart the app."
    )


#: Starter prompts for the sample corpus: (mode, file it needs, label, message).
_SAMPLE_STARTERS = [
    (
        "qa",
        "aurora-x1-spec.pdf",
        "Aurora battery",
        "How long does the Aurora X1 battery last, and how long does it take to charge?",
    ),
    (
        "qa",
        "remote-work-policy.md",
        "Remote stipends",
        "What equipment stipends do remote employees get?",
    ),
    (
        "qa",
        "q2-2026-business-review.pdf",
        "Q2 revenue",
        "What was Skylark Dynamics' revenue in Q2 2026, and how did it change?",
    ),
    (
        "qa",
        "customer-support-faq.txt",
        "Returns",
        "What is the return policy for opened products?",
    ),
    (
        "summarize",
        "borealis-s2-product-sheet.docx",
        "Borealis S2",
        "Summarize the Borealis S2 product sheet",
    ),
    (
        "summarize",
        "q2-2026-business-review.pdf",
        "Q2 review",
        "Summarize the Q2 2026 business review",
    ),
    (
        "compare",
        "borealis-s2-product-sheet.docx",
        "Aurora vs Borealis",
        "Compare the Aurora X1 and the Borealis S2 on flight time, range, wind resistance "
        "and warranty",
    ),
]


def starter_prompts(mode: str, documents: Sequence[DocumentRecord]) -> list[tuple[str, str]]:
    """``(label, message)`` starters that fit the documents actually indexed."""
    filenames = {d.filename for d in documents}
    curated = [
        (label, message)
        for starter_mode, needs, label, message in _SAMPLE_STARTERS
        if starter_mode == mode and needs in filenames
    ]
    if curated:
        return curated
    if not documents:
        return []
    if mode == "summarize":
        return [(d.title[:30], f"Summarize {d.title}") for d in documents[:3]]
    if mode == "compare":
        if len(documents) < 2:
            return []  # nothing to compare yet
        a, b = documents[0], documents[1]
        return [("Compare two documents", f"Compare {a.title} and {b.title}")]
    return [("What's in here?", "What topics do these documents cover?")]
