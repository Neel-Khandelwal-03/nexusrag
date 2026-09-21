"""Markdown rendering for chat messages and source panels.

Kept free of Chainlit imports so the formatting can be unit-tested.
"""

from __future__ import annotations

from collections.abc import Sequence

from nexusrag.models import AgentStep, Citation
from nexusrag.store.registry import DocumentRecord


def citation_element_name(citation: Citation) -> str:
    """Name of the side-panel element for a citation.

    Chainlit turns every occurrence of an element's name in the message into a link,
    so naming it ``[n]`` makes the inline citation markers clickable.
    """
    return f"[{citation.index}]"


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


def sources_footer(citations: Sequence[Citation]) -> str:
    """Compact list of the cited sources, appended below the answer."""
    rows = [f"- {citation_element_name(c)} {c.location}" for c in citations]
    return "**Sources**\n" + "\n".join(rows)


def match_element_name(citation: Citation) -> str:
    """Side-panel name for a "closest match" (distinct from answer citations)."""
    return f"match {citation.index}"


def closest_matches_footer(matches: Sequence[Citation]) -> str:
    """Shown under a refusal: what the knowledge base *does* contain near the question."""
    rows = [f"- {match_element_name(m)}: {m.location}" for m in matches]
    return "**Closest matches in your documents**\n" + "\n".join(rows)


def steps_markdown(steps: Sequence[AgentStep]) -> str:
    """Compact pipeline trace, one line per agent step."""
    return "\n".join(f"{i}. {s.label} ({s.ms:.0f} ms)" for i, s in enumerate(steps, start=1))


def welcome_markdown(collection: str, documents: Sequence[DocumentRecord]) -> str:
    """First message of a chat: what's indexed and how to start."""
    if not documents:
        return (
            f"👋 Welcome to **NexusRAG**. The knowledge base **{collection}** is empty.\n\n"
            "Index the sample documents with `python -m nexusrag.ingest data/`, "
            "then start a new chat."
        )
    listing = "\n".join(f"- {d.title} (`{d.filename}`)" for d in documents[:12])
    more = f"\n- …and {len(documents) - 12} more" if len(documents) > 12 else ""
    chunks = sum(d.num_chunks for d in documents)
    return (
        f"👋 Welcome to **NexusRAG**. Knowledge base **{collection}**: "
        f"{len(documents)} documents, {chunks} chunks.\n\n{listing}{more}\n\n"
        "Ask a question and every answer will cite its sources; click a `[n]` marker "
        "to see the exact passage."
    )


def setup_error_markdown(message: str) -> str:
    return (
        f"⚠️ {message}\n\nAdd `GEMINI_API_KEY` to your `.env` file (see `.env.example`) "
        "and restart the app."
    )
