"""Intent routing: decide what kind of request a message is before doing any retrieval.

One fast-model call with a JSON schema returns three things:

* the **route** (chitchat, doc_qa, summarize_document, compare_documents, out_of_scope);
* the **documents** the user means, as numbers from a catalog of the knowledge base, so
  "summarise the Borealis sheet" resolves to a doc_id. Documents the user just uploaded
  are marked in the catalog, so "what is this file about?" resolves to them;
* the **standalone question**, with follow-ups rewritten using the history. Retrieval
  can then skip its own condensation step, which saves a call per turn.

Obvious greetings and thanks skip the LLM entirely, and any routing failure defaults to
``doc_qa``. The worst case of that is a grounded refusal, never an ungrounded answer.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.llm.prompts import ROUTER_PROMPT, render
from nexusrag.log import get_logger
from nexusrag.models import ChatTurn, Route
from nexusrag.retrieval.query_transform import format_history
from nexusrag.store.registry import DocumentRecord

log = get_logger(__name__)

#: Catalog entries shown to the router; very large knowledge bases are truncated.
MAX_CATALOG_ENTRIES = 100
RECENT_MARK = "[just uploaded]"
#: Appended to the "Document catalog:" heading when some documents were just uploaded.
RECENT_NOTE = (
    f"\n(documents marked {RECENT_MARK} were added moments ago: 'this', 'this document',"
    " 'this file', 'this project' and a bare 'it' refer to them unless the conversation"
    " says otherwise)"
)

_GREETING_RE = re.compile(
    r"^\s*("
    r"(hi|hello|hey|hiya|yo)( there| all| everyone| nexusrag)?"
    r"|good (morning|afternoon|evening)"
    r"|thanks( a lot| so much)?|thank you( so much)?|thx|cheers"
    r"|ok(ay)?|cool|great|bye|goodbye|see you"
    r")[\s!.,:)]*$",
    re.IGNORECASE,
)


class _RouterOutput(BaseModel):
    route: Literal["doc_qa", "summarize_document", "compare_documents", "chitchat", "out_of_scope"]
    documents: list[int] = Field(default_factory=list)
    standalone_question: str
    reason: str = ""


@dataclass
class RouteDecision:
    """Where a message goes, which documents it targets, and its standalone form."""

    route: Route
    standalone_question: str
    doc_ids: list[str] = field(default_factory=list)
    reason: str = ""
    #: False when the decision came from a shortcut or a fallback rather than the model.
    from_model: bool = True


def catalog_order(
    documents: Sequence[DocumentRecord], recent_doc_ids: Collection[str] = ()
) -> list[DocumentRecord]:
    """Documents in catalog order: just-uploaded ones first, so truncation never hides them."""
    recent = [d for d in documents if d.doc_id in recent_doc_ids]
    return recent + [d for d in documents if d.doc_id not in recent_doc_ids]


def format_catalog(
    documents: Sequence[DocumentRecord], recent_doc_ids: Collection[str] = ()
) -> str:
    """Numbered list of documents for the router prompt."""
    if not documents:
        return "(the knowledge base is empty)"
    lines = [
        f"{i}. {doc.title} ({doc.filename})"
        + (f" {RECENT_MARK}" if doc.doc_id in recent_doc_ids else "")
        for i, doc in enumerate(documents[:MAX_CATALOG_ENTRIES], start=1)
    ]
    if len(documents) > MAX_CATALOG_ENTRIES:
        lines.append(f"... and {len(documents) - MAX_CATALOG_ENTRIES} more")
    return "\n".join(lines)


def is_greeting(message: str) -> bool:
    return bool(_GREETING_RE.match(message))


class Router:
    """Classifies messages with the fast model."""

    def __init__(self, gemini: GeminiClient) -> None:
        self.gemini = gemini

    async def route(
        self,
        message: str,
        *,
        documents: Sequence[DocumentRecord],
        history: Sequence[ChatTurn] = (),
        recent_doc_ids: Collection[str] = (),
    ) -> RouteDecision:
        """Route ``message`` given the catalog, the conversation and any fresh uploads."""
        if is_greeting(message):
            return RouteDecision(Route.CHITCHAT, message, reason="greeting", from_model=False)
        recent = {d.doc_id for d in documents if d.doc_id in recent_doc_ids}
        catalog = catalog_order(documents, recent)[:MAX_CATALOG_ENTRIES]
        prompt = render(
            ROUTER_PROMPT,
            recent_note=RECENT_NOTE if recent else "",
            catalog=format_catalog(catalog, recent),
            history=format_history(history) or "(no previous messages)",
            message=message,
        )
        try:
            output = await self.gemini.generate_structured(
                prompt, _RouterOutput, role="fast", stage="route"
            )
        except GeminiError as exc:
            log.warning("agent.route_failed", error_type=type(exc).__name__)
            return RouteDecision(
                Route.DOC_QA, message, reason="router unavailable", from_model=False
            )

        doc_ids = list(
            dict.fromkeys(catalog[n - 1].doc_id for n in output.documents if 1 <= n <= len(catalog))
        )
        decision = RouteDecision(
            route=Route(output.route),
            standalone_question=output.standalone_question.strip() or message,
            doc_ids=doc_ids,
            reason=output.reason.strip(),
        )
        log.info(
            "agent.route",
            route=decision.route.value,
            documents=len(doc_ids),
            condensed=decision.standalone_question != message,
        )
        return decision
