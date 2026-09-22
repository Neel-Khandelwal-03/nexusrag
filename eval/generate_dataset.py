"""Draft the evaluation dataset from the indexed documents, for review by hand afterwards.

    python -m eval.generate_dataset                    # about 40 questions -> eval/dataset.jsonl
    python -m eval.generate_dataset --per-doc 4 --cross 2 --unanswerable 5 --out draft.jsonl

* **Single-section questions.** Sections are sampled evenly from each document. The
  generation model writes a question, a reference answer and exact evidence quotes. Quotes
  are checked against the section text, and items whose quotes can't be found are dropped.
  The quotes are mapped to the child chunks that contain them: these are the ground-truth
  chunk IDs.
* **Cross-document questions** pair same-named sections of two documents (e.g. both
  products' "Performance"), so answering needs both.
* **Unanswerable questions** are written from the section headings only, and kept only if
  retrieval plus the relevance grader agree that the documents can't answer them.

Every item starts with ``"reviewed": false``. Read the file, fix or delete weak items, and
flip the flag.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from eval.dataset import DEFAULT_PATH, EvalItem, normalize, save_dataset
from nexusrag.agent.grader import RelevanceGrader
from nexusrag.config import get_settings
from nexusrag.llm.gemini_client import GeminiClient, GeminiError, GeminiQuotaExhaustedError
from nexusrag.llm.prompts import (
    EVAL_CROSS_QUESTION_PROMPT,
    EVAL_QUESTION_PROMPT,
    EVAL_UNANSWERABLE_PROMPT,
    render,
)
from nexusrag.log import configure_logging
from nexusrag.models import ParentSection
from nexusrag.retrieval.retriever import RetrievalOptions
from nexusrag.service import RAGService
from nexusrag.store import Stores
from nexusrag.store.registry import DocumentRecord

#: Sections shorter than this rarely hold a testable fact.
MIN_SECTION_TOKENS = 40
#: Passage text sent to the generator is capped to keep calls cheap.
MAX_SECTION_CHARS = 6000


class _Question(BaseModel):
    question: str
    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    kind: Literal["fact", "numeric", "table", "comparison", "unanswerable"] = "fact"


class _Questions(BaseModel):
    questions: list[_Question] = Field(default_factory=list)


def pick_sections(parents: Sequence[ParentSection], n: int) -> list[ParentSection]:
    """Up to ``n`` substantial sections, spread evenly through the document."""
    usable = [p for p in parents if p.token_count >= MIN_SECTION_TOKENS]
    if len(usable) <= n:
        return list(usable)
    step = len(usable) / n
    return [usable[int(i * step)] for i in range(n)]


def heading(section_path: str) -> str:
    """Last heading of a section path without its number: "3 Performance" -> "performance"."""
    last = section_path.split(">")[-1]
    return re.sub(r"^[\d.\s]+", "", last).strip().lower()


def valid_quotes(quotes: Sequence[str], parents: Sequence[ParentSection]) -> list[str]:
    """Quotes that really occur in one of ``parents`` (after normalisation)."""
    texts = [normalize(p.text) for p in parents]
    return [q for q in quotes if q.strip() and any(normalize(q) in t for t in texts)]


def chunks_with_quotes(
    stores: Stores, collection: str, parent: ParentSection, quotes: Sequence[str]
) -> list[str]:
    """Child chunks of ``parent`` that contain any quote (all children if none match)."""
    chunks = stores.vectors.chunks_for_parent(collection, parent.parent_id)
    hits = [c.chunk_id for c in chunks if any(normalize(q) in normalize(c.text) for q in quotes)]
    return hits or [c.chunk_id for c in chunks]


class DatasetGenerator:
    def __init__(self, service: RAGService, collection: str, delay_s: float = 1.0) -> None:
        self.service = service
        self.gemini: GeminiClient = service.gemini
        self.stores = service.stores
        self.collection = collection
        self.delay_s = delay_s
        #: Set when a daily quota runs out: stop asking and keep what was written.
        self.exhausted = False

    async def _ask(self, prompt: str, stage: str) -> list[_Question]:
        if self.exhausted:
            return []
        try:
            out = await self.gemini.generate_structured(
                prompt, _Questions, role="main", stage=stage
            )
        except GeminiQuotaExhaustedError as exc:
            print(f"  stopping: {exc.user_message}", file=sys.stderr)
            self.exhausted = True
            return []
        except GeminiError as exc:
            print(f"  skipped: {exc.user_message}", file=sys.stderr)
            return []
        finally:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
        return out.questions

    async def single_section(self, doc: DocumentRecord, per_doc: int) -> list[EvalItem]:
        parents = self.stores.parents.for_document(self.collection, doc.doc_id)
        items = []
        for parent in pick_sections(parents, per_doc):
            questions = await self._ask(
                render(
                    EVAL_QUESTION_PROMPT,
                    n=1,
                    title=doc.title,
                    filename=doc.filename,
                    section=parent.section_path or "(untitled)",
                    text=parent.text[:MAX_SECTION_CHARS],
                ),
                stage="eval.generate",
            )
            for q in questions[:1]:
                quotes = valid_quotes(q.evidence, [parent])
                if not quotes or not q.answer.strip():
                    print(f"  dropped (evidence not found): {q.question}", file=sys.stderr)
                    continue
                items.append(
                    EvalItem(
                        id="",
                        question=q.question.strip(),
                        answer=q.answer.strip(),
                        kind=q.kind if q.kind in ("fact", "numeric", "table") else "fact",
                        documents=[doc.filename],
                        source_parent_ids=[parent.parent_id],
                        source_chunk_ids=chunks_with_quotes(
                            self.stores, self.collection, parent, quotes
                        ),
                        evidence=quotes,
                    )
                )
        return items

    async def cross_document(self, docs: Sequence[DocumentRecord], n: int) -> list[EvalItem]:
        """Questions that need same-named sections from two different documents."""
        sections: dict[str, list[tuple[DocumentRecord, ParentSection]]] = {}
        for doc in docs:
            for parent in self.stores.parents.for_document(self.collection, doc.doc_id):
                if parent.token_count >= MIN_SECTION_TOKENS and heading(parent.section_path):
                    sections.setdefault(heading(parent.section_path), []).append((doc, parent))
        pairs = [
            (group[0], group[1])
            for group in sections.values()
            if len(group) >= 2 and group[0][0].doc_id != group[1][0].doc_id
        ]
        items = []
        for (doc_a, a), (doc_b, b) in pairs[:n]:
            questions = await self._ask(
                render(
                    EVAL_CROSS_QUESTION_PROMPT,
                    title_a=doc_a.title,
                    section_a=a.section_path,
                    text_a=a.text[: MAX_SECTION_CHARS // 2],
                    title_b=doc_b.title,
                    section_b=b.section_path,
                    text_b=b.text[: MAX_SECTION_CHARS // 2],
                ),
                stage="eval.generate_cross",
            )
            for q in questions[:1]:
                quotes_a, quotes_b = valid_quotes(q.evidence, [a]), valid_quotes(q.evidence, [b])
                if not quotes_a or not quotes_b or not q.answer.strip():
                    print(f"  dropped (needs evidence from both): {q.question}", file=sys.stderr)
                    continue
                items.append(
                    EvalItem(
                        id="",
                        question=q.question.strip(),
                        answer=q.answer.strip(),
                        kind="comparison",
                        documents=[doc_a.filename, doc_b.filename],
                        source_parent_ids=[a.parent_id, b.parent_id],
                        source_chunk_ids=chunks_with_quotes(
                            self.stores, self.collection, a, quotes_a
                        )
                        + chunks_with_quotes(self.stores, self.collection, b, quotes_b),
                        evidence=quotes_a + quotes_b,
                    )
                )
        return items

    async def unanswerable(self, docs: Sequence[DocumentRecord], n: int) -> list[EvalItem]:
        """Plausible questions the documents don't answer, verified by retrieval + grading."""
        outline = []
        for doc in docs:
            parents = self.stores.parents.for_document(self.collection, doc.doc_id)
            headings = sorted({p.section_path for p in parents if p.section_path})
            outline.append(f"- {doc.title}: " + "; ".join(headings))
        candidates = await self._ask(
            render(EVAL_UNANSWERABLE_PROMPT, n=n * 2, outline="\n".join(outline)),
            stage="eval.generate_unanswerable",
        )
        grader = RelevanceGrader(self.gemini)
        # No reranking: verification only needs to know whether anything relevant exists.
        options = replace(RetrievalOptions.from_settings(self.service.settings), rerank=False)
        items: list[EvalItem] = []
        for q in candidates:
            if len(items) >= n:
                break
            retrieval = await self.service.retriever.retrieve(
                q.question, collection=self.collection, options=options
            )
            verdict = await grader.grade(q.question, retrieval.passages)
            if verdict.sufficient or not verdict.checked:
                print(f"  dropped (documents may answer it): {q.question}", file=sys.stderr)
                continue
            items.append(EvalItem(id="", question=q.question.strip(), answerable=False,
                                  kind="unanswerable"))  # fmt: skip
        return items


def finalize(items: Sequence[EvalItem]) -> list[EvalItem]:
    """Drop duplicate questions and number the rest q001, q002, ..."""
    seen: set[str] = set()
    out: list[EvalItem] = []
    for item in items:
        key = normalize(item.question)
        if key in seen:
            continue
        seen.add(key)
        out.append(item.model_copy(update={"id": f"q{len(out) + 1:03d}"}))
    return out


async def generate(
    service: RAGService, collection: str, *, per_doc: int, cross: int, unanswerable: int,
    delay_s: float,
) -> list[EvalItem]:  # fmt: skip
    generator = DatasetGenerator(service, collection, delay_s=delay_s)
    docs = service.stores.registry.list_documents(collection)
    if not docs:
        raise SystemExit(f"Knowledge base '{collection}' is empty: ingest documents first.")
    items: list[EvalItem] = []
    for doc in docs:
        print(f"{doc.filename}: writing {per_doc} questions")
        items += await generator.single_section(doc, per_doc)
    print(f"cross-document: writing {cross} questions")
    items += await generator.cross_document(docs, cross)
    print(f"unanswerable: writing and verifying {unanswerable} questions")
    items += await generator.unanswerable(docs, unanswerable)
    if generator.exhausted:
        print("warning: a daily quota ran out, so the dataset is incomplete", file=sys.stderr)
    return finalize(items)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", default=None)
    parser.add_argument("--per-doc", type=int, default=6)
    parser.add_argument("--cross", type=int, default=4)
    parser.add_argument("--unanswerable", type=int, default=8)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between model calls")
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--force", action="store_true", help="overwrite an existing dataset")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        parser.error(f"{args.out} exists (it may contain reviewed items); pass --force")

    settings = get_settings().model_copy(update={"log_level": "WARNING"})
    configure_logging(settings)
    service = RAGService.create(settings)
    try:
        items = asyncio.run(
            generate(
                service,
                args.collection or settings.default_collection,
                per_doc=args.per_doc,
                cross=args.cross,
                unanswerable=args.unanswerable,
                delay_s=args.delay,
            )
        )
    finally:
        service.close()
    save_dataset(items, args.out)
    answerable = sum(item.answerable for item in items)
    print(
        f"\nWrote {len(items)} questions ({answerable} answerable, "
        f"{len(items) - answerable} unanswerable) to {args.out}. Review them before evaluating."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
