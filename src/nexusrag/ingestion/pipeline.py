"""Ingestion orchestration: load -> chunk -> dedupe -> embed -> store.

Incremental indexing:

* **Unchanged sources are skipped.** A document whose ``content_hash`` matches its
  registry row is already indexed.
* **Changed sources are replaced**, and their stale chunks are removed from Chroma, BM25
  and the parent store.
* **Unchanged chunks keep their vectors.** When a changed document is re-indexed,
  chunks whose exact embedding input is unchanged reuse their existing vectors, so a
  one-paragraph edit doesn't re-embed the whole file.
* **Pruning** (optional, for directory ingests) removes documents whose files are gone.

Crash safety: Chroma can't join the SQLite transaction, so the order matters.

1. Embed everything first (no side effects).
2. Replace the document's vectors in Chroma.
3. In one SQLite transaction: replace parents and BM25 rows, upsert the registry row
   and bump the collection version.

The registry row is the commit marker. If the process dies between steps 2 and 3,
the registry still holds the old hash, so the next run re-indexes the document and
converges.
"""

from __future__ import annotations

import asyncio
import inspect
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nexusrag.config import Settings
from nexusrag.ingestion.chunker import (
    ChunkedDocument,
    ChunkingConfig,
    chunk_document,
    contextual_text,
    keyword_text,
)
from nexusrag.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    LoaderError,
    document_from_html,
    fetch_url,
    load_file,
)
from nexusrag.ingestion.metadata import short_hash
from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.log import get_logger
from nexusrag.models import Document, SourceType, utcnow
from nexusrag.store import Stores
from nexusrag.store.registry import DocumentRecord, InvalidCollectionName, validate_collection_name
from nexusrag.store.vector_store import EmbeddingMismatchError

log = get_logger(__name__)

# Failures with messages that are safe and useful to show users as-is.
EXPECTED_ERRORS = (LoaderError, GeminiError, EmbeddingMismatchError, InvalidCollectionName)

Status = Literal["indexed", "updated", "skipped", "failed", "deleted"]
Stage = Literal["parsing", "chunking", "embedding", "storing", "done", "skipped", "failed"]


@dataclass(frozen=True)
class ProgressEvent:
    """Emitted as each file moves through the pipeline (drives the UI progress messages)."""

    filename: str
    stage: Stage
    detail: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], Awaitable[None] | None]


@dataclass
class IngestResult:
    """Outcome for one source."""

    filename: str
    status: Status
    doc_id: str | None = None
    title: str | None = None
    num_parents: int = 0
    num_chunks: int = 0
    reused_embeddings: int = 0
    error: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


def embed_hash(chunk_text: str, title: str, model: str, dim: int) -> str:
    """Identity of an embedding input, so identical inputs can reuse stored vectors."""
    return short_hash(f"{model}|{dim}|{title}|{chunk_text}")


def iter_source_files(directory: Path) -> list[Path]:
    """Supported files under ``directory`` (recursive, hidden files skipped), sorted."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


class IngestionPipeline:
    """Indexes files and URLs into a knowledge base."""

    def __init__(self, settings: Settings, gemini: GeminiClient, stores: Stores) -> None:
        self.settings = settings
        self.gemini = gemini
        self.stores = stores
        self.chunking = ChunkingConfig.from_settings(settings)

    # ------------------------------------------------------------------ public API

    async def ingest_file(
        self,
        path: Path,
        *,
        collection: str,
        filename: str | None = None,
        source_key: str | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestResult:
        """Index one file. ``filename`` overrides the display name (e.g. for uploads)."""
        name = filename or path.name
        timings: dict[str, float] = {}
        try:
            await _emit(progress, ProgressEvent(name, "parsing"))
            started = time.perf_counter()
            document = await asyncio.to_thread(
                load_file, path, filename=name, source_key=source_key
            )
            timings["parse"] = _ms(started)
            result = await self._index(
                document, collection, force=force, progress=progress, timings=timings
            )
            if document.source_type == SourceType.PDF:
                kb = validate_collection_name(collection)
                changed = result.status in ("indexed", "updated")
                # Unchanged PDFs indexed before copies were kept get one now.
                missing = result.status == "skipped" and not self._kept(kb, document.doc_id)
                if changed or missing:
                    await asyncio.to_thread(self._keep_pdf, path, kb, document.doc_id)
            return result
        except EXPECTED_ERRORS as exc:
            return await self._failed(name, exc, progress, timings)
        except Exception as exc:
            log.exception("ingest.unexpected_error", filename=name)
            return await self._failed(name, exc, progress, timings)

    async def ingest_url(
        self,
        url: str,
        *,
        collection: str,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestResult:
        """Fetch a web page (SSRF-guarded), extract its main content and index it."""
        timings: dict[str, float] = {}
        try:
            await _emit(progress, ProgressEvent(url, "parsing"))
            started = time.perf_counter()
            html, final_url = await fetch_url(
                url,
                timeout_s=self.settings.llm_timeout_s,
                max_bytes=self.settings.max_upload_mb * 1024 * 1024,
            )
            document = await asyncio.to_thread(document_from_html, html, final_url)
            timings["parse"] = _ms(started)
            return await self._index(
                document, collection, force=force, progress=progress, timings=timings
            )
        except EXPECTED_ERRORS as exc:
            return await self._failed(url, exc, progress, timings)
        except Exception as exc:
            log.exception("ingest.unexpected_error", source_type="url")
            return await self._failed(url, exc, progress, timings)

    async def ingest_directory(
        self,
        directory: Path,
        *,
        collection: str,
        force: bool = False,
        prune: bool = False,
        progress: ProgressCallback | None = None,
    ) -> list[IngestResult]:
        """Index every supported file under ``directory``.

        Files are keyed by their path relative to ``directory``. With ``prune``, file-based
        documents in the collection that no longer exist on disk are deleted.
        """
        files = iter_source_files(directory)
        results = []
        for path in files:
            key = path.relative_to(directory).as_posix()
            results.append(
                await self.ingest_file(
                    path,
                    collection=collection,
                    source_key=key,
                    filename=key,
                    force=force,
                    progress=progress,
                )
            )
        if prune:
            present = {path.relative_to(directory).as_posix() for path in files}
            collection = validate_collection_name(collection)
            for record in self.stores.registry.list_documents(collection):
                if record.source_type != SourceType.URL and record.source not in present:
                    await self.delete_document(collection, record.doc_id)
                    results.append(
                        IngestResult(
                            record.filename, "deleted", doc_id=record.doc_id, title=record.title
                        )
                    )
        return results

    def _kept(self, collection: str, doc_id: str) -> bool:
        return self.settings.source_file_path(collection, doc_id).is_file()

    def _keep_pdf(self, path: Path, collection: str, doc_id: str) -> None:
        """Keep a copy of an ingested PDF so the UI can open it at the cited page."""
        target = self.settings.source_file_path(collection, doc_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    async def delete_document(self, collection: str, doc_id: str) -> bool:
        """Remove a document from every store. Returns False if it wasn't indexed."""
        collection = validate_collection_name(collection)
        stores = self.stores
        if stores.registry.get_document(collection, doc_id) is None:
            return False
        await asyncio.to_thread(stores.vectors.delete_document, collection, doc_id)
        with stores.db.transaction():
            stores.parents.delete_document(collection, doc_id)
            stores.bm25.delete_document(collection, doc_id)
            stores.registry.delete_document(collection, doc_id)
            stores.registry.bump_version(collection)
            stores.cache.invalidate(collection)  # cached answers may cite this document
        self.settings.source_file_path(collection, doc_id).unlink(missing_ok=True)
        log.info("ingest.deleted", collection=collection, doc_id=doc_id)
        return True

    # ------------------------------------------------------------------ core

    async def _index(
        self,
        document: Document,
        collection: str,
        *,
        force: bool,
        progress: ProgressCallback | None,
        timings: dict[str, float],
    ) -> IngestResult:
        stores = self.stores
        collection = stores.registry.ensure_collection(collection)
        name = document.filename
        existing = stores.registry.get_document(collection, document.doc_id)

        if existing is not None and existing.content_hash == document.content_hash and not force:
            await _emit(progress, ProgressEvent(name, "skipped", {"reason": "unchanged"}))
            log.info("ingest.skipped", collection=collection, doc_id=document.doc_id, filename=name)
            return IngestResult(
                name,
                "skipped",
                doc_id=document.doc_id,
                title=document.title,
                num_parents=existing.num_parents,
                num_chunks=existing.num_chunks,
                timings_ms=timings,
            )

        await _emit(progress, ProgressEvent(name, "chunking"))
        started = time.perf_counter()
        chunked: ChunkedDocument = await asyncio.to_thread(
            chunk_document, document, collection, self.chunking
        )
        timings["chunk"] = _ms(started)
        if not chunked.chunks:
            raise LoaderError(f"No indexable text found in {name}.")

        await _emit(
            progress,
            ProgressEvent(
                name, "embedding", {"parents": len(chunked.parents), "chunks": len(chunked.chunks)}
            ),
        )
        started = time.perf_counter()
        vectors, hashes, reused = await self._embed(
            document, chunked, collection, existing is not None
        )
        timings["embed"] = _ms(started)

        await _emit(progress, ProgressEvent(name, "storing"))
        started = time.perf_counter()
        await asyncio.to_thread(self._replace, collection, document, chunked, vectors, hashes)
        timings["store"] = _ms(started)

        status: Status = "updated" if existing is not None else "indexed"
        result = IngestResult(
            name,
            status,
            doc_id=document.doc_id,
            title=document.title,
            num_parents=len(chunked.parents),
            num_chunks=len(chunked.chunks),
            reused_embeddings=reused,
            timings_ms=timings,
        )
        await _emit(
            progress,
            ProgressEvent(
                name,
                "done",
                {
                    "status": status,
                    "parents": result.num_parents,
                    "chunks": result.num_chunks,
                    "reused": reused,
                },
            ),
        )
        log.info(
            "ingest.document",
            collection=collection,
            doc_id=document.doc_id,
            filename=name,
            status=status,
            parents=result.num_parents,
            chunks=result.num_chunks,
            reused_embeddings=reused,
            **{f"{k}_ms": round(v, 1) for k, v in timings.items()},
        )
        return result

    async def _embed(
        self, document: Document, chunked: ChunkedDocument, collection: str, has_previous: bool
    ) -> tuple[list[list[float]], list[str], int]:
        """Embed chunks, reusing vectors from the previous version where the input is identical."""
        s = self.settings
        texts = [contextual_text(c) for c in chunked.chunks]
        hashes = [embed_hash(t, document.title, s.embedding_model, s.embedding_dim) for t in texts]
        reusable: dict[str, list[float]] = {}
        if has_previous:
            reusable = await asyncio.to_thread(
                self.stores.vectors.reusable_embeddings, collection, document.doc_id
            )
        missing = [i for i, h in enumerate(hashes) if h not in reusable]
        fresh = await self.gemini.embed_documents(
            [texts[i] for i in missing], titles=[document.title] * len(missing)
        )
        by_index = dict(zip(missing, fresh, strict=True))
        vectors = [by_index[i] if i in by_index else reusable[h] for i, h in enumerate(hashes)]
        return vectors, hashes, len(hashes) - len(missing)

    def _replace(
        self,
        collection: str,
        document: Document,
        chunked: ChunkedDocument,
        vectors: Sequence[Sequence[float]],
        hashes: Sequence[str],
    ) -> None:
        """Swap the stored version of a document for the new one (see module docstring)."""
        stores = self.stores
        stores.vectors.delete_document(collection, document.doc_id)
        stores.vectors.upsert(collection, chunked.chunks, vectors, hashes)
        with stores.db.transaction():
            stores.parents.delete_document(collection, document.doc_id)
            stores.bm25.delete_document(collection, document.doc_id)
            stores.parents.add_many(chunked.parents)
            stores.bm25.add(collection, chunked.chunks, [keyword_text(c) for c in chunked.chunks])
            stores.registry.upsert_document(
                DocumentRecord(
                    collection=collection,
                    doc_id=document.doc_id,
                    source=document.source,
                    filename=document.filename,
                    source_type=document.source_type,
                    title=document.title,
                    content_hash=document.content_hash,
                    num_parents=len(chunked.parents),
                    num_chunks=len(chunked.chunks),
                    ingested_at=utcnow(),
                )
            )
            stores.registry.bump_version(collection)
            # Same transaction: no answer can be served from before this document changed.
            stores.cache.invalidate(collection)

    async def _failed(
        self,
        name: str,
        exc: Exception,
        progress: ProgressCallback | None,
        timings: dict[str, float],
    ) -> IngestResult:
        if isinstance(exc, GeminiError):
            message = exc.user_message
        elif isinstance(exc, EXPECTED_ERRORS):
            message = str(exc)
        else:
            message = "Unexpected error while processing this file."
        log.warning("ingest.failed", filename=name, error_type=type(exc).__name__, detail=message)
        await _emit(progress, ProgressEvent(name, "failed", {"error": message}))
        return IngestResult(name, "failed", error=message, timings_ms=timings)


async def _emit(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
