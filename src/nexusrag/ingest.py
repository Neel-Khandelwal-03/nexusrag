"""Command-line ingestion.

Examples::

    python -m nexusrag.ingest data/                      # index a folder (incremental)
    python -m nexusrag.ingest report.pdf https://example.com/guide --collection research
    python -m nexusrag.ingest data/ --prune              # also drop docs whose files are gone
    python -m nexusrag.ingest --list                     # show indexed documents
    python -m nexusrag.ingest --delete <doc_id>

Don't run a write command while the chat app is serving the same storage directory:
ChromaDB's local mode isn't designed for concurrent writers across processes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

from nexusrag.config import Settings, get_settings
from nexusrag.ingestion.pipeline import IngestionPipeline, IngestResult, ProgressEvent
from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.llm.usage import track_usage
from nexusrag.log import configure_logging, request_context
from nexusrag.store import Stores
from nexusrag.store.registry import InvalidCollectionName, validate_collection_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nexusrag.ingest",
        description="Index files, folders and web pages into a NexusRAG knowledge base.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="Files, directories or http(s) URLs (default: DATA_DIR when nothing else is asked)",
    )
    parser.add_argument(
        "-c", "--collection", help="Knowledge base name (default: DEFAULT_COLLECTION)"
    )
    parser.add_argument("--force", action="store_true", help="Re-index even if unchanged")
    parser.add_argument(
        "--prune", action="store_true", help="Remove documents whose files are gone"
    )
    parser.add_argument("--list", action="store_true", help="List indexed documents and exit")
    parser.add_argument(
        "--list-collections", action="store_true", help="List knowledge bases and exit"
    )
    parser.add_argument("--delete", metavar="DOC_ID", help="Delete a document and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print the summary")
    return parser


def _print_progress(event: ProgressEvent) -> None:
    if event.stage == "embedding":
        print(
            f"  {event.filename}: embedding {event.detail['chunks']} chunks "
            f"({event.detail['parents']} sections)"
        )
    elif event.stage in ("parsing", "chunking", "storing"):
        print(f"  {event.filename}: {event.stage}")


def _print_result(result: IngestResult) -> None:
    if result.status == "failed":
        print(f"[failed ] {result.filename}: {result.error}")
    elif result.status == "deleted":
        print(f"[deleted] {result.filename}")
    elif result.status == "skipped":
        print(f"[skipped] {result.filename} (unchanged)")
    else:
        total = sum(result.timings_ms.values()) / 1000
        reused = f", {result.reused_embeddings} vectors reused" if result.reused_embeddings else ""
        print(
            f"[{result.status}] {result.filename}: {result.num_parents} sections, "
            f"{result.num_chunks} chunks{reused} ({total:.1f}s)"
        )


def _list_documents(stores: Stores, collection: str) -> None:
    records = stores.registry.list_documents(collection)
    if not records:
        print(f"No documents in knowledge base {collection!r}.")
        return
    print(f"Knowledge base {collection!r}: {len(records)} documents")
    for r in records:
        print(
            f"  {r.doc_id}  {r.source_type.value:<8} {r.num_chunks:>5} chunks  {r.filename}  "
            f"({r.title})"
        )


def _list_collections(stores: Stores) -> None:
    infos = stores.registry.list_collections()
    if not infos:
        print("No knowledge bases yet.")
    for info in infos:
        print(
            f"  {info.name:<24} {info.num_documents:>4} docs {info.num_chunks:>6} chunks  "
            f"v{info.version}  updated {info.updated_at:%Y-%m-%d %H:%M}"
        )


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    collection = validate_collection_name(args.collection or settings.default_collection)
    stores = Stores.open(settings)
    try:
        if args.list_collections:
            _list_collections(stores)
            return 0
        if args.list:
            _list_documents(stores, collection)
            return 0

        pipeline = IngestionPipeline(settings, GeminiClient(settings), stores)
        if args.delete:
            deleted = await pipeline.delete_document(collection, args.delete)
            print("Deleted." if deleted else f"No document {args.delete!r} in {collection!r}.")
            return 0 if deleted else 1

        sources = args.sources or [str(settings.data_dir)]
        progress = None if args.quiet else _print_progress
        results: list[IngestResult] = []
        with request_context(command="ingest", collection=collection), track_usage() as usage:
            for source in sources:
                if source.startswith(("http://", "https://")):
                    batch = [
                        await pipeline.ingest_url(
                            source, collection=collection, force=args.force, progress=progress
                        )
                    ]
                elif await asyncio.to_thread(Path(source).is_dir):
                    batch = await pipeline.ingest_directory(
                        Path(source),
                        collection=collection,
                        force=args.force,
                        prune=args.prune,
                        progress=progress,
                    )
                else:
                    batch = [
                        await pipeline.ingest_file(
                            Path(source), collection=collection, force=args.force, progress=progress
                        )
                    ]
                for result in batch:
                    _print_result(result)
                results.extend(batch)

        counts = Counter(r.status for r in results)
        totals = usage.totals()
        summary = (
            ", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "nothing to do"
        )
        print(
            f"\nKnowledge base {collection!r}: {summary}. "
            f"Embedded ~{totals.embedding_tokens} tokens (est. ${totals.cost_usd:.4f})."
        )
        return 1 if counts.get("failed") else 0
    finally:
        stores.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    # The CLI prints its own progress lines; keep structured logs to warnings unless debugging.
    if settings.log_level.upper() != "DEBUG":
        settings = settings.model_copy(update={"log_level": "WARNING"})
    configure_logging(settings)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        return asyncio.run(_run(args, settings))
    except (GeminiError, InvalidCollectionName) as exc:
        message = exc.user_message if isinstance(exc, GeminiError) else str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
