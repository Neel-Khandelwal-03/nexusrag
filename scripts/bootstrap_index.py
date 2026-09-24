"""Index the sample documents if the knowledge base is empty (run at container start).

    python scripts/bootstrap_index.py                 # index DATA_DIR if nothing is indexed
    python scripts/bootstrap_index.py --force         # re-index even when documents exist
    python scripts/bootstrap_index.py --require       # fail (exit 1) if indexing fails

By default a failure only logs a warning and exits 0: a container should still start and
explain itself in the chat (a missing API key, say) rather than crash-looping.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from nexusrag.config import get_settings
from nexusrag.ingestion.pipeline import IngestionPipeline
from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.log import configure_logging, get_logger
from nexusrag.store import Stores

log = get_logger("nexusrag.bootstrap")


async def bootstrap(data_dir: Path, collection: str, *, force: bool) -> int:
    """Index ``data_dir`` into ``collection``; returns the number of documents indexed."""
    settings = get_settings()
    stores = Stores.open(settings)
    try:
        existing = stores.registry.list_documents(collection)
        if existing and not force:
            log.info("bootstrap.skipped", collection=collection, documents=len(existing))
            return 0
        if not await asyncio.to_thread(data_dir.is_dir):
            log.warning("bootstrap.no_data_dir", path=str(data_dir))
            return 0
        pipeline = IngestionPipeline(settings, GeminiClient(settings), stores)
        results = await pipeline.ingest_directory(data_dir, collection=collection, force=force)
        indexed = sum(r.status in ("indexed", "updated") for r in results)
        failed = [r.filename for r in results if r.status == "failed"]
        log.info("bootstrap.done", collection=collection, indexed=indexed, failed=len(failed))
        if failed:
            raise RuntimeError(f"could not index: {', '.join(failed)}")
        return indexed
    finally:
        stores.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="default: DATA_DIR")
    parser.add_argument("--collection", default=None, help="default: DEFAULT_COLLECTION")
    parser.add_argument("--force", action="store_true", help="re-index even if not empty")
    parser.add_argument("--require", action="store_true", help="exit 1 if indexing fails")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    try:
        indexed = asyncio.run(
            bootstrap(
                args.data_dir or settings.data_dir,
                args.collection or settings.default_collection,
                force=args.force,
            )
        )
    except (GeminiError, RuntimeError, OSError) as exc:
        # The app can still start and explain the problem in the chat.
        log.warning("bootstrap.failed", error_type=type(exc).__name__, detail=str(exc)[:200])
        return 1 if args.require else 0
    print(f"Bootstrap: {indexed} document(s) indexed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
