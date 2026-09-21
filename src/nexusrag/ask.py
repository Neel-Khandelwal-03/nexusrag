"""Ask a question from the terminal (handy for testing without the UI).

Examples::

    python -m nexusrag.ask "How long does the Aurora X1 battery last?"
    python -m nexusrag.ask "Summarise the remote work stipends" --style concise --show-context
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from nexusrag.config import get_settings
from nexusrag.llm.gemini_client import GeminiError
from nexusrag.log import configure_logging
from nexusrag.models import Route
from nexusrag.service import RAGService
from nexusrag.store.registry import InvalidCollectionName


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nexusrag.ask", description="Ask your knowledge base a question."
    )
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("-c", "--collection", help="Knowledge base (default: DEFAULT_COLLECTION)")
    parser.add_argument("--style", choices=["concise", "detailed"], default="detailed")
    parser.add_argument(
        "--show-context", action="store_true", help="Print the agent steps and context passages"
    )
    parser.add_argument(
        "--mode", choices=["summarize", "compare"], help="Force a summary or comparison"
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    service = RAGService.create(settings)
    try:

        async def on_token(token: str) -> None:
            print(token, end="", flush=True)

        async def on_reset() -> None:
            print("\n[discarding the previous draft and regenerating]\n")

        result = await service.ask(
            args.question,
            collection=args.collection,
            style=args.style,
            on_token=on_token,
            on_reset=on_reset,
            mode={"summarize": Route.SUMMARIZE, "compare": Route.COMPARE}.get(args.mode or ""),
        )
        answer = result.answer
        if answer.citations:
            print("\n\nSources:")
            for citation in answer.citations:
                print(f"  [{citation.index}] {citation.location}")
        if answer.closest_matches:
            print("\n\nClosest matches in your documents:")
            for match in answer.closest_matches:
                print(f"  - {match.location}")
        if args.show_context:
            print("\nAgent steps:")
            for i, step in enumerate(answer.steps, start=1):
                print(f"  {i}. {step.label} ({step.ms:.0f} ms)")
            if result.passages:
                print("Context passages:")
            for passage in result.passages:
                best = passage.chunks[0].scores if passage.chunks else None
                parts = []
                if best is not None and best.rrf_score is not None:
                    parts.append(f"rrf={best.rrf_score:.4f}")
                if best is not None and best.rerank_score is not None:
                    parts.append(f"rerank={best.rerank_score:.3f}")
                label = " ".join(parts)
                print(f"  [{passage.index}] {label}  {passage.to_citation().location}")
        u = answer.usage
        total_s = sum(t.ms for t in answer.timings) / 1000
        print(
            f"\n(route {answer.route.value}; {total_s:.1f}s; {u.calls} model calls; "
            f"{u.prompt_tokens}+{u.output_tokens} tokens, est. ${u.cost_usd:.5f})"
        )
        return 0
    finally:
        service.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    if settings.log_level.upper() != "DEBUG":
        settings = settings.model_copy(update={"log_level": "WARNING"})
    configure_logging(settings)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        return asyncio.run(_run(args))
    except (GeminiError, InvalidCollectionName) as exc:
        message = exc.user_message if isinstance(exc, GeminiError) else str(exc)
        print(f"\nError: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
