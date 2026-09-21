"""Live smoke test of the Gemini configuration.

Makes a handful of real, cheap API calls (fast model, structured output, streaming
with the main model, embeddings) to confirm the API key and model names in `.env`
are valid. Costs a fraction of a cent. Not run in CI.

Usage:
    python scripts/smoke_gemini.py
"""

from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel

from nexusrag.config import get_settings
from nexusrag.llm.gemini_client import GeminiClient, GeminiError
from nexusrag.llm.usage import track_usage
from nexusrag.log import configure_logging, request_context


class SentenceFacts(BaseModel):
    topic: str
    word_count: int


async def run() -> None:
    settings = get_settings()
    configure_logging(settings)
    client = GeminiClient(settings)

    with request_context(stage="smoke"), track_usage() as usage:
        pong = await client.generate("Reply with exactly the word: pong", stage="smoke.fast")
        print(f"\n[fast · {pong.model}] {pong.text.strip()}")

        facts = await client.generate_structured(
            "Describe this sentence: 'Hybrid retrieval fuses dense vectors with BM25.'",
            SentenceFacts,
            stage="smoke.structured",
        )
        print(f"[structured] {facts.model_dump()}")

        print(f"[stream · {settings.generation_model}] ", end="", flush=True)
        async for delta in client.stream(
            "In one sentence, why do RAG systems rerank retrieved passages?", stage="smoke.stream"
        ):
            print(delta, end="", flush=True)
        print()

        query = await client.embed_query("What does BM25 measure?")
        docs = await client.embed_documents(
            [
                "BM25 ranks documents by term frequency and inverse document frequency.",
                "The Eiffel Tower is in Paris.",
            ],
            titles=["Lexical search", None],
        )
        sims = [sum(q * d for q, d in zip(query, doc, strict=True)) for doc in docs]
        print(
            f"[embed · {settings.embedding_model}] dim={len(query)} "
            f"cos(relevant)={sims[0]:.3f} cos(irrelevant)={sims[1]:.3f}"
        )

    totals = usage.totals()
    print(
        f"\nOK: {totals.calls} calls, {totals.prompt_tokens} prompt + {totals.output_tokens} output"
        f" + {totals.thoughts_tokens} thinking tokens, ~{totals.embedding_tokens} embedding tokens,"
        f" est. cost ${totals.cost_usd:.5f}"
    )


def main() -> int:
    # Windows consoles default to cp1252; never crash on a model's Unicode output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        asyncio.run(run())
    except GeminiError as exc:
        print(f"\nFAILED: {exc.user_message}\n  detail: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
