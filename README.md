# NexusRAG

> Advanced Retrieval-Augmented Generation chatbot: hybrid search, reranking, a self-correcting agent, inline citations and a measurable evaluation suite, built on **Chainlit + ChromaDB + Google Gemini**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

> **Status:** under active development. The live demo link, CI badge and full documentation arrive in later phases.

## What this is

NexusRAG goes beyond the naive *chunk → embed → retrieve → generate* loop. The retrieval and orchestration logic is written by hand, with no LangChain or LlamaIndex, so every step of the pipeline is visible and explainable.

Planned capabilities:

- **Structure-aware, parent–child chunking.** Small chunks are searched and full sections are sent to the LLM.
- **Hybrid retrieval.** Dense vectors (Gemini embeddings in ChromaDB) and BM25 are fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking** with `BAAI/bge-reranker-base`.
- **Query transformation.** Follow-up condensation, multi-query expansion and optional HyDE.
- **Self-correcting agent.** An intent router, a relevance grader with query rewrites, and a groundedness check.
- **Grounded answers** with inline `[n]` citations that open the exact source chunk.
- **Semantic cache** for near-duplicate questions.
- **Evaluation suite** covering hit@k, MRR, context precision/recall, faithfulness, answer relevance, refusal rate, latency and cost.

## Branching model

| Branch      | Purpose                                                        |
|-------------|----------------------------------------------------------------|
| `main`      | Production. Only updated by a reviewed PR from `staging`.       |
| `staging`   | Default integration branch; deploys to the staging environment. |
| `feat/*`    | Feature work, branched from `staging` and merged back via PR.   |

## License

[MIT](LICENSE)
