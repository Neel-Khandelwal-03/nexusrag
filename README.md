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

## Local development

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then set GEMINI_API_KEY
```

Check your Gemini key and model names with a few real, cheap API calls:

```bash
python scripts/smoke_gemini.py
```

Run the quality gates. CI runs the same commands, with Gemini mocked:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

### Ingesting documents

```bash
python -m nexusrag.ingest data/                       # index the sample corpus (incremental)
python -m nexusrag.ingest report.pdf https://example.com/guide --collection research
python -m nexusrag.ingest data/ --prune               # also remove docs whose files were deleted
python -m nexusrag.ingest --list                      # what's indexed
python -m nexusrag.ingest --list-collections          # knowledge bases
```

Supported sources: **PDF** (with page numbers and tables), **DOCX**, **Markdown**, **TXT** and **web URLs**.

How documents are processed:
- **Parsing.** Loaders extract structure: headings, paragraphs, lists, tables (rendered as Markdown) and code blocks.
  - PDF: headings come from the outline or from font sizes; running headers, footers and page numbers are removed; tables and paragraphs split by a page break are re-joined.
  - URLs: fetched with SSRF protection (private and link-local addresses are refused on every redirect), and the main content is extracted with trafilatura.
- **Parent–child chunking.** Documents are split along their headings into *parent* sections of up to about 1,200 tokens, which is what the LLM reads. Each parent is cut into *child* chunks of about 256 tokens with a 40-token sentence-aligned overlap; children are what get embedded and searched.
  - Small sibling sections are merged into one parent.
  - Tables are never split mid-table.
  - Every chunk records its section path, e.g. `2 Hardware > 2.3 Battery System`.
- **Incremental indexing.**
  - Unchanged files are skipped (SHA-256 of the file).
  - Changed files replace their old chunks in Chroma, BM25 and the parent store.
  - Unchanged chunks inside a changed file reuse their existing vectors, so an edit costs only the changed chunks.
- **Storage.** Vectors live in ChromaDB (one collection per knowledge base). The document registry, parent sections and BM25 token lists share one SQLite file, so each document update commits atomically.

The `data/` folder holds a small fictional corpus about "Skylark Dynamics", a made-up drone maker, used for the demo and the evaluation suite. It contains two product specs (PDF and DOCX), an HR policy (Markdown), a support FAQ (TXT) and a quarterly report (PDF with financial tables). To regenerate the binary files, run `python scripts/build_sample_docs.py`.

### Models

All model IDs are configured via environment variables (see [.env.example](.env.example)):

| Role | Default | Used for |
|------|---------|----------|
| `GENERATION_MODEL` | `gemini-3.8-flash` | User-facing answers (streamed) |
| `FAST_MODEL` | `gemini-3.5-flash-lite` | Query rewriting, routing, grading |
| `EMBEDDING_MODEL` | `gemini-embedding-2` (768-d) | Chunk and query embeddings |

`gemini-embedding-2` has no `task_type` parameter. Queries are embedded as `task: search result | query: …` and documents as `title: … | text: …`, following Google's guidance for asymmetric retrieval. Temperature is left at the Gemini 3 default unless you set it explicitly.

## Branching model

| Branch      | Purpose                                                        |
|-------------|----------------------------------------------------------------|
| `main`      | Production. Only updated by a reviewed PR from `staging`.       |
| `staging`   | Default integration branch; deploys to the staging environment. |
| `feat/*`    | Feature work, branched from `staging` and merged back via PR.   |

## License

[MIT](LICENSE)
