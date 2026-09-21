# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository scaffolding: `.gitignore`, MIT license, README, PR and issue templates, minimal CI workflow.
- Foundation (phase 1):
  - `pyproject.toml` packaging with ruff, mypy and pytest configuration.
  - `nexusrag.config`: pydantic-settings configuration covering every tunable, with `.env.example`.
  - `nexusrag.llm.gemini_client`: async Gemini wrapper providing per-role models, tenacity retries with jittered backoff, JSON-schema structured output, streaming, batched and normalised embeddings, and UI-safe error translation.
  - `nexusrag.llm.usage`: per-request and process-wide token and cost accounting.
  - `nexusrag.log`: structlog logging with request IDs, stage timers, and redaction of secrets and document content.
  - `nexusrag.models`: Pydantic models for documents, chunks, parents, retrieval scores, citations and answers.
  - `scripts/smoke_gemini.py`: live check of the API key and model configuration.
  - Unit tests with a scripted fake Gemini client (the real SDK client is blocked in tests).
- CI now installs the package and runs ruff lint, ruff format, mypy and pytest.
- Ingestion (phase 2):
  - Loaders for PDF, DOCX, Markdown, TXT and URLs.
    - PDF: PyMuPDF text with outline- or font-based headings, pdfplumber tables as Markdown, header/footer/page-number removal, and re-joining of tables and paragraphs split across pages.
    - URLs: SSRF-guarded fetching, with trafilatura extracting the main content.
  - Structure-aware parent/child chunker. It splits along headings, merges small sibling sections, keeps tables whole (oversized tables split between rows with the header repeated), and adds sentence-aligned overlap between child chunks.
  - Stores: ChromaDB vector store (cosine, one collection per knowledge base, embedding-model guard), plus a SQLite registry, parent store and BM25 index (rank-bm25 with Lucene IDF) sharing one database for atomic document replacement.
  - Incremental pipeline: skips unchanged files, removes stale chunks, reuses vectors for unchanged chunks, optionally prunes deleted files, and reports progress events for the UI.
  - `python -m nexusrag.ingest` CLI (`nexusrag-ingest` console script).
  - Fictional "Skylark Dynamics" sample corpus in `data/` and `scripts/build_sample_docs.py`.
