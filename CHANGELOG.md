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
- Basic retrieval and generation (phase 3):
  - `Retriever`: dense search plus parent expansion. Parents are deduplicated, kept in rank order and packed within a token budget (with a missing-parent fallback), and metadata filters are supported.
  - Grounded answer generation: streaming, numbered `<passage>` context, cite-every-sentence rules, an exact refusal phrase, and a prompt-injection guard.
  - Citation mapping: `[n]`, `[1, 2]` and `[1-3]` markers are parsed, invalid references removed, and each citation carries its parent passage plus the matched chunks.
  - `RAGService` facade shared by the UI, CLI and (later) evaluation; `python -m nexusrag.ask` terminal client.
  - Minimal Chainlit app: welcome message listing the knowledge base, streamed answers, clickable `[n]` citations opening side panels, and friendly error messages.
- Advanced retrieval (phase 4):
  - Query transformation with the fast model:
    - condensation of follow-ups using chat history (skipped when there's no history);
    - multi-query paraphrases, deduplicated and capped;
    - optional HyDE, embedded as a document;
    - each step falls back to the original question on failure.
  - Hybrid search: dense and BM25 for every query, with a hand-written Reciprocal Rank Fusion (k=60, deterministic tie-breaking). Per-stage scores and matched queries are recorded on each chunk. HyDE is excluded from BM25.
  - Reranking:
    - `bge-reranker-base` cross-encoder, lazily loaded once and applied to the top N candidates with a score threshold;
    - Gemini-as-reranker backend, with automatic one-time fallback when the local model can't load;
    - retrieval keeps the RRF order if reranking fails.
  - `RetrievalOptions` for per-request stage switches (used by the UI settings and the evaluation configs).
  - `RetrievalResult` now exposes the query plan, fused candidates, individual rankings and the reranker used.
  - The chat app passes conversation history for follow-up questions.
  - Rerankers read tables linearised as `Header: value` lines (`linearize_tables`). On the sample corpus this lifted table chunks from, for example, 0.28 to 0.81 and from rank 4 to rank 2.
  - `RERANK_THRESHOLD` raised from 0.05 to 0.10 and `RERANK_CANDIDATES` set to 20, after calibrating against real `bge-reranker-base` scores on the sample corpus.
  - New settings `RERANK_CANDIDATES` and `RERANKER_MAX_LENGTH`; optional `rerank` extra (sentence-transformers).
  - Logging writes to the current `sys.stdout`, so swapped streams (test runners, servers) can't cause writes to a closed file.
- Hardening from the first live runs against the Gemini API:
  - Model fallback (`GENERATION_FALLBACK_MODEL`, `FAST_FALLBACK_MODEL`). A primary still overloaded (429/5xx) after retries, or returning 404, is replaced once by the backup model. Usage and cost are recorded against the model that actually answered.
  - Streams stay retryable until the first *visible* text. An answer interrupted after text has appeared is cleared (`on_reset`) and regenerated once on the fallback model.
  - Retry defaults tuned for interactive latency: 3 attempts per model, backoff capped at 8 s.
  - The SDK's automatic function calling is disabled (we never pass tools), which removes a warning and an INFO log line on every call.
  - `.env.example` keeps only secrets active, with every tunable commented out, so a copied `.env` no longer pins old defaults. `.env` is read BOM-tolerantly for Windows editors.
- Self-correcting agent (phase 5):
  - Hand-written state machine (`agent/graph.py`) with recorded steps. Every node emits an `AgentStep` for the UI and logs, and a hard step budget guards against loops.
  - Router: one fast-model structured call returns the route (chitchat, doc_qa, summarize_document, compare_documents or out_of_scope), catalog-resolved target documents and the condensed standalone question, so retrieval doesn't condense separately. Greetings skip the model, and failures default to doc_qa.
  - Relevance grader: rewrites the query and retries up to `MAX_RETRIEVAL_RETRIES`, merging attempts round-robin. It stops early when a rewrite finds nothing.
  - Groundedness grader: an unsupported answer is regenerated once with the claims called out (the UI draft is reset). If still unsupported, the agent declines and shows closest matches. Both graders fail open.
  - Summaries (single pass, or map-reduce for large documents with passage markers preserved) and comparisons (per-document retrieval, globally numbered passages, cited table).
  - Refusals carry `closest_matches`. `Answer` now also carries `steps` and `grounded`, and chat profiles can force summarize/compare via `mode`.
  - `RAGService.ask()` delegates to the agent. `AskResult` exposes every retrieval and the final context passages.
- Full chat experience (phase 6):
  - Password login (`AUTH_USERNAME`/`AUTH_PASSWORD`, constant-time comparison; fails closed in staging/production), with a session-signing secret generated for local runs only.
  - Persistent chat history on a SQLite schema for Chainlit's SQLAlchemy data layer: resumable threads, follow-up context rebuilt from the saved messages, and thumbs up/down feedback.
  - Drag-and-drop uploads with per-file progress (parsing, embedding, then indexed with section and chunk counts), checked server-side for type, size and count. Web pages can be added with a button or the `/url` command.
  - Settings panel: knowledge base selector, a field that creates a new knowledge base, a document filter, top-k, stage toggles (hybrid, multi-query, HyDE, rerank, self-correction) and answer style.
  - Chat profiles Q&A, Summarize and Compare, each with starter questions matched to the indexed documents.
  - Each agent node is shown as a timed step nested under one *Pipeline* step: route, standalone question, hybrid candidates and reranked passages as rank tables, grader verdicts. `AgentGraph.run()` gained an `on_node_start` callback.
  - PDF citations link to the original file at the cited page. Ingested PDFs are copied to `storage/files/`, backfilled for PDFs indexed earlier, and removed with the document.
  - Suggested follow-up questions as buttons after grounded answers (`ENABLE_FOLLOW_UPS`; one fast-model call).
  - `/stats` command (`RAGService.stats()`): documents, sections, chunks, vectors, model usage and cost.
  - Per-user rate limits for messages and uploads, and a friendly message instead of raw errors if a UI callback fails.
  - The cross-encoder is loaded in the background at startup (`RAGService.warm_up()`), and its load time is logged.
- Comparisons no longer drop sections below the rerank threshold. Each search is scoped to a named document, and on a multi-attribute question ("flight time and warranty") the threshold had discarded the performance tables. `RetrievalOptions.rerank_threshold` allows a per-request override.
- Tests no longer read the developer's `.env` when Chainlit is imported, and every settings variable is cleared from the test environment.
- Retrieval fixes found in live testing:
  - The final order now fuses the hybrid ranking with the reranker's ranking (RRF, `RERANK_FUSION_WEIGHT`). Pure reranking had buried a table that BM25, dense search and fusion all ranked #1; on the calibration questions, target-in-top-5 went from 10 to 11 of 12, with no rank worse.
  - The router no longer guesses whether the documents cover a question: company questions go to doc_qa, which gives grounded refusals.
  - The default generation fallback is now `gemini-3.6-flash`, which Google recommends and which stayed available while 3.7 was overloaded.
  - Tests can no longer load the real cross-encoder or reach the Hugging Face Hub, and `RAGService` accepts an injected reranker.
