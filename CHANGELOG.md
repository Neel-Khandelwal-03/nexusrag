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
