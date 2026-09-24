# NexusRAG: slim runtime image, non-root, with the index on a writable volume.
#
# Build:  docker build -t nexusrag .
# Run:    docker run --rm -p 8000:8000 --env-file .env -v nexusrag-data:/data nexusrag
#
# The image uses the Gemini reranker by default: the cross-encoder pulls in torch and
# sentence-transformers (hundreds of MB) and needs ~8 s per query on 2 vCPUs, and the
# evaluation found it didn't improve answers on this corpus. For the local cross-encoder,
# build with --build-arg INSTALL_RERANK=true and set RERANKER_BACKEND=cross_encoder.

ARG PYTHON_VERSION=3.11

# --------------------------------------------------------------------- build
FROM python:${PYTHON_VERSION}-slim AS builder

ARG INSTALL_RERANK=false
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build
RUN python -m venv "$VIRTUAL_ENV"

# Only what the package build needs, so edits to source don't refetch dependencies.
COPY pyproject.toml README.md ./
COPY src/nexusrag/__init__.py src/nexusrag/__init__.py
RUN pip install --upgrade pip \
 && if [ "$INSTALL_RERANK" = "true" ]; then \
        pip install ".[rerank]" --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install .; \
    fi

COPY src/ src/
RUN pip install --no-deps .

# --------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    # Writable paths for the index, chat history and model cache (see the volume below).
    STORAGE_DIR=/data/storage \
    HF_HOME=/data/models \
    # Index the sample documents on first start when the knowledge base is empty.
    BOOTSTRAP_INDEX=true \
    # Matches the default build (no torch); override when built with INSTALL_RERANK=true.
    RERANKER_BACKEND=gemini

# uid 1000 keeps file ownership predictable on mounted volumes and on Spaces.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --create-home app \
 && mkdir -p /data && chown -R app:app /data

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app.py chainlit.md ./
COPY --chown=app:app .chainlit/ .chainlit/
COPY --chown=app:app data/ data/
COPY --chown=app:app scripts/bootstrap_index.py scripts/healthcheck.py scripts/
COPY --chown=app:app docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# Chainlit writes uploads under the working directory, so the app user must own it.
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
 && mkdir -p /app/.files \
 && chown -R app:app /app

USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "/app/scripts/healthcheck.py"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--headless"]
