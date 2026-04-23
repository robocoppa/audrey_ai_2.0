# Audrey FastAPI orchestrator
#
# Build from the repo root:
#   docker build -f docker/audrey.Dockerfile -t audrey-ai:latest .
#
# Run locally:
#   docker run --rm -p 8000:8000 \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     -v $PWD/config.yaml:/app/config.yaml:ro \
#     audrey-ai:latest

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    AUDREY_CONFIG=/app/config.yaml

# System packages needed for some Python wheels (lxml, pillow) and diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Runtime deps. Phase 8 adds the KB stack: qdrant-client for the vector
# store, sentence-transformers+torch for CLIP image embeddings (text
# embeddings go through ollama), pypdf/python-docx/beautifulsoup4 for
# document loaders, watchdog for the dataset auto-ingest watcher.
RUN uv pip install --system \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.32" \
      "httpx>=0.28" \
      "pydantic>=2.9" \
      "pydantic-settings>=2.6" \
      "pyyaml>=6.0.2" \
      "tenacity>=9.0" \
      "tiktoken>=0.8" \
      "langgraph>=0.2.60" \
      "langchain-core>=0.3.28" \
      "python-dotenv>=1.0" \
      "qdrant-client>=1.12" \
      "sentence-transformers>=3.2" \
      "pypdf>=5.1" \
      "python-docx>=1.1" \
      "beautifulsoup4>=4.12" \
      "lxml>=5.3" \
      "watchdog>=6.0" \
      "pillow>=11.0"

# Copy the package + config
COPY pyproject.toml /app/pyproject.toml
COPY config.yaml    /app/config.yaml
COPY src/audrey     /app/audrey

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

# PYTHONPATH so `audrey.*` resolves without an editable install
ENV PYTHONPATH=/app

CMD ["uvicorn", "audrey.main:app", "--host", "0.0.0.0", "--port", "8000"]
