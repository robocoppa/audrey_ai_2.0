# Audrey custom-tools FastAPI server
#
# Build from the repo root:
#   docker build -f docker/custom-tools.Dockerfile -t audrey-custom-tools:latest .
#
# Run locally:
#   docker run --rm -p 8001:8001 \
#     -e BRAVE_API_KEY=... \
#     -e AUDREY_URL=http://host.docker.internal:8000 \
#     -v $PWD/tools-server/data:/app/data \
#     audrey-custom-tools:latest

# Phase 31: pinned to digest for reproducibility. Tag (`python:3.12-slim`)
# is what this digest pointed to on 2026-05-02. Same digest as audrey's
# Dockerfile — keep them in sync when bumping.
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# uv for fast, reproducible installs. Pinned to digest (Phase 31).
COPY --from=ghcr.io/astral-sh/uv@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first so they cache independently of source changes.
#
# Phase 21 note: the audrey-ai Dockerfile installs from `pyproject.toml` via
# `uv pip install --system .`, but tools-server has a flat-script layout
# (no `src/<pkg>/` wrapper, just `app.py` / `brave.py` / `db.py` / `settings.py`)
# which hatchling can't cleanly wheel-build. So this list stays manual.
# **If you add a dep to tools-server/pyproject.toml, also add it here** —
# Phase 12 hit this with qdrant-client when only pyproject was edited.
COPY tools-server/pyproject.toml /app/pyproject.toml
RUN uv pip install --system \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.32" \
      "httpx>=0.28" \
      "pydantic>=2.9" \
      "pydantic-settings>=2.6" \
      "aiosqlite>=0.20" \
      "tenacity>=9.0" \
      "python-dotenv>=1.0" \
      "qdrant-client>=1.12"

# Copy application code
COPY tools-server/app.py      /app/app.py
COPY tools-server/brave.py    /app/brave.py
COPY tools-server/db.py       /app/db.py
COPY tools-server/settings.py /app/settings.py

# Data dir for memory.db (bind-mounted in production)
RUN mkdir -p /app/data
ENV TOOLS_DATA_DIR=/app/data

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8001/health', timeout=3).raise_for_status()" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
