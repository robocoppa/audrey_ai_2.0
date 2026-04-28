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
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Phase 21: install from pyproject.toml so deps live in one place. Adding
# a new runtime dep is now a single edit to pyproject.toml — no Dockerfile
# change needed. Tradeoff: the install layer invalidates on any source
# change because hatchling needs the package source to build the wheel,
# so `docker build` runs install on every code change rather than only
# when deps change. Acceptable to never re-trip over the pre-Phase-21
# footgun: adding a dep to pyproject alone (forgetting the Dockerfile's
# hardcoded list) crashed the container at import time. Bit us with
# aiosqlite (Phase 15) and prometheus-client (Phase 17).
#
# Layout note: the wheel is built with `[tool.hatch.build] packages = ["src/audrey"]`,
# which expects the source at /app/src/audrey at build time. We copy it
# there for the build, then expose it at /app/audrey via symlink so the
# historical PYTHONPATH=/app + `import audrey` still resolves the live
# source. (PYTHONPATH wins over site-packages — runtime edits to
# /app/audrey/* are reflected without rebuilding the wheel.)
COPY pyproject.toml /app/pyproject.toml
COPY README.md      /app/README.md
COPY src/audrey     /app/src/audrey
RUN uv pip install --system . && ln -s /app/src/audrey /app/audrey

COPY config.yaml /app/config.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

# PYTHONPATH so `audrey.*` resolves without an editable install
ENV PYTHONPATH=/app

CMD ["uvicorn", "audrey.main:app", "--host", "0.0.0.0", "--port", "8000"]
