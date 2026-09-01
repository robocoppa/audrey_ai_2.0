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

# Phase 31: pinned to digest for reproducibility. Tag (`python:3.12-slim`)
# is what this digest pointed to on 2026-05-02. To bump:
#   docker pull python:3.12-slim
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim
# then replace the digest below.
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    AUDREY_CONFIG=/app/config.yaml

# System packages needed for some Python wheels (lxml, pillow) and diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libxml2 \
        libxslt1.1 \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs. Pinned to digest (Phase 31); the tag
# `ghcr.io/astral-sh/uv:latest` is what this digest pointed to on 2026-05-02.
COPY --from=ghcr.io/astral-sh/uv@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 /uv /usr/local/bin/uv

WORKDIR /app

# The workspace lock is the dependency authority. `--locked` refuses a stale
# lock instead of silently resolving whatever satisfies pyproject today. The
# first sync installs only third-party dependencies, preserving the fast source
# edit layer; the second adds Audrey itself without changing the resolution.

# ── Layer 1: deps only ──────────────────────────────
COPY pyproject.toml uv.lock /app/
COPY tools-server/pyproject.toml /app/tools-server/pyproject.toml
RUN uv sync --locked --no-dev --package audrey \
        --no-install-workspace --no-cache

# ── Layer 2: audrey package only ────────────────────────────
COPY README.md  /app/README.md
COPY src/audrey /app/src/audrey
RUN uv sync --locked --no-dev --package audrey --no-editable --no-cache

COPY config.yaml /app/config.yaml

# UID/GID 99:100 is Unraid's normal nobody:users ownership. Bind mounts replace
# these image directories at runtime, so the host paths must carry the same
# numeric owner before the first non-root start.
ARG APP_UID=99
ARG APP_GID=100
RUN useradd --no-log-init --system --create-home --uid "${APP_UID}" --gid "${APP_GID}" \
        --shell /usr/sbin/nologin audrey \
    && install -d -o "${APP_UID}" -g "${APP_GID}" /data /home/audrey/.cache/clip

ENV HOME=/home/audrey
USER audrey

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1


CMD ["uvicorn", "audrey.main:app", "--host", "0.0.0.0", "--port", "8000"]
