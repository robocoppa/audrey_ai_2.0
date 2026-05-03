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

# uv for fast, reproducible installs. Pinned to digest (Phase 31); the tag
# `ghcr.io/astral-sh/uv:latest` is what this digest pointed to on 2026-05-02.
COPY --from=ghcr.io/astral-sh/uv@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 /uv /usr/local/bin/uv

WORKDIR /app

# Phase 21: install from pyproject.toml so deps live in one place. Adding
# a new runtime dep is now a single edit to pyproject.toml — no Dockerfile
# change needed. Bit us pre-Phase-21 with aiosqlite (Phase 15) and
# prometheus-client (Phase 17): adding a dep to pyproject alone, while the
# Dockerfile kept a separate hardcoded list, crashed the container at
# import time.
#
# Phase 24b: split the install into two layers so source-only edits don't
# re-resolve all deps:
#   Layer 1 (deps)    — cache key is `pyproject.toml` content. Reruns only
#                       when deps change.
#   Layer 2 (package) — cache key is the `src/audrey/` tree + README.md.
#                       Reruns on every code edit, but cheap (~5-10s) since
#                       it skips dep resolution via `--no-deps`.
# Pre-Phase-24b a single source edit re-resolved the full dep tree (~30-60s
# warm, ~3min cold) AND produced a fresh ~10 GB layer in build cache. After
# the split, source edits add ~MB-scale layers and run in seconds.
#
# Layout note: the wheel is built with `[tool.hatch.build] packages =
# ["src/audrey"]`, which expects the source at /app/src/audrey at build time.
# We copy it there for the build, then expose it at /app/audrey via symlink
# so the historical PYTHONPATH=/app + `import audrey` still resolves the
# live source. (PYTHONPATH wins over site-packages — runtime edits to
# /app/audrey/* are reflected without rebuilding the wheel.)

# ── Layer 1: deps only ───────────────────────────────────────────────
# Compile pyproject.toml's deps to a frozen list, then install them.
# The COPY's cache key is pyproject.toml's contents; this whole layer is
# cached unless deps change.
COPY pyproject.toml /app/pyproject.toml
RUN uv pip compile --quiet /app/pyproject.toml -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Layer 2: audrey package only ─────────────────────────────────────
# Build + install the audrey wheel without re-resolving deps. README is
# required because pyproject.toml's `readme = "README.md"` field tells
# hatchling to embed it in the wheel metadata at build time.
COPY README.md  /app/README.md
COPY src/audrey /app/src/audrey
RUN uv pip install --system --no-deps --no-cache /app \
    && ln -sf /app/src/audrey /app/audrey

COPY config.yaml /app/config.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

# PYTHONPATH so `audrey.*` resolves without an editable install
ENV PYTHONPATH=/app

CMD ["uvicorn", "audrey.main:app", "--host", "0.0.0.0", "--port", "8000"]
