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

# Install from pyproject.toml — deps and source in one place. Matches the
# audrey-ai Dockerfile pattern (Phase 21). Two-layer split per Phase 24b:
# dep changes rebuild Layer 1 (~30s), source changes rebuild Layer 2 (~5s).
#
# Layout note: the wheel is built with `[tool.hatch.build.targets.wheel]
# packages = ["."]` (Phase 5). That packs every `.py` file in tools-server/
# at the wheel root, so `from brave import …` keeps working. Adding a new
# .py file to tools-server/ no longer requires a Dockerfile edit.

# ── Layer 1: deps only ───────────────────────────────────────────────
# Compile pyproject.toml's deps to a frozen list, then install them.
# Cache key is `pyproject.toml`'s contents — re-runs only when deps
# change.
COPY tools-server/pyproject.toml /app/pyproject.toml
RUN uv pip compile --quiet /app/pyproject.toml -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Layer 2: package only ────────────────────────────────────────────
# Build + install the wheel without re-resolving deps. The wildcard
# COPY picks up new tools-server *.py files automatically — no
# Dockerfile edit needed when adding modules. README.md is required
# because pyproject.toml's `readme = "README.md"` field embeds it in
# the wheel metadata at build time.
COPY tools-server/README.md /app/README.md
COPY tools-server/*.py      /app/
RUN uv pip install --system --no-deps --no-cache /app

# Data dir for memory.db and chat_archive.db (bind-mounted in production)
RUN mkdir -p /app/data
ENV TOOLS_DATA_DIR=/app/data

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8001/health', timeout=3).raise_for_status()" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
