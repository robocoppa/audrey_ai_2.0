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
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# uv for fast, reproducible installs. Pinned to digest (Phase 31).
COPY --from=ghcr.io/astral-sh/uv@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 /uv /usr/local/bin/uv

WORKDIR /app

# Select this workspace member from the repository lock. The sidecar executes
# its copied modules directly from /app, so only its locked third-party
# dependencies belong in the environment layer.

# ── Layer 1: deps only ──────────────────────────────
COPY pyproject.toml uv.lock /app/
COPY tools-server/pyproject.toml /app/tools-server/pyproject.toml
RUN uv sync --locked --no-dev --package audrey-custom-tools \
        --no-install-workspace --no-cache

# ── Layer 2: source only ──────────────────────────────
COPY tools-server/README.md /app/README.md
COPY tools-server/*.py      /app/

# Data dir for memory.db and chat_archive.db (bind-mounted in production).
ARG APP_UID=99
ARG APP_GID=100
RUN useradd --no-log-init --system --create-home --uid "${APP_UID}" --gid "${APP_GID}" \
        --shell /usr/sbin/nologin tools \
    && install -d -o "${APP_UID}" -g "${APP_GID}" /app/data

ENV HOME=/home/tools \
    TOOLS_DATA_DIR=/app/data
USER tools

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8001/health', timeout=3).raise_for_status()" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
