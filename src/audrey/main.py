"""Audrey FastAPI entrypoint.

Phase 4: app skeleton wired to the Ollama client + model registry. Exposes
`/health`, `/v1/models`, `/v1/chat/completions` (pass-through). Classifier,
pipeline, tools, and KB endpoints attach in later phases.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from audrey import __version__
from audrey.config import get_config
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.graph import build_graph
from audrey.routes.openai import router as openai_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("audrey")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    log.info("audrey starting; version=%s", __version__)
    log.info("config loaded: %d model types, tools=%s", len(cfg.model_registry), cfg.tools.get("servers"))

    default_timeout = float(cfg.timeouts.get("medium", 180))
    ollama = OllamaClient(cfg.env.ollama_host, default_timeout_s=default_timeout)
    registry = ModelRegistry(cfg)
    health = HealthTracker()

    graph = build_graph(cfg, ollama, registry, health)

    app.state.cfg = cfg
    app.state.ollama = ollama
    app.state.registry = registry
    app.state.health = health
    app.state.graph = graph

    log.info("ready: ollama=%s; task types=%s; pipeline=compiled",
             cfg.env.ollama_host, registry.all_task_types())
    try:
        yield
    finally:
        await ollama.aclose()


app = FastAPI(
    title="Audrey AI",
    version=__version__,
    description=(
        "OpenAI-compatible orchestrator. Exposes three virtual models — "
        "`audrey_deep`, `audrey_cloud`, `audrey_local` — each a different "
        "pipeline mode over the same model registry. Phase 4 build is a "
        "pass-through; routing/panels/tools land in later phases."
    ),
    lifespan=lifespan,
)

app.include_router(openai_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def run() -> None:
    """Console-script entry point for `audrey` command."""
    import uvicorn

    uvicorn.run(
        "audrey.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
