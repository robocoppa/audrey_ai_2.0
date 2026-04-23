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
from audrey.pipeline.semaphore import GpuGate
from audrey.routes.openai import router as openai_router
from audrey.tools.discovery import discover_all

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
    gpu_concurrency = int(cfg.raw.get("gpu", {}).get("concurrency", 1))
    gate = GpuGate(concurrency=gpu_concurrency)

    tool_servers: list[str] = list(cfg.tools.get("servers", []) or [])
    tools_enabled = bool(cfg.tools.get("enabled", True))
    if tools_enabled and tool_servers:
        tool_registry = await discover_all(tool_servers)
    else:
        from audrey.tools.discovery import ToolRegistry
        tool_registry = ToolRegistry()
        log.info("tools: disabled or no servers configured")

    graph = build_graph(cfg, ollama, registry, health, gate, tool_registry)

    app.state.cfg = cfg
    app.state.ollama = ollama
    app.state.registry = registry
    app.state.health = health
    app.state.gate = gate
    app.state.tools = tool_registry
    app.state.graph = graph

    log.info("ready: ollama=%s; task types=%s; gpu_concurrency=%d; tools=%d (%s); pipeline=compiled",
             cfg.env.ollama_host, registry.all_task_types(), gpu_concurrency,
             len(tool_registry.by_name), tool_registry.names())
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


@app.get("/v1/tools", tags=["tools"])
async def list_tools() -> dict[str, list[dict]]:
    """Inspect what tools are currently registered for the ReAct loop."""
    reg = app.state.tools
    return {
        "tools": [
            {
                "name": s.name,
                "description": s.description,
                "server_url": s.server_url,
                "path": s.path,
                "parameters": s.parameters,
            }
            for s in reg.specs()
        ],
    }


@app.post("/v1/tools/rediscover", tags=["tools"])
async def rediscover_tools() -> dict[str, list[str] | int]:
    """Re-fetch /openapi.json from every configured tool server.

    Mutates the live ToolRegistry in place — the graph keeps its closure
    over the same registry instance, so changes take effect on the next
    request without a graph rebuild.
    """
    cfg = app.state.cfg
    reg = app.state.tools
    tool_servers = list(cfg.tools.get("servers", []) or [])
    fresh = await discover_all(tool_servers)
    reg.by_name.clear()
    reg.by_name.update(fresh.by_name)
    log.info("tools: rediscover -> %d tool(s): %s", len(reg.by_name), reg.names())
    return {"tools": reg.names(), "count": len(reg.by_name)}


def run() -> None:
    """Console-script entry point for `audrey` command."""
    import uvicorn

    uvicorn.run(
        "audrey.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
