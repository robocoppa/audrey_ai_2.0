"""Audrey FastAPI entrypoint.

Wires the orchestrator, tool registry, and KB stack into a single
FastAPI app. The KB pieces (Qdrant client, text/image embedders, and
the optional filesystem watcher) are instantiated in the lifespan and
attached to `app.state` so routes and the ReAct loop can read them.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response

from audrey import __version__
from audrey.config import get_config
from audrey.metrics import render as render_metrics
from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.qdrant import QdrantKB
from audrey.kb.uploads_db import UploadsDB, reconcile_with_qdrant
from audrey.kb.watcher import KBWatcher
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.graph import build_graph
from audrey.pipeline.semaphore import GpuGate
from audrey.routes.admin import router as admin_router
from audrey.routes.files import router as files_router
from audrey.routes.kb import router as kb_router
from audrey.routes.openai import router as openai_router
from audrey.routes.upload_ui import router as upload_ui_router
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

    # ─── KB stack ────────────────────────────────────────────────────
    kb_cfg = cfg.raw.get("kb", {}) or {}
    qdrant = QdrantKB(
        host=cfg.env.qdrant_host,
        port=cfg.env.qdrant_port,
        text_collection=kb_cfg.get("text_collection", "kb_text"),
        image_collection=kb_cfg.get("image_collection", "kb_images"),
    )
    try:
        await qdrant.ensure_collections()
    except Exception as e:  # noqa: BLE001 — Qdrant outage shouldn't kill boot
        log.warning("qdrant: ensure_collections failed: %s (KB endpoints will 503)", e)

    # Phase 15: sqlite index over per-user upload metadata. Reconciled
    # against qdrant on every boot — ghost rows pruned, missing rows
    # backfilled from the user collections.
    uploads_db = UploadsDB(kb_cfg.get("uploads_db_path", "/data/uploads.sqlite"))
    try:
        await reconcile_with_qdrant(uploads_db, qdrant)
    except Exception as e:  # noqa: BLE001 — reconciliation is a tune-up, not load-bearing
        log.warning("uploads_db: reconcile failed: %s (sqlite still usable)", e)

    text_embedder = TextEmbedder(
        ollama=ollama,
        model=kb_cfg.get("text_embedder", "nomic-embed-text"),
    )
    image_embedder = ImageEmbedder(
        model_name=kb_cfg.get("image_model", "clip-ViT-B-32"),
        cache_folder="/root/.cache/clip",
    )

    watcher: KBWatcher | None = None
    if os.environ.get("KB_WATCHER_ENABLED", "").strip() in ("1", "true", "yes"):
        roots = [Path(p) for p in (kb_cfg.get("dataset_paths") or [])]
        watcher = KBWatcher(
            roots=roots,
            qdrant=qdrant,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            debounce_s=float(kb_cfg.get("watcher_debounce_seconds", 2)),
            chunk_tokens=int(kb_cfg.get("chunk_tokens", 1000)),
            overlap_tokens=int(kb_cfg.get("chunk_overlap", 100)),
        )
        await watcher.start()

    app.state.cfg = cfg
    app.state.ollama = ollama
    app.state.registry = registry
    app.state.health = health
    app.state.gate = gate
    app.state.tools = tool_registry
    app.state.graph = graph
    app.state.qdrant = qdrant
    app.state.uploads_db = uploads_db
    app.state.text_embedder = text_embedder
    app.state.image_embedder = image_embedder
    app.state.kb_watcher = watcher

    log.info(
        "ready: ollama=%s; task types=%s; gpu_concurrency=%d; tools=%d (%s); "
        "qdrant=%s:%d; kb_watcher=%s; pipeline=compiled",
        cfg.env.ollama_host, registry.all_task_types(), gpu_concurrency,
        len(tool_registry.by_name), tool_registry.names(),
        cfg.env.qdrant_host, cfg.env.qdrant_port,
        "on" if watcher is not None else "off",
    )
    try:
        yield
    finally:
        if watcher is not None:
            await watcher.stop()
        uploads_db.close()
        qdrant.close()
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
app.include_router(kb_router)
app.include_router(files_router)
app.include_router(upload_ui_router)
app.include_router(admin_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/metrics", tags=["system"], include_in_schema=False)
async def metrics() -> Response:
    """Prometheus text-format exposition.

    Unauthenticated by design — Prometheus convention, and we don't
    publish the route via cloudflared, so it's effectively LAN-only
    (Unraid scrapes from the same docker network as audrey-ai).
    """
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


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
