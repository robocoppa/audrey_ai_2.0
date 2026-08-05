"""Audrey FastAPI entrypoint.

Wires the orchestrator, tool registry, and KB stack into a single
FastAPI app. The KB pieces (Qdrant client, text/image embedders, and
the optional filesystem watcher) are instantiated in the lifespan and
attached to `app.state` so routes and the ReAct loop can read them.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI
from fastapi.responses import Response

from audrey import __version__
from audrey.auth import AuthedUser, require_admin
from audrey.config import get_config
from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.qdrant import QdrantKB
from audrey.kb.reconcile import KBReconciler
from audrey.kb.uploads_db import UploadsDB, reconcile_with_qdrant
from audrey.kb.watcher import KBWatcher
from audrey.metrics import render as render_metrics
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.chat_archive import ChatArchiveClient
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.graph import build_graph
from audrey.routes.admin import router as admin_router
from audrey.routes.files import router as files_router
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.kb import router as kb_router
from audrey.routes.media import router as media_router
from audrey.routes.openai import router as openai_router
from audrey.routes.upload_ui import router as upload_ui_router
from audrey.tools.discovery import ToolRegistry, discover_all
from audrey.tools.dispatch import audit_user_scoping

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
    gate = FairLocalGate(concurrency=gpu_concurrency)
    fairness_cfg = cfg.raw.get("fairness", {}) or {}
    inflight = UserInflightRegistry(
        max_inflight_per_user=int(fairness_cfg.get("max_inflight_per_user", 3)),
        max_tracked_users=int(fairness_cfg.get("max_tracked_users", 1024)),
    )

    tool_servers: list[str] = list(cfg.tools.get("servers", []) or [])
    tools_enabled = bool(cfg.tools.get("enabled", True))
    if tools_enabled and tool_servers:
        tool_registry = await discover_all(tool_servers)
        audit_user_scoping(tool_registry)
    else:
        tool_registry = ToolRegistry()
        log.info("tools: disabled or no servers configured")

    # If the first discovery came up empty despite servers being configured,
    # custom-tools probably wasn't healthy yet (depends_on race, slow Qdrant
    # init, etc.). Retry in the background so Audrey doesn't sit at tools=0
    # for the whole session and silently skip everything that needs a tool.
    # The graph closes over the same ToolRegistry instance, so in-place
    # mutation is enough — no rebuild needed.
    tools_retry_task: asyncio.Task[None] | None = None
    if tools_enabled and tool_servers and not tool_registry.by_name:
        tools_retry_task = asyncio.create_task(
            _retry_tool_discovery(tool_registry, tool_servers),
            name="audrey.tools.retry_discovery",
        )

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

    # SQLite index over per-user upload metadata. Reconciled against qdrant
    # on every boot — ghost rows pruned, missing rows backfilled from the
    # user collections. Must run BEFORE uvicorn starts accepting traffic:
    # `reconcile_with_qdrant`'s step-2 prune isn't safe under concurrent
    # uploads (see its docstring). The lifespan runs everything before
    # `yield`, so this ordering is structural — don't move the call below.
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
    if cfg.env.kb_watcher_enabled:
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

    reconciler: KBReconciler | None = None
    reconcile_cfg = kb_cfg.get("reconcile", {}) or {}
    if reconcile_cfg.get("enabled", True):
        reconciler = KBReconciler(
            qdrant=qdrant,
            interval_s=float(reconcile_cfg.get("interval_s", 1800)),
            text_collection=kb_cfg.get("text_collection", "kb_text"),
            image_collection=kb_cfg.get("image_collection", "kb_images"),
        )
        await reconciler.start()

    app.state.cfg = cfg
    app.state.ollama = ollama
    app.state.registry = registry
    app.state.health = health
    app.state.gate = gate
    app.state.inflight = inflight
    app.state.tools = tool_registry
    app.state.graph = graph
    app.state.qdrant = qdrant
    app.state.uploads_db = uploads_db
    app.state.text_embedder = text_embedder
    app.state.image_embedder = image_embedder
    app.state.kb_watcher = watcher
    app.state.kb_reconciler = reconciler

    # Shared httpx client + chat-archive writer. The client is reused
    # across requests to avoid the per-call connection setup cost; the
    # archive writer is a thin wrapper that resolves the host server
    # from the tool registry on each call so a tools-server reload
    # doesn't strand the writer on a stale URL.
    archive_http = httpx.AsyncClient(timeout=10.0)
    archive_client = ChatArchiveClient(archive_http)
    app.state.archive_http = archive_http
    app.state.archive_client = archive_client

    log.info(
        "ready: ollama=%s; task types=%s; gpu_concurrency=%d; "
        "max_inflight_per_user=%d; tools=%d (%s); "
        "qdrant=%s:%d; kb_watcher=%s; kb_reconcile=%s; pipeline=compiled",
        cfg.env.ollama_host, registry.all_task_types(), gpu_concurrency,
        inflight.max_per_user,
        len(tool_registry.by_name), tool_registry.names(),
        cfg.env.qdrant_host, cfg.env.qdrant_port,
        "on" if watcher is not None else "off",
        "on" if reconciler is not None else "off",
    )
    try:
        yield
    finally:
        if tools_retry_task is not None and not tools_retry_task.done():
            tools_retry_task.cancel()
            try:
                await tools_retry_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 — shutdown path
                pass
        if reconciler is not None:
            await reconciler.stop()
        if watcher is not None:
            await watcher.stop()
        uploads_db.close()
        qdrant.close()
        await ollama.aclose()
        await archive_http.aclose()


async def _retry_tool_discovery(
    registry: ToolRegistry,
    tool_servers: list[str],
    *,
    attempts: int = 30,
    interval_s: float = 4.0,
) -> None:
    """Retry `discover_all` in the background when initial discovery was empty.

    Custom-tools may not be healthy yet when Audrey starts (depends_on
    races, slow Qdrant init). Without this retry the live registry sits
    at zero tools until somebody hits `/v1/tools/rediscover`, and every
    request that wanted a tool quietly skips.

    The loop bails out the moment any tool is discovered, so it costs
    nothing on a healthy startup. Bounded so a permanently-broken
    custom-tools doesn't generate retries forever — after the window
    expires, manual rediscover stays available. Default window:
    30 × 4s = 2 minutes, which covers the worst observed cold-start.
    """
    for attempt in range(1, attempts + 1):
        await asyncio.sleep(interval_s)
        try:
            fresh = await discover_all(tool_servers)
        except Exception as e:  # noqa: BLE001 — discovery is best-effort here
            log.warning("tools: retry %d/%d failed: %s", attempt, attempts, e)
            continue
        if not fresh.by_name:
            log.info("tools: retry %d/%d still empty", attempt, attempts)
            continue
        registry.by_name.clear()
        registry.by_name.update(fresh.by_name)
        audit_user_scoping(registry)
        log.info(
            "tools: retry %d/%d succeeded -> %d tool(s): %s",
            attempt, attempts, len(registry.by_name), registry.names(),
        )
        return
    log.warning(
        "tools: gave up after %d retries (%.0fs); use /v1/tools/rediscover to retry manually",
        attempts, attempts * interval_s,
    )


app = FastAPI(
    title="Audrey AI",
    version=__version__,
    description=(
        "OpenAI-compatible orchestrator over Ollama (local + cloud-bridge "
        "models) with a shared LangGraph pipeline: classify → complexity "
        "gate → fast path or planner → deep panel → synth → reflect. Six "
        "virtual models — `audrey_deep`/`audrey_cloud`/`audrey_local` (always "
        "deep, different pools), `audrey_research` (always deep, staged "
        "research → verify → write for grounded answers), `audrey_auto` "
        "(adaptive), `audrey_fast` (always fast, no escalation). Tool dispatch "
        "via custom-tools (Brave "
        "web search, Qdrant text + CLIP-image KB, per-user memory, per-user "
        "chat-history search). Per-user fair scheduling at the local-GPU "
        "gate, OWUI-backed auth, Prometheus metrics at `/metrics`, KB "
        "watcher + periodic reconcile keeping global collections drift-free, "
        "streaming progress banners + per-worker tools-used footer on "
        "streamed responses."
    ),
    lifespan=lifespan,
)

app.include_router(openai_router)
app.include_router(kb_router)
app.include_router(files_router)
app.include_router(media_router)
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
async def rediscover_tools(
    _admin: AuthedUser = Depends(require_admin),
) -> dict[str, list[str] | int]:
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
    audit_user_scoping(reg)
    log.info("tools: rediscover -> %d tool(s): %s", len(reg.by_name), reg.names())
    return {"tools": reg.names(), "count": len(reg.by_name)}


def run() -> None:
    """Console-script entry point for `audrey` command."""
    import uvicorn

    uvicorn.run(
        "audrey.main:app",
        host="0.0.0.0",  # noqa: S104 — server bind, all-interfaces is the point
        port=8000,
        log_level="info",
    )
