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
from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
from audrey.kb.qdrant import QdrantKB
from audrey.kb.reconcile import KBReconciler
from audrey.kb.storage_lifecycle import StorageLifecycle
from audrey.kb.uploads_db import UploadsDB, reconcile_with_qdrant
from audrey.kb.watcher import KBWatcher
from audrey.metrics import render as render_metrics
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.chat_archive import ChatArchiveClient, ChatArchiveQueue
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.graph import build_graph
from audrey.routes.admin import router as admin_router
from audrey.routes.files import router as files_router
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.kb import router as kb_router
from audrey.routes.media import router as media_router
from audrey.routes.openai import router as openai_router
from audrey.routes.upload_ui import router as upload_ui_router
from audrey.routes.user_data import router as user_data_router
from audrey.tools.discovery import ToolRegistry, discover_all
from audrey.tools.dispatch import audit_user_scoping
from audrey.user_data_purge import UserDataPurgeCoordinator

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
    # ⚠️ Always logged, including the empty case. An env override silently
    # displaces a committed YAML value, so `config.yaml` stops describing what
    # is running and nothing in the file admits it — the exact trap the
    # `VIDEO_LEASE_MINUTES` comment warns about. Printing "none" when there are
    # none is what makes the line trustworthy on the runs that do have some.
    if overrides := cfg.active_env_overrides:
        log.warning("config: %d ENV OVERRIDE(S) active, config.yaml is not the "
                    "whole truth: %s", len(overrides),
                    ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items())))
    else:
        log.info("config: no env overrides; config.yaml is authoritative")

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

    # If first discovery is empty or partially degraded, custom-tools is not
    # fully ready yet. Retry in the background so the live registry can recover
    # without an Audrey restart.
    # The graph closes over the same ToolRegistry instance, so in-place
    # mutation is enough — no rebuild needed.
    tools_retry_task: asyncio.Task[None] | None = None
    if (
        tools_enabled
        and tool_servers
        and (
            not tool_registry.by_name
            or any(
                not spec.available
                for spec in tool_registry.policy_records()
            )
        )
    ):
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
    file_operation_locks = FileOperationLocks()
    try:
        await reconcile_with_qdrant(uploads_db, qdrant)
    except Exception as e:  # noqa: BLE001 — reconciliation is a tune-up, not load-bearing
        log.warning("uploads_db: reconcile failed: %s (sqlite still usable)", e)
    storage_lifecycle = StorageLifecycle(uploads_db)
    file_deletion_cfg = kb_cfg.get("file_deletion", {}) or {}
    upload_root = Path(kb_cfg.get("upload_root", "/data/uploads"))
    file_deletions = FileDeletionWorker(
        db=uploads_db,
        qdrant=qdrant,
        upload_root=upload_root,
        locks=file_operation_locks,
        retry_interval_s=float(file_deletion_cfg.get("retry_interval_s", 30.0)),
        batch_size=int(file_deletion_cfg.get("batch_size", 50)),
    )
    await file_deletions.start()
    fetch_cfg = kb_cfg.get("fetch", {}) or {}
    fetch_recovery = await storage_lifecycle.restore_pending_url_fetches(
        ceiling_bytes=int(fetch_cfg.get("max_bytes_mb", 2048)) * 1024 * 1024,
    )
    if fetch_recovery.restored_pending or fetch_recovery.released_stranded:
        log.info(
            "uploads_db: URL quota recovery restored=%d pending, "
            "released=%d stranded",
            fetch_recovery.restored_pending,
            fetch_recovery.released_stranded,
        )
    text_embedder = TextEmbedder(
        ollama=ollama,
        model=kb_cfg.get("text_embedder", "nomic-embed-text"),
        keep_alive=kb_cfg.get("text_embedder_keep_alive", "24h"),
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
    app.state.storage_lifecycle = storage_lifecycle
    app.state.file_operation_locks = file_operation_locks
    app.state.file_deletions = file_deletions
    app.state.text_embedder = text_embedder
    app.state.image_embedder = image_embedder
    app.state.kb_watcher = watcher
    app.state.kb_reconciler = reconciler

    # Shared transport + durable chat-archive outbox. Response handlers commit
    # only the small local source row; this lifecycle-owned worker performs the
    # remote custom-tools call and resumes unfinished rows after restart.
    archive_http = httpx.AsyncClient(timeout=10.0)
    archive_transport = ChatArchiveClient(
        archive_http,
        service_token=cfg.env.kb_service_token,
    )
    archive_cfg = cfg.raw.get("chat_archive", {}) or {}
    archive_queue_cfg = archive_cfg.get("queue", {}) or {}
    archive_enabled = bool(archive_cfg.get("enabled", True))
    archive_queue = ChatArchiveQueue(
        client=archive_transport,
        registry=tool_registry,
        sqlite_path=Path(
            archive_queue_cfg.get(
                "sqlite_path",
                "/data/chat_archive_outbox.sqlite",
            )
        ),
        maxsize=int(archive_queue_cfg.get("maxsize", 128)),
        retry_interval_s=float(
            archive_queue_cfg.get("retry_interval_s", 30.0)
        ),
    )
    await archive_queue.start(run_worker=archive_enabled)
    archive_client: ChatArchiveQueue | None = (
        archive_queue if archive_enabled else None
    )
    app.state.archive_http = archive_http
    app.state.archive_client = archive_client
    app.state.archive_transport = archive_transport
    app.state.kb_service_token = cfg.env.kb_service_token

    purge_cfg = ((cfg.raw.get("user_data", {}) or {}).get("purge", {}) or {})
    user_data_purges = UserDataPurgeCoordinator(
        db=uploads_db,
        file_deletions=file_deletions,
        archive_queue=archive_queue,
        archive_transport=archive_transport,
        registry=tool_registry,
        upload_root=upload_root,
        retry_interval_s=float(purge_cfg.get("retry_interval_s", 30.0)),
        batch_size=int(purge_cfg.get("batch_size", 50)),
    )
    await user_data_purges.start()
    app.state.user_data_purges = user_data_purges

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
        await user_data_purges.stop()
        await archive_queue.stop()
        await file_deletions.stop()
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
    """Retry discovery while the initial registry is empty or degraded.

    Custom-tools may be missing or may have optional dependencies still
    recovering when Audrey starts. Without this retry the live registry would
    stay empty or partial until somebody calls `/v1/tools/rediscover`.

    The loop bails out when every declared tool is available, so it costs
    nothing on a healthy startup and also recovers a partially degraded set.
    Bounded so a permanently broken custom-tools does not generate retries
    forever — after the window
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
        unavailable = sorted(
            spec.name
            for spec in registry.policy_records()
            if not spec.available
        )
        if unavailable:
            log.info(
                "tools: retry %d/%d still degraded; unavailable=%s",
                attempt, attempts, unavailable,
            )
            continue
        log.info(
            "tools: retry %d/%d succeeded -> %d tool(s): %s",
            attempt, attempts, len(registry.names()), registry.names(),
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
app.include_router(user_data_router)
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
                "visibility": s.visibility.value,
                "user_scoped": s.user_scoped,
                "user_scope": s.user_scope.value,
                "dependencies": sorted(s.dependencies),
                "available": s.available,
                "unavailable_reason": s.unavailable_reason,
                "parameters": s.parameters,
            }
            for s in reg.policy_records()
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
    log.info(
        "tools: rediscover -> %d/%d available: %s",
        len(reg.names()), len(reg.by_name), reg.names(),
    )
    return {
        "tools": reg.names(),
        "count": len(reg.names()),
        "declared_count": len(reg.by_name),
    }


def run() -> None:
    """Console-script entry point for `audrey` command."""
    import uvicorn

    uvicorn.run(
        "audrey.main:app",
        host="0.0.0.0",  # noqa: S104 — server bind, all-interfaces is the point
        port=8000,
        log_level="info",
    )
