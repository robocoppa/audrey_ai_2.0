"""Audrey's API entrypoint and shared service lifecycle.

The lifespan owns authentication, durable application state and run recovery,
model orchestration, tools and skills, the knowledge-base stack, and background
workers for native runs, titles, archive delivery, file deletion and data purge.
These services live on `app.state`; routers expose native and compatible APIs.
The browser application is served separately by audrey-ui.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response

from audrey import __version__
from audrey.agent_executor import ResearchAgentExecutor, agent_settings
from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.auth import AuthedUser, require_admin, require_principal
from audrey.chat_projection import ChatProjectionPromoter
from audrey.config import get_config
from audrey.content_search import ContentSearchError, ContentSearchIndex
from audrey.content_search_service import ContentSearchService
from audrey.content_sources import ContentSourceReader
from audrey.conversation_titles import ConversationTitleGenerator
from audrey.embedding_profile import profile_from_mapping
from audrey.identity import Principal, build_cloudflare_access_verifier
from audrey.kb.embed import ImageEmbedder
from audrey.kb.embed import text_embedder as build_text_embedder
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
from audrey.readiness import ReadinessCollector
from audrey.references.embedding import reference_embedder
from audrey.references.library import ReferenceLibraryRegistry
from audrey.references.releases import ReleaseError, ReleaseStore
from audrey.references.search import ReferenceSearchService
from audrey.routes.admin import router as admin_router
from audrey.routes.app import router as application_router
from audrey.routes.app.runs import NativeRunManager
from audrey.routes.evidence_tools import router as evidence_tools_router
from audrey.routes.files import router as files_router
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.kb import router as kb_router
from audrey.routes.media import router as media_router
from audrey.routes.openai import VIRTUAL_MODELS
from audrey.routes.openai import router as openai_router
from audrey.routes.references import router as references_router
from audrey.routes.tool_history import router as tool_history_router
from audrey.routes.user_data import router as user_data_router
from audrey.skills import SkillRegistry
from audrey.tax_exports import TaxReviewExports
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, discover_all
from audrey.tools.dispatch import audit_user_scoping
from audrey.user_data_purge import UserDataPurgeCoordinator
from audrey.watch_executor import WatchExecutor, watch_settings
from audrey.watch_runtime import WatchRuntime

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

    cloudflare_access_verifier = build_cloudflare_access_verifier(
        enabled=cfg.env.cloudflare_access_enabled,
        team_domain=cfg.env.cloudflare_access_team_domain,
        audience=cfg.env.cloudflare_access_audience,
    )
    if cloudflare_access_verifier is not None:
        log.info(
            "auth: Cloudflare Access enabled team_domain=%s",
            cloudflare_access_verifier.team_domain,
        )
    else:
        log.info("auth: Cloudflare Access disabled")

    application_cfg = cfg.raw.get("application", {}) or {}
    application_store = ApplicationStore(
        application_cfg.get("sqlite_path", "/data/audrey_app.sqlite")
    )
    log.info(
        "application_store: ready sqlite=%s schema=%d",
        application_store.path,
        application_store.schema_version,
    )
    from audrey.private_reviews import replay_private_deliveries

    # Deliver already committed results before failing genuinely interrupted
    # native attempts. Recovery never launches new model work.
    await replay_private_deliveries(application_store)
    await application_store.private_tasks.recover_for_restart()
    recovered_runs = await application_store.conversations.recover_interrupted_runs()
    if recovered_runs:
        log.warning(
            "application_store: recovered %d interrupted native run(s)",
            recovered_runs,
        )

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

    skill_registry = SkillRegistry.from_config(
        cfg.skills,
        known_tools=frozenset(TOOL_DECLARATIONS),
        available_tools=frozenset(tool_registry.names()),
    )
    skill_snapshot = skill_registry.snapshot()
    log.info(
        "skills: status=%s loaded=%d available=%d degraded=%d invalid=%d",
        skill_snapshot.status,
        skill_snapshot.loaded_count,
        skill_snapshot.available_count,
        skill_snapshot.degraded_count,
        skill_snapshot.invalid_count,
    )

    # Discovery refreshes the same registry captured by the graph. Serializing
    # fetch and publication prevents a background snapshot from overwriting a
    # newer manual rediscovery.
    tools_discovery_lock = asyncio.Lock()

    # One lifespan-owned custom-tools transport is reused by memory recall and
    # durable archive delivery. Per-call timeouts remain owned by each caller.
    archive_http = httpx.AsyncClient(timeout=10.0)
    graph = build_graph(
        cfg, ollama, registry, health, gate, tool_registry, archive_http,
    )

    # ─── KB stack ────────────────────────────────────────────────────
    kb_cfg = cfg.raw.get("kb", {}) or {}
    qdrant = QdrantKB(
        host=cfg.env.qdrant_host,
        port=cfg.env.qdrant_port,
        text_collection=kb_cfg.get("text_collection", "kb_text"),
        image_collection=kb_cfg.get("image_collection", "kb_images"),
        embedding_profile=profile_from_mapping(cfg.raw["text_embedding"]),
    )
    try:
        await qdrant.ensure_collections()
    except Exception as e:  # noqa: BLE001 — Qdrant outage shouldn't kill boot
        log.warning("qdrant: ensure_collections failed: %s (KB endpoints will 503)", e)

    # SQLite owns the durable upload catalog and storage lifecycle. Startup
    # reconciliation backfills missing legacy rows and reports lost Qdrant
    # content without deleting files or their catalog records. Run before
    # serving traffic so recovery and drift reporting see a stable catalog.
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
    text_embedder = build_text_embedder(ollama, cfg.raw)
    image_embedder = ImageEmbedder(
        model_name=kb_cfg.get("image_model", "clip-ViT-B-32"),
        cache_folder=kb_cfg.get("image_cache_folder", "/home/audrey/.cache/clip"),
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
    app.state.application_store = application_store
    tax_review_exports = TaxReviewExports(
        root=application_store.path.parent / "tax-review-exports"
    )
    app.state.tax_review_exports = tax_review_exports
    app.state.cloudflare_access_verifier = cloudflare_access_verifier
    app.state.ollama = ollama
    app.state.registry = registry
    app.state.health = health
    app.state.gate = gate
    app.state.inflight = inflight
    app.state.tools = tool_registry
    app.state.tools_discovery_lock = tools_discovery_lock
    app.state.skills = skill_registry
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
    reference_cfg = cfg.raw.get("reference_libraries", {})
    app.state.reference_libraries = ReferenceLibraryRegistry(reference_cfg)
    app.state.reference_releases = None
    app.state.reference_search = None
    if app.state.reference_libraries.enabled:
        try:
            releases = ReleaseStore(Path(reference_cfg.get("index_root", "/data/reference-libraries")))
            app.state.reference_search = ReferenceSearchService(
                app.state.reference_libraries, releases, qdrant,
                reference_embedder(ollama, cfg.raw),
                config=reference_cfg.get("search", {}),
            )
            app.state.reference_releases = releases
        except (OSError, ReleaseError):
            log.warning("Public reference search state unavailable; catalog/page reader remain available")

    # Durable chat-archive outbox. Response handlers commit only the small
    # local source row; this lifecycle-owned worker performs the remote
    # custom-tools call and resumes unfinished rows after restart.
    archive_transport = ChatArchiveClient(
        archive_http,
        service_token=cfg.env.kb_service_token,
    )
    archive_cfg = cfg.raw.get("chat_archive", {}) or {}
    archive_queue_cfg = archive_cfg.get("queue", {}) or {}
    archive_projection_cfg = archive_cfg.get("projection", {}) or {}
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
    app.state.archive_queue = archive_queue
    app.state.archive_client = archive_client
    app.state.archive_transport = archive_transport
    app.state.kb_service_token = cfg.env.kb_service_token

    archive_projector = ChatProjectionPromoter(
        store=application_store,
        archive_queue=archive_queue,
        retry_interval_s=float(
            archive_projection_cfg.get("retry_interval_s", 30.0)
        ),
        batch_size=int(archive_projection_cfg.get("batch_size", 50)),
    )
    await archive_projector.start(run_worker=archive_enabled)
    app.state.archive_projector = archive_projector

    purge_cfg = ((cfg.raw.get("user_data", {}) or {}).get("purge", {}) or {})

    async def purge_content_search(owner_id: str) -> None:
        service = getattr(app.state, "content_search", None)
        if service is not None:
            await service.purge_owner(owner_id)

    user_data_purges = UserDataPurgeCoordinator(
        db=uploads_db,
        file_deletions=file_deletions,
        archive_queue=archive_queue,
        archive_transport=archive_transport,
        registry=tool_registry,
        upload_root=upload_root,
        application_store=application_store,
        tax_review_exports=tax_review_exports,
        local_search_cleanup=purge_content_search,
        retry_interval_s=float(purge_cfg.get("retry_interval_s", 30.0)),
        batch_size=int(purge_cfg.get("batch_size", 50)),
    )
    await user_data_purges.start()
    app.state.user_data_purges = user_data_purges

    title_cfg = cfg.raw.get("title_generation", {}) or {}
    conversation_titles = None
    if title_cfg.get("enabled", True):
        conversation_titles = ConversationTitleGenerator(
            ollama=ollama,
            registry=registry,
            gate=gate,
            inflight=inflight,
            model=str(title_cfg.get("model") or cfg.raw["router"]["model"]),
            timeout_s=float(title_cfg.get("timeout_s", 10.0)),
            max_prompt_chars=int(title_cfg.get("max_prompt_chars", 6_000)),
            max_tokens=int(title_cfg.get("max_tokens", 32)),
            no_thinking=bool(title_cfg.get("no_thinking", True)),
        )
    app.state.conversation_titles = conversation_titles

    native_runs = NativeRunManager(
        app=app,
        store=application_store,
        archive_wake=archive_projector.wake if archive_enabled else None,
    )
    app.state.native_runs = native_runs

    saved_agent_settings = agent_settings(app)
    agent_runtime = AgentRuntime(
        application_store.agents,
        ResearchAgentExecutor(app),
        poll_seconds=saved_agent_settings.poll_seconds,
        lease_seconds=saved_agent_settings.lease_seconds,
        max_attempts=saved_agent_settings.max_attempts,
        step_timeout_seconds=saved_agent_settings.step_timeout_seconds,
    )
    app.state.agent_runtime = agent_runtime

    saved_watch_settings = watch_settings(app)
    watch_runtime = WatchRuntime(
        application_store.watches,
        WatchExecutor(app),
        poll_seconds=saved_watch_settings.poll_seconds,
        lease_seconds=saved_watch_settings.lease_seconds,
        max_attempts=saved_watch_settings.max_attempts,
        check_timeout_seconds=saved_watch_settings.check_timeout_seconds,
        run_limits=saved_watch_settings.run_limits(),
    )
    app.state.watch_runtime = watch_runtime

    search_cfg = cfg.raw.get("content_search", {}) or {}
    content_search = None
    if search_cfg.get("enabled", True):
        try:
            search_index = ContentSearchIndex(search_cfg.get("sqlite_path", "/data/content-search/index.sqlite"))
            content_search = ContentSearchService(
                application_store,
                ContentSourceReader(application_store, uploads_db, upload_root),
                search_index,
            )
        except (ContentSearchError, OSError):
            log.warning("Local content search unavailable; canonical sources remain available")
    app.state.content_search = content_search

    readiness_cfg = cfg.raw.get("readiness", {}) or {}
    required_components = set(
        readiness_cfg.get("required_components", ["ollama"]) or []
    )
    readiness = ReadinessCollector(
        app,
        required_components=required_components,
        probe_timeout_s=float(readiness_cfg.get("probe_timeout_s", 2.0)),
        cache_ttl_s=float(readiness_cfg.get("cache_ttl_s", 5.0)),
    )
    app.state.readiness = readiness
    await readiness.collect(force=True)

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
    # Start only once startup has succeeded, so a later startup exception
    # cannot strand the monitor outside the lifespan's shutdown guard.
    tools_discovery_task: asyncio.Task[None] | None = None
    if tools_enabled and tool_servers:
        tools_discovery_task = asyncio.create_task(
            _maintain_tool_discovery(
                tool_registry,
                tool_servers,
                skills=skill_registry,
                lock=tools_discovery_lock,
            ),
            name="audrey.tools.discovery",
        )
    try:
        if content_search is not None:
            try:
                await content_search.start()
            except ContentSearchError:
                await content_search.stop()
                content_search.index.close()
                content_search = None
                app.state.content_search = None
                log.warning("Local content search recovery failed; search is unavailable")
        if saved_agent_settings.enabled:
            await agent_runtime.start()
        if saved_agent_settings.enabled and saved_watch_settings.enabled:
            await watch_runtime.start()
        yield
    finally:
        await user_data_purges.stop()
        if content_search is not None:
            await content_search.stop()
            content_search.index.close()
        await watch_runtime.stop()
        await agent_runtime.stop()
        await native_runs.stop()
        if tools_discovery_task is not None and not tools_discovery_task.done():
            tools_discovery_task.cancel()
            try:
                await tools_discovery_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 — shutdown path
                pass
        if reconciler is not None:
            await reconciler.stop()
        if watcher is not None:
            await watcher.stop()
        await archive_projector.stop()
        await archive_queue.stop()
        await file_deletions.stop()
        application_store.close()
        uploads_db.close()
        qdrant.close()
        await ollama.aclose()
        await archive_http.aclose()


async def _refresh_tool_discovery(
    registry: ToolRegistry,
    tool_servers: list[str],
    *,
    skills: SkillRegistry | None = None,
    lock: asyncio.Lock | None = None,
) -> bool:
    """Fetch and publish one validated snapshot without replacing live objects.

    An empty result from configured servers is inconclusive: discovery already
    isolates unreachable and rejected servers. Keep the existing view in that
    case; explicit unavailable records still withdraw their individual tools.
    """
    async with lock if lock is not None else asyncio.Lock():
        fresh = await discover_all(tool_servers)
        if tool_servers and not fresh.by_name:
            return False
        audit_user_scoping(fresh)
        registry.by_name.clear()
        registry.by_name.update(fresh.by_name)
        if skills is not None:
            skills.refresh_availability(frozenset(registry.names()))
        return True


async def _maintain_tool_discovery(
    registry: ToolRegistry,
    tool_servers: list[str],
    *,
    skills: SkillRegistry | None = None,
    lock: asyncio.Lock | None = None,
    interval_s: float = 60.0,
) -> None:
    """Recover cold startup quickly, then track live sidecar capabilities.

    The slow refresh also runs after a healthy startup. No request waits for
    discovery; dependencies becoming unavailable or recovering are reflected
    in model tools and skill availability on a subsequent snapshot.
    """
    if not registry.by_name or any(
        not spec.available for spec in registry.policy_records()
    ):
        await _retry_tool_discovery(registry, tool_servers, skills=skills, lock=lock)
    while True:
        await asyncio.sleep(interval_s)
        try:
            refreshed = await _refresh_tool_discovery(
                registry, tool_servers, skills=skills, lock=lock,
            )
            if not refreshed:
                log.warning("tools: periodic discovery empty; keeping the live registry")
        except Exception as exc:  # noqa: BLE001 — a later snapshot may recover
            log.warning("tools: periodic discovery failed: %s", exc)


async def _retry_tool_discovery(
    registry: ToolRegistry,
    tool_servers: list[str],
    *,
    skills: SkillRegistry | None = None,
    lock: asyncio.Lock | None = None,
    attempts: int = 30,
    interval_s: float = 4.0,
) -> None:
    """Retry discovery while the initial registry is empty or degraded.

    Custom-tools may be missing or may have optional dependencies still
    recovering when Audrey starts. These rapid retries shorten recovery
    before the lifecycle monitor settles into its minute-by-minute refresh.

    The loop bails out when every declared tool is available. These rapid
    startup retries are bounded; the lifecycle monitor continues at its slow
    interval afterwards. Manual rediscovery also stays available. Default window:
    30 × 4s = 2 minutes, which covers the worst observed cold-start.
    """
    for attempt in range(1, attempts + 1):
        await asyncio.sleep(interval_s)
        try:
            refreshed = await _refresh_tool_discovery(
                registry, tool_servers, skills=skills, lock=lock,
            )
        except Exception as e:  # noqa: BLE001 — discovery is best-effort here
            log.warning("tools: retry %d/%d failed: %s", attempt, attempts, e)
            continue
        if not refreshed:
            log.info("tools: retry %d/%d still empty", attempt, attempts)
            continue
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
        "tools: fast recovery ended after %d retries (%.0fs); slow refresh and manual rediscovery remain available",
        attempts, attempts * interval_s,
    )


app = FastAPI(
    title="Audrey AI",
    version=__version__,
    description=(
        "OpenAI-compatible orchestrator over Ollama (local + cloud-bridge "
        "models) with a shared LangGraph pipeline: classify → complexity "
        "gate → fast path or planner → deep panel → synth → reflect. "
        f"{len(VIRTUAL_MODELS)} virtual models are listed by /v1/models. "
        "Tool dispatch uses the declarative catalogue at authenticated "
        "/v1/tools, discovered from custom-tools OpenAPI. Per-user fair "
        "scheduling at the local-GPU "
        "gate, Cloudflare Access and Audrey personal-token identity, "
        "Prometheus metrics at `/metrics`, KB "
        "watcher + periodic reconcile keeping global collections drift-free, "
        "streaming progress banners + per-worker tools-used footer on "
        "streamed responses."
    ),
    lifespan=lifespan,
)

app.include_router(openai_router)
app.include_router(application_router)
app.include_router(kb_router)
app.include_router(references_router)
app.include_router(evidence_tools_router)
app.include_router(tool_history_router)
app.include_router(files_router)
app.include_router(media_router)
app.include_router(user_data_router)
app.include_router(admin_router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/metrics", tags=["system"], include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Prometheus text-format exposition.

    Unauthenticated by design — Prometheus convention, and we don't
    publish the route via cloudflared, so it's effectively LAN-only
    (Unraid scrapes from the same docker network as audrey).
    """
    readiness = getattr(request.app.state, "readiness", None)
    if readiness is not None:
        try:
            await readiness.collect()
        except Exception as exc:  # noqa: BLE001 — metrics must stay scrapeable
            log.warning("readiness: metrics refresh failed: %s", exc)
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@app.get("/v1/tools", tags=["tools"])
async def list_tools(
    _principal: Principal = Depends(require_principal),
) -> dict[str, list[dict]]:
    """Let an active account inspect the registered ReAct tool catalogue."""
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
    skills = getattr(app.state, "skills", None)
    refreshed = await _refresh_tool_discovery(
        reg,
        tool_servers,
        skills=skills,
        lock=getattr(app.state, "tools_discovery_lock", None),
    )
    if not refreshed:
        log.warning("tools: rediscover returned no records; keeping the live registry")
        raise HTTPException(
            status_code=503,
            detail="Tool discovery returned no records; existing tools are unchanged.",
        )
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
