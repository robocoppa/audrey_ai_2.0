"""Admin-only ops endpoints.

All routes here depend on `require_admin`, which pins access to OWUI
users with `role == "admin"`. Non-admins get 403; missing/invalid token
gets 401 from the underlying `require_user` chain.

Endpoints:
  POST /v1/admin/auth/clear          — evict every cached AuthedUser.
                                       Used after revoking a token in OWUI,
                                       or as an incident-response lever.
  POST /v1/admin/auth/clear/{email}  — evict cache entries for one user.
                                       Used after deleting/banning a user in
                                       OWUI; surgically clears their sessions
                                       without disturbing other users.
  GET  /v1/admin/auth/status         — cache size visibility.
  GET  /v1/admin/repair-status       — aggregate durable repair counts.
  GET  /v1/admin/readiness           — components, queues, workers, pressure.
  POST /v1/admin/repair              — wake local repair owners and run one
                                       bounded sidecar repair pass.
  POST /v1/admin/chat_archive/prune  — apply the chat archive's retention
                                       policy on demand (SQLite rows + Qdrant
                                       points older than the cutoff).
  GET  /v1/admin/chat_archive/stats  — chat-archive row counts (messages,
                                       chunks, chunks_unindexed).
  POST /v1/admin/kb/reconcile        — trigger one KB reconcile sweep on
                                       demand. Useful after a bulk delete on
                                       disk, or to confirm the periodic loop
                                       is functioning.

The tool-rediscovery admin route (`POST /v1/tools/rediscover`) is
admin-gated by the same `require_admin` dependency but lives in `main.py`,
not here — it closes over `app.state` to mutate the live ToolRegistry.

Eviction is intentionally manual — OWUI doesn't notify Audrey of user
lifecycle events, so the admin operator (you) signals via these
endpoints. The 30s cache TTL is the upper bound on staleness if you
forget; the targeted endpoint cuts that to ~0s for a known-deleted user.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ValidationError

from audrey.auth import (
    AuthedUser,
    cache_size,
    clear_auth_cache,
    clear_auth_cache_for_email,
    require_admin,
)
from audrey.kb.reconcile import reconcile_once
from audrey.readiness import ReadinessStatus
from audrey.routes.user_data import RepairQueueStatus, UserDataRepairStatus

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


class AuthClearResponse(BaseModel):
    cleared: int
    by: str


class AuthClearForUserResponse(BaseModel):
    cleared: int
    email: str
    by: str


class AuthStatusResponse(BaseModel):
    cached_entries: int


class RepairTriggerComponent(BaseModel):
    available: bool
    accepted: bool


class AdminRepairTriggerResponse(BaseModel):
    schema_version: int = 1
    status: Literal["accepted", "partial"]
    file_deletions: RepairTriggerComponent
    chat_delivery: RepairTriggerComponent
    chat_archive: RepairTriggerComponent
    account_purges: RepairTriggerComponent


def _repair_queue(
    value: dict | None = None,
    *,
    available: bool = True,
) -> RepairQueueStatus:
    if not available:
        return RepairQueueStatus(available=False)
    if not isinstance(value, dict):
        raise HTTPException(status_code=502, detail="repair_backend_invalid_response")
    try:
        return RepairQueueStatus.model_validate(value)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail="repair_backend_invalid_response",
        ) from exc


def _repair_state(queues: tuple[RepairQueueStatus, ...]) -> str:
    if any(queue.exhausted for queue in queues):
        return "attention_required"
    if any(not queue.available for queue in queues):
        return "degraded"
    if any(queue.pending for queue in queues):
        return "repairing"
    return "ready"


@router.post("/auth/clear", response_model=AuthClearResponse)
async def auth_clear(me: AuthedUser = Depends(require_admin)) -> AuthClearResponse:
    """Force every cached token to re-probe OWUI on its next request.

    The caller's own cache entry goes too. Their next admin call will
    transparently re-probe — this is the correct behavior, not a bug.
    """
    n = clear_auth_cache()
    log.warning("admin: auth cache cleared by %s (%d entries evicted)", me.email, n)
    return AuthClearResponse(cleared=n, by=me.email)


@router.post("/auth/clear/{email}", response_model=AuthClearForUserResponse)
async def auth_clear_for_user(
    email: str,
    me: AuthedUser = Depends(require_admin),
) -> AuthClearForUserResponse:
    """Evict cache entries for one user by email. Other users untouched.

    Surgical replacement for blanket-clearing the cache after deleting a
    user in OWUI. Idempotent — calling twice for the same email returns 0
    the second time. Case-insensitive match.
    """
    n = clear_auth_cache_for_email(email)
    log.warning("admin: auth cache cleared for %s by %s (%d entries evicted)",
                email, me.email, n)
    return AuthClearForUserResponse(cleared=n, email=email, by=me.email)


@router.get("/auth/status", response_model=AuthStatusResponse)
async def auth_status(_: AuthedUser = Depends(require_admin)) -> AuthStatusResponse:
    """Quick visibility: how many entries the auth cache currently holds."""
    return AuthStatusResponse(cached_entries=cache_size())


@router.get("/readiness", response_model=ReadinessStatus)
async def readiness_status(
    request: Request,
    response: Response,
    _: AuthedUser = Depends(require_admin),
) -> ReadinessStatus:
    """Current sanitized readiness; 503 only for required component failure."""
    collector = getattr(request.app.state, "readiness", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="readiness_unavailable")
    snapshot = await collector.collect(force=True)
    if snapshot.status == "unready":
        response.status_code = 503
    return snapshot


@router.get("/repair-status", response_model=UserDataRepairStatus)
async def repair_status(
    request: Request,
    _: AuthedUser = Depends(require_admin),
) -> UserDataRepairStatus:
    """Global repair counts without user identities, payloads, or raw errors."""
    uploads_db = getattr(request.app.state, "uploads_db", None)
    file_stats = getattr(uploads_db, "file_deletion_stats", None)
    file_deletions = (
        _repair_queue(await file_stats())
        if callable(file_stats)
        else _repair_queue(available=False)
    )
    purge_stats = getattr(uploads_db, "data_purge_stats", None)
    account_purges = (
        _repair_queue(await purge_stats())
        if callable(purge_stats)
        else _repair_queue(available=False)
    )

    archive_queue = getattr(request.app.state, "archive_client", None)
    delivery_stats = getattr(archive_queue, "repair_stats", None)
    chat_delivery = (
        _repair_queue(await delivery_stats())
        if callable(delivery_stats)
        else _repair_queue(available=False)
    )

    transport = getattr(request.app.state, "archive_transport", None)
    remote_status = getattr(transport, "repair_status", None)
    if callable(remote_status):
        try:
            remote = await remote_status(registry=request.app.state.tools)
        except (httpx.HTTPError, RuntimeError, TimeoutError, ValueError):
            remote = None
    else:
        remote = None
    if remote is None:
        chat_indexing = _repair_queue(available=False)
        chat_deletions = _repair_queue(available=False)
        conversation_deletions = _repair_queue(available=False)
    else:
        chat_indexing = _repair_queue(remote.get("indexing"))
        chat_deletions = _repair_queue(remote.get("deletions"))
        conversation_deletions = _repair_queue(
            remote.get("conversation_deletions")
        )

    queues = (
        file_deletions,
        chat_delivery,
        chat_indexing,
        chat_deletions,
        conversation_deletions,
        account_purges,
    )
    return UserDataRepairStatus(
        status=_repair_state(queues),
        file_deletions=file_deletions,
        chat_delivery=chat_delivery,
        chat_indexing=chat_indexing,
        chat_deletions=chat_deletions,
        conversation_deletions=conversation_deletions,
        account_purges=account_purges,
    )


@router.post(
    "/repair",
    response_model=AdminRepairTriggerResponse,
    status_code=202,
)
async def repair(
    request: Request,
    me: AuthedUser = Depends(require_admin),
) -> AdminRepairTriggerResponse:
    """Wake local owners and run one bounded sidecar repair pass."""
    file_worker = getattr(request.app.state, "file_deletions", None)
    file_wake = getattr(file_worker, "wake", None)
    file_component = RepairTriggerComponent(
        available=callable(file_wake),
        accepted=callable(file_wake),
    )
    if callable(file_wake):
        file_wake()

    archive_queue = getattr(request.app.state, "archive_client", None)
    retry_delivery = getattr(archive_queue, "retry_now", None)
    delivery_component = RepairTriggerComponent(
        available=callable(retry_delivery),
        accepted=callable(retry_delivery),
    )
    if callable(retry_delivery):
        await retry_delivery()

    purge_coordinator = getattr(request.app.state, "user_data_purges", None)
    purge_wake = getattr(purge_coordinator, "wake", None)
    purge_component = RepairTriggerComponent(
        available=callable(purge_wake),
        accepted=callable(purge_wake),
    )
    if callable(purge_wake):
        purge_wake()

    transport = getattr(request.app.state, "archive_transport", None)
    remote_repair = getattr(transport, "repair", None)
    remote_available = callable(remote_repair)
    remote_accepted = False
    if callable(remote_repair):
        try:
            await remote_repair(registry=request.app.state.tools)
        except (httpx.HTTPError, RuntimeError, TimeoutError, ValueError):
            pass
        else:
            remote_accepted = True
    remote_component = RepairTriggerComponent(
        available=remote_available and remote_accepted,
        accepted=remote_accepted,
    )

    components = (
        file_component,
        delivery_component,
        remote_component,
        purge_component,
    )
    status_value = (
        "accepted"
        if all(item.available and item.accepted for item in components)
        else "partial"
    )
    log.warning(
        "admin: repair triggered by %s status=%s local_file=%s "
        "local_delivery=%s sidecar=%s account_purge=%s",
        me.email,
        status_value,
        file_component.accepted,
        delivery_component.accepted,
        remote_component.accepted,
        purge_component.accepted,
    )
    return AdminRepairTriggerResponse(
        status=status_value,
        file_deletions=file_component,
        chat_delivery=delivery_component,
        chat_archive=remote_component,
        account_purges=purge_component,
    )


@router.post("/chat_archive/prune")
async def chat_archive_prune(
    request: Request,
    me: AuthedUser = Depends(require_admin),
) -> dict:
    """Apply the chat archive's retention policy on demand.

    Honors `CHAT_ARCHIVE_RETENTION_DAYS` and also retries an exhausted
    deletion outbox after an operator has repaired the upstream failure.
    Scheduled maintenance enforces the same policy between manual calls.
    """
    archive_client = getattr(request.app.state, "archive_client", None)
    registry = request.app.state.tools
    if archive_client is None:
        raise HTTPException(status_code=503, detail="archive_client_unavailable")
    host = archive_client.host_url(registry)
    if host is None:
        raise HTTPException(status_code=503, detail="chat_history_search_not_registered")
    async with httpx.AsyncClient(timeout=30.0) as http:
        r = await http.post(f"{host}/chat_history/prune")
    log.warning("admin: chat_archive prune triggered by %s; result=%s",
                me.email, r.text[:200])
    if r.status_code >= 400:
        return {"error": "prune_failed", "status": r.status_code, "body": r.text[:500]}
    return r.json()


@router.get("/chat_archive/stats")
async def chat_archive_stats(
    request: Request,
    _: AuthedUser = Depends(require_admin),
) -> dict:
    """Counts and latest repair state from the chat archive.

    Includes retryable/exhausted index and deletion counts plus the latest
    attempt timestamps and bounded error strings.
    """
    archive_client = getattr(request.app.state, "archive_client", None)
    registry = request.app.state.tools
    if archive_client is None:
        raise HTTPException(status_code=503, detail="archive_client_unavailable")
    host = archive_client.host_url(registry)
    if host is None:
        raise HTTPException(status_code=503, detail="chat_history_search_not_registered")
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.get(f"{host}/chat_history/stats")
    if r.status_code >= 400:
        return {"error": "stats_failed", "status": r.status_code}
    result = r.json()
    queue_stats = getattr(archive_client, "stats", None)
    if callable(queue_stats):
        result["delivery_queue"] = await queue_stats()
    return result


@router.post("/kb/reconcile")
async def kb_reconcile(
    request: Request,
    me: AuthedUser = Depends(require_admin),
) -> dict:
    """Trigger one KB reconcile sweep on demand.

    Scrolls `kb_text` + `kb_images`, deletes vectors for any source path
    that no longer exists on disk. Returns the structured summary
    (per-collection: checked/orphans_deleted/points_in_orphans/elapsed_s).

    Per-user collections are NOT touched — those are reconciled by the
    sqlite uploads index at startup.

    Synchronous: call returns when the sweep completes. On a 10k-point
    KB this is sub-second; on a much larger one it can take longer.
    The periodic loop (configured via `kb.reconcile.interval_s`) runs
    independently — calling this endpoint doesn't reset its timer.
    """
    qdrant = request.app.state.qdrant
    reconciler = getattr(request.app.state, "kb_reconciler", None)
    run_once = getattr(reconciler, "run_once", None)
    if callable(run_once):
        result = await run_once()
    else:
        result = await reconcile_once(
            qdrant,
            text_collection=qdrant.text_collection,
            image_collection=qdrant.image_collection,
        )
    log.warning("admin: kb reconcile triggered by %s; orphans_deleted=%d",
                me.email, result.total_orphans_deleted)
    return result.to_dict()


__all__ = ["router"]
