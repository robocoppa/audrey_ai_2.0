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

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from audrey.auth import (
    AuthedUser,
    cache_size,
    clear_auth_cache,
    clear_auth_cache_for_email,
    require_admin,
)
from audrey.kb.reconcile import reconcile_once

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
    result = await reconcile_once(
        qdrant,
        text_collection=qdrant.text_collection,
        image_collection=qdrant.image_collection,
    )
    log.warning("admin: kb reconcile triggered by %s; orphans_deleted=%d",
                me.email, result.total_orphans_deleted)
    return result.to_dict()


__all__ = ["router"]
