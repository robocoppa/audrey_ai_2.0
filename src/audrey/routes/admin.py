"""Admin-only ops endpoints.

All routes here depend on `require_admin`, which pins access to OWUI
users with `role == "admin"`. Non-admins get 403; missing/invalid token
gets 401 from the underlying `require_user` chain.

Endpoints:
  POST /v1/admin/auth/clear         — evict every cached AuthedUser.
                                      Used after revoking a token in OWUI,
                                      or as an incident-response lever.
  POST /v1/admin/auth/clear/{email} — evict only entries for one user
                                      (Phase 22). Used after deleting a
                                      user in OWUI — eviction without
                                      disturbing other sessions. Phase 22
                                      shipped this because OWUI v0.9.x
                                      doesn't emit user-deletion webhooks.
  GET  /v1/admin/auth/status        — cache size visibility.

Eviction is intentionally manual — OWUI doesn't notify Audrey of user
lifecycle events, so the admin operator (you) signals via these
endpoints. The 30s cache TTL is the upper bound on staleness if you
forget; the targeted endpoint cuts that to ~0s for a known-deleted user.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from audrey.auth import (
    AuthedUser,
    cache_size,
    clear_auth_cache,
    clear_auth_cache_for_email,
    require_admin,
)

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

    Phase 22: surgical replacement for blanket-clearing the cache after
    deleting a user in OWUI. Idempotent — calling twice for the same
    email returns 0 the second time. Case-insensitive match.
    """
    n = clear_auth_cache_for_email(email)
    log.warning("admin: auth cache cleared for %s by %s (%d entries evicted)",
                email, me.email, n)
    return AuthClearForUserResponse(cleared=n, email=email, by=me.email)


@router.get("/auth/status", response_model=AuthStatusResponse)
async def auth_status(_: AuthedUser = Depends(require_admin)) -> AuthStatusResponse:
    """Quick visibility: how many entries the auth cache currently holds."""
    return AuthStatusResponse(cached_entries=cache_size())


__all__ = ["router"]
