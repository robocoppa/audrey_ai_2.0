"""Admin-only ops endpoints.

All routes here depend on `require_admin`, which pins access to OWUI
users with `role == "admin"`. Non-admins get 403; missing/invalid token
gets 401 from the underlying `require_user` chain.

Currently a single endpoint:
  POST /v1/admin/auth/clear — evict every cached AuthedUser. Used to
                              force a re-probe across all active sessions
                              (e.g. after revoking a token in OWUI, or
                              when the cache TTL is too long for an
                              incident response).

Eviction is blunt by design — there's no per-token or per-email variant.
The auth cache is small (capped at ~1024 entries) and the cost of a
clear is one extra OWUI probe per active user on their next request,
which is fine. If that ever becomes painful, add targeted eviction.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from audrey.auth import AuthedUser, cache_size, clear_auth_cache, require_admin

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


class AuthClearResponse(BaseModel):
    cleared: int
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


@router.get("/auth/status", response_model=AuthStatusResponse)
async def auth_status(_: AuthedUser = Depends(require_admin)) -> AuthStatusResponse:
    """Quick visibility: how many entries the auth cache currently holds."""
    return AuthStatusResponse(cached_entries=cache_size())


__all__ = ["router"]
