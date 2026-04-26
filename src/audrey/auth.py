"""OWUI-backed authentication (Phase 14).

Validates `Authorization: Bearer <jwt>` tokens by proxying them to
Open WebUI's session endpoint (`GET /api/v1/auths/`, trailing slash
load-bearing — OWUI 0.9.2 specifically). OWUI is the single source of
identity truth; Audrey never issues or signs its own tokens.

Flow:
    browser --Bearer <jwt>--> cloudflared --same-origin--> audrey
                                                            │
                                                            ▼
                                              GET http://open-webui:8080
                                              /api/v1/auths/
                                              Authorization: Bearer <jwt>
                                                            │
                                                            ▼
                                              {id, email, role, ...}
                                                            │
                                            AuthedUser(email, role, owui_id)

Token results are cached per-token for `_TTL_S` seconds (default 30s)
to spare OWUI on bursty dashboards. The cache is a plain dict with an
opportunistic sweep at 1024 entries — fine for our scale (dozens of
users, not thousands).

We only ever read the `Authorization` header. OWUI also sets an
HttpOnly `token` cookie on same-origin responses — we ignore it.
Reasoning: (1) the upload page carries the token explicitly from
localStorage, so we never *need* the cookie; (2) accepting cookies
opens a CSRF path from any logged-in OWUI user's browser. Requiring
the explicit header closes that.

401 propagates from OWUI as-is. 502 on any other OWUI error — we want
the client to see "auth is broken" distinctly from "you're not logged
in." Timeouts are short (5s); OWUI lives on `ollama-net` at sub-ms
latency, so a 5s ceiling means OWUI is actually down, not slow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx
from fastapi import Depends, Header, HTTPException, Request

log = logging.getLogger(__name__)

_TTL_S: float = 30.0
_SWEEP_AT: int = 1024
_PROBE_TIMEOUT_S: float = 5.0

# Roles allowed past the gate. OWUI emits these as lowercase. `pending` —
# a user the admin hasn't activated — has a valid JWT but no chat access
# in OWUI, so we reject them too. Anything outside this set (a future
# `disabled` state, garbled response) also fails closed.
_ALLOWED_ROLES: frozenset[str] = frozenset({"user", "admin"})


@dataclass(slots=True)
class AuthedUser:
    """Identity returned from OWUI. `email` is the canonical user id.

    Matches what Phase 11 memory + Phase 13 uploads have been keying on
    since those phases shipped, so threading this through `user_id`
    everywhere requires zero downstream changes.
    """

    email: str
    role: str
    owui_id: str


_cache: dict[str, tuple[float, AuthedUser]] = {}


def _sweep_cache(now: float) -> None:
    """Drop expired entries. Opportunistic — only called when cache is large."""
    cutoff = now - _TTL_S
    for k, (t, _) in list(_cache.items()):
        if t < cutoff:
            _cache.pop(k, None)


async def _probe_owui(owui_url: str, token: str) -> AuthedUser:
    """Single OWUI probe call. Raises HTTPException on 401 / upstream failure."""
    url = f"{owui_url.rstrip('/')}/api/v1/auths/"
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as http:
            r = await http.get(url, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as e:
        log.warning("auth: OWUI unreachable at %s: %s", url, e)
        raise HTTPException(status_code=502, detail="Auth backend unreachable.") from e

    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="Token rejected by OWUI.")
    if r.status_code >= 400:
        log.warning("auth: OWUI probe -> %d: %s", r.status_code, r.text[:200])
        raise HTTPException(status_code=502, detail=f"Auth probe failed ({r.status_code}).")

    try:
        body = r.json()
    except ValueError as e:
        raise HTTPException(status_code=502, detail="Auth probe returned non-JSON.") from e

    email = body.get("email")
    if not email:
        raise HTTPException(status_code=502, detail="OWUI response missing email.")
    role = str(body.get("role") or "").lower()
    if role not in _ALLOWED_ROLES:
        # OWUI returns 200 for `pending` users — the JWT is valid; the user
        # just isn't activated. Reject here so they can't sneak past the
        # gate via Audrey while OWUI itself blocks them.
        log.info("auth: rejecting %s with role=%r (not in %s)", email, role, sorted(_ALLOWED_ROLES))
        raise HTTPException(status_code=401, detail=f"Account not activated (role={role!r}).")
    return AuthedUser(
        email=str(email),
        role=role,
        owui_id=str(body.get("id") or ""),
    )


async def require_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> AuthedUser:
    """FastAPI dependency — inject `AuthedUser` into route handlers.

    Use as `me: AuthedUser = Depends(require_user)`. Returns 401 on
    missing/invalid token, 502 if OWUI is down. Every route that writes
    or lists user-scoped data MUST depend on this.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token.")

    now = time.monotonic()
    cached = _cache.get(token)
    if cached is not None and now - cached[0] < _TTL_S:
        return cached[1]

    owui_url = request.app.state.cfg.env.owui_url
    user = await _probe_owui(owui_url, token)

    _cache[token] = (now, user)
    if len(_cache) > _SWEEP_AT:
        _sweep_cache(now)
    return user


async def require_admin(me: AuthedUser = Depends(require_user)) -> AuthedUser:
    """Like `require_user`, but additionally enforces `role == "admin"`.

    OWUI v0.9.2 emits role strings as lowercase (`"admin"`, `"user"`,
    `"pending"`). 403 — not 401 — because the caller is authenticated;
    they just don't have permission.
    """
    if me.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return me


def clear_auth_cache() -> int:
    """Drop every cached AuthedUser. Returns the number of entries evicted.

    Used by `POST /v1/admin/auth/clear` and tests. Self-evicts the caller's
    own cache row too — that's intentional. Their next request re-probes
    OWUI, which is exactly the point.
    """
    n = len(_cache)
    _cache.clear()
    return n


def cache_size() -> int:
    """Current count of cached AuthedUser entries. For admin observability."""
    return len(_cache)


__all__ = [
    "AuthedUser", "require_user", "require_admin",
    "clear_auth_cache", "cache_size",
]
