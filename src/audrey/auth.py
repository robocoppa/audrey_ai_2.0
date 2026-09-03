"""Authentication adapters and provider-neutral Audrey principal resolution.

Accepts `Authorization: Bearer <credential>`. Audrey personal tokens use an
`aud_pat_` discriminator and resolve locally; other bearers are proxied to the
Open WebUI session endpoint (`GET /api/v1/auths/`, trailing slash
load-bearing — OWUI 0.9.2 specifically). During native-application migration,
OWUI proves the external identity while Audrey owns the stable user id and
private-storage namespace.

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
                                                            ▼
                                              ApplicationStore resolves
                                              provider subject -> user_id
                                                            │
                                                            ▼
                                              AuthedUser(..., principal)

OWUI results are cached per-token for `_TTL_S` seconds (default 30s)
to spare OWUI on bursty dashboards. The cache is a plain dict with an
opportunistic sweep at 1024 entries — fine for our scale (dozens of
users, not thousands).

We only ever read the `Authorization` header. OWUI also sets an
HttpOnly `token` cookie on same-origin responses — we ignore it.
Reasoning: (1) the upload page carries the token explicitly from
localStorage, so we never *need* the cookie; (2) accepting cookies
opens a CSRF path from any logged-in OWUI user's browser. Requiring
the explicit header closes that.

Invalid, expired, revoked, and disabled-account personal tokens return 401 and
are never cached. OWUI 401 propagates as-is. 502 on any other OWUI error — we want
the client to see "auth is broken" distinctly from "you're not logged
in." Timeouts are short (5s); OWUI lives on `ollama-net` at sub-ms
latency, so a 5s ceiling means OWUI is actually down, not slow.
"""

from __future__ import annotations

import hmac
import logging
import time
from dataclasses import dataclass

import httpx
from fastapi import Depends, Header, HTTPException, Request

from audrey.app_state import (
    IdentityConflictError,
    InvalidIdentityError,
    PersonalTokenAuthenticationError,
)
from audrey.identity import Principal
from audrey.metrics import auth_cache_size as _auth_cache_size_gauge

log = logging.getLogger(__name__)

_TTL_S: float = 30.0
_SWEEP_AT: int = 1024
_PROBE_TIMEOUT_S: float = 5.0
_PERSONAL_TOKEN_PREFIX = "aud_pat_"  # noqa: S105 — public format discriminator

# Roles allowed past the gate. OWUI emits these as lowercase. `pending` —
# a user the admin hasn't activated — has a valid JWT but no chat access
# in OWUI, so we reject them too. Anything outside this set (a future
# `disabled` state, garbled response) also fails closed.
_ALLOWED_ROLES: frozenset[str] = frozenset({"user", "admin"})


@dataclass(slots=True)
class AuthedUser:
    """Compatibility identity returned from the current OWUI adapter.

    Existing `/v1` memory and upload routes still key on exact ``email`` during
    migration. Native `/api` resources consume ``principal`` and its stable
    Audrey user id. There is deliberately no ambiguous ``id`` field here.
    """

    email: str
    role: str
    owui_id: str
    display_name: str = ""
    principal: Principal | None = None


@dataclass(slots=True)
class KBCaller:
    email: str | None
    is_service: bool


_cache: dict[str, tuple[float, AuthedUser]] = {}


def _sweep_cache(now: float) -> None:
    """Drop expired entries. Opportunistic — only called when cache is large."""
    cutoff = now - _TTL_S
    for k, (t, _) in list(_cache.items()):
        if t < cutoff:
            _cache.pop(k, None)
    _auth_cache_size_gauge.set(len(_cache))


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
        display_name=str(body.get("name") or body.get("display_name") or ""),
    )


async def _resolve_personal_token(
    request: Request,
    token: str,
) -> AuthedUser:
    """Resolve an Audrey bearer locally and enforce compatibility policy.

    Personal tokens never inherit provider-admin authority on `/v1`; that
    avoids stale admin access if the external provider later demotes a user.
    """

    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application identity is not initialized.",
        )
    try:
        principal = await store.authenticate_personal_token(token)
    except PersonalTokenAuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="Personal access token is invalid.",
        ) from exc

    if request.url.path.startswith("/v1/") and "compat:full" not in principal.scopes:
        raise HTTPException(
            status_code=403,
            detail="Personal access token lacks compat:full scope.",
        )

    return AuthedUser(
        email=principal.storage_namespace,
        role="user",
        owui_id="",
        display_name=principal.display_name,
        principal=principal,
    )


async def _bind_audrey_principal(
    request: Request,
    user: AuthedUser,
) -> AuthedUser:
    """Map OWUI evidence to one durable Audrey account when the store exists."""

    store = getattr(request.app.state, "application_store", None)
    if store is None:
        return user
    if not user.owui_id:
        raise HTTPException(
            status_code=502,
            detail="Auth provider response missing stable subject.",
        )
    try:
        user.principal = await store.resolve_external_identity(
            provider="owui",
            subject=user.owui_id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            auth_method="owui_bearer",
            legacy_storage_namespace=user.email,
        )
    except InvalidIdentityError as exc:
        log.error("auth: invalid identity evidence from OWUI: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Auth provider returned invalid identity evidence.",
        ) from exc
    except IdentityConflictError as exc:
        log.error("auth: refusing implicit account merge: %s", exc)
        raise HTTPException(
            status_code=409,
            detail="Audrey identity binding conflict.",
        ) from exc
    return user


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

    if token.startswith(_PERSONAL_TOKEN_PREFIX):
        return await _resolve_personal_token(request, token)

    now = time.monotonic()
    cached = _cache.get(token)
    if cached is not None and now - cached[0] < _TTL_S:
        return cached[1]

    owui_url = request.app.state.cfg.env.owui_url
    user = await _bind_audrey_principal(
        request,
        await _probe_owui(owui_url, token),
    )

    _cache[token] = (now, user)
    _auth_cache_size_gauge.set(len(_cache))
    if len(_cache) > _SWEEP_AT:
        _sweep_cache(now)
    return user


async def require_principal(
    me: AuthedUser = Depends(require_user),
) -> Principal:
    """Resolve native application routes to one active Audrey principal."""

    if me.principal is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application identity is not initialized.",
        )
    if me.principal.status != "active":
        raise HTTPException(status_code=403, detail="Audrey account is disabled.")
    return me.principal


def require_scope(scope: str):
    """Build a dependency that enforces scopes only for personal tokens."""

    async def _dependency(
        principal: Principal = Depends(require_principal),
    ) -> Principal:
        if principal.auth_method == "personal_token" and scope not in principal.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Personal access token lacks {scope} scope.",
            )
        return principal

    return _dependency


async def require_provider_principal(
    principal: Principal = Depends(require_principal),
) -> Principal:
    """Require external-provider authentication for credential management."""

    if principal.auth_method == "personal_token":
        raise HTTPException(
            status_code=403,
            detail="External provider authentication is required for this operation.",
        )
    return principal


async def require_admin(me: AuthedUser = Depends(require_user)) -> AuthedUser:
    """Like `require_user`, but additionally enforces `role == "admin"`.

    OWUI v0.9.2 emits role strings as lowercase (`"admin"`, `"user"`,
    `"pending"`). 403 — not 401 — because the caller is authenticated;
    they just don't have permission.
    """
    if me.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return me


def verify_service_token(presented: str | None, expected: str) -> bool:
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented, expected)


async def require_service(
    request: Request,
    x_audrey_service_token: str | None = Header(default=None),
) -> None:
    """Service-token-only gate, for routes no browser should ever reach.

    Distinct from `resolve_kb_caller`, which accepts *either* a service token
    or a user JWT. The video job routes (Phase 33) hand out filesystem paths
    and accept results on behalf of an arbitrary user, so holding a valid user
    token must not be enough to reach them — otherwise any logged-in user could
    claim another's video and post a transcript into their collection.
    """
    expected = request.app.state.cfg.env.kb_service_token
    if not verify_service_token(x_audrey_service_token, expected):
        raise HTTPException(status_code=401, detail="Service token required.")


async def resolve_kb_caller(
    request: Request,
    x_audrey_service_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> KBCaller:
    expected = request.app.state.cfg.env.kb_service_token
    if verify_service_token(x_audrey_service_token, expected):
        return KBCaller(email=None, is_service=True)
    me = await require_user(request, authorization)
    return KBCaller(email=me.email, is_service=False)


def clear_auth_cache() -> int:
    """Drop every cached AuthedUser. Returns the number of entries evicted.

    Used by `POST /v1/admin/auth/clear` and tests. Self-evicts the caller's
    own cache row too — that's intentional. Their next request re-probes
    OWUI, which is exactly the point.
    """
    n = len(_cache)
    _cache.clear()
    _auth_cache_size_gauge.set(0)
    return n


def clear_auth_cache_for_email(email: str) -> int:
    """Drop every cached entry whose `AuthedUser.email == email`.

    Targeted variant of `clear_auth_cache`. A user can have multiple cached
    tokens (multi-device, multi-session) — this clears all of theirs at once
    without disturbing other users' entries. Returns the number evicted.

    OWUI v0.9.x doesn't emit user-deletion webhooks, so when an admin deletes
    or bans a user in OWUI, this endpoint is the supported way to evict
    immediately rather than waiting for TTL or wiping the whole cache.
    """
    if not email:
        return 0
    target = email.strip().lower()
    if not target:
        return 0
    to_evict = [k for k, (_, u) in _cache.items() if u.email.lower() == target]
    for k in to_evict:
        _cache.pop(k, None)
    if to_evict:
        _auth_cache_size_gauge.set(len(_cache))
    return len(to_evict)


def cache_size() -> int:
    """Current count of cached AuthedUser entries. For admin observability."""
    return len(_cache)


__all__ = [
    "AuthedUser",
    "KBCaller",
    "require_user",
    "require_principal",
    "require_scope",
    "require_provider_principal",
    "require_admin",
    "verify_service_token",
    "resolve_kb_caller",
    "require_service",
    "clear_auth_cache",
    "clear_auth_cache_for_email",
    "cache_size",
]
