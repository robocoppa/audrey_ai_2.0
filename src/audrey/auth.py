"""Cloudflare Access and Audrey personal-token authentication.

External providers prove identity while Audrey owns the stable user id, role,
and storage namespace. Personal tokens resolve locally on every request so
revocation and account changes take effect immediately. Unknown bearers and
cookies never authenticate a caller.
"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from audrey.app_state import (
    IdentityConflictError,
    InvalidIdentityError,
    PersonalTokenAuthenticationError,
)
from audrey.identity import (
    CloudflareAccessTokenError,
    CloudflareAccessUnavailableError,
    Principal,
)

log = logging.getLogger(__name__)

_PERSONAL_TOKEN_PREFIX = "aud_pat_"  # noqa: S105 — public format discriminator


async def _resolve_personal_token(
    request: Request,
    token: str,
) -> Principal:
    """Resolve an Audrey bearer locally with current account and token state."""

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

    return principal


async def _resolve_cloudflare_access(
    request: Request,
    token: str,
    *,
    allow_inactive: bool = False,
) -> Principal:
    """Bind verified Access email to Audrey while keeping roles Audrey-owned."""

    verifier = getattr(request.app.state, "cloudflare_access_verifier", None)
    if verifier is None:
        raise HTTPException(status_code=401, detail="Cloudflare Access authentication is disabled.")
    try:
        claims = await verifier.verify(token)
    except CloudflareAccessTokenError as exc:
        raise HTTPException(status_code=401, detail="Cloudflare Access token is invalid.") from exc
    except CloudflareAccessUnavailableError as exc:
        log.warning("auth: Cloudflare Access verification unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Auth provider verification unavailable.",
        ) from exc

    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application identity is not initialized.",
        )
    try:
        principal = await store.resolve_external_identity(
            provider="cloudflare_access",
            subject=claims.subject,
            email=claims.email,
            display_name="",
            role="user",
            auth_method="cloudflare_access",
            legacy_storage_namespace=None,
            sync_role=False,
            sync_display_name=False,
            initial_status="pending",
        )
    except InvalidIdentityError as exc:
        log.error("auth: invalid identity evidence from Cloudflare Access: %s", exc)
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

    if not allow_inactive and principal.status != "active":
        raise HTTPException(status_code=403, detail=_inactive_account_detail(principal))
    return principal


async def require_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    cloudflare_access_jwt: Annotated[
        str | None,
        Header(alias="Cf-Access-Jwt-Assertion"),
    ] = None,
) -> Principal:
    """Authenticate one active Audrey account from Access or a personal token."""
    verifier = getattr(request.app.state, "cloudflare_access_verifier", None)
    if verifier is not None and cloudflare_access_jwt:
        # A bad Access assertion must fail closed. Never fall through to a
        # bearer that might happen to be present on the same request.
        return await _resolve_cloudflare_access(request, cloudflare_access_jwt)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token.")

    if token.startswith(_PERSONAL_TOKEN_PREFIX):
        return await _resolve_personal_token(request, token)

    raise HTTPException(status_code=401, detail="Unsupported bearer token.")


async def require_account_user(
    request: Request,
    authorization: str | None = Header(default=None),
    cloudflare_access_jwt: Annotated[
        str | None,
        Header(alias="Cf-Access-Jwt-Assertion"),
    ] = None,
) -> Principal:
    """Authenticate `/api/me` while exposing pending Cloudflare account status."""

    verifier = getattr(request.app.state, "cloudflare_access_verifier", None)
    if verifier is not None and cloudflare_access_jwt:
        return await _resolve_cloudflare_access(
            request,
            cloudflare_access_jwt,
            allow_inactive=True,
        )
    return await require_principal(request, authorization, cloudflare_access_jwt)


async def require_account_principal(
    principal: Principal = Depends(require_account_user),
) -> Principal:
    """Return account status, enforcing account:read only for personal tokens."""
    if principal.auth_method == "personal_token" and "account:read" not in principal.scopes:
        raise HTTPException(
            status_code=403,
            detail="Personal access token lacks account:read scope.",
        )
    return principal


async def require_user(
    request: Request,
    authorization: str | None = Header(default=None),
    cloudflare_access_jwt: Annotated[
        str | None,
        Header(alias="Cf-Access-Jwt-Assertion"),
    ] = None,
) -> Principal:
    """Require an active compatibility-API caller with compat:full for tokens."""
    principal = await require_principal(request, authorization, cloudflare_access_jwt)
    if principal.auth_method == "personal_token" and "compat:full" not in principal.scopes:
        raise HTTPException(
            status_code=403,
            detail="Personal access token lacks compat:full scope.",
        )
    return principal


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


async def require_admin_principal(
    principal: Principal = Depends(require_provider_principal),
) -> Principal:
    """Require provider auth from an active Audrey administrator."""

    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return principal


def verify_service_token(presented: str | None, expected: str) -> bool:
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented, expected)


async def require_service(
    request: Request,
    x_audrey_service_token: str | None = Header(default=None),
) -> None:
    """Require the shared internal credential for trusted service operations."""
    expected = request.app.state.cfg.env.kb_service_token
    if not verify_service_token(x_audrey_service_token, expected):
        raise HTTPException(status_code=401, detail="Service token required.")


def _inactive_account_detail(principal: Principal) -> str:
    if principal.status == "pending":
        return "Audrey account is awaiting administrator approval."
    return "Audrey account is disabled."


__all__ = [
    "require_account_principal",
    "require_account_user",
    "require_admin_principal",
    "require_user",
    "require_principal",
    "require_scope",
    "require_provider_principal",
    "verify_service_token",
    "require_service",
]
