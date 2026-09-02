"""Authenticated native account resource."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from audrey.app_state import ApplicationStore, InvalidIdentityError
from audrey.auth import require_provider_principal, require_scope
from audrey.identity import PersonalTokenSummary, Principal

router = APIRouter(tags=["application"])


class TokenCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[Literal["account:read", "compat:full"]] = Field(
        default_factory=lambda: ["compat:full"],
        min_length=1,
    )
    expires_in_days: int = Field(default=90, ge=1, le=365)


class TokenRecordResponse(BaseModel):
    id: str
    name: str
    scopes: list[str]
    created_at: str
    expires_at: str
    last_used_at: str | None
    revoked_at: str | None


class TokenCreateResponse(TokenRecordResponse):
    token: str


class TokenListResponse(BaseModel):
    items: list[TokenRecordResponse]


class TokenRevokeResponse(BaseModel):
    id: str
    revoked: bool


_account_read = require_scope("account:read")


def _store(request: Request) -> ApplicationStore:
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application identity is not initialized.",
        )
    return store


def _token_response(record: PersonalTokenSummary) -> TokenRecordResponse:
    return TokenRecordResponse(
        id=record.token_id,
        name=record.name,
        scopes=list(record.scopes),
        created_at=record.created_at,
        expires_at=record.expires_at or None,
        last_used_at=record.last_used_at or None,
        revoked_at=record.revoked_at or None,
    )


class MeResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    status: str
    auth_provider: str


@router.get("/me", response_model=MeResponse)
async def get_me(
    principal: Principal = Depends(_account_read),
) -> MeResponse:
    """Return the Audrey-owned account behind current auth evidence.

    Provider subjects and storage namespaces stay server-side. The browser gets
    the stable Audrey id and mutable profile fields, never an identity selector.
    """

    return MeResponse(
        id=principal.user_id,
        email=principal.email,
        display_name=principal.display_name,
        role=principal.role,
        status=principal.status,
        auth_provider=principal.provider,
    )


@router.post(
    "/tokens",
    response_model=TokenCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_token(
    payload: TokenCreateRequest,
    request: Request,
    principal: Principal = Depends(require_provider_principal),
) -> TokenCreateResponse:
    """Issue a bearer secret once; only its SHA-256 digest remains at rest."""

    expires_at = (
        dt.datetime.now(dt.UTC) + dt.timedelta(days=payload.expires_in_days)
    ).isoformat(timespec="microseconds")
    try:
        issued = await _store(request).create_personal_token(
            user_id=principal.user_id,
            name=payload.name,
            scopes=payload.scopes,
            expires_at=expires_at,
        )
    except InvalidIdentityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = _token_response(issued.record)
    return TokenCreateResponse(token=issued.token, **record.model_dump())


@router.get("/tokens", response_model=TokenListResponse)
async def list_tokens(
    request: Request,
    principal: Principal = Depends(require_provider_principal),
) -> TokenListResponse:
    records = await _store(request).list_personal_tokens(
        user_id=principal.user_id,
    )
    return TokenListResponse(items=[_token_response(record) for record in records])


@router.delete("/tokens/{token_id}", response_model=TokenRevokeResponse)
async def revoke_token(
    token_id: str,
    request: Request,
    principal: Principal = Depends(require_provider_principal),
) -> TokenRevokeResponse:
    revoked = await _store(request).revoke_personal_token(
        user_id=principal.user_id,
        token_id=token_id,
    )
    if not revoked:
        raise HTTPException(status_code=404, detail="Token not found.")
    return TokenRevokeResponse(id=token_id, revoked=True)


__all__ = ["router"]
