"""Authenticated native account resource."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from audrey.app_state import (
    ApplicationStore,
    InvalidApplicationStateError,
    InvalidIdentityError,
    UserPreferences,
)
from audrey.auth import (
    clear_auth_cache_for_email,
    require_provider_principal,
    require_scope,
)
from audrey.identity import PersonalTokenSummary, Principal
from audrey.pipeline.context import normalize_response_preferences

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


class MeUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)


class PreferencesResponse(BaseModel):
    timezone: str
    persona: str
    detail: Literal["concise", "balanced", "detailed"]
    tone: Literal["natural", "professional", "casual"]
    show_progress: bool
    created_at: str
    updated_at: str


class PreferencesUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str = Field(min_length=1, max_length=100)
    persona: str = Field(max_length=4_000)
    detail: Literal["concise", "balanced", "detailed"]
    tone: Literal["natural", "professional", "casual"]
    show_progress: bool


def _preferences_response(record: UserPreferences) -> PreferencesResponse:
    response = normalize_response_preferences(record.response_preferences)
    return PreferencesResponse(
        timezone=record.timezone,
        persona=record.persona,
        detail=response["detail"],
        tone=response["tone"],
        show_progress=response["show_progress"],
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


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


@router.patch("/me", response_model=MeResponse)
async def update_me(
    payload: MeUpdateRequest,
    request: Request,
    principal: Principal = Depends(require_provider_principal),
) -> MeResponse:
    """Update the current Audrey account's profile name only."""

    try:
        display_name = await _store(request).update_display_name(
            user_id=principal.user_id,
            display_name=payload.display_name,
        )
    except InvalidIdentityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    clear_auth_cache_for_email(principal.email)
    return MeResponse(
        id=principal.user_id,
        email=principal.email,
        display_name=display_name,
        role=principal.role,
        status=principal.status,
        auth_provider=principal.provider,
    )


@router.get("/me/preferences", response_model=PreferencesResponse)
async def get_preferences(
    request: Request,
    principal: Principal = Depends(_account_read),
) -> PreferencesResponse:
    preferences = await _store(request).preferences.get(user_id=principal.user_id)
    if preferences is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey account preferences are not initialized.",
        )
    return _preferences_response(preferences)


@router.put("/me/preferences", response_model=PreferencesResponse)
async def update_preferences(
    payload: PreferencesUpdateRequest,
    request: Request,
    principal: Principal = Depends(require_provider_principal),
) -> PreferencesResponse:
    try:
        preferences = await _store(request).preferences.replace(
            user_id=principal.user_id,
            timezone=payload.timezone,
            persona=payload.persona,
            response_preferences={
                "detail": payload.detail,
                "tone": payload.tone,
                "show_progress": payload.show_progress,
            },
        )
    except InvalidApplicationStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if preferences is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey account preferences are not initialized.",
        )
    return _preferences_response(preferences)


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
