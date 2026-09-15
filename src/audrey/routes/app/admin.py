"""Native account approval, group membership, and model access controls."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from audrey.app_state import AccountAdministrationError, AdminUserRecord
from audrey.auth import clear_auth_cache_for_user_id, require_admin_principal
from audrey.identity import Principal
from audrey.model_catalog import catalog_for_principal, configured_models
from audrey.routes.app.models import (
    ModelResponse,
    application_store,
    model_response,
)

router = APIRouter(prefix="/admin", tags=["application-admin"])


class AdminUserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    status: Literal["pending", "active", "disabled"]
    groups: list[Literal["users", "testers", "admins"]]
    auth_provider: str
    created_at: str
    updated_at: str
    last_seen_at: str


class AdminUserListResponse(BaseModel):
    items: list[AdminUserResponse]


class UserApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tester: bool = False


class AdminUserPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["active", "disabled"] | None = None
    groups: list[Literal["users", "testers", "admins"]] | None = None


class AdminModelResponse(ModelResponse):
    concrete_model: str
    policy_overridden: bool


class AdminModelListResponse(BaseModel):
    items: list[AdminModelResponse]


class AdminModelPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    audience: Literal["users", "testers", "admins"]


def _user_response(record: AdminUserRecord) -> AdminUserResponse:
    return AdminUserResponse(
        id=record.user_id,
        email=record.email,
        display_name=record.display_name,
        role=record.role,
        status=record.status,
        groups=list(record.groups),
        auth_provider=record.auth_provider,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_seen_at=record.last_seen_at,
    )


def _admin_model_response(model, *, policy_overridden: bool) -> AdminModelResponse:
    base = model_response(model).model_dump()
    return AdminModelResponse(
        **base,
        concrete_model=model.concrete_model,
        policy_overridden=policy_overridden,
    )


def _admin_error(exc: AccountAdministrationError) -> HTTPException:
    detail = str(exc)
    status_code = 404 if "does not exist" in detail else 409
    return HTTPException(status_code=status_code, detail=detail)


@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    request: Request,
    status: Literal["pending", "active", "disabled"] | None = None,
    group: Literal["users", "testers", "admins"] | None = None,
    search: str = Query(default="", max_length=200),
    principal: Principal = Depends(require_admin_principal),
) -> AdminUserListResponse:
    del principal
    records = await application_store(request).list_admin_users(
        status=status or "",
        group=group or "",
        search=search,
    )
    return AdminUserListResponse(items=[_user_response(record) for record in records])


@router.post("/users/{user_id}/approve", response_model=AdminUserResponse)
async def approve_user(
    user_id: str,
    payload: UserApprovalRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminUserResponse:
    groups = ["users", *(["testers"] if payload.tester else [])]
    try:
        record = await application_store(request).admin_update_user(
            actor_user_id=principal.user_id,
            target_user_id=user_id,
            status="active",
            groups=groups,
            action="approve_user",
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    clear_auth_cache_for_user_id(user_id)
    return _user_response(record)


@router.post("/users/{user_id}/deny", response_model=AdminUserResponse)
async def deny_user(
    user_id: str,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminUserResponse:
    try:
        record = await application_store(request).admin_update_user(
            actor_user_id=principal.user_id,
            target_user_id=user_id,
            status="disabled",
            groups=[],
            action="deny_user",
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    clear_auth_cache_for_user_id(user_id)
    return _user_response(record)


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
async def update_user(
    user_id: str,
    payload: AdminUserPatchRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminUserResponse:
    if payload.status is None and payload.groups is None:
        raise HTTPException(status_code=422, detail="Account update has no fields.")
    try:
        record = await application_store(request).admin_update_user(
            actor_user_id=principal.user_id,
            target_user_id=user_id,
            status=payload.status,
            groups=payload.groups,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    clear_auth_cache_for_user_id(user_id)
    return _user_response(record)


@router.get("/models", response_model=AdminModelListResponse)
async def list_models(
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelListResponse:
    store = application_store(request)
    overridden = {
        policy.model_id for policy in await store.list_model_access_policies()
    }
    models = await catalog_for_principal(
        request.app.state.cfg,
        store,
        principal,
        include_hidden=True,
    )
    return AdminModelListResponse(
        items=[
            _admin_model_response(
                model,
                policy_overridden=model.id in overridden,
            )
            for model in models
        ]
    )


@router.patch("/models/{model_id:path}", response_model=AdminModelResponse)
async def update_model(
    model_id: str,
    payload: AdminModelPatchRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelResponse:
    known = {model.id for model in configured_models(request.app.state.cfg)}
    if model_id not in known:
        raise HTTPException(status_code=404, detail="Model does not exist.")
    try:
        await application_store(request).set_model_access_policy(
            actor_user_id=principal.user_id,
            model_id=model_id,
            enabled=payload.enabled,
            audience=payload.audience,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    models = await catalog_for_principal(
        request.app.state.cfg,
        application_store(request),
        principal,
        include_hidden=True,
    )
    updated = next(model for model in models if model.id == model_id)
    return _admin_model_response(updated, policy_overridden=True)


@router.delete(
    "/model-policies/{model_id:path}",
    response_model=AdminModelResponse,
)
async def delete_model_policy(
    model_id: str,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelResponse:
    known = {model.id for model in configured_models(request.app.state.cfg)}
    if model_id not in known:
        raise HTTPException(status_code=404, detail="Model does not exist.")
    store = application_store(request)
    try:
        await store.delete_model_access_policy(
            actor_user_id=principal.user_id,
            model_id=model_id,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    models = await catalog_for_principal(
        request.app.state.cfg,
        store,
        principal,
        include_hidden=True,
    )
    updated = next(model for model in models if model.id == model_id)
    return _admin_model_response(updated, policy_overridden=False)


__all__ = ["router"]
