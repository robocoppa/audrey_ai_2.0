"""Native account approval, group membership, and model access controls."""

from __future__ import annotations

import io
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from audrey.app_state import AccessRoleRecord, AccountAdministrationError, AdminUserRecord
from audrey.auth import clear_auth_cache_for_user_id, require_admin_principal
from audrey.identity import Principal
from audrey.model_catalog import catalog_for_principal, discover_models
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
    groups: list[str]
    auth_provider: str
    created_at: str
    updated_at: str
    last_seen_at: str
    deletion_pending: bool = False


class AdminUserListResponse(BaseModel):
    items: list[AdminUserResponse]


class AdminUserCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    display_name: str = ""


class UserApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tester: bool = False


class AdminUserPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["active", "disabled"] | None = None
    groups: list[str] | None = None


class AdminRoleResponse(BaseModel):
    id: str
    name: str
    description: str
    system: bool
    user_count: int


class AdminRoleListResponse(BaseModel):
    items: list[AdminRoleResponse]


class AdminRoleCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""


class AdminRolePatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""


class AdminModelProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visibility: Literal["public", "private"]
    roles: list[str]
    display_name: str = ""


class AdminModelResponse(ModelResponse):
    concrete_model: str
    policy_overridden: bool


class AdminModelListResponse(BaseModel):
    items: list[AdminModelResponse]
    source: Literal["ollama", "configuration"]
    warning: str


class AdminModelPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    audience: Literal["users", "testers", "admins"]


def _role_response(record: AccessRoleRecord) -> AdminRoleResponse:
    return AdminRoleResponse(
        id=record.role_id,
        name=record.label,
        description=record.description,
        system=record.system,
        user_count=record.user_count,
    )


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
        deletion_pending=record.deletion_pending,
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


@router.post("/users", response_model=AdminUserResponse, status_code=201)
async def create_pending_user(
    payload: AdminUserCreateRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminUserResponse:
    try:
        record = await application_store(request).admin_create_pending_user(
            actor_user_id=principal.user_id,
            email=payload.email,
            display_name=payload.display_name,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    return _user_response(record)


@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    request: Request,
    status: Literal["pending", "active", "disabled"] | None = None,
    group: str | None = Query(default=None, max_length=32),
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


@router.delete("/users/{user_id}", status_code=202)
async def delete_user(
    user_id: str,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> dict[str, str]:
    coordinator = getattr(request.app.state, "user_data_purges", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="user_data_purge_unavailable")
    try:
        purge_id, _ = await application_store(request).begin_admin_account_deletion(
            actor_user_id=principal.user_id,
            target_user_id=user_id,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    clear_auth_cache_for_user_id(user_id)
    coordinator.wake()
    return {"id": user_id, "status": "deleting", "purge_id": purge_id}


@router.get("/roles", response_model=AdminRoleListResponse)
async def list_roles(
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminRoleListResponse:
    del principal
    roles = await application_store(request).list_access_roles()
    return AdminRoleListResponse(items=[_role_response(role) for role in roles])


@router.post("/roles", response_model=AdminRoleResponse, status_code=201)
async def create_role(
    payload: AdminRoleCreateRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminRoleResponse:
    try:
        role = await application_store(request).create_access_role(
            actor_user_id=principal.user_id,
            role_id=payload.id,
            label=payload.name,
            description=payload.description,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    return _role_response(role)


@router.patch("/roles/{role_id}", response_model=AdminRoleResponse)
async def update_role(
    role_id: str,
    payload: AdminRolePatchRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminRoleResponse:
    try:
        role = await application_store(request).update_access_role(
            actor_user_id=principal.user_id,
            role_id=role_id,
            label=payload.name,
            description=payload.description,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    return _role_response(role)


@router.delete("/roles/{role_id}", status_code=204)
async def delete_role(
    role_id: str,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> None:
    try:
        await application_store(request).delete_access_role(
            actor_user_id=principal.user_id, role_id=role_id
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc


@router.get("/models", response_model=AdminModelListResponse)
async def list_models(
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelListResponse:
    store = application_store(request)
    overridden = {
        policy.model_id for policy in await store.list_model_access_policies()
    } | {
        profile.model_id for profile in await store.list_model_publication_profiles()
    }
    inventory = await discover_models(
        request.app.state.cfg,
        getattr(request.app.state, "ollama", None),
    )
    models = await catalog_for_principal(
        request.app.state.cfg,
        store,
        principal,
        include_hidden=True,
        inventory=inventory.models,
    )
    return AdminModelListResponse(
        items=[
            _admin_model_response(
                model,
                policy_overridden=model.id in overridden,
            )
            for model in models
        ],
        source=inventory.source,
        warning=inventory.warning,
    )


@router.patch("/models/{model_id:path}", response_model=AdminModelResponse)
async def update_model(
    model_id: str,
    payload: AdminModelPatchRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelResponse:
    inventory = await discover_models(
        request.app.state.cfg,
        getattr(request.app.state, "ollama", None),
    )
    known = {model.id for model in inventory.models}
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
        inventory=inventory.models,
    )
    updated = next(model for model in models if model.id == model_id)
    return _admin_model_response(updated, policy_overridden=True)


@router.patch(
    "/model-profiles/{model_id:path}", response_model=AdminModelResponse
)
async def update_model_profile(
    model_id: str,
    payload: AdminModelProfileRequest,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelResponse:
    inventory = await discover_models(
        request.app.state.cfg, getattr(request.app.state, "ollama", None)
    )
    if model_id not in {model.id for model in inventory.models}:
        raise HTTPException(status_code=404, detail="Model does not exist.")
    store = application_store(request)
    try:
        await store.set_model_publication_profile(
            actor_user_id=principal.user_id,
            model_id=model_id,
            visibility=payload.visibility,
            roles=payload.roles,
            display_name=payload.display_name,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    models = await catalog_for_principal(
        request.app.state.cfg, store, principal,
        include_hidden=True, inventory=inventory.models,
    )
    return _admin_model_response(
        next(model for model in models if model.id == model_id),
        policy_overridden=True,
    )


@router.put(
    "/model-portraits/{model_id:path}", response_model=AdminModelResponse
)
async def upload_model_portrait(
    model_id: str,
    request: Request,
    portrait: UploadFile = File(...),
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelResponse:
    inventory = await discover_models(
        request.app.state.cfg, getattr(request.app.state, "ollama", None)
    )
    if model_id not in {model.id for model in inventory.models}:
        raise HTTPException(status_code=404, detail="Model does not exist.")
    raw = await portrait.read(2_000_001)
    await portrait.close()
    if len(raw) > 2_000_000 or not raw:
        raise HTTPException(status_code=422, detail="Portrait must be under 2 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Unsupported image format.")
            if source.width > 2048 or source.height > 2048:
                raise ValueError("Portrait dimensions must be at most 2048 pixels.")
            image = ImageOps.exif_transpose(source)
            output = io.BytesIO()
            image.save(output, format="WEBP", quality=85)
            data = output.getvalue()
            if len(data) > 2_000_000:
                raise ValueError("Encoded portrait exceeds 2 MB.")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid model portrait.") from exc
    try:
        store = application_store(request)
        await store.set_model_portrait(
            actor_user_id=principal.user_id, model_id=model_id,
            mime="image/webp", data=data,
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    models = await catalog_for_principal(
        request.app.state.cfg, store, principal,
        include_hidden=True, inventory=inventory.models,
    )
    return _admin_model_response(
        next(model for model in models if model.id == model_id),
        policy_overridden=True,
    )


@router.delete(
    "/model-portraits/{model_id:path}", response_model=AdminModelResponse
)
async def remove_model_portrait(
    model_id: str,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelResponse:
    inventory = await discover_models(
        request.app.state.cfg, getattr(request.app.state, "ollama", None)
    )
    if model_id not in {model.id for model in inventory.models}:
        raise HTTPException(status_code=404, detail="Model does not exist.")
    store = application_store(request)
    try:
        await store.delete_model_portrait(
            actor_user_id=principal.user_id, model_id=model_id
        )
    except AccountAdministrationError as exc:
        raise _admin_error(exc) from exc
    models = await catalog_for_principal(
        request.app.state.cfg, store, principal,
        include_hidden=True, inventory=inventory.models,
    )
    return _admin_model_response(
        next(model for model in models if model.id == model_id),
        policy_overridden=True,
    )


@router.delete(
    "/model-policies/{model_id:path}",
    response_model=AdminModelResponse,
)
async def delete_model_policy(
    model_id: str,
    request: Request,
    principal: Principal = Depends(require_admin_principal),
) -> AdminModelResponse:
    inventory = await discover_models(
        request.app.state.cfg,
        getattr(request.app.state, "ollama", None),
    )
    known = {model.id for model in inventory.models}
    if model_id not in known:
        raise HTTPException(status_code=404, detail="Model does not exist.")
    store = application_store(request)
    try:
        await store.delete_model_access_policy(
            actor_user_id=principal.user_id,
            model_id=model_id,
        )
        await store.delete_model_publication_profile(
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
        inventory=inventory.models,
    )
    updated = next(model for model in models if model.id == model_id)
    return _admin_model_response(updated, policy_overridden=False)


__all__ = ["router"]
