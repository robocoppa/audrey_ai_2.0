"""Group-filtered model catalog for the native Audrey application."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.identity import Principal
from audrey.model_catalog import ServedModel, catalog_for_principal, resolve_model

router = APIRouter(tags=["application-models"])


class ModelResponse(BaseModel):
    id: str
    label: str
    description: str
    kind: Literal["workflow", "direct"]
    mode: str
    presentation: str
    capabilities: list[str]
    enabled: bool
    audience: Literal["users", "testers", "admins"]
    visibility: Literal["public", "private"]
    roles: list[str]
    portrait_url: str


class ModelListResponse(BaseModel):
    items: list[ModelResponse]


def model_response(model: ServedModel) -> ModelResponse:
    return ModelResponse(
        id=model.id,
        label=model.label,
        description=model.description,
        kind=model.kind,
        mode=model.mode,
        presentation=model.presentation,
        capabilities=list(model.capabilities),
        enabled=model.enabled,
        audience=model.audience,
        visibility=model.visibility,
        roles=list(model.roles),
        portrait_url=model.portrait_url,
    )


def application_store(request: Request) -> ApplicationStore:
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application identity is not initialized.",
        )
    return store


@router.get("/models", response_model=ModelListResponse)
async def list_application_models(
    request: Request,
    principal: Principal = Depends(require_principal),
) -> ModelListResponse:
    models = await catalog_for_principal(
        request.app.state.cfg,
        application_store(request),
        principal,
        ollama=getattr(request.app.state, "ollama", None),
    )
    return ModelListResponse(items=[model_response(model) for model in models])


@router.get("/model-portraits/{model_id:path}")
async def model_portrait(
    model_id: str,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> Response:
    store = application_store(request)
    model = await resolve_model(
        request.app.state.cfg,
        store,
        principal,
        model_id,
        ollama=getattr(request.app.state, "ollama", None),
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Model does not exist.")
    portrait = await store.get_model_portrait(model_id=model_id)
    if portrait is None:
        raise HTTPException(status_code=404, detail="Model portrait does not exist.")
    mime, data = portrait
    return Response(content=data, media_type=mime, headers={"Cache-Control": "private, no-store"})


__all__ = ["ModelListResponse", "ModelResponse", "application_store", "model_response", "router"]
