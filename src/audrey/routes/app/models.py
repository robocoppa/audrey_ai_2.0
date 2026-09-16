"""Group-filtered model catalog for the native Audrey application."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.identity import Principal
from audrey.model_catalog import ServedModel, catalog_for_principal

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


__all__ = ["ModelListResponse", "ModelResponse", "application_store", "model_response", "router"]
