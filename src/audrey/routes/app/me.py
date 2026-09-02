"""Authenticated native account resource."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from audrey.auth import require_principal
from audrey.identity import Principal

router = APIRouter(tags=["application"])


class MeResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    status: str
    auth_provider: str


@router.get("/me", response_model=MeResponse)
async def get_me(
    principal: Principal = Depends(require_principal),
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


__all__ = ["router"]
