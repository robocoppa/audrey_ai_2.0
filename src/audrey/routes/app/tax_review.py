"""Owner-only saved Tax review case and explicit user confirmation actions."""
from fastapi import APIRouter, Depends, Request

from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.tax_review import (
    CreateTaxCaseRequest,
    TaxCaseActionRequest,
    apply_tax_case_action,
    create_tax_case,
    get_tax_case,
)

router = APIRouter(tags=["application-tax-review"])
_access = require_scope("compat:full")


@router.get("/conversations/{conversation_id}/tax-review")
async def get_case(
    conversation_id: str, request: Request, principal: Principal = Depends(_access),
) -> dict:
    return {"case": await get_tax_case(request.app, principal, conversation_id)}


@router.post("/conversations/{conversation_id}/tax-review", status_code=201)
async def create_case(
    conversation_id: str, body: CreateTaxCaseRequest, request: Request,
    principal: Principal = Depends(_access),
) -> dict:
    return await create_tax_case(request.app, principal, conversation_id, body)


@router.post("/conversations/{conversation_id}/tax-review/actions")
async def case_action(
    conversation_id: str, body: TaxCaseActionRequest, request: Request,
    principal: Principal = Depends(_access),
) -> dict:
    return await apply_tax_case_action(request.app, principal, conversation_id, body)
