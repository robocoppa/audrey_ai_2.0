"""Owner-only saved Tax review case and explicit user confirmation actions."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.tax_review import (
    CreateTaxCaseRequest,
    TaxCaseActionRequest,
    apply_tax_case_action,
    create_tax_case,
    ensure_tax_project,
    get_project_tax_cases,
    get_tax_case,
)

router = APIRouter(tags=["application-tax-review"])
_access = require_scope("compat:full")


@router.get("/conversations/{conversation_id}/tax-review")
async def get_case(
    conversation_id: str, request: Request, principal: Principal = Depends(_access),
) -> dict:
    return {"case": await get_tax_case(request.app, principal, conversation_id)}


@router.get("/projects/{project_id}/tax-reviews")
async def project_cases(
    project_id: str, request: Request, principal: Principal = Depends(_access),
) -> dict:
    return {"items": await get_project_tax_cases(request.app, principal, project_id)}


@router.post("/conversations/{conversation_id}/tax-review", status_code=201)
async def create_case(
    conversation_id: str, body: CreateTaxCaseRequest, request: Request,
    principal: Principal = Depends(_access),
) -> dict:
    return await create_tax_case(request.app, principal, conversation_id, body)


@router.post("/conversations/{conversation_id}/tax-review/project")
async def ensure_case_project(
    conversation_id: str, request: Request, principal: Principal = Depends(_access),
) -> dict:
    return await ensure_tax_project(request.app, principal, conversation_id)


@router.post("/conversations/{conversation_id}/tax-review/actions")
async def case_action(
    conversation_id: str, body: TaxCaseActionRequest, request: Request,
    principal: Principal = Depends(_access),
) -> dict:
    return await apply_tax_case_action(request.app, principal, conversation_id, body)


class PrivateTaskAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    action: Literal["pause", "cancel"]


@router.get("/conversations/{conversation_id}/private-task")
async def get_private_task(conversation_id: str, request: Request,
                           principal: Principal = Depends(_access)) -> dict:
    from audrey.app_state.private_tasks import PrivateTaskAccessError
    from audrey.private_reviews import public_private_task
    from audrey.routes.app.runs import _manager

    try:
        task = await _manager(request).private_reviews.latest(principal.user_id, conversation_id)
    except PrivateTaskAccessError:
        raise HTTPException(status_code=404, detail="Private review not found.") from None
    return {"task": public_private_task(task)}


@router.post("/private-tasks/{task_id}/actions")
async def private_task_action(task_id: str, body: PrivateTaskAction, request: Request,
                              principal: Principal = Depends(_access)) -> dict:
    from audrey.app_state.private_tasks import PrivateTaskAccessError, PrivateTaskError
    from audrey.private_reviews import public_private_task
    from audrey.routes.app.runs import _manager

    try:
        task = await _manager(request).private_reviews.control(
            user_id=principal.user_id, task_id=task_id,
            expected_revision=body.expected_revision, action=body.action,
        )
    except PrivateTaskAccessError:
        raise HTTPException(status_code=404, detail="Private review not found.") from None
    except PrivateTaskError:
        raise HTTPException(status_code=409, detail="Private review changed; reload it.") from None
    if task is None:
        raise HTTPException(status_code=404, detail="Private review not found.")
    return {"task": public_private_task(task)}
