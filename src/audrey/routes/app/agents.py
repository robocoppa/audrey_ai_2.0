"""Native owner-bound saved research agents and lifecycle controls."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_executor import agent_settings, research_authority
from audrey.app_state.agents import AgentTaskAccessError, AgentTaskConflictError, AgentTaskError
from audrey.auth import require_scope
from audrey.identity import Principal

router = APIRouter(tags=["application-agents"])
_agents_access = require_scope("compat:full")


class AgentStepResponse(BaseModel):
    id: Literal["plan", "research", "review"]
    label: str
    status: Literal["pending", "running", "completed"]
    result: str


class AgentTaskResponse(BaseModel):
    id: str
    objective: str
    status: Literal["queued", "running", "paused", "completed", "cancelled", "failed"]
    revision: int
    created_at: str
    updated_at: str
    steps: list[AgentStepResponse]
    result: str
    error: str
    attempts: int


class AgentTaskSummaryResponse(AgentTaskResponse):
    summary: Literal[True] = True


class AgentListResponse(BaseModel):
    items: list[AgentTaskSummaryResponse]


class AgentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    objective: str = Field(min_length=1, max_length=8000)
    request_id: str = Field(min_length=1, max_length=200)


class AgentActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "cancel", "rerun"]
    expected_revision: int = Field(ge=1, strict=True)


def _repository(request: Request):
    store = getattr(request.app.state, "application_store", None)
    if store is None or not hasattr(store, "agents"):
        raise HTTPException(status_code=503, detail="Saved agents are unavailable.")
    return store.agents


def _error(error: AgentTaskError) -> HTTPException:
    code = 404 if isinstance(error, AgentTaskAccessError) else (
        409 if isinstance(error, AgentTaskConflictError) else 422
    )
    return HTTPException(status_code=code, detail=str(error))


def _wake(request: Request) -> None:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is not None:
        runtime.wake()


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(request: Request, response: Response,
                      principal: Principal = Depends(_agents_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return {"items": await _repository(request).list(user_id=principal.user_id, limit=200, summary=True)}
    except AgentTaskError as error:
        raise _error(error) from None


@router.post("/agents", response_model=AgentTaskResponse, status_code=201)
async def create_agent(body: AgentCreateRequest, request: Request, response: Response,
                       principal: Principal = Depends(_agents_access)):
    response.headers["Cache-Control"] = "private, no-store"
    repository = _repository(request)
    try:
        await research_authority(request.app, principal.user_id)
        settings = agent_settings(request.app)
        task = await repository.create(user_id=principal.user_id, objective=body.objective,
                                       request_id=body.request_id, limits=settings.task_limits())
    except AgentDispatchStoppedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except AgentTaskError as error:
        raise _error(error) from None
    _wake(request)
    return task


@router.delete("/agents/{task_id}")
async def delete_agent(task_id: str, request: Request, response: Response,
                       expected_revision: int = Query(ge=1),
                       principal: Principal = Depends(_agents_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        deleted = await _repository(request).delete(user_id=principal.user_id, task_id=task_id,
                                                   expected_revision=expected_revision)
    except AgentTaskError as error:
        raise _error(error) from None
    return {"deleted": deleted}


@router.get("/agents/{task_id}", response_model=AgentTaskResponse)
async def get_agent(task_id: str, request: Request, response: Response,
                    principal: Principal = Depends(_agents_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await _repository(request).get(user_id=principal.user_id, task_id=task_id)
    except AgentTaskError as error:
        raise _error(error) from None


@router.post("/agents/{task_id}/actions", response_model=AgentTaskResponse)
async def agent_action(
    task_id: str,
    body: AgentActionRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(_agents_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    repository = _repository(request)
    try:
        # Resolve ownership before checking execution grants for queued work.
        await repository.get(user_id=principal.user_id, task_id=task_id)
        if body.action in {"resume", "rerun"}:
            await research_authority(request.app, principal.user_id)
        options = {}
        if body.action == "rerun":
            settings = agent_settings(request.app)
            options = {
                "max_attempts": settings.max_attempts,
                "admission_limits": settings.task_limits(),
            }
        task = await repository.control(
            user_id=principal.user_id,
            task_id=task_id,
            expected_revision=body.expected_revision,
            action=body.action,
            **options,
        )
    except AgentDispatchStoppedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except AgentTaskError as error:
        raise _error(error) from None
    _wake(request)
    return task
