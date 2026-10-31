"""Native owner-bound saved research agents and lifecycle controls."""

from __future__ import annotations

import logging
import sqlite3
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
logger = logging.getLogger(__name__)


class AgentStepResponse(BaseModel):
    id: Literal["plan", "research", "review"]
    label: str
    status: Literal["pending", "running", "completed"]
    result: str


class AgentInputResponse(BaseModel):
    id: str
    step_id: Literal["plan"]
    question: str
    answer: str | None
    created_at: str
    answered_at: str | None


class AgentTaskResponse(BaseModel):
    id: str
    objective: str
    title: str = ""
    status: Literal["queued", "running", "paused", "waiting_for_input", "completed", "cancelled", "failed"]
    revision: int
    created_at: str
    updated_at: str
    steps: list[AgentStepResponse]
    result: str
    error: str
    attempts: int
    input_requests: list[AgentInputResponse] = Field(default_factory=list)


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


class AgentInputAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_revision: int = Field(ge=1, strict=True)
    answer: str = Field(min_length=1, max_length=8000)


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
        repository = _repository(request)
        tasks = await repository.list(user_id=principal.user_id, limit=200, summary=True)
        repository.fence_response(user_id=principal.user_id, tasks=tasks)
        return {"items": tasks}
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
        repository.fence_response(user_id=principal.user_id, tasks=[task])
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
        repository = _repository(request)
        task = await repository.get(user_id=principal.user_id, task_id=task_id)
        repository.fence_response(user_id=principal.user_id, tasks=[task])
        return task
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
        repository.fence_response(user_id=principal.user_id, tasks=[task])
    except AgentDispatchStoppedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except AgentTaskError as error:
        raise _error(error) from None
    _wake(request)
    return task


@router.post("/agents/{task_id}/inputs/{request_id}/answer", response_model=AgentTaskResponse)
async def answer_agent_input(
    task_id: str,
    request_id: str,
    body: AgentInputAnswerRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(_agents_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    repository = _repository(request)
    try:
        await repository.get(user_id=principal.user_id, task_id=task_id)
        await research_authority(request.app, principal.user_id)
        settings = agent_settings(request.app)
        task = await repository.answer_input(
            user_id=principal.user_id, task_id=task_id, request_id=request_id,
            expected_revision=body.expected_revision, answer=body.answer,
            max_attempts=settings.max_attempts, admission_limits=settings.task_limits(),
        )
        repository.fence_response(user_id=principal.user_id, tasks=[task])
    except AgentDispatchStoppedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except AgentTaskError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        logger.warning("Agent input storage unavailable (%s)", type(error).__name__)
        raise HTTPException(status_code=503, detail="Saved agent input is temporarily unavailable.") from None
    _wake(request)
    return task
