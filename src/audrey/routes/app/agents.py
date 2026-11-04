"""Native owner-bound saved agents and lifecycle controls."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Coroutine
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_executor import agent_settings, research_authority
from audrey.agent_knowledge_drafts import DraftError, ReportKnowledgeDrafts
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


class AgentKnowledgePinResponse(BaseModel):
    id: str
    revision: int


class AgentProjectContextResponse(BaseModel):
    project_id: str
    project_name: str | None
    available: bool
    current: bool
    instructions_included: bool
    knowledge: list[AgentKnowledgePinResponse]
    origin_entry: AgentKnowledgePinResponse | None = Field(default=None, exclude_if=lambda value: value is None)


class AgentOutputResponse(BaseModel):
    file_id: str
    filename: str
    download_url: str
    created_at: str


class AgentFileInputResponse(BaseModel):
    file_id: str
    filename: str
    artifact: Literal["text", "transcript"]
    chars: int = Field(ge=1)
    total_chars: int = Field(ge=1)
    truncated: bool
    origin: Literal["upload", "generated"]


class AgentFileInputsResponse(BaseModel):
    current: bool
    files: list[AgentFileInputResponse] = Field(min_length=1, max_length=3)


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
    project_context: AgentProjectContextResponse | None = None
    outputs: list[AgentOutputResponse] = Field(
        default_factory=list, max_length=8, exclude_if=lambda value: not value,
    )
    file_inputs: AgentFileInputsResponse | None = Field(default=None, exclude_if=lambda value: value is None)


class AgentTaskSummaryResponse(AgentTaskResponse):
    summary: Literal[True] = True
    outputs: list[AgentOutputResponse] = Field(default_factory=list, exclude=True)


class AgentListResponse(BaseModel):
    items: list[AgentTaskSummaryResponse]


class AgentEventResponse(BaseModel):
    seq: int = Field(ge=1)
    attempt: int = Field(ge=1)
    stage: Literal["plan", "research", "review", "title"]
    kind: Literal["stage", "model", "tool"]
    state: Literal["started", "completed", "failed", "interrupted"]
    tool: Literal["web_search", "web_fetch", "create_file", "read_selected_file"] | None
    created_at: str
    finished_at: str | None
    duration_ms: int | None = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)


class AgentEventsResponse(BaseModel):
    items: list[AgentEventResponse] = Field(max_length=50)
    has_more: bool
    truncated: bool


class AgentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    objective: str = Field(min_length=1, max_length=8000)
    request_id: str = Field(min_length=1, max_length=200)
    project_id: str | None = Field(default=None, min_length=1, max_length=300)
    knowledge_entry_id: str | None = Field(default=None, min_length=1, max_length=300)
    knowledge_entry_revision: int | None = Field(default=None, ge=1, le=2**63 - 1, strict=True)
    file_ids: list[Annotated[str, Field(min_length=1, max_length=200, strict=True)]] = Field(
        default_factory=list, max_length=3,
    )

    @model_validator(mode="after")
    def validate_knowledge_origin(self):
        if (self.knowledge_entry_id is None) != (self.knowledge_entry_revision is None):
            raise ValueError("Project knowledge identity and revision must be supplied together.")
        if self.knowledge_entry_id is not None and self.project_id is None:
            raise ValueError("Project knowledge tasks require a Project.")
        if self.file_ids and self.project_id is None:
            raise ValueError("Selected files require a Project.")
        if len(set(self.file_ids)) != len(self.file_ids):
            raise ValueError("Choose each file only once.")
        return self


class AgentActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "cancel", "rerun"]
    expected_revision: int = Field(ge=1, strict=True)


class AgentInputAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_revision: int = Field(ge=1, strict=True)
    answer: str = Field(min_length=1, max_length=8000)


class AgentKnowledgeDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)


class AgentKnowledgeSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fact", "decision", "question", "action"]
    text: str = Field(min_length=1, max_length=1200)


class AgentKnowledgeDraftResponse(BaseModel):
    task_id: str
    task_revision: int
    suggestions: list[AgentKnowledgeSuggestion] = Field(max_length=3)


def _repository(request: Request):
    store = getattr(request.app.state, "application_store", None)
    if store is None or not hasattr(store, "agents"):
        raise HTTPException(status_code=503, detail="Saved agents are unavailable.")
    from audrey.agent_file_inputs import get_agent_file_inputs

    get_agent_file_inputs(request.app)
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
                      project_id: str | None = Query(default=None, min_length=1, max_length=300),
                      principal: Principal = Depends(_agents_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        repository = _repository(request)
        tasks = await repository.list(user_id=principal.user_id, limit=200, summary=True, project_id=project_id)
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
        from audrey.agent_file_inputs import get_agent_file_inputs

        inputs = get_agent_file_inputs(request.app)
        if body.file_ids and inputs is None:
            raise HTTPException(status_code=503, detail="Agent file inputs are unavailable.")
        pins = await inputs.capture(principal, body.project_id, body.file_ids) if body.file_ids else None
        from audrey.agent_knowledge_files import get_agent_knowledge_files

        knowledge_files = get_agent_knowledge_files(request.app, inputs=inputs)
        knowledge_sources = await knowledge_files.prepare(
            user_id=principal.user_id, project_id=body.project_id, objective=body.objective,
            knowledge_entry_id=body.knowledge_entry_id,
            knowledge_entry_revision=body.knowledge_entry_revision,
        ) if knowledge_files is not None and body.project_id else None
        task = await repository.create(user_id=principal.user_id, objective=body.objective,
                                       request_id=body.request_id, limits=settings.task_limits(), project_id=body.project_id,
                                       knowledge_entry_id=body.knowledge_entry_id,
                                       knowledge_entry_revision=body.knowledge_entry_revision, file_inputs=pins,
                                       knowledge_file_sources=knowledge_sources)
        repository.fence_response(user_id=principal.user_id, tasks=[task])
    except AgentDispatchStoppedError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except AgentTaskError as error:
        raise _error(error) from None
    _wake(request)
    return task


async def _knowledge_drafts_while_connected(request: Request, operation: Coroutine) -> dict:
    """Cancel a bounded, unsaved helper when its requesting browser goes away."""
    stop_watching = asyncio.Event()

    async def disconnected():
        while not stop_watching.is_set():
            if await request.is_disconnected():
                return
            try:
                await asyncio.wait_for(stop_watching.wait(), timeout=0.1)
            except TimeoutError:
                pass

    work = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _pending = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            await watcher
            raise HTTPException(499, "Suggestion generation was cancelled.")
        return await work
    finally:
        # Starlette's nonblocking disconnect probe uses a cancelled AnyIO
        # scope, which can swallow an external task cancellation. End the
        # watcher cooperatively so even an immediate worker error cleans up.
        stop_watching.set()
        if not work.done():
            work.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


@router.post("/agents/{task_id}/knowledge-drafts", response_model=AgentKnowledgeDraftResponse)
async def generate_agent_knowledge_drafts(
    task_id: str,
    body: AgentKnowledgeDraftRequest,
    request: Request,
    response: Response,
    principal: Principal = Depends(_agents_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    _repository(request)
    service = getattr(request.app.state, "agent_knowledge_drafts", None)
    if service is None:
        service = ReportKnowledgeDrafts(request.app)
        request.app.state.agent_knowledge_drafts = service
    scope = {"user_id": principal.user_id, "task_id": task_id, "expected_revision": body.expected_revision}
    try:
        result = await _knowledge_drafts_while_connected(request, service.generate(**scope))
        # The generation coroutine can finish before this route resumes. Recheck
        # the canonical report and current authority after that last await.
        service.fence_response(**scope)
        return result
    except DraftError as error:
        raise HTTPException(error.status_code, str(error)) from None
    except (sqlite3.Error, OSError) as error:
        logger.warning("Knowledge suggestion storage unavailable (%s)", type(error).__name__)
        raise HTTPException(503, "Knowledge suggestions are temporarily unavailable.") from None


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


@router.get("/agents/{task_id}/events", response_model=AgentEventsResponse)
async def agent_events(
    task_id: str, request: Request, response: Response,
    before: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=50),
    principal: Principal = Depends(_agents_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        repository = _repository(request)
        page = await repository.events(user_id=principal.user_id, task_id=task_id, before=before, limit=limit)
        repository.fence_events_response(user_id=principal.user_id, task_id=task_id, page=page)
        return page
    except AgentTaskError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        logger.warning("Agent events unavailable (%s)", type(error).__name__)
        raise HTTPException(503, "Agent activity is temporarily unavailable.") from None


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
