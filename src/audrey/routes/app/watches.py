"""Explicit, owner-bound public-page watches and their in-app activity."""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state.watches import (
    WatchAccessError,
    WatchConflictError,
    WatchError,
    WatchLimitError,
)
from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.watch_executor import watch_authority, watch_settings
from audrey.watch_schedule import preview_schedule

router = APIRouter(tags=["application-watches"])
_access = require_scope("compat:full")
logger = logging.getLogger(__name__)


class WatchRunInputs(BaseModel):
    title: str
    url: str
    criterion: str
    schedule: dict[str, Any]


class WatchRunSummary(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    kind: Literal["scheduled", "catchup", "manual"]
    scheduled_at: str
    created_at: str
    completed_at: str | None
    outcome: Literal["baseline", "unchanged", "changed"] | None
    significant: bool | None
    error: str
    report: str
    diff: str
    definition: WatchRunInputs


class WatchRunResponse(WatchRunSummary):
    watch_id: str
    watch_revision: int
    updated_at: str
    attempts: int


class WatchResponse(BaseModel):
    id: str
    title: str
    url: str
    criterion: str
    schedule: dict[str, Any]
    enabled: bool
    revision: int
    created_at: str
    updated_at: str
    next_run_at: str | None
    active_run_id: str | None
    latest_run: WatchRunSummary | None


class NoticeResponse(BaseModel):
    id: str
    watch_id: str
    run_id: str
    kind: Literal["change", "failure"]
    title: str
    body: str
    created_at: str
    read_at: str | None


class WatchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=1, max_length=4096)
    criterion: str = Field(min_length=1, max_length=2000)
    schedule: dict[str, Any]
    enabled: bool = Field(default=True, strict=True)
    request_id: str = Field(min_length=1, max_length=200)


class WatchUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_revision: int = Field(ge=1, strict=True)
    title: str | None = Field(default=None, min_length=1, max_length=100)
    url: str | None = Field(default=None, min_length=1, max_length=4096)
    criterion: str | None = Field(default=None, min_length=1, max_length=2000)
    schedule: dict[str, Any] | None = None
    enabled: bool | None = Field(default=None, strict=True)


class WatchAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "run"]
    expected_revision: int = Field(ge=1, strict=True)


class SchedulePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schedule: dict[str, Any]


def _repository(request: Request):
    store = getattr(request.app.state, "application_store", None)
    if store is None or not hasattr(store, "watches"):
        raise HTTPException(503, "Saved watches are unavailable.")
    return store.watches


def _error(error: WatchError) -> HTTPException:
    code = 404 if isinstance(error, WatchAccessError) else (
        409 if isinstance(error, WatchConflictError) else (
            429 if isinstance(error, WatchLimitError) else 422
        )
    )
    return HTTPException(code, str(error))


def _unavailable(error: Exception) -> HTTPException:
    logger.warning("Watch storage unavailable (%s)", type(error).__name__)
    return HTTPException(503, "Saved watches are temporarily unavailable.")


def _wake(request: Request) -> None:
    runtime = getattr(request.app.state, "watch_runtime", None)
    if runtime is not None:
        runtime.wake()


def _fence(request: Request, principal: Principal, **payloads) -> None:
    # Reads run in a thread. Recheck canonical identities after the last await,
    # so a deletion/purge while that thread returns cannot expose removed bodies.
    _repository(request).fence_response(user_id=principal.user_id, **payloads)


@router.post("/watches/preview")
async def preview_watch(body: SchedulePreview, response: Response, principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return {"occurrences": preview_schedule(body.schedule, after=dt.datetime.now(dt.UTC), count=3)}
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


@router.get("/watches", response_model=dict[str, list[WatchResponse]])
async def list_watches(request: Request, response: Response, principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        items = await _repository(request).list(user_id=principal.user_id, limit=20)
        _fence(request, principal, watches=items)
        return {"items": items}
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.post("/watches", response_model=WatchResponse, status_code=201)
async def create_watch(body: WatchCreate, request: Request, response: Response,
                       principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        await watch_authority(request.app, principal.user_id)
        settings = watch_settings(request.app)
        watch = await _repository(request).create(
            user_id=principal.user_id, **body.model_dump(), limits=settings.run_limits(),
            max_per_user=settings.max_watches_per_user, max_global=settings.max_watches_global,
        )
        _fence(request, principal, watches=[watch])
    except AgentDispatchStoppedError as error:
        raise HTTPException(503, str(error)) from None
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None
    _wake(request)
    return watch


@router.get("/watches/{watch_id}", response_model=WatchResponse)
async def get_watch(watch_id: str, request: Request, response: Response,
                    principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        watch = await _repository(request).get(user_id=principal.user_id, watch_id=watch_id)
        _fence(request, principal, watches=[watch])
        return watch
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.patch("/watches/{watch_id}", response_model=WatchResponse)
async def update_watch(watch_id: str, body: WatchUpdate, request: Request, response: Response,
                       principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    repository = _repository(request)
    values = body.model_dump(exclude_unset=True)
    if any(value is None for value in values.values()) or len(values) <= 1:
        raise HTTPException(422, "Choose at least one watch field to change; fields cannot be null.")
    try:
        await repository.get(user_id=principal.user_id, watch_id=watch_id)
        # Pausing and deletion remain available after grants/configuration change.
        if values.keys() - {"expected_revision", "enabled"} or values.get("enabled") is True:
            await watch_authority(request.app, principal.user_id)
        watch = await repository.update(user_id=principal.user_id, watch_id=watch_id, **values)
        _fence(request, principal, watches=[watch])
    except AgentDispatchStoppedError as error:
        raise HTTPException(503, str(error)) from None
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None
    _wake(request)
    return watch


@router.post("/watches/{watch_id}/actions", response_model=WatchResponse)
async def watch_action(watch_id: str, body: WatchAction, request: Request, response: Response,
                       principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    repository = _repository(request)
    try:
        await repository.get(user_id=principal.user_id, watch_id=watch_id)
        if body.action != "pause":
            await watch_authority(request.app, principal.user_id)
        if body.action == "run":
            watch = await repository.run_now(user_id=principal.user_id, watch_id=watch_id,
                                             expected_revision=body.expected_revision)
        else:
            watch = await repository.update(user_id=principal.user_id, watch_id=watch_id,
                                            expected_revision=body.expected_revision,
                                            enabled=body.action == "resume")
        _fence(request, principal, watches=[watch])
    except AgentDispatchStoppedError as error:
        raise HTTPException(503, str(error)) from None
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None
    _wake(request)
    return watch


@router.delete("/watches/{watch_id}")
async def delete_watch(watch_id: str, request: Request, response: Response,
                       expected_revision: int = Query(ge=1), principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        deleted = await _repository(request).delete(user_id=principal.user_id, watch_id=watch_id,
                                                   expected_revision=expected_revision)
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None
    _wake(request)
    return {"deleted": deleted}


@router.get("/watches/{watch_id}/runs", response_model=dict[str, list[WatchRunResponse]])
async def list_watch_runs(watch_id: str, request: Request, response: Response,
                          principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        items = await _repository(request).list_runs(user_id=principal.user_id, watch_id=watch_id, limit=50)
        _fence(request, principal, runs=items)
        return {"items": items}
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.get("/watches/{watch_id}/runs/{run_id}", response_model=WatchRunResponse)
async def get_watch_run(watch_id: str, run_id: str, request: Request, response: Response,
                       principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        run = await _repository(request).get_run(user_id=principal.user_id, watch_id=watch_id, run_id=run_id)
        _fence(request, principal, runs=[run])
        return run
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.get("/notifications")
async def list_notifications(request: Request, response: Response, principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        feed = await _repository(request).notification_feed(user_id=principal.user_id, limit=50)
        _fence(request, principal, notices=feed["items"])
        return {"items": [NoticeResponse.model_validate(item).model_dump() for item in feed["items"]],
                "unread_count": feed["unread_count"]}
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.get("/notifications/unread")
async def unread_notifications(request: Request, response: Response, principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return {"unread_count": await _repository(request).unread_count(user_id=principal.user_id)}
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.post("/notifications/{notice_id}/read", response_model=NoticeResponse)
async def read_notification(notice_id: str, request: Request, response: Response,
                            principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        notice = await _repository(request).mark_notice_read(user_id=principal.user_id, notice_id=notice_id)
        _fence(request, principal, notices=[notice])
        return notice
    except WatchError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None
