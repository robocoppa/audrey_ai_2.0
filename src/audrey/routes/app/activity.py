"""Owned in-app activity for saved agents and public-page watches."""

from __future__ import annotations

import logging
import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from audrey.app_state.agents import AgentTaskAccessError, AgentTaskConflictError, AgentTaskError
from audrey.app_state.notifications import (
    NotificationAccessError,
    NotificationConflictError,
    NotificationError,
)
from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.routes.app.agents import AgentTaskResponse

router = APIRouter(tags=["application-activity"])
_access = require_scope("compat:full")
logger = logging.getLogger(__name__)


class AgentActivityTarget(BaseModel):
    kind: Literal["agent"]
    task_id: str
    revision: int


class WatchActivityTarget(BaseModel):
    kind: Literal["watch"]
    watch_id: str
    run_id: str


class ActivityResponse(BaseModel):
    id: str
    origin: Literal["agent", "watch"]
    kind: Literal["completed", "failure", "change"]
    title: str
    body: str
    created_at: str
    read_at: str | None
    target: Annotated[AgentActivityTarget | WatchActivityTarget, Field(discriminator="kind")]


class ActivityFeedResponse(BaseModel):
    items: list[ActivityResponse]
    unread_count: int


def _repository(request: Request):
    store = getattr(request.app.state, "application_store", None)
    if store is None or not hasattr(store, "notifications"):
        raise HTTPException(503, "Activity is unavailable.")
    return store.notifications


def _error(error: NotificationError | AgentTaskError) -> HTTPException:
    code = 404 if isinstance(error, (NotificationAccessError, AgentTaskAccessError)) else (
        409 if isinstance(error, (NotificationConflictError, AgentTaskConflictError)) else 422
    )
    return HTTPException(code, str(error))


def _unavailable(error: Exception) -> HTTPException:
    logger.warning("Activity storage unavailable (%s)", type(error).__name__)
    return HTTPException(503, "Activity is temporarily unavailable.")


@router.get("/activity", response_model=ActivityFeedResponse)
async def list_activity(request: Request, response: Response, principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        repository = _repository(request)
        feed = await repository.feed(user_id=principal.user_id, limit=50)
        repository.fence_response(user_id=principal.user_id, notices=feed["items"])
        return feed
    except NotificationError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.get("/activity/unread")
async def unread_activity(request: Request, response: Response, principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        repository = _repository(request)
        count = await repository.unread_count(user_id=principal.user_id)
        repository.fence_response(user_id=principal.user_id)
        return {"unread_count": count}
    except NotificationError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.get("/activity/agent/{notice_id}/task", response_model=AgentTaskResponse)
async def activity_agent_task(notice_id: str, request: Request, response: Response,
                              principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        repository = _repository(request)
        notice = await repository.get(user_id=principal.user_id, origin="agent", notice_id=notice_id)
        target = notice["target"]
        task = await request.app.state.application_store.agents.get(
            user_id=principal.user_id, task_id=target["task_id"],
        )
        if task["id"] != target["task_id"] or task["revision"] != target["revision"]:
            raise NotificationConflictError("This agent changed. Refresh Activity before opening it.")
        repository.fence_response(user_id=principal.user_id, notices=[notice])
        return task
    except (NotificationError, AgentTaskError) as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None


@router.post("/activity/{origin}/{notice_id}/read", response_model=ActivityResponse)
async def read_activity(origin: Literal["agent", "watch"], notice_id: str, request: Request, response: Response,
                        principal: Principal = Depends(_access)):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        repository = _repository(request)
        notice = await repository.mark_read(user_id=principal.user_id, origin=origin, notice_id=notice_id)
        repository.fence_response(user_id=principal.user_id, notices=[notice])
        return notice
    except NotificationError as error:
        raise _error(error) from None
    except (sqlite3.Error, OSError) as error:
        raise _unavailable(error) from None
