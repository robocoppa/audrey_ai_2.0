"""Owner-bound native run lifecycle and typed server-sent events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from audrey.app_state import (
    ApplicationStore,
    ConversationArchivedError,
    ConversationHasActiveRunError,
    InvalidApplicationStateError,
    MessageRecord,
    RunAlreadyTerminalError,
    RunRecord,
    StartedRun,
)
from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.pipeline.agui import (
    AgUiCursor,
    AgUiCursorError,
    AgUiRunEventAdapter,
    dump_agui_event,
    format_agui_cursor,
    parse_agui_cursor,
)
from audrey.pipeline.run_events import (
    RunEvent,
    RunEventContext,
    RunEventEmitter,
    RunFinishedEvent,
    TextDeltaEvent,
    UsageReportedEvent,
    dump_run_event,
)
from audrey.routes.openai.pipeline import _stream_via_pipeline
from audrey.routes.openai.responses import _options_from_request
from audrey.routes.openai.schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])
_run_access = require_scope("compat:full")
_Mode = Literal["auto", "fast", "deep", "research", "local", "cloud"]
_MODELS: dict[str, str] = {
    "auto": "audrey_auto",
    "fast": "audrey_fast",
    "deep": "audrey_deep",
    "research": "audrey_research",
    "local": "audrey_local",
    "cloud": "audrey_cloud",
}
_StreamFactory = Callable[..., AsyncIterator[str]]


class NativeRunUnavailableError(RuntimeError):
    """The owner has a durable run row but its transient events are gone."""


class NativeRunCursorExpiredError(RuntimeError):
    """The requested sequence predates the bounded in-memory event window."""


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=1_000_000)
    mode: _Mode | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class RunResponse(BaseModel):
    id: str
    conversation_id: str
    mode: _Mode
    status: Literal["running", "succeeded", "cancelled", "failed"]
    started_at: str
    completed_at: str | None
    finish_reason: str
    error_code: str
    virtual_model: str
    concrete_model: str
    prompt_tokens: int
    completion_tokens: int


class RunCreateResponse(RunResponse):
    user_message_id: str
    assistant_message_id: str
    events_url: str
    agui_events_url: str
    cancel_url: str


@dataclass(slots=True)
class _LiveRun:
    owner_user_id: str
    storage_namespace: str
    started: StartedRun
    emitter: RunEventEmitter
    events: deque[RunEvent]
    changed: asyncio.Event
    settled: asyncio.Event
    task: asyncio.Task[None] | None = None
    cancel_error_code: str = "cancelled_by_user"
    answer_parts: list[str] = field(default_factory=list)
    latest_usage: UsageReportedEvent | None = None

    def publish(self, event: RunEvent) -> None:
        expected = self.events[-1].sequence + 1 if self.events else 1
        if event.run_id != self.started.run.run_id or event.sequence != expected:
            raise RuntimeError("native run event identity or sequence is invalid")
        if isinstance(event, TextDeltaEvent):
            self.answer_parts.append(event.delta)
        elif isinstance(event, UsageReportedEvent):
            self.latest_usage = event
        self.events.append(event)
        previous = self.changed
        self.changed = asyncio.Event()
        previous.set()

    @property
    def terminal_event(self) -> RunFinishedEvent | None:
        if self.events and isinstance(self.events[-1], RunFinishedEvent):
            return self.events[-1]
        return None


class NativeRunManager:
    """Own live tasks and a bounded reconnect window for native run events."""

    def __init__(
        self,
        *,
        app: Any,
        store: ApplicationStore,
        stream_factory: _StreamFactory = _stream_via_pipeline,
        max_events_per_run: int = 20_000,
        max_completed_runs: int = 128,
    ) -> None:
        if max_events_per_run < 100:
            raise ValueError("native run event capacity must be at least 100")
        if max_completed_runs < 1:
            raise ValueError("native completed-run capacity must be positive")
        self._app = app
        self._store = store
        self._stream_factory = stream_factory
        self._max_events_per_run = max_events_per_run
        self._max_completed_runs = max_completed_runs
        self._runs: dict[str, _LiveRun] = {}
        self._completed: deque[str] = deque()
        self._lock = asyncio.Lock()

    async def launch(
        self,
        *,
        principal: Principal,
        started: StartedRun,
        payload: ChatCompletionRequest,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> None:
        live: _LiveRun
        live = _LiveRun(
            owner_user_id=principal.user_id,
            storage_namespace=principal.storage_namespace,
            started=started,
            emitter=RunEventEmitter(
                run_id=started.run.run_id,
                conversation_id=started.conversation.conversation_id,
                assistant_message_id=started.assistant_message.message_id,
                mode=started.run.mode,
                virtual_model=payload.model,
            ),
            events=deque(maxlen=self._max_events_per_run),
            changed=asyncio.Event(),
            settled=asyncio.Event(),
        )
        live.emitter.set_sink(live.publish)
        async with self._lock:
            if started.run.run_id in self._runs:
                raise RuntimeError("native run is already registered")
            self._runs[started.run.run_id] = live
            live.task = asyncio.create_task(
                self._execute(live, payload, messages, options),
                name=f"audrey.native_run.{started.run.run_id}",
            )

    async def _execute(
        self,
        live: _LiveRun,
        payload: ChatCompletionRequest,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> None:
        context = RunEventContext(
            run_id=live.started.run.run_id,
            conversation_id=live.started.conversation.conversation_id,
            assistant_message_id=live.started.assistant_message.message_id,
            mode=live.started.run.mode,
            sink=live.publish,
            emitter=live.emitter,
        )
        try:
            async for _frame in self._stream_factory(
                self._app,
                payload,
                messages,
                options,
                user_id=live.storage_namespace,
                conversation_id=live.started.conversation.conversation_id,
                user_turn_text=live.started.user_message.content,
                event_context=context,
            ):
                pass
        except asyncio.CancelledError:
            if not live.emitter.is_finished:
                live.emitter.terminate_incomplete(
                    status="cancelled",
                    finish_reason="cancelled",
                    error_code=live.cancel_error_code,
                )
        except Exception:
            log.exception("native run failed run_id=%s", live.started.run.run_id)
            if not live.emitter.is_finished:
                live.emitter.terminate_incomplete(
                    status="failed",
                    finish_reason="error",
                    error_code="pipeline_error",
                )
        finally:
            try:
                await self._persist_terminal(live)
            except Exception:
                log.exception(
                    "native run terminal persistence failed run_id=%s",
                    live.started.run.run_id,
                )
            finally:
                live.settled.set()
                await self._retain_completed(live)

    async def _persist_terminal(self, live: _LiveRun) -> None:
        terminal = live.terminal_event
        if terminal is None:
            live.emitter.terminate_incomplete(
                status="failed",
                finish_reason="error",
                error_code="missing_terminal_event",
            )
            terminal = live.terminal_event
        assert terminal is not None
        assistant_content = "".join(live.answer_parts)
        usage = live.latest_usage
        try:
            await self._store.conversations.finish_run(
                user_id=live.owner_user_id,
                run_id=live.started.run.run_id,
                outcome=terminal.status,
                assistant_content=assistant_content,
                finish_reason=terminal.finish_reason,
                error_code=terminal.error_code,
                virtual_model=terminal.virtual_model,
                concrete_model=terminal.concrete_model,
                prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                completion_tokens=usage.completion_tokens if usage is not None else 0,
            )
        except RunAlreadyTerminalError:
            log.warning(
                "native run terminal row was already finalized run_id=%s",
                live.started.run.run_id,
            )

    async def _retain_completed(self, live: _LiveRun) -> None:
        async with self._lock:
            self._completed.append(live.started.run.run_id)
            while len(self._completed) > self._max_completed_runs:
                expired_id = self._completed.popleft()
                expired = self._runs.get(expired_id)
                if expired is not None and expired.terminal_event is not None:
                    self._runs.pop(expired_id, None)

    async def open_events(
        self,
        *,
        user_id: str,
        run_id: str,
        after_sequence: int,
    ) -> _LiveRun:
        async with self._lock:
            live = self._runs.get(run_id)
        if live is None or live.owner_user_id != user_id:
            record = await self._store.conversations.get_run(
                user_id=user_id,
                run_id=run_id,
            )
            if record is None:
                raise KeyError(run_id)
            raise NativeRunUnavailableError(run_id)
        self._validate_cursor(live, after_sequence)
        return live

    @staticmethod
    def _validate_cursor(live: _LiveRun, after_sequence: int) -> None:
        if live.events and after_sequence < live.events[0].sequence - 1:
            raise NativeRunCursorExpiredError(live.started.run.run_id)

    async def iter_events(
        self,
        live: _LiveRun,
        *,
        after_sequence: int,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[RunEvent | None]:
        cursor = after_sequence
        while True:
            self._validate_cursor(live, cursor)
            pending = tuple(event for event in live.events if event.sequence > cursor)
            waiter = live.changed
            if pending:
                for event in pending:
                    if isinstance(event, RunFinishedEvent):
                        await live.settled.wait()
                    cursor = event.sequence
                    yield event
                continue
            if live.terminal_event is not None:
                return
            try:
                await asyncio.wait_for(waiter.wait(), timeout=heartbeat_seconds)
            except TimeoutError:
                yield None

    async def cancel(self, *, user_id: str, run_id: str) -> RunRecord | None:
        async with self._lock:
            live = self._runs.get(run_id)
        if live is None or live.owner_user_id != user_id:
            return await self._store.conversations.get_run(
                user_id=user_id,
                run_id=run_id,
            )
        task = live.task
        if not live.emitter.is_finished and task is not None and not task.done():
            live.cancel_error_code = "cancelled_by_user"
            task.cancel()
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)
        return await self._store.conversations.get_run(user_id=user_id, run_id=run_id)

    async def stop(self) -> None:
        async with self._lock:
            active = [
                live
                for live in self._runs.values()
                if live.task is not None and not live.task.done()
            ]
        for live in active:
            live.cancel_error_code = "server_shutdown"
            assert live.task is not None
            live.task.cancel()
        if active:
            await asyncio.gather(
                *(live.task for live in active if live.task is not None),
                return_exceptions=True,
            )


def _store(request: Request) -> ApplicationStore:
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Audrey application state is unavailable.")
    return store


def _manager(request: Request) -> NativeRunManager:
    manager = getattr(request.app.state, "native_runs", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Audrey run service is unavailable.")
    return manager


def _run_response(record: RunRecord) -> RunResponse:
    return RunResponse(
        id=record.run_id,
        conversation_id=record.conversation_id,
        mode=record.mode,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        finish_reason=record.finish_reason,
        error_code=record.error_code,
        virtual_model=record.virtual_model,
        concrete_model=record.concrete_model,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
    )


def _history_messages(
    started: StartedRun,
    records: tuple[MessageRecord, ...],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for record in records:
        if record.message_id == started.assistant_message.message_id:
            continue
        if record.role not in {"user", "assistant"}:
            continue
        if record.role == "assistant" and not record.content:
            continue
        messages.append({"role": record.role, "content": record.content})
    return messages


@router.post(
    "/conversations/{conversation_id}/runs",
    response_model=RunCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_run(
    conversation_id: str,
    payload: RunCreateRequest,
    request: Request,
    principal: Principal = Depends(_run_access),
) -> RunCreateResponse:
    manager = _manager(request)
    store = _store(request)
    try:
        started = await store.conversations.begin_run(
            user_id=principal.user_id,
            conversation_id=conversation_id,
            user_content=payload.content,
            mode=payload.mode,
        )
    except (ConversationArchivedError, ConversationHasActiveRunError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidApplicationStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if started is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    records = await store.conversations.list_messages(
        user_id=principal.user_id,
        conversation_id=conversation_id,
    )
    assert records is not None
    messages = _history_messages(started, records)
    virtual_model = _MODELS[started.run.mode]
    pipeline_payload = ChatCompletionRequest(
        model=virtual_model,
        messages=messages,
        stream=True,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
    await manager.launch(
        principal=principal,
        started=started,
        payload=pipeline_payload,
        messages=messages,
        options=_options_from_request(pipeline_payload),
    )
    base = _run_response(started.run).model_dump()
    return RunCreateResponse(
        **base,
        user_message_id=started.user_message.message_id,
        assistant_message_id=started.assistant_message.message_id,
        events_url=f"/api/runs/{started.run.run_id}/events",
        agui_events_url=f"/api/runs/{started.run.run_id}/ag-ui-events",
        cancel_url=f"/api/runs/{started.run.run_id}/cancel",
    )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    request: Request,
    principal: Principal = Depends(_run_access),
) -> RunResponse:
    record = await _store(request).conversations.get_run(
        user_id=principal.user_id,
        run_id=run_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return _run_response(record)


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
async def cancel_run(
    run_id: str,
    request: Request,
    principal: Principal = Depends(_run_access),
) -> RunResponse:
    record = await _manager(request).cancel(
        user_id=principal.user_id,
        run_id=run_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return _run_response(record)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    principal: Principal = Depends(_run_access),
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    if last_event_id is not None:
        try:
            resumed_after = int(last_event_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Last-Event-ID is invalid.") from exc
        if resumed_after < 0 or (after and after != resumed_after):
            raise HTTPException(status_code=422, detail="Event cursor is inconsistent.")
        after = resumed_after
    manager = _manager(request)
    try:
        live = await manager.open_events(
            user_id=principal.user_id,
            run_id=run_id,
            after_sequence=after,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except NativeRunUnavailableError as exc:
        raise HTTPException(
            status_code=410,
            detail="Run events are no longer available; read the persisted run and messages.",
        ) from exc
    except NativeRunCursorExpiredError as exc:
        raise HTTPException(status_code=409, detail="Run event cursor has expired.") from exc

    async def _events() -> AsyncIterator[str]:
        async for event in manager.iter_events(live, after_sequence=after):
            if event is None:
                yield ": keep-alive\n\n"
                continue
            data = json.dumps(dump_run_event(event), separators=(",", ":"))
            yield f"id: {event.sequence}\nevent: {event.type}\ndata: {data}\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def _resolve_agui_cursor(*, after: str, last_event_id: str | None) -> AgUiCursor:
    try:
        query_cursor = parse_agui_cursor(after)
        if last_event_id is None:
            return query_cursor
        header_cursor = parse_agui_cursor(last_event_id)
    except AgUiCursorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if after != "0" and header_cursor != query_cursor:
        raise HTTPException(status_code=422, detail="AG-UI event cursor is inconsistent.")
    return header_cursor


def _validate_agui_fanout_cursor(
    *,
    live: _LiveRun,
    cursor: AgUiCursor,
    adapter: AgUiRunEventAdapter,
) -> None:
    if cursor.part is None:
        return
    source = next(
        (event for event in live.events if event.sequence == cursor.source_sequence),
        None,
    )
    if source is None:
        raise HTTPException(status_code=422, detail="AG-UI event cursor is unavailable.")
    if cursor.part > len(adapter.adapt(source)):
        raise HTTPException(status_code=422, detail="AG-UI event cursor is invalid.")


@router.get("/runs/{run_id}/ag-ui-events")
async def stream_run_agui_events(
    run_id: str,
    request: Request,
    principal: Principal = Depends(_run_access),
    after: Annotated[str, Query(min_length=1, max_length=64)] = "0",
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Adapt one owner-bound live run to the current AG-UI SSE vocabulary."""

    cursor = _resolve_agui_cursor(after=after, last_event_id=last_event_id)
    manager = _manager(request)
    try:
        live = await manager.open_events(
            user_id=principal.user_id,
            run_id=run_id,
            after_sequence=cursor.native_after_sequence,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except NativeRunUnavailableError as exc:
        raise HTTPException(
            status_code=410,
            detail="Run events are no longer available; read the persisted run and messages.",
        ) from exc
    except NativeRunCursorExpiredError as exc:
        raise HTTPException(status_code=409, detail="Run event cursor has expired.") from exc

    adapter = AgUiRunEventAdapter(
        thread_id=live.started.conversation.conversation_id,
        run_id=live.started.run.run_id,
        assistant_message_id=live.started.assistant_message.message_id,
        latest_usage=live.latest_usage,
    )
    _validate_agui_fanout_cursor(live=live, cursor=cursor, adapter=adapter)

    async def _events() -> AsyncIterator[str]:
        async for event in manager.iter_events(
            live,
            after_sequence=cursor.native_after_sequence,
        ):
            if event is None:
                yield ": keep-alive\n\n"
                continue
            for part, agui_event in enumerate(adapter.adapt(event), start=1):
                if cursor.consumed(source_sequence=event.sequence, part=part):
                    continue
                event_cursor = format_agui_cursor(
                    source_sequence=event.sequence,
                    part=part,
                )
                data = json.dumps(
                    dump_agui_event(agui_event),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                yield f"id: {event_cursor}\ndata: {data}\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["NativeRunManager", "router"]
