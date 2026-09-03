"""Typed, client-neutral events for one Audrey run.

Pipeline implementations may use narrower private events while they work, but
anything observable by a client crosses this boundary as one of these models.
Protocol adapters serialize the models; they do not infer lifecycle state from
rendered text or parse another adapter's wire format.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class RunEventProtocolError(RuntimeError):
    """An event producer attempted an invalid lifecycle transition."""


class _RunEventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    created_at: dt.datetime


class RunStartedEvent(_RunEventBase):
    type: Literal["run.started"] = "run.started"
    conversation_id: str = Field(min_length=1)
    mode: str = Field(min_length=1)


class StageStartedEvent(_RunEventBase):
    type: Literal["stage.started"] = "stage.started"
    stage: str = Field(min_length=1)
    label: str = ""


class StageProgressEvent(_RunEventBase):
    type: Literal["stage.progress"] = "stage.progress"
    stage: str = ""
    delta: str


class StageFinishedEvent(_RunEventBase):
    type: Literal["stage.finished"] = "stage.finished"
    stage: str = Field(min_length=1)
    status: Literal["succeeded", "failed", "cancelled"]
    detail: str = ""


class AssistantMessageStartedEvent(_RunEventBase):
    type: Literal["message.started"] = "message.started"
    message_id: str = Field(min_length=1)
    role: Literal["assistant"] = "assistant"


class TextDeltaEvent(_RunEventBase):
    type: Literal["text.delta"] = "text.delta"
    message_id: str = Field(min_length=1)
    delta: str = Field(min_length=1)


class ToolCallStartedEvent(_RunEventBase):
    type: Literal["tool.started"] = "tool.started"
    tool_call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class ToolCallArgumentsEvent(_RunEventBase):
    type: Literal["tool.arguments"] = "tool.arguments"
    tool_call_id: str = Field(min_length=1)
    arguments: dict[str, Any]


class ToolCallFinishedEvent(_RunEventBase):
    type: Literal["tool.finished"] = "tool.finished"
    tool_call_id: str = Field(min_length=1)
    status: Literal["succeeded", "failed"]
    result: Any | None = None
    error: str = ""


class SourceObservedEvent(_RunEventBase):
    type: Literal["source.observed"] = "source.observed"
    source_id: str = Field(min_length=1)
    title: str = ""
    url: str = ""
    source_type: str = ""


class UsageReportedEvent(_RunEventBase):
    type: Literal["usage.reported"] = "usage.reported"
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class AssistantMessageFinishedEvent(_RunEventBase):
    type: Literal["message.finished"] = "message.finished"
    message_id: str = Field(min_length=1)
    status: Literal["completed", "incomplete"]


class RunFinishedEvent(_RunEventBase):
    type: Literal["run.finished"] = "run.finished"
    status: Literal["succeeded", "cancelled", "failed"]
    finish_reason: str = ""
    error_code: str = ""
    virtual_model: str = ""
    concrete_model: str = ""


type RunEvent = Annotated[
    RunStartedEvent
    | StageStartedEvent
    | StageProgressEvent
    | StageFinishedEvent
    | AssistantMessageStartedEvent
    | TextDeltaEvent
    | ToolCallStartedEvent
    | ToolCallArgumentsEvent
    | ToolCallFinishedEvent
    | SourceObservedEvent
    | UsageReportedEvent
    | AssistantMessageFinishedEvent
    | RunFinishedEvent,
    Field(discriminator="type"),
]
RUN_EVENT_ADAPTER = TypeAdapter(RunEvent)


@dataclass(frozen=True, slots=True)
class RunEventContext:
    """Stable identities and optional observer for one streamed run."""

    run_id: str
    conversation_id: str
    assistant_message_id: str
    mode: str
    sink: Callable[[RunEvent], None] | None = None
    emitter: RunEventEmitter | None = None


class RunEventEmitter:
    """Issue ordered events while enforcing one run/message terminal path."""

    def __init__(
        self,
        *,
        run_id: str,
        conversation_id: str,
        assistant_message_id: str,
        mode: str,
        virtual_model: str,
        sink: Callable[[RunEvent], None] | None = None,
    ) -> None:
        self.run_id = _required(run_id, "run id")
        self.conversation_id = _required(conversation_id, "conversation id")
        self.assistant_message_id = _required(
            assistant_message_id,
            "assistant message id",
        )
        self.mode = _required(mode, "mode")
        self.virtual_model = str(virtual_model)
        self._sink = sink
        self._sequence = 0
        self._run_started = False
        self._message_started = False
        self._message_finished = False
        self._run_finished = False
        self._open_stages: list[str] = []
        self._tool_states: dict[str, str] = {}

    @property
    def is_started(self) -> bool:
        return self._run_started

    @property
    def is_message_started(self) -> bool:
        return self._message_started

    @property
    def is_message_finished(self) -> bool:
        return self._message_finished

    @property
    def is_finished(self) -> bool:
        return self._run_finished

    def set_sink(self, sink: Callable[[RunEvent], None] | None) -> None:
        """Attach the observer before the first event is emitted."""

        if self._run_started:
            raise RunEventProtocolError("event sink cannot change after run start")
        self._sink = sink

    def _event[EventT: _RunEventBase](
        self,
        event_type: type[EventT],
        **fields: Any,
    ) -> EventT:
        self._sequence += 1
        event = event_type(
            event_id=f"evt_{uuid.uuid4().hex}",
            run_id=self.run_id,
            sequence=self._sequence,
            created_at=dt.datetime.now(dt.UTC),
            **fields,
        )
        if self._sink is not None:
            self._sink(event)
        return event

    def _require_active(self) -> None:
        if not self._run_started:
            raise RunEventProtocolError("run has not started")
        if self._run_finished:
            raise RunEventProtocolError("run already finished")

    def run_started(self) -> RunStartedEvent:
        if self._run_started:
            raise RunEventProtocolError("run already started")
        self._run_started = True
        return self._event(
            RunStartedEvent,
            conversation_id=self.conversation_id,
            mode=self.mode,
        )

    def stage_started(self, stage: str, *, label: str = "") -> StageStartedEvent:
        self._require_active()
        stage = _required(stage, "stage")
        if stage in self._open_stages:
            raise RunEventProtocolError(f"stage {stage!r} already started")
        self._open_stages.append(stage)
        return self._event(StageStartedEvent, stage=stage, label=str(label))

    def stage_progress(self, delta: str, *, stage: str = "") -> StageProgressEvent:
        self._require_active()
        if stage and stage not in self._open_stages:
            raise RunEventProtocolError(f"stage {stage!r} is not active")
        delta = str(delta)
        if not delta:
            raise RunEventProtocolError("stage progress delta cannot be empty")
        return self._event(StageProgressEvent, stage=stage, delta=delta)

    def stage_finished(
        self,
        stage: str,
        *,
        status: Literal["succeeded", "failed", "cancelled"],
        detail: str = "",
    ) -> StageFinishedEvent:
        self._require_active()
        stage = _required(stage, "stage")
        if stage not in self._open_stages:
            raise RunEventProtocolError(f"stage {stage!r} is not active")
        self._open_stages.remove(stage)
        return self._event(
            StageFinishedEvent,
            stage=stage,
            status=status,
            detail=str(detail),
        )

    def finish_open_stages(
        self,
        *,
        status: Literal["succeeded", "failed", "cancelled"],
        detail: str = "",
    ) -> tuple[StageFinishedEvent, ...]:
        """Close unfinished stages in reverse nesting order at termination."""

        return tuple(
            self.stage_finished(stage, status=status, detail=detail)
            for stage in reversed(tuple(self._open_stages))
        )

    def message_started(self) -> AssistantMessageStartedEvent:
        self._require_active()
        if self._message_started:
            raise RunEventProtocolError("assistant message already started")
        self._message_started = True
        return self._event(
            AssistantMessageStartedEvent,
            message_id=self.assistant_message_id,
        )

    def text_delta(self, delta: str) -> TextDeltaEvent:
        self._require_active()
        if not self._message_started or self._message_finished:
            raise RunEventProtocolError("assistant message is not active")
        if not delta:
            raise RunEventProtocolError("text delta cannot be empty")
        return self._event(
            TextDeltaEvent,
            message_id=self.assistant_message_id,
            delta=delta,
        )

    def tool_started(self, tool_call_id: str, *, name: str) -> ToolCallStartedEvent:
        self._require_active()
        tool_call_id = _required(tool_call_id, "tool call id")
        if tool_call_id in self._tool_states:
            raise RunEventProtocolError(f"tool call {tool_call_id!r} already started")
        self._tool_states[tool_call_id] = "started"
        return self._event(
            ToolCallStartedEvent,
            tool_call_id=tool_call_id,
            name=_required(name, "tool name"),
        )

    def tool_arguments(
        self,
        tool_call_id: str,
        *,
        arguments: dict[str, Any],
    ) -> ToolCallArgumentsEvent:
        self._require_active()
        tool_call_id = _required(tool_call_id, "tool call id")
        if self._tool_states.get(tool_call_id) != "started":
            raise RunEventProtocolError(f"tool call {tool_call_id!r} is not awaiting arguments")
        self._tool_states[tool_call_id] = "arguments"
        return self._event(
            ToolCallArgumentsEvent,
            tool_call_id=tool_call_id,
            arguments=dict(arguments),
        )

    def tool_finished(
        self,
        tool_call_id: str,
        *,
        status: Literal["succeeded", "failed"],
        result: Any | None = None,
        error: str = "",
    ) -> ToolCallFinishedEvent:
        self._require_active()
        tool_call_id = _required(tool_call_id, "tool call id")
        if self._tool_states.get(tool_call_id) != "arguments":
            raise RunEventProtocolError(f"tool call {tool_call_id!r} is not active")
        self._tool_states[tool_call_id] = "finished"
        return self._event(
            ToolCallFinishedEvent,
            tool_call_id=tool_call_id,
            status=status,
            result=result,
            error=str(error),
        )

    def source_observed(
        self,
        source_id: str,
        *,
        title: str = "",
        url: str = "",
        source_type: str = "",
    ) -> SourceObservedEvent:
        self._require_active()
        return self._event(
            SourceObservedEvent,
            source_id=_required(source_id, "source id"),
            title=str(title),
            url=str(url),
            source_type=str(source_type),
        )

    def usage_reported(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> UsageReportedEvent:
        self._require_active()
        return self._event(
            UsageReportedEvent,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def message_finished(
        self,
        *,
        status: Literal["completed", "incomplete"],
    ) -> AssistantMessageFinishedEvent:
        self._require_active()
        if not self._message_started:
            raise RunEventProtocolError("assistant message has not started")
        if self._message_finished:
            raise RunEventProtocolError("assistant message already finished")
        self._message_finished = True
        return self._event(
            AssistantMessageFinishedEvent,
            message_id=self.assistant_message_id,
            status=status,
        )

    def run_finished(
        self,
        *,
        status: Literal["succeeded", "cancelled", "failed"],
        finish_reason: str = "",
        error_code: str = "",
        concrete_model: str = "",
    ) -> RunFinishedEvent:
        self._require_active()
        if not self._message_finished:
            raise RunEventProtocolError("assistant message has not finished")
        if self._open_stages:
            stages = ", ".join(sorted(self._open_stages))
            raise RunEventProtocolError(f"run has active stages: {stages}")
        active_tools = sorted(
            tool_call_id
            for tool_call_id, state in self._tool_states.items()
            if state != "finished"
        )
        if active_tools:
            raise RunEventProtocolError(
                f"run has active tool calls: {', '.join(active_tools)}"
            )
        self._run_finished = True
        return self._event(
            RunFinishedEvent,
            status=status,
            finish_reason=str(finish_reason),
            error_code=str(error_code),
            virtual_model=self.virtual_model,
            concrete_model=str(concrete_model),
        )

    def terminate_incomplete(
        self,
        *,
        status: Literal["cancelled", "failed"],
        finish_reason: str = "",
        error_code: str = "",
        concrete_model: str = "",
    ) -> RunFinishedEvent:
        """Finish an interrupted producer through the same ordered emitter."""

        if self._run_finished:
            raise RunEventProtocolError("run already finished")
        if not self._run_started:
            self.run_started()
        if not self._message_started:
            self.message_started()
        stage_status: Literal["cancelled", "failed"] = status
        self.finish_open_stages(
            status=stage_status,
            detail=error_code or finish_reason,
        )
        for tool_call_id, tool_state in tuple(self._tool_states.items()):
            if tool_state == "started":
                self.tool_arguments(tool_call_id, arguments={})
            if self._tool_states[tool_call_id] != "finished":
                self.tool_finished(
                    tool_call_id,
                    status="failed",
                    error=error_code or status,
                )
        if not self._message_finished:
            self.message_finished(status="incomplete")
        return self.run_finished(
            status=status,
            finish_reason=finish_reason,
            error_code=error_code,
            concrete_model=concrete_model,
        )


def dump_run_event(event: RunEvent) -> dict[str, Any]:
    """Return the stable JSON-compatible event envelope."""

    return RUN_EVENT_ADAPTER.dump_python(event, mode="json")


def _required(value: str, label: str) -> str:
    clean = str(value).strip()
    if not clean:
        raise ValueError(f"{label} is required")
    return clean


__all__ = [
    "AssistantMessageFinishedEvent",
    "AssistantMessageStartedEvent",
    "RUN_EVENT_ADAPTER",
    "RunEvent",
    "RunEventContext",
    "RunEventEmitter",
    "RunEventProtocolError",
    "RunFinishedEvent",
    "RunStartedEvent",
    "SourceObservedEvent",
    "StageFinishedEvent",
    "StageProgressEvent",
    "StageStartedEvent",
    "TextDeltaEvent",
    "ToolCallArgumentsEvent",
    "ToolCallFinishedEvent",
    "ToolCallStartedEvent",
    "UsageReportedEvent",
    "dump_run_event",
]
