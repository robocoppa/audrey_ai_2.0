"""AG-UI wire adapter for Audrey's client-neutral run events.

Audrey's :mod:`audrey.pipeline.run_events` models are the application contract.
This module is deliberately a boundary adapter: AG-UI naming, camel-case fields,
and transport cursors do not leak back into pipeline producers or persistence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from audrey.pipeline.run_events import (
    AssistantMessageFinishedEvent,
    AssistantMessageStartedEvent,
    RunEvent,
    RunFinishedEvent,
    RunStartedEvent,
    SourceObservedEvent,
    StageFinishedEvent,
    StageProgressEvent,
    StageStartedEvent,
    TextDeltaEvent,
    ToolCallArgumentsEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    UsageReportedEvent,
)

_CURSOR_RE = re.compile(r"^(0|[1-9][0-9]*)(?:\.([1-9][0-9]*))?$")
_MAX_CURSOR_COMPONENT = 2**63 - 1


def _to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


class AgUiCursorError(ValueError):
    """An AG-UI SSE cursor is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AgUiCursor:
    """Resume position in the fan-out AG-UI stream.

    A bare sequence means every AG-UI frame derived from that Audrey event was
    consumed. ``sequence.part`` identifies a particular frame when one Audrey
    event expands to multiple AG-UI events (currently a finished tool call).
    """

    source_sequence: int = 0
    part: int | None = None

    @property
    def native_after_sequence(self) -> int:
        if self.part is None:
            return self.source_sequence
        return max(0, self.source_sequence - 1)

    def consumed(self, *, source_sequence: int, part: int) -> bool:
        if source_sequence < self.source_sequence:
            return True
        if source_sequence > self.source_sequence:
            return False
        if self.part is None:
            return True
        return part <= self.part


def parse_agui_cursor(value: str) -> AgUiCursor:
    """Parse Audrey's SSE cursor extension without accepting ambiguous forms."""

    if len(value) > 64:
        raise AgUiCursorError("AG-UI event cursor is invalid.")
    match = _CURSOR_RE.fullmatch(value)
    if match is None:
        raise AgUiCursorError("AG-UI event cursor is invalid.")
    source_sequence = int(match.group(1))
    raw_part = match.group(2)
    part = int(raw_part) if raw_part is not None else None
    if source_sequence > _MAX_CURSOR_COMPONENT or (
        part is not None and part > _MAX_CURSOR_COMPONENT
    ):
        raise AgUiCursorError("AG-UI event cursor is invalid.")
    if source_sequence == 0 and part is not None:
        raise AgUiCursorError("AG-UI event cursor is invalid.")
    return AgUiCursor(source_sequence=source_sequence, part=part)


def format_agui_cursor(*, source_sequence: int, part: int) -> str:
    if source_sequence < 1 or part < 1:
        raise AgUiCursorError("AG-UI event cursor components must be positive.")
    return f"{source_sequence}.{part}"


class _AgUiBaseEvent(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    type: str
    timestamp: int = Field(ge=0)


class AgUiTokenUsage(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    model: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class AgUiSuccessOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["success"] = "success"


class AgUiRunStartedEvent(_AgUiBaseEvent):
    type: Literal["RUN_STARTED"] = "RUN_STARTED"
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


class AgUiRunFinishedEvent(_AgUiBaseEvent):
    type: Literal["RUN_FINISHED"] = "RUN_FINISHED"
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    result: dict[str, Any] | None = None
    outcome: AgUiSuccessOutcome = Field(default_factory=AgUiSuccessOutcome)
    usage: list[AgUiTokenUsage] | None = None


class AgUiRunErrorEvent(_AgUiBaseEvent):
    type: Literal["RUN_ERROR"] = "RUN_ERROR"
    message: str = Field(min_length=1)
    code: str | None = None
    usage: list[AgUiTokenUsage] | None = None


class AgUiStepStartedEvent(_AgUiBaseEvent):
    type: Literal["STEP_STARTED"] = "STEP_STARTED"
    step_name: str = Field(min_length=1)


class AgUiStepFinishedEvent(_AgUiBaseEvent):
    type: Literal["STEP_FINISHED"] = "STEP_FINISHED"
    step_name: str = Field(min_length=1)


class AgUiTextMessageStartEvent(_AgUiBaseEvent):
    type: Literal["TEXT_MESSAGE_START"] = "TEXT_MESSAGE_START"
    message_id: str = Field(min_length=1)
    role: Literal["assistant"] = "assistant"


class AgUiTextMessageContentEvent(_AgUiBaseEvent):
    type: Literal["TEXT_MESSAGE_CONTENT"] = "TEXT_MESSAGE_CONTENT"
    message_id: str = Field(min_length=1)
    delta: str = Field(min_length=1)


class AgUiTextMessageEndEvent(_AgUiBaseEvent):
    type: Literal["TEXT_MESSAGE_END"] = "TEXT_MESSAGE_END"
    message_id: str = Field(min_length=1)


class AgUiToolCallStartEvent(_AgUiBaseEvent):
    type: Literal["TOOL_CALL_START"] = "TOOL_CALL_START"
    tool_call_id: str = Field(min_length=1)
    tool_call_name: str = Field(min_length=1)
    parent_message_id: str = Field(min_length=1)


class AgUiToolCallArgsEvent(_AgUiBaseEvent):
    type: Literal["TOOL_CALL_ARGS"] = "TOOL_CALL_ARGS"
    tool_call_id: str = Field(min_length=1)
    delta: str


class AgUiToolCallEndEvent(_AgUiBaseEvent):
    type: Literal["TOOL_CALL_END"] = "TOOL_CALL_END"
    tool_call_id: str = Field(min_length=1)


class AgUiToolCallResultEvent(_AgUiBaseEvent):
    type: Literal["TOOL_CALL_RESULT"] = "TOOL_CALL_RESULT"
    message_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    content: str
    role: Literal["tool"] = "tool"


class AgUiCustomEvent(_AgUiBaseEvent):
    type: Literal["CUSTOM"] = "CUSTOM"
    name: str = Field(min_length=1)
    value: Any


type AgUiEvent = Annotated[
    AgUiRunStartedEvent
    | AgUiRunFinishedEvent
    | AgUiRunErrorEvent
    | AgUiStepStartedEvent
    | AgUiStepFinishedEvent
    | AgUiTextMessageStartEvent
    | AgUiTextMessageContentEvent
    | AgUiTextMessageEndEvent
    | AgUiToolCallStartEvent
    | AgUiToolCallArgsEvent
    | AgUiToolCallEndEvent
    | AgUiToolCallResultEvent
    | AgUiCustomEvent,
    Field(discriminator="type"),
]
AGUI_EVENT_ADAPTER = TypeAdapter(AgUiEvent)


class AgUiRunEventAdapter:
    """Translate one Audrey run into current AG-UI event models."""

    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str,
        assistant_message_id: str,
        latest_usage: UsageReportedEvent | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self.assistant_message_id = assistant_message_id
        self._latest_usage = latest_usage

    def adapt(self, event: RunEvent) -> tuple[AgUiEvent, ...]:
        if event.run_id != self.run_id:
            raise ValueError("AG-UI adapter received an event for another run")
        timestamp = int(event.created_at.timestamp() * 1000)

        if isinstance(event, RunStartedEvent):
            if event.conversation_id != self.thread_id:
                raise ValueError("AG-UI adapter received an event for another thread")
            return (
                AgUiRunStartedEvent(
                    timestamp=timestamp,
                    thread_id=self.thread_id,
                    run_id=self.run_id,
                ),
            )
        if isinstance(event, AssistantMessageStartedEvent):
            return (
                AgUiTextMessageStartEvent(
                    timestamp=timestamp,
                    message_id=event.message_id,
                ),
            )
        if isinstance(event, StageStartedEvent):
            return (
                AgUiStepStartedEvent(timestamp=timestamp, step_name=event.stage),
            )
        if isinstance(event, StageProgressEvent):
            return (
                AgUiCustomEvent(
                    timestamp=timestamp,
                    name="audrey.stage.progress",
                    value={"stage": event.stage, "delta": event.delta},
                ),
            )
        if isinstance(event, StageFinishedEvent):
            return (
                AgUiStepFinishedEvent(timestamp=timestamp, step_name=event.stage),
            )
        if isinstance(event, TextDeltaEvent):
            return (
                AgUiTextMessageContentEvent(
                    timestamp=timestamp,
                    message_id=event.message_id,
                    delta=event.delta,
                ),
            )
        if isinstance(event, ToolCallStartedEvent):
            return (
                AgUiToolCallStartEvent(
                    timestamp=timestamp,
                    tool_call_id=event.tool_call_id,
                    tool_call_name=event.name,
                    parent_message_id=self.assistant_message_id,
                ),
            )
        if isinstance(event, ToolCallArgumentsEvent):
            return (
                AgUiToolCallArgsEvent(
                    timestamp=timestamp,
                    tool_call_id=event.tool_call_id,
                    delta=json.dumps(
                        event.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            )
        if isinstance(event, ToolCallFinishedEvent):
            result_message_id = f"{self.assistant_message_id}:{event.tool_call_id}:result"
            content = _tool_result_content(event)
            return (
                AgUiToolCallEndEvent(
                    timestamp=timestamp,
                    tool_call_id=event.tool_call_id,
                ),
                AgUiToolCallResultEvent(
                    timestamp=timestamp,
                    message_id=result_message_id,
                    tool_call_id=event.tool_call_id,
                    content=content,
                ),
            )
        if isinstance(event, SourceObservedEvent):
            return (
                AgUiCustomEvent(
                    timestamp=timestamp,
                    name="audrey.source.observed",
                    value={
                        "sourceId": event.source_id,
                        "title": event.title,
                        "url": event.url,
                        "sourceType": event.source_type,
                    },
                ),
            )
        if isinstance(event, UsageReportedEvent):
            self._latest_usage = event
            usage = _usage_value(event)
            return (
                AgUiCustomEvent(
                    timestamp=timestamp,
                    name="audrey.usage.reported",
                    value=usage.model_dump(by_alias=True, exclude_none=True),
                ),
            )
        if isinstance(event, AssistantMessageFinishedEvent):
            return (
                AgUiTextMessageEndEvent(
                    timestamp=timestamp,
                    message_id=event.message_id,
                ),
            )
        if isinstance(event, RunFinishedEvent):
            usage = self._terminal_usage(event)
            if event.status == "succeeded":
                return (
                    AgUiRunFinishedEvent(
                        timestamp=timestamp,
                        thread_id=self.thread_id,
                        run_id=self.run_id,
                        result={"finishReason": event.finish_reason},
                        usage=usage,
                    ),
                )
            message = "Run was cancelled." if event.status == "cancelled" else "Run failed."
            return (
                AgUiRunErrorEvent(
                    timestamp=timestamp,
                    message=message,
                    code=event.error_code or event.status,
                    usage=usage,
                ),
            )
        raise TypeError(f"unsupported Audrey run event: {type(event).__name__}")

    def _terminal_usage(self, event: RunFinishedEvent) -> list[AgUiTokenUsage] | None:
        if self._latest_usage is None:
            return None
        return [_usage_value(self._latest_usage, model=event.concrete_model)]


def _usage_value(
    event: UsageReportedEvent,
    *,
    model: str = "",
) -> AgUiTokenUsage:
    return AgUiTokenUsage(
        model=model or None,
        input_tokens=event.prompt_tokens,
        output_tokens=event.completion_tokens,
        total_tokens=event.total_tokens,
    )


def _tool_result_content(event: ToolCallFinishedEvent) -> str:
    if event.status == "failed":
        return '{"error":"tool_failed","status":"failed"}'
    if isinstance(event.result, str):
        return event.result
    return json.dumps(
        event.result,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def dump_agui_event(event: AgUiEvent) -> dict[str, Any]:
    """Return the canonical camel-case AG-UI JSON object."""

    return AGUI_EVENT_ADAPTER.dump_python(
        event,
        mode="json",
        by_alias=True,
        exclude_none=True,
    )


__all__ = [
    "AGUI_EVENT_ADAPTER",
    "AgUiCursor",
    "AgUiCursorError",
    "AgUiEvent",
    "AgUiRunEventAdapter",
    "dump_agui_event",
    "format_agui_cursor",
    "parse_agui_cursor",
]
