"""OpenAI SSE framing for Audrey's client-neutral stream lifecycle."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from audrey import __version__
from audrey.pipeline.run_events import (
    AssistantMessageFinishedEvent,
    AssistantMessageStartedEvent,
    RunEvent,
    RunEventEmitter,
    RunFinishedEvent,
    RunStartedEvent,
    StageProgressEvent,
    TextDeltaEvent,
    UsageReportedEvent,
)
from audrey.pipeline.streaming import StreamOutcome, StreamTerminal
from audrey.routes.openai.responses import _responses_api_response_object
from audrey.routes.openai.schemas import ResponseCreateRequest
from audrey.routes.openai.structured_outputs import (
    StructuredOutputError,
    response_json_schema,
    validate_structured_output,
)


@dataclass(slots=True)
class OpenAIStreamAdapter:
    """Render client-neutral run events as OpenAI chat-completion chunks."""

    virtual_model: str
    fingerprint_model: str
    completion_id: str
    created: int
    _role_emitted: bool = False
    _terminal_emitted: bool = False

    @property
    def fingerprint(self) -> str:
        return f"audrey-{__version__}/{self.fingerprint_model}"

    def _frame(self, delta: dict[str, str], finish_reason: str | None) -> str:
        frame = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.virtual_model,
            "system_fingerprint": self.fingerprint,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(frame)}\n\n"

    def render(self, event: RunEvent) -> str | None:
        if isinstance(event, AssistantMessageStartedEvent):
            if self._role_emitted:
                raise RuntimeError("assistant role frame already emitted")
            if self._terminal_emitted:
                raise RuntimeError("cannot emit assistant role after stream terminal")
            self._role_emitted = True
            return self._frame({"role": "assistant"}, None)
        if isinstance(event, (TextDeltaEvent, StageProgressEvent)):
            if not self._role_emitted:
                raise RuntimeError("assistant role frame must precede content")
            if self._terminal_emitted:
                raise RuntimeError("cannot emit content after stream terminal")
            return self._frame({"content": event.delta}, None)
        if isinstance(event, RunFinishedEvent):
            if not self._role_emitted:
                raise RuntimeError("assistant role frame must precede stream terminal")
            if self._terminal_emitted:
                raise RuntimeError("stream terminal frame already emitted")
            self._terminal_emitted = True
            return self._frame({}, event.finish_reason or None)
        return None


@dataclass(slots=True)
class OpenAIStreamSession:
    """One OpenAI SSE identity and lifecycle for a streamed response.

    Banner text and model text both pass through this owner. It refuses a
    second role frame, terminal frame, or ``[DONE]`` marker, which makes the
    one-response/one-identity rule executable instead of relying on every
    nested generator to remember it.
    """

    virtual_model: str
    fingerprint_model: str
    completion_id: str = field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}"
    )
    created: int = field(default_factory=lambda: int(time.time()))
    terminal: StreamTerminal = field(default_factory=StreamTerminal)
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    conversation_id: str = field(default_factory=lambda: f"con_{uuid.uuid4().hex}")
    assistant_message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    mode: str = ""
    event_sink: Callable[[RunEvent], None] | None = field(default=None, repr=False)
    event_emitter: RunEventEmitter | None = field(default=None, repr=False)
    concrete_model: str = ""
    _events: RunEventEmitter = field(init=False, repr=False)
    _adapter: OpenAIStreamAdapter = field(init=False, repr=False)
    _done_emitted: bool = False

    def __post_init__(self) -> None:
        mode = self.mode or self.virtual_model.removeprefix("audrey_")
        if mode == "video":
            mode = "auto"
        self._events = self.event_emitter or RunEventEmitter(
            run_id=self.run_id,
            conversation_id=self.conversation_id,
            assistant_message_id=self.assistant_message_id,
            mode=mode,
            virtual_model=self.virtual_model,
            sink=self.event_sink,
        )
        self._adapter = OpenAIStreamAdapter(
            virtual_model=self.virtual_model,
            fingerprint_model=self.fingerprint_model,
            completion_id=self.completion_id,
            created=self.created,
        )

    @property
    def run_event_emitter(self) -> RunEventEmitter:
        """The shared emitter used by native-only observation adapters."""

        return self._events

    def role_frame(self) -> str:
        if self._adapter._role_emitted:
            raise RuntimeError("assistant role frame already emitted")
        if self._adapter._terminal_emitted:
            raise RuntimeError("cannot emit assistant role after stream terminal")
        self._events.run_started()
        frame = self._adapter.render(self._events.message_started())
        assert frame is not None
        return frame

    def content_frame(self, text: str) -> str:
        if not self._adapter._role_emitted:
            raise RuntimeError("assistant role frame must precede content")
        if self._adapter._terminal_emitted:
            raise RuntimeError("cannot emit content after stream terminal")
        frame = self._adapter.render(self._events.text_delta(text))
        assert frame is not None
        return frame

    def status_frame(self, text: str, *, stage: str = "") -> str:
        frame = self._adapter.render(self._events.stage_progress(text, stage=stage))
        assert frame is not None
        return frame

    def stage_started(self, stage: str, *, label: str = "") -> None:
        self._events.stage_started(stage, label=label)

    def stage_finished(
        self,
        stage: str,
        *,
        status: str = "succeeded",
        detail: str = "",
    ) -> None:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError(f"unsupported stage status {status!r}")
        self._events.stage_finished(stage, status=status, detail=detail)  # type: ignore[arg-type]

    def set_concrete_model(self, model: str) -> None:
        self.concrete_model = str(model)

    def usage_reported(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self._events.usage_reported(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def terminal_frame(self) -> str:
        if not self._adapter._role_emitted:
            raise RuntimeError("assistant role frame must precede stream terminal")
        if self._adapter._terminal_emitted:
            raise RuntimeError("stream terminal frame already emitted")
        outcome = self.terminal.outcome
        finish_reason = self.terminal.finish_reason
        run_status = {
            StreamOutcome.OK: "succeeded",
            StreamOutcome.CANCELLED: "cancelled",
            StreamOutcome.ERROR: "failed",
            StreamOutcome.TRUNCATED: "failed",
        }[outcome]
        stage_status = {
            StreamOutcome.OK: "succeeded",
            StreamOutcome.CANCELLED: "cancelled",
            StreamOutcome.ERROR: "failed",
            StreamOutcome.TRUNCATED: "failed",
        }[outcome]
        self._events.finish_open_stages(
            status=stage_status,  # type: ignore[arg-type]
            detail="stream ended before the stage reported completion",
        )
        self._events.message_finished(
            status="completed" if outcome is StreamOutcome.OK else "incomplete"
        )
        error_code = {
            StreamOutcome.OK: "",
            StreamOutcome.CANCELLED: "cancelled",
            StreamOutcome.ERROR: "pipeline_error",
            StreamOutcome.TRUNCATED: "stream_truncated",
        }[outcome]
        event = self._events.run_finished(
            status=run_status,  # type: ignore[arg-type]
            finish_reason=finish_reason or "",
            error_code=error_code,
            concrete_model=self.concrete_model or self.fingerprint_model,
        )
        frame = self._adapter.render(event)
        assert frame is not None
        return frame

    def done_frame(self) -> str:
        if not self._adapter._terminal_emitted:
            raise RuntimeError("stream terminal frame must precede [DONE]")
        if self._done_emitted:
            raise RuntimeError("stream [DONE] marker already emitted")
        self._done_emitted = True
        return "data: [DONE]\n\n"


@dataclass(slots=True)
class ResponsesStreamAdapter:
    """Render client-neutral run events as typed Responses API SSE events."""

    request: ResponseCreateRequest
    response_id: str
    message_id: str
    created: int
    _sequence: int = -1
    _text_parts: list[str] = field(default_factory=list)
    _usage: tuple[int, int] | None = None
    _response_started: bool = False
    _message_started: bool = False
    _message_finished: bool = False
    _terminal_emitted: bool = False

    @property
    def text(self) -> str:
        return "".join(self._text_parts)

    def _frame(self, event_type: str, **fields: Any) -> str:
        self._sequence += 1
        payload = {
            "type": event_type,
            **fields,
            "sequence_number": self._sequence,
        }
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    def _response(
        self,
        *,
        status: str,
        completed_at: int | None = None,
        output_started: bool,
        error: dict[str, str] | None = None,
        incomplete_details: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        input_tokens: int | None = None
        output_tokens: int | None = None
        if self._usage is not None:
            input_tokens, output_tokens = self._usage
        return _responses_api_response_object(
            request=self.request,
            response_id=self.response_id,
            message_id=self.message_id,
            created_at=self.created,
            completed_at=completed_at,
            status=status,
            content=self.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_started=output_started,
            error=error,
            incomplete_details=incomplete_details,
        )

    def _item(self, *, status: str, include_content: bool) -> dict[str, Any]:
        return {
            "id": self.message_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [self._part()] if include_content else [],
        }

    def _part(self) -> dict[str, Any]:
        return {
            "type": "output_text",
            "text": self.text,
            "annotations": [],
            "logprobs": [],
        }

    def render(self, event: RunEvent) -> str | None:
        if isinstance(event, RunStartedEvent):
            if self._response_started:
                raise RuntimeError("response stream already started")
            self._response_started = True
            response = self._response(status="in_progress", output_started=False)
            return self._frame("response.created", response=response) + self._frame(
                "response.in_progress",
                response=response,
            )
        if isinstance(event, AssistantMessageStartedEvent):
            if not self._response_started:
                raise RuntimeError("response must start before its output item")
            if self._message_started:
                raise RuntimeError("response output item already started")
            self._message_started = True
            return self._frame(
                "response.output_item.added",
                output_index=0,
                item=self._item(status="in_progress", include_content=False),
            ) + self._frame(
                "response.content_part.added",
                item_id=self.message_id,
                output_index=0,
                content_index=0,
                part=self._part(),
            )
        if isinstance(event, TextDeltaEvent):
            if not self._message_started or self._message_finished:
                raise RuntimeError("response output text is not active")
            if self._terminal_emitted:
                raise RuntimeError("cannot emit response text after terminal")
            self._text_parts.append(event.delta)
            return self._frame(
                "response.output_text.delta",
                item_id=self.message_id,
                output_index=0,
                content_index=0,
                delta=event.delta,
                logprobs=[],
            )
        if isinstance(event, UsageReportedEvent):
            self._usage = (event.prompt_tokens, event.completion_tokens)
            return None
        if isinstance(event, AssistantMessageFinishedEvent):
            if not self._message_started:
                raise RuntimeError("response output item has not started")
            if self._message_finished:
                raise RuntimeError("response output item already finished")
            self._message_finished = True
            item_status = "completed" if event.status == "completed" else "incomplete"
            return self._frame(
                "response.output_text.done",
                item_id=self.message_id,
                output_index=0,
                content_index=0,
                text=self.text,
                logprobs=[],
            ) + self._frame(
                "response.content_part.done",
                item_id=self.message_id,
                output_index=0,
                content_index=0,
                part=self._part(),
            ) + self._frame(
                "response.output_item.done",
                output_index=0,
                item=self._item(status=item_status, include_content=True),
            )
        if isinstance(event, RunFinishedEvent):
            if not self._message_finished:
                raise RuntimeError("response output item must finish before terminal")
            if self._terminal_emitted:
                raise RuntimeError("response terminal event already emitted")
            self._terminal_emitted = True
            completed_at = int(time.time())
            if event.status == "succeeded":
                event_type = "response.completed"
                status = "completed"
                error = None
                incomplete_details = None
            elif event.error_code == "stream_truncated":
                event_type = "response.incomplete"
                status = "incomplete"
                error = None
                incomplete_details = {"reason": "max_output_tokens"}
            else:
                event_type = "response.failed"
                status = "cancelled" if event.status == "cancelled" else "failed"
                code = event.error_code or "pipeline_error"
                error = {"code": code, "message": "Audrey could not complete the response."}
                incomplete_details = None
            response = self._response(
                status=status,
                completed_at=completed_at if status == "completed" else None,
                output_started=True,
                error=error,
                incomplete_details=incomplete_details,
            )
            return self._frame(event_type, response=response)
        return None


@dataclass(slots=True)
class ResponsesStreamSession:
    """Responses API stream identity backed by Audrey's shared run events."""

    request: ResponseCreateRequest
    virtual_model: str
    fingerprint_model: str
    response_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    created: int = field(default_factory=lambda: int(time.time()))
    terminal: StreamTerminal = field(default_factory=StreamTerminal)
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    conversation_id: str = field(default_factory=lambda: f"con_{uuid.uuid4().hex}")
    assistant_message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    mode: str = ""
    event_sink: Callable[[RunEvent], None] | None = field(default=None, repr=False)
    event_emitter: RunEventEmitter | None = field(default=None, repr=False)
    concrete_model: str = ""
    _events: RunEventEmitter = field(init=False, repr=False)
    _adapter: ResponsesStreamAdapter = field(init=False, repr=False)
    _structured: bool = field(init=False, repr=False)
    _done_emitted: bool = False

    def __post_init__(self) -> None:
        mode = self.mode or self.virtual_model.removeprefix("audrey_")
        if mode == "video":
            mode = "auto"
        self._events = self.event_emitter or RunEventEmitter(
            run_id=self.run_id,
            conversation_id=self.conversation_id,
            assistant_message_id=self.assistant_message_id,
            mode=mode,
            virtual_model=self.virtual_model,
            sink=self.event_sink,
        )
        self._adapter = ResponsesStreamAdapter(
            request=self.request,
            response_id=self.response_id,
            message_id=self.assistant_message_id,
            created=self.created,
        )
        self._structured = response_json_schema(self.request) is not None

    @property
    def run_event_emitter(self) -> RunEventEmitter:
        return self._events

    def role_frame(self) -> str:
        started = self._adapter.render(self._events.run_started())
        message = self._adapter.render(self._events.message_started())
        assert started is not None and message is not None
        return started + message

    def content_frame(self, text: str) -> str:
        frame = self._adapter.render(self._events.text_delta(text))
        assert frame is not None
        return frame

    def status_frame(self, text: str, *, stage: str = "") -> str:
        # Keep progress in the internal run trace. Responses output_text
        # contains only answer text, for plain and structured output alike.
        event = self._events.stage_progress(text, stage=stage)
        return self._adapter.render(event) or ""

    def stage_started(self, stage: str, *, label: str = "") -> None:
        self._events.stage_started(stage, label=label)

    def stage_finished(
        self,
        stage: str,
        *,
        status: str = "succeeded",
        detail: str = "",
    ) -> None:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError(f"unsupported stage status {status!r}")
        self._events.stage_finished(stage, status=status, detail=detail)  # type: ignore[arg-type]

    def set_concrete_model(self, model: str) -> None:
        self.concrete_model = str(model)

    def usage_reported(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        event = self._events.usage_reported(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._adapter.render(event)

    def terminal_frame(self) -> str:
        outcome = self.terminal.outcome
        # A provider's length finish can never be a completed response, even
        # if another pipeline owner has reported an inconsistent OK outcome.
        if outcome is StreamOutcome.OK and self.terminal.finish_reason == "length":
            outcome = StreamOutcome.TRUNCATED
        structured_error = ""
        empty_answer = False
        if outcome is StreamOutcome.OK and self._structured:
            try:
                validate_structured_output(self._adapter.text, self.request)
            except StructuredOutputError as exc:
                outcome = StreamOutcome.ERROR
                structured_error = str(exc)
        if outcome is StreamOutcome.OK and not self._adapter.text.strip():
            outcome = StreamOutcome.ERROR
            empty_answer = True
        run_status = {
            StreamOutcome.OK: "succeeded",
            StreamOutcome.CANCELLED: "cancelled",
            StreamOutcome.ERROR: "failed",
            StreamOutcome.TRUNCATED: "failed",
        }[outcome]
        stage_status = {
            StreamOutcome.OK: "succeeded",
            StreamOutcome.CANCELLED: "cancelled",
            StreamOutcome.ERROR: "failed",
            StreamOutcome.TRUNCATED: "failed",
        }[outcome]
        self._events.finish_open_stages(
            status=stage_status,  # type: ignore[arg-type]
            detail="stream ended before the stage reported completion",
        )
        message = self._events.message_finished(
            status="completed" if outcome is StreamOutcome.OK else "incomplete"
        )
        if structured_error:
            error_code = "structured_output_invalid"
        elif empty_answer:
            error_code = "empty_answer"
        else:
            error_code = {
                StreamOutcome.OK: "",
                StreamOutcome.CANCELLED: "cancelled",
                StreamOutcome.ERROR: "pipeline_error",
                StreamOutcome.TRUNCATED: "stream_truncated",
            }[outcome]
        finished = self._events.run_finished(
            status=run_status,  # type: ignore[arg-type]
            finish_reason=self.terminal.finish_reason or "",
            error_code=error_code,
            concrete_model=self.concrete_model or self.fingerprint_model,
        )
        message_frame = self._adapter.render(message)
        terminal_frame = self._adapter.render(finished)
        assert message_frame is not None and terminal_frame is not None
        return message_frame + terminal_frame

    def done_frame(self) -> str:
        if not self._adapter._terminal_emitted:
            raise RuntimeError("response terminal event must precede stream end")
        if self._done_emitted:
            raise RuntimeError("response stream end already emitted")
        self._done_emitted = True
        return ""


__all__ = [
    "OpenAIStreamAdapter",
    "OpenAIStreamSession",
    "ResponsesStreamAdapter",
    "ResponsesStreamSession",
    "StreamOutcome",
    "StreamTerminal",
]
