"""OpenAI SSE framing for Audrey's client-neutral stream lifecycle."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from audrey import __version__
from audrey.pipeline.run_events import (
    AssistantMessageStartedEvent,
    RunEvent,
    RunEventEmitter,
    RunFinishedEvent,
    StageProgressEvent,
    TextDeltaEvent,
)
from audrey.pipeline.streaming import StreamOutcome, StreamTerminal


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


__all__ = [
    "OpenAIStreamAdapter",
    "OpenAIStreamSession",
    "StreamOutcome",
    "StreamTerminal",
]
