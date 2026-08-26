"""OpenAI SSE framing for Audrey's client-neutral stream lifecycle."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from audrey import __version__
from audrey.pipeline.streaming import StreamOutcome, StreamTerminal


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
    _role_emitted: bool = False
    _terminal_emitted: bool = False
    _done_emitted: bool = False

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

    def role_frame(self) -> str:
        if self._role_emitted:
            raise RuntimeError("assistant role frame already emitted")
        if self._terminal_emitted:
            raise RuntimeError("cannot emit assistant role after stream terminal")
        self._role_emitted = True
        return self._frame({"role": "assistant"}, None)

    def content_frame(self, text: str) -> str:
        if not self._role_emitted:
            raise RuntimeError("assistant role frame must precede content")
        if self._terminal_emitted:
            raise RuntimeError("cannot emit content after stream terminal")
        return self._frame({"content": text}, None)

    def terminal_frame(self) -> str:
        if not self._role_emitted:
            raise RuntimeError("assistant role frame must precede stream terminal")
        if self._terminal_emitted:
            raise RuntimeError("stream terminal frame already emitted")
        finish_reason = self.terminal.finish_reason
        self._terminal_emitted = True
        return self._frame({}, finish_reason)

    def done_frame(self) -> str:
        if not self._terminal_emitted:
            raise RuntimeError("stream terminal frame must precede [DONE]")
        if self._done_emitted:
            raise RuntimeError("stream [DONE] marker already emitted")
        self._done_emitted = True
        return "data: [DONE]\n\n"


__all__ = [
    "OpenAIStreamSession",
    "StreamOutcome",
    "StreamTerminal",
]
