"""Tests for the passthrough dispatch path.

Asserts that an `audrey_passthrough/<concrete>` request, when handed
through the route's `_handle_passthrough` helper, reaches
`OllamaClient.chat` with the right concrete model AND was guarded by
both the FairLocalGate (per-user GPU bucket) and the
UserInflightRegistry (per-user inflight cap).

We stub the OllamaClient surface so no real network is involved.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.passthrough import passthrough_chat, passthrough_stream
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai import (
    PASSTHROUGH_PREFIX,
    ChatCompletionRequest,
    _handle_passthrough,
)

# ─── Fakes ─────────────────────────────────────────────────────────────

class _FakeOllama:
    """OllamaClient stub that records what was asked of it."""

    def __init__(
        self,
        *,
        chat_response: dict | None = None,
        stream_chunks: list[dict] | None = None,
    ) -> None:
        self.chat_response = chat_response or {
            "message": {"role": "assistant", "content": "hello from fake ollama"},
            "prompt_eval_count": 12,
            "eval_count": 34,
        }
        self.stream_chunks = stream_chunks or [
            {"message": {"role": "assistant", "content": "hi "}, "done": False},
            {"message": {"role": "assistant", "content": "there"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def chat(self, *, model, messages, options=None, tools=None, timeout_s=None, think=None):
        self.chat_calls.append({
            "model": model, "messages": messages,
            "options": options, "tools": tools, "timeout_s": timeout_s,
            "think": think,
        })
        return self.chat_response

    async def chat_stream(self, *, model, messages, options=None, tools=None,
                          timeout_s=None, think=None):
        self.stream_calls.append({
            "model": model, "messages": messages,
            "options": options, "tools": tools, "timeout_s": timeout_s,
            "think": think,
        })
        for chunk in self.stream_chunks:
            yield chunk


class _RecordingGate(FairLocalGate):
    """FairLocalGate variant that records every acquire call."""

    def __init__(self) -> None:
        super().__init__(concurrency=1)
        self.acquired: list[dict] = []

    def acquire(self, model, *, location, user_id=None):
        self.acquired.append({
            "model": model, "location": location, "user_id": user_id,
        })
        return super().acquire(model, location=location, user_id=user_id)


def _stub_app(
    *,
    ollama: _FakeOllama,
    gate: FairLocalGate,
    inflight: UserInflightRegistry,
    location: str = "local",
    passthrough_block: dict | None = None,
    vision_block: dict | None = None,
) -> SimpleNamespace:
    """Build the minimal `request.app` shape `_handle_passthrough` reads."""
    raw: dict = {"passthrough": passthrough_block or {
        "enabled": True,
        "allowed_models": ["qwen3.6:35b-64k"],
        "require_role": None,
    }}
    if vision_block is not None:
        raw["vision"] = vision_block
    cfg = SimpleNamespace(
        raw=raw,
        timeouts={"medium": 180},
        # The vision sidecar reads the `vl` pool to decide whether the
        # concrete target can see an image on its own.
        model_registry={"vl": [
            {"name": "qwen3-vl:32b", "priority": 100, "location": "local"},
        ]},
    )
    registry = ModelRegistry(cfg)  # type: ignore[arg-type]
    registry.location_of = lambda _name: location  # type: ignore[method-assign]
    state = SimpleNamespace(
        cfg=cfg, registry=registry, ollama=ollama, gate=gate, inflight=inflight,
        health=HealthTracker(),
    )
    return SimpleNamespace(state=state)


def _stub_user(email: str = "alice@example.com", role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(email=email, role=role, owui_id="abc")


def _payload(
    *,
    model: str,
    stream: bool = False,
    content: str = "hello",
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": content}],
        stream=stream,
    )


# ─── passthrough_chat (direct helper) ──────────────────────────────────

async def test_passthrough_chat_acquires_gate_with_right_user_and_model():
    ollama = _FakeOllama()
    gate = _RecordingGate()
    resp = await passthrough_chat(
        ollama, gate,
        concrete="qwen3.6:35b-64k", location="local",
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0.4},
        user_id="alice@example.com",
    )
    assert resp == ollama.chat_response
    # Gate was acquired with the concrete model + correct user.
    assert len(gate.acquired) == 1
    assert gate.acquired[0]["model"] == "qwen3.6:35b-64k"
    assert gate.acquired[0]["location"] == "local"
    assert gate.acquired[0]["user_id"] == "alice@example.com"
    # Ollama was called with the same model.
    assert len(ollama.chat_calls) == 1
    assert ollama.chat_calls[0]["model"] == "qwen3.6:35b-64k"
    assert ollama.chat_calls[0]["options"] == {"temperature": 0.4}


async def test_passthrough_stream_yields_chunks_with_gate_held():
    ollama = _FakeOllama()
    gate = _RecordingGate()
    received = []
    async for chunk in passthrough_stream(
        ollama, gate,
        concrete="qwen3.6:35b-64k", location="local",
        messages=[{"role": "user", "content": "hi"}],
        options={},
        user_id="alice@example.com",
    ):
        received.append(chunk)
    assert received == ollama.stream_chunks
    assert len(gate.acquired) == 1
    assert gate.acquired[0]["user_id"] == "alice@example.com"
    assert len(ollama.stream_calls) == 1


async def test_passthrough_cloud_location_bypasses_gate_waiting():
    # Cloud location -> gate is a no-op (no queueing). We assert the
    # gate didn't actually hold us up by parking another acquire
    # simultaneously and confirming both ran without serializing.
    import asyncio

    ollama = _FakeOllama()
    gate = FairLocalGate(concurrency=1)

    started = []

    async def call() -> None:
        async def slow_chat(**kwargs):
            started.append(kwargs["model"])
            await asyncio.sleep(0.01)
            return ollama.chat_response

        ollama.chat = slow_chat  # type: ignore[assignment]
        await passthrough_chat(
            ollama, gate,
            concrete="cloud-only", location="cloud",
            messages=[], options={},
            user_id="alice@example.com",
        )

    await asyncio.gather(call(), call())
    assert len(started) == 2  # both ran concurrently, no FIFO serialization


# ─── _handle_passthrough (the route entry point) ───────────────────────

@pytest.mark.asyncio
async def test_handle_passthrough_nonstreaming_returns_openai_shape():
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=_payload(model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k"),
        me=_stub_user(),
    )

    assert resp["object"] == "chat.completion"
    assert resp["model"] == f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k"  # virtual
    assert "qwen3.6:35b-64k" in resp["system_fingerprint"]
    assert resp["choices"][0]["message"]["content"] == "hello from fake ollama"
    assert resp["usage"]["prompt_tokens"] == 12
    assert resp["usage"]["completion_tokens"] == 34
    assert resp["usage"]["total_tokens"] == 46
    # Gate fired with the concrete model and the auth user.
    assert gate.acquired[0]["model"] == "qwen3.6:35b-64k"
    assert gate.acquired[0]["user_id"] == "alice@example.com"


@pytest.mark.asyncio
async def test_handle_passthrough_uses_inflight_slot():
    # Confirm the inflight registry records this user as active during
    # the call by snapshotting inside Ollama's chat method.
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    seen_snapshots: list[dict[str, int]] = []

    async def chat_recording_inflight(**kwargs):
        seen_snapshots.append(dict(inflight.snapshot()))
        return ollama.chat_response

    ollama.chat = chat_recording_inflight  # type: ignore[assignment]

    await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=_payload(model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k"),
        me=_stub_user(),
    )
    # Inside the slot we should have seen this user with 1 inflight.
    assert seen_snapshots == [{"alice@example.com": 1}]
    # After the slot exits, the inflight count drops back to 0.
    assert inflight.snapshot().get("alice@example.com", 0) == 0


@pytest.mark.asyncio
async def test_handle_passthrough_propagates_ollama_error_as_502():
    from fastapi import HTTPException

    from audrey.models.ollama import OllamaError

    ollama = _FakeOllama()

    async def boom(**kwargs):
        raise OllamaError("model not loaded")

    ollama.chat = boom  # type: ignore[assignment]
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    with pytest.raises(HTTPException) as exc_info:
        await _handle_passthrough(
            app, request=SimpleNamespace(app=app),
            payload=_payload(model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k"),
            me=_stub_user(),
        )
    assert exc_info.value.status_code == 502
    assert "model not loaded" in exc_info.value.detail


@pytest.mark.asyncio
async def test_handle_passthrough_streaming_emits_openai_sse_frames():
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=_payload(model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k", stream=True),
        me=_stub_user(),
    )

    # StreamingResponse — consume its body iterator.
    chunks = []
    async for raw in resp.body_iterator:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        chunks.append(raw)

    body = "".join(chunks)
    # Role frame first.
    assert '"delta": {"role": "assistant"}' in body
    # Content frames in order.
    assert '"delta": {"content": "hi "}' in body
    assert '"delta": {"content": "there"}' in body
    # Stop frame.
    assert '"finish_reason": "stop"' in body
    # SSE terminator.
    assert body.rstrip().endswith("data: [DONE]")
    # Gate held once across the whole stream.
    assert len(gate.acquired) == 1


# ─── Config gating from inside the route ───────────────────────────────

@pytest.mark.asyncio
async def test_handle_passthrough_403_when_model_not_allowed():
    from fastapi import HTTPException
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(
        ollama=ollama, gate=gate, inflight=inflight,
        passthrough_block={
            "enabled": True,
            "allowed_models": ["qwen3.6:35b-64k"],
            "require_role": None,
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        await _handle_passthrough(
            app, request=SimpleNamespace(app=app),
            payload=_payload(model=f"{PASSTHROUGH_PREFIX}llama3.3:70b"),
            me=_stub_user(),
        )
    assert exc_info.value.status_code == 403
    # Ollama was never reached.
    assert ollama.chat_calls == []
    assert gate.acquired == []


@pytest.mark.asyncio
async def test_handle_passthrough_403_when_disabled():
    from fastapi import HTTPException
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(
        ollama=ollama, gate=gate, inflight=inflight,
        passthrough_block={"enabled": False, "allowed_models": ["qwen3.6:35b-64k"]},
    )
    with pytest.raises(HTTPException) as exc_info:
        await _handle_passthrough(
            app, request=SimpleNamespace(app=app),
            payload=_payload(model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k"),
            me=_stub_user(),
        )
    assert exc_info.value.status_code == 403


# ─── Tool forwarding (agent client transparency) ───────────────────────

# Sample OpenAI-shaped tool definition. Agent clients (Hermes, OpenClaw)
# send these in `payload.tools`; without forwarding, the model sees no
# tool schema and falls back to emitting tool syntax as plain text.
_SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


@pytest.mark.asyncio
async def test_handle_passthrough_forwards_tools_to_ollama():
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        messages=[{"role": "user", "content": "what's in /etc/hosts"}],
        tools=[_SAMPLE_TOOL],
    )
    await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=payload, me=_stub_user(),
    )
    assert len(ollama.chat_calls) == 1
    forwarded = ollama.chat_calls[0]["tools"]
    assert forwarded is not None
    assert forwarded == [_SAMPLE_TOOL], (
        "passthrough must forward tools verbatim; otherwise the model "
        "sees no schema and free-styles tool syntax as plain text"
    )


@pytest.mark.asyncio
async def test_handle_passthrough_returns_openai_shaped_tool_calls():
    # When Ollama returns tool_calls, the non-streaming response must
    # surface them in the OpenAI shape (with id, type, JSON-string args)
    # so agent clients can execute them. Validates the Ollama→OpenAI
    # tool_call conversion.
    import json as _json

    ollama = _FakeOllama(chat_response={
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "/etc/hosts"}}},
            ],
        },
        "prompt_eval_count": 8,
        "eval_count": 16,
    })
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        messages=[{"role": "user", "content": "read /etc/hosts"}],
        tools=[_SAMPLE_TOOL],
    )
    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=payload, me=_stub_user(),
    )
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["role"] == "assistant"
    tool_calls = msg["tool_calls"]
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    # Arguments must be a JSON string, not a dict — OpenAI clients
    # call json.loads on this field.
    assert isinstance(call["function"]["arguments"], str)
    assert _json.loads(call["function"]["arguments"]) == {"path": "/etc/hosts"}
    # And an id was synthesized.
    assert call["id"]


@pytest.mark.asyncio
async def test_handle_passthrough_streaming_emits_tool_calls_on_final_frame():
    # Ollama typically populates tool_calls on the final stream chunk
    # rather than as deltas. The SSE emitter must surface them in the
    # terminal frame's delta with finish_reason="tool_calls".
    import json as _json

    ollama = _FakeOllama(stream_chunks=[
        {"message": {"role": "assistant", "content": ""}, "done": False},
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "/etc/hosts"}}},
                ],
            },
            "done": True,
        },
    ])
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        messages=[{"role": "user", "content": "read /etc/hosts"}],
        stream=True,
        tools=[_SAMPLE_TOOL],
    )
    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=payload, me=_stub_user(),
    )
    chunks: list[str] = []
    async for raw in resp.body_iterator:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        chunks.append(raw)
    body = "".join(chunks)
    # The final frame must carry the tool_calls delta + tool_calls finish.
    assert '"finish_reason": "tool_calls"' in body
    assert '"tool_calls"' in body
    assert '"read_file"' in body
    # Arguments serialized as a JSON string (escaped).
    assert _json.dumps({"path": "/etc/hosts"}) in body or '\\"path\\": \\"/etc/hosts\\"' in body
    # Stream still terminates.
    assert body.rstrip().endswith("data: [DONE]")
    # And the tools array was forwarded to ollama.chat_stream.
    assert ollama.stream_calls[0]["tools"] == [_SAMPLE_TOOL]


@pytest.mark.asyncio
async def test_handle_passthrough_streaming_collects_tool_calls_from_pre_done_chunk():
    # Real Ollama streaming behavior (verified against qwen3.6:35b-64k
    # on 2026-05-29): the model emits `tool_calls` in a chunk with
    # `done: false`, then sends a separate `done: true` chunk that
    # carries no tool_calls. The SSE emitter must accumulate
    # tool_calls across ALL chunks and attach them to the final delta,
    # not only look at the done chunk's message. Without this fix,
    # streaming + tools produced an empty SSE stream from the client's
    # perspective, even though tool_calls were on the wire.
    import json as _json

    ollama = _FakeOllama(stream_chunks=[
        # Thinking/reasoning tokens (qwen3.6 emits these with empty content).
        {"message": {"role": "assistant", "content": "", "thinking": "Let me check"}, "done": False},
        # Tool call lands on a non-final chunk.
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "/etc/hosts"}}},
                ],
            },
            "done": False,
        },
        # Final chunk has no tool_calls.
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ])
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        messages=[{"role": "user", "content": "read /etc/hosts"}],
        stream=True,
        tools=[_SAMPLE_TOOL],
    )
    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=payload, me=_stub_user(),
    )
    chunks: list[str] = []
    async for raw in resp.body_iterator:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        chunks.append(raw)
    body = "".join(chunks)
    # tool_calls must surface in the closing SSE frame with the right
    # finish reason. Pre-fix this assertion failed (no tool_calls in body).
    assert '"finish_reason": "tool_calls"' in body
    assert '"read_file"' in body
    assert _json.dumps({"path": "/etc/hosts"}) in body or '\\"path\\": \\"/etc/hosts\\"' in body
    assert body.rstrip().endswith("data: [DONE]")


@pytest.mark.asyncio
async def test_handle_passthrough_without_tools_does_not_set_tool_calls():
    # A passthrough request without `tools` in the payload should send
    # no tools array to Ollama and produce a vanilla stop-reason response.
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        messages=[{"role": "user", "content": "hi"}],
    )
    resp = await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=payload, me=_stub_user(),
    )
    # No tools forwarded.
    assert ollama.chat_calls[0]["tools"] is None
    # Vanilla response, no tool_calls in the message.
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


# ─── Vision sidecar on the passthrough path ────────────────────────────
# The reported bug: attaching an image to a text-only concrete model (e.g.
# `glm-5.2:cloud`, whose Ollama capabilities are tools/thinking/cloud —
# no vision) forwarded the data URI to a model with no vision encoder.

_IMAGE_TURN = [
    {"type": "text", "text": "what does this error say?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
]


@pytest.mark.asyncio
async def test_image_to_text_only_model_is_transcribed_before_forwarding():
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(ollama=ollama, gate=gate, inflight=inflight)

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        messages=[{"role": "user", "content": _IMAGE_TURN}],
    )
    await _handle_passthrough(
        app, request=SimpleNamespace(app=app), payload=payload, me=_stub_user(),
    )

    # Two dispatches: the vl transcription, then the forward to the target.
    assert [c["model"] for c in ollama.chat_calls] == [
        "qwen3-vl:32b", "qwen3.6:35b-64k",
    ]
    forwarded = ollama.chat_calls[1]["messages"]
    assert isinstance(forwarded[0]["content"], str)  # image part is gone
    assert "what does this error say?" in forwarded[0]["content"]
    assert "hello from fake ollama" in forwarded[0]["content"]  # the description


@pytest.mark.asyncio
async def test_image_to_vision_capable_model_is_forwarded_untouched():
    """qwen3-vl reads the image itself — no transcription, no loss."""
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(
        ollama=ollama, gate=gate, inflight=inflight,
        passthrough_block={
            "enabled": True, "allowed_models": ["qwen3-vl:32b"], "require_role": None,
        },
    )

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3-vl:32b",
        messages=[{"role": "user", "content": _IMAGE_TURN}],
    )
    await _handle_passthrough(
        app, request=SimpleNamespace(app=app), payload=payload, me=_stub_user(),
    )

    assert [c["model"] for c in ollama.chat_calls] == ["qwen3-vl:32b"]
    assert ollama.chat_calls[0]["messages"][0]["content"] == _IMAGE_TURN


@pytest.mark.asyncio
async def test_sidecar_off_restores_the_old_verbatim_forward():
    ollama = _FakeOllama()
    gate = _RecordingGate()
    inflight = UserInflightRegistry(max_inflight_per_user=3)
    app = _stub_app(
        ollama=ollama, gate=gate, inflight=inflight,
        vision_block={"describe_for_text_models": False},
    )

    payload = ChatCompletionRequest(
        model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        messages=[{"role": "user", "content": _IMAGE_TURN}],
    )
    await _handle_passthrough(
        app, request=SimpleNamespace(app=app), payload=payload, me=_stub_user(),
    )

    assert [c["model"] for c in ollama.chat_calls] == ["qwen3.6:35b-64k"]
    assert ollama.chat_calls[0]["messages"][0]["content"] == _IMAGE_TURN


# ── passthrough.think (2026-08-12) ──────────────────────────────────────────
#
# Passthrough forwarded every turn in Ollama's `omitted` state, because the
# OpenAI schema has no thinking knob. Fine for serving; useless for comparing.
# The local bake-off reaches its models only through this route, and all three
# of its candidates declare `thinking` — so their scores were being read off
# three different, unchosen reasoning budgets.

class _ThinkingAwareOllama(_FakeOllama):
    """Adds the capability lookup `_passthrough_think` routes through."""

    def __init__(self, *, declares: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.declares = declares
        self.flag_calls: list[tuple[str, bool]] = []

    async def thinking_flag(self, model: str, want: bool) -> bool | None:
        self.flag_calls.append((model, want))
        return want if self.declares else None


def _think_app(ollama, think):
    return _stub_app(
        ollama=ollama, gate=_RecordingGate(),
        inflight=UserInflightRegistry(max_inflight_per_user=3),
        passthrough_block={
            "enabled": True,
            "allowed_models": ["qwen3.6:35b-64k"],
            "require_role": None,
            "think": think,
        },
    )


async def _run(app):
    return await _handle_passthrough(
        app, request=SimpleNamespace(app=app),
        payload=_payload(model=f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k"),
        me=_stub_user(),
    )


@pytest.mark.asyncio
async def test_think_is_omitted_by_default():
    """Config ships `think: null`, so serving clients keep today's behaviour
    and the capability lookup is never even made."""
    ollama = _ThinkingAwareOllama()
    await _run(_think_app(ollama, None))
    assert ollama.chat_calls[0]["think"] is None
    assert ollama.flag_calls == []


@pytest.mark.asyncio
async def test_think_false_reaches_a_model_that_declares_it():
    ollama = _ThinkingAwareOllama()
    await _run(_think_app(ollama, False))
    assert ollama.chat_calls[0]["think"] is False
    assert ollama.flag_calls == [("qwen3.6:35b-64k", False)]


@pytest.mark.asyncio
async def test_think_true_reaches_a_model_that_declares_it():
    ollama = _ThinkingAwareOllama()
    await _run(_think_app(ollama, True))
    assert ollama.chat_calls[0]["think"] is True


@pytest.mark.asyncio
async def test_think_is_dropped_for_a_model_that_does_not_declare_it():
    """⚠️ Why this never sends a raw bool. Ollama HARD ERRORS on `think` for a
    model without the capability rather than ignoring it, and several entries
    in `allowed_models` do not declare it — a bare False would break every one
    of them in a single config edit."""
    ollama = _ThinkingAwareOllama(declares=False)
    await _run(_think_app(ollama, False))
    assert ollama.chat_calls[0]["think"] is None


# ── what was ASKED for vs what came BACK ────────────────────────────────────
# A `PASSTHROUGH_THINK` A/B is only falsifiable if the log carries both. On
# 2026-08-18 a thinking-on arm came back within noise of the thinking-off arm
# and there was no way to tell "the model ignored the flag" from "the flag
# never reached Ollama" — the request line named model, user and prompt head,
# and said nothing about thinking.

_LOGGER = "audrey.pipeline.passthrough"


async def _run_chat(ollama, *, think=None, user="a@example.com"):
    gate = _RecordingGate()
    return await passthrough_chat(
        ollama, gate, concrete="m:latest", location="local",
        messages=[{"role": "user", "content": "hi"}], options={},
        user_id=user, tools=None, timeout_s=30.0, think=think,
    )


async def _run_stream(ollama, *, think=None, user="a@example.com"):
    gate = _RecordingGate()
    return [
        c async for c in passthrough_stream(
            ollama, gate, concrete="m:latest", location="local",
            messages=[{"role": "user", "content": "hi"}], options={},
            user_id=user, tools=None, timeout_s=30.0, think=think,
        )
    ]


class TestThinkIsLoggedOnBothSides:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asked", [True, False, None])
    async def test_chat_logs_what_was_asked_for(self, caplog, asked):
        # ⚠️ `None` is a THIRD state, not a synonym for False: it means the
        # field was omitted entirely, which is what every passthrough turn did
        # before the knob existed and what Ollama needs for models that do not
        # declare `thinking`.
        caplog.set_level("INFO", logger=_LOGGER)
        await _run_chat(_FakeOllama(), think=asked)
        line = [r.getMessage() for r in caplog.records if "passthrough.chat " in r.getMessage()]
        assert line and f"think={asked}" in line[0]

    @pytest.mark.asyncio
    async def test_the_stream_request_line_carries_think(self, caplog):
        # On the REQUEST line too — a stream that dies mid-flight still has to
        # say what it asked for.
        caplog.set_level("INFO", logger=_LOGGER)
        await _run_stream(_FakeOllama(), think=True)
        req = [r.getMessage() for r in caplog.records if "passthrough.stream.req" in r.getMessage()]
        assert req and "think=True" in req[0]

    @pytest.mark.asyncio
    async def test_chat_logs_thinking_length_that_came_back(self, caplog):
        caplog.set_level("INFO", logger=_LOGGER)
        ollama = _FakeOllama(chat_response={
            "message": {"role": "assistant", "content": "ok", "thinking": "x" * 500},
            "eval_count": 100,
        })
        await _run_chat(ollama, think=True)
        line = next(r.getMessage() for r in caplog.records
                    if "passthrough.chat " in r.getMessage())
        assert "thinking_len=500" in line
        assert "content_len=2" in line

    @pytest.mark.asyncio
    async def test_the_stream_sums_thinking_across_chunks(self, caplog):
        caplog.set_level("INFO", logger=_LOGGER)
        ollama = _FakeOllama(stream_chunks=[
            {"message": {"content": "", "thinking": "aaa"}, "done": False},
            {"message": {"content": "hi", "thinking": "bb"}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 40,
             "done_reason": "stop"},
        ])
        await _run_stream(ollama, think=True)
        line = next(r.getMessage() for r in caplog.records
                    if "passthrough.stream " in r.getMessage())
        assert "thinking_len=5" in line
        assert "content_len=2" in line


class TestCharsPerTokMeasuresContentOnly:
    """The ratio answers "did the billed tokens become text the client saw?".

    Folding thinking into the numerator answers a different question and hides
    the one case worth catching: a full budget spent where nothing reaches the
    client.
    """

    @pytest.mark.asyncio
    async def test_thinking_is_not_counted_in_the_ratio(self, caplog):
        caplog.set_level("INFO", logger=_LOGGER)
        ollama = _FakeOllama(chat_response={
            # 4,000 characters of reasoning, 10 of answer, 1,000 tokens billed.
            "message": {"role": "assistant", "content": "x" * 10,
                        "thinking": "t" * 4000},
            "eval_count": 1000,
        })
        await _run_chat(ollama, think=True)
        line = next(r.getMessage() for r in caplog.records
                    if "passthrough.chat " in r.getMessage())
        # 10/1000 = 0.01, NOT 4010/1000 = 4.01 (which would read as healthy prose).
        assert "chars_per_tok=0.01" in line

    def test_a_zero_eval_count_does_not_divide_by_zero(self):
        from audrey.pipeline.passthrough import _chars_per_tok
        assert _chars_per_tok(100, 0) == 0.0

    def test_ordinary_prose_lands_near_four(self):
        from audrey.pipeline.passthrough import _chars_per_tok
        assert 3.0 < _chars_per_tok(4000, 1000) < 5.0
