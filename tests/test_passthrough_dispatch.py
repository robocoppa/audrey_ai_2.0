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

    async def chat(self, *, model, messages, options=None, timeout_s=None):
        self.chat_calls.append({
            "model": model, "messages": messages,
            "options": options, "timeout_s": timeout_s,
        })
        return self.chat_response

    async def chat_stream(self, *, model, messages, options=None, timeout_s=None):
        self.stream_calls.append({
            "model": model, "messages": messages,
            "options": options, "timeout_s": timeout_s,
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
) -> SimpleNamespace:
    """Build the minimal `request.app` shape `_handle_passthrough` reads."""
    cfg = SimpleNamespace(
        raw={"passthrough": passthrough_block or {
            "enabled": True,
            "allowed_models": ["qwen3.6:35b-64k"],
            "require_role": None,
        }},
        timeouts={"medium": 180},
    )
    registry = SimpleNamespace(location_of=lambda _name: location)
    state = SimpleNamespace(
        cfg=cfg, registry=registry, ollama=ollama, gate=gate, inflight=inflight,
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
