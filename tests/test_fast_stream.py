"""Regression tests for the unified no-tools Fast stream."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline import fast_path as fast_path_module
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.fast_path import (
    FastStreamEventType,
    stream_fast_path,
)
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai import pipeline as route_pipeline
from audrey.routes.openai.pipeline import _stream_via_pipeline
from audrey.routes.openai.schemas import ChatCompletionRequest
from audrey.routes.openai.streaming import (
    OpenAIStreamSession,
    StreamOutcome,
    StreamTerminal,
)


class _Cfg:
    def __init__(
        self,
        *models: tuple[str, int, str],
        no_thinking: bool = True,
        no_thinking_prose: bool | None = False,
    ) -> None:
        self.model_registry = {
            "general": [
                {
                    "name": name,
                    "priority": priority,
                    "location": location,
                }
                for name, priority, location in models
            ],
        }
        fast_path: dict[str, Any] = {
            "tool_capable_models": [],
            "no_thinking": no_thinking,
        }
        if no_thinking_prose is not None:
            fast_path["no_thinking_prose"] = no_thinking_prose
        self.raw = {
            "complexity": {"token_threshold": 500},
            "fast_path": fast_path,
        }
        self.router: dict[str, Any] = {}
        self.timeouts = {"fast_path": 5.0}


class _ScriptedOllama:
    """Raw stream outcomes per model, with call and thinking-policy capture."""

    def __init__(self, outcomes: dict[str, list[dict[str, Any] | BaseException]]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []
        self.flag_calls: list[tuple[str, bool]] = []

    async def thinking_flag(self, model: str, want: bool) -> bool | None:
        self.flag_calls.append((model, want))
        return want

    async def chat_stream(
        self,
        *,
        model: str,
        messages,
        options=None,
        tools=None,
        timeout_s=None,
        think=None,
    ):
        self.calls.append({
            "model": model,
            "messages": messages,
            "options": options,
            "tools": tools,
            "timeout_s": timeout_s,
            "think": think,
        })
        for item in self.outcomes[model]:
            if isinstance(item, BaseException):
                raise item
            yield item


class _BlockingOllama:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.settled = asyncio.Event()

    async def thinking_flag(self, model: str, want: bool) -> bool | None:
        return want

    async def chat_stream(self, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.settled.set()
        if False:
            yield {}


class _Metric:
    def __init__(self) -> None:
        self.labels_seen: list[dict[str, str]] = []
        self.values: list[float] = []

    def labels(self, **labels):
        self.labels_seen.append(labels)
        return self

    def inc(self) -> None:
        self.values.append(1.0)

    def observe(self, value: float) -> None:
        self.values.append(value)


@pytest.fixture
def recorded_metrics(monkeypatch):
    dispatch = _Metric()
    total = _Metric()
    seconds = _Metric()
    monkeypatch.setattr(fast_path_module, "dispatch_total", dispatch)
    monkeypatch.setattr(fast_path_module, "pipeline_total", total)
    monkeypatch.setattr(fast_path_module, "pipeline_seconds", seconds)
    return dispatch, total, seconds


def _registry(*models: tuple[str, int, str]) -> ModelRegistry:
    return ModelRegistry(_Cfg(*models))  # type: ignore[arg-type]


async def _events(
    ollama,
    registry,
    health,
    *,
    no_thinking_prose: bool = False,
    pipeline_started_at: float | None = None,
    terminal: StreamTerminal | None = None,
):
    terminal = terminal or StreamTerminal()
    events = [
        event
        async for event in stream_fast_path(
            ollama,
            registry,
            health,
            FairLocalGate(concurrency=1),
            task="general",
            messages=[{"role": "user", "content": "hello"}],
            options={},
            timeout_s=5.0,
            user_id="alice@example.com",
            pipeline_started_at=pipeline_started_at,
            no_thinking_prose=no_thinking_prose,
            terminal=terminal,
        )
    ]
    return events, terminal


async def test_plain_fast_success_has_one_terminal_and_matching_metrics(
    recorded_metrics,
):
    ollama = _ScriptedOllama({
        "a": [{
            "message": {"content": "hello"},
            "done": True,
            "done_reason": "stop",
        }],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama,
        _registry(("a", 100, "local")),
        health,
    )

    assert [event.type for event in events] == [
        FastStreamEventType.ATTEMPT,
        FastStreamEventType.STARTED,
        FastStreamEventType.TEXT,
    ]
    assert "".join(event.text for event in events) == "hello"
    assert terminal.outcome == StreamOutcome.OK
    assert terminal.finish_reason == "stop"
    assert health.is_healthy("a")
    dispatch, total, seconds = recorded_metrics
    assert dispatch.labels_seen == [{
        "model": "a",
        "task_type": "general",
        "path": "fast",
    }]
    assert total.labels_seen == [{
        "mode": "fast",
        "task_type": "general",
        "outcome": "ok",
    }]
    assert len(seconds.values) == 1


async def test_plain_fast_latency_includes_preclassification_time(
    monkeypatch,
    recorded_metrics,
):
    monkeypatch.setattr(fast_path_module.time, "perf_counter", lambda: 15.0)
    ollama = _ScriptedOllama({
        "a": [{
            "message": {"content": "answer"},
            "done": True,
        }],
    })

    await _events(
        ollama,
        _registry(("a", 100, "local")),
        HealthTracker(),
        pipeline_started_at=10.0,
    )

    _dispatch, _total, seconds = recorded_metrics
    assert seconds.values == [5.0]


async def test_plain_fast_falls_back_only_before_first_text(recorded_metrics):
    ollama = _ScriptedOllama({
        "a": [OllamaError("a down")],
        "b": [{
            "message": {"content": "fallback answer"},
            "done": True,
        }],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama,
        _registry(("a", 100, "local"), ("b", 50, "local")),
        health,
        no_thinking_prose=True,
    )

    assert [call["model"] for call in ollama.calls] == ["a", "b"]
    assert [call["think"] for call in ollama.calls] == [False, False]
    assert [event.model for event in events if event.type == FastStreamEventType.STARTED] == ["b"]
    assert "".join(event.text for event in events) == "fallback answer"
    assert terminal.outcome == StreamOutcome.OK
    assert not health.is_healthy("a")
    assert health.is_healthy("b")
    dispatch, total, _seconds = recorded_metrics
    assert [labels["model"] for labels in dispatch.labels_seen] == ["a", "b"]
    assert total.labels_seen[0]["outcome"] == "ok"


async def test_plain_fast_empty_missing_done_falls_back(recorded_metrics):
    ollama = _ScriptedOllama({
        "a": [],
        "b": [{
            "message": {"content": "fallback answer"},
            "done": True,
        }],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama,
        _registry(("a", 100, "local"), ("b", 50, "local")),
        health,
    )

    assert [call["model"] for call in ollama.calls] == ["a", "b"]
    assert [
        event.model
        for event in events
        if event.type == FastStreamEventType.STARTED
    ] == ["b"]
    assert "".join(event.text for event in events) == "fallback answer"
    assert terminal.outcome == StreamOutcome.OK
    assert not health.is_healthy("a")
    assert health.is_healthy("b")


async def test_plain_fast_does_not_fallback_after_text(recorded_metrics):
    ollama = _ScriptedOllama({
        "a": [
            {"message": {"content": "partial answer"}, "done": False},
            OllamaError("a failed mid-stream"),
        ],
        "b": [{
            "message": {"content": "must not run"},
            "done": True,
        }],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama,
        _registry(("a", 100, "local"), ("b", 50, "local")),
        health,
    )

    assert [call["model"] for call in ollama.calls] == ["a"]
    assert "".join(
        event.text for event in events
        if event.type == FastStreamEventType.TEXT
    ) == "partial answer"
    assert events[-1].type == FastStreamEventType.ERROR
    assert "mid-stream" in events[-1].text
    assert terminal.outcome == StreamOutcome.ERROR
    assert terminal.finish_reason == "stop"
    assert recorded_metrics[1].labels_seen[0]["outcome"] == "error"


async def test_plain_fast_missing_done_is_truncated_and_unhealthy(recorded_metrics):
    ollama = _ScriptedOllama({
        "a": [{"message": {"content": "partial answer"}, "done": False}],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama,
        _registry(("a", 100, "local")),
        health,
    )

    assert "".join(
        event.text for event in events
        if event.type == FastStreamEventType.TEXT
    ) == "partial answer"
    assert terminal.outcome == StreamOutcome.TRUNCATED
    assert terminal.finish_reason == "length"
    assert not health.is_healthy("a")
    assert recorded_metrics[1].labels_seen[0]["outcome"] == "truncated"


async def test_plain_fast_cancellation_records_cancelled_once(recorded_metrics):
    ollama = _BlockingOllama()
    terminal = StreamTerminal()
    registry = _registry(("a", 100, "local"))
    health = HealthTracker()

    async def drain() -> None:
        async for _event in stream_fast_path(
            ollama,  # type: ignore[arg-type]
            registry,
            health,
            FairLocalGate(concurrency=1),
            task="general",
            messages=[{"role": "user", "content": "hello"}],
            options={},
            timeout_s=5.0,
            user_id="alice@example.com",
            terminal=terminal,
        ):
            pass

    consumer = asyncio.create_task(drain())
    await asyncio.wait_for(ollama.started.wait(), timeout=1)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.wait_for(ollama.settled.wait(), timeout=1)

    assert terminal.outcome == StreamOutcome.CANCELLED
    assert recorded_metrics[1].labels_seen == [{
        "mode": "fast",
        "task_type": "general",
        "outcome": "cancelled",
    }]
    assert health.is_healthy("a")


async def test_inline_dangling_think_replay_renders_once(recorded_metrics):
    ollama = _ScriptedOllama({
        "a": [
            {"message": {"content": "H4-OWUI"}, "done": False},
            {"message": {"content": "-READY</"}, "done": False},
            {"message": {"content": "think>\nH4-"}, "done": False},
            {"message": {"content": "OWUI-READY"}, "done": True},
        ],
    })

    events, terminal = await _events(
        ollama,
        _registry(("a", 100, "local")),
        HealthTracker(),
    )
    answer = "".join(
        event.text for event in events
        if event.type == FastStreamEventType.TEXT
    )

    assert answer == "H4-OWUI-READY"
    assert "</think>" not in answer
    assert terminal.outcome == StreamOutcome.OK


def test_openai_stream_session_refuses_duplicate_lifecycle_frames():
    session = OpenAIStreamSession(
        virtual_model="audrey_fast",
        fingerprint_model="audrey_fast",
        completion_id="chatcmpl-fixed",
        created=123,
    )
    role = json.loads(session.role_frame()[6:].strip())
    content = json.loads(session.content_frame("hello")[6:].strip())
    session.terminal.finish(StreamOutcome.OK, finish_reason="stop")
    terminal = json.loads(session.terminal_frame()[6:].strip())

    assert {role["id"], content["id"], terminal["id"]} == {"chatcmpl-fixed"}
    assert role["choices"][0]["delta"] == {"role": "assistant"}
    assert content["choices"][0]["delta"] == {"content": "hello"}
    assert terminal["choices"][0]["finish_reason"] == "stop"
    assert session.done_frame() == "data: [DONE]\n\n"
    with pytest.raises(RuntimeError, match="role frame already"):
        session.role_frame()
    with pytest.raises(RuntimeError, match="terminal frame already"):
        session.terminal_frame()
    with pytest.raises(RuntimeError, match="DONE"):
        session.done_frame()


class _NoTools:
    by_name: ClassVar[dict[str, Any]] = {}



class _RecordingArchive:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def archive_turn(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _route_app(cfg: _Cfg, ollama, archive: _RecordingArchive):
    return SimpleNamespace(state=SimpleNamespace(
        cfg=cfg,
        ollama=ollama,
        registry=ModelRegistry(cfg),  # type: ignore[arg-type]
        health=HealthTracker(),
        inflight=UserInflightRegistry(max_inflight_per_user=2),
        gate=FairLocalGate(concurrency=1),
        tools=_NoTools(),
        archive_client=archive,
    ))

async def test_route_uses_one_id_one_role_and_the_configured_thinking_policy(
    monkeypatch,
):
    cfg = _Cfg(
        ("a", 100, "local"),
        no_thinking=True,
        no_thinking_prose=False,
    )
    ollama = _ScriptedOllama({
        "a": [{
            "message": {"content": "one answer"},
            "done": True,
        }],
    })

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    archive = _RecordingArchive()
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    app = _route_app(cfg, ollama, archive)
    messages = [{"role": "user", "content": "hello"}]
    payload = ChatCompletionRequest(
        model="audrey_fast",
        messages=messages,
        stream=True,
    )

    frames = [
        frame async for frame in _stream_via_pipeline(
            app,
            payload,
            messages,
            {},
            user_id="alice@example.com",
            conversation_id="conversation-1",
            user_turn_text="hello",
        )
    ]
    chunks = [
        json.loads(frame[6:].strip())
        for frame in frames
        if frame.startswith("data: ") and frame.strip() != "data: [DONE]"
    ]

    assert len({chunk["id"] for chunk in chunks}) == 1
    assert sum(
        chunk["choices"][0]["delta"].get("role") == "assistant"
        for chunk in chunks
    ) == 1
    assert sum(
        chunk["choices"][0]["finish_reason"] is not None
        for chunk in chunks
    ) == 1
    answer = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
    )
    assert answer.count("one answer") == 1
    assert frames[-1] == "data: [DONE]\n\n"
    assert ollama.calls[0]["think"] is None
    assert len(archive.calls) == 1
    assert archive.calls[0]["assistant_content"] == "one answer"
    assert archive.calls[0]["partial"] is False
    assert archive.calls[0]["concrete_model"] == "a"


async def test_route_archives_missing_done_as_partial_without_banner_text(
    monkeypatch,
):
    cfg = _Cfg(("a", 100, "local"))
    ollama = _ScriptedOllama({
        "a": [{
            "message": {"content": "partial answer"},
            "done": False,
        }],
    })

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    archive = _RecordingArchive()
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    app = _route_app(cfg, ollama, archive)
    messages = [{"role": "user", "content": "hello"}]
    payload = ChatCompletionRequest(
        model="audrey_fast",
        messages=messages,
        stream=True,
    )

    frames = [
        frame async for frame in _stream_via_pipeline(
            app,
            payload,
            messages,
            {},
            user_id="alice@example.com",
            conversation_id="conversation-1",
            user_turn_text="hello",
        )
    ]
    chunks = [
        json.loads(frame[6:].strip())
        for frame in frames
        if frame.startswith("data: ") and frame.strip() != "data: [DONE]"
    ]

    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert frames[-1] == "data: [DONE]\n\n"
    assert len(archive.calls) == 1
    assert archive.calls[0]["assistant_content"] == "partial answer"
    assert archive.calls[0]["partial"] is True
    assert archive.calls[0]["concrete_model"] == "a"
