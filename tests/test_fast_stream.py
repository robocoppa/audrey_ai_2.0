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
from audrey.pipeline import memory as memory_module
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.fast_path import (
    FastStreamEventType,
    stream_fast_path,
)
from audrey.pipeline.run_events import RunEvent, RunEventContext
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai import pipeline as route_pipeline
from audrey.routes.openai.pipeline import (
    _generate_via_pipeline,
    _stream_via_pipeline,
)
from audrey.routes.openai.schemas import ChatCompletionRequest, ResponseCreateRequest
from audrey.routes.openai.streaming import (
    OpenAIStreamSession,
    ResponsesStreamSession,
    StreamOutcome,
    StreamTerminal,
)
from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.dispatch import ToolResult


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
        format=None,
    ):
        self.calls.append({
            "model": model,
            "messages": messages,
            "options": options,
            "tools": tools,
            "timeout_s": timeout_s,
            "think": think,
            "format": format,
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


async def test_plain_fast_stream_probes_when_all_candidates_are_cooling(recorded_metrics):
    health = HealthTracker()
    health.record_failure("a", "daemon restarted")
    ollama = _ScriptedOllama({"a": [{
        "message": {"content": "recovered"}, "done": True, "done_reason": "stop",
    }]})
    events, terminal = await _events(ollama, _registry(("a", 100, "local")), health)
    assert "".join(event.text for event in events) == "recovered"
    assert terminal.outcome == StreamOutcome.OK
    assert health.is_healthy("a")


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
        FastStreamEventType.USAGE,
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


@pytest.mark.parametrize("content", ["", "<think>hidden reasoning</think>", "partial answer"])
async def test_plain_fast_token_limit_is_incomplete_with_usage_and_no_retry(
    recorded_metrics, content,
):
    ollama = _ScriptedOllama({
        "a": [{
            "message": {"content": content, "thinking": "hidden reasoning"},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 1177,
            "eval_count": 4096,
        }],
        "b": [{"message": {"content": "must not run"}, "done": True}],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama, _registry(("a", 100, "local"), ("b", 50, "local")), health,
    )

    assert [call["model"] for call in ollama.calls] == ["a"]
    visible = "partial answer" if content == "partial answer" else ""
    assert "".join(
        event.text for event in events if event.type == FastStreamEventType.TEXT
    ) == visible
    assert sum(event.type == FastStreamEventType.STARTED for event in events) == bool(visible)
    usage = next(event for event in events if event.type == FastStreamEventType.USAGE)
    assert (usage.prompt_tokens, usage.completion_tokens) == (1177, 4096)
    assert terminal.outcome == StreamOutcome.TRUNCATED
    assert terminal.finish_reason == "length"
    assert health.is_healthy("a")
    assert recorded_metrics[1].labels_seen[0]["outcome"] == "truncated"


@pytest.mark.parametrize("content", ["", " \n" * 20, "<think>hidden reasoning</think>"])
async def test_plain_fast_empty_done_falls_back_before_announcing_answer(
    recorded_metrics, content,
):
    ollama = _ScriptedOllama({
        "a": [{
            "message": {"content": content, "thinking": "hidden reasoning"},
            "done": True,
            "done_reason": "stop",
            "eval_count": 12,
        }],
        "b": [{
            "message": {"content": "fallback answer"},
            "done": True,
            "eval_count": 3,
        }],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama, _registry(("a", 100, "local"), ("b", 50, "local")), health,
    )

    assert [call["model"] for call in ollama.calls] == ["a", "b"]
    assert [event.model for event in events if event.type == FastStreamEventType.STARTED] == ["b"]
    assert "".join(
        event.text for event in events if event.type == FastStreamEventType.TEXT
    ) == "fallback answer"
    assert [
        event.completion_tokens for event in events if event.type == FastStreamEventType.USAGE
    ] == [12, 3]
    assert terminal.outcome == StreamOutcome.OK
    assert not health.is_healthy("a")
    assert health.is_healthy("b")


async def test_plain_fast_empty_done_exhaustion_is_bounded_failure(recorded_metrics):
    empty = {"message": {"thinking": "hidden"}, "done": True, "done_reason": "stop"}
    ollama = _ScriptedOllama({
        "a": [empty], "b": [empty],
        "c": [{"message": {"content": "must not run"}, "done": True}],
    })
    health = HealthTracker()

    events, terminal = await _events(
        ollama,
        _registry(("a", 100, "local"), ("b", 50, "local"), ("c", 10, "local")),
        health,
    )

    assert [call["model"] for call in ollama.calls] == ["a", "b"]
    assert not any(event.type == FastStreamEventType.STARTED for event in events)
    assert events[-1].type == FastStreamEventType.ERROR
    assert "completed without answer text" in events[-1].text
    assert terminal.outcome == StreamOutcome.ERROR
    assert not health.is_healthy("a")
    assert not health.is_healthy("b")
    assert health.is_healthy("c")


async def test_plain_fast_preserves_leading_whitespace_once_answer_starts(recorded_metrics):
    ollama = _ScriptedOllama({
        "a": [
            {"message": {"content": " \n" * 20}, "done": False},
            {"message": {"content": "answer"}, "done": True},
        ],
    })

    events, terminal = await _events(
        ollama, _registry(("a", 100, "local")), HealthTracker(),
    )

    assert "".join(
        event.text for event in events if event.type == FastStreamEventType.TEXT
    ) == " \n" * 20 + "answer"
    assert sum(event.type == FastStreamEventType.STARTED for event in events) == 1
    assert terminal.outcome == StreamOutcome.OK


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


class _FixedGraph:
    async def ainvoke(self, _state):
        return {
            "content": "utility result",
            "mode": "fast",
            "task_type": "general",
            "classify_reason": "test",
            "classify_confidence": 1.0,
            "concrete_model": "a",
            "prompt_eval_count": 10,
            "eval_count": 2,
        }


class _ObservedGraph:
    def __init__(self) -> None:
        self.observer_seen: bool | None = None

    async def ainvoke(self, state):
        observer = state["tool_observer"]
        self.observer_seen = observer is not None
        if observer is not None:
            call_id = observer.started({
                "function": {
                    "name": "web_search",
                    "arguments": {
                        "query": "coffee history",
                        "api_key": "private-argument",
                    },
                },
            })
            observer.finished(
                call_id,
                ToolResult(
                    name="web_search",
                    call_id=None,
                    content='{"provider_debug":"private-result-body"}',
                    elapsed_s=0.01,
                    is_error=False,
                ),
                sources=[{
                    "title": "Coffee history",
                    "url": "https://example.com/coffee?token=private",
                    "tool": "web_search",
                }],
            )
        return {
            "content": "fast observed answer",
            "mode": "fast",
            "task_type": "general",
            "classify_reason": "test",
            "classify_confidence": 1.0,
            "concrete_model": "a",
            "prompt_eval_count": 10,
            "eval_count": 2,
        }


def _route_app(cfg: _Cfg, ollama, archive: _RecordingArchive):
    return SimpleNamespace(state=SimpleNamespace(
        cfg=cfg,
        ollama=ollama,
        registry=ModelRegistry(cfg),  # type: ignore[arg-type]
        health=HealthTracker(),
        inflight=UserInflightRegistry(max_inflight_per_user=2),
        gate=FairLocalGate(concurrency=1),
        tools=_NoTools(),
        archive_http=object(),
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
    events: list[RunEvent] = []

    frames = [
        frame async for frame in _stream_via_pipeline(
            app,
            payload,
            messages,
            {},
            user_id="alice@example.com",
            conversation_id="conversation-1",
            user_turn_text="hello",
            event_context=RunEventContext(
                run_id="run-fast",
                conversation_id="conversation-1",
                assistant_message_id="message-fast",
                mode="fast",
                sink=events.append,
            ),
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
    assert archive.calls == []
    assert [event.type for event in events] == [
        "run.started",
        "message.started",
        "stage.started",
        "stage.progress",
        "stage.progress",
        "stage.finished",
        "stage.progress",
        "text.delta",
        "text.delta",
        "usage.reported",
        "message.finished",
        "run.finished",
    ]
    assert "".join(event.delta for event in events if event.type == "text.delta") == (
        "one answer"
    )
    usage = next(event for event in events if event.type == "usage.reported")
    assert (usage.prompt_tokens, usage.completion_tokens) == (0, 0)


@pytest.mark.parametrize("content", ["", "partial answer"])
async def test_responses_vision_route_token_limit_reports_incomplete(content):
    cfg = _Cfg(("qwen3-vl:32b", 100, "local"))
    cfg.model_registry["vl"] = cfg.model_registry["general"]
    ollama = _ScriptedOllama({
        "qwen3-vl:32b": [{
            "message": {"content": content, "thinking": "hidden reasoning"},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 1177,
            "eval_count": 4096,
        }],
    })
    app = _route_app(cfg, ollama, _RecordingArchive())
    app.state.archive_client = None
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Read the document and image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ],
    }]
    payload = ChatCompletionRequest(
        model="audrey_fast", messages=messages, stream=True, max_tokens=4096,
    )
    response_request = ResponseCreateRequest(
        model="audrey_fast", input="Read the document and image.",
        stream=True, max_output_tokens=4096,
    )
    run_events: list[RunEvent] = []

    def session_factory(**kwargs):
        return ResponsesStreamSession(
            request=response_request, event_sink=run_events.append, **kwargs,
        )

    frames = [
        frame async for frame in _stream_via_pipeline(
            app, payload, messages, {"num_predict": 4096},
            user_id="alice@example.com", conversation_id="", user_turn_text="Read the image.",
            stream_session_factory=session_factory,
        )
    ]
    wire_events = [
        json.loads(line[6:])
        for frame in frames for line in frame.splitlines() if line.startswith("data: ")
    ]
    terminal = wire_events[-1]

    assert terminal["type"] == "response.incomplete"
    assert not any(event["type"] == "response.completed" for event in wire_events)
    response = terminal["response"]
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["usage"]["input_tokens"] == 1177
    assert response["usage"]["output_tokens"] == 4096
    assert "".join(event.delta for event in run_events if event.type == "text.delta") == content
    assert ollama.calls[0]["think"] is None
    assert ollama.calls[0]["options"] == {"num_predict": 4096}
    if not content:
        assert "✅" not in response["output_text"]


async def test_native_preference_context_reaches_model_but_not_auto_routing(
    monkeypatch,
):
    cfg = _Cfg(("a", 100, "local"))
    ollama = _ScriptedOllama({
        "a": [{
            "message": {"content": "short answer"},
            "done": True,
        }],
    })
    routing_messages = [{"role": "user", "content": "A short request."}]
    model_messages = [
        {
            "role": "system",
            "name": "audrey_user_preferences",
            "content": "Very long persona guidance. " * 600,
        },
        *routing_messages,
    ]
    gate_calls: list[list[dict[str, Any]]] = []
    classify_calls: list[list[dict[str, Any]]] = []

    def capture_complexity(messages, *, threshold):
        assert threshold == 500
        gate_calls.append(messages)
        return False, 4

    async def classify(*args, **kwargs):
        classify_calls.append(kwargs["messages"])
        return "general", "test", 1.0

    monkeypatch.setattr(route_pipeline, "is_complex", capture_complexity)
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    app = _route_app(cfg, ollama, _RecordingArchive())
    payload = ChatCompletionRequest(
        model="audrey_auto",
        messages=model_messages,
        stream=True,
    )

    frames = [
        frame async for frame in _stream_via_pipeline(
            app,
            payload,
            model_messages,
            {},
            user_id="alice@example.com",
            conversation_id="conversation-preferences",
            user_turn_text="A short request.",
            routing_messages=routing_messages,
        )
    ]

    assert gate_calls == [routing_messages]
    assert classify_calls == [routing_messages]
    assert any(
        message.get("name") == "audrey_user_preferences"
        for message in ollama.calls[0]["messages"]
    )
    chunks = [
        json.loads(frame[6:].strip())
        for frame in frames
        if frame.startswith("data: ") and frame.strip() != "data: [DONE]"
    ]
    answer = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
    )
    assert "short answer" in answer
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize("cooling", [False, True])
async def test_fast_tool_route_emits_native_observations_without_leaking_to_v1(
    monkeypatch, cooling,
):
    cfg = _Cfg(("a", 100, "local"))
    cfg.raw["fast_path"]["tool_capable_models"] = ["a"]
    archive = _RecordingArchive()
    app = _route_app(cfg, _ScriptedOllama({}), archive)
    if cooling:
        app.state.health.record_failure("a", "Ollama was temporarily offline")
    app.state.tools = ToolRegistry(by_name={
        "web_search": ToolSpec(
            name="web_search",
            description="search",
            parameters={"type": "object", "properties": {}},
            server_url="http://unused",
            path="/web_search",
        ),
    })
    graph = _ObservedGraph()
    app.state.graph = graph

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "search coffee history"}]
    payload = ChatCompletionRequest(
        model="audrey_fast",
        messages=messages,
        stream=True,
    )
    events: list[RunEvent] = []

    frames = [
        frame async for frame in _stream_via_pipeline(
            app,
            payload,
            messages,
            {},
            user_id="alice@example.com",
            conversation_id="conversation-fast-tool",
            user_turn_text="search coffee history",
            event_context=RunEventContext(
                run_id="run-fast-tool",
                conversation_id="conversation-fast-tool",
                assistant_message_id="message-fast-tool",
                mode="fast",
                sink=events.append,
            ),
        )
    ]

    assert graph.observer_seen is True
    observed = [
        event for event in events
        if event.type.startswith("tool.") or event.type == "source.observed"
    ]
    assert [event.type for event in observed] == [
        "tool.started",
        "tool.arguments",
        "tool.finished",
        "source.observed",
    ]
    assert observed[1].arguments["api_key"] == "[redacted]"
    assert observed[-1].url == "https://example.com/coffee"
    rendered = "".join(frames)
    assert "fast observed answer" in rendered
    assert "private-argument" not in rendered
    assert "private-result-body" not in rendered
    assert "token=private" not in rendered
    assert frames[-1] == "data: [DONE]\n\n"


async def test_compatibility_tool_route_does_not_create_native_observer(monkeypatch):
    cfg = _Cfg(("a", 100, "local"))
    cfg.raw["fast_path"]["tool_capable_models"] = ["a"]
    app = _route_app(cfg, _ScriptedOllama({}), _RecordingArchive())
    app.state.tools = ToolRegistry(by_name={
        "web_search": ToolSpec(
            name="web_search",
            description="search",
            parameters={"type": "object", "properties": {}},
            server_url="http://unused",
            path="/web_search",
        ),
    })
    graph = _ObservedGraph()
    app.state.graph = graph

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "search coffee history"}]
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
            conversation_id="conversation-fast-tool",
            user_turn_text="search coffee history",
        )
    ]

    assert graph.observer_seen is False
    rendered = "".join(frames)
    assert "fast observed answer" in rendered
    assert frames[-1] == "data: [DONE]\n\n"


async def test_cancelled_tool_stream_joins_graph_cleanup_before_returning(monkeypatch):
    cfg = _Cfg(("a", 100, "local"))
    cfg.raw["fast_path"]["tool_capable_models"] = ["a"]
    app = _route_app(cfg, _ScriptedOllama({}), _RecordingArchive())
    app.state.tools = _context_tools()
    graph_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    graph_tasks = []

    class Graph:
        async def ainvoke(self, state):
            graph_tasks.append(asyncio.current_task())
            graph_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()

    app.state.graph = Graph()

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "hello"}]
    stream = _stream_via_pipeline(
        app, ChatCompletionRequest(model="audrey_fast", messages=messages, stream=True),
        messages, {}, user_id="owner@example.com", conversation_id="", user_turn_text="hello",
    )

    async def consume():
        return [frame async for frame in stream]

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(graph_started.wait(), timeout=1)
        consumer.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        assert not consumer.done()
        assert not cleanup_finished.is_set()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=1)
        assert cleanup_finished.is_set()
        assert graph_tasks[0].cancelled()
    finally:
        release_cleanup.set()
        if not consumer.done():
            consumer.cancel()
        for task in graph_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(consumer, *graph_tasks, return_exceptions=True)
        await stream.aclose()


@pytest.mark.parametrize("provider_error", [OllamaError("provider failed"), RuntimeError("provider failed")])
async def test_tool_stream_cleanup_preserves_provider_failure(monkeypatch, provider_error):
    cfg = _Cfg(("a", 100, "local"))
    cfg.raw["fast_path"]["tool_capable_models"] = ["a"]
    app = _route_app(cfg, _ScriptedOllama({}), _RecordingArchive())
    app.state.tools = _context_tools()

    class Graph:
        async def ainvoke(self, state):
            raise provider_error

    app.state.graph = Graph()

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "hello"}]
    stream = _stream_via_pipeline(
        app, ChatCompletionRequest(model="audrey_fast", messages=messages, stream=True),
        messages, {}, user_id="owner@example.com", conversation_id="", user_turn_text="hello",
    )
    if isinstance(provider_error, OllamaError):
        frames = [frame async for frame in stream]
        assert "[ollama error: provider failed]" in "".join(frames)
        assert frames[-1] == "data: [DONE]\n\n"
    else:
        with pytest.raises(RuntimeError, match="provider failed"):
            _ = [frame async for frame in stream]


async def test_nonstream_task_header_is_ordinary_archived_content():
    archive = _RecordingArchive()
    cfg = _Cfg(("a", 100, "local"))
    app = SimpleNamespace(state=SimpleNamespace(
        graph=_FixedGraph(),
        inflight=UserInflightRegistry(max_inflight_per_user=2),
        archive_client=archive,
        tools=_NoTools(),
        cfg=cfg,
    ))
    messages = [{"role": "user", "content": "### Task:\nGenerate a title"}]
    payload = ChatCompletionRequest(
        model="audrey_fast",
        messages=messages,
        stream=False,
    )

    await _generate_via_pipeline(
        app,
        payload,
        messages,
        {},
        user_id="alice@example.com",
        conversation_id="conversation-1",
        user_turn_text="### Task:\nGenerate a title",
    )

    assert len(archive.calls) == 1


@pytest.mark.parametrize("native", [False, True])
async def test_stream_task_header_obeys_the_requested_deep_mode(monkeypatch, native):
    cfg = _Cfg(("a", 100, "local"))
    ollama = _ScriptedOllama({})
    user_text = "### Task:\nWrite a detailed migration plan"

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    deep_calls: list[dict[str, Any]] = []

    async def stream_deep(*args, **kwargs):
        deep_calls.append(kwargs)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    monkeypatch.setattr(
        route_pipeline,
        "_stream_deep_with_banners",
        stream_deep,
    )
    app = _route_app(cfg, ollama, _RecordingArchive())
    messages = [{"role": "user", "content": user_text}]
    payload = ChatCompletionRequest(
        model="audrey_deep",
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
            conversation_id="conversation-native-task",
            user_turn_text=user_text,
            event_context=RunEventContext(
                run_id="run-native-task",
                conversation_id="conversation-native-task",
                assistant_message_id="message-native-task",
                mode="deep",
                sink=lambda _event: None,
            ) if native else None,
        )
    ]

    assert frames == ["data: [DONE]\n\n"]


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


@pytest.mark.parametrize(
    ("mode", "done_reason", "expected_finish"),
    [("fast", "length", "length"), ("fast", "stop", "stop"), ("deep", "length", "stop")],
)
async def test_nonstream_preserves_final_fast_truncation_without_tainting_deep(
    mode, done_reason, expected_finish,
):
    class FinalGraph(_FixedGraph):
        async def ainvoke(self, state):
            final = await super().ainvoke(state)
            final.update(mode=mode, fast_done_reason=done_reason)
            return final

    archive = _RecordingArchive()
    app = SimpleNamespace(state=SimpleNamespace(
        graph=FinalGraph(),
        inflight=UserInflightRegistry(max_inflight_per_user=2),
        archive_client=archive, tools=_NoTools(), cfg=_Cfg(("a", 100, "local")),
    ))
    messages = [{"role": "user", "content": "Explain this."}]
    payload = ChatCompletionRequest(model="audrey_fast", messages=messages)

    result = await _generate_via_pipeline(
        app, payload, messages, {}, user_id="alice@example.com",
        conversation_id="conversation-1", user_turn_text="Explain this.",
    )

    assert result["choices"][0]["finish_reason"] == expected_finish
    assert result["choices"][0]["message"]["content"] == "utility result"
    assert archive.calls[0]["partial"] is (expected_finish == "length")


def _answer_from_frames(frames: list[str]) -> str:
    return "".join(
        json.loads(frame[6:])["choices"][0]["delta"].get("content", "")
        for frame in frames
        if frame.startswith("data: ") and frame.strip() != "data: [DONE]"
    )


def _context_tools() -> ToolRegistry:
    return ToolRegistry(by_name={
        name: ToolSpec(
            name=name,
            description=name,
            parameters={"type": "object", "properties": {}},
            server_url="http://unused",
            path=f"/{name}",
        )
        for name in ("memory_search", "memory_store", "chat_history_search")
    })


def _recall_result(*, failed: bool = False) -> ToolResult:
    return ToolResult(
        name="memory_search", call_id=None,
        content=(
            "search timed out"
            if failed
            else json.dumps({"results": [{"key": "color", "value": "blue"}]})
        ),
        elapsed_s=0.01, is_error=failed,
    )


@pytest.mark.parametrize("kind", ["text", "image", "structured"])
@pytest.mark.parametrize("native", [False, True])
async def test_plain_fast_adds_owner_context_without_tool_guidance(
    monkeypatch, kind, native,
):
    cfg = _Cfg(("a", 100, "local"))
    cfg.model_registry["vl"] = cfg.model_registry["general"]
    cfg.raw["agentic"] = {"memory": {"top_k": 2, "timeout_s": 0.75}}
    if kind == "structured":
        # Format alone must disable ReAct even with a capable model and tools.
        cfg.raw["fast_path"]["tool_capable_models"] = ["a"]
    ollama = _ScriptedOllama({
        "a": [{"message": {"content": "context answer"}, "done": True}],
    })
    app = _route_app(cfg, ollama, _RecordingArchive())
    app.state.tools = _context_tools()
    content: Any = "What is my favorite color?"
    if kind == "image":
        content = [
            {"type": "text", "text": content},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]
    routing_messages = [{"role": "user", "content": content}]
    original_messages = [
        {"role": "system", "content": "Original caller or skill instruction."},
        *routing_messages,
    ]
    if native:
        original_messages.insert(0, {
            "role": "system", "name": "audrey_user_preferences",
            "content": "User-local date and time: 2026-10-10T10:00:00-06:00",
        })
    original_copy = json.loads(json.dumps(original_messages))
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    payload = ChatCompletionRequest(
        model="audrey_auto", messages=original_messages, stream=True,
        user="someone-else@example.com",
    )
    recalls: list[dict[str, Any]] = []
    decisions: list[list[dict[str, Any]]] = []

    async def dispatch(http, registry, call, **kwargs):
        assert http is app.state.archive_http
        assert registry is app.state.tools
        recalls.append({**call["function"]["arguments"], **kwargs})
        return _recall_result()

    async def classify(*args, **kwargs):
        decisions.append(kwargs["messages"])
        return "general", "test", 1.0

    def complexity(messages, *, threshold):
        decisions.append(messages)
        return False, 6

    monkeypatch.setattr(memory_module, "dispatch_one", dispatch)
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    monkeypatch.setattr(route_pipeline, "is_complex", complexity)
    frames = [frame async for frame in _stream_via_pipeline(
        app, payload, original_messages, {},
        user_id="owner@example.com", conversation_id="", user_turn_text="color",
        routing_messages=routing_messages,
        # A restricted skill can hide memory tools without disabling the
        # server's owner recall. Structured output uses the full offered set.
        model_tools=app.state.tools if kind == "structured" else ToolRegistry(),
        response_format=schema if kind == "structured" else None,
        event_context=(
            RunEventContext(
                run_id="run-context", conversation_id="conversation-context",
                assistant_message_id="message-context", mode="auto", sink=lambda event: None,
            ) if native else None
        ),
    )]

    assert original_messages == original_copy
    assert len(recalls) == 1
    assert recalls[0]["user"] == "owner@example.com"
    assert recalls[0]["query"] == "What is my favorite color?"
    assert recalls[0]["top_k"] == 2
    assert recalls[0]["timeout_s"] == 0.75
    assert all(messages == routing_messages for messages in decisions)
    sent = ollama.calls[0]["messages"]
    assert sent[-len(original_messages):] == original_messages
    assert sent[-1] is original_messages[-1]
    assert sent[-1]["content"] is content
    assert sum(
        str(message.get("content", "")).startswith("Current server date and time:")
        for message in sent
    ) == 1
    injected = "\n".join(message["content"] for message in sent[:-len(original_messages)])
    assert "(color) blue" in injected
    assert "memory_store" not in injected
    assert "chat_history_search" not in injected
    assert ollama.calls[0]["tools"] is None
    assert ollama.calls[0]["format"] == (schema if kind == "structured" else None)
    assert "context answer" in _answer_from_frames(frames)
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize("skip", ["disabled", "no_owner", "no_search_tool"])
async def test_plain_fast_keeps_date_when_recall_is_unavailable(monkeypatch, skip):
    cfg = _Cfg(("a", 100, "local"))
    cfg.raw["agentic"] = {"memory": {"enabled": skip != "disabled"}}
    ollama = _ScriptedOllama({
        "a": [{"message": {"content": "answer without recall"}, "done": True}],
    })
    app = _route_app(cfg, ollama, _RecordingArchive())
    app.state.tools = _context_tools() if skip != "no_search_tool" else ToolRegistry()

    async def dispatch(*args, **kwargs):
        pytest.fail("Recall must not dispatch when disabled, unidentified, or unavailable")

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(memory_module, "dispatch_one", dispatch)
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "hello"}]
    frames = [frame async for frame in _stream_via_pipeline(
        app, ChatCompletionRequest(model="audrey_fast", messages=messages, stream=True),
        messages, {}, user_id="" if skip == "no_owner" else "owner@example.com",
        conversation_id="", user_turn_text="hello",
    )]

    sent = ollama.calls[0]["messages"]
    assert len(sent) == len(messages) + 1
    assert sent[0]["content"].startswith("Current server date and time:")
    assert "answer without recall" in _answer_from_frames(frames)


async def test_plain_fast_ack_precedes_recall_and_failed_recall_still_answers(monkeypatch):
    cfg = _Cfg(("a", 100, "local"))
    cfg.raw["agentic"] = {"memory": {"timeout_s": 0.125}}
    ollama = _ScriptedOllama({
        "a": [{"message": {"content": "answer after recall failure"}, "done": True}],
    })
    app = _route_app(cfg, ollama, _RecordingArchive())
    app.state.tools = _context_tools()
    recall_started = asyncio.Event()
    release = asyncio.Event()

    async def dispatch(*args, **kwargs):
        assert kwargs["timeout_s"] == 0.125
        recall_started.set()
        await release.wait()
        return _recall_result(failed=True)

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(memory_module, "dispatch_one", dispatch)
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "hello"}]
    stream = _stream_via_pipeline(
        app, ChatCompletionRequest(model="audrey_fast", messages=messages, stream=True),
        messages, {}, user_id="owner@example.com", conversation_id="", user_turn_text="hello",
    )
    frames = [await anext(stream), await anext(stream)]
    assert '"role": "assistant"' in frames[0]
    assert "Thinking" in frames[1]
    assert not recall_started.is_set()
    pending = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(recall_started.wait(), timeout=1)
        assert ollama.calls == []
        release.set()
        frames.append(await asyncio.wait_for(pending, timeout=1))
        frames.extend([frame async for frame in stream])
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()

    assert "answer after recall failure" in _answer_from_frames(frames)
    assert len(ollama.calls[0]["messages"]) == len(messages) + 1
    assert frames[-1] == "data: [DONE]\n\n"


async def test_plain_fast_fallback_reuses_context_without_recalling_again(monkeypatch):
    cfg = _Cfg(("a", 100, "local"), ("b", 50, "local"))
    ollama = _ScriptedOllama({
        "a": [OllamaError("first model unavailable")],
        "b": [{"message": {"content": "fallback answer"}, "done": True}],
    })
    app = _route_app(cfg, ollama, _RecordingArchive())
    app.state.tools = _context_tools()
    recalls = []

    async def dispatch(*args, **kwargs):
        recalls.append(args)
        return _recall_result()

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(memory_module, "dispatch_one", dispatch)
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "hello"}]
    frames = [frame async for frame in _stream_via_pipeline(
        app, ChatCompletionRequest(model="audrey_fast", messages=messages, stream=True),
        messages, {}, user_id="owner@example.com", conversation_id="", user_turn_text="hello",
    )]

    assert len(recalls) == 1
    assert [call["model"] for call in ollama.calls] == ["a", "b"]
    assert ollama.calls[0]["messages"] is ollama.calls[1]["messages"]
    assert len(ollama.calls[1]["messages"]) == len(messages) + 2
    assert "fallback answer" in _answer_from_frames(frames)


async def test_tool_graph_branch_does_not_prepare_stream_context_twice(monkeypatch):
    cfg = _Cfg(("a", 100, "local"))
    cfg.raw["fast_path"]["tool_capable_models"] = ["a"]
    app = _route_app(cfg, _ScriptedOllama({}), _RecordingArchive())
    app.state.tools = _context_tools()
    graph_states = []

    class Graph(_FixedGraph):
        async def ainvoke(self, state):
            graph_states.append(state)
            return await super().ainvoke(state)

    app.state.graph = Graph()

    async def forbidden_context(**kwargs):
        pytest.fail("Graph date/recall nodes own context on the tool-capable branch")

    async def classify(*args, **kwargs):
        return "general", "test", 1.0

    monkeypatch.setattr(route_pipeline, "_prepare_stream_context", forbidden_context)
    monkeypatch.setattr(route_pipeline, "classify_with_registry", classify)
    messages = [{"role": "user", "content": "hello"}]
    frames = [frame async for frame in _stream_via_pipeline(
        app, ChatCompletionRequest(model="audrey_fast", messages=messages, stream=True),
        messages, {}, user_id="owner@example.com", conversation_id="", user_turn_text="hello",
    )]

    assert len(graph_states) == 1
    assert graph_states[0]["messages"] is messages
    assert "utility result" in "".join(frames)


async def test_deep_thinking_reuses_context_and_keeps_offered_tool_guidance(monkeypatch):
    registry = _context_tools()
    recalls = []

    async def dispatch(*args, **kwargs):
        recalls.append(args)
        return _recall_result()

    monkeypatch.setattr(memory_module, "dispatch_one", dispatch)
    messages = [{"role": "user", "content": "hello"}]
    sent, subtasks = await route_pipeline._phase_thinking(
        ollama=None, tools=registry, http=object(), model_tools=registry,
        user_id="owner@example.com", messages=messages,
        memory_enabled=True, memory_top_k=3, memory_timeout_s=5,
        planning_enabled=False, planning_min_tokens=40, planning_max_subtasks=3,
        prompt_tokens=1, router_cfg={},
    )

    assert len(recalls) == 1
    assert sent[0]["content"].startswith("Current server date and time:")
    assert sent[-1] is messages[-1]
    injected = "\n".join(message["content"] for message in sent[:-1])
    assert "(color) blue" in injected
    assert "memory_store" in injected
    assert "chat_history_search" in injected
    assert subtasks == []
