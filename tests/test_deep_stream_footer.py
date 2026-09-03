"""The deep stream's two reports on what its workers did with tools: the
user-facing `Tools used:` footer, and the operator-facing summary log line.

Phase 42's eval artifacts showed every deep-routed turn arriving without a
footer while every fast turn had one, and that got written down as "the deep
footer does not render". This pins the actual behaviour so the next reader
does not have to re-derive it from answer files: given workers that called
tools, the deep stream DOES emit the footer, in the right place. A deep turn
with no footer therefore means its workers called no tools — which is a
question about the panel, not about `_stream_deep_with_banners`.

Answering that question then took four rounds of greps against the box,
because the streaming path logged nothing about the panel at all — the
`deep_panel: … tool_grounded=N` line lives in `graph.py`, which serves
non-streaming requests, and OWUI streams questions while NOT streaming title
generation. So every grep found only utility turns. The summary line now
carries the same three fields on both paths.

Structured like `test_research_stream.py`: fakes on `app.state` so no real
model runs.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import audrey.pipeline.graph
from audrey.config import Config, EnvOverrides, get_config
from audrey.metrics import pipeline_total
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.run_events import RunEvent, RunEventContext, dump_run_event
from audrey.routes.openai import pipeline as route_pipeline
from audrey.routes.openai.pipeline import _stream_deep_with_banners
from audrey.routes.openai.schemas import ChatCompletionRequest
from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.dispatch import ToolResult

# Read at import: the cross-path wording check runs inside an async test, and
# blocking file IO there trips ASYNC240.
_GRAPH_SRC = Path(inspect.getfile(audrey.pipeline.graph)).read_text()


class _FakeOllama:
    """Canned content per model. Workers run one-shot chat unless the model is
    tool-capable; the synthesizer streams."""

    def __init__(self, responses: dict[str, str], *, tool_call_models: set[str] | None = None):
        self.responses = responses
        self.tool_call_models = tool_call_models or set()
        self._called: set[str] = set()

    async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None, think=None):
        # First round for a tool-calling model returns a tool call; the second
        # returns prose, which is what ends the ReAct loop.
        if model in self.tool_call_models and model not in self._called:
            self._called.add(model)
            return {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "kb_search", "arguments": {"query": "x"}}}],
                },
                "prompt_eval_count": 1, "eval_count": 1,
            }
        return {"message": {"content": self.responses.get(model, "")},
                "prompt_eval_count": 1, "eval_count": 1}

    async def chat_stream(self, *, model, messages, options=None, timeout_s=0):
        text = self.responses.get(model, "")
        yield {"message": {"content": text}, "done": True,
               "prompt_eval_count": 1, "eval_count": 1}

    async def aclose(self):
        pass


class _NoTools:
    by_name: ClassVar[dict[str, Any]] = {}


def _kb_registry() -> ToolRegistry:
    spec = ToolSpec(name="kb_search", description="search the kb",
                    parameters={"type": "object", "properties": {}},
                    server_url="http://unused.invalid", path="/kb_search")
    return ToolRegistry(by_name={"kb_search": spec})


def _fake_app(responses: dict[str, str], *, tool_capable: list[str] | None = None):
    base = get_config()
    cfg = Config(copy.deepcopy(base.raw), EnvOverrides())
    cfg.raw["deep_panel"] = {
        "reasoning": {"workers": ["w1", "w2"], "synthesizer": "s", "fallback_synth": "fb"},
    }
    cfg.raw.setdefault("model_registry", {})["reasoning"] = [
        {"name": "w1", "priority": 100, "location": "cloud"},
        {"name": "w2", "priority": 90, "location": "cloud"},
        {"name": "s", "priority": 80, "location": "cloud"},
        {"name": "fb", "priority": 70, "location": "cloud"},
    ]
    cfg.raw.setdefault("fast_path", {})["tool_capable_models"] = list(tool_capable or [])
    # Planner off: it would need its own canned model, and the footer is
    # downstream of planning either way.
    cfg.raw.setdefault("agentic", {})["planning"] = {"enabled": False}
    cfg.raw["agentic"]["memory"] = {"enabled": False}
    state = SimpleNamespace(
        cfg=cfg,
        ollama=_FakeOllama(responses, tool_call_models=set(tool_capable or [])),
        registry=ModelRegistry(cfg),
        health=HealthTracker(),
        gate=FairLocalGate(concurrency=1),
        tools=_kb_registry() if tool_capable else _NoTools(),
        archive_http=object(),
    )
    return SimpleNamespace(state=state)


def _joined_content(frames: list[str]) -> str:
    out: list[str] = []
    for f in frames:
        if not f.startswith("data: ") or f.strip() == "data: [DONE]":
            continue
        delta = json.loads(f[len("data: "):])["choices"][0].get("delta", {})
        if delta.get("content"):
            out.append(delta["content"])
    return "".join(out)


async def _collect(app, events: list[RunEvent] | None = None):
    msgs = [{"role": "user", "content": "what do my videos say about the london system"}]
    payload = ChatCompletionRequest(model="audrey_deep", messages=msgs, stream=True)
    return [
        frame async for frame in _stream_deep_with_banners(
            app, payload, msgs, {}, task="reasoning", conf=0.9,
            user_id="", conversation_id="", user_turn_text=msgs[0]["content"],
            event_context=(
                RunEventContext(
                    run_id="run-deep",
                    conversation_id="conversation-deep",
                    assistant_message_id="message-deep",
                    mode="deep",
                    sink=events.append,
                )
                if events is not None
                else None
            ),
        )
    ]


async def test_deep_stream_emits_typed_lifecycle_and_answer_only_deltas():
    events: list[RunEvent] = []
    app = _fake_app({"w1": "draft one", "w2": "draft two", "s": "Deep answer."})

    await _collect(app, events)

    assert [event.stage for event in events if event.type == "stage.started"] == [
        "planning",
        "dispatching",
        "synthesizing",
    ]
    assert [event.stage for event in events if event.type == "stage.finished"] == [
        "planning",
        "dispatching",
        "synthesizing",
    ]
    assert "".join(event.delta for event in events if event.type == "text.delta") == (
        "Deep answer."
    )
    usage = next(event for event in events if event.type == "usage.reported")
    assert (usage.prompt_tokens, usage.completion_tokens) == (1, 1)
    assert events[-2].type == "message.finished"
    assert events[-1].type == "run.finished"
    assert events[-1].status == "succeeded"


async def test_deep_stream_projects_real_tool_activity_and_deduped_sources(
    monkeypatch,
):
    result_body = json.dumps({
        "results": [{
            "title": "London System",
            "url": "https://example.com/london?token=private",
            "snippet": "private-result-body",
        }],
    })

    async def _search(http, registry, tc, *, max_result_chars, timeout_s, user_id=None):
        return ToolResult(
            name="kb_search",
            call_id=tc.get("id"),
            content=result_body,
            elapsed_s=0.01,
            is_error=False,
        )

    monkeypatch.setattr("audrey.pipeline.react.dispatch_one", _search)
    events: list[RunEvent] = []
    app = _fake_app(
        {"w1": "draft one", "w2": "draft two", "s": "Deep answer."},
        tool_capable=["w1", "w2"],
    )

    await _collect(app, events)

    observed = [
        event for event in events
        if event.type.startswith("tool.") or event.type == "source.observed"
    ]
    assert [event.type for event in observed].count("tool.started") == 2
    assert [event.type for event in observed].count("tool.arguments") == 2
    assert [event.type for event in observed].count("tool.finished") == 2
    sources = [event for event in observed if event.type == "source.observed"]
    assert len(sources) == 1
    assert sources[0].url == "https://example.com/london"
    assert all(
        event.status == "succeeded"
        for event in observed
        if event.type == "tool.finished"
    )
    wire = json.dumps([dump_run_event(event) for event in events])
    assert "private-result-body" not in wire
    assert "token=private" not in wire


async def test_deep_stream_renders_the_tools_used_footer():
    """The claim under test. Workers call `kb_search`, so the footer must show."""
    app = _fake_app({"w1": "draft one", "w2": "draft two", "s": "The London System is a setup."},
                    tool_capable=["w1", "w2"])
    joined = _joined_content(await _collect(app))

    assert "_Tools used:_" in joined, "deep path dropped the footer"
    assert "`kb_search`" in joined
    # Both workers get a row.
    assert "**w1**" in joined
    assert "**w2**" in joined


async def test_the_footer_lands_after_the_answer_body():
    app = _fake_app({"w1": "draft one", "w2": "draft two", "s": "The London System is a setup."},
                    tool_capable=["w1", "w2"])
    joined = _joined_content(await _collect(app))
    assert joined.index("The London System is a setup.") < joined.index("_Tools used:_")


async def test_no_footer_when_workers_called_no_tools():
    """The other half — and the likelier reading of a footerless deep turn in an
    eval artifact. Nothing here is broken; there was simply nothing to report."""
    app = _fake_app({"w1": "draft one", "w2": "draft two", "s": "An answer."})
    joined = _joined_content(await _collect(app))
    assert "_Tools used:_" not in joined
    assert "An answer." in joined


# ─── The operator-facing half ─────────────────────────────────────────


def _summary_line(caplog) -> str:
    lines = [r.message for r in caplog.records if r.message.startswith("stream deep done")]
    assert len(lines) == 1, f"expected one summary line, got {lines}"
    return lines[0]


async def test_the_summary_line_reports_grounding(caplog):
    caplog.set_level(logging.INFO)
    app = _fake_app({"w1": "draft one", "w2": "draft two", "s": "An answer."},
                    tool_capable=["w1", "w2"])
    await _collect(app)

    assert "workers=2 ok=2 tool_grounded=2" in _summary_line(caplog)


async def test_the_summary_line_reports_zero_grounding_rather_than_omitting_it(caplog):
    """⚠️ The field has to be present even at zero.

    A missing field and a zero are the same thing to a reader scrolling a log,
    and that ambiguity is what sent the last investigation down the wrong
    path: `deep_panel:` was genuinely absent from streamed turns, which looked
    like "no deep turns ran" rather than "this path never logged it".
    """
    caplog.set_level(logging.INFO)
    app = _fake_app({"w1": "draft one", "w2": "draft two", "s": "An answer."})
    await _collect(app)

    assert "workers=2 ok=2 tool_grounded=0" in _summary_line(caplog)


async def test_the_summary_line_matches_the_non_streaming_wording(caplog):
    """Both pipelines must be greppable with ONE pattern. `graph.py` writes
    `deep_panel: … workers=%d ok=%d tool_grounded=%d`; drift in either
    direction quietly re-splits the diagnostic surface that this change
    exists to merge."""
    caplog.set_level(logging.INFO)
    app = _fake_app({"w1": "draft one", "w2": "draft two", "s": "An answer."},
                    tool_capable=["w1", "w2"])
    await _collect(app)

    assert re.search(r"workers=\d+ ok=\d+ tool_grounded=\d+", _summary_line(caplog))

    assert "workers=%d ok=%d tool_grounded=%d" in _GRAPH_SRC

# ─── Request-owned task cancellation ──────────────────────────────────


class _RecordingArchive:
    def __init__(self):
        self.calls = []

    async def archive_turn(self, **kwargs):
        self.calls.append(kwargs)


def _deep_generator(app):
    msgs = [{"role": "user", "content": "cancel this request"}]
    payload = ChatCompletionRequest(model="audrey_deep", messages=msgs, stream=True)
    return _stream_deep_with_banners(
        app,
        payload,
        msgs,
        {},
        task="reasoning",
        conf=0.9,
        user_id="alice@example.com",
        conversation_id="conversation-1",
        user_turn_text=msgs[0]["content"],
    )


async def _drain_stream(stream):
    async for _frame in stream:
        pass


async def _assert_cancel_settles_child(stream, started, settled, child_tasks):
    consumer = asyncio.create_task(_drain_stream(stream))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("stream consumer did not propagate cancellation")
        await asyncio.wait_for(settled.wait(), timeout=1)
    finally:
        consumer.cancel()
        for task in child_tasks:
            task.cancel()
        await asyncio.gather(consumer, *child_tasks, return_exceptions=True)
        await stream.aclose()


async def test_deep_disconnect_during_planning_cancels_and_archives_partial(monkeypatch):
    app = _fake_app({})
    archive = _RecordingArchive()
    app.state.archive_client = archive
    started = asyncio.Event()
    settled = asyncio.Event()
    child_tasks = []

    async def blocked_planning(**kwargs):
        child_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    monkeypatch.setattr(route_pipeline, "_phase_thinking", blocked_planning)
    await _assert_cancel_settles_child(
        _deep_generator(app),
        started,
        settled,
        child_tasks,
    )
    assert len(archive.calls) == 1
    assert archive.calls[0]["partial"] is True


async def test_deep_disconnect_during_panel_cancels_panel_task(monkeypatch):
    app = _fake_app({})
    started = asyncio.Event()
    settled = asyncio.Event()
    child_tasks = []

    async def immediate_planning(**kwargs):
        return kwargs["messages"], []

    async def blocked_panel(**kwargs):
        child_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    monkeypatch.setattr(route_pipeline, "_phase_thinking", immediate_planning)
    monkeypatch.setattr(route_pipeline, "_phase_dispatch", blocked_panel)
    await _assert_cancel_settles_child(
        _deep_generator(app),
        started,
        settled,
        child_tasks,
    )


async def test_deep_disconnect_during_synthesis_cancels_producer(monkeypatch):
    app = _fake_app({})
    started = asyncio.Event()
    settled = asyncio.Event()
    child_tasks = []

    async def immediate_planning(**kwargs):
        return kwargs["messages"], []

    async def immediate_panel(**kwargs):
        return []

    async def blocked_synthesis(*args, **kwargs):
        child_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()
        yield {"type": "done"}

    monkeypatch.setattr(route_pipeline, "_phase_thinking", immediate_planning)
    monkeypatch.setattr(route_pipeline, "_phase_dispatch", immediate_panel)
    monkeypatch.setattr(route_pipeline, "synthesize_stream", blocked_synthesis)
    await _assert_cancel_settles_child(
        _deep_generator(app),
        started,
        settled,
        child_tasks,
    )


async def test_deep_queue_full_cleanup_does_not_wait_for_sentinel_space(monkeypatch):
    app = _fake_app({})
    put_attempted = asyncio.Event()
    producer_tasks = []

    async def immediate_planning(**kwargs):
        return kwargs["messages"], []

    async def immediate_panel(**kwargs):
        return []

    async def fill_event_queue(*args, **kwargs):
        producer_tasks.append(asyncio.current_task())
        for index in range(129):
            if index == 128:
                put_attempted.set()
            yield {"type": "fallback_attempt", "model": "s", "error": "retry"}

    monkeypatch.setattr(route_pipeline, "_phase_thinking", immediate_planning)
    monkeypatch.setattr(route_pipeline, "_phase_dispatch", immediate_panel)
    monkeypatch.setattr(route_pipeline, "synthesize_stream", fill_event_queue)

    stream = _deep_generator(app)
    try:
        while True:
            frame = await asyncio.wait_for(anext(stream), timeout=1)
            if "_Synthesizing_" in frame:
                break
        await asyncio.wait_for(put_attempted.wait(), timeout=1)
        await asyncio.sleep(0)
        assert producer_tasks and not producer_tasks[0].done()
        await asyncio.wait_for(stream.aclose(), timeout=1)
        assert producer_tasks[0].done()
    finally:
        for task in producer_tasks:
            task.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)


def _deep_outcome_counts() -> dict[str, float]:
    return {
        outcome: pipeline_total.labels(
            mode="deep",
            task_type="reasoning",
            outcome=outcome,
        )._value.get()
        for outcome in ("ok", "error", "cancelled", "truncated")
    }


async def test_deep_missing_done_has_one_identity_and_truncated_outcome(monkeypatch):
    app = _fake_app({})
    archive = _RecordingArchive()
    app.state.archive_client = archive

    async def immediate_planning(**kwargs):
        return kwargs["messages"], []

    async def immediate_panel(**kwargs):
        return []

    async def truncated_synthesis(*args, **kwargs):
        yield {"type": "first_token", "model": "s"}
        yield {"type": "delta", "text": "partial answer"}

    monkeypatch.setattr(route_pipeline, "_phase_thinking", immediate_planning)
    monkeypatch.setattr(route_pipeline, "_phase_dispatch", immediate_panel)
    monkeypatch.setattr(route_pipeline, "synthesize_stream", truncated_synthesis)

    before = _deep_outcome_counts()
    frames = [frame async for frame in _deep_generator(app)]
    after = _deep_outcome_counts()

    assert after["truncated"] == before["truncated"] + 1
    for outcome in ("ok", "error", "cancelled"):
        assert after[outcome] == before[outcome]

    payloads = [
        json.loads(frame.removeprefix("data: "))
        for frame in frames
        if frame.startswith("data: ") and frame.strip() != "data: [DONE]"
    ]
    assert len({item["id"] for item in payloads}) == 1
    assert sum(
        item["choices"][0]["delta"].get("role") == "assistant"
        for item in payloads
    ) == 1
    assert sum(
        item["choices"][0]["finish_reason"] == "stop"
        for item in payloads
    ) == 1
    assert frames.count("data: [DONE]\n\n") == 1

    assert len(archive.calls) == 1
    assert archive.calls[0]["assistant_content"] == "partial answer"
    assert archive.calls[0]["partial"] is True
