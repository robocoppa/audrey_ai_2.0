"""Integration test for the streaming `audrey_research` route (Phase 24).

`_stream_research_with_banners` is the orchestration glue: it drives the
staged pipeline (`run_research_pipeline_streaming`) and turns its stage events
into the Planning → Researching → Verifying → Writing banner stream plus the
live answer. The executor's event order is unit-tested in `test_deep_panel.py`;
this pins the *route* — that the phase banners appear in order, the separator
lands before the answer, and the writer's tokens stream through as content.

We stub `app.state` with fakes so no real model runs.
"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from typing import Any, ClassVar

from audrey.config import Config, EnvOverrides, get_config
from audrey.metrics import pipeline_total
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.run_events import RunEvent, RunEventContext
from audrey.routes.openai import pipeline as route_pipeline
from audrey.routes.openai.pipeline import _stream_research_with_banners
from audrey.routes.openai.schemas import ChatCompletionRequest


class _FakeOllama:
    """Returns canned content per model for chat (researchers/verifier) and
    chat_stream (writer). Mirrors the stub in test_deep_panel.py."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses

    async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None,
                   format=None, think=None):
        # ⚠️ Every kwarg the real `OllamaClient.chat` accepts must appear here,
        # even the ones this fake ignores. When `think=` was added to the panel's
        # call, this double still had the older, narrower signature: the
        # TypeError escaped `_run_one_worker` and the streaming test HUNG rather
        # than failed, which is far more expensive to diagnose than a red test.
        # `tools=` is likewise here so the fact-checker's run_react loop can call us.
        return {"message": {"content": self.responses.get(model, "")},
                "prompt_eval_count": 1, "eval_count": 1}

    async def chat_stream(self, *, model, messages, options=None, timeout_s=0):
        text = self.responses.get(model, "")
        mid = len(text) // 2
        yield {"message": {"content": text[:mid]}, "done": False}
        yield {"message": {"content": text[mid:]}, "done": True,
               "prompt_eval_count": 1, "eval_count": 1}

    async def aclose(self):
        pass


class _FakeTools:
    """Empty tool registry — `_phase_thinking` reads `.by_name`; researchers
    run tool-free (not in tool_capable_models)."""
    by_name: ClassVar[dict[str, Any]] = {}


def _one_tool_registry():
    """A real ToolRegistry with one web_search tool, so the fact-check stage's
    run_react has something to offer (and `_phase_thinking` sees it non-empty)."""
    from audrey.tools.discovery import ToolRegistry, ToolSpec
    spec = ToolSpec(name="web_search", description="search",
                    parameters={"type": "object", "properties": {}},
                    server_url="http://unused", path="/web_search")
    return ToolRegistry(by_name={"web_search": spec})


def _fake_app(responses: dict[str, str], *, factchecker: str | None = None):
    # Build a FRESH Config from a deep-copied raw so mutating the research
    # pool can't leak into the shared `get_config()` singleton other tests use.
    base = get_config()
    cfg = Config(copy.deepcopy(base.raw), EnvOverrides())
    # Point the research pool at the fake models so the route resolves them.
    body = {
        "researchers": ["r1", "r2"],
        "verifier": "v",
        "writer": "w",
        "fallback_synth": "fb",
    }
    reg_models = [
        {"name": "r1", "priority": 100, "location": "cloud"},
        {"name": "r2", "priority": 90, "location": "cloud"},
        {"name": "v", "priority": 80, "location": "cloud"},
        {"name": "w", "priority": 70, "location": "local"},
        {"name": "fb", "priority": 60, "location": "cloud"},
    ]
    if factchecker:
        body["factchecker"] = factchecker
        reg_models.append({"name": factchecker, "priority": 75, "location": "cloud"})
        # The fact-checker must be tool-capable for the stage to run.
        cfg.raw.setdefault("fast_path", {})["tool_capable_models"] = [factchecker]
    cfg.raw["deep_panel_research"] = {"reasoning": body}
    cfg.raw.setdefault("model_registry", {})["reasoning"] = reg_models
    registry = ModelRegistry(cfg)
    state = SimpleNamespace(
        cfg=cfg,
        ollama=_FakeOllama(responses),
        registry=registry,
        health=HealthTracker(),
        gate=FairLocalGate(concurrency=1),
        tools=_one_tool_registry() if factchecker else _FakeTools(),
        archive_http=object(),
        # archive_client intentionally absent → getattr default None.
    )
    return SimpleNamespace(state=state)


def _content_frames(frames: list[str]) -> list[str]:
    """Extract the delta content strings from raw SSE frames."""
    out: list[str] = []
    for f in frames:
        if not f.startswith("data: ") or f.strip() == "data: [DONE]":
            continue
        payload = json.loads(f[len("data: "):])
        delta = payload["choices"][0].get("delta", {})
        if delta.get("content"):
            out.append(delta["content"])
    return out


async def _collect(
    app,
    model="audrey_research",
    events: list[RunEvent] | None = None,
):
    # The route receives `messages` as plain dicts (the caller converts the
    # pydantic request into dicts before dispatching), so pass dicts here.
    msgs = [{"role": "user", "content": "tell me about euclid"}]
    payload = ChatCompletionRequest(model=model, messages=msgs, stream=True)
    frames = [
        frame async for frame in _stream_research_with_banners(
            app, payload, msgs, {}, task="reasoning", conf=0.9,
            user_id="", conversation_id="", user_turn_text="tell me about euclid",
            event_context=(
                RunEventContext(
                    run_id="run-research",
                    conversation_id="conversation-research",
                    assistant_message_id="message-research",
                    mode="research",
                    sink=events.append,
                )
                if events is not None
                else None
            ),
        )
    ]
    return frames


async def test_research_stream_banner_order_and_answer():
    app = _fake_app({"r1": "fact A", "r2": "fact B", "v": "looks fine",
                     "w": "Euclid was a Greek mathematician."})
    frames = await _collect(app)
    content = _content_frames(frames)
    joined = "".join(content)

    # All four phase banners appear, in order.
    for banner in ("_Planning_", "_Researching_", "_Verifying_", "_Writing_"):
        assert banner in joined, f"missing banner {banner}"
    assert (joined.index("_Researching_") < joined.index("_Verifying_")
            < joined.index("_Writing_"))

    # Separator precedes the answer body; the writer's text streamed through.
    assert "\n\n---\n\n" in joined
    answer_region = joined.split("\n\n---\n\n", 1)[1]
    assert "Euclid was a Greek mathematician." in answer_region

    # Terminates with stop + DONE.
    assert frames[-1] == "data: [DONE]\n\n"


async def test_research_stream_emits_typed_lifecycle_and_writer_usage():
    events: list[RunEvent] = []
    app = _fake_app({
        "r1": "fact A",
        "r2": "fact B",
        "v": "looks fine",
        "w": "Research answer.",
    })

    await _collect(app, events=events)

    assert [event.stage for event in events if event.type == "stage.started"] == [
        "planning",
        "researching",
        "verifying",
        "writing",
    ]
    assert [event.stage for event in events if event.type == "stage.finished"] == [
        "planning",
        "researching",
        "verifying",
        "writing",
    ]
    assert "".join(event.delta for event in events if event.type == "text.delta") == (
        "Research answer."
    )
    usage = next(event for event in events if event.type == "usage.reported")
    assert (usage.prompt_tokens, usage.completion_tokens) == (1, 1)
    assert events[-1].type == "run.finished"
    assert events[-1].status == "succeeded"


async def test_research_stream_factcheck_banner_in_order():
    # With a factchecker configured + tool-capable, the Fact-checking banner
    # appears between Verifying and Writing, and the answer still streams.
    app = _fake_app(
        {"r1": "fact A", "r2": "fact B", "v": "looks fine",
         "fc": "CONFIRMED: fact A (source)", "w": "Euclid was a Greek mathematician."},
        factchecker="fc",
    )
    frames = await _collect(app)
    joined = "".join(_content_frames(frames))

    for banner in ("_Researching_", "_Verifying_", "_Fact-checking_", "_Writing_"):
        assert banner in joined, f"missing banner {banner}"
    assert (joined.index("_Verifying_") < joined.index("_Fact-checking_")
            < joined.index("_Writing_"))
    answer_region = joined.split("\n\n---\n\n", 1)[1]
    assert "Euclid was a Greek mathematician." in answer_region
    assert frames[-1] == "data: [DONE]\n\n"


async def test_research_stream_empty_research_skips_verify_banner():
    # No researchers healthy → no findings → verify skipped. Writer still runs
    # (flagged), so the answer still streams and the stream still terminates.
    app = _fake_app({"w": "Caveat: unverified. Euclid..."})
    app.state.health.record_failure("r1", "down")
    app.state.health.record_failure("r2", "down")
    frames = await _collect(app)
    joined = "".join(_content_frames(frames))

    assert "_Researching_" in joined
    assert "_Writing_" in joined
    # Answer still streamed despite zero grounding.
    answer_region = joined.split("\n\n---\n\n", 1)[1]
    assert "Caveat: unverified." in answer_region
    assert frames[-1] == "data: [DONE]\n\n"


# ─── Research trace block (opt-in via agentic.debug_research_trace) ────

_LEDGER_JSON = json.dumps({
    "summary_notes": "",
    "claims": [{"id": "c1", "text": "Euclid wrote the Elements.",
                "source_ids": ["s1"], "risk": "low", "needs_hedge": False}],
    "sources": [{"id": "s1", "title": "Euclid — Britannica",
                 "url": "https://britannica.com/euclid",
                 "source_type": "reference", "supports": ["c1"]}],
    "unresolved_questions": [],
})


class _StructuringFakeOllama(_FakeOllama):
    """Also answers the ledger-structuring calls: a `format=`-pinned chat
    returns canned ResearchResult JSON instead of the researcher prose, so
    the pipeline builds a real merged ledger from the fake stack."""

    async def thinking_flag(self, model, want):
        # The structuring passes ask this before every `format=` call. Absent it,
        # the AttributeError is caught by the gather's blanket handler and the
        # ONLY symptom is a missing ledger — which reads as a structuring bug.
        return want

    async def chat(self, *, model, messages, options=None, timeout_s=0,
                   tools=None, format=None, think=None):
        if format is not None:
            return {"message": {"content": _LEDGER_JSON},
                    "prompt_eval_count": 1, "eval_count": 1}
        return await super().chat(model=model, messages=messages,
                                  options=options, timeout_s=timeout_s,
                                  tools=tools, think=think)


async def test_research_stream_trace_block_when_flag_on():
    # Flag on → the staged-pipeline trace renders after the answer: researcher
    # notes, the merged ledger (2 workers × same source URL → 2 claims,
    # 1 deduped source), and the verifier critique.
    responses = {"r1": "fact A", "r2": "fact B", "v": "looks fine",
                 "w": "Euclid was a Greek mathematician."}
    app = _fake_app(responses)
    app.state.ollama = _StructuringFakeOllama(responses)
    app.state.cfg.raw.setdefault("agentic", {})["debug_research_trace"] = True
    frames = await _collect(app)
    joined = "".join(_content_frames(frames))

    answer_region = joined.split("\n\n---\n\n", 1)[1]
    assert "## Research trace (debug)" in answer_region
    assert "### Researcher notes" in answer_region
    assert "#### r1" in answer_region and "#### r2" in answer_region
    assert "fact A" in answer_region and "fact B" in answer_region
    assert "### Ledger — 2 claims, 1 sources" in answer_region
    assert "https://britannica.com/euclid" in answer_region
    assert "### Verifier critique" in answer_region
    # The trace lands after the answer prose, and the stream still terminates.
    assert (answer_region.index("Euclid was a Greek mathematician.")
            < answer_region.index("## Research trace (debug)"))
    assert frames[-1] == "data: [DONE]\n\n"


async def test_research_stream_no_trace_block_by_default():
    # Ships dark: the default config leaves the flag off and the trace absent.
    app = _fake_app({"r1": "fact A", "r2": "fact B", "v": "looks fine",
                     "w": "Euclid was a Greek mathematician."})
    frames = await _collect(app)
    joined = "".join(_content_frames(frames))

    assert "## Research trace (debug)" not in joined
    assert frames[-1] == "data: [DONE]\n\n"

# ─── Request-owned task cancellation ──────────────────────────────────


class _ResearchArchive:
    def __init__(self):
        self.calls = []

    async def archive_turn(self, **kwargs):
        self.calls.append(kwargs)


def _research_generator(app):
    msgs = [{"role": "user", "content": "cancel this research"}]
    payload = ChatCompletionRequest(model="audrey_research", messages=msgs, stream=True)
    return _stream_research_with_banners(
        app,
        payload,
        msgs,
        {},
        task="reasoning",
        conf=0.9,
        user_id="alice",
        conversation_id="conversation-1",
        user_turn_text=msgs[0]["content"],
    )


async def _drain_research_stream(stream, expected_text, observed):
    async for frame in stream:
        if expected_text and expected_text in frame:
            observed.set()


async def _cancel_research_consumer(
    stream,
    started,
    settled,
    child_tasks,
    expected_text="",
):
    observed = asyncio.Event()
    consumer = asyncio.create_task(_drain_research_stream(stream, expected_text, observed))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        if expected_text:
            await asyncio.wait_for(observed.wait(), timeout=1)
        consumer.cancel()
        try:
            await asyncio.wait_for(consumer, timeout=1)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("research consumer did not propagate cancellation")
        await asyncio.wait_for(settled.wait(), timeout=1)
    finally:
        consumer.cancel()
        for task in child_tasks:
            task.cancel()
        await asyncio.gather(consumer, *child_tasks, return_exceptions=True)
        await stream.aclose()


async def test_research_disconnect_during_planning_cancels_planner(monkeypatch):
    app = _fake_app({})
    archive = _ResearchArchive()
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
    await _cancel_research_consumer(
        _research_generator(app),
        started,
        settled,
        child_tasks,
    )
    assert len(archive.calls) == 1
    assert archive.calls[0]["partial"] is True


async def test_research_disconnect_cancels_pipeline_at_every_later_stage(monkeypatch):
    stages = [
        ("research", []),
        ("verification", [{"type": "findings_ready", "grounded": True}]),
        (
            "factcheck",
            [
                {"type": "findings_ready", "grounded": True},
                {"type": "verify_done", "ok": True},
            ],
        ),
        (
            "writing",
            [
                {"type": "findings_ready", "grounded": True},
                {"type": "verify_done", "ok": True},
                {"type": "write_delta", "text": "partial answer"},
            ],
        ),
    ]

    async def immediate_planning(**kwargs):
        return kwargs["messages"], []

    monkeypatch.setattr(route_pipeline, "_phase_thinking", immediate_planning)

    for stage, prefix in stages:
        app = _fake_app({})
        archive = _ResearchArchive()
        app.state.archive_client = archive
        started = asyncio.Event()
        settled = asyncio.Event()
        child_tasks = []

        async def blocked_pipeline(
            *args,
            _prefix=prefix,
            _child_tasks=child_tasks,
            _started=started,
            _settled=settled,
            **kwargs,
        ):
            _child_tasks.append(asyncio.current_task())
            try:
                for event in _prefix:
                    yield event
                _started.set()
                await asyncio.Event().wait()
            finally:
                _settled.set()

        monkeypatch.setattr(
            route_pipeline,
            "run_research_pipeline_streaming",
            blocked_pipeline,
        )
        await _cancel_research_consumer(
            _research_generator(app),
            started,
            settled,
            child_tasks,
            expected_text="partial answer" if stage == "writing" else "",
        )
        assert len(archive.calls) == 1, stage
        assert archive.calls[0]["partial"] is True, stage
        if stage == "writing":
            assert archive.calls[0]["assistant_content"] == "partial answer"


async def test_research_queue_full_cleanup_cancels_producer(monkeypatch):
    app = _fake_app({})
    put_attempted = asyncio.Event()
    producer_tasks = []

    async def immediate_planning(**kwargs):
        return kwargs["messages"], []

    async def fill_event_queue(*args, **kwargs):
        producer_tasks.append(asyncio.current_task())
        for index in range(257):
            if index == 256:
                put_attempted.set()
            yield {
                "type": "researcher_done",
                "model": f"r{index}",
                "ok": True,
                "elapsed_s": 0.0,
            }

    monkeypatch.setattr(route_pipeline, "_phase_thinking", immediate_planning)
    monkeypatch.setattr(
        route_pipeline,
        "run_research_pipeline_streaming",
        fill_event_queue,
    )

    stream = _research_generator(app)
    try:
        while True:
            frame = await asyncio.wait_for(anext(stream), timeout=1)
            if "_Researching_" in frame:
                break
        await asyncio.wait_for(put_attempted.wait(), timeout=1)
        await asyncio.sleep(0)
        assert producer_tasks and not producer_tasks[0].done()
        await asyncio.wait_for(stream.aclose(), timeout=1)
        await asyncio.sleep(0)
        assert producer_tasks[0].done()
    finally:
        for task in producer_tasks:
            task.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)


def _research_outcome_counts() -> dict[str, float]:
    return {
        outcome: pipeline_total.labels(
            mode="deep",
            task_type="reasoning",
            outcome=outcome,
        )._value.get()
        for outcome in ("ok", "error", "cancelled", "truncated")
    }


async def test_research_missing_done_has_one_identity_and_truncated_outcome(monkeypatch):
    app = _fake_app({})
    archive = _ResearchArchive()
    app.state.archive_client = archive

    async def immediate_planning(**kwargs):
        return kwargs["messages"], []

    async def truncated_pipeline(*args, **kwargs):
        yield {"type": "findings_ready", "grounded": True}
        yield {"type": "write_delta", "text": "partial research answer"}

    monkeypatch.setattr(route_pipeline, "_phase_thinking", immediate_planning)
    monkeypatch.setattr(
        route_pipeline,
        "run_research_pipeline_streaming",
        truncated_pipeline,
    )

    before = _research_outcome_counts()
    frames = [frame async for frame in _research_generator(app)]
    after = _research_outcome_counts()

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
    assert archive.calls[0]["assistant_content"] == "partial research answer"
    assert archive.calls[0]["partial"] is True
