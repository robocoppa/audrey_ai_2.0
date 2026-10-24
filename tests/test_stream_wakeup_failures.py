"""Real producer timeouts must terminate streams, not act as banner signals."""
from __future__ import annotations

import asyncio

import pytest
from test_deep_stream_footer import _collect as collect_deep
from test_deep_stream_footer import _fake_app as deep_app
from test_research_stream import _collect as collect_research
from test_research_stream import _fake_app as research_app

from audrey.routes.openai import pipeline


@pytest.mark.parametrize("mode,after_progress", [
    ("deep", False), ("deep", True), ("research", False), ("research", True),
])
async def test_producer_timeout_terminates_before_and_after_stage_progress(
    monkeypatch, mode, after_progress,
):
    async def prepare(**kwargs):
        return kwargs["messages"], []

    async def dispatch(**_kwargs):
        return []

    async def fail(*_args, **_kwargs):
        if after_progress:
            if mode == "deep":
                yield {"type": "first_token", "model": "s"}
                yield {"type": "delta", "text": "Partial answer"}
            else:
                yield {"type": "findings_ready"}
        raise TimeoutError("Producer failed, not a banner wake")

    monkeypatch.setattr(pipeline, "_phase_thinking", prepare)
    monkeypatch.setattr(pipeline, "_phase_dispatch", dispatch)
    events = []
    if mode == "deep":
        monkeypatch.setattr(pipeline, "synthesize_stream", fail)
        app, collect = deep_app({}), collect_deep
    else:
        monkeypatch.setattr(pipeline, "run_research_pipeline_streaming", fail)
        app, collect = research_app({}), collect_research
    async with asyncio.timeout(2):
        frames = await collect(app, events=events)

    assert frames[-1] == "data: [DONE]\n\n"
    terminal = [event for event in events if event.type == "run.finished"]
    assert len(terminal) == 1 and terminal[0].status == "failed"
