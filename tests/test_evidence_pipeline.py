"""Evidence tools remain usable through local ReAct and safe source projection."""
from __future__ import annotations

import asyncio
import json

from audrey.models.health import HealthTracker
from audrey.pipeline import deep_panel, react
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.run_observations import _observable_arguments
from audrey.tools.discovery import ToolRegistry, ToolSpec, ToolUserScope
from audrey.tools.dispatch import ToolResult


async def test_deep_local_worker_releases_gpu_for_visual_tool(monkeypatch):
    gate = FairLocalGate(concurrency=1)
    stages = []

    class Ollama:
        async def chat(self, **kwargs):
            assert gate._available == 0
            stages.append("answer-model")
            if len(stages) == 1:
                return {"message": {"tool_calls": [{
                    "function": {"name": "inspect_pdf_page", "arguments": {
                        "filename": "diagram.pdf", "page": 2,
                    }},
                }]}}
            assert "visible chart" in kwargs["messages"][-1]["content"]
            return {"message": {"content": "The chart shows an increase."}}

    async def dispatch(*args, **kwargs):
        async with gate.acquire("vision", location="local", user_id="owner"):
            stages.append("vision-model")
            return ToolResult(
                name="inspect_pdf_page", call_id=None,
                content=json.dumps({
                    "filename": "diagram.pdf", "artifact": "pdf_page", "page": 2,
                    "description": "visible chart",
                }), elapsed_s=0.01, is_error=False,
            )

    monkeypatch.setattr(react, "dispatch_one", dispatch)
    registry = ToolRegistry(by_name={
        "inspect_pdf_page": ToolSpec(
            name="inspect_pdf_page", description="inspect page", parameters={},
            server_url="http://tools", path="/inspect_pdf_page",
            user_scope=ToolUserScope.ARGUMENT,
        ),
    })
    result = await asyncio.wait_for(deep_panel._run_one_worker(
        Ollama(), HealthTracker(), gate, model="local", location="local",
        messages=[{"role": "user", "content": "Inspect page 2"}],
        options={}, timeout_s=1, tools=registry, tool_capable=True,
        react_max_rounds=2, react_compress_after=2, react_max_tool_chars=4000,
        react_dispatch_timeout_s=150, user_id="owner",
    ), timeout=2)
    assert result["content"] == "The chart shows an increase."
    assert stages == ["answer-model", "vision-model", "answer-model"]
    assert gate._available == 1


def test_page_and_timestamp_sources_are_distinct_without_private_contents():
    results = [
        ToolResult(
            name=name, call_id=None, is_error=False, elapsed_s=0,
            content=json.dumps({
                "filename": filename, "file_id": fid, "artifact": artifact,
                **position, "description": "private observed content",
            }),
        )
        for name, filename, fid, artifact, position in [
            ("inspect_pdf_page", "report.pdf", "pdf-id", "pdf_page", {"page": 1}),
            ("inspect_pdf_page", "report.pdf", "pdf-id", "pdf_page", {"page": 2}),
            ("inspect_video_frame", "clip.mp4", "video-id", "video_frame", {"timestamp_s": 135}),
            ("analyze_table", "costs.xlsx", "table-id", "table", {"sheet": "Costs"}),
        ]
    ]
    sources = react._observed_evidence(results)
    assert len(sources) == len({source["identity"] for source in sources}) == 4
    assert [source["title"] for source in sources] == [
        "report.pdf · Page 1", "report.pdf · Page 2",
        "clip.mp4 · Frame at 135s", "costs.xlsx · Sheet Costs",
    ]
    assert "private observed content" not in json.dumps(sources)


def test_new_tools_do_not_publish_private_inputs_in_run_events():
    assert _observable_arguments("calculate", {
        "operation": "evaluate", "expression": "123-private-value", "user": "forged",
    }) == {"operation": "evaluate", "expression": "[redacted]", "user": "[redacted]"}
    assert _observable_arguments("analyze_table", {
        "filename": "costs.csv", "filters": [{"value": "private client"}],
    }) == {"filename": "costs.csv", "filters": "[redacted]"}
    assert _observable_arguments("inspect_pdf_page", {
        "filename": "report.pdf", "page": 2, "question": "private question",
    }) == {"filename": "report.pdf", "page": 2, "question": "[redacted]"}


def test_visual_model_event_is_kept_in_native_persisted_model_snapshot():
    from collections import deque
    from types import SimpleNamespace

    from audrey.pipeline.run_events import RunEventEmitter
    from audrey.pipeline.run_observations import RunEventToolObserver
    from audrey.routes.app.runs import _LiveRun

    emitter = RunEventEmitter(
        run_id="run-model", conversation_id="conversation", assistant_message_id="assistant",
        mode="auto", virtual_model="audrey_auto", sink=lambda event: live.publish(event),
    )
    live = _LiveRun(
        owner_user_id="owner", storage_namespace="owner",
        started=SimpleNamespace(run=SimpleNamespace(run_id="run-model")), emitter=emitter,
        events=deque(), changed=asyncio.Event(), settled=asyncio.Event(),
    )
    emitter.run_started()
    emitter.message_started()
    observer = RunEventToolObserver(emitter)
    call = observer.started({"function": {"name": "inspect_video_frame", "arguments": {}}})
    observer.finished(call, ToolResult(
        name="inspect_video_frame", call_id=None, elapsed_s=0, is_error=False,
        content=json.dumps({"model": "vision-model", "description": "frame detail"}),
    ))
    assert len(live.model_snapshots) == 1
    assert live.model_snapshots[0].model == "vision-model"
    assert live.model_snapshots[0].calls == 1
