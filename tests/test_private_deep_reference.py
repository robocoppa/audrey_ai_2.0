"""Optional Private planning failures preserve the verified local answer path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_reference_library import _entry, _pdf, _record, _registry

from audrey.config import Config, EnvOverrides
from audrey.identity import Principal
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.planner import plan
from audrey.pipeline.run_events import RunEventContext, RunFinishedEvent, ToolCallFinishedEvent
from audrey.private_processing import PrivateProcessingRuntime
from audrey.processing_policy import ProcessingPolicy, ProcessingPolicyError
from audrey.references.irs_archive import ArchiveStore
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai.pipeline import _stream_via_pipeline
from audrey.routes.openai.schemas import ChatCompletionRequest
from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.private_dispatch import private_tool_scope
from audrey.tools.private_handlers import make_private_tool_handlers

ROUTER = "local-router:latest"
WORKER = "local-worker:latest"
SERVER = "http://custom-tools:8001"


@pytest.mark.parametrize("code", [
    "local_model_not_allowed", "local_model_digest_mismatch", "private_transport_not_allowed",
])
async def test_optional_planning_does_not_swallow_hard_private_policy_refusals(code):
    ollama = SimpleNamespace(chat=AsyncMock(side_effect=ProcessingPolicyError(code)))
    with pytest.raises(ProcessingPolicyError) as error:
        await plan(ollama, planner_model=ROUTER, user_text="Synthetic private request",
                   timeout_s=1, max_subtasks=3)
    assert error.value.code == code


async def test_native_private_deep_stream_reads_owned_irs_after_optional_planner_timeout(tmp_path):
    root = tmp_path / "irs"
    entry = _entry("p17.pdf", product="Publication 17", title="Your Federal Income Tax",
                   revision="2025")
    with ArchiveStore(root) as archive:
        _record(archive, [entry], content=_pdf("For use in preparing 2025 Returns"))

    cfg = Config({
        "router": {"model": ROUTER, "timeout_s": 20, "pin_schema": True},
        "title_generation": {"enabled": False},
        "agentic": {"planning": {"enabled": True, "min_prompt_tokens": 1},
                    "react": {"deep_worker": {"max_rounds": 3, "max_tool_result_chars": 4000}}},
        "model_registry": {"general": [{"name": WORKER, "location": "local"}]},
        "deep_panel_local": {"general": {"workers": [WORKER], "synthesizer": WORKER,
                                         "fallback_synth": WORKER}},
        "fast_path": {"tool_capable_models": [WORKER]},
    }, EnvOverrides(_env_file=None))
    tools = ToolRegistry({
        name: ToolSpec(name=name, description=name, parameters={"type": "object"},
                       server_url=SERVER, path=f"/{name}")
        for name in ("reference_lookup", "read_reference")
    })
    app = SimpleNamespace(state=SimpleNamespace(
        cfg=cfg, registry=ModelRegistry(cfg), health=HealthTracker(),
        gate=FairLocalGate(concurrency=1), inflight=UserInflightRegistry(max_inflight_per_user=3),
        tools=tools, reference_libraries=_registry(root),
    ))
    owner = Principal("usr_test", "owner", "test", "subject", "owner@example.com",
                      "Owner", "user", "active", "test")
    policy = ProcessingPolicy({ROUTER: "a" * 64, WORKER: "b" * 64}, "http://ollama:11434",
                              allowed_tools=frozenset(tools.names()),
                              tool_servers=frozenset({SERVER}))
    calls, planner_calls = [], []

    def respond(request):
        body = json.loads(request.content) if request.content else {}
        assert request.url.host == "ollama"
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": model, "model": model, "digest": digest}
                for model, digest in policy.local_models.items()
            ]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={
                "details": {"format": "gguf"}, "capabilities": ["completion", "tools"],
                "model_info": {"general.architecture": "local", "general.parameter_count": 100},
            })
        assert request.url.path == "/api/chat"
        calls.append(body)
        if body["model"] == ROUTER:
            if any("subtasks" in str(message.get("content"))
                   for message in body["messages"]):
                planner_calls.append(body)
                raise httpx.ReadTimeout("SYNTHETIC_PRIVATE_PROVIDER_ERROR", request=request)
            message = {"content": json.dumps({"task": "general", "confidence": 1.0})}
        else:
            message = {"content": "Publication 17 covers tax year 2025, PDF page 1."}
            if body.get("tools"):
                results = [message for message in body["messages"] if message["role"] == "tool"]
                if not results:
                    name, arguments = "reference_lookup", {"product_number": "Publication 17"}
                elif len(results) == 1:
                    document = json.loads(results[0]["content"])["items"][0]
                    name, arguments = "read_reference", {"document_id": document["document_id"], "page": 1}
                else:
                    name = None
                    assert "2025 Returns" in results[-1]["content"]
                if name:
                    message = {"content": "", "tool_calls": [
                        {"function": {"name": name, "arguments": arguments}},
                    ]}
        result = {"done": True, "message": message}
        if body.get("stream"):
            return httpx.Response(200, content=json.dumps(result) + "\n")
        return httpx.Response(200, json=result)

    runtime = PrivateProcessingRuntime(policy, ollama_transport=httpx.MockTransport(respond))
    question = "Use the owned IRS library to identify Publication 17's tax year from PDF page 1."
    messages = [{"role": "user", "content": question}]
    events = []
    context = RunEventContext(run_id="run_private_irs", conversation_id="con_private_irs",
                              assistant_message_id="msg_private_irs", mode="local", sink=events.append)
    async with runtime.scope(app) as execution:
        with private_tool_scope(user_id="owner", handlers=make_private_tool_handlers(execution, owner)):
            frames = [frame async for frame in _stream_via_pipeline(
                execution, ChatCompletionRequest(model="audrey_local", messages=messages, stream=True),
                messages, {}, user_id="owner", conversation_id=context.conversation_id,
                user_turn_text=question, event_context=context, model_tools=execution.state.tools,
            )]
    assert len(planner_calls) == 1
    assert {call["model"] for call in calls} == {ROUTER, WORKER}
    finished = [event for event in events if isinstance(event, ToolCallFinishedEvent)]
    assert [event.status for event in finished] == ["succeeded", "succeeded"]
    assert isinstance(events[-1], RunFinishedEvent) and events[-1].status == "succeeded"
    assert "2025" in "".join(frames)
    assert "SYNTHETIC_PRIVATE_PROVIDER_ERROR" not in "".join(frames)
    assert "internal error" not in "".join(frames)
