"""Request-level routing and planner scheduling regressions; no model server."""

from __future__ import annotations

from copy import deepcopy

import pytest

from audrey.config import get_config
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline import graph as graph_module
from audrey.pipeline.classify import DEFAULT_ROUTER_MODEL
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.openai import pipeline as route_pipeline
from audrey.tools.discovery import ToolRegistry


def _graph():
    cfg = deepcopy(get_config())
    gate = FairLocalGate(concurrency=1)
    compiled = graph_module.build_graph(
        cfg, object(), ModelRegistry(cfg), HealthTracker(), gate,
        ToolRegistry(), object(),
    )
    return cfg, gate, compiled


async def test_greeting_does_not_purchase_a_panel_after_a_successful_fast_reply(monkeypatch):
    cfg, _gate, compiled = _graph()
    cfg.raw["agentic"]["memory"]["enabled"] = False
    calls = []

    async def fast(*args, **kwargs):
        calls.append(kwargs)
        return "local", {"message": {"content": "Hello!"}, "done_reason": "stop"}

    async def unexpected(*args, **kwargs):
        pytest.fail("A completed greeting must not start a planner or deep panel")

    monkeypatch.setattr(graph_module, "run_fast_path", fast)
    monkeypatch.setattr(graph_module, "planner_plan", unexpected)
    monkeypatch.setattr(graph_module, "run_panel", unexpected)
    result = await compiled.ainvoke({
        "virtual_model": "audrey_auto",
        "messages": [{"role": "user", "content": "Hello"}],
    })

    assert len(calls) == 1
    assert result["classify_reason"] == "short_skip:general"
    assert result["content"] == "Hello!"
    assert result["mode"] == "fast"
    assert result["request_prompt_tokens"] == 1
    # Audrey's timestamp is still injected, but isn't mistaken for user context.
    assert result["messages"][0]["content"].startswith("Current server date and time:")


async def test_original_request_size_keeps_prior_context_and_native_transcript():
    _cfg, _gate, compiled = _graph()
    node = compiled.nodes["datetime"].bound
    history = [{"role": "assistant", "content": "Background " * 40},
               {"role": "user", "content": "Yes"}]
    ordinary = await node.ainvoke({"messages": history})
    assert ordinary["request_prompt_tokens"] > 8

    native = await node.ainvoke({
        "messages": [{"role": "system", "content": "Server instructions " * 40},
                     {"role": "user", "content": "Hello"}],
        "routing_messages": [{"role": "user", "content": "Hello"}],
    })
    assert native["request_prompt_tokens"] == 1


async def test_graph_planner_carries_the_request_gate_and_owner(monkeypatch):
    cfg, gate, compiled = _graph()
    cfg.router.pop("model")
    calls = []

    async def planner(*args, **kwargs):
        calls.append(kwargs)
        return ["one focused question"]

    monkeypatch.setattr(graph_module, "planner_plan", planner)
    result = await compiled.nodes["planner"].bound.ainvoke({
        "virtual_model": "audrey_deep", "prompt_tokens": 500,
        "messages": [{"role": "user", "content": "Plan this work"}],
        "user_id": "owner@example.com",
    })
    assert result["subtasks"] == ["one focused question"]
    assert calls[0]["gate"] is gate
    assert calls[0]["user_id"] == "owner@example.com"
    assert calls[0]["planner_model"] == DEFAULT_ROUTER_MODEL


async def test_stream_planner_carries_the_request_gate_and_owner(monkeypatch):
    gate = FairLocalGate(concurrency=1)
    calls = []

    async def planner(*args, **kwargs):
        calls.append(kwargs)
        return ["one focused question"]

    monkeypatch.setattr(route_pipeline, "planner_plan", planner)
    _messages, subtasks = await route_pipeline._phase_thinking(
        ollama=object(), tools=ToolRegistry(), http=object(), model_tools=ToolRegistry(),
        user_id="owner@example.com", messages=[{"role": "user", "content": "Plan this work"}],
        memory_enabled=False, memory_top_k=3, memory_timeout_s=5,
        planning_enabled=True, planning_min_tokens=40, planning_max_subtasks=3,
        prompt_tokens=500, router_cfg={}, gate=gate,
    )
    assert subtasks == ["one focused question"]
    assert calls[0]["gate"] is gate
    assert calls[0]["user_id"] == "owner@example.com"
    assert calls[0]["planner_model"] == DEFAULT_ROUTER_MODEL
