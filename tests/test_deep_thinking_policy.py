"""Reasoning settings must reach final-answer roles on both drivers."""
from __future__ import annotations

from collections import defaultdict

import pytest

from audrey.config import Config, EnvOverrides
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.deep_panel import run_research_pipeline, select_researchers
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.synthesize import synthesize, synthesize_stream


class CapturingOllama:
    def __init__(self, *, capable=True):
        self.capable = capable
        self.calls = defaultdict(list)

    async def thinking_flag(self, model, want):
        return want if self.capable else None

    async def chat(self, *, model, think=None, **kwargs):
        self.calls[model].append(think)
        return {"message": {"content": "A grounded draft or answer."}}

    async def chat_stream(self, *, model, **kwargs):
        self.calls[model].append(kwargs.get("think"))
        yield {"message": {"content": "The final answer."}, "done": True}


def configuration(thinking):
    names = ["researcher", "verifier", "writer"]
    return Config({
        "thinking": thinking,
        "model_registry": {"reasoning": [
            {"name": name, "location": "cloud", "priority": 10} for name in names
        ]},
        "deep_panel": {"reasoning": {"synthesizer": "writer"}},
        "deep_panel_research": {"reasoning": {
            "researchers": ["researcher"], "verifier": "verifier", "writer": "writer",
        }},
    }, EnvOverrides(_env_file=None))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("policy,capable,expected", [
    ({"deep_synth": True}, True, False),
    ({"no_thinking_models": ["writer"]}, True, False),
    ({}, True, None),
    ({"deep_synth": True}, False, None),
])
async def test_synthesis_reasoning_policy_is_the_same_on_both_drivers(streaming, policy, capable, expected):
    cfg, ollama = configuration(policy), CapturingOllama(capable=capable)
    args = (cfg, ollama, ModelRegistry(cfg), HealthTracker(), FairLocalGate(concurrency=1))
    kwargs = dict(pool_key="deep_panel", task="reasoning", messages=[{"role": "user", "content": "Explain"}],
                  drafts=[{"model": "researcher", "content": "Draft"}], subtasks=[], timeout_s=1)
    if streaming:
        events = [event async for event in synthesize_stream(*args, **kwargs)]
        assert events[-1]["synth_error"] == ""
    else:
        assert (await synthesize(*args, **kwargs))["synth_error"] == ""
    assert ollama.calls["writer"] == [expected]


@pytest.mark.parametrize("policy,expected", [
    ({"deep_synth": True}, False),
    ({"no_thinking_models": ["verifier", "writer"]}, False),
    ({}, None),
])
async def test_research_verifier_and_writer_receive_final_answer_thinking_policy(policy, expected):
    cfg, ollama = configuration(policy), CapturingOllama()
    result = await run_research_pipeline(
        cfg, ollama, ModelRegistry(cfg), HealthTracker(), FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "Research this topic"}],
        options={}, timeout_s=1, max_researchers_cloud=2,
    )
    assert result["error"] == ""
    assert ollama.calls["researcher"] == [None]
    assert ollama.calls["verifier"] == [expected]
    assert ollama.calls["writer"] == [expected]


def test_dropped_cloud_researcher_is_logged(caplog):
    cfg = configuration({})
    cfg.raw["deep_panel_research"]["reasoning"]["researchers"] = ["researcher", "verifier"]
    with caplog.at_level("WARNING"):
        chosen = select_researchers(cfg, ModelRegistry(cfg), HealthTracker(),
                                   task="reasoning", max_researchers_cloud=1)
    assert chosen == [("researcher", "cloud")]
    assert "verifier" in caplog.text
    assert "max_research_workers_cloud=1" in caplog.text


async def test_named_factchecker_receives_worker_thinking_policy():
    from audrey.tools.discovery import ToolRegistry, ToolSpec

    cfg, ollama = configuration({"no_thinking_models": ["verifier"]}), CapturingOllama()
    cfg.raw["deep_panel_research"]["reasoning"].update(factchecker="verifier")
    tools = ToolRegistry({"web_search": ToolSpec(
        name="web_search", description="Search", parameters={},
        server_url="http://unused", path="/web_search",
    )})
    result = await run_research_pipeline(
        cfg, ollama, ModelRegistry(cfg), HealthTracker(), FairLocalGate(concurrency=1),
        task="reasoning", messages=[{"role": "user", "content": "Research this topic"}],
        options={}, timeout_s=1, max_researchers_cloud=2,
        tools=tools, tool_capable_models={"verifier"},
    )
    assert result["error"] == ""
    # The same named model fills verifier and prose factchecker roles.
    assert ollama.calls["verifier"] == [False, False]
