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
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.openai.pipeline import _stream_deep_with_banners
from audrey.routes.openai.schemas import ChatCompletionRequest
from audrey.tools.discovery import ToolRegistry, ToolSpec

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


async def _collect(app):
    msgs = [{"role": "user", "content": "what do my videos say about the london system"}]
    payload = ChatCompletionRequest(model="audrey_deep", messages=msgs, stream=True)
    return [
        frame async for frame in _stream_deep_with_banners(
            app, payload, msgs, {}, task="reasoning", conf=0.9,
            user_id="", conversation_id="", user_turn_text=msgs[0]["content"],
        )
    ]


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
