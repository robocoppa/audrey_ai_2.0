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

import copy
import json
from types import SimpleNamespace
from typing import Any, ClassVar

from audrey.config import Config, EnvOverrides, get_config
from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.openai.pipeline import _stream_research_with_banners
from audrey.routes.openai.schemas import ChatCompletionRequest


class _FakeOllama:
    """Returns canned content per model for chat (researchers/verifier) and
    chat_stream (writer). Mirrors the stub in test_deep_panel.py."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses

    async def chat(self, *, model, messages, options=None, timeout_s=0, tools=None):
        # `tools=` accepted so the fact-checker's run_react loop can call us.
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


async def _collect(app, model="audrey_research"):
    # The route receives `messages` as plain dicts (the caller converts the
    # pydantic request into dicts before dispatching), so pass dicts here.
    msgs = [{"role": "user", "content": "tell me about euclid"}]
    payload = ChatCompletionRequest(model=model, messages=msgs, stream=True)
    frames = [
        frame async for frame in _stream_research_with_banners(
            app, payload, msgs, {}, task="reasoning", conf=0.9,
            user_id="", conversation_id="", user_turn_text="tell me about euclid",
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

    async def chat(self, *, model, messages, options=None, timeout_s=0,
                   tools=None, format=None):  # `format` mirrors OllamaClient.chat
        if format is not None:
            return {"message": {"content": _LEDGER_JSON},
                    "prompt_eval_count": 1, "eval_count": 1}
        return await super().chat(model=model, messages=messages,
                                  options=options, timeout_s=timeout_s,
                                  tools=tools)


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
