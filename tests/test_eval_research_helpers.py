"""Hermetic tests for the pure helpers in scripts/eval_research.py.

The eval script talks to a live stack, so most of it can't be unit-tested
offline. But its classification logic — route inference from the banner family,
and the latency/route formatting — is pure and worth pinning, because the
fast/deep routing-correctness check depends on infer_route being right.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_research.py"
_spec = importlib.util.spec_from_file_location("eval_research", _SCRIPT)
assert _spec and _spec.loader
eval_research = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module via sys.modules
# (string annotations from `from __future__ import annotations` look it up there).
sys.modules["eval_research"] = eval_research
_spec.loader.exec_module(eval_research)


class TestInferRoute:
    """Route is read from the banner FAMILY, not banner presence."""

    def test_fast_turn_reads_fast(self):
        # Fast emits only 'Thinking' — must NOT be misread as deep/unknown.
        assert eval_research.infer_route(["Thinking"]) == "fast"

    def test_deep_turn_reads_deep(self):
        assert eval_research.infer_route(
            ["Planning", "Dispatching panel", "Synthesizing"]) == "deep"

    def test_research_turn_reads_research(self):
        # Research shares 'Planning' with deep; its unique banners win.
        assert eval_research.infer_route(
            ["Planning", "Researching", "Verifying", "Writing"]) == "research"

    def test_research_with_factcheck_still_research(self):
        assert eval_research.infer_route(
            ["Planning", "Researching", "Verifying", "Fact-checking", "Writing"]
        ) == "research"

    def test_planning_only_reads_deep(self):
        # Deep shares 'Planning'; alone it's still the deep family, not fast.
        assert eval_research.infer_route(["Planning"]) == "deep"

    def test_no_banner_reads_unknown(self):
        # e.g. an error turn before any banner, or an OWUI utility task.
        assert eval_research.infer_route([]) == "unknown"

    def test_research_family_beats_deep_family(self):
        # If both deep-shared and research-unique banners appear, research wins
        # (the ordering in infer_route checks research first by design).
        assert eval_research.infer_route(
            ["Planning", "Synthesizing", "Writing"]) == "research"


class TestFmtLatency:
    def test_full_latency(self):
        r = eval_research.CaseResult(
            name="x", model="audrey_fast", ok=True, checks={}, answer="hi",
            route="fast", ttft_s=0.42, total_s=1.37)
        out = eval_research._fmt_latency(r)
        assert "route:fast" in out
        assert "ttft:0.4s" in out
        assert "total:1.4s" in out

    def test_missing_timings_omitted(self):
        r = eval_research.CaseResult(
            name="x", model="audrey_deep", ok=False, checks={}, answer="",
            route="unknown")
        out = eval_research._fmt_latency(r)
        assert out == "route:unknown"


class TestExpectRouteCheck:
    """run_case wires expect_route into a pass/fail check via infer_route.

    We don't hit the network; we monkeypatch _post_stream to return a canned
    (content, banners, error, timing) tuple and assert the route check.
    """

    def _patch_stream(self, monkeypatch, banners):
        timing = eval_research.StreamTiming(ttft_s=0.1, total_s=0.2)
        monkeypatch.setattr(
            eval_research, "_post_stream",
            lambda *a, **k: ("answer body long enough to pass has_answer", banners, "", timing),
        )

    def test_expect_route_fast_pass(self, monkeypatch):
        self._patch_stream(monkeypatch, ["Thinking"])
        case = {"name": "auto-short", "prompt": "hi there friend",
                "model": "audrey_auto", "expect_route": "fast",
                "expect_banners": False, "expect_sources": False}
        r = eval_research.run_case("http://x", "k", case, "audrey_auto", 1.0)
        assert r.route == "fast"
        assert r.checks["route"] is True
        assert r.ok

    def test_expect_route_deep_mismatch_fails(self, monkeypatch):
        # Case expects deep but the stream shows a fast turn → route check fails.
        self._patch_stream(monkeypatch, ["Thinking"])
        case = {"name": "auto-deepintent", "prompt": "think hard, deep dive",
                "model": "audrey_auto", "expect_route": "deep",
                "expect_banners": False, "expect_sources": False}
        r = eval_research.run_case("http://x", "k", case, "audrey_auto", 1.0)
        assert r.route == "fast"
        assert r.checks["route"] is False
        assert not r.ok

    def test_no_expect_route_is_na(self, monkeypatch):
        self._patch_stream(monkeypatch, ["Thinking"])
        case = {"name": "fast-plain", "prompt": "what is 2+2",
                "model": "audrey_fast", "expect_banners": False,
                "expect_sources": False}
        r = eval_research.run_case("http://x", "k", case, "audrey_fast", 1.0)
        assert r.checks["route"] is None  # not applicable
        assert r.ok


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
