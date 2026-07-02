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


class TestDetectBanners:
    """Banners are read from the pre-separator region in the server's `> _X_`
    blockquote form only — so panel-drafts PROSE can't fake a banner (the
    deep-pythagoras 'Writing' / einstein 'Thinking' false positives)."""

    SEP = "\n\n---\n\n"
    DEEP = "> _Planning_ ✅\n> _Dispatching panel_ ✅\n> _Synthesizing_ ✅"

    def test_deep_banners_detected_in_order(self):
        assert eval_research._detect_banners(self.DEEP + self.SEP + "answer") == \
            ["Planning", "Dispatching panel", "Synthesizing"]

    def test_draft_prose_writing_does_not_register(self):
        # A draft opening "Writing in the 4th century…" after the answer must
        # NOT add a 'Writing' banner (would mislabel deep as research).
        content = (self.DEEP + self.SEP + "answer body"
                   "\n\n## Panel drafts (debug)\n\n### m\n\n"
                   "Writing in the 4th century BCE, Aristotle is our source.")
        b = eval_research._detect_banners(content)
        assert "Writing" not in b
        assert eval_research.infer_route(b) == "deep"

    def test_draft_prose_thinking_does_not_register(self):
        content = (self.DEEP + self.SEP + "answer body"
                   "\n\n## Panel drafts (debug)\n\n### m\n\n"
                   "Einstein's Thinking was intensely visual.")
        b = eval_research._detect_banners(content)
        assert "Thinking" not in b
        assert eval_research.infer_route(b) == "deep"

    def test_real_research_banners_still_detected(self):
        content = ("> _Planning_\n> _Researching_\n> _Verifying_\n> _Writing_"
                   + self.SEP + "answer")
        assert eval_research._detect_banners(content) == \
            ["Planning", "Researching", "Verifying", "Writing"]

    def test_real_fast_banner_still_detected(self):
        assert eval_research._detect_banners("> _Thinking_ ✅" + self.SEP + "x") == \
            ["Thinking"]

    def test_bare_word_in_answer_body_ignored(self):
        # Even a capitalized banner word in the ANSWER prose (not just drafts)
        # is ignored — detection is scoped to the pre-separator region.
        content = self.DEEP + self.SEP + "Planning is important. We are Writing now."
        assert eval_research._detect_banners(content) == \
            ["Planning", "Dispatching panel", "Synthesizing"]

    def test_no_banners_returns_empty(self):
        assert eval_research._detect_banners("just an answer, no banners") == []


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


class TestAnswerBody:
    """`_answer_body` splits on the FIRST banner separator, so an in-prose
    `---` rule can't swallow the answer (the deep-ssh-keys / missing-drafts bug)."""

    SEP = "\n\n---\n\n"

    def test_plain_answer_after_banners(self):
        content = f"> _Planning_ ✅{self.SEP}The real answer."
        assert eval_research._answer_body(content) == "The real answer."

    def test_in_prose_rule_does_not_truncate(self):
        # The synth wrote a section-break `---` inside the answer. Splitting on
        # the LAST separator would keep only "Part two"; the FIRST keeps both.
        content = (
            f"> _Planning_ ✅{self.SEP}"
            f"Part one of the answer.{self.SEP}Part two of the answer."
        )
        body = eval_research._answer_body(content)
        assert body.startswith("Part one")
        assert "Part two of the answer." in body

    def test_footer_and_drafts_blocks_are_retained(self):
        # Footer (`\n\n---\n>`) and drafts (`\n\n---\n#`) are NOT separators,
        # so they stay in the body even after an in-prose rule.
        content = (
            f"> _Synthesizing_ ✅{self.SEP}"
            f"Answer with a rule.{self.SEP}More answer."
            "\n\n---\n> _Tools used:_\n> - **m** — `web_search` ✅1"
            "\n\n## Panel drafts (debug)\n\n### m\n\ndraft text"
        )
        body = eval_research._answer_body(content)
        assert "Answer with a rule." in body
        assert "## Panel drafts (debug)" in body
        assert "_Tools used:_" in body

    def test_no_separator_returns_stripped_whole(self):
        assert eval_research._answer_body("  just text  ") == "just text"


class TestClassifyHost:
    """Domain buckets for the informational source breakdown."""

    def test_academic_domains(self):
        assert eval_research._classify_host("https://arxiv.org/abs/1706.03762") == "academic"
        assert eval_research._classify_host("https://stanford.edu/~x") == "academic"

    def test_low_quality_domains(self):
        assert eval_research._classify_host("https://www.facebook.com/groups/1") == "low_quality"
        assert eval_research._classify_host("https://x.wordpress.com/p") == "low_quality"

    def test_official_domains(self):
        assert eval_research._classify_host("https://en.wikipedia.org/wiki/X") == "official"
        assert eval_research._classify_host("https://deepseek.com/news") == "official"

    def test_academic_beats_official_ordering(self):
        # A .edu also matches nothing on the official list; ordering keeps it academic.
        assert eval_research._classify_host("https://cs.washington.edu/x") == "academic"

    def test_unlisted_is_other_not_official(self):
        assert eval_research._classify_host("https://random-startup.io/blog") == "other"

    def test_no_host_is_other(self):
        assert eval_research._classify_host("not-a-url") == "other"


class TestSourceStats:
    """The reported breakdown + one-word quality summary. Informational only."""

    def _answer(self, *urls):
        lines = "\n".join(f"- [t]({u})" for u in urls)
        return f"Answer body.\n\n## Sources\n{lines}\n"

    def test_good_when_authoritative_and_no_junk(self):
        s = eval_research.source_stats(
            self._answer("https://arxiv.org/abs/1", "https://en.wikipedia.org/wiki/X"),
            expected=True)
        assert (s.total, s.quality) == (2, "GOOD")
        assert s.academic == 1 and s.official == 1 and s.low_quality == 0

    def test_partial_when_junk_present(self):
        s = eval_research.source_stats(
            self._answer("https://arxiv.org/abs/1", "https://www.facebook.com/groups/9"),
            expected=True)
        assert (s.total, s.quality) == (2, "PARTIAL")
        assert s.low_quality == 1

    def test_thin_when_single_url(self):
        s = eval_research.source_stats(self._answer("https://arxiv.org/abs/1"), expected=True)
        assert (s.total, s.quality) == (1, "THIN")

    def test_na_when_no_block_and_not_expected(self):
        # A creative control has no Sources block and none expected → N/A, not THIN.
        s = eval_research.source_stats("A birthday toast, no sources.", expected=False)
        assert (s.total, s.quality) == (0, "N/A")

    def test_thin_when_block_expected_but_absent(self):
        s = eval_research.source_stats("No sources rendered.", expected=True)
        assert (s.total, s.quality) == (0, "THIN")


class TestSourceStatsNeverGates:
    """source_stats is reported on run_case results but must never affect `ok`."""

    def _patch_stream(self, monkeypatch, content, banners=None):
        # Full research banner set so the (unrelated) banners check passes and
        # can't mask what we're actually asserting about source_stats.
        banners = banners or ["Planning", "Researching", "Verifying", "Writing"]
        timing = eval_research.StreamTiming(ttft_s=0.1, total_s=0.2)
        monkeypatch.setattr(
            eval_research, "_post_stream",
            lambda *a, **k: (content, banners, "", timing),
        )

    def test_research_case_gets_stats(self, monkeypatch):
        answer = "Grounded answer body.\n\n## Sources\n- [a](https://arxiv.org/abs/1)\n"
        self._patch_stream(monkeypatch, answer)
        case = {"name": "bio", "prompt": "tell me about Euclid at length",
                "model": "audrey_research"}
        r = eval_research.run_case("http://x", "k", case, "audrey_research", 1.0)
        assert r.source_stats is not None
        assert r.source_stats.total == 1

    def test_junk_sources_do_not_fail_the_case(self, monkeypatch):
        # All-junk Sources → quality PARTIAL, but the case still passes its gates
        # (a well-formed URL is present); quality is informational, not a check.
        answer = ("Answer.\n\n## Sources\n- [a](https://www.facebook.com/groups/1)\n"
                  "- [b](https://www.scribd.com/doc/2)\n")
        self._patch_stream(monkeypatch, answer)
        case = {"name": "junk", "prompt": "some grounding prompt here",
                "model": "audrey_research"}
        r = eval_research.run_case("http://x", "k", case, "audrey_research", 1.0)
        assert r.source_stats.quality == "PARTIAL"
        assert r.source_stats.low_quality == 2
        assert r.checks["sources"] is True and r.checks["url_wellformed"] is True
        assert r.ok  # quality never gates

    def test_non_research_case_has_no_stats(self, monkeypatch):
        self._patch_stream(monkeypatch, "plain fast answer, long enough to pass")
        case = {"name": "fast", "prompt": "what is 2+2", "model": "audrey_fast",
                "expect_banners": False, "expect_sources": False}
        r = eval_research.run_case("http://x", "k", case, "audrey_fast", 1.0)
        assert r.source_stats is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
