"""`analyze_escalations` counts what escalation costs — and says when it cannot.

The script exists because `Standing gotchas` asserts that every `audrey_auto`
escalation silently buys a 3-worker panel with two cloud models, and nothing
ever counted them. Cloud credits are a hard budget, so the number matters.

Two things are pinned here beyond ordinary parsing:

1. **The format strings it reads still exist in `graph.py`.** A log-parsing
   script whose patterns have drifted does not fail — it reports zero, and
   zero reads as "escalation is not happening". The repo has burned six
   sessions on parsers that produced confident, plausible, completely wrong
   tables, so the parser is asserted against the source that emits it.

2. **A pool it cannot price is reported, never counted as free.** Understating
   spend is the one error direction that matters for a budget.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import analyze_escalations as ae  # noqa: E402

# Real lines, with the prefix `logging.basicConfig` actually puts on them.
_PREFIX = "2026-08-15 17:48:20,304 INFO audrey.pipeline.graph: "
_LOG = [
    _PREFIX + "complexity: 463 tokens -> fast (owui_task)",
    _PREFIX + "classify: general (short_skip:general, conf=0.50)",
    _PREFIX + "complexity: 88 tokens -> fast (short)",
    _PREFIX + "classify: general (keyword:code_fence, conf=0.95)",
    _PREFIX + "complexity: 120 tokens -> fast (short)",
    _PREFIX + "classify: reasoning (router:reasoning, conf=0.70)",
    _PREFIX + "escalate: fast→deep (chars=42, conf=0.70, reason=too_short)",
    _PREFIX + "deep_panel: pool=deep_panel task=reasoning workers=3 ok=3 "
              "tool_grounded=1 attempted=['qwen3.6:35b', 'kimi-k2.6:cloud']",
    _PREFIX + "complexity: 2100 tokens -> deep (long)",
]


class TestTheParserMatchesWhatGraphActuallyLogs:
    """If `graph.py` changes its format string, this file must fail loudly."""

    def test_the_four_format_strings_are_still_in_graph_py(self):
        src = (_ROOT / "src" / "audrey" / "pipeline" / "graph.py").read_text()
        # Substrings, not whole lines: the `%`-args differ, the literal shape
        # is what the regexes anchor on.
        assert '"complexity: %d tokens -> %s (%s)"' in src
        assert '"classify: %s (%s, conf=%.2f)"' in src
        assert '"escalate: fast→deep (chars=%d, conf=%.2f, reason=%s)"' in src
        assert '"deep_panel: pool=%s task=%s workers=%d ok=%d' in src

    def test_each_regex_matches_a_real_prefixed_line(self):
        t = ae.parse_log(_LOG, since=None, ceiling=0.95)
        assert t.fast_turns == 3
        assert t.deep_turns == 1
        assert t.fast_owui == 1
        assert t.escalations == 1
        assert t.panels == 1
        assert t.unparsed == {}

    def test_the_escalate_arrow_is_the_unicode_one(self):
        # graph.py writes U+2192, not "->". A regex with the ASCII arrow would
        # match nothing and report a 0% escalation rate.
        assert "→" in ae._ESCALATE_RE.pattern
        ascii_arrow = _PREFIX + "escalate: fast->deep (chars=42, conf=0.7, reason=x)"
        assert ae._ESCALATE_RE.search(ascii_arrow) is None


class TestALineThatLooksOursButDoesNotParseIsReported:
    """Silent non-matching is the failure mode that makes the output a lie."""

    def test_drifted_line_lands_in_unparsed(self):
        drifted = [_PREFIX + "escalate: fast→deep because the answer was thin"]
        t = ae.parse_log(drifted, since=None, ceiling=0.95)
        assert t.escalations == 0
        assert t.unparsed["escalate"] == 1

    def test_the_report_shouts_about_it(self):
        t = ae.parse_log(
            [*_LOG, _PREFIX + "deep_panel: pool=deep_panel (shape changed)"],
            since=None, ceiling=0.95,
        )
        out = ae.render(ae.build_report(t, {}, {}, 0.95))
        assert "DID NOT PARSE" in out
        assert "deep_panel: 1" in out

    def test_an_unrelated_line_is_not_flagged(self):
        t = ae.parse_log(
            [_PREFIX + "research: factcheck batches — 9/10 answered"],
            since=None, ceiling=0.95,
        )
        assert t.unparsed == {}


class TestTheFallbackSuppressionIsOneUnderscoreWide:
    """`escalation_decision` suppresses `fallback:` — and only that.

    `classify.py` emits both `fallback:general` (0.25, every routing attempt
    failed → no signal → suppressed) and `fallback_keyword:...` (0.6, a real
    keyword hit after the router failed → still escalates). They differ by one
    character and mean opposite things.
    """

    def test_fallback_general_is_suppressed(self):
        t = ae.parse_log(
            [_PREFIX + "classify: general (fallback:general, conf=0.25)"],
            since=None, ceiling=0.95,
        )
        assert t.classify_fallback == 1
        assert t.classify_below_ceiling == 0

    def test_fallback_keyword_still_counts_as_would_escalate(self):
        t = ae.parse_log(
            [_PREFIX + "classify: code (fallback_keyword:code_fence, conf=0.60)"],
            since=None, ceiling=0.95,
        )
        assert t.classify_fallback == 0
        assert t.classify_below_ceiling == 1

    def test_an_exact_keyword_hit_sits_on_the_ceiling_and_does_not_trigger(self):
        # `keyword:` is pinned at 0.95 and the trigger is `conf < ceiling`,
        # strictly. The one confidence the classifier emits most is exactly
        # the boundary value — an off-by-one here would invert the finding.
        t = ae.parse_log(
            [_PREFIX + "classify: code (keyword:code_fence, conf=0.95)"],
            since=None, ceiling=0.95,
        )
        assert t.classify_below_ceiling == 0

    def test_reason_families_collapse_the_suffix(self):
        t = ae.parse_log(
            [_PREFIX + "classify: code (keyword:code_fence, conf=0.95)",
             _PREFIX + "classify: code (keyword:diff_marker, conf=0.95)"],
            since=None, ceiling=0.95,
        )
        assert t.classify_reasons["keyword"] == 2


class TestCloudDetection:
    def test_registry_wins_over_the_name(self):
        assert ae.is_cloud("weird-name", {"weird-name": "cloud"}) is True
        assert ae.is_cloud("looks:cloud-ish", {"looks:cloud-ish": "local"}) is False

    def test_both_real_cloud_suffix_shapes_are_caught_without_a_registry(self):
        # The suffix is not uniform in config.yaml.
        assert ae.is_cloud("kimi-k2.6:cloud", {}) is True
        assert ae.is_cloud("qwen3.5:397b-cloud", {}) is True
        assert ae.is_cloud("cogito-2.1:671b-cloud", {}) is True

    def test_local_models_are_not_cloud(self):
        assert ae.is_cloud("qwen3.6:35b", {}) is False
        assert ae.is_cloud("qwen3-coder-next:latest", {}) is False


class TestPanelPricing:
    _POOLS: ClassVar[dict[str, dict[str, dict[str, object]]]] = {
        "deep_panel": {
            "general": {
                "workers": ["qwen3.6:35b", "kimi-k2.6:cloud", "deepseek-v4-pro:cloud"],
                "synthesizer": "glm-5.2:cloud",
                "fallback_synth": "qwen3.6:35b",
            }
        },
        "deep_panel_research": {
            "general": {
                "researchers": ["qwen3.6:35b", "glm-5.2:cloud"],
                "verifier": "deepseek-v4-pro:cloud",
                "writer": "glm-5.2:cloud",
            }
        },
    }

    def test_a_mixed_panel_prices_workers_plus_synthesizer(self):
        # 3 workers + synth = 4 calls, of which 3 are cloud. `fallback_synth`
        # is NOT counted — it only runs when the primary fails.
        assert ae.panel_cost("deep_panel", "general", self._POOLS, {}) == (3, 4)

    def test_the_staged_research_pool_prices_its_own_roles(self):
        assert ae.panel_cost("deep_panel_research", "general", self._POOLS, {}) == (3, 4)

    def test_an_unknown_pool_returns_none_rather_than_zero(self):
        # Zero would silently understate spend, which for a hard budget is the
        # one direction that must not happen quietly.
        assert ae.panel_cost("deep_panel", "vl", self._POOLS, {}) is None
        assert ae.panel_cost("deep_panel_nonexistent", "general", self._POOLS, {}) is None

    def test_unpriced_panels_are_surfaced_as_a_floor(self):
        t = ae.parse_log(
            [_PREFIX + "deep_panel: pool=deep_panel task=vl workers=2 ok=2 "
                       "tool_grounded=0 attempted=[]"],
            since=None, ceiling=0.95,
        )
        rep = ae.build_report(t, self._POOLS, {}, 0.95)
        assert rep["panels"]["unattributed"] == 1
        assert rep["panels"]["cloud_calls"] == 0
        assert "floor" in ae.render(rep)


class TestConfigReading:
    def test_a_missing_config_degrades_to_a_census_without_prices(self, tmp_path):
        assert ae.read_config(tmp_path / "nope.yaml") == {}
        assert ae.load_pools({}) == ({}, {})

    def test_unparseable_yaml_degrades_too(self, tmp_path):
        bad = tmp_path / "config.yaml"
        bad.write_text("deep_panel: [unclosed\n")
        assert ae.read_config(bad) == {}

    def test_the_real_config_yields_pools_and_locations(self):
        raw = ae.read_config(_ROOT / "config.yaml")
        pools, locations = ae.load_pools(raw)
        assert "deep_panel" in pools
        assert locations, "model_registry should give a name→location map"
        # Cross-check the two cloud-detection paths agree on a real model.
        assert ae.is_cloud("glm-5.2:cloud", locations) is True
        assert ae.is_cloud("qwen3.6:35b", locations) is False

    def test_the_ceiling_comes_from_config_and_defaults_to_graphs_own(self):
        raw = ae.read_config(_ROOT / "config.yaml")
        assert ae._ceiling_from(raw) == 0.95
        assert ae._ceiling_from({}) == 0.95
        assert ae._ceiling_from({"agentic": {"escalation": {"confidence_ceiling": 0.5}}}) == 0.5

    def test_an_escalated_panel_costs_more_cloud_calls_than_the_gotcha_says(self):
        """`Standing gotchas` says "two cloud models" — it counts workers only.

        The synthesizer is `glm-5.2:cloud` too, so the unit of spend was already
        three. Pinned against the LIVE config so a pool edit surfaces here rather
        than quietly changing what an escalation costs — which is exactly what
        happened on 2026-08-15, when `minimax-m3:cloud` joined `code` and this
        test failed with `(4, 5) != (3, 4)`. That is the guard working; update
        the number deliberately, never by loosening the assert.
        """
        raw = ae.read_config(_ROOT / "config.yaml")
        pools, locations = ae.load_pools(raw)
        expected = {
            "general": (3, 4),
            "reasoning": (3, 4),
            "code": (4, 5),      # 3 cloud workers + cloud synth (2026-08-15)
        }
        for task, want in expected.items():
            got = ae.panel_cost("deep_panel", task, pools, locations)
            assert got == want, f"{task} pool changed shape: {got} != {want}"

    def test_the_code_pool_is_exactly_at_the_cloud_worker_cap(self):
        # `_healthy_workers` drops cloud workers past `max_deep_workers_cloud`
        # with NO log line, so a fifth entry here would be dead config that
        # looks live. This test is the thing that notices.
        raw = ae.read_config(_ROOT / "config.yaml")
        cap = int(((raw.get("agentic") or {}).get("max_deep_workers_cloud")) or 3)
        workers = raw["deep_panel"]["code"]["workers"]
        cloud = [w for w in workers if ae.is_cloud(w, ae.load_pools(raw)[1])]
        assert len(cloud) <= cap, (
            f"{len(cloud)} cloud workers against a cap of {cap} — the extras are "
            "silently dropped at dispatch"
        )


class TestWilsonInterval:
    def test_zero_trials_is_not_a_crash(self):
        assert ae.wilson(0, 0) == (0.0, 0.0)

    def test_the_interval_brackets_the_point_estimate(self):
        lo, hi = ae.wilson(5, 100)
        assert lo < 0.05 < hi

    def test_it_never_goes_negative_at_zero_successes(self):
        # The normal approximation does, which is why this uses Wilson.
        lo, hi = ae.wilson(0, 20)
        assert lo == 0.0
        assert 0.0 < hi < 0.25

    def test_a_small_sample_gives_a_uselessly_wide_interval(self):
        # Pinning the honesty, not the arithmetic: 1-of-4 must not read as 25%.
        lo, hi = ae.wilson(1, 4)
        assert hi - lo > 0.5


class TestTheReportRefusesToOverclaim:
    def test_a_thin_sample_is_labelled_a_mechanism_read(self):
        t = ae.parse_log(_LOG, since=None, ceiling=0.95)
        out = ae.render(ae.build_report(t, {}, {}, 0.95))
        assert "mechanism read, not a rate" in out

    def test_no_complexity_lines_is_flagged_rather_than_reported_as_zero(self):
        t = ae.parse_log([_PREFIX + "classify: general (router:general, conf=0.7)"],
                         since=None, ceiling=0.95)
        out = ae.render(ae.build_report(t, {}, {}, 0.95))
        assert "No `complexity:` lines" in out

    def test_a_large_clean_sample_gets_no_warning(self):
        lines = ([_PREFIX + "complexity: 100 tokens -> fast (short)"] * 200
                 + [_PREFIX + "escalate: fast→deep (chars=10, conf=0.5, reason=too_short)"] * 20)
        out = ae.render(ae.build_report(ae.parse_log(lines, since=None, ceiling=0.95),
                                        {}, {}, 0.95))
        assert "mechanism read" not in out
        assert "10.0%" in out


class TestSinceFilter:
    def test_lines_before_the_cutoff_are_dropped(self):
        early = "2026-08-01 09:00:00,000 INFO audrey.pipeline.graph: " \
                "complexity: 100 tokens -> fast (short)"
        t = ae.parse_log([early, *_LOG], since="2026-08-15", ceiling=0.95)
        assert t.fast_turns == 3  # the early one is gone

    def test_an_undated_line_is_kept(self):
        # Wrapped continuation lines carry no timestamp; dropping signal we
        # cannot date is worse than including it.
        t = ae.parse_log(["complexity: 100 tokens -> fast (short)"],
                         since="2026-08-15", ceiling=0.95)
        assert t.fast_turns == 1


class TestEndToEnd:
    def test_the_json_report_carries_the_headline_numbers(self):
        t = ae.parse_log(_LOG, since=None, ceiling=0.95)
        rep = ae.build_report(t, TestPanelPricing._POOLS, {}, 0.95)
        assert rep["turns"]["fast"] == 3
        assert rep["escalations"]["count"] == 1
        assert abs(rep["escalations"]["rate"] - 1 / 3) < 1e-9
        assert rep["classification"]["below_ceiling"] == 2   # 0.50 and 0.70
        assert rep["classification"]["observed"] == 3

    def test_escalated_share_of_panels_needs_no_pairing(self):
        # Deliberately NOT pairing escalate→panel across interleaved turns:
        # both are exact counts, and their ratio is well defined without
        # attributing individual panels under concurrency.
        t = ae.parse_log(_LOG, since=None, ceiling=0.95)
        rep = ae.build_report(t, {}, {}, 0.95)
        assert rep["panels"]["escalated_share"] == 1.0
