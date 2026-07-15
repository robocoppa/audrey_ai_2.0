"""Hermetic tests for the eval harness's pure pieces (scripts/eval_research.py).

The harness itself is a LIVE tool — it needs the box, OWUI, and a network. But
its newest checks (code extraction + execution, answer_contains), the sweep
expansion, the JSON results writer, and eval_compare's table builder are pure
functions we can pin offline. The subprocess in _run_code_check runs
`sys.executable` on a temp file — no network, no stack, still hermetic.

Same import pattern as test_web_search_routing.py: scripts/ isn't a packaged
module, so we add it to sys.path and import the harness directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eval_compare  # noqa: E402
import eval_research as er  # noqa: E402

# ── code-block extraction ───────────────────────────────────────────────────

def test_extract_largest_python_block():
    answer = (
        "Here's a usage example:\n\n```python\nf(1)\n```\n\n"
        "And the implementation:\n\n```python\ndef f(x):\n    return x + 1\n```\n"
    )
    code = er._extract_code_block(answer, "python")
    assert code is not None and "def f(x):" in code


def test_extract_accepts_py_alias():
    answer = "```py\ndef g():\n    return 7\n```\n"
    code = er._extract_code_block(answer, "python")
    assert code is not None and "def g():" in code


def test_extract_none_when_no_python_block():
    assert er._extract_code_block("no code here", "python") is None
    # A bash block is not a python block.
    assert er._extract_code_block("```bash\nls -la\n```", "python") is None


def test_extract_ignores_untagged_fence():
    assert er._extract_code_block("```\nprint('hi')\n```", "python") is None


def test_has_tagged_code_block():
    assert er._has_tagged_code_block("```bash\nls\n```") is True
    assert er._has_tagged_code_block("```\nls\n```") is False
    assert er._has_tagged_code_block("plain prose") is False


def test_extraction_stops_at_debug_blocks():
    # With debug_panel_drafts on, worker drafts stream AFTER the answer and can
    # carry their own (bigger) code fences — extraction must not grab those.
    answer = (
        "```python\ndef real():\n    return 1\n```\n\n"
        "## Panel drafts (debug)\n\n"
        "```python\ndef draft():\n    # a much longer draft implementation\n"
        "    total = 0\n    for i in range(100):\n        total += i\n"
        "    return total\n```\n"
    )
    code = er._extract_code_block(answer, "python")
    assert code is not None and "def real():" in code and "draft" not in code


# ── code execution ──────────────────────────────────────────────────────────

def test_run_code_check_passes():
    ok, detail = er._run_code_check(
        "def f(x):\n    return x * 2\n", "assert f(3) == 6\n", 10.0)
    assert ok is True
    assert detail == "exit 0"


def test_run_code_check_assert_failure_carries_detail():
    ok, detail = er._run_code_check(
        "def f(x):\n    return x\n", "assert f(3) == 6\n", 10.0)
    assert ok is False
    assert "AssertionError" in detail


def test_run_code_check_times_out():
    ok, detail = er._run_code_check("while True:\n    pass\n", "pass\n", 1.0)
    assert ok is False
    assert "timeout" in detail


def test_run_code_check_syntax_error():
    ok, detail = er._run_code_check("def f(:\n", "pass\n", 10.0)
    assert ok is False
    assert "exit" in detail


# ── answer_contains ─────────────────────────────────────────────────────────

def test_contains_all_case_insensitive():
    assert er._contains_all("The answer is K2 at 8611m.", ["k2", "8611"]) is True
    assert er._contains_all("The answer is K2.", ["k2", "8611"]) is False


def test_contains_ignores_debug_region():
    answer = "Final answer: unsure.\n\n## Research trace (debug)\n\nnotes say 82.8"
    assert er._contains_all(answer, ["82.8"]) is False


# ── sweep expansion ─────────────────────────────────────────────────────────

def test_expand_sweep_crosses_and_groups_by_model():
    cases = [
        {"name": "a", "prompt": "pa", "code_test": "t"},
        {"name": "b", "prompt": "pb"},
    ]
    out = er._expand_sweep(cases, ["m1", "m2"])
    assert [(c["name"], c["model"]) for c in out] == [
        ("a [m1]", "m1"), ("b [m1]", "m1"),   # all of model 1 first (GPU-load
        ("a [m2]", "m2"), ("b [m2]", "m2"),   # friendly), then model 2
    ]
    # Original case fields survive the copy; originals aren't mutated.
    assert out[0]["code_test"] == "t"
    assert "model" not in cases[0]


def test_expand_sweep_name_falls_back_to_prompt():
    out = er._expand_sweep([{"prompt": "what is love"}], ["m1"])
    assert out[0]["name"] == "what is love [m1]"


# ── save_json ───────────────────────────────────────────────────────────────

def test_save_json_round_trips(tmp_path):
    # No source_stats on this result → the "sources" field serializes to null,
    # keeping the record shape stable (e.g. a code case).
    r = er.CaseResult(
        name="c1 [m1]", model="m1", ok=False,
        checks={"reachable": True, "code_runs": False, "banners": None},
        answer="x" * 40, route="unknown", ttft_s=1.5, total_s=12.0,
        code_detail="exit 1: AssertionError",
    )
    out = tmp_path / "results.json"
    er.save_json([r], out)
    records = json.loads(out.read_text())
    assert records == [{
        "name": "c1 [m1]", "model": "m1", "ok": False,
        "checks": {"reachable": True, "code_runs": False, "banners": None},
        "route": "unknown", "ttft_s": 1.5, "total_s": 12.0,
        "answer_len": 40, "banners": [], "error": "",
        "code_detail": "exit 1: AssertionError", "sources": None,
    }]


def test_save_json_includes_source_stats(tmp_path):
    # A research case carries grounding-quality numbers into the record so runs
    # can be compared on source quality, not just pass/fail + latency.
    r = er.CaseResult(
        name="attn", model="audrey_research", ok=True,
        checks={"reachable": True}, answer="ans", route="research",
        source_stats=er.SourceStats(
            total=5, official=1, academic=1, low_quality=0, other=3, quality="GOOD",
        ),
    )
    out = tmp_path / "results.json"
    er.save_json([r], out)
    rec = json.loads(out.read_text())[0]
    assert rec["sources"] == {
        "total": 5, "official": 1, "academic": 1,
        "low_quality": 0, "other": 3, "quality": "GOOD",
    }


# ── eval_compare.build_table ────────────────────────────────────────────────

def _rec(name, model, ok, total=10.0, **extra):
    rec = {"name": name, "model": model, "ok": ok, "checks": {"reachable": True},
           "route": "unknown", "ttft_s": 1.0, "total_s": total,
           "answer_len": 100, "banners": [], "error": "", "code_detail": ""}
    rec.update(extra)
    return rec


def test_build_table_matrix_strips_sweep_suffix():
    table = eval_compare.build_table([
        _rec("case-a [m1]", "m1", True),
        _rec("case-a [m2]", "m2", False,
             checks={"code_runs": False}, code_detail="exit 1: AssertionError"),
    ])
    # One row for case-a, both model columns, pass/fail marks with latency.
    assert "| case-a | ✅ 10s | ❌ 10s |" in table
    assert "`m1`" in table and "`m2`" in table
    # Failures section names the failing check and the code detail.
    assert "**case-a** on `m2` — code_runs; code: exit 1: AssertionError" in table


def test_build_table_summary_and_missing_cells():
    table = eval_compare.build_table([
        _rec("a", "m1", True, total=10.0),
        _rec("b", "m1", True, total=20.0),
        _rec("a", "m2", True, total=30.0),   # m2 never ran case b → "—" cell
    ])
    assert "| b | ✅ 20s | — |" in table
    assert "| `m1` | 2/2 | 1.0s | 15.0s | 100.0 |" in table
    assert "| `m2` | 1/1 | 1.0s | 30.0s | 100.0 |" in table
    assert "## Failures" not in table
