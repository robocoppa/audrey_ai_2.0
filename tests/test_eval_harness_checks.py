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


# ── expect_names_files ──────────────────────────────────────────────────────

_SHORT = "How to WIN with the London System.mp4"
_LONG = "Magnus Carlsen Teaches How to Win with the London System.mp4"
# The shape used by the video suite: alternatives per file.
_FILES = [
    [_SHORT, "How to WIN with the London System"],
    [_LONG, "Magnus Carlsen Teaches How to Win with the London System",
     "Magnus Carlsen video", "Carlsen video", "Magnus Carlsen"],
]


def test_naming_both_files_passes():
    answer = f"You have two: {_SHORT} and {_LONG}. Which did you mean?"
    assert er._names_all_files(answer, _FILES) is True


def test_naming_one_file_fails():
    """The regression this exists for. `video-ambiguous-singular` answered from
    one of two same-topic candidates, and every other check scored it PASS."""
    answer = f"Based on {_SHORT}, the move order begins with 1. d4."
    assert er._names_all_files(answer, _FILES) is False


def test_the_longer_filename_alone_does_not_satisfy_both():
    """⚠️ Why this is not `_contains_all`.

    `_LONG` contains `_SHORT` as a substring, so a plain contains-check on
    both is satisfied by an answer that named only the longer file — reporting
    a pass on the exact failure being tested. Resolving most-specific-first
    and consuming the match forces a second, separate mention.
    """
    answer = f"Based on {_LONG}, the move order begins with 1. d4."

    assert er._contains_all(answer, [_SHORT, _LONG]) is True  # the trap
    assert er._names_all_files(answer, _FILES) is False


def test_an_informal_reference_counts():
    """⚠️ A check that false-fails gets ignored, which is worse than no check.

    Models write "the Magnus Carlsen video" at least as often as the filename,
    and an answer that clearly distinguishes both files has done the thing
    being tested. Taken from a real `audrey_auto` answer, 2026-08-10.
    """
    answer = (
        'Based on your two London System videos: in the Magnus Carlsen video, '
        'Carlsen calls it a "solid choice". Your longer instructional video '
        '("How to WIN with the London System") emphasises middlegame plans.'
    )
    assert er._names_all_files(answer, _FILES) is True


def test_a_plain_string_still_works():
    """One file, one string — the list-of-alternatives form is opt-in."""
    assert er._names_all_files(f"read {_SHORT}", [_SHORT]) is True


def test_naming_is_case_insensitive():
    answer = "i checked how to win with the london system.mp4 and " + _LONG.lower()
    assert er._names_all_files(answer, _FILES) is True


def test_naming_ignores_the_debug_region():
    """Same rule as `contains`: a filename mentioned only in the trace is not
    the model naming it to the user."""
    answer = f"Based on {_SHORT}.\n\n## Research trace (debug)\n\nread {_LONG}"
    assert er._names_all_files(answer, _FILES) is False


# ── answer_not_contains ─────────────────────────────────────────────────────
#
# ⚠️ Added 2026-08-10 after `video-long-transcript-paging` scored PASS on the
# exact failure it was written to catch: three `get_file_text` pages in hand,
# no bounds given, nothing offered to continue, and the user told to "access
# the file directly". Some failures have a signature WORDING; that is
# checkable, where the good behaviour has a hundred valid phrasings.


def test_a_banned_phrase_fails():
    answer = ("I cannot provide the full transcript because I only retrieved "
              "partial sections due to output length constraints.")
    assert er._contains_any(answer, ["output length constraint"]) is True


def test_a_good_answer_trips_nothing():
    """The real `audrey_auto` answer from the same run — bounds and an offer."""
    answer = ("The transcript is quite long (33,626 characters total). That's "
              "the first ~4,005 characters, with about 29,600 remaining. Would "
              "you like me to continue reading it page by page?")
    assert er._contains_any(
        answer, ["system limitation", "output length constraint",
                 "access the file directly"]) is False


def test_any_one_banned_phrase_is_enough():
    """`_contains_any`, not `_contains_all` — one signature phrase is the
    failure; requiring all of them would pass every real case."""
    answer = "You would need to access the file directly."
    assert er._contains_any(answer, ["system limitation",
                                     "access the file directly"]) is True


def test_banned_phrases_ignore_the_debug_region():
    answer = "Here is page 1.\n\n## Research trace (debug)\n\nsystem limitation"
    assert er._contains_any(answer, ["system limitation"]) is False


# ── expect_continuation_offer ───────────────────────────────────────────────
#
# ⚠️ The second attempt at this case, and the reason the first one was wrong.
# `answer_not_contains` blacklisted the observed wordings ("output length
# constraints") on 2026-08-10 and was defeated on the NEXT run by both models
# paraphrasing without trying — "exceeds available context limits", "exceeds
# the available context window". A model rewords for free, so the check has to
# describe the shape of the failure, not the sentence that expressed it.
#
# The shape: refusing to hand over the whole thing is fine. Refusing with no
# way to continue is the failure. Both halves required.

_GOOD_PAGED = (
    "The transcript is quite long (33,626 characters total). Here's the first "
    "portion: [00:00:01] ladies and gentlemen... That's the first ~4,005 "
    "characters. Would you like me to continue reading it page by page?"
)
_BAD_CONTEXT_LIMIT = (
    "I cannot provide the full transcript because it exceeds available context "
    "limits. To get the full transcript, you would need to request it in "
    "sections using an offset parameter, or consult the file directly."
)
_BAD_GO_ELSEWHERE = (
    "I cannot provide a full transcript, which exceeds the available context "
    "window. You would need to use video transcription software."
)


def test_a_paged_answer_with_an_offer_passes():
    assert er._declines_without_offering(_GOOD_PAGED) is False


def test_refusing_with_no_way_forward_fails():
    assert er._declines_without_offering(_BAD_CONTEXT_LIMIT) is True


def test_being_sent_to_another_tool_entirely_fails():
    """Worst of the family: told to go transcribe a file Audrey has already
    transcribed in full."""
    assert er._declines_without_offering(_BAD_GO_ELSEWHERE) is True


def test_paraphrase_does_not_escape_it():
    """⚠️ The exact regression. Both of these slip past a substring blacklist
    of the previously-observed wordings; neither slips past the shape."""
    banned = ["system limitation", "output length constraint",
              "access the file directly"]

    for answer in (_BAD_CONTEXT_LIMIT, _BAD_GO_ELSEWHERE):
        assert er._contains_any(answer, banned) is False       # blacklist misses
        assert er._declines_without_offering(answer) is True   # shape catches


def test_refusing_but_offering_to_continue_is_allowed():
    """Declining to dump 33,000 characters is reasonable. The offer is what
    separates a judgement call from a dead end, so only the pair fails."""
    answer = ("I cannot provide the whole transcript in one message — it is "
              "33,626 characters. Would you like me to continue page by page?")
    assert er._declines_without_offering(answer) is False


def test_an_answer_that_never_refuses_is_untouched():
    """No refusal, no requirement — a case that simply answers must not be
    dragged into needing an offer."""
    answer = "Here is the continued transcript (pages 15-24): [00:06:48] ..."
    assert er._declines_without_offering(answer) is False


def test_the_suite_case_is_wired_up():
    """The helper is only worth having if a case actually opts in. Pins that
    `video-ambiguous-singular` — the case that regressed — carries it."""
    cases = json.loads(
        (Path(er.__file__).parent / "eval_prompts_video.json").read_text())
    by_name = {c["name"]: c for c in cases}

    assert by_name["video-ambiguous-singular"].get("expect_names_files")
    assert by_name["video-control-unscoped-plural"].get("expect_names_files")
    assert by_name["video-long-transcript-paging"].get("answer_not_contains")
    assert by_name["video-long-transcript-paging"].get("expect_continuation_offer")


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
