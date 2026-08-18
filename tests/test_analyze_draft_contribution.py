"""`analyze_draft_contribution` says which panel workers the answer resembles.

The script exists to replace a hand read. On `171841` a person read five
artifacts and concluded two of four workers were dead weight — a verdict about
a RATE, formed at n=1, of exactly the kind `PROJECT_STATE.md` keeps calling
unreadable. A panel worker is real cloud credit, so the claim needs counting.

Three things are pinned here beyond ordinary parsing:

1. **The artifact format it reads still matches what `eval_research.py`
   writes.** A parser that has drifted does not fail — it finds no drafts and
   reports nothing, which reads as "no artifacts had a drafts block". That is
   the same silent-zero failure `test_analyze_escalations` was written for.

2. **Code is matched verbatim, prose by shingle.** The two axes need different
   metrics because synthesis treats them differently: it lifts code and
   rewrites prose. Matching prose by whole sentences scored a worker at 0.00
   on a case where it had visibly supplied the entire explanation. That
   regression is pinned directly.

3. **Recall runs final→draft, not draft→final.** The reverse rewards a draft
   for being short, and a one-line draft that happened to match would score a
   perfect 1.0 while contributing nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import analyze_draft_contribution as adc  # noqa: E402

# Mirrors the real artifact shape: case header, harness metadata list, the
# final answer, then the drafts block with `### model — Ns` sections.
_ARTIFACT = """# eval run — fixture

1 cases, 1 passed all applicable checks.

---

## code-hard-tokenizer

- model: `audrey_deep`
- status: PASS
- checks: reachable:OK  code_block:OK
- code: exit 0

The scanner walks the string once and never backtracks.

```python
def tokenize(s):
    i = 0
    return i
```

## Panel drafts (debug)

### twin:cloud — 2.1s

The scanner walks the string once and never backtracks.

```python
def tokenize(s):
    i = 0
    return i
```

### other:cloud — 9.9s

Something else entirely, phrased in completely different words here.

```python
def tokenize(s):
    raise NotImplementedError
```
"""


def _cases():
    return adc.parse_artifact(_ARTIFACT)


def test_parses_the_case_its_drafts_and_their_elapsed_times():
    cases = _cases()
    assert len(cases) == 1
    c = cases[0]
    assert c.name == "code-hard-tokenizer"
    assert [d.model for d in c.drafts] == ["twin:cloud", "other:cloud"]
    assert [d.elapsed_s for d in c.drafts] == [2.1, 9.9]


def test_the_harness_metadata_list_is_not_read_as_the_answer():
    # `- checks: reachable:OK` is harness output. Counting it as prose would
    # give every draft free overlap on boilerplate none of them wrote.
    final = _cases()[0].final
    assert "checks:" not in final
    assert "status: PASS" not in final
    assert "never backtracks" in final


def test_a_verbatim_draft_scores_one_and_an_unrelated_draft_scores_low():
    rows = adc.score_case(_cases()[0], check_syntax=True)
    twin = next(r for r in rows if r["model"] == "twin:cloud")
    other = next(r for r in rows if r["model"] == "other:cloud")
    assert twin["code_recall"] == 1.0
    assert twin["prose_recall"] == 1.0
    assert other["code_recall"] < 0.5
    assert other["prose_recall"] == 0.0


def test_prose_recall_survives_a_rewritten_sentence_boundary():
    """The regression that made the metric worth having.

    Synthesis re-cuts sentences: it keeps a span and changes what precedes it.
    Whole-sentence matching scores that 0.0 and reports a contributing worker
    as dead weight. Shingles must find the surviving span.
    """
    final = ("Passing return_exceptions=True makes gather store each task's "
             "exception object in the results list, and a one-line post-process "
             "maps those exception objects to None per the contract.")
    draft = ("With return_exceptions=True, gather instead stores each failed "
             "task's exception object in the results list, and a one-line "
             "post-process maps those exception objects to None per the contract.")
    recall = adc._recall(adc._prose_units(final), adc._prose_units(draft))
    assert recall is not None and recall > 0.4, (
        f"a rewritten opening must not zero out a surviving tail; got {recall}"
    )


def test_recall_is_final_over_draft_so_a_short_draft_cannot_score_one():
    # draft ⊂ final: the draft supplies one of the final's three lines.
    final = adc._code_units("a = 1\nb = 2\nc = 3")
    draft = adc._code_units("a = 1")
    assert adc._recall(final, draft) < 0.4


def test_comments_and_blank_lines_do_not_count_as_code():
    # Otherwise a chatty draft inflates its own recall on commentary alone.
    assert adc._code_units("# just a comment\n\n   \nx = 1") == ["x = 1"]


def test_an_unterminated_fence_counts_as_code_not_prose():
    """An odd fence count is the truncation/thinking signature. Scoring the
    dangling tail as prose would move a draft's code recall onto its prose
    recall and hide the very shape this campaign was chasing."""
    code, prose = adc._split_code_and_prose("intro text\n```python\nx = 1\ny = 2")
    assert "x = 1" in code
    assert "x = 1" not in prose
    assert "intro text" in prose


def test_no_code_in_a_draft_is_reported_when_the_final_has_code():
    rows = adc.score_case(_cases()[0], check_syntax=True)
    assert all(r["final_has_code"] for r in rows)
    stripped = adc.Case(name="x", final=_cases()[0].final,
                        drafts=[adc.Draft(model="m", elapsed_s=1.0,
                                          text="prose only, no fence at all")])
    row = adc.score_case(stripped, check_syntax=True)[0]
    assert row["has_code"] is False
    assert row["code_recall"] == 0.0


def test_a_draft_that_will_not_parse_is_flagged():
    case = adc.Case(name="x", final="```python\nx = 1\n```",
                    drafts=[adc.Draft(model="m", elapsed_s=1.0,
                                      text="```python\ndef broken(\n```")])
    assert adc.score_case(case, check_syntax=True)[0]["syntax_ok"] is False


def _row(artifact, case, model, code_recall):
    return {"artifact": artifact, "case": case, "key": (artifact, case),
            "model": model, "code_recall": code_recall, "prose_recall": None,
            "elapsed_s": 1.0, "has_code": True, "final_has_code": True}


def test_a_tied_top_score_counts_for_every_model_tied():
    """Convergence is the case the metric cannot resolve. Picking one winner
    would hide that; the summary must show the tie."""
    rows = [_row("f.md", "c", "a", 1.0), _row("f.md", "c", "b", 1.0)]
    summary = {s["model"]: s for s in adc.summarise(rows)}
    assert summary["a"]["top_code_cases"] == 1
    assert summary["b"]["top_code_cases"] == 1


def test_the_same_case_name_in_two_artifacts_is_two_cases():
    """The bug this pins made the first real run unreadable.

    Case NAMES repeat across runs — `code-hard-tokenizer` is in every
    `code_hard` artifact. Keying on the name alone merged six runs into one
    "case" and then picked a `top` by comparing one run's draft against a
    DIFFERENT run's. Both the denominator and the winner were wrong.
    """
    rows = [
        _row("run1.md", "shared-name", "a", 1.0),
        _row("run1.md", "shared-name", "b", 0.1),
        _row("run2.md", "shared-name", "a", 0.1),
        _row("run2.md", "shared-name", "b", 1.0),
    ]
    summary = {s["model"]: s for s in adc.summarise(rows)}
    # One top each — not one model taking both, and not a single merged case.
    assert summary["a"]["top_code_cases"] == 1
    assert summary["b"]["top_code_cases"] == 1
    assert summary["a"]["cases"] == 2


def test_the_summary_carries_the_denominator_for_its_top_count():
    """`top` alone is unreadable: 2 of 5 and 2 of 40 are opposite findings."""
    rows = [_row("r.md", "c1", "a", 1.0), _row("r.md", "c2", "a", 0.0),
            _row("r.md", "c2", "b", 1.0)]
    summary = {s["model"]: s for s in adc.summarise(rows)}
    assert summary["a"]["code_cases"] == 2
    assert summary["a"]["top_code_cases"] == 1


def test_an_answer_with_no_code_yields_no_code_verdict_rather_than_zero():
    """`None`, not 0.0. A research answer has no code, and scoring every worker
    0.0 there would read as universal non-contribution."""
    case = adc.Case(name="x", final="just prose, no fence anywhere in sight",
                    drafts=[adc.Draft(model="m", elapsed_s=1.0, text="also prose")])
    assert adc.score_case(case, check_syntax=True)[0]["code_recall"] is None


def test_an_artifact_with_no_drafts_block_yields_no_cases():
    # The pre-`debug_panel_drafts` artifacts are all of this shape. They must
    # drop out rather than arrive as cases with zero drafts and skew the mean.
    assert adc.parse_artifact("## a-case\n\n- status: PASS\n\nAn answer.\n") == []
