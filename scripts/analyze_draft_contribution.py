#!/usr/bin/env python3
"""Measure which panel workers a synthesised answer actually resembles.

WHY

`171841` raised a question the eval harness structurally cannot answer: all
five cases passed, and yet reading the drafts by hand suggested only two of
the four workers were doing anything. `deepseek-v4-pro` supplied the winning
code in every case, `minimax-m3` the exposition in three, and `nemotron` and
`kimi-k2.7-code` appeared to contribute nothing that survived synthesis.

"Appeared to" is the problem. That verdict was one person reading five
artifacts, which is exactly the n=1 rate call `PROJECT_STATE.md` keeps warning
about. A panel costs real cloud credit per worker, so "does this slot earn its
call" deserves a number, not an impression.

The drafts are already in the artifact — `agentic.debug_panel_drafts` writes a
`## Panel drafts (debug)` block beside every answer. So this script needs no
new instrumentation and no new eval run: it reads artifacts you already have,
and it keeps working on every future one for free.

⚠️⚠️ WHAT THIS MEASURES, AND THE ONE THING IT CANNOT

It measures RESEMBLANCE: how much of the final answer's content also appears in
a given worker's draft. That is not the same as contribution, and the gap runs
in one direction only:

  • A LOW score is strong evidence. If a worker's draft never resembles the
    output across many cases, the synthesizer is not using it.
  • A HIGH score is ambiguous. When four models converge on the obvious
    solution to `toposort`, all four score high and none of them is thereby
    shown to have caused anything. Agreement between drafts is itself a signal
    the synthesizer uses, and that use is invisible here.

⚠️⚠️ **RECALL RISES WHEN THE FIELD SHRINKS — NEVER COMPARE IT ACROSS LINEUPS.**
The score is resemblance to the FINAL, and the final is a merge of whoever
drafted. Take a worker out and the synthesizer has fewer sources to blend, so
every survivor's recall goes up without anyone improving. Dropping `minimax-m3`
moved `deepseek-v4-pro` from 0.69 to 0.90 code recall overnight; that is
arithmetic, not a better model. Compare within one lineup only.

⚠️ Latency and recall both move with CASE MIX too. The same model read 11.1s
across a mixed corpus and 3.2s on five hard code cases — and 3.8s on those same
five cases in the previous run, so the "speedup" was the mix, not the model.

So read this as a NEGATIVE instrument — it identifies dead weight, it does not
award credit. Do not rank two high scorers against each other with it.

⚠️ It also cannot see a draft's CORRECTNESS. Nemotron's `parse-duration` draft
on `171841` scored a respectable code recall and raises `AttributeError` on
every input. Resemblance and quality are different axes; `--check-syntax` only
catches drafts that will not even parse.

USAGE

⚠️ Unraid's shell has no python3, and eval artifacts land on the BOX at
`${APPDATA}/testing-out/`. So on-box runs go through a container. This is pure
local parsing — no model calls, no network, exits in under a second — so it
wants a plain `docker run --rm`, NOT `probe-onbox.sh` (which self-detaches for
long probes and would be all overhead here).

  # Every artifact with a drafts block, on the box. Read-only mounts; the
  # glob must expand INSIDE the container, hence `sh -c`.
  docker run --rm --entrypoint sh \\
    -v /mnt/user/appdata/audrey_ai_2.0/scripts:/s:ro \\
    -v /mnt/user/appdata/audrey_ai_2.0/testing-out:/out:ro \\
    audrey-eval:latest \\
    -c 'python /s/analyze_draft_contribution.py /out/*answers.md'

  # One run, with per-draft detail as well as the summary:
  … -c 'python /s/analyze_draft_contribution.py \\
        /out/2026-08-17-171841-code-hard-onbox-answers.md --per-case'

  # On the laptop, against artifacts pulled back from the box:
  uv run python scripts/analyze_draft_contribution.py docs/testing/*answers.md

  # Machine-readable, for accumulating across runs:
  … --json

Only the standard library is imported, so any python image serves if
`audrey-eval:latest` is not present.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# A case header is any `## …` that is not the drafts block; the drafts block
# is a `## …` too, which is why titles are matched rather than levels.
_H2 = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
_DRAFTS_TITLE = "Panel drafts (debug)"

# `### deepseek-v4-pro:cloud — 5.3s`. The separator is an em dash in the
# artifact, but a hand-edited file may carry a hyphen, so accept either.
_DRAFT_H3 = re.compile(r"^### (.+?)\s+[—–-]\s+([\d.]+)s\s*$", re.MULTILINE)

# The metadata list the harness writes directly under a case header. Dropped
# before the answer body is read, so `- checks: …` cannot be mistaken for prose.
_META_LINE = re.compile(
    r"^- (model|status|route|latency|banners|checks|code|sources|error|note):",
)

_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*$", re.MULTILINE)


@dataclass
class Draft:
    model: str
    elapsed_s: float
    text: str


@dataclass
class Case:
    name: str
    final: str
    drafts: list[Draft] = field(default_factory=list)


def _split_code_and_prose(text: str) -> tuple[str, str]:
    """Return (code, prose) — everything inside fences, everything outside.

    An unterminated fence — the truncation/thinking signature this campaign has
    been chasing — lands in CODE, which is what we want, and it does so with no
    special case: `split` alternates prose/code from index 0, so an odd fence
    count leaves an even part count whose last index is odd, i.e. code.

    ⚠️ DO NOT "fix" that by moving a trailing segment across. An earlier cut
    added exactly that branch and it popped the last PROSE element instead —
    on a one-fence draft it fed the intro paragraph into the code score and
    left prose empty. The alternation is already correct; a guard here can only
    break it.
    """
    parts = _FENCE.split(text)
    code = [p for i, p in enumerate(parts) if i % 2 == 1]
    prose = [p for i, p in enumerate(parts) if i % 2 == 0]
    return "\n".join(code), "\n".join(prose)


def _code_units(code: str) -> list[str]:
    """Significant code lines, normalised so formatting noise cannot score.

    Blank lines and whole-line comments are dropped — two models that differ
    only in commentary are the same code, and counting comments would let a
    verbose draft inflate its own recall.
    """
    out = []
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(" ".join(line.split()))
    return out


#: Word-shingle width for prose. 5 is long enough that ordinary English
#: ("the first exception it encounters") does not collide by chance, short
#: enough to survive the synthesizer's habit of re-cutting sentence boundaries.
_SHINGLE = 5


def _prose_units(prose: str) -> list[str]:
    """Overlapping word 5-grams, lowercased and punctuation-insensitive.

    ⚠️ Shingles, NOT sentences — and the difference is the whole measurement.
    Code survives synthesis verbatim; prose does not. The synthesizer merges
    two workers' explanations into one sentence, swaps a clause, keeps the
    ending. Matching whole sentences scored `minimax-m3` at 0.00 on a case
    where it had visibly supplied the entire explanation, because the shipped
    sentence started differently and ended identically. Shingles catch the
    surviving span wherever its boundaries land.
    """
    words = re.findall(r"[a-z0-9_]+", prose.lower())
    if len(words) < _SHINGLE:
        return []
    return [" ".join(words[i:i + _SHINGLE])
            for i in range(len(words) - _SHINGLE + 1)]


def _recall(final_units: list[str], draft_units: list[str]) -> float | None:
    """Fraction of the FINAL's units that also appear in the draft.

    Direction matters and is deliberate. Recall asks "how much of what shipped
    came from here", which is the question. The reverse (precision) would
    reward a draft for being short, and a one-line draft that happens to match
    would score 1.0.

    `None` when the final has no units of this kind — an answer with no code
    cannot tell you anything about who supplied its code.
    """
    if not final_units:
        return None
    have = set(draft_units)
    return sum(1 for u in final_units if u in have) / len(final_units)


def parse_artifact(text: str) -> list[Case]:
    """Pull cases and their drafts out of one `*-answers.md` artifact."""
    heads = list(_H2.finditer(text))
    cases: list[Case] = []
    for i, h in enumerate(heads):
        title = h.group(1).strip()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[h.end():end]
        if title == _DRAFTS_TITLE:
            if cases:
                cases[-1].drafts = _parse_drafts(body)
            continue
        cases.append(Case(name=title, final=_strip_meta(body)))
    return [c for c in cases if c.drafts]


def _strip_meta(body: str) -> str:
    lines = body.splitlines()
    start = 0
    for n, line in enumerate(lines):
        if _META_LINE.match(line) or not line.strip():
            start = n + 1
            continue
        break
    return "\n".join(lines[start:])


def _parse_drafts(body: str) -> list[Draft]:
    marks = list(_DRAFT_H3.finditer(body))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append(Draft(model=m.group(1).strip(),
                         elapsed_s=float(m.group(2)),
                         text=body[m.end():end]))
    return out


def _syntax_ok(code: str) -> bool | None:
    """`None` when there is nothing that looks like Python to check."""
    if not code.strip():
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    except (ValueError, RecursionError):
        return None
    return True


def score_case(case: Case, *, check_syntax: bool, artifact: str = "") -> list[dict]:
    """⚠️ `artifact` is part of case IDENTITY, not decoration.

    Case NAMES repeat across runs — `code-hard-tokenizer` appears in every
    `code_hard` artifact ever saved. Keying on the name alone silently merges
    six runs into one "case" and then picks a `top` by comparing one run's
    draft against another run's, which is not a comparison at all. Every
    grouping below keys on `(artifact, case)`.
    """
    fin_code, fin_prose = _split_code_and_prose(case.final)
    fin_c, fin_p = _code_units(fin_code), _prose_units(fin_prose)
    rows = []
    for d in case.drafts:
        d_code, d_prose = _split_code_and_prose(d.text)
        row = {
            "artifact": artifact,
            "case": case.name,
            "key": (artifact, case.name),
            "model": d.model,
            "elapsed_s": d.elapsed_s,
            "code_recall": _recall(fin_c, _code_units(d_code)),
            "prose_recall": _recall(fin_p, _prose_units(d_prose)),
            "has_code": bool(d_code.strip()),
            "final_has_code": bool(fin_c),
            # ⚠️ Recall confounds with LENGTH, and only this disambiguates it.
            # A draft that writes no prose scores ~0 prose recall for the same
            # reason a draft whose prose was rejected does — and those are
            # opposite findings. "Contributes no explanation" is a lineup
            # problem; "writes bare code" may just be the prompt working.
            "prose_words": len(re.findall(r"[a-z0-9_]+", d_prose.lower())),
        }
        if check_syntax:
            row["syntax_ok"] = _syntax_ok(d_code)
        rows.append(row)
    return rows


def _fmt(v: float | None) -> str:
    return " —  " if v is None else f"{v:5.2f}"


def summarise(rows: list[dict]) -> list[dict]:
    """Per-model aggregate, plus the counts that make a low score readable."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    # "Top" is per case, over drafts that have a code recall at all. Ties count
    # for everyone tied — a shared top spot is exactly the convergence case
    # this script cannot resolve, and silently picking one would hide that.
    tops: dict[str, int] = defaultdict(int)
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[r["key"]].append(r)
    for group in by_key.values():
        scored = [r for r in group if r["code_recall"] is not None]
        if not scored:
            continue
        best = max(r["code_recall"] for r in scored)
        for r in scored:
            if r["code_recall"] == best:
                tops[r["model"]] += 1

    out = []
    for model, rs in by_model.items():
        code = [r["code_recall"] for r in rs if r["code_recall"] is not None]
        prose = [r["prose_recall"] for r in rs if r["prose_recall"] is not None]
        out.append({
            "model": model,
            "cases": len(rs),
            "code_cases": len(code),
            "mean_code_recall": sum(code) / len(code) if code else None,
            "mean_prose_recall": sum(prose) / len(prose) if prose else None,
            "mean_prose_words": sum(r["prose_words"] for r in rs) / len(rs),
            "top_code_cases": tops.get(model, 0),
            "missing_code": sum(1 for r in rs if r["final_has_code"] and not r["has_code"]),
            "syntax_bad": sum(1 for r in rs if r.get("syntax_ok") is False),
            "mean_elapsed_s": sum(r["elapsed_s"] for r in rs) / len(rs),
        })
    out.sort(key=lambda r: (r["mean_code_recall"] is None, -(r["mean_code_recall"] or 0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure which panel workers the final answer resembles.")
    ap.add_argument("artifacts", nargs="+", type=Path,
                    help="one or more *-answers.md files with a drafts block")
    ap.add_argument("--per-case", action="store_true",
                    help="print every draft's scores, not just the summary")
    ap.add_argument("--check-syntax", action="store_true", default=True,
                    help="ast.parse each draft's code (default on)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    rows: list[dict] = []
    skipped: list[str] = []
    for p in args.artifacts:
        try:
            cases = parse_artifact(p.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        if not cases:
            skipped.append(p.name)
            continue
        for c in cases:
            rows.extend(score_case(c, check_syntax=args.check_syntax,
                                  artifact=p.name))

    if not rows:
        print("no artifacts with a `## Panel drafts (debug)` block.", file=sys.stderr)
        print("Turn on `agentic.debug_panel_drafts` for measurement runs.",
              file=sys.stderr)
        return 2

    summary = summarise(rows)
    if args.json:
        json.dump({"summary": summary, "rows": rows}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if skipped:
        print(f"({len(skipped)} artifact(s) had no drafts block, ignored)\n")

    if args.per_case:
        for key in dict.fromkeys(r["key"] for r in rows):
            print(f"\n{key[1]}   [{key[0]}]")
            for r in (x for x in rows if x["key"] == key):
                flags = []
                if r["final_has_code"] and not r["has_code"]:
                    flags.append("NO CODE")
                if r.get("syntax_ok") is False:
                    flags.append("SYNTAX")
                print(f"  {r['model']:<34} code {_fmt(r['code_recall'])}"
                      f"  prose {_fmt(r['prose_recall'])}"
                      f"  {r['elapsed_s']:6.1f}s  {' '.join(flags)}")

    n_keys = len({r["key"] for r in rows})
    n_art = len({r["artifact"] for r in rows})
    print(f"\n── per-model over {n_keys} case-run(s) in {n_art} artifact(s) ──")
    print(f"{'model':<34} {'n':>4} {'code':>6} {'prose':>6} {'pwords':>7} "
          f"{'top':>5} {'/n':>6} {'nocode':>7} {'badsyn':>7} {'mean s':>8}")
    for s in summary:
        # `top` is unreadable without its denominator: 2 of 5 and 2 of 40 are
        # opposite findings and the raw count cannot tell them apart.
        rate = (f"{s['top_code_cases'] / s['code_cases']:5.2f}"
                if s["code_cases"] else "   — ")
        print(f"{s['model']:<34} {s['cases']:>4} {_fmt(s['mean_code_recall'])} "
              f"{_fmt(s['mean_prose_recall'])} {s['mean_prose_words']:>7.0f} "
              f"{s['top_code_cases']:>5} {rate:>6} {s['missing_code']:>7} "
              f"{s['syntax_bad']:>7} {s['mean_elapsed_s']:>8.1f}")

    print("\ncode = fraction of the FINAL answer's significant code lines "
          f"present verbatim in that draft;\nprose = same for its word "
          f"{_SHINGLE}-grams (prose gets paraphrased, code does not).")
    print("pwords = mean words of prose the draft WROTE. ⚠️  Read it beside "
          "`prose`: a near-zero score\n   at near-zero pwords means the model "
          "wrote no explanation, NOT that its explanation lost.")
    print("⚠️  A low score is evidence of non-contribution. A high score is NOT "
          "evidence of contribution —")
    print("   converging models all score high. `top` counts ties for everyone "
          "tied, on purpose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
