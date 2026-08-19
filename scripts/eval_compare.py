#!/usr/bin/env python3
"""Build a case-by-model comparison table from eval harness JSON results.

WHAT THIS DOES

`scripts/eval_research.py --save-json` writes one flat JSON array of per-case
records (checks + latency, no answer text). This script reads one or more of
those files and renders a single markdown comparison: a case × model matrix
(pass/fail + total latency per cell), a per-model summary (pass rate, mean
TTFT/total, mean answer length), and a short list of every failure with its
failing checks. It is the seed for the hand-written report — the quality read
of the paired answers file stays yours.

Sweep runs (`--models`) suffix case names with ` [<model>]`, and `--repeat N`
suffixes them with `#2`, `#3`, …. BOTH suffixes are stripped here, so a case
lands on one row across models AND its repeats pool into ONE cell rendered as a
pass RATE (`⚠️ 3/5`). That is the point of the tool: a single sample of a hard
case says almost nothing, because the harness sends no `seed` and no
`temperature` (see `_options_from_request` — options come only from the request
body, and eval_research.py sets none), so every case is one draw from the
model's default sampler. Cells therefore lead with how OFTEN it passed.

⚠️ ACROSS FILES the later file WINS per (case, model) — rerun semantics, applied
to the whole repeat GROUP rather than record-by-record, so a 5-repeat rerun
REPLACES a 5-repeat original instead of interleaving with it. It does not pool.
▶ That is deliberate, and it is also a trap: on 2026-08-19 a thinking-ON run and
a thinking-OFF run of the same model were globbed into ONE invocation, and the
thinking-ON arm vanished from the table with no warning — the printed matrix
described one arm while appearing to describe both. It now prints a WARNING
naming every clobbered pair. **Compare two ARMS in two SEPARATE invocations.**
Pooling repeats of the SAME arm is the job this tool does for you.

USAGE

    # One sweep run:
    .venv/bin/python scripts/eval_compare.py \\
        docs/testing/2026-07-10-code-sweep-results.json

    # Merge a sweep with a deep-protocol run, write the compare file:
    .venv/bin/python scripts/eval_compare.py \\
        docs/testing/2026-07-10-code-sweep-results.json \\
        docs/testing/2026-07-10-code-deep-results.json \\
        --out docs/testing/2026-07-10-code-compare.md

Exit 0 on success (regardless of pass/fail content — this is a renderer, not a
gate); 2 on unreadable input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The sweep suffix eval_research.py's _expand_sweep appends to case names.
_SWEEP_SUFFIX = re.compile(r"\s*\[[^\]]+\]\s*$")
# The repeat marker _expand_repeats appends. It sits INSIDE the sweep suffix
# (`a#2 [m1]`), so the sweep strip has to run first.
_REPEAT_SUFFIX = re.compile(r"#\d+\s*$")


def _case_key(name: str) -> str:
    """Case name with the ' [<model>]' sweep suffix AND the '#N' repeat marker gone.

    ⚠️ Stripping `#N` here is what pools repeats into one cell, and it REVERSES
    the original design (a test used to pin `a#2 [m1]` → `a#2`, one row per
    repeat). Rows per repeat made the matrix grow without making it say
    anything: five rows of ✅/❌ for one case is a pass rate written the long
    way, and it silently pushed the real signal — how often — onto the reader.
    ▶ If you ever need the per-repeat detail back, it is in the paired answers
    markdown, which keeps every sample under its own `#N` header.
    """
    return _REPEAT_SUFFIX.sub("", _SWEEP_SUFFIX.sub("", name))


def _fmt_cell(group: list[dict]) -> str:
    """One cell: pass mark for a single sample, pass RATE for repeats.

    ⚠️ The mark for repeats is three-valued on purpose. `⚠️` (mixed) is a
    different fact from `❌` (never passed) and the distinction is the one that
    matters most for a model you are choosing: 4/5 is a model that can do the
    case, 0/5 is a model that cannot. Collapsing both to ❌ throws that away.
    Latency is the MEDIAN, not the mean — the first case of a run pays a cold
    model load (40–60s against 0.4s warm) and one such sample drags a mean of
    five far enough to invert a comparison.
    """
    n = len(group)
    passed = sum(1 for r in group if r.get("ok"))
    totals = [r["total_s"] for r in group
              if isinstance(r.get("total_s"), (int, float))]
    lat = f" {_median(totals):.0f}s" if totals else ""
    if n == 1:
        return ("✅" if passed else "❌") + lat
    mark = "✅" if passed == n else ("❌" if passed == 0 else "⚠️")
    return f"{mark} {passed}/{n}{lat}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float:
    """Median of a non-empty list (callers guard emptiness)."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _fmt_mean(value: float | None, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value is not None else "—"


def build_table(records: list[dict]) -> str:
    """Render the full comparison markdown from flat result records.

    Pure string-in/string-out (no I/O) so the hermetic tests can pin it.
    Models become columns in first-seen order; cases become rows in
    first-seen order — both follow the run order, which the sweep groups
    by model, so feeding files in a deliberate order controls the layout.
    """
    models: list[str] = []
    cases: list[str] = []
    # (case, model) -> EVERY sample for that pair. Repeats pool here; see
    # `_case_key` for why they are not their own rows.
    grid: dict[str, dict[str, list[dict]]] = {}
    for rec in records:
        model = rec.get("model") or "?"
        key = _case_key(rec.get("name") or "?")
        if model not in models:
            models.append(model)
        if key not in cases:
            cases.append(key)
        grid.setdefault(key, {}).setdefault(model, []).append(rec)

    lines: list[str] = ["## Case × model matrix", ""]
    lines.append("| case | " + " | ".join(f"`{m}`" for m in models) + " |")
    lines.append("|---" * (len(models) + 1) + "|")
    for key in cases:
        cells = [
            _fmt_cell(grid[key][m]) if m in grid[key] else "—"
            for m in models
        ]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-model summary", ""]
    # `flaky` counts CASES the model passed sometimes and failed sometimes. It
    # is the column to read first on a repeated run: a model at 12/15 with ZERO
    # flaky cases has three cases it simply cannot do, which is a different and
    # more actionable problem than the same 12/15 spread over three unreliable
    # ones. The old single-sample table could not tell those apart at all.
    lines.append("| model | pass | flaky | mean ttft | mean total "
                 "| mean answer chars |")
    lines.append("|---|---|---|---|---|---|")
    for m in models:
        groups = [grid[key][m] for key in cases if m in grid[key]]
        recs = [r for g in groups for r in g]
        passed = sum(1 for r in recs if r.get("ok"))
        flaky = sum(1 for g in groups
                    if 0 < sum(1 for r in g if r.get("ok")) < len(g))
        ttft = _mean([r["ttft_s"] for r in recs
                      if isinstance(r.get("ttft_s"), (int, float))])
        total = _mean([r["total_s"] for r in recs
                       if isinstance(r.get("total_s"), (int, float))])
        alen = _mean([float(r["answer_len"]) for r in recs
                      if isinstance(r.get("answer_len"), (int, float))])
        lines.append(
            f"| `{m}` | {passed}/{len(recs)} | {flaky} | {_fmt_mean(ttft, 's')} "
            f"| {_fmt_mean(total, 's')} | {_fmt_mean(alen)} |"
        )

    failures = [
        (key, m, grid[key][m])
        for key in cases for m in models
        if m in grid[key] and any(not r.get("ok") for r in grid[key][m])
    ]
    if failures:
        lines += ["", "## Failures", ""]
        for key, m, group in failures:
            bad = [r for r in group if not r.get("ok")]
            # Across repeats the same case can fail DIFFERENT checks each time,
            # so checks are COUNTED, not merged: "code_runs (3×)" next to
            # "not_truncated (1×)" says the failure has two distinct causes and
            # only one of them is about the code being wrong.
            counts: dict[str, int] = {}
            for r in bad:
                for c, v in (r.get("checks") or {}).items():
                    if v is False:
                        counts[c] = counts.get(c, 0) + 1
            if len(group) > 1:
                named = [f"{c} ({n}×)" for c, n in counts.items()]
                bits = [f"{len(bad)}/{len(group)} runs failed",
                        ", ".join(named) or "(no check recorded)"]
            else:
                # n=1 output stays byte-identical to the pre-repeat format.
                bits = [", ".join(counts) or "(no check recorded)"]
            details = [r["code_detail"] for r in bad
                       if r.get("code_detail")
                       and (r.get("checks") or {}).get("code_runs") is False]
            if details:
                bits.append(f"code: {_first_distinct(details)}")
            errors = [r["error"] for r in bad if r.get("error")]
            if errors:
                bits.append(f"error: {_first_distinct(errors)}")
            lines.append(f"- **{key}** on `{m}` — {'; '.join(bits)}")

    return "\n".join(lines) + "\n"


def _first_distinct(values: list[str], limit: int = 2) -> str:
    """First `limit` distinct strings, order-preserving, ' | '-joined.

    Repeats of one case usually fail the same way every time, and printing
    `exit 1: AssertionError` five times buries the one run that failed
    differently. Distinct-and-capped keeps that outlier on the same line.
    """
    seen: list[str] = []
    for v in values:
        if v not in seen:
            seen.append(v)
        if len(seen) == limit:
            break
    return " | ".join(seen)


def merge_files(per_file: list[tuple[str, list[dict]]]) -> tuple[
        list[dict], list[str]]:
    """Apply rerun semantics ACROSS files, by (case, model) GROUP.

    A later file's samples for a pair REPLACE an earlier file's — they do not
    pool. Pooling across files would silently blend two different ARMS (a
    thinking-ON run and a thinking-OFF run of one model) into a single
    meaningless rate, which is a worse failure than the clobber it would fix.

    Returns the merged records plus human-readable clobber warnings; the caller
    decides where to print them. Row/column order follows FIRST sight of a
    pair, so replacing a group never reshuffles the table.
    """
    merged: dict[tuple[str, str], list[dict]] = {}
    source: dict[tuple[str, str], str] = {}
    order: list[tuple[str, str]] = []
    warnings: list[str] = []
    for label, recs in per_file:
        here: dict[tuple[str, str], list[dict]] = {}
        for rec in recs:
            k = (_case_key(rec.get("name") or "?"), rec.get("model") or "?")
            here.setdefault(k, []).append(rec)
        for k, group in here.items():
            if k in merged:
                warnings.append(
                    f"{k[0]} on {k[1]}: {len(merged[k])} sample(s) from "
                    f"{source[k]} REPLACED by {len(group)} from {label}"
                )
            else:
                order.append(k)
            merged[k] = group
            source[k] = label
    return [r for k in order for r in merged[k]], warnings


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=Path,
                   help="one or more --save-json files from eval_research.py")
    p.add_argument("--out", type=Path, default=None,
                   help="write the markdown here instead of stdout "
                        "(convention: docs/testing/<date>-<desc>-compare.md)")
    args = p.parse_args()

    records: list[dict] = []
    per_file: list[tuple[str, list[dict]]] = []
    for path in args.results:
        if not path.exists():
            print(f"error: results file not found: {path}", file=sys.stderr)
            return 2
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(loaded, list):
            print(f"error: {path} is not a JSON array of records", file=sys.stderr)
            return 2
        per_file.append((path.name, loaded))
        records.extend(loaded)

    if not records:
        print("error: no records in the given files", file=sys.stderr)
        return 2

    # ⚠️ Warn LOUDLY before rendering. A clobber is invisible in the output —
    # the table looks complete and simply describes fewer arms than it appears
    # to. That is exactly how a thinking-ON arm disappeared on 2026-08-19.
    records, clobbered = merge_files(per_file)
    if clobbered:
        print(f"WARNING: {len(clobbered)} (case, model) pair(s) were replaced "
              f"by a later file — this is rerun semantics, NOT pooling. If "
              f"these files are different ARMS of one experiment, run them in "
              f"separate invocations:", file=sys.stderr)
        for w in clobbered:
            print(f"  - {w}", file=sys.stderr)

    table = build_table(records)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(table, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
