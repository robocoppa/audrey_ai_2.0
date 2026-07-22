#!/usr/bin/env python3
"""Lint lesson prose against the durable lesson-writing rules.

WHY

AGENTS.md (and auto-memory) encodes a handful of "always/never" rules
for lesson prose:

  - No "Phase N" references — readers landing on lessons have no idea
    what build phases were.
  - No real emails — `alice@example.com` is the placeholder.
  - No "Bart" in lesson text — use the placeholder above.
  - No forward-references to later lessons by number — say "the next
    lesson" or "when we cover X."
  - No specific codebase counts ("~16k chunks", "110 tests", "~430
    lines", "~9,800 LOC"). File:line citations are exempt.

This checker mechanizes those rules so a sweep doesn't depend on
eyeballing. It is a companion to scripts/check-lesson-links.py, which
validates source-code cite freshness. This script validates prose
hygiene.

HOW IT WORKS

For each lesson under docs/lesson-*/lesson-*.md (or any file passed on
the command line), the checker walks the file line-by-line outside
fenced code blocks (` ``` ` and ` ~~~ `) and applies five rules. Each
finding prints:

    <file>:<line>: <RULE>: <message>
       <offending line excerpt>
       fix: <suggested replacement, when mechanical>

Exit code is the number of findings (capped at 1 so CI can use plain
non-zero). Pass --quiet to suppress the per-finding output and only
print the count. Pass --json for machine-readable output.

RULES

  PHASE_N        Matches "Phase N" / "phase N" / "Phase NN" where the
                 token is a build-phase marker (not e.g. "phase space"
                 in physics prose — Audrey lessons don't talk about
                 physics, so any Phase+digit is flagged).

  REAL_EMAIL     Matches RFC-shaped emails whose domain is NOT one of
                 the RFC 2606 reserved test domains (example.com,
                 example.org, example.net, *.example, *.test,
                 *.invalid, *.localhost). Catches @proton.me,
                 @gmail.com, @anthropic.com, etc.

  BART           Case-insensitive "bart" as a whole word. Excludes
                 false-positive substrings (bartender etc.) by
                 requiring word boundaries.

  FORWARD_REF    A lesson NN file mentioning "Lesson M" where M >= NN+2.
                 The immediate next-lesson handoff (NN+1) is allowed
                 because every lesson footer points at it by title.
                 Catches the harmful pattern: mid-prose claims about
                 lessons that may not exist yet, or that skip ahead and
                 will rot when the course reshuffles. URL targets
                 (lesson-12-...md) inside `](...)` are excluded.

  COUNT          Specific numeric counts that bake in transient state:
                 "~Nk chunks", "N,NNN chunks", "N tests" (>= 10),
                 "~N LOC", "~N lines" (as code-base size, not as line
                 ranges). File:line citations like "main.py:53" and
                 short-line ranges like "lines 5-9" are excluded.

USAGE

    .venv/bin/python scripts/check-lesson-conventions.py
    .venv/bin/python scripts/check-lesson-conventions.py docs/lesson-ai/lesson-04-*.md
    .venv/bin/python scripts/check-lesson-conventions.py --json
    .venv/bin/python scripts/check-lesson-conventions.py --quiet

ENVIRONMENT

    DOCS_GLOB     Default `docs/lesson-*/lesson-*.md` — covers BOTH courses
                  (docs/lesson-ai/ and docs/lesson-python/). Override to scan
                  a different directory (e.g. for testing).

Run after editing a lesson; run periodically across the corpus.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────

# `docs/lesson-*/` matches BOTH courses: docs/lesson-ai/ and docs/lesson-python/.
# The old default was `docs/lessons/lesson-*.md`, a directory that stopped existing
# when the courses were renamed — so the checker silently matched zero files and
# exited clean while AGENTS.md told every session to run it after lesson edits.
DOCS_GLOB = os.environ.get("DOCS_GLOB", "docs/lesson-*/lesson-*.md")

# RFC 2606 reserved test domains. Everything outside this set is
# considered a "real" email for the REAL_EMAIL check.
RESERVED_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "example",
    "test",
    "invalid",
    "localhost",
}


# ─── Helpers ─────────────────────────────────────────────────────────


@dataclass
class Finding:
    file: str
    line: int
    rule: str
    message: str
    excerpt: str
    fix: str | None = None


def lesson_number_from_filename(path: str) -> int | None:
    """Extract NN from `lesson-NN-slug.md`. Returns None if not a lesson file."""
    name = Path(path).name
    m = re.match(r"lesson-(\d+)-", name)
    return int(m.group(1)) if m else None


def iter_prose_lines(text: str):
    """Yield (line_number, line_text) for lines outside fenced code blocks.

    Tracks both ``` and ~~~ fences. A fence opens on first occurrence,
    closes on the next. Indented-code-block-by-4-spaces is NOT excluded
    (lesson prose rarely uses it; cite checker treats those as prose).
    """
    in_fence = False
    fence_marker: str | None = None
    for i, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence_marker = stripped[:3]
            in_fence = True
            continue
        if in_fence and stripped.startswith(fence_marker or "```"):
            in_fence = False
            fence_marker = None
            continue
        if in_fence:
            continue
        yield i, raw


def strip_inline_code(line: str) -> str:
    """Remove `inline code spans` so rule regexes don't match inside them.

    Keeps the same line length by replacing each span with spaces, so
    column offsets in the original line stay meaningful (we don't use
    them for output, but it's cheap insurance against future regex
    anchoring).
    """
    out = []
    i = 0
    while i < len(line):
        if line[i] == "`":
            # find the closing backtick
            j = line.find("`", i + 1)
            if j == -1:
                out.append(line[i:])
                break
            out.append(" " * (j - i + 1))
            i = j + 1
        else:
            out.append(line[i])
            i += 1
    return "".join(out)


def strip_link_targets(line: str) -> str:
    """Remove the `(target)` half of markdown links — keep the `[text]`.

    `[Lesson 12](lesson-12-foo.md)` becomes `[Lesson 12]                `.
    This lets FORWARD_REF check link text without false-positiving on
    file paths.
    """
    return re.sub(r"\]\([^)]*\)", lambda m: "]" + " " * (len(m.group(0)) - 1), line)


# ─── Rules ───────────────────────────────────────────────────────────


_PHASE_N = re.compile(r"\bphase\s+\d+\b", re.IGNORECASE)


def check_phase_n(line: str) -> list[tuple[str, str | None]]:
    if _PHASE_N.search(line):
        return [
            (
                "Lesson prose refers to a build-phase number; describe by substance instead.",
                None,
            )
        ]
    return []


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


def check_real_email(line: str) -> list[tuple[str, str | None]]:
    findings: list[tuple[str, str | None]] = []
    for m in _EMAIL.finditer(line):
        domain = m.group(1).lower()
        if domain in RESERVED_EMAIL_DOMAINS:
            continue
        # Also allow any subdomain of an RFC-reserved root (e.g. mail.example.com).
        parts = domain.split(".")
        root = ".".join(parts[-2:]) if len(parts) >= 2 else domain
        tld = parts[-1] if parts else ""
        if root in RESERVED_EMAIL_DOMAINS or tld in RESERVED_EMAIL_DOMAINS:
            continue
        findings.append(
            (
                f"Real-looking email '{m.group(0)}' — use alice@example.com.",
                f"replace {m.group(0)} with alice@example.com",
            )
        )
    return findings


_BART = re.compile(r"\bbart\b", re.IGNORECASE)


def check_bart(line: str) -> list[tuple[str, str | None]]:
    if _BART.search(line):
        return [
            (
                "'Bart' appears in lesson text — use alice@example.com / a generic placeholder.",
                None,
            )
        ]
    return []


_LESSON_REF = re.compile(r"\bLesson\s+(\d+)\b")


def check_forward_ref(line: str, own_lesson: int | None) -> list[tuple[str, str | None]]:
    if own_lesson is None:
        return []
    # Strip link targets so we don't false-positive on lesson-NN-...md
    # file paths inside the (url) half of markdown links.
    scrubbed = strip_link_targets(line)
    findings: list[tuple[str, str | None]] = []
    for m in _LESSON_REF.finditer(scrubbed):
        ref = int(m.group(1))
        # NN+1 handoffs are allowed — every lesson footer points at
        # the next one by title. Anything two or more lessons ahead
        # risks rot when the course reshuffles.
        if ref >= own_lesson + 2:
            findings.append(
                (
                    f"Forward reference to Lesson {ref} from Lesson {own_lesson:02d}. "
                    "Use 'the next lesson' or 'when we cover X' instead.",
                    f"replace 'Lesson {ref}' with 'the next lesson' / 'a later lesson'",
                )
            )
    return findings


# Counts that bake in transient codebase/KB state. Each pattern matches
# whole tokens around a number with a unit/noun that implies "size of
# the project right now."
_COUNT_PATTERNS = [
    # ~16k chunks, ~110 tests, ~430 lines, ~9,800 LOC, ~9800 LOC
    (re.compile(r"~?\d{1,3}(?:,\d{3})*\s*(?:k|K)?\s+chunks?\b"), "chunk count"),
    (re.compile(r"~?\d+\s+tests?\b(?!\s+(?:pass|fail))"), "test count"),
    (re.compile(r"~?\d+\s+LOC\b"), "LOC count"),
    (re.compile(r"~?\d{2,}\s+lines?\s+of\s+code\b"), "lines-of-code count"),
    (re.compile(r"~?\d{3,}\s+(?:hermetic\s+)?pytests?\b"), "pytest count"),
]


# File:line citations like main.py:53 are exempt — they're stable
# pointers, not codebase-size claims. Also exempt: "lines 5-9" style
# ranges, "5 lines" of context (small, not project-size).
_LINE_RANGE = re.compile(r"\blines?\s+\d+(?:[-–]\d+)?\b", re.IGNORECASE)


def check_count(line: str) -> list[tuple[str, str | None]]:
    findings: list[tuple[str, str | None]] = []
    for pat, label in _COUNT_PATTERNS:
        for m in pat.finditer(line):
            # Skip "lines 5-9" style ranges that just happen to look
            # like a count.
            if _LINE_RANGE.fullmatch(m.group(0).strip()):
                continue
            findings.append(
                (
                    f"Specific {label} ('{m.group(0).strip()}') bakes in transient state. "
                    "Use ballpark language ('a few thousand', 'hundreds of').",
                    None,
                )
            )
    return findings


# ─── Driver ──────────────────────────────────────────────────────────


def check_file(path: str) -> list[Finding]:
    findings: list[Finding] = []
    own = lesson_number_from_filename(path)
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw in iter_prose_lines(text):
        scrubbed = strip_inline_code(raw)
        if not scrubbed.strip():
            continue
        excerpt = raw.strip()
        for rule, fn in (
            ("PHASE_N", lambda line: check_phase_n(line)),
            ("REAL_EMAIL", lambda line: check_real_email(line)),
            ("BART", lambda line: check_bart(line)),
            ("FORWARD_REF", lambda line: check_forward_ref(line, own)),
            ("COUNT", lambda line: check_count(line)),
        ):
            for message, fix in fn(scrubbed):
                findings.append(
                    Finding(
                        file=path,
                        line=lineno,
                        rule=rule,
                        message=message,
                        excerpt=excerpt,
                        fix=fix,
                    )
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="Lesson files to check (default: DOCS_GLOB).")
    parser.add_argument("--quiet", action="store_true", help="Only print finding count.")
    parser.add_argument("--json", action="store_true", help="Print findings as JSON.")
    args = parser.parse_args()

    if args.files:
        targets: list[str] = []
        for arg in args.files:
            matches = glob.glob(arg)  # noqa: PTH207
            targets.extend(matches if matches else [arg])
    else:
        targets = sorted(glob.glob(DOCS_GLOB))  # noqa: PTH207

    if not targets:
        print(f"No files matched (looked for: {args.files or DOCS_GLOB})", file=sys.stderr)
        return 0

    all_findings: list[Finding] = []
    for path in targets:
        if not Path(path).is_file():
            print(f"skip: {path} (not a file)", file=sys.stderr)
            continue
        all_findings.extend(check_file(path))

    if args.json:
        print(json.dumps([asdict(f) for f in all_findings], indent=2))
        return 0 if not all_findings else 1

    if not args.quiet:
        for f in all_findings:
            print(f"{f.file}:{f.line}: {f.rule}: {f.message}")
            print(f"    {f.excerpt}")
            if f.fix:
                print(f"    fix: {f.fix}")
            print()

    print(f"{len(all_findings)} finding(s) across {len(targets)} file(s)")
    return 0 if not all_findings else 1


if __name__ == "__main__":
    sys.exit(main())
