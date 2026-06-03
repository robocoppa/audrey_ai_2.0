#!/usr/bin/env python3
"""Audit Markdown line-cites in long-form docs against the current source.

WHAT IT CHECKS

For every Markdown link of the form

    [<label>](relative/path#L42)

or

    [<label>](relative/path#L42-L51)

in your docs, the script opens the target file and verifies the cite.
Three strategies, in order of strength:

  1. **Snippet match (strongest).** When the cite is followed shortly
     by a fenced code block in the same doc, the script treats that
     block's first code line as the canonical anchor — the line the
     prose is showing the reader. If the cited line number doesn't
     match where that first line actually lives in the file, the script
     prints the correct line so you can fix the cite in place.

  2. **Label-symbol anchor.** When the cite has no snippet but its
     *visible label* is a code identifier — `` [`require_user`](...#L126) ``,
     `` [`Class.method`](...) `` — that symbol is a content anchor. The
     script finds where the symbol is *defined* (def/class/assignment)
     and checks the cite points there. This closes the landmark
     heuristic's blind spot: a cite that drifted to a different line
     which still *looks* load-bearing (another `def`) passes the shape
     check but fails here, because the named symbol isn't at the cited
     line. Confident DRIFT with a proposed fix.

  3. **Landmark heuristic (weakest fallback).** When there's neither a
     snippet nor a label symbol, the script falls back to checking
     whether the cited line "looks like a landmark" — a
     def/class/constant/decorator/YAML key. This catches gross drift
     but yields false positives on deliberate "into the body" cites
     (an `if`, a `raise`, a field assignment); in fallback mode it only
     emits a soft `DRIFT?`, never a confident fix. The bare `file.py:NN`
     cites that carry no symbol in their label land here.

WHAT IT DOES NOT DO

- Validate the prose around the cite.
- Rewrite cites. It only reports; the human applies the fix.
- Track cite history. There is no database; every run starts cold.

USAGE

    scripts/check-lesson-links.py
        Check every cite in every doc file under DOCS_GLOB.

    scripts/check-lesson-links.py path/to/changed.py [more.py ...]
        Only report cites that target one of the given paths. Use as
        a pre-commit / post-edit step: pass the files you just edited
        and it tells you which docs need an update.

    scripts/check-lesson-links.py --list-only
        Print every (doc, cite, target) tuple without checking. Useful
        for building a one-time CSV index.

ENVIRONMENT (override if your repo is shaped differently)

    DOCS_GLOB    Glob for the docs to audit. Default: docs/lessons/*.md
                 Example: DOCS_GLOB="docs/**/*.md" check-lesson-links.py
    REPO_ROOT    Path the cite URLs are relative to. Default: the
                 enclosing git repo root. Cites in lessons typically use
                 "../../" to escape the lesson directory; the script
                 resolves each cite relative to the lesson file.

EXIT CODES

    0   No drift found (or only soft "drift?" hints in fallback mode).
    1   At least one cite is broken or has a confident fix proposal.
    2   Usage error / missing dependency.

PROJECT-AGNOSTIC NOTES

The script makes no assumption about your codebase. The only knobs
are DOCS_GLOB (where the docs are) and the cite syntax (Markdown link
with #L<num> fragment, optionally a #L<a>-L<b> range). The landmark
patterns are tuned for Python/YAML/shell/markdown sources; add more
in LANDMARK_PATTERNS below to cover other languages.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────

DOCS_GLOB = os.environ.get("DOCS_GLOB", "docs/lessons/lesson-*.md")

# Docs to skip even when they match DOCS_GLOB. Useful for excluding
# scaffolding files (e.g. AUDIT.md) that contain illustrative cites the
# checker shouldn't validate. Set DOCS_EXCLUDE to a colon-separated list
# of glob patterns.
DOCS_EXCLUDE = [
    p for p in os.environ.get("DOCS_EXCLUDE", "").split(":") if p
]

# How far past a cite to look for a fenced code block before giving up.
# Cites in well-formed lessons land within 1-4 lines of the block.
SNIPPET_LOOKAHEAD = 6

# When the displayed snippet's first line doesn't appear verbatim in
# the target file, try shorter prefixes of that line. Stops at this
# minimum length to avoid matching every "if " in a file.
MIN_SNIPPET_PREFIX = 12

# When multiple matches for the same snippet exist in the file, prefer
# the one closest to the original cited line. This keeps re-positioning
# stable when the cited block is a common pattern (e.g. `def __init__`).
PREFER_NEAREST = True

# When the snippet is found within this many lines of the cited line,
# accept the cite as OK rather than propose a fix. Common case: the
# cite points at a function signature and the displayed snippet shows
# the function body, which lives a few lines later. The reader opens
# the cite and the snippet is right there — no real drift.
NEAR_CITE_RANGE = 10

# Patterns that mark "load-bearing" lines for the fallback heuristic.
# Each is a Python regex applied with `re.match` (anchored at line start
# minus leading whitespace). Add patterns for new languages here.
LANDMARK_PATTERNS = [
    r"def\s",                       # python def
    r"async\s+def\s",               # python async def
    r"class\s",                     # python class
    r"@",                           # python decorator
    r"[A-Z_][A-Z0-9_]*\s*[:=]",     # ALL_CAPS = / ALL_CAPS:
    r"[a-z_][a-zA-Z0-9_]*\s*=\s*\w",  # name = expr (top-level binding)
    r"#\s*[─=]{3,}",                # box-drawing section header
    r"#\s*[A-Z]",                   # comment starting with a capital
    r"function\s",                  # bash function keyword
    r"[A-Za-z0-9_]+\(\)\s*\{",      # bash function (POSIX style)
]
# Anchored landmark regex — strips leading whitespace before matching.
LANDMARK_RE = re.compile(r"^\s*(?:" + "|".join(LANDMARK_PATTERNS) + r")")

# YAML keys: a separate check because they have their own shape.
YAML_KEY_RE = re.compile(r"^\s*[A-Za-z0-9_-]+:\s*(#.*)?$")

# Cite extractor: matches a Markdown link whose URL has a #L<num>
# anchor, optionally a range. Captures the label text and the URL.
CITE_RE = re.compile(r"\[([^\]]+)\]\(([^)]*#L\d+(?:-L\d+)?)\)")

# Identifier extractor for the link label. When a cite's visible label is
# a backticked code symbol — `require_user`, `ChatArchiveStore.archive_turn`,
# `get_config()` — that symbol is a content anchor we can verify against the
# cited line even when there's no fenced snippet. We take the last
# dotted/parenthesized component (the method/function name) since that's
# what appears at a `def`/`class` line. Labels that are file:line
# (`app.py:117`) or plain prose yield no symbol and fall through.
_LABEL_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)(?:\(\))?`")


# ─── Data types ──────────────────────────────────────────────────────


@dataclass
class Cite:
    """One Markdown line-cite extracted from a doc."""
    doc: Path           # the .md file the cite came from
    doc_line: int       # the doc line the cite appears on (for reporting)
    url: str            # the raw URL from the link
    target: Path        # resolved absolute path to the cited file
    start: int          # cited line number (start of range)
    end: int            # cited line number (end of range; == start if not a range)
    snippet: str | None  # first code line of the following fenced block, if any
    label_symbol: str | None  # code identifier from the link label, if any


@dataclass
class Finding:
    cite: Cite
    severity: str       # "BROKEN" | "DRIFT" | "DRIFT?" | "OK"
    message: str
    proposed_line: int | None  # when severity == DRIFT and we have a confident fix


# ─── Resolution ──────────────────────────────────────────────────────


def repo_root() -> Path:
    """Locate REPO_ROOT — env var, then `git rev-parse`, then bail."""
    env = os.environ.get("REPO_ROOT")
    if env:
        return Path(env).resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL,
        )
        return Path(out.decode().strip()).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.stderr.write("error: REPO_ROOT not set and not in a git repo\n")
        sys.exit(2)


def resolve_cite_path(doc: Path, rel: str) -> Path:
    """Resolve a cite URL relative to the citing doc's directory.

    Markdown links are resolved relative to the file they appear in, so
    a lesson at `docs/lessons/foo.md` citing `../../src/x.py` resolves
    to `<repo>/src/x.py`. This mirrors how a Markdown viewer follows
    the link.
    """
    return (doc.parent / rel).resolve()


def parse_anchor(anchor: str) -> tuple[int, int]:
    """Parse an `#L42` or `#L42-L51` anchor into (start, end) line nums.

    Anchor comes in without the leading `#`.
    """
    # anchor: "L42" or "L42-L51"
    if "-L" in anchor:
        a, b = anchor.split("-L", 1)
        return int(a[1:]), int(b)
    return int(anchor[1:]), int(anchor[1:])


# ─── Doc parsing ─────────────────────────────────────────────────────


def extract_cites(doc: Path) -> Iterable[Cite]:
    """Walk the doc line by line, emit one Cite per `#L<n>` link.

    The snippet is set when the cite is followed (within
    SNIPPET_LOOKAHEAD lines) by an opening code fence (```), in which
    case the next non-empty content line is captured as the snippet's
    first line. Snippet capture is conservative — we never look across
    a blank-line-then-prose gap, so cites with no nearby snippet just
    get snippet=None and fall through to the landmark heuristic.
    """
    lines = doc.read_text().splitlines()
    for i, line in enumerate(lines):
        for match in CITE_RE.finditer(line):
            label, url = match.group(1), match.group(2)
            rel_path, anchor = url.split("#", 1)
            target = resolve_cite_path(doc, rel_path)
            start, end = parse_anchor(anchor)
            snippet = _find_following_snippet(lines, i)
            yield Cite(
                doc=doc, doc_line=i + 1, url=url, target=target,
                start=start, end=end, snippet=snippet,
                label_symbol=_extract_label_symbol(label),
            )


def _extract_label_symbol(label: str) -> str | None:
    """Pull a verifiable code identifier out of a link label, or None.

    `require_user`              → "require_user"
    `ChatArchiveStore.archive_turn` → "archive_turn"  (last component)
    `get_config()`             → "get_config"
    `app.py:117`               → None  (file:line, not a symbol)
    plain prose                → None

    We take the trailing dotted component because that's the name that
    appears at the `def`/`class` line the cite points to — for
    `Class.method` the method is what's defined there.
    """
    m = _LABEL_SYMBOL_RE.fullmatch(label.strip())
    if not m:
        return None
    ident = m.group(1)
    # A file basename like `app.py` matches the identifier shape; reject
    # anything that looks like a filename (has a known source suffix).
    last = ident.split(".")[-1]
    if last in {"py", "yaml", "yml", "sh", "md", "txt"}:
        return None
    return last


def _find_following_snippet(lines: list[str], cite_line_idx: int) -> str | None:
    """Find the first code line of the fenced block following a cite.

    Returns None when:
      - no fenced block opens within SNIPPET_LOOKAHEAD lines, or
      - the fenced block's language tag indicates non-source content
        (``` ``text``, ``` ``mermaid``, ``` ``diagram``, ``` ``markdown``)
        — those are pseudocode / diagrams, not snippets the script
        should look for in the target file, or
      - the first content line is too generic to be a useful anchor
        (e.g. `\"\"\"` opening a docstring). We advance past those lines
        to find a substantive first content line, falling back to None
        if nothing substantive appears in the first ~6 lines of the
        block.

    The "first line inside the fence" is trimmed of trailing whitespace
    only; leading whitespace is preserved so the match logic can
    normalize for indentation.
    """
    end = min(cite_line_idx + SNIPPET_LOOKAHEAD + 1, len(lines))
    for j in range(cite_line_idx + 1, end):
        fence_line = lines[j].lstrip()
        # If another cite appears before any fence, the upcoming fence
        # belongs to *that* cite, not this one. Stop the search so we
        # don't poach a later cite's snippet. Common in list-style
        # walkthroughs: "1. X at file:1 / 2. Y at file:2 / ```snippet```"
        # — the snippet belongs to step 2, not step 1.
        if CITE_RE.search(lines[j]):
            return None
        if not fence_line.startswith("```"):
            continue
        # Read the language tag (everything after ```). Empty tag means
        # "plain" — could be code or could be prose; treat as ambiguous
        # and try to use it (covers untagged python snippets).
        tag = fence_line[3:].strip().lower()
        if tag in _NON_SOURCE_FENCE_TAGS:
            return None
        # Scan inside the fence for a useful first line.
        for k in range(j + 1, min(j + 1 + _SNIPPET_FIRST_LINE_SCAN, len(lines))):
            content = lines[k].rstrip()
            if content.lstrip().startswith("```"):
                return None  # empty/closed fence — no usable snippet
            stripped = content.strip()
            if not stripped:
                continue
            if stripped in _GENERIC_SNIPPET_OPENERS:
                continue  # `"""` etc — keep scanning for substance
            return content
        return None
    return None


# Fence language tags whose contents are not source code. Cites
# followed by these should fall back to the landmark heuristic.
_NON_SOURCE_FENCE_TAGS: frozenset[str] = frozenset({
    "text", "txt", "plain",
    "mermaid", "diagram", "graphviz", "dot",
    "markdown", "md",
    "ascii",
})

# Snippet first lines that are too generic to anchor on. Triple quotes
# open and close every docstring; matching them tells us nothing about
# *which* docstring the snippet meant.
_GENERIC_SNIPPET_OPENERS: frozenset[str] = frozenset({
    '"""', "'''",
})

# When scanning inside a fence for a substantive first line, stop after
# this many lines so we don't drag through long preamble.
_SNIPPET_FIRST_LINE_SCAN = 6


# ─── Checking ────────────────────────────────────────────────────────


def check_cite(cite: Cite) -> Finding:
    """Return a Finding describing whether the cite still points correctly."""
    if not cite.target.exists():
        return Finding(cite, "BROKEN",
                       f"target file not found: {cite.target}", None)
    try:
        text = cite.target.read_text().splitlines()
    except OSError as e:
        return Finding(cite, "BROKEN", f"unreadable: {e}", None)
    line_count = len(text)
    if cite.start > line_count or cite.start < 1:
        return Finding(
            cite, "BROKEN",
            f"line {cite.start} past end of file ({line_count} lines)", None,
        )

    cited_text = text[cite.start - 1]

    # Strategy 1: snippet match. If we know what the displayed snippet's
    # first line is, that's a much stronger anchor than "is this a
    # landmark." We try the full first line, then progressively
    # shorter prefixes, until we either match or give up.
    #
    # We compare with leading whitespace stripped on both sides. Lessons
    # often un-indent the snippet (showing top-of-block code from a
    # nested context as if it were top-level), and `startswith` against
    # a left-stripped source line gives the right semantics: "does the
    # file at the cited line begin with this content, regardless of
    # indentation?"
    if cite.snippet is not None:
        snippet_norm = cite.snippet.strip()
        if cited_text.lstrip().rstrip() == snippet_norm:
            return Finding(cite, "OK", "snippet matches cited line", None)
        # Snippet didn't match at the cited line. Look for it elsewhere
        # in the file using progressively shorter prefixes.
        found = _find_snippet_line(text, cite.snippet, cite.start)
        if found is not None:
            if found == cite.start:
                # Match landed exactly where the cite already points —
                # this happens when the snippet's prefix matched the
                # line but its full first line didn't (e.g. truncated
                # `await classify_fn(...)` vs full call). Treat as OK.
                return Finding(cite, "OK", "snippet prefix matches cited line", None)
            # "Cite is in the neighborhood" — common when the cite
            # points at a function signature and the snippet shows the
            # function body. Accept as OK; the reader opens the cite
            # and the snippet is right there.
            if abs(found - cite.start) <= NEAR_CITE_RANGE:
                return Finding(
                    cite, "OK",
                    f"snippet at line {found} is within {NEAR_CITE_RANGE} of cited line {cite.start}",
                    None,
                )
            return Finding(
                cite, "DRIFT",
                f"snippet found at line {found}, not {cite.start}",
                proposed_line=found,
            )
        # Last resort: maybe the snippet just wasn't a good anchor (a
        # generic line that no longer exists, or a paraphrase). Fall
        # back to landmark on the cited line so we don't over-report.
        # If the cited line still looks load-bearing, treat as a soft
        # `DRIFT?` rather than a confident `DRIFT`.
        if _is_landmark(cited_text):
            return Finding(
                cite, "DRIFT?",
                f"snippet not found in file but cited line is a landmark: "
                f"{cited_text.strip()!r}",
                None,
            )
        return Finding(
            cite, "DRIFT",
            f"snippet not found in file; cited line {cite.start} reads: "
            f"{cited_text.strip()!r}",
            proposed_line=None,
        )

    # Strategy 2: label-symbol anchor. When the cite's visible label is a
    # code identifier (`require_user`, `Class.method`), it's a real content
    # anchor even without a fenced snippet. Verify the symbol is *defined*
    # at or near the cited line. This catches the landmark heuristic's
    # blind spot: a cite that drifted to a different-but-still-landmark
    # line (e.g. `def some_other_fn`) — the shape check passes, but the
    # symbol won't be there.
    if cite.label_symbol is not None:
        defined_at = _find_definition_line(text, cite.label_symbol, cite.start)
        if defined_at is not None:
            if abs(defined_at - cite.start) <= NEAR_CITE_RANGE:
                return Finding(cite, "OK",
                               f"label symbol {cite.label_symbol!r} defined near cited line", None)
            return Finding(
                cite, "DRIFT",
                f"label symbol {cite.label_symbol!r} is defined at line "
                f"{defined_at}, not near cited line {cite.start}",
                proposed_line=defined_at,
            )
        # Symbol not defined anywhere — either removed (real drift) or the
        # cite points into a call site rather than a definition. If the
        # cited line itself mentions the symbol, accept; else flag soft.
        if cite.label_symbol in cited_text:
            return Finding(cite, "OK",
                           f"label symbol {cite.label_symbol!r} on cited line", None)
        return Finding(
            cite, "DRIFT?",
            f"label symbol {cite.label_symbol!r} not found at/near cited line "
            f"{cite.start} (reads: {cited_text.strip()!r})",
            None,
        )

    # Strategy 3: landmark heuristic. No snippet and no label symbol to
    # anchor against, so all we can do is judge whether the cited line
    # "looks load-bearing" — a function/class/constant/etc. False
    # positives happen for deliberate "into the body" cites, hence
    # "DRIFT?" (with the question mark) rather than confident "DRIFT".
    if _is_landmark(cited_text):
        return Finding(cite, "OK", "landmark heuristic passed", None)
    return Finding(
        cite, "DRIFT?",
        f"cited line is not a landmark: {cited_text.strip()!r}",
        None,
    )


def _find_definition_line(
    text: list[str], symbol: str, near: int,
) -> int | None:
    """Locate where `symbol` is defined: `def symbol`, `class symbol`,
    `async def symbol`, or `symbol =` / `symbol:` (constant/field).

    Returns the 1-indexed line nearest to `near`, or None if the symbol
    is defined nowhere. A definition is a much stronger signal than a
    bare mention — we only want the line that *introduces* the symbol,
    not every call site, so the cite resolves to the right place.
    """
    pat = re.compile(
        r"^\s*(?:async\s+def\s+|def\s+|class\s+)"
        + re.escape(symbol)
        + r"\b"
        + r"|^\s*" + re.escape(symbol) + r"\s*[:=]"
    )
    candidates = [i + 1 for i, line in enumerate(text) if pat.match(line)]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - near))


def _is_landmark(line: str) -> bool:
    """Return True iff `line` matches one of the landmark patterns."""
    if LANDMARK_RE.match(line):
        return True
    if YAML_KEY_RE.match(line):
        return True
    return False


def _find_snippet_line(
    text: list[str], snippet: str, near: int,
) -> int | None:
    """Locate `snippet` (or a long-enough prefix of it) in `text`.

    Tries the full snippet first, then shorter prefixes down to
    MIN_SNIPPET_PREFIX chars. Returns 1-indexed line numbers.

    When multiple matches exist and PREFER_NEAREST is set, returns
    the match closest to `near`. This biases corrections toward "the
    block moved slightly" rather than "a totally different block in
    the file happens to start the same way".
    """
    snippet = snippet.strip()
    # Try full snippet down to MIN_SNIPPET_PREFIX-char prefix — but
    # when the whole snippet is shorter than that, use the snippet's
    # full length as the floor. Otherwise short anchors like `try:` or
    # `return x` would never get tried.
    # We compare with leading whitespace stripped on the source side
    # too so an indented `async def node_classify(...)` matches an
    # un-indented snippet.
    floor = min(MIN_SNIPPET_PREFIX, len(snippet))
    for length in range(len(snippet), floor - 1, -1):
        needle = snippet[:length].rstrip()
        if not needle:
            continue
        candidates = [
            i + 1 for i, line in enumerate(text)
            if line.lstrip().startswith(needle)
        ]
        if not candidates:
            continue
        if PREFER_NEAREST:
            return min(candidates, key=lambda x: abs(x - near))
        return candidates[0]
    return None


# ─── Reporting ───────────────────────────────────────────────────────


def report(finding: Finding) -> str:
    c = finding.cite
    head = f"{finding.severity:7s} {c.doc}:{c.doc_line}\n"
    body = f"  cite: {c.url}\n"
    body += f"  detail: {finding.message}\n"
    if finding.proposed_line is not None:
        suggested_anchor = f"L{finding.proposed_line}"
        if c.end != c.start:
            span = c.end - c.start
            suggested_anchor += f"-L{finding.proposed_line + span}"
        body += f"  fix: change #L{c.start}"
        if c.end != c.start:
            body += f"-L{c.end}"
        body += f" → #{suggested_anchor}\n"
    return head + body


# ─── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Audit Markdown line-cites against current source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "paths", nargs="*",
        help="Only report cites whose target is one of these files. "
             "When omitted, all cites are reported.",
    )
    ap.add_argument(
        "--list-only", action="store_true",
        help="Print (doc, cite, target) tuples without checking.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    os.chdir(root)

    docs = sorted(Path().glob(DOCS_GLOB))
    if DOCS_EXCLUDE:
        # Each exclude is a glob; drop any doc whose path matches.
        excluded = set()
        for pattern in DOCS_EXCLUDE:
            excluded.update(Path().glob(pattern))
        docs = [d for d in docs if d not in excluded]
    if not docs:
        sys.stderr.write(f"no docs found matching: {DOCS_GLOB}\n")
        return 0

    # Filter target set, if the user passed paths.
    filter_set: set[Path] | None = None
    if args.paths:
        filter_set = {Path(p).resolve() for p in args.paths}

    findings: list[Finding] = []
    for doc in docs:
        for cite in extract_cites(doc):
            if filter_set is not None and cite.target not in filter_set:
                continue
            if args.list_only:
                print(f"{doc}\t{cite.url}\t{cite.target}")
                continue
            findings.append(check_cite(cite))

    if args.list_only:
        return 0

    # Print findings in severity order; group BROKEN + DRIFT (real
    # problems) before DRIFT? (heuristic noise) and OK (silent).
    severity_rank = {"BROKEN": 0, "DRIFT": 1, "DRIFT?": 2, "OK": 3}
    findings.sort(key=lambda f: (severity_rank[f.severity], str(f.cite.doc)))

    counts = {"OK": 0, "BROKEN": 0, "DRIFT": 0, "DRIFT?": 0}
    for f in findings:
        counts[f.severity] += 1
        if f.severity != "OK":
            print(report(f))

    print("─────────────────────────────────────────")
    print(
        f"cites checked: {sum(counts.values())}  "
        f"ok: {counts['OK']}  drift: {counts['DRIFT']}  "
        f"drift?: {counts['DRIFT?']}  broken: {counts['BROKEN']}"
    )
    # Confident problems fail the run; "drift?" hints don't.
    return 1 if counts["BROKEN"] + counts["DRIFT"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
