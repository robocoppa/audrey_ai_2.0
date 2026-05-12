"""Smoke tests for scripts/check-lesson-links.py.

Builds synthetic lesson/source fixtures in `tmp_path` and runs the
checker via subprocess. Each test pins one behavior end-to-end:
extract → resolve → match → report → exit code.

The script is a side-tool, not part of the Audrey runtime, so we run
it as an external process to keep the test honest about the actual
CLI contract (DOCS_GLOB, REPO_ROOT, exit codes, stdout shape).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check-lesson-links.py"


# ─── Fixture helpers ──────────────────────────────────────────────────


def _write(path: Path, content: str) -> None:
    """Create parent dirs and write `content` with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content.endswith("\n") else content + "\n")


def _run(repo: Path, *args: str, docs_glob: str = "docs/lessons/*.md") -> subprocess.CompletedProcess:
    """Run the checker against `repo` with the given CLI args.

    `repo` is treated as REPO_ROOT. `docs_glob` overrides the default
    glob; tests use the permissive `docs/lessons/*.md` so they can use
    any filename. The script's production default is tighter
    (`lesson-*.md`) to exclude scaffolding files.
    """
    env = {
        **os.environ,
        "REPO_ROOT": str(repo),
        "DOCS_GLOB": docs_glob,
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _setup_minimal_repo(tmp: Path) -> Path:
    """Make `tmp` look enough like a repo for the script to operate.

    Creates the lessons dir + a single src file. Caller adds whatever
    cites they want. Skips git init — the script falls back to
    REPO_ROOT env when not in a git repo.
    """
    (tmp / "docs" / "lessons").mkdir(parents=True)
    (tmp / "src").mkdir()
    return tmp


# ─── Snippet-match path ───────────────────────────────────────────────


def test_clean_cite_with_matching_snippet_exits_0(tmp_path):
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "# header\n\ndef hello():\n    return 'hi'\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/lib.py:3`](../../src/lib.py#L3):\n\n"
        "```python\n"
        "def hello():\n"
        "    return 'hi'\n"
        "```\n",
    )
    proc = _run(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok: 1" in proc.stdout


def test_drifted_cite_proposes_correct_line(tmp_path):
    """Snippet moved far enough that NEAR_CITE_RANGE tolerance can't
    excuse the drift — checker should propose the new line."""
    repo = _setup_minimal_repo(tmp_path)
    # 20 padding lines push the snippet well past NEAR_CITE_RANGE.
    pad = "\n".join(f"# pad {i}" for i in range(20))
    _write(
        repo / "src" / "lib.py",
        f"{pad}\n\ndef hello():\n    return 'hi'\n",
    )
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/lib.py:3`](../../src/lib.py#L3):\n\n"
        "```python\n"
        "def hello():\n"
        "    return 'hi'\n"
        "```\n",
    )
    proc = _run(repo)
    assert proc.returncode == 1
    assert "DRIFT" in proc.stdout
    # def hello is now at line 22 (20 pad lines + blank + def).
    assert "snippet found at line 22" in proc.stdout, proc.stdout
    assert "fix: change #L3 → #L22" in proc.stdout, proc.stdout


def test_drifted_cite_with_indentation_difference(tmp_path):
    """Source is indented (nested function); snippet is un-indented.

    Lessons often un-indent code from nested contexts. The checker must
    tolerate that — comparing on left-stripped content, not verbatim.
    """
    repo = _setup_minimal_repo(tmp_path)
    _write(
        repo / "src" / "lib.py",
        "def outer():\n    def inner():\n        return 42\n",
    )
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "See [`src/lib.py:2`](../../src/lib.py#L2):\n\n"
        "```python\n"
        "def inner():\n"
        "    return 42\n"
        "```\n",
    )
    proc = _run(repo)
    assert proc.returncode == 0, proc.stdout
    assert "ok: 1" in proc.stdout


def test_snippet_not_found_anywhere_reports_drift(tmp_path):
    """When the snippet's first line was renamed/removed AND the cited
    line is no longer a landmark, the checker emits a confident DRIFT
    with no fix proposal."""
    repo = _setup_minimal_repo(tmp_path)
    # Cited line 1 is a non-landmark expression — bare arithmetic mid-
    # function. No def/class/CONST shape.
    _write(repo / "src" / "lib.py", "    return 1 + 2 + 3\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/lib.py:1`](../../src/lib.py#L1):\n\n"
        "```python\n"
        "def hello():\n"
        "    return 'hi'\n"
        "```\n",
    )
    proc = _run(repo)
    assert proc.returncode == 1, proc.stdout
    assert "DRIFT" in proc.stdout
    assert "snippet not found" in proc.stdout
    # No fix proposal when we couldn't locate the snippet.
    assert "fix:" not in proc.stdout


def test_snippet_not_found_but_cited_line_is_landmark_is_softer(tmp_path):
    """When the snippet's first line is missing but the cited line still
    looks load-bearing, the checker emits the softer DRIFT? rather than
    confident DRIFT — the cite is probably still correct, the snippet
    is paraphrased or stale."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def helper():\n    return 1\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/lib.py:1`](../../src/lib.py#L1):\n\n"
        "```python\n"
        "totally unrelated paraphrased text\n"
        "```\n",
    )
    proc = _run(repo)
    # DRIFT? is advisory only; exit 0.
    assert proc.returncode == 0, proc.stdout
    assert "DRIFT?" in proc.stdout
    assert "drift?: 1" in proc.stdout


def test_prefer_nearest_match_when_multiple(tmp_path):
    """Duplicate snippet shape — when both candidates are outside
    NEAR_CITE_RANGE, checker proposes the one closer to the cite.
    Stabilizes corrections when the same pattern (e.g.
    `def __init__(self):`) appears in many places."""
    repo = _setup_minimal_repo(tmp_path)
    pad_top = "\n".join("# top" for _ in range(30))
    pad_mid = "\n".join("# mid" for _ in range(40))
    _write(
        repo / "src" / "lib.py",
        f"{pad_top}\n\n"
        "class A:\n    def __init__(self):\n        pass\n"
        f"\n{pad_mid}\n\n"
        "class B:\n    def __init__(self):\n        pass\n",
    )
    # Source layout:
    #   1..30  # top
    #   31  (blank)
    #   32  class A:
    #   33      def __init__(self):    ← A's init
    #   34          pass
    #   35  (blank)
    #   36..75 # mid
    #   76  (blank)
    #   77  class B:
    #   78      def __init__(self):    ← B's init
    #   79          pass
    # Cite at line 60 (in the middle pad). Both inits are outside
    # NEAR_CITE_RANGE=10. Distance from 60: A's at 33 → 27, B's at
    # 78 → 18. Nearest is B's at 78.
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "B's init is at [`src/lib.py:60`](../../src/lib.py#L60):\n\n"
        "```python\n"
        "def __init__(self):\n"
        "    pass\n"
        "```\n",
    )
    proc = _run(repo)
    assert "snippet found at line 78" in proc.stdout, proc.stdout
    assert "fix: change #L60 → #L78" in proc.stdout, proc.stdout


# ─── Range cites ──────────────────────────────────────────────────────


def test_range_cite_proposes_shifted_range(tmp_path):
    """Range cite drifts far enough that NEAR_CITE_RANGE can't excuse
    it — proposal preserves the span."""
    repo = _setup_minimal_repo(tmp_path)
    pad = "\n".join(f"# pad {i}" for i in range(25))
    _write(
        repo / "src" / "lib.py",
        f"{pad}\n\ndef hello():\n    a = 1\n    return a\n",
    )
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Block: [`src/lib.py:3-L5`](../../src/lib.py#L3-L5)\n\n"
        "```python\n"
        "def hello():\n"
        "    a = 1\n"
        "    return a\n"
        "```\n",
    )
    proc = _run(repo)
    assert proc.returncode == 1, proc.stdout
    # def hello is at line 27 (25 pad + blank + def). Span was 3..5
    # (2 lines), so the proposed end is 27 + 2 = 29.
    assert "fix: change #L3-L5 → #L27-L29" in proc.stdout, proc.stdout


# ─── Broken cites ─────────────────────────────────────────────────────


def test_missing_target_file_is_broken(tmp_path):
    repo = _setup_minimal_repo(tmp_path)
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/missing.py:1`](../../src/missing.py#L1).\n",
    )
    proc = _run(repo)
    assert proc.returncode == 1
    assert "BROKEN" in proc.stdout
    assert "target file not found" in proc.stdout


def test_line_past_eof_is_broken(tmp_path):
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "tiny.py", "x = 1\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/tiny.py:99`](../../src/tiny.py#L99).\n",
    )
    proc = _run(repo)
    assert proc.returncode == 1
    assert "BROKEN" in proc.stdout
    assert "past end of file" in proc.stdout


# ─── Landmark fallback (no snippet) ───────────────────────────────────


def test_inline_cite_landmark_def_is_ok(tmp_path):
    """Cite without a following ``` block — falls back to landmark
    heuristic. A `def` line passes."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def hello():\n    return 1\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "See `hello()` at [`src/lib.py:1`](../../src/lib.py#L1) for details.\n",
    )
    proc = _run(repo)
    assert proc.returncode == 0
    assert "ok: 1" in proc.stdout


def test_inline_cite_non_landmark_is_drift_q(tmp_path):
    """Cite into the middle of an expression — `DRIFT?` advisory."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def hello():\n    return 1 + 2\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Return at [`src/lib.py:2`](../../src/lib.py#L2).\n",
    )
    proc = _run(repo)
    # DRIFT? is advisory only; exit 0.
    assert proc.returncode == 0, proc.stdout
    assert "DRIFT?" in proc.stdout
    assert "drift?: 1" in proc.stdout


# ─── Filter mode ─────────────────────────────────────────────────────


def test_filter_mode_only_checks_targeted_files(tmp_path):
    """Pass a path; cites pointing elsewhere are filtered out."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "a.py", "def fn_a():\n    pass\n")
    _write(repo / "src" / "b.py", "def fn_b():\n    pass\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "A: [`src/a.py:1`](../../src/a.py#L1)\n\n"
        "```python\ndef fn_a():\n```\n\n"
        "B: [`src/b.py:1`](../../src/b.py#L1)\n\n"
        "```python\ndef fn_b():\n```\n",
    )
    proc = _run(repo, "src/a.py")
    # Only one cite should be checked (cites checked: 1).
    assert "cites checked: 1" in proc.stdout, proc.stdout
    assert proc.returncode == 0


def test_filter_mode_accepts_absolute_paths(tmp_path):
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "a.py", "def fn_a():\n    pass\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "A: [`src/a.py:1`](../../src/a.py#L1)\n\n"
        "```python\ndef fn_a():\n```\n",
    )
    abs_path = str((repo / "src" / "a.py").resolve())
    proc = _run(repo, abs_path)
    assert "cites checked: 1" in proc.stdout


# ─── List-only mode ──────────────────────────────────────────────────


def test_list_only_prints_tuples_and_exits_0(tmp_path):
    """--list-only emits one `doc\\tcite\\ttarget` line per cite, exit 0."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def hello(): pass\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "[`src/lib.py:1`](../../src/lib.py#L1)\n",
    )
    proc = _run(repo, "--list-only")
    assert proc.returncode == 0
    assert "\t" in proc.stdout
    # One tab-separated row per cite.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    doc, url, target = lines[0].split("\t")
    assert doc.endswith("01.md")
    assert url == "../../src/lib.py#L1"
    assert target.endswith("/src/lib.py")


# ─── No-docs / empty-glob path ───────────────────────────────────────


def test_no_docs_found_exits_0_with_stderr(tmp_path):
    """Empty glob is a no-op, not an error. Saves CI from failing
    when the repo has no docs yet."""
    repo = _setup_minimal_repo(tmp_path)
    # Remove the lessons dir entirely.
    (repo / "docs" / "lessons").rmdir()
    proc = _run(repo)
    assert proc.returncode == 0
    assert "no docs found" in proc.stderr


# ─── Snippet capture edge cases ──────────────────────────────────────


def test_text_fence_block_is_skipped(tmp_path):
    """Cite followed by a ```text block (pseudocode/diagram, not code)
    should fall through to landmark mode — not try to find prose in the
    source file."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def classify():\n    pass\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "The function at [`src/lib.py:1`](../../src/lib.py#L1) decides:\n\n"
        "```text\n"
        "if strong keyword -> use\n"
        "else -> ask router\n"
        "```\n",
    )
    proc = _run(repo)
    # Should pass via landmark — cite at line 1 is `def classify():`.
    assert proc.returncode == 0, proc.stdout
    assert "ok: 1" in proc.stdout


def test_mermaid_fence_block_is_skipped(tmp_path):
    """Same as text fence — mermaid/diagram blocks aren't source."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def flow():\n    pass\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "[`src/lib.py:1`](../../src/lib.py#L1)\n\n"
        "```mermaid\n"
        "graph TD\n"
        "  A --> B\n"
        "```\n",
    )
    proc = _run(repo)
    assert proc.returncode == 0, proc.stdout


def test_docstring_opener_is_not_used_as_snippet(tmp_path):
    """Snippet starting with `\"\"\"` is too generic to be an anchor —
    every docstring would match. The script should advance past it to
    the next content line, or fall back to landmark."""
    repo = _setup_minimal_repo(tmp_path)
    _write(
        repo / "src" / "lib.py",
        'def hello():\n    """Greet."""\n    return 1\n',
    )
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/lib.py:1`](../../src/lib.py#L1):\n\n"
        "```python\n"
        '"""\n'
        "Greeting function.\n"
        '"""\n'
        "```\n",
    )
    proc = _run(repo)
    # The snippet's first line is `\"\"\"` — useless as an anchor.
    # Script should fall back to landmark (def hello at L1 passes).
    assert proc.returncode == 0, proc.stdout


def test_short_snippet_anchor_still_matched(tmp_path):
    """Snippet first line shorter than MIN_SNIPPET_PREFIX (e.g. `try:`).
    The script should still try to match it — using the snippet's full
    length as the floor when it's shorter than the default minimum."""
    repo = _setup_minimal_repo(tmp_path)
    pad = "\n".join(f"# pad {i}" for i in range(20))
    _write(
        repo / "src" / "lib.py",
        f"{pad}\n\ntry:\n    foo()\nexcept Exception:\n    pass\n",
    )
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/lib.py:5`](../../src/lib.py#L5):\n\n"
        "```python\n"
        "try:\n"
        "    foo()\n"
        "```\n",
    )
    proc = _run(repo)
    # `try:` is at line 22; cite at 5. Should propose 22.
    assert "snippet found at line 22" in proc.stdout, proc.stdout


def test_snippet_near_cite_passes(tmp_path):
    """Cite points at the function signature; snippet shows code from
    inside the function body. The snippet's first line lives a few
    lines after the cite. Script should accept this as "cite is in the
    neighborhood of the snippet" rather than flag drift."""
    repo = _setup_minimal_repo(tmp_path)
    _write(
        repo / "src" / "lib.py",
        "def helper():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n",
    )
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Open [`src/lib.py:1`](../../src/lib.py#L1) and look at the body:\n\n"
        "```python\n"
        "x = 1\n"
        "y = 2\n"
        "return x + y\n"
        "```\n",
    )
    proc = _run(repo)
    # Cite is at line 1 (def helper); snippet `x = 1` is at line 2.
    # Within neighborhood threshold — should pass.
    assert proc.returncode == 0, proc.stdout


def test_cite_followed_by_another_cite_does_not_poach_snippet(tmp_path):
    """In list-style prose, cite A is immediately followed by cite B,
    and the fenced block belongs to cite B (or a later step). Cite A
    must not claim cite B's snippet — it should fall through to the
    landmark heuristic instead."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def fn_a():\n    pass\n\ndef fn_b():\n    return 99\n")
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "1. Call A at [`src/lib.py:1`](../../src/lib.py#L1).\n"
        "2. Call B at [`src/lib.py:4`](../../src/lib.py#L4):\n\n"
        "```python\n"
        "def fn_b():\n"
        "    return 99\n"
        "```\n",
    )
    proc = _run(repo)
    # Both cites should pass:
    #   - Cite A passes via landmark (def fn_a at line 1).
    #   - Cite B passes via snippet (def fn_b matches at line 4).
    # The bug we're guarding against would have cite A grab cite B's
    # snippet and report DRIFT because `def fn_b` is at line 4, not 1.
    assert proc.returncode == 0, proc.stdout
    assert "ok: 2" in proc.stdout


def test_cite_with_no_following_fence_uses_landmark(tmp_path):
    """The fence-search has a small lookahead. A cite followed by lots
    of prose before any fence should fall through to landmark mode,
    not pick up an unrelated later fence."""
    repo = _setup_minimal_repo(tmp_path)
    _write(repo / "src" / "lib.py", "def fn():\n    pass\n")
    # 20 lines of prose between the cite and any ```.
    prose = "\n".join(f"Paragraph {i}." for i in range(20))
    _write(
        repo / "docs" / "lessons" / "01.md",
        "# Lesson\n\n"
        "Look at [`src/lib.py:1`](../../src/lib.py#L1).\n\n"
        f"{prose}\n\n"
        "```python\nthis is unrelated code\n```\n",
    )
    proc = _run(repo)
    # Should pass via landmark (def fn() at line 1); no snippet match.
    assert proc.returncode == 0, proc.stdout
