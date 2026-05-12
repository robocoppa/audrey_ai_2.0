"""Stress test for scripts/check-lesson-links.py.

Generates synthetic doc+source pairs at varying scales, injects line
shifts, and confirms the script either (a) proposes the right
correction or (b) honestly says it can't find the snippet — never
silently passes drifted cites.

Run as part of the normal pytest suite. It's still hermetic: temp
dirs, no network, no real-source dependencies.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check-lesson-links.py"


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "REPO_ROOT": str(repo),
        "DOCS_GLOB": "docs/lessons/*.md",
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _gen_source(rng: random.Random, num_funcs: int) -> tuple[str, list[tuple[str, int]]]:
    """Generate a synthetic Python file with `num_funcs` functions.

    Returns (source_text, [(function_name, 1-indexed_line), ...]).
    Lines between functions are filler comments so insertions are
    visible in test output.
    """
    lines: list[str] = []
    fn_at: list[tuple[str, int]] = []
    for i in range(num_funcs):
        # Random preceding filler so functions don't always start at
        # predictable offsets.
        for _ in range(rng.randint(1, 4)):
            lines.append(f"# filler {i}-{rng.randint(0, 99)}")
        fn_at.append((f"fn_{i}", len(lines) + 1))
        lines.append(f"def fn_{i}() -> int:")
        lines.append(f"    return {i}")
    return "\n".join(lines) + "\n", fn_at


def test_stress_correct_proposals_after_random_insertion(tmp_path):
    """Generate a source file with 10 functions, then prepend N random
    blank/comment lines. Every cite that pointed at a function header
    should now be proposed at original_line + N."""
    rng = random.Random(42)  # noqa: S311 — deterministic test fuzz, not crypto
    repo = tmp_path
    (repo / "docs" / "lessons").mkdir(parents=True)
    (repo / "src").mkdir()

    source, fns = _gen_source(rng, num_funcs=10)
    src_path = repo / "src" / "lib.py"
    src_path.write_text(source)

    # Write a lesson that cites each function with the original line and
    # displays the function's first line as the snippet.
    lesson_lines = ["# Stress\n"]
    for name, line in fns:
        lesson_lines.append(
            f"Open [`src/lib.py:{line}`](../../src/lib.py#L{line}):\n\n"
            f"```python\n"
            f"def {name}() -> int:\n"
            f"    return {name.split('_')[1]}\n"
            f"```\n"
        )
    (repo / "docs" / "lessons" / "00.md").write_text("\n".join(lesson_lines))

    # Sanity: with no mutation the run should be clean.
    proc = _run(repo)
    assert proc.returncode == 0, proc.stdout
    assert "drift: 0" in proc.stdout

    # Now insert N blank lines at the top of the source.
    inserted = 25
    src_path.write_text("\n" * inserted + source)

    proc = _run(repo)
    assert proc.returncode == 1, proc.stdout
    # Every cite should propose original + 25 (we inserted 25 lines).
    for name, line in fns:
        expected_fix = f"fix: change #L{line} → #L{line + inserted}"
        assert expected_fix in proc.stdout, (
            f"missing proposal for {name}: expected '{expected_fix}'\n"
            f"got:\n{proc.stdout}"
        )


def test_stress_no_false_positives_under_unrelated_edits(tmp_path):
    """Make code changes that don't touch any cited function. Script
    should not flag any cite as drift."""
    rng = random.Random(7)  # noqa: S311 — deterministic test fuzz, not crypto
    repo = tmp_path
    (repo / "docs" / "lessons").mkdir(parents=True)
    (repo / "src").mkdir()

    source, fns = _gen_source(rng, num_funcs=5)
    src_path = repo / "src" / "lib.py"
    src_path.write_text(source)

    # Cite only fn_0, fn_2, fn_4 (skip 1 and 3).
    cited = [(name, line) for name, line in fns if int(name.split("_")[1]) % 2 == 0]
    lesson_lines = ["# Stress\n"]
    for name, line in cited:
        lesson_lines.append(
            f"Open [`src/lib.py:{line}`](../../src/lib.py#L{line}):\n\n"
            f"```python\n"
            f"def {name}() -> int:\n"
            f"    return {name.split('_')[1]}\n"
            f"```\n"
        )
    (repo / "docs" / "lessons" / "00.md").write_text("\n".join(lesson_lines))

    # Append unrelated code at the end of the source — should not
    # shift cited line numbers.
    src_path.write_text(source + "\n# bunch\n# of\n# unrelated\n# lines\n")

    proc = _run(repo)
    assert "drift: 0" in proc.stdout, proc.stdout


def test_stress_function_rename_reports_drift_without_fix(tmp_path):
    """When a cited function is renamed and its body changes, the
    script should report drift (no fix proposal) rather than silently
    pass or propose a misleading line number."""
    repo = tmp_path
    (repo / "docs" / "lessons").mkdir(parents=True)
    (repo / "src").mkdir()

    (repo / "src" / "lib.py").write_text(
        "def renamed() -> int:\n    return 99\n"
    )
    (repo / "docs" / "lessons" / "00.md").write_text(
        "Open [`src/lib.py:1`](../../src/lib.py#L1):\n\n"
        "```python\n"
        "def original_name() -> int:\n"
        "    return 42\n"
        "```\n"
    )
    proc = _run(repo)
    # Renamed function — script can't find original_name anywhere.
    # Cited line still reads `def renamed()` which IS a landmark,
    # so this should be a soft DRIFT? not confident DRIFT.
    assert "drift?: 1" in proc.stdout, proc.stdout
    assert proc.returncode == 0  # DRIFT? is advisory


def test_stress_many_random_shifts(tmp_path):
    """Run a fuzz pass: 20 trials of (random source, random shift),
    confirm proposed corrections are always exactly shift + original."""
    rng = random.Random(123)  # noqa: S311 — deterministic test fuzz, not crypto
    for trial in range(20):
        repo = tmp_path / f"trial-{trial}"
        (repo / "docs" / "lessons").mkdir(parents=True)
        (repo / "src").mkdir()

        n_funcs = rng.randint(3, 12)
        source, fns = _gen_source(rng, num_funcs=n_funcs)
        src_path = repo / "src" / "lib.py"

        lesson_parts = ["# Stress\n"]
        for name, line in fns:
            lesson_parts.append(
                f"Open [`src/lib.py:{line}`](../../src/lib.py#L{line}):\n\n"
                f"```python\n"
                f"def {name}() -> int:\n"
                f"    return {name.split('_')[1]}\n"
                f"```\n"
            )
        (repo / "docs" / "lessons" / "00.md").write_text("\n".join(lesson_parts))

        shift = rng.randint(15, 60)  # Big enough to exceed NEAR_CITE_RANGE
        src_path.write_text("\n" * shift + source)

        proc = _run(repo)
        if proc.returncode != 1:
            raise AssertionError(
                f"trial {trial}: expected non-zero exit, got {proc.returncode}\n"
                f"{proc.stdout}"
            )
        for name, line in fns:
            expected = f"fix: change #L{line} → #L{line + shift}"
            if expected not in proc.stdout:
                raise AssertionError(
                    f"trial {trial}, fn {name}: missing '{expected}'\n"
                    f"stdout:\n{proc.stdout}"
                )
