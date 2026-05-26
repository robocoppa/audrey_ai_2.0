"""Smoke tests for scripts/check-lesson-conventions.py.

Builds synthetic lesson fixtures in `tmp_path` and runs the checker
via subprocess. Each test pins one rule's behavior end-to-end:
input prose → finding → exit code.

Same posture as test_check_lesson_links.py — external process so the
CLI contract (DOCS_GLOB, args, exit code) stays honest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check-lesson-conventions.py"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content.endswith("\n") else content + "\n")


def _run(repo: Path, *args: str, docs_glob: str = "docs/lessons/lesson-*.md") -> subprocess.CompletedProcess:
    env = {**os.environ, "DOCS_GLOB": docs_glob}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# ─── PHASE_N ──────────────────────────────────────────────────────────


def test_phase_n_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "This was added in Phase 9.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "PHASE_N" in result.stdout


def test_phase_word_alone_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "This phase of the project is different.")
    result = _run(tmp_path)
    assert result.returncode == 0


def test_phase_n_inside_code_block_ignored(tmp_path: Path) -> None:
    body = "Some intro.\n\n```\nPhase 9 deploy notes\n```\n\nOutro.\n"
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", body)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_phase_n_inside_inline_code_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "See the `Phase 9` deploy doc.")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


# ─── REAL_EMAIL ───────────────────────────────────────────────────────


def test_real_email_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Contact alice@proton.me for details.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "REAL_EMAIL" in result.stdout


def test_example_email_allowed(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Use alice@example.com as the test user.")
    result = _run(tmp_path)
    assert result.returncode == 0


def test_example_subdomain_allowed(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Send to bob@mail.example.com.")
    result = _run(tmp_path)
    assert result.returncode == 0


# ─── BART ─────────────────────────────────────────────────────────────


def test_bart_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Bart wrote this code.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "BART" in result.stdout


def test_bart_substring_not_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "The bartender knew everyone.")
    result = _run(tmp_path)
    assert result.returncode == 0


# ─── FORWARD_REF ──────────────────────────────────────────────────────


def test_forward_ref_skipping_ahead_flagged(tmp_path: Path) -> None:
    # Lesson 4 → Lesson 12 is NN+8 — well past the next-lesson handoff.
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Lesson 12 covers the watcher.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "FORWARD_REF" in result.stdout


def test_next_lesson_handoff_allowed(tmp_path: Path) -> None:
    # Lesson 4 → Lesson 5 is the standard footer handoff, allowed.
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Lesson 5 picks up from here.")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_backward_ref_allowed(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-08-foo.md", "Lesson 6 introduced the model layer.")
    result = _run(tmp_path)
    assert result.returncode == 0


def test_same_lesson_ref_allowed(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-08-foo.md", "This is Lesson 8.")
    result = _run(tmp_path)
    assert result.returncode == 0


def test_forward_ref_in_link_target_ignored(tmp_path: Path) -> None:
    # The URL target is `lesson-12-...md` — `Lesson 12` appears as
    # filename text, not lesson-prose claim. The link *text* is
    # backward-safe ("Lesson 5") so this should pass.
    body = "See [Lesson 5](lesson-12-streaming.md) for details.\n"
    _write(tmp_path / "docs/lessons/lesson-08-foo.md", body)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout


def test_forward_ref_in_link_text_flagged_when_skipping_ahead(tmp_path: Path) -> None:
    # Lesson 4 → Lesson 14 in link text is still a violation (skips
    # 9 lessons ahead, may not even exist).
    body = "See [Lesson 14](lesson-14-streaming.md) for the frame format.\n"
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", body)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "FORWARD_REF" in result.stdout


# ─── COUNT ────────────────────────────────────────────────────────────


def test_count_chunks_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Audrey indexes ~16k chunks of KB content.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "COUNT" in result.stdout


def test_count_tests_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "The suite has 110 tests.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "COUNT" in result.stdout


def test_count_loc_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Audrey is ~9,800 LOC across the package.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "COUNT" in result.stdout


def test_count_pytests_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "343 hermetic pytests pass on every change.")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "COUNT" in result.stdout


def test_file_line_citation_allowed(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "See `main.py:53` for the lifespan hook.")
    result = _run(tmp_path)
    assert result.returncode == 0


def test_line_range_allowed(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Lines 5-9 set up the FastAPI app.")
    result = _run(tmp_path)
    assert result.returncode == 0


def test_small_chunk_count_in_per_file_context_flagged(tmp_path: Path) -> None:
    # Even "~5 chunks" technically violates the spirit (specific count),
    # though it's per-file rather than codebase-size. The rule fires
    # regardless — the user can either rephrase or accept.
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Each PDF produces ~5 chunks on average.")
    result = _run(tmp_path)
    assert result.returncode == 1


# ─── Driver behavior ──────────────────────────────────────────────────


def test_clean_corpus_passes(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs/lessons/lesson-04-foo.md",
        "This is clean prose with no rule violations.\n"
        "We reference `alice@example.com` and earlier lessons safely.\n",
    )
    result = _run(tmp_path)
    assert result.returncode == 0
    assert "0 finding" in result.stdout


def test_json_output_shape(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Phase 9 was the deploy.")
    result = _run(tmp_path, "--json")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert len(data) == 1
    assert data[0]["rule"] == "PHASE_N"
    assert data[0]["line"] == 1
    assert "Phase 9" in data[0]["excerpt"]


def test_quiet_mode_only_prints_count(tmp_path: Path) -> None:
    _write(tmp_path / "docs/lessons/lesson-04-foo.md", "Phase 9 was the deploy.")
    result = _run(tmp_path, "--quiet")
    assert result.returncode == 1
    # No per-finding "PHASE_N" preamble, just the summary line.
    assert "PHASE_N" not in result.stdout
    assert "1 finding" in result.stdout


def test_explicit_file_argument(tmp_path: Path) -> None:
    target = tmp_path / "docs/lessons/lesson-04-foo.md"
    _write(target, "Phase 9 deploy.")
    # Pass the file directly; DOCS_GLOB is ignored when args are given.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(target)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "PHASE_N" in result.stdout


def test_non_lesson_file_skips_forward_ref(tmp_path: Path) -> None:
    # README.md has no lesson number → FORWARD_REF can't fire.
    _write(tmp_path / "docs/lessons/README.md", "Lesson 14 will cover streaming.")
    result = _run(tmp_path, docs_glob="docs/lessons/*.md")
    assert result.returncode == 0, result.stdout
