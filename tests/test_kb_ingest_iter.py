"""Tests for `_iter_files` walk-filter behavior.

The crawler skips:
  - dot-named files at any depth (e.g. `.DS_Store`)
  - any file whose path includes a dot-prefixed *directory* (e.g. `.git/`)

The dot-directory skip is defensive — `/datasets` is curated, but a stray
`.git/` or `.cache/` slipping into a topic dir would otherwise have its
non-dot children ingested as if they were KB content.
"""
from __future__ import annotations

from pathlib import Path

from audrey.kb.ingest import _iter_files


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_iter_files_yields_plain_files(tmp_path: Path):
    _touch(tmp_path / "a.md")
    _touch(tmp_path / "sub" / "b.txt")
    got = sorted(p.name for p in _iter_files(tmp_path))
    assert got == ["a.md", "b.txt"]


def test_iter_files_skips_dotfiles_at_any_depth(tmp_path: Path):
    _touch(tmp_path / "a.md")
    _touch(tmp_path / ".hidden")
    _touch(tmp_path / "sub" / ".also_hidden")
    got = sorted(p.name for p in _iter_files(tmp_path))
    assert got == ["a.md"]


def test_iter_files_skips_dot_directory_contents(tmp_path: Path):
    # The non-dot children of a dot-directory must also be skipped — this is
    # the case the old `p.name.startswith(".")` check missed.
    _touch(tmp_path / "a.md")
    _touch(tmp_path / ".git" / "HEAD")
    _touch(tmp_path / ".git" / "objects" / "pack")
    _touch(tmp_path / "topic" / ".cache" / "tmp.txt")
    _touch(tmp_path / "topic" / "real.md")
    got = sorted(p.name for p in _iter_files(tmp_path))
    assert got == ["a.md", "real.md"]


def test_iter_files_single_file_root(tmp_path: Path):
    f = tmp_path / "only.md"
    _touch(f)
    got = list(_iter_files(f))
    assert got == [f]
