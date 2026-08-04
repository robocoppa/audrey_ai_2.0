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

import pytest

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


class TestTranscriptPayloadBytes:
    """The transcript's points must report the SOURCE video's size.

    `reconcile_with_qdrant` rebuilds each uploads row from its points' payload
    on every boot, and `user_total_bytes` — the quota gate — sums that column.
    So a payload carrying the sidecar's size instead of the video's silently
    re-bills a 288 MB upload as 9 KB, one restart after ingest. Found in
    production on the first real transcript.
    """

    @pytest.mark.asyncio
    async def test_the_payload_carries_the_video_size_not_the_sidecar(
        self, tmp_path: Path,
    ):
        from audrey.kb.ingest import ingest_transcript_segments

        sidecar = tmp_path / "f1.transcript.txt"
        sidecar.write_text("[00:00:00] a short transcript", encoding="utf-8")
        video_bytes = 301_936_597

        captured: list = []

        class _Q:
            async def delete_by_file_id(self, *a, **k) -> None: ...
            async def upsert_text(self, points, *, collection) -> None:
                captured.extend(points)

        class _E:
            async def embed_many(self, texts):
                return [[0.1] * 8 for _ in texts]

        n = await ingest_transcript_segments(
            [{"t_start": 0.0, "t_end": 2.0, "text": "hello there this is speech"}],
            sidecar=sidecar, qdrant=_Q(), embedder=_E(), collection="c",
            user="a@b.c", file_id="f1", filename="jasonRetirement.mp4",
            mime="video/mp4", source_bytes=video_bytes,
        )

        assert n == 1
        payload = captured[0].payload
        assert payload["bytes"] == video_bytes
        # The sidecar is ~29 bytes; if that value ever appears here, the quota
        # is being computed from the wrong file again.
        assert payload["bytes"] != sidecar.stat().st_size

    @pytest.mark.asyncio
    async def test_the_timestamps_are_payload_not_text(self, tmp_path: Path):
        from audrey.kb.ingest import ingest_transcript_segments

        sidecar = tmp_path / "f1.transcript.txt"
        sidecar.write_text("[00:00:00] hello", encoding="utf-8")
        captured: list = []

        class _Q:
            async def delete_by_file_id(self, *a, **k) -> None: ...
            async def upsert_text(self, points, *, collection) -> None:
                captured.extend(points)

        class _E:
            async def embed_many(self, texts):
                return [[0.1] * 8 for _ in texts]

        await ingest_transcript_segments(
            [{"t_start": 12.0, "t_end": 15.0, "text": "hello there"}],
            sidecar=sidecar, qdrant=_Q(), embedder=_E(), collection="c",
            user="a@b.c", file_id="f1", filename="v.mp4", mime="video/mp4",
            source_bytes=1000,
        )

        payload = captured[0].payload
        assert payload["t_start"] == 12.0
        assert payload["t_end"] == 15.0
        assert "[" not in payload["text"]
        assert payload["artifact"] == "transcript"
