"""Tests for the ffmpeg wrapper (Phase 34).

These run against **real ffmpeg on generated fixtures**, not mocks. The whole
value of this module is its handling of what ffmpeg actually does — how it
reports a missing audio stream, where it puts the reason for a rejection,
which durations it omits for a given container. A mocked `subprocess.run`
would only assert that we wrote down what we already believed, which is the
belief that would be wrong.

Fixtures are a fraction of a second long and built once per session.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from audrey.media.audio import (
    CHANNELS,
    SAMPLE_RATE,
    FFmpegFailedError,
    FFmpegMissingError,
    extract_audio,
    probe,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", *args],
        check=True, capture_output=True,
    )


@pytest.fixture(scope="session")
def with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One second of test pattern with a 440 Hz tone."""
    path = tmp_path_factory.mktemp("media") / "with_audio.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
    )
    return path


@pytest.fixture(scope="session")
def silent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Video with no audio stream at all — not a silent track, none."""
    path = tmp_path_factory.mktemp("media") / "silent.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
        "-c:v", "mpeg4", str(path),
    )
    return path


@pytest.fixture
def garbage(tmp_path: Path) -> Path:
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"this is not a container, it is a sentence")
    return path


class TestProbe:
    def test_it_finds_the_audio_stream(self, with_audio: Path):
        info = probe(with_audio)
        assert info.has_audio is True
        assert info.audio_duration_s == pytest.approx(1.0, abs=0.3)

    def test_a_video_with_no_audio_stream_is_not_an_error(self, silent: Path):
        """The case that must not raise. A silent screen recording still has
        frames worth describing in Phase 36, and failing here would deny it
        that on the basis of something it was never required to have."""
        info = probe(silent)
        assert info.has_audio is False
        assert info.audio_duration_s == 0.0

    def test_the_container_duration_survives_having_no_audio(self, silent: Path):
        """Audio duration is 0 but the file is still a second long — the two
        are separate facts and Phase 37 will want the second one."""
        assert probe(silent).container_duration_s == pytest.approx(1.0, abs=0.3)

    def test_an_unreadable_file_raises_with_ffmpegs_reason(self, garbage: Path):
        with pytest.raises(FFmpegFailedError) as e:
            probe(garbage)
        # The message must carry ffmpeg's own words. A generic "probe failed"
        # sends whoever reads the failed row back to the container logs.
        assert str(e.value).strip() != "ffprobe rejected the file:"
        assert len(str(e.value)) > len("ffprobe rejected the file: ")

    def test_a_missing_file_raises_rather_than_reporting_zero(self, tmp_path: Path):
        with pytest.raises(FFmpegFailedError):
            probe(tmp_path / "nope.mp4")


class TestExtractAudio:
    def test_it_writes_a_16k_mono_wav(self, with_audio: Path, tmp_path: Path):
        dest = tmp_path / "out.wav"
        duration = extract_audio(with_audio, dest)

        assert dest.exists()
        assert duration == pytest.approx(1.0, abs=0.3)
        info = probe(dest)
        assert info.has_audio is True

    def test_the_sample_rate_and_channels_are_what_whisper_wants(
        self, with_audio: Path, tmp_path: Path,
    ):
        """Phase 35 feeds this straight to whisper. Resampling here means the
        model doesn't have to, and pins the format before anything depends on
        it implicitly."""
        dest = tmp_path / "out.wav"
        extract_audio(with_audio, dest)

        out = subprocess.run(
            [shutil.which("ffprobe") or "ffprobe", "-v", "error",
             "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,channels", "-of", "csv=p=0", str(dest)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert out == f"{SAMPLE_RATE},{CHANNELS}"

    def test_a_silent_video_returns_zero_and_writes_nothing(
        self, silent: Path, tmp_path: Path,
    ):
        """Success, not failure. The caller gets an empty result rather than
        an exception it would only have to translate back into success."""
        dest = tmp_path / "out.wav"
        assert extract_audio(silent, dest) == 0.0
        assert not dest.exists()

    def test_a_rejected_file_leaves_no_partial_wav(
        self, garbage: Path, tmp_path: Path,
    ):
        """A half-written wav is worse than none — the next step would read it
        as a complete one."""
        dest = tmp_path / "out.wav"
        with pytest.raises(FFmpegFailedError):
            extract_audio(garbage, dest)
        assert not dest.exists()

    def test_it_creates_the_destination_directory(
        self, with_audio: Path, tmp_path: Path,
    ):
        dest = tmp_path / "nested" / "deeper" / "out.wav"
        extract_audio(with_audio, dest)
        assert dest.exists()

    def test_the_source_is_not_modified(self, with_audio: Path, tmp_path: Path):
        """We extract, we never transform. The uploads mount is read-only in
        compose, so a write attempt would fail in production — this catches it
        before it gets there."""
        before = with_audio.read_bytes()
        extract_audio(with_audio, tmp_path / "out.wav")
        assert with_audio.read_bytes() == before


class TestMissingBinary:
    def test_a_missing_ffmpeg_is_its_own_error_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        """An image defect, not a file defect. The worker treats the two
        differently: one fails the row, the other must not."""
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(FFmpegMissingError):
            probe(tmp_path / "any.mp4")
