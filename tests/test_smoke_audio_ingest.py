"""The audio live smoke must generate a real audio-only MP3 fixture."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from audrey.kb.extract import sniff_mime
from audrey.media.audio import probe

SMOKE_DIR = Path(__file__).parent / "smoke"
sys.path.insert(0, str(SMOKE_DIR))
import smoke_audio_ingest as smoke  # noqa: E402


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_generated_fixture_is_mp3_with_audio_and_no_video(tmp_path):
    path = tmp_path / "fixture.mp3"
    path.write_bytes(smoke._spoken_mp3())

    info = probe(path)

    assert sniff_mime(path) == "audio/mpeg"
    assert info.has_audio is True
    assert info.has_video is False
    assert info.audio_duration_s > 0


def test_required_words_describe_the_spoken_fixture():
    spoken = smoke._SPOKEN_TEXT.lower()
    assert smoke._REQUIRED_WORDS == {"blue", "lantern", "audio", "ready"}
    assert all(word in spoken for word in smoke._REQUIRED_WORDS)

