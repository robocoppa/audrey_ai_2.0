"""Tests for keyframe extraction and thinning (Phase 36).

Same posture as `test_media_audio.py`: real ffmpeg on generated fixtures. The
value of this module is entirely in what ffmpeg actually does — how it numbers
output files, whether a filtergraph expression parses, what it emits for a
source with no video stream — and a mocked `subprocess.run` would only assert
that we wrote down what we already believed.

The fixtures are deliberately *textured*. An earlier draft used flat colour
fields and the gate merged red, blue and white into a single keyframe, which
looked like a gate bug and is not one: dHash asks "is this pixel brighter than
the one to its right", so a flat field has zero gradient everywhere and every
solid colour hashes identically. Real footage has texture; a fixture that does
not is testing a case the gate was never meant to handle.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from audrey.media.audio import FFmpegFailedError, probe
from audrey.media.frames import (
    Frame,
    SelectedFrame,
    _apply_limit,
    extract_frames,
    select_frames,
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
def three_scenes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """90 seconds, three visually distinct 30-second textured scenes."""
    path = tmp_path_factory.mktemp("frames") / "scenes.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=5:duration=30",
        "-f", "lavfi", "-i", "smptebars=size=640x480:rate=5:duration=30",
        "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=5:duration=30",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
        "-map", "[out]", "-pix_fmt", "yuv420p", str(path),
    )
    return path


@pytest.fixture(scope="session")
def audio_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An m4a — a real upload shape with no video stream at all."""
    path = tmp_path_factory.mktemp("frames") / "podcast.m4a"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:a", "aac", str(path))
    return path


@pytest.fixture(scope="session")
def big_frame_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("frames") / "big.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=5:duration=4",
            "-pix_fmt", "yuv420p", str(path))
    return path


class TestProbeSeesVideo:
    def test_a_video_file_reports_a_video_stream(self, three_scenes: Path):
        assert probe(three_scenes).has_video

    def test_an_audio_only_file_does_not(self, audio_only: Path):
        """The distinction the whole visual pass branches on."""
        info = probe(audio_only)
        assert info.has_audio
        assert not info.has_video


class TestExtractFrames:
    def test_frames_are_sampled_at_the_interval(self, three_scenes: Path, tmp_path: Path):
        frames = extract_frames(three_scenes, tmp_path, interval_s=10.0)
        assert len(frames) == 9
        assert [f.t_start for f in frames] == [0, 10, 20, 30, 40, 50, 60, 70, 80]

    def test_frames_come_back_in_capture_order(self, three_scenes: Path, tmp_path: Path):
        """`frame_10.jpg` sorts before `frame_2.jpg` as a string. The gate
        compares each frame against the last one *kept*, so a scrambled order
        does not just reorder the output — it changes which frames survive."""
        frames = extract_frames(three_scenes, tmp_path, interval_s=5.0)
        assert len(frames) > 10
        assert [f.index for f in frames] == sorted(f.index for f in frames)
        assert [f.path.name for f in frames] == sorted(f.path.name for f in frames)

    def test_frames_are_downscaled_to_the_width_cap(
        self, big_frame_video: Path, tmp_path: Path,
    ):
        """A 4K frame carries no more legible text than a 1080p one once the
        vision encoder resizes it, and costs decode time at both ends."""
        from PIL import Image
        frames = extract_frames(big_frame_video, tmp_path, interval_s=1.0, max_width=640)
        with Image.open(frames[0].path) as image:
            # 1920x1080 is 16:9, so the height follows from the width cap.
            # Aspect must be preserved — a squashed frame makes on-screen text
            # harder to read, which is the one thing this pass is for.
            assert image.size == (640, 360)

    def test_a_frame_narrower_than_the_cap_is_not_upscaled(
        self, three_scenes: Path, tmp_path: Path,
    ):
        from PIL import Image
        frames = extract_frames(three_scenes, tmp_path, interval_s=30.0, max_width=1280)
        with Image.open(frames[0].path) as image:
            assert image.size == (640, 480)

    def test_an_audio_only_file_yields_no_frames_rather_than_failing(
        self, audio_only: Path, tmp_path: Path,
    ):
        """A podcast upload must still get its transcript. Failing here would
        turn phase 35's successful job into a failed one."""
        assert extract_frames(audio_only, tmp_path, interval_s=1.0) == []

    def test_a_clip_shorter_than_the_interval_still_yields_one_frame(
        self, tmp_path: Path,
    ):
        """`fps=1/30` on a 4-second video emits *nothing* — the filter never
        reaches its first output time. Left alone that fails the whole job on
        a video which decodes perfectly well, and it would have hit every
        short upload."""
        path = tmp_path / "short.mp4"
        _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=5:duration=4",
                "-pix_fmt", "yuv420p", str(path))

        frames = extract_frames(path, tmp_path / "out", interval_s=30.0)

        assert len(frames) == 1
        assert frames[0].t_start == 0.0

    def test_a_corrupt_source_raises(self, tmp_path: Path):
        bad = tmp_path / "not-a-video.mp4"
        bad.write_bytes(b"this is not a video")
        with pytest.raises(FFmpegFailedError):
            extract_frames(bad, tmp_path / "out")

    def test_a_non_positive_interval_is_refused(self, three_scenes: Path, tmp_path: Path):
        """`fps=1/0` is a filtergraph that fails deep inside ffmpeg with a
        message about nothing in particular."""
        with pytest.raises(ValueError, match="interval_s"):
            extract_frames(three_scenes, tmp_path, interval_s=0)


class TestSelectFrames:
    def test_each_distinct_scene_earns_one_describe_call(
        self, three_scenes: Path, tmp_path: Path,
    ):
        """Nine candidates, three scenes, three GPU calls — a 67% saving that
        no description cache could find, since every frame differs in bytes."""
        frames = extract_frames(three_scenes, tmp_path, interval_s=10.0)
        selected = select_frames(frames, interval_s=10.0)

        assert len(selected) == 3
        assert [s.represents for s in selected] == [3, 3, 3]

    def test_a_keyframe_is_attributed_to_the_span_it_speaks_for(
        self, three_scenes: Path, tmp_path: Path,
    ):
        """Not to the instant it happened to be sampled at. A description of a
        static stretch belongs to the whole stretch."""
        frames = extract_frames(three_scenes, tmp_path, interval_s=10.0)
        selected = select_frames(frames, interval_s=10.0)

        assert [(s.t_start, s.t_end) for s in selected] == [
            (0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]

    def test_no_frames_selects_nothing(self):
        assert select_frames([], interval_s=10.0) == []


class TestTheCap:
    """`keyframes_max` is a hard cap, never a scene-detection byproduct. The
    gate ranks and thins; the cap decides how many survive."""

    def _fake(self, n: int) -> list[SelectedFrame]:
        return [SelectedFrame(path=Path(f"{i}.jpg"), t_start=i * 10.0,
                              t_end=i * 10.0 + 10, represents=1) for i in range(n)]

    def test_the_cap_is_never_exceeded(self):
        for limit in (1, 2, 3, 7):
            assert len(_apply_limit(self._fake(20), limit)) <= limit

    def test_the_first_and_last_frames_always_survive(self):
        """Truncating the tail would leave a two-hour video searchable only
        for its opening. In practice the endpoints are the title card and the
        conclusion."""
        capped = _apply_limit(self._fake(20), 5)
        assert capped[0].t_start == 0.0
        assert capped[-1].t_start == 190.0

    def test_the_survivors_are_spread_evenly(self):
        capped = _apply_limit(self._fake(21), 5)
        assert [f.t_start for f in capped] == [0.0, 50.0, 100.0, 150.0, 200.0]

    def test_a_limit_of_one_keeps_the_opening(self):
        assert len(_apply_limit(self._fake(9), 1)) == 1

    def test_under_the_cap_nothing_is_dropped(self):
        assert len(_apply_limit(self._fake(3), 10)) == 3

    def test_no_cap_means_no_cap(self):
        for limit in (None, 0, -1):
            assert len(_apply_limit(self._fake(30), limit)) == 30

    def test_order_is_preserved(self):
        capped = _apply_limit(self._fake(20), 6)
        assert [f.t_start for f in capped] == sorted(f.t_start for f in capped)


class TestGateBlindSpot:
    """dHash is a gradient hash, so it cannot see a cut between two flat
    colour fields — every solid colour has zero gradient and hashes to zero.

    Pinned rather than fixed. Real footage has texture, and the alternatives
    (average hash, colour histograms) trade this rare case for sensitivity to
    exposure and colour-grade drift, which is the common one. Worth knowing
    before someone tests with a fixture full of solid colours and concludes
    the gate is broken.
    """

    def test_a_cut_between_flat_colours_is_not_detected(self, tmp_path: Path):
        path = tmp_path / "flat.mp4"
        _ffmpeg(
            "-f", "lavfi", "-i", "color=c=red:size=320x240:rate=5:duration=10",
            "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=5:duration=10",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map", "[out]", "-pix_fmt", "yuv420p", str(path),
        )
        frames = extract_frames(path, tmp_path / "out", interval_s=5.0)
        assert len(frames) == 4

        assert len(select_frames(frames, interval_s=5.0)) == 1


class TestFrameDataclass:
    def test_a_frame_knows_where_it_came_from(self):
        frame = Frame(index=2, path=Path("/x/frame_00003.jpg"), t_start=60.0)
        assert (frame.index, frame.t_start) == (2, 60.0)
