"""ffmpeg audio extraction and probing (Phase 34).

Separated from the worker loop so it can be tested against a real fixture
file without a container, and so the worker stays a thing that talks HTTP.

Nothing here imports from `audrey` — see this package's `__init__`. The whole
module is stdlib plus two subprocesses, which is what lets the media-worker
image be `python:slim` + ffmpeg instead of the full app image.

**We extract, we never transform.** The source video is opened read-only and
never rewritten. The audio comes out as 16 kHz mono PCM because that is what
whisper wants in Phase 35 — resampling once here is cheaper than making the
model do it, and it makes the intermediate file small enough to sit in the
container's own /tmp rather than needing a writable mount.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# 16 kHz mono is whisper's native input. Anything else gets resampled inside
# the model, so we may as well do it during a demux we are running anyway.
SAMPLE_RATE = 16000
CHANNELS = 1

DEFAULT_PROBE_TIMEOUT_S = 60
DEFAULT_EXTRACT_TIMEOUT_S = 1800


class FFmpegMissingError(RuntimeError):
    """ffmpeg or ffprobe is not on PATH — an image problem, not a file problem."""


class FFmpegFailedError(RuntimeError):
    """ffmpeg ran and refused the file. Carries the tail of stderr."""


@dataclass(frozen=True)
class Probe:
    """What we learned about a media file without decoding it.

    `audio_duration_s` is 0.0 for a file with no audio stream. That is a fact
    about the file, not an error: a silent screen recording still has frames
    worth describing in Phase 36, and failing it here would deny it that.

    `has_video` is the mirror image, added in Phase 36 for the same reason. An
    audio-only file in a video container has nothing to describe and must
    still get its transcript — the visual pass reports zero frames, not a
    failure.
    """

    container_duration_s: float
    audio_duration_s: float
    has_audio: bool
    has_video: bool = False


def _binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise FFmpegMissingError(
            f"{name} is not on PATH — the media-worker image is built wrong",
        )
    return found


def _run(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    # `argv` is a list with an absolute binary path resolved by `shutil.which`
    # and no shell, so nothing here is interpreted by a shell. The only
    # caller-supplied element is a filesystem path.
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FFmpegFailedError(f"{argv[0]} timed out after {timeout}s") from e


def _tail(stderr: str, limit: int = 400) -> str:
    """Last of stderr. ffmpeg puts the actual reason at the end, after banners."""
    return stderr.strip()[-limit:]


def probe(path: Path, *, timeout_s: int = DEFAULT_PROBE_TIMEOUT_S) -> Probe:
    """Read stream metadata without decoding. Cheap even on a large file."""
    result = _run(
        [
            _binary("ffprobe"), "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise FFmpegFailedError(f"ffprobe rejected the file: {_tail(result.stderr)}")

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise FFmpegFailedError(f"ffprobe returned unparseable output: {e}") from e

    streams = parsed.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    # Cover art in an MP3 is a video stream by codec_type, and extracting
    # "frames" from it would describe the same JPEG once per sample interval.
    # A real video track has more than one frame; `nb_frames` is absent on
    # some containers, so treat unknown as real rather than discarding a
    # genuine track on a missing field.
    video = [
        s for s in streams
        if s.get("codec_type") == "video"
        and s.get("disposition", {}).get("attached_pic", 0) != 1
        and _as_float(s.get("nb_frames")) != 1.0
    ]

    # Duration can be absent on either the format or the stream depending on
    # the container — matroska routinely omits the stream-level one. Take
    # whichever is present rather than assuming a shape.
    container = _as_float((parsed.get("format") or {}).get("duration"))
    audio_duration = 0.0
    if audio:
        audio_duration = _as_float(audio[0].get("duration")) or container

    return Probe(
        container_duration_s=container,
        audio_duration_s=audio_duration,
        has_audio=bool(audio),
        has_video=bool(video),
    )


def _as_float(value: object) -> float:
    """ffprobe reports durations as strings, and sometimes as 'N/A'."""
    try:
        return max(0.0, float(str(value)))
    except (TypeError, ValueError):
        return 0.0


def extract_audio(
    src: Path, dest: Path, *, timeout_s: int = DEFAULT_EXTRACT_TIMEOUT_S,
) -> float:
    """Demux `src`'s audio to a 16 kHz mono WAV at `dest`. Returns its duration.

    Returns 0.0 and writes nothing when the source has no audio stream — the
    caller gets a successful, empty result rather than an exception it would
    only have to special-case back into success.
    """
    info = probe(src, timeout_s=DEFAULT_PROBE_TIMEOUT_S)
    if not info.has_audio:
        log.info("audio: %s has no audio stream, nothing to extract", src.name)
        return 0.0

    dest.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            _binary("ffmpeg"), "-nostdin", "-y",
            "-i", str(src),
            "-vn",                        # drop video; we want the audio only
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            str(dest),
        ],
        timeout=timeout_s,
    )
    if result.returncode != 0:
        # Don't leave a half-written wav behind for the next step to read as
        # if it were complete.
        dest.unlink(missing_ok=True)
        raise FFmpegFailedError(f"ffmpeg could not extract audio: {_tail(result.stderr)}")

    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise FFmpegFailedError("ffmpeg reported success but produced no audio")

    return info.audio_duration_s


__all__ = [
    "CHANNELS",
    "SAMPLE_RATE",
    "FFmpegFailedError",
    "FFmpegMissingError",
    "Probe",
    "extract_audio",
    "probe",
]
