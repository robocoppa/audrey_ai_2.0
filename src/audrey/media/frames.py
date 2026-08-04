"""ffmpeg keyframe extraction, and the thinning that decides what gets described (Phase 36).

The CPU half of the visual pass. Sampling produces candidates,
[`framegate`](framegate.py) decides which of them are worth a GPU call, and a
hard cap decides how many survive at all. None of it touches a model, which is
the point — every frame discarded here is exclusive GPU time never spent.

Nothing imports from `audrey` outside this package, so the media-worker image
stays `python:slim` + ffmpeg + Pillow rather than the full app image.

## Why sampling is by time, not by ffmpeg's own keyframes

`-skip_frame nokey` would hand back the encoder's I-frames, which sound like
the right thing and are not: their spacing is a function of the encoder's GOP
settings and the bitrate ladder, not of the content. A talking-head video
recorded at a high bitrate can emit an I-frame every two seconds with nothing
changing between them, while a screen recording of a static slide may go a
minute without one. Sampling at a fixed interval gives a count that depends
only on duration, which is predictable, and then the gate does the part that
depends on content.

## Why frames are downscaled on the way out

A 4K frame carries no more legible text than a 1080p one by the time the
vision encoder has resized it, and costs decode time at both ends. The width
cap is applied by ffmpeg during the same pass that samples, so the large
version never exists on disk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from audrey.media.audio import (
    DEFAULT_EXTRACT_TIMEOUT_S,
    FFmpegFailedError,
    _binary,
    _run,
    _tail,
    probe,
)
from audrey.media.framegate import (
    DEFAULT_MAX_RUN,
    DEFAULT_MIN_DISTANCE,
    select_keyframes,
)

log = logging.getLogger(__name__)

#: Seconds between sampled frames. 30s over a 9½-minute video gives 19
#: candidates, which is the run the gate was calibrated against.
DEFAULT_INTERVAL_S = 30.0

#: Longest edge of an extracted frame. See the module docstring.
DEFAULT_MAX_WIDTH = 1280

#: ffmpeg's JPEG quality scale, where 2 is best and 31 is worst. 3 is visually
#: lossless for this purpose and roughly a third the size of 2 — and size
#: matters here because every frame is base64'd into a JSON request body.
DEFAULT_QUALITY = 3

_FRAME_GLOB = "frame_*.jpg"
_FRAME_NUM = re.compile(r"frame_(\d+)\.jpg$")


@dataclass(frozen=True)
class Frame:
    """One sampled frame and where it came from in the video."""

    index: int
    path: Path
    t_start: float


@dataclass(frozen=True)
class SelectedFrame:
    """A frame chosen for description, and the span of video it speaks for.

    `t_start`/`t_end` bound every sampled frame this one stands in for, so a
    description of a static ten-minute stretch is attributed to ten minutes
    rather than to the instant it happened to be sampled at.
    """

    path: Path
    t_start: float
    t_end: float
    represents: int


def extract_frames(
    src: Path,
    dest_dir: Path,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: int = DEFAULT_QUALITY,
    timeout_s: int = DEFAULT_EXTRACT_TIMEOUT_S,
) -> list[Frame]:
    """Sample `src` every `interval_s` into `dest_dir`, in capture order.

    Returns an empty list when the source has no video stream. That is the
    same treatment `extract_audio` gives a file with no audio, and for the
    same reason: an audio-only file in a video container has nothing to
    describe and must still get its transcript. Failing here would turn a
    podcast upload into a failed job.

    A source that *does* have a video stream and still yields no frames is a
    different matter and raises — that is a decode failure wearing the
    disguise of an empty success, and it is the one shape this must not
    report as fine.
    """
    if interval_s <= 0:
        raise ValueError(f"interval_s must be > 0, got {interval_s}")

    info = probe(src)
    if not info.has_video:
        log.info("frames: %s has no video stream, nothing to extract", src.name)
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)

    # The single-quoted expression is for *ffmpeg's* filtergraph parser, not a
    # shell — there is no shell here. A bare comma inside min() would be read
    # as a filter separator and the graph would fail to parse.
    scale = f"scale='min({max_width},iw)':-2"
    _sample(src, dest_dir, vf=f"fps=1/{interval_s},{scale}",
            quality=quality, timeout_s=timeout_s)
    frames = _collect(dest_dir, interval_s)

    if not frames:
        # A clip shorter than one sample interval produces *nothing* under
        # `fps=1/N` — a 10-second upload at the 30-second default comes back
        # empty, which would fail the whole job over a video that decodes
        # perfectly well. Take its first frame instead.
        #
        # This doubles as the general fallback for a source whose duration
        # ffmpeg does not report, which is common enough in matroska that
        # deciding up front from the probe would be its own bug.
        log.info(
            "frames: %s yielded nothing at %.0fs sampling — taking one frame",
            src.name, interval_s,
        )
        _sample(src, dest_dir, vf=scale, quality=quality, timeout_s=timeout_s,
                extra=["-frames:v", "1"])
        frames = _collect(dest_dir, interval_s)

    if not frames:
        raise FFmpegFailedError(
            "ffmpeg reported success but produced no frames — the video stream "
            "is present but did not decode",
        )
    log.info(
        "frames: %s sampled %d frames every %.0fs", src.name, len(frames), interval_s,
    )
    return frames


def _sample(
    src: Path, dest_dir: Path, *, vf: str, quality: int, timeout_s: int,
    extra: list[str] | None = None,
) -> None:
    """One ffmpeg pass writing `frame_NNNNN.jpg` into `dest_dir`."""
    result = _run(
        [
            _binary("ffmpeg"), "-nostdin", "-y",
            "-i", str(src),
            "-vf", vf,
            "-an",                    # no audio in a still; phase 35 owns that
            "-q:v", str(quality),
            *(extra or []),
            str(dest_dir / "frame_%05d.jpg"),
        ],
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise FFmpegFailedError(f"ffmpeg could not extract frames: {_tail(result.stderr)}")


def _collect(dest_dir: Path, interval_s: float) -> list[Frame]:
    """Read back what ffmpeg wrote, ordered by frame number.

    Sorted numerically rather than lexically: `frame_10.jpg` sorts before
    `frame_2.jpg` as a string, and the gate compares each frame against the
    last one *kept*, so scrambled order does not merely reorder the output —
    it changes which frames are chosen.
    """
    numbered: list[tuple[int, Path]] = []
    for path in dest_dir.glob(_FRAME_GLOB):
        match = _FRAME_NUM.search(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort()
    # ffmpeg numbers from 1; the first sampled frame is at t=0.
    return [
        Frame(index=i, path=path, t_start=i * interval_s)
        for i, (_n, path) in enumerate(numbered)
    ]


def select_frames(
    frames: list[Frame],
    *,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    max_run: int = DEFAULT_MAX_RUN,
    limit: int | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> list[SelectedFrame]:
    """Thin `frames` down to the ones worth a describe call.

    Two stages, and the order matters. The gate drops frames that are
    *redundant* — it is content-aware and its output is the good answer. The
    limit then drops frames because there are too many, which is a budget
    decision and always a loss.
    """
    if not frames:
        return []

    chosen = select_keyframes(
        [f.path for f in frames], min_distance=min_distance, max_run=max_run,
    )
    selected = [
        SelectedFrame(
            path=frames[k.index].path,
            t_start=frames[k.index].t_start,
            # The end of the window this keyframe speaks for, not the instant
            # its last sibling was sampled at.
            t_end=frames[k.represents[-1]].t_start + interval_s,
            represents=k.span,
        )
        for k in chosen
    ]
    return _apply_limit(selected, limit)


def _apply_limit(
    selected: list[SelectedFrame], limit: int | None,
) -> list[SelectedFrame]:
    """Thin to `limit` frames, spread evenly, keeping the first and last.

    Truncating the tail would be simpler and is wrong: it would describe the
    first N scenes of a long video in detail and the rest not at all, so a
    two-hour recording would be searchable only for its opening. Spreading the
    loss evenly keeps coverage proportional across the whole file, and pinning
    the endpoints keeps the opening and closing shots — which in practice are
    the title card and the conclusion.
    """
    if limit is None or limit <= 0 or len(selected) <= limit:
        return selected
    if limit == 1:
        return [selected[0]]
    last = len(selected) - 1
    picked = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    log.info("frames: capping %d keyframes to %d", len(selected), len(picked))
    return [selected[i] for i in picked]


__all__ = [
    "DEFAULT_INTERVAL_S",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_QUALITY",
    "Frame",
    "SelectedFrame",
    "extract_frames",
    "select_frames",
]
