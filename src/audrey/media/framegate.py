"""Decide which sampled frames are worth a describe call.

Sampling a video at a fixed rate produces frames whose count is a function of
duration, not of content. Measured against real footage on 2026-08-02: a ~9½
minute retirement video sampled at 1 frame/30s gave 19 frames, and three of them
90 seconds apart came back from `qwen3-vl:32b` describing the same two people in
the same chairs against the same backdrop, differing only in a gesture. At
`gpu.concurrency: 1` each of those calls was exclusive GPU that bought nothing.

Sampling rate cannot fix that. Lower it and real cuts get missed; raise it and
you pay more for the same answer. The decision has to be driven by whether the
picture actually changed, which is what this module does — on CPU, in
milliseconds, before any GPU time is committed.

`vision.py` already caches descriptions, but it keys on the sha256 of the data
URI. Two frames a second apart from a locked-off camera differ in sensor noise
alone and hash differently, so that cache catches none of this.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

#: Bits of a 64-bit dHash that must differ before a frame earns its own describe
#: call. Tuned toward keeping too much rather than too little: a redundant
#: description costs one GPU call, a dropped scene change costs coverage that
#: nothing downstream can recover. Calibrate per corpus with `--min-distance`.
DEFAULT_MIN_DISTANCE = 8

#: No keyframe may stand in for more than this many consecutive frames. Without
#: it a three-hour lecture on one slide reduces to a single description, which is
#: technically accurate and useless. With it, long static stretches still get
#: periodic coverage.
DEFAULT_MAX_RUN = 20


@dataclass(frozen=True)
class Keyframe:
    """A frame chosen for description, and the frames it stands in for.

    `represents` always includes `index` itself, so `len(represents)` is the span
    this one describe call covers. The worker needs that to attribute a
    description back to a time range rather than a single instant.
    """

    index: int
    represents: tuple[int, ...]
    path: Path | None = None

    @property
    def span(self) -> int:
        return len(self.represents)


def dhash(image: PILImage, *, size: int = 8) -> int:
    """Difference hash: `size`x`size` bits of "is this pixel brighter than its right neighbour".

    Gradient-based rather than absolute, so it survives the exposure and encoder
    drift that make two visually identical frames differ byte-for-byte. Greyscale
    because a colour-grade shift is not a scene change.
    """
    from PIL import Image

    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    # tobytes() rather than getdata(): row-major, one byte per pixel in mode "L",
    # and not on Pillow's deprecation list.
    pixels = small.tobytes()
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits <<= 1
            if pixels[base + col] > pixels[base + col + 1]:
                bits |= 1
    return bits


def hamming(left: int, right: int) -> int:
    """How many bits two hashes disagree on."""
    return (left ^ right).bit_count()


def select(
    hashes: Sequence[int],
    *,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    max_run: int = DEFAULT_MAX_RUN,
) -> list[Keyframe]:
    """Choose which frames to describe, given their hashes in capture order.

    Pure and hash-only so the policy can be tested without decoding an image.

    Each candidate is compared against the last frame *kept*, not against its
    immediate predecessor. That distinction is the whole design: under pairwise
    comparison a slow pan differs from its neighbour by almost nothing at every
    step, so every frame would be dropped and a shot that ends somewhere
    completely different would never be described. Measuring against the last
    kept frame lets that drift accumulate until it crosses the threshold on its
    own.
    """
    if not hashes:
        return []
    if min_distance < 0:
        raise ValueError(f"min_distance must be >= 0, got {min_distance}")
    if max_run < 1:
        raise ValueError(f"max_run must be >= 1, got {max_run}")

    kept: list[int] = [0]  # the first frame has nothing to be redundant with
    members: list[list[int]] = [[0]]

    for index in range(1, len(hashes)):
        anchor = kept[-1]
        changed = hamming(hashes[anchor], hashes[index]) >= min_distance
        if changed or len(members[-1]) >= max_run:
            kept.append(index)
            members.append([index])
        else:
            members[-1].append(index)

    return [
        Keyframe(index=index, represents=tuple(group))
        for index, group in zip(kept, members, strict=True)
    ]


def select_keyframes(
    paths: Sequence[Path],
    *,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    max_run: int = DEFAULT_MAX_RUN,
) -> list[Keyframe]:
    """`select` over frame files, in the order given.

    Order is the caller's responsibility and matters: these are frames in capture
    order, and sorting `frame_10.jpg` before `frame_2.jpg` would scramble the
    adjacency the whole method depends on.
    """
    from PIL import Image

    hashes: list[int] = []
    for path in paths:
        with Image.open(path) as image:
            image.load()
            hashes.append(dhash(image))

    chosen = select(hashes, min_distance=min_distance, max_run=max_run)
    return [
        Keyframe(index=frame.index, represents=frame.represents, path=paths[frame.index])
        for frame in chosen
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Report the reduction for a set of frames, so a threshold can be calibrated.

    Exists to be run against real footage before the media worker does: the
    default distance is a starting point, and the only way to know whether it is
    right for a corpus is to look at what it drops.
    """
    parser = argparse.ArgumentParser(
        description="Select which sampled frames are worth describing.",
    )
    parser.add_argument("frames", nargs="+", type=Path,
                        help="frame files, in capture order")
    parser.add_argument("--min-distance", type=int, default=DEFAULT_MIN_DISTANCE,
                        help=f"dHash bits that must differ (default {DEFAULT_MIN_DISTANCE})")
    parser.add_argument("--max-run", type=int, default=DEFAULT_MAX_RUN,
                        help=f"max frames one keyframe may cover (default {DEFAULT_MAX_RUN})")
    args = parser.parse_args(argv)

    chosen = select_keyframes(
        args.frames, min_distance=args.min_distance, max_run=args.max_run,
    )
    total = len(args.frames)
    for frame in chosen:
        covered = "" if frame.span == 1 else f"  (+{frame.span - 1} near-identical)"
        print(f"  keep {frame.path}{covered}")

    saved = total - len(chosen)
    pct = (saved / total * 100) if total else 0.0
    print(f"\n{len(chosen)}/{total} frames need a describe call "
          f"— {saved} skipped, {pct:.0f}% fewer GPU calls")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
