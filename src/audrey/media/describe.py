"""The worker's side of the keyframe describe call (Phase 36).

Sampling and thinning happen in [`frames.py`](frames.py); this turns what
survives into prose by asking `audrey-ai` for it, one frame at a time.

## Why it goes through Audrey rather than straight to Ollama

`FairLocalGate` is in-process to `audrey-ai`. A worker calling Ollama directly
would share no gate at all and would contend with live chat at the Ollama
level with no fairness whatsoever — a long ingest would simply starve the box.
The worker's compose network is `internal: true` precisely so this cannot be
done by accident.

`POST /v1/media/describe` is the door. It is service-token authenticated and
takes the uploading user's email, which is what puts the ingest in *that
user's* round-robin slice: a giant video slows its own owner's chat and leaves
everyone else's alone.

## Why a partial result is acceptable here, unlike a transcript

Phase 35 holds that a partial transcript is worse than none, because a
truncated transcript looks complete — nothing in the artifact says where it
stopped, so it is silently wrong in the way that never gets noticed.

Frame descriptions are not like that. Each one is independently timestamped
and stands alone, so a set that stops early is *correct about what it covers*
and merely incomplete. Given a budget shorter than the work, returning eight
described keyframes beats returning nothing and burning an attempt — so this
stops at the budget, reports how many it meant to do, and lets the row record
the difference.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable
from pathlib import Path

from audrey.media.frames import SelectedFrame

log = logging.getLogger(__name__)

#: Seconds to spend on the whole visual pass. The binding constraint is the
#: lease, not patience: `kb.video.lease_minutes` is 30, and `keyframes_max` 24
#: frames at `vision.timeout_s` 120s each is 48 minutes — so an unbudgeted
#: pass on a slow model would have its job swept out from under it and retried
#: forever. 900s leaves half the lease for demux and transcription.
DEFAULT_BUDGET_S = 900.0

#: One frame's own ceiling, above `vision.timeout_s` so the server's timeout
#: is the one that fires. This exists to catch a hung connection, not a slow
#: model.
FRAME_TIMEOUT_S = 180


class DescribeFailedError(RuntimeError):
    """The describe endpoint refused, in a way that will not fix itself."""


def describe_frames(
    frames: list[SelectedFrame],
    *,
    user: str,
    post: Callable[..., tuple[int, dict]],
    endpoint: str,
    token: str,
    budget_s: float | None = DEFAULT_BUDGET_S,
) -> tuple[list[dict], int]:
    """Describe each frame. Returns `(descriptions, planned)`.

    `planned` is `len(frames)`, so a caller can see that the budget cut the
    pass short rather than inferring it from a count that looks complete.

    `post` is injected rather than imported so this stays testable without a
    server and without patching a module global — the worker passes its own.

    A single frame failing does not fail the pass. One unreadable frame in a
    hundred is a gap in coverage; refusing the whole video over it would throw
    away ninety-nine good descriptions and the transcript besides. A failure
    that repeats will show up as an empty result, which the caller can see.
    """
    described: list[dict] = []
    started = time.monotonic()

    for i, frame in enumerate(frames):
        # `is not None`, not truthiness. A budget of 0.0 means the lease is
        # already spent and nothing may be described — under a falsy check
        # that reads as "no budget configured" and describes *everything*,
        # which is the exact failure the caller computed a 0.0 to prevent.
        if budget_s is not None and (time.monotonic() - started) >= budget_s:
            log.warning(
                "describe: budget of %.0fs spent after %d/%d frames — "
                "posting what we have", budget_s, len(described), len(frames),
            )
            break

        try:
            payload = _read_b64(frame.path)
        except OSError as e:
            log.warning("describe: could not read %s: %s", frame.path.name, e)
            continue

        status, body = post(
            endpoint, "/v1/media/describe", token,
            {"user": user, "image_b64": payload, "mime": "image/jpeg"},
            timeout=FRAME_TIMEOUT_S,
        )
        if status != 200:
            # 503 means no vl model is healthy. That is not going to change
            # within this job, and grinding through the remaining frames would
            # spend the lease to collect the same error N more times.
            if status == 503:
                raise DescribeFailedError(
                    f"vision is unavailable: {body.get('detail', status)}")
            log.warning(
                "describe: frame %d/%d rejected (%s): %s",
                i + 1, len(frames), status, body.get("detail", ""),
            )
            continue

        text = str(body.get("description") or "").strip()
        if text:
            described.append({
                "t_start": frame.t_start, "t_end": frame.t_end, "text": text,
            })

    return described, len(frames)


def _read_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


__all__ = ["DEFAULT_BUDGET_S", "DescribeFailedError", "describe_frames"]
