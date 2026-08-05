"""Model access for the media worker (Phase 36).

    POST /v1/media/describe   — one keyframe in, prose out

## Why this route exists at all

The phase 36 plan had the worker call `POST /v1/chat/completions` with
`audrey_passthrough/qwen3-vl:32b`, "acting as the uploading user via the Phase
31 service-token act-as". That does not work: `chat_completions` depends on
`require_user`, which demands a real OWUI bearer. Phase 31's act-as is
`resolve_kb_caller`, and it lives only on the KB *query* routes. The worker
holds `KB_SERVICE_TOKEN` and cannot obtain a user JWT, so it would get a 401.

The fix is not to widen `require_user`. `/v1/chat/completions` is the endpoint
every OWUI user hits, and teaching it to accept a service token plus an act-as
header would put the entire chat surface behind a header that grants any
identity — to close a gap for one background client. This route is the narrow
alternative: service-token only, one verb, no message history, no streaming,
no model selection.

## What it preserves

The reason the plan wanted passthrough was fairness, and that survives intact.
`FairLocalGate` is in-process to `audrey-ai` and keys on `user_id`, so passing
the *uploader's* email puts the ingest in that user's round-robin slice: a
giant video slows its own owner's chat and leaves everyone else's alone. A
worker calling Ollama directly would share no gate at all and would starve the
box; a shared service identity would pool every ingest into one slice and blur
exactly the distinction the gate exists to draw.

`UserInflightRegistry` wraps the call for the same reason, mirroring what
`_handle_passthrough` does around its forwards.

## Why the caller sends base64, not a URL

`ollama.py` silently drops `http(s)://` image URLs, which yields a confidently
blind answer rather than an error — found the hard way during phase 32's
manual testing. Taking raw base64 and building the `data:` URI here means a
caller cannot make that mistake: there is no field to put a URL in.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from audrey.auth import require_service
from audrey.metrics import video_describe_seconds, vision_eval_tokens, vision_stage_seconds
from audrey.models.ollama import OllamaError
from audrey.pipeline.vision import VisionTiming, VisionUnavailableError, describe_one_image

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/media", tags=["media"])

# A 1280px JPEG at quality 3 is 30-100 KB, so ~140 KB of base64. The cap is
# an order of magnitude above that: generous enough that a legitimately dense
# frame is never refused, small enough that a malformed caller cannot post a
# gigabyte into the event loop.
MAX_IMAGE_B64 = 8 * 1024 * 1024

ALLOWED_MIME = frozenset({"image/jpeg", "image/png", "image/webp"})


class DescribeRequest(BaseModel):
    user: str = Field(
        min_length=1, max_length=200,
        description=(
            "The uploading user's email. Not decoration — both fairness "
            "layers key on it, so this is what puts the ingest in that "
            "user's slice rather than in a shared one."
        ),
    )
    image_b64: str = Field(
        min_length=1, max_length=MAX_IMAGE_B64,
        description="Raw base64 image bytes, with no `data:` prefix.",
    )
    mime: str = Field(default="image/jpeg")
    hint: str = Field(
        default="", max_length=2000,
        description=(
            "Optional context, e.g. what was being said over this frame. "
            "Reaches the model as the 'question asked about this image', "
            "which makes it describe the relevant parts in more detail — it "
            "never licenses answering."
        ),
    )


class DescribeResponse(BaseModel):
    description: str
    model: str
    elapsed_s: float


def _record(timing: VisionTiming, wall_s: float) -> None:
    """Attribute one describe call across its four disjoint stages (Phase 38).

    Phase 36 shipped the wall clock and phase 38 has to spend it, so this is
    the measurement that decides which lever is worth building. `queue` is
    observed even at zero — a flat line at zero is the evidence that the gate
    is *not* the problem, and a stage that is only reported when it is large
    cannot say that.
    """
    vision_stage_seconds.labels(stage="queue").observe(timing.queue_s(wall_s))
    vision_stage_seconds.labels(stage="load").observe(timing.load_s)
    vision_stage_seconds.labels(stage="prompt_eval").observe(timing.prompt_eval_s)
    vision_stage_seconds.labels(stage="eval").observe(timing.eval_s)
    vision_eval_tokens.observe(timing.eval_tokens)


@router.post("/describe", response_model=DescribeResponse)
async def describe(
    req: DescribeRequest,
    request: Request,
    _: None = Depends(require_service),
) -> DescribeResponse:
    """Describe one image as the named user, through the `vl` pool and the gate."""
    app = request.app
    for attr in ("ollama", "registry", "health", "gate", "inflight"):
        if getattr(app.state, attr, None) is None:
            raise HTTPException(status_code=503, detail=f"{attr} is not initialized")

    if req.mime not in ALLOWED_MIME:
        raise HTTPException(
            status_code=422,
            detail=f"mime must be one of {sorted(ALLOWED_MIME)}, got {req.mime!r}",
        )
    # Validate here rather than letting Ollama discover it. A bad payload
    # would otherwise cost a GPU slot before failing.
    try:
        base64.b64decode(req.image_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"image_b64 is not valid base64: {e}") from e

    data_url = f"data:{req.mime};base64,{req.image_b64}"
    t0 = time.perf_counter()
    try:
        async with app.state.inflight.slot(req.user):
            description, model, timing = await describe_one_image(
                data_url,
                ollama=app.state.ollama, registry=app.state.registry,
                health=app.state.health, gate=app.state.gate, cfg=app.state.cfg,
                user_question=req.hint, user_id=req.user,
            )
    except VisionUnavailableError as e:
        # 503, not 502: the deployment has no vision model, and a worker that
        # retries later may well succeed.
        raise HTTPException(status_code=503, detail=str(e)) from e
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=f"vision model failed: {e}") from e

    elapsed = time.perf_counter() - t0
    video_describe_seconds.observe(elapsed)
    _record(timing, elapsed)
    if not description:
        # An empty description is a failure wearing the shape of a success. It
        # would be ingested as an empty chunk and nobody would ever notice.
        #
        # Name the cause when it is knowable. `done_reason == "length"` means
        # the model spent its entire `num_predict` budget on reasoning and
        # never began the description — measured on 2026-08-04, three of six
        # probe runs on a cluttered frame emitted zero characters after 2,048
        # tokens. That is a config problem with a one-line fix, and it is
        # indistinguishable from a broken vision model unless the error says
        # so. The generic wording sent a previous investigation at the lease.
        if timing.done_reason == "length":
            detail = (
                f"{model} hit the num_predict cap ({timing.eval_tokens} tokens) "
                f"while reasoning and never produced a description "
                f"({timing.thinking_chars} chars of thinking). Raise "
                f"vision.num_predict."
            )
        else:
            detail = f"{model} returned an empty description"
        log.warning("media: describe produced nothing for %s — %s", req.user, detail)
        raise HTTPException(status_code=502, detail=detail)
    # The breakdown goes in the log line, not only in Prometheus. Tuning this
    # means reading a handful of consecutive frames from one video and seeing
    # which number moved — a histogram aggregates exactly the per-frame
    # variation that phase 36 measured as a 4x spread and could not explain.
    # `think=` closes the gap between tokens billed and characters kept. A
    # describe that generates 1,156 tokens and returns 267 characters is
    # spending its time somewhere, and the only honest way to find out is to
    # print what the response actually carried rather than divide one number
    # by another and reason about the remainder.
    log.info(
        "media: described a frame for %s via %s in %.1fs (%d chars) "
        "queue=%.1fs load=%.1fs prefill=%.1fs/%dtok gen=%.1fs/%dtok think=%dch",
        req.user, model, elapsed, len(description),
        timing.queue_s(elapsed), timing.load_s,
        timing.prompt_eval_s, timing.prompt_tokens,
        timing.eval_s, timing.eval_tokens, timing.thinking_chars,
    )
    return DescribeResponse(description=description, model=model, elapsed_s=elapsed)


__all__ = ["router"]
