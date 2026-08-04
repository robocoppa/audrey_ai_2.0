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
from audrey.metrics import video_describe_seconds
from audrey.models.ollama import OllamaError
from audrey.pipeline.vision import VisionUnavailableError, describe_one_image

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
            description, model = await describe_one_image(
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
    if not description:
        # An empty description is a failure wearing the shape of a success. It
        # would be ingested as an empty chunk and nobody would ever notice.
        raise HTTPException(
            status_code=502,
            detail=f"{model} returned an empty description",
        )
    log.info(
        "media: described a frame for %s via %s in %.1fs (%d chars)",
        req.user, model, elapsed, len(description),
    )
    return DescribeResponse(description=description, model=model, elapsed_s=elapsed)


__all__ = ["router"]
