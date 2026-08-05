"""Show what a keyframe describe call ACTUALLY returns (Phase 38).

Phase 38 spent two deploys reasoning about a characters-per-token ratio and
was wrong both times. A describe call generates ~1,000 tokens and returns
~250 characters; dividing one by the other and arguing about the remainder
produced a confident thinking-token diagnosis, a `think: false` that changed
nothing measurable, and a `num_predict` cap that silently dropped three of six
keyframes.

This prints the response instead. Every top-level key, every key inside
`message`, and the length of each — so "where did 900 tokens go" is a thing
you read rather than infer.

## Running it

The worker cannot reach Ollama (its compose network is `internal: true`, by
design), so this runs inside `audrey-ai`, which is the only container on both
networks. Fed over stdin so no rebuild is needed:

    # Unraid box
    docker exec -i -e IMAGES=/tmp/a.jpg,/tmp/b.jpg audrey-ai python3 - \
        < scripts/vision_probe.py

Getting real frames in there is two steps, because `audrey-ai` has no ffmpeg
(that is the whole reason the sidecar exists) and the worker deletes its
frames when a job ends:

    # Unraid box — pull two frames out of a source video
    docker exec media-worker sh -c \
      'ffmpeg -loglevel error -ss 30 -i /data/uploads/<user>/<file_id>.mp4 \
         -frames:v 1 -vf scale=1280:-2 -q:v 3 /tmp/a.jpg && \
       ffmpeg -loglevel error -ss 510 -i /data/uploads/<user>/<file_id>.mp4 \
         -frames:v 1 -vf scale=1280:-2 -q:v 3 /tmp/b.jpg'
    docker cp media-worker:/tmp/a.jpg /tmp/a.jpg && docker cp /tmp/a.jpg audrey-ai:/tmp/a.jpg
    docker cp media-worker:/tmp/b.jpg /tmp/b.jpg && docker cp /tmp/b.jpg audrey-ai:/tmp/b.jpg

Pick the two deliberately: one frame from a static stretch and one from a
scene with text on screen. The token/character ratio measured 1.1-1.5 on
static frames and 0.23-0.36 on new scenes with text, so a probe of only one
kind answers half the question.

## What each variant isolates

    baseline      think omitted entirely — the pre-phase-38 request
    think-false   what is deployed now
    think-true    thinking asked for explicitly, to prove the field is
                  honoured at all. If `think: false` and `think: true` return
                  the same token count, the model is ignoring the flag and no
                  amount of config will change that.
    screenshot    the old DESCRIBE_SYSTEM, to separate prompt cost from
                  sampling cost

Environment:

    IMAGES      comma-separated paths inside the container (default /tmp/probe.jpg)
    MODEL       default qwen3-vl:32b
    OLLAMA_HOST default http://ollama:11434
    HINT        transcript context to send, as the worker now does
    TIMEOUT_S   default 180
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import httpx

from audrey.pipeline.vision import DESCRIBE_SYSTEM, KEYFRAME_SYSTEM, TRANSCRIPT_HINT

MODEL = os.environ.get("MODEL", "qwen3-vl:32b")
HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
HINT = os.environ.get("HINT", "")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "180"))
NS = 1_000_000_000.0

#: Path inside the container, not on the host — the probe runs via
#: `docker exec`, so this is the container's own scratch space.
_DEFAULT_IMAGE = "/tmp/probe.jpg"  # noqa: S108


def load(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call(
    b64: str, *, system: str, think: bool | None,
    num_predict: int | None = None, temperature: float | None = None,
) -> dict:
    """One /api/chat call, built exactly as `_transcribe_one` builds it."""
    hint = TRANSCRIPT_HINT.format(hint=HINT) if HINT.strip() else ""
    payload: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Describe this image.{hint}",
                "images": [b64],
            },
        ],
        "stream": False,
    }
    options: dict = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if temperature is not None:
        options["temperature"] = temperature
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think

    r = httpx.post(f"{HOST}/api/chat", json=payload, timeout=TIMEOUT_S)
    if r.status_code != 200:
        return {"__error__": f"HTTP {r.status_code}: {r.text[:400]}"}
    return r.json()


def report(label: str, resp: dict) -> None:
    if "__error__" in resp:
        print(f"  {label:<14} ERROR  {resp['__error__']}")
        return

    message = resp.get("message") or {}
    content = str(message.get("content") or "")
    thinking = message.get("thinking")
    thinking = thinking if isinstance(thinking, str) else ""
    eval_count = int(resp.get("eval_count") or 0)
    eval_s = float(resp.get("eval_duration") or 0) / NS

    # The number the whole investigation turns on. If content alone is far
    # under the token count but content+thinking accounts for it, the tokens
    # are reasoning. If NEITHER accounts for it, they are something else and
    # the extra keys printed below are where to look.
    per_tok = (len(content) / eval_count) if eval_count else 0.0
    both_tok = ((len(content) + len(thinking)) / eval_count) if eval_count else 0.0

    print(
        f"  {label:<14} {eval_s:6.1f}s  {eval_count:5d}tok  "
        f"content={len(content):5d}ch ({per_tok:.2f}/tok)  "
        f"thinking={len(thinking):5d}ch  both={both_tok:.2f}/tok"
    )

    # Anything in the payload that is not accounted for above. A key nobody
    # expected is the most likely explanation left once thinking is ruled out.
    extra_msg = sorted(set(message) - {"role", "content", "thinking"})
    if extra_msg:
        print(f"                 message also carries: {extra_msg}")
    if resp.get("done_reason") not in (None, "stop"):
        print(f"                 done_reason={resp.get('done_reason')!r}")


def main() -> int:
    images = [
        Path(p.strip())
        for p in os.environ.get("IMAGES", _DEFAULT_IMAGE).split(",")
        if p.strip()
    ]
    missing = [p for p in images if not p.exists()]
    if missing:
        print(f"No such image(s): {[str(p) for p in missing]}")
        print("See this file's docstring for how to extract frames into the container.")
        return 2

    print(f"model={MODEL} host={HOST} hint={'yes' if HINT.strip() else 'no'}")
    print(
        "\nA large `thinking` is the thinking hypothesis confirmed. A small one "
        "beside a\nlow content/tok means the tokens are going somewhere else "
        "entirely — and\n`think-true` vs `think-false` says whether the flag "
        "does anything at all.\n"
    )

    for image in images:
        b64 = load(image)
        size_kb = image.stat().st_size // 1024
        print(f"── {image.name} ({size_kb} KB) " + "─" * 40)
        variants: list[tuple[str, dict]] = [
            ("baseline", {"system": KEYFRAME_SYSTEM, "think": None}),
            ("think-false", {"system": KEYFRAME_SYSTEM, "think": False}),
            ("think-true", {"system": KEYFRAME_SYSTEM, "think": True}),
            ("screenshot", {"system": DESCRIBE_SYSTEM, "think": False}),
        ]
        for label, kwargs in variants:
            report(label, call(b64, **kwargs))
        print()

    print(
        "If think-true and think-false report the same token count, the model "
        "ignores\nthe flag — stop tuning it and either accept the cost or move "
        "to a model whose\n`ollama show` does not list `thinking`."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
