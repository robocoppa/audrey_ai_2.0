"""Show what a keyframe describe call ACTUALLY returns (Phase 38).

Phase 38 spent two deploys reasoning about a characters-per-token ratio and
was wrong both times. A describe call generates ~1,000 tokens and returns
~250 characters; dividing one by the other and arguing about the remainder
produced a confident thinking-token diagnosis, a `think: false` that changed
nothing measurable, and a `num_predict` cap that silently dropped three of six
keyframes.

This prints the response instead — and now the thinking text itself, because
"where did 900 tokens go" is a question with a literal answer sitting in the
payload.

## What the first run established (2026-08-04, qwen3-vl:32b)

**The tokens are thinking, and `think: false` does not stop it.**

    a.jpg  think-false   325tok  content=478ch   thinking=947ch    both=4.38/tok
    b.jpg  think-false  2496tok  content=350ch   thinking=8476ch   both=3.54/tok

`content + thinking` came to **3.4-4.4 characters per token in every variant**,
exactly the ratio prose runs at — so the accounting closes and nothing
unexplained is left. It also settles the per-frame variance: `b.jpg` (an office
scene with text on screen) produced **ten times** the thinking of `a.jpg` (a
static two-shot). Ollama accepted `think: false` without error and the model
reasoned anyway.

## What each variant isolates

    think-false     what is deployed
    think-true      the control. Thinking is present either way, so this is
                    what says whether the API field does anything at all.
    no_think-tag    Qwen3's documented `/no_think` prompt-level soft switch, a
                    different mechanism from the API field — worth trying
                    precisely because the API field is not landing.
    no_think+false  both together, in case they are additive
    screenshot      the old DESCRIBE_SYSTEM, to separate prompt cost from
                    sampling cost

Every variant runs at the DEPLOYED sampling settings. The first pass of this
probe sent neither `temperature` nor `num_predict` and ran at the model default
of 1.0 — it reported `think: true` as *cheaper* than `think: false`, which is
backwards and was almost certainly variance. Hence `SAMPLES`.

## Running it

The worker cannot reach Ollama (its compose network is `internal: true`, by
design), so this runs inside `audrey-ai`, the only container on both networks.
Fed over stdin so no rebuild is needed:

    # Unraid box
    docker exec -i -e IMAGES=/tmp/a.jpg,/tmp/b.jpg audrey-ai python3 - \
        < scripts/vision_probe.py

Getting real frames in there is two steps, because `audrey-ai` has no ffmpeg
(that is the whole reason the sidecar exists) and the worker deletes its frames
when a job ends:

    # Unraid box — pull two frames out of a source video
    docker exec media-worker sh -c \
      'ffmpeg -loglevel error -ss 30 -i /data/uploads/<user>/<file_id>.mp4 \
         -frames:v 1 -vf scale=1280:-2 -q:v 3 /tmp/a.jpg && \
       ffmpeg -loglevel error -ss 510 -i /data/uploads/<user>/<file_id>.mp4 \
         -frames:v 1 -vf scale=1280:-2 -q:v 3 /tmp/b.jpg'
    docker cp media-worker:/tmp/a.jpg /tmp/a.jpg && docker cp /tmp/a.jpg audrey-ai:/tmp/a.jpg
    docker cp media-worker:/tmp/b.jpg /tmp/b.jpg && docker cp /tmp/b.jpg audrey-ai:/tmp/b.jpg

Pick the two deliberately: one from a static stretch, one from a scene with
text on screen. Thinking differed 10x between those two cases, so a probe of
only one kind answers half the question.

Environment:

    IMAGES      comma-separated paths inside the container (default /tmp/probe.jpg)
    MODEL       default qwen3-vl:32b
    OLLAMA_HOST default http://ollama:11434
    HINT        transcript context to send, as the worker now does
    TIMEOUT_S   default 180
    SAMPLES     runs per variant (default 2) — thinking length is noisy
    NUM_PREDICT default 2048, matching config.yaml
    TEMPERATURE default 0.3, matching config.yaml
    ONLY        comma-separated variant names, to re-run just one
    EXCERPT     characters of thinking/content to print (default 500, 0 = off)
"""

from __future__ import annotations

import base64
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
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

#: Match `config.yaml`'s `vision:` block, so a result here transfers to the
#: deployed path rather than describing a configuration nobody runs.
_NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "2048"))
_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))

#: Runs per variant. Thinking length is noisy enough that one sample cannot
#: rank two variants — it can only show a difference of the order seen between
#: a.jpg (913ch) and b.jpg (9,884ch).
_SAMPLES = int(os.environ.get("SAMPLES", "2"))

_EXCERPT = int(os.environ.get("EXCERPT", "500"))

#: A variant that leaves thinking under this fraction of `think-false`'s is
#: reported as suppressing it. Deliberately loose: the useful answer is "this
#: mechanism works at all", and a 5x reduction settles that without needing to
#: distinguish 3% from 8% out of two samples.
_SUPPRESSED_AT = 0.25


@dataclass
class Variant:
    name: str
    system: str
    think: bool | None
    suffix: str = ""


VARIANTS: list[Variant] = [
    Variant("think-false", KEYFRAME_SYSTEM, False),
    Variant("think-true", KEYFRAME_SYSTEM, True),
    Variant("no_think-tag", KEYFRAME_SYSTEM, None, " /no_think"),
    Variant("no_think+false", KEYFRAME_SYSTEM, False, " /no_think"),
    Variant("screenshot", DESCRIBE_SYSTEM, False),
]


@dataclass
class Result:
    image: str
    variant: str
    ok: bool = True
    error: str = ""
    tokens: int = 0
    content: str = ""
    thinking: str = ""
    eval_s: float = 0.0
    done_reason: str = ""
    extra_keys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def both_per_tok(self) -> float:
        """Characters of content+thinking per generated token.

        The number that closes the accounting. Near 4 means every token is
        explained by text we can see; well under it means something is being
        generated that neither field carries, and the extra keys are where to
        look next.
        """
        return (len(self.content) + len(self.thinking)) / self.tokens if self.tokens else 0.0


def load(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call(b64: str, variant: Variant, image: str) -> Result:
    """One /api/chat call, built exactly as `_transcribe_one` builds it."""
    hint = TRANSCRIPT_HINT.format(hint=HINT) if HINT.strip() else ""
    payload: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": variant.system},
            {
                "role": "user",
                "content": f"Describe this image.{hint}{variant.suffix}",
                "images": [b64],
            },
        ],
        "stream": False,
        "options": {"num_predict": _NUM_PREDICT, "temperature": _TEMPERATURE},
    }
    if variant.think is not None:
        payload["think"] = variant.think

    try:
        r = httpx.post(f"{HOST}/api/chat", json=payload, timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        # One variant erroring must not lose the other nineteen calls' data.
        return Result(image, variant.name, ok=False, error=f"{type(e).__name__}: {e}")
    if r.status_code != 200:
        return Result(image, variant.name, ok=False,
                      error=f"HTTP {r.status_code}: {r.text[:300]}")

    body = r.json()
    message = body.get("message") or {}
    thinking = message.get("thinking")
    return Result(
        image=image,
        variant=variant.name,
        tokens=int(body.get("eval_count") or 0),
        content=str(message.get("content") or ""),
        thinking=thinking if isinstance(thinking, str) else "",
        eval_s=float(body.get("eval_duration") or 0) / NS,
        done_reason=str(body.get("done_reason") or ""),
        extra_keys=tuple(sorted(set(message) - {"role", "content", "thinking"})),
    )


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarise(results: list[Result], images: list[str]) -> None:
    print("\n" + "=" * 78)
    print("PER-VARIANT MEANS")
    print("=" * 78)
    for image in images:
        print(f"\n{image}")
        print(f"  {'variant':<16} {'gen':>7} {'tokens':>8} {'content':>9} "
              f"{'thinking':>10} {'both/tok':>9}")
        for variant in VARIANTS:
            rows = [r for r in results
                    if r.image == image and r.variant == variant.name and r.ok]
            if not rows:
                failed = [r for r in results
                          if r.image == image and r.variant == variant.name]
                if failed:
                    print(f"  {variant.name:<16} ERROR  {failed[0].error[:50]}")
                continue
            print(
                f"  {variant.name:<16} "
                f"{_mean([r.eval_s for r in rows]):6.1f}s "
                f"{_mean([float(r.tokens) for r in rows]):8.0f} "
                f"{_mean([float(len(r.content)) for r in rows]):8.0f}ch "
                f"{_mean([float(len(r.thinking)) for r in rows]):9.0f}ch "
                f"{_mean([r.both_per_tok for r in rows]):9.2f}"
            )
            truncated = [r for r in rows if r.done_reason == "length"]
            if truncated:
                print(f"  {'':<16} ^ {len(truncated)}/{len(rows)} hit num_predict "
                      f"({_NUM_PREDICT}) — description was CUT")
            extras = {k for r in rows for k in r.extra_keys}
            if extras:
                print(f"  {'':<16} ^ message also carries: {sorted(extras)}")


def verdict(results: list[Result], images: list[str]) -> None:
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    ok = [r for r in results if r.ok]
    if not ok:
        print("Every call failed — nothing to conclude.")
        return

    # Only meaningful on a substantial generation. A 30-token reply swings
    # this ratio on rounding alone, and reporting that as "unexplained tokens"
    # would manufacture the exact mystery the probe exists to dispel.
    unexplained = [r for r in ok if r.tokens >= 100 and r.both_per_tok < 2.0]
    if unexplained:
        print(
            f"\n{len(unexplained)}/{len(ok)} calls generated tokens that neither "
            "content nor thinking\naccounts for (both/tok < 2.0). Something else "
            "is being generated — look at\nthe extra message keys above before "
            "tuning anything."
        )
    else:
        print(
            "\nAccounting closes: content + thinking explains every generated "
            "token in\nevery call. There is no third thing to find."
        )

    for image in images:
        base = [r for r in ok
                if r.image == image and r.variant == "think-false"]
        if not base:
            continue
        baseline = _mean([float(len(r.thinking)) for r in base])
        print(f"\n{image} — think-false thinking: {baseline:.0f}ch")
        if baseline == 0:
            print("  No thinking at all. `think: false` is honoured here.")
            continue
        for variant in VARIANTS:
            if variant.name == "think-false":
                continue
            rows = [r for r in ok if r.image == image and r.variant == variant.name]
            if not rows:
                continue
            mean_think = _mean([float(len(r.thinking)) for r in rows])
            ratio = mean_think / baseline
            mark = "  <<< SUPPRESSES IT" if ratio <= _SUPPRESSED_AT else ""
            print(f"  {variant.name:<16} {mean_think:7.0f}ch  {ratio * 100:5.1f}% "
                  f"of baseline{mark}")

    print(
        "\nA variant marked SUPPRESSES IT is the mechanism to wire into "
        "`_transcribe_one`.\nIf none is marked, thinking cannot be turned off "
        "for this model and the choice\nis to accept the cost or move to a model "
        "whose `ollama show` omits `thinking`."
    )


def excerpts(results: list[Result], images: list[str]) -> None:
    """Print what the model actually thought and said.

    The point of the whole exercise. A token count says how much was spent; the
    text says on what — and whether it is something a prompt could prevent.
    """
    if _EXCERPT <= 0:
        return
    print("\n" + "=" * 78)
    print(f"WHAT IT ACTUALLY WROTE (first {_EXCERPT} chars)")
    print("=" * 78)
    for image in images:
        for variant in VARIANTS:
            row = next((r for r in results
                        if r.image == image and r.variant == variant.name
                        and r.ok and r.thinking), None)
            if row is None:
                continue
            print(f"\n── {image} · {variant.name} · thinking "
                  f"({len(row.thinking)}ch) " + "─" * 20)
            print(row.thinking[:_EXCERPT].strip())
            print(f"\n── {image} · {variant.name} · content "
                  f"({len(row.content)}ch) " + "─" * 21)
            print(row.content[:_EXCERPT].strip())
            break  # one variant per image is enough to see the shape


def main() -> int:
    paths = [
        Path(p.strip())
        for p in os.environ.get("IMAGES", _DEFAULT_IMAGE).split(",")
        if p.strip()
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"No such image(s): {[str(p) for p in missing]}")
        print("See this file's docstring for how to extract frames into the container.")
        return 2

    only = {v.strip() for v in os.environ.get("ONLY", "").split(",") if v.strip()}
    variants = [v for v in VARIANTS if not only or v.name in only]
    if not variants:
        print(f"ONLY={sorted(only)} matched no variant. "
              f"Known: {[v.name for v in VARIANTS]}")
        return 2

    total = len(paths) * len(variants) * _SAMPLES
    print(f"model={MODEL} host={HOST} hint={'yes' if HINT.strip() else 'no'}")
    print(f"num_predict={_NUM_PREDICT} temperature={_TEMPERATURE} samples={_SAMPLES}")
    print(f"{total} calls. At 10-80s each this is roughly "
          f"{total * 10 // 60}-{total * 80 // 60} minutes.\n")

    results: list[Result] = []
    started = time.monotonic()
    for path in paths:
        b64 = load(path)
        for variant in variants:
            for run in range(_SAMPLES):
                res = call(b64, variant, path.name)
                results.append(res)
                done = len(results)
                if not res.ok:
                    print(f"[{done}/{total}] {path.name:<8} {variant.name:<16} "
                          f"ERROR {res.error[:60]}")
                    continue
                print(
                    f"[{done}/{total}] {path.name:<8} {variant.name:<16} "
                    f"run{run + 1}  {res.eval_s:6.1f}s {res.tokens:5d}tok  "
                    f"content={len(res.content):5d}ch thinking={len(res.thinking):6d}ch"
                )

    summarise(results, [p.name for p in paths])
    verdict(results, [p.name for p in paths])
    excerpts(results, [p.name for p in paths])
    print(f"\nTotal probe time: {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
