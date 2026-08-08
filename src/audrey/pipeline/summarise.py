"""One-call video summary over the transcript and frame descriptions (Phase 37).

The smallest stage of the video work and the one that makes the rest legible.
A file list reading `jasonRetirement.mp4 · 288 MB · ready` tells you nothing
you did not already know.

## Why this runs in `audrey-ai` and not in the worker

The phase plan put it in `media/summarise.py`, calling passthrough "acting-as
the uploader as in phase 36". Phase 36 discovered there is no act-as on
`/v1/chat/completions` and answered it with a narrow service route. A second
route would work here too — and would be the wrong shape.

A summary is derived from the *artifacts*, not from the video file. By the
time one can be written, `ingest_result` is already holding the segments and
the descriptions in memory. Asking the worker to do it would mean shipping the
whole transcript to a summarise endpoint and then shipping it again in the
result post, to produce something the worker never looks at.

So the worker's job ends where the artifacts end, and this runs where they
land.

## Why a cloud model is the right default

Summarising is a text task over text inputs, so it carries none of the
image-capability risk that made the `vl` pool local-only — the failure that
prompted that rule was a model answering an image question blind, and there is
no image here. `FairLocalGate.acquire` is a no-op for a non-local location,
so this is the one stage of video ingest that costs the box no GPU at all.

It should stay that way. A local default here would put a summary in the same
queue as the chat turn waiting behind it, for a stage nobody is waiting on.
"""

from __future__ import annotations

import logging
from typing import Any

from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate

log = logging.getLogger(__name__)

SUMMARY_SYSTEM = (
    "You are summarising a video for someone deciding whether to watch it and "
    "for a search index that will store your summary as one chunk.\n"
    "- Open with one sentence saying what the video IS — the kind of thing it "
    "is, who is in it, and what happens.\n"
    "- Then the substance: what is discussed, what is shown, what is decided "
    "or concluded. Name people, places, products and figures that appear.\n"
    "- Write plain prose in one or two short paragraphs. No headings, no "
    "bullet lists, no preamble like 'This video shows'.\n"
    "- Use only what is in the material below. Do not speculate about what "
    "happens off-camera or between the excerpts, and do not describe the "
    "material itself ('the transcript mentions')."
)

#: Characters of transcript + descriptions handed to the model. A two-hour
#: transcript is ~100k characters and will not fit any context we want to pay
#: for, so the input is bounded here rather than discovered by a truncation
#: partway through a sentence. 24k is comfortably inside every model in the
#: pool and is roughly 45 minutes of speech.
DEFAULT_INPUT_BUDGET = 24_000

DEFAULT_MODEL = "glm-5.2:cloud"
DEFAULT_TIMEOUT_S = 180.0


class SummaryUnavailableError(RuntimeError):
    """No usable summariser. A missing field, never a failed video."""


def _thin(lines: list[str], budget: int) -> tuple[list[str], bool]:
    """Reduce `lines` to fit `budget` characters, spread evenly.

    Evenly rather than truncating the tail, for the same reason the keyframe
    cap spreads its losses: a video summarised from its first fifteen minutes
    is confidently wrong about the other forty-five, and says nothing to
    indicate it. Sampling across the whole thing keeps the summary's coverage
    proportional to the video's.

    Returns `(lines, was_thinned)` so the caller can tell the model it is
    reading excerpts — a model that thinks it has the whole transcript will
    happily assert what the video concluded.
    """
    total = sum(len(x) + 1 for x in lines)
    if total <= budget or not lines:
        return lines, False

    keep = max(1, int(len(lines) * budget / total))
    step = len(lines) / keep
    picked = [lines[min(len(lines) - 1, int(i * step))] for i in range(keep)]
    return picked, True


def build_input(
    segments: list[dict],
    frames: list[dict],
    *,
    budget: int = DEFAULT_INPUT_BUDGET,
) -> str:
    """Lay the two artifacts out for the model, labelled and bounded.

    Labelled because they answer different questions and the model should not
    blend them: the transcript is what was *said*, the descriptions are what
    was *shown*. A summary that reports a whiteboard as something someone
    stated is worse than one that omits it.
    """
    spoken = [str(s.get("text") or "").strip() for s in segments]
    spoken = [s for s in spoken if s]
    shown = [str(f.get("text") or "").strip() for f in frames]
    shown = [s for s in shown if s]

    # Split the budget by what is actually present, so a silent video gives
    # its whole allowance to the descriptions rather than reserving half of it
    # for a transcript that does not exist.
    if spoken and shown:
        spoken_budget, shown_budget = int(budget * 0.6), int(budget * 0.4)
    elif spoken:
        spoken_budget, shown_budget = budget, 0
    else:
        spoken_budget, shown_budget = 0, budget

    spoken, spoken_cut = _thin(spoken, spoken_budget)
    shown, shown_cut = _thin(shown, shown_budget)

    parts: list[str] = []
    if spoken:
        note = " (excerpts, evenly sampled across the video)" if spoken_cut else ""
        parts.append(f"WHAT WAS SAID{note}:\n" + "\n".join(spoken))
    if shown:
        note = " (excerpts, evenly sampled across the video)" if shown_cut else ""
        parts.append(f"WHAT WAS ON SCREEN{note}:\n\n" + "\n\n".join(shown))
    return "\n\n".join(parts)


async def summarise_video(
    segments: list[dict],
    frames: list[dict],
    *,
    filename: str,
    duration_s: float,
    ollama: OllamaClient,
    registry: ModelRegistry,
    gate: FairLocalGate,
    cfg: Any,
    user_id: str | None = None,
) -> str:
    """One model call over both artifacts. Returns the summary text.

    Raises `SummaryUnavailableError` when there is nothing to summarise or no
    model to do it with, and lets `OllamaError` through. Both are the caller's
    to swallow — by this point the transcript and descriptions are already
    ingested and already useful, so a summary failure is a missing field and
    never a failed row.
    """
    material = build_input(
        segments, frames,
        budget=int(_cfg(cfg).get("summary_input_chars", DEFAULT_INPUT_BUDGET)),
    )
    if not material:
        raise SummaryUnavailableError("no transcript or descriptions to summarise")

    model = str(_cfg(cfg).get("summarise_model") or DEFAULT_MODEL)
    location = registry.location_of(model)
    minutes = duration_s / 60.0
    header = (
        f"Video file: {filename}\n"
        f"Length: {minutes:.0f} minutes\n\n" if duration_s else f"Video file: {filename}\n\n"
    )

    think = await _think_flag(ollama, model, cfg)

    # A no-op for a cloud location, and correct rather than redundant for a
    # deployment that pins a local summariser: it would then queue behind chat
    # like anything else, in the uploader's own slice.
    async with gate.acquire(model, location=location, user_id=user_id):
        resp = await ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": header + material},
            ],
            timeout_s=float(_cfg(cfg).get("summary_timeout_s", DEFAULT_TIMEOUT_S)),
            think=think,
        )
    text = str((resp.get("message") or {}).get("content") or "").strip()
    if not text:
        raise SummaryUnavailableError(f"{model} returned an empty summary")
    # `think=` and `thinking=` are both here on purpose, and they are different
    # facts: the first is what we asked for, the second is what the model did.
    # A model that declares `thinking` and ignores the flag — `qwen3-vl:32b`
    # does exactly this — shows as `think=False thinking=8994c`, and without
    # both numbers side by side that is indistinguishable from the flag
    # working. `eval` is the billed total and includes reasoning tokens, which
    # is the whole reason this setting exists.
    thinking = str((resp.get("message") or {}).get("thinking") or "")
    log.info(
        "summarise: %s -> %d chars via %s (%d segments, %d descriptions) "
        "think=%s thinking=%dc eval=%s",
        filename, len(text), model, len(segments), len(frames),
        "unset" if think is None else think, len(thinking),
        resp.get("eval_count", "?"),
    )
    return text


async def _think_flag(ollama: Any, model: str, cfg: Any) -> bool | None:
    """Whether to send `think`, and what. `None` means do not send the field.

    ## Why this role turns thinking off

    Summarising is the clearest case in the whole registry of **reasoning that
    is billed and thrown away**: the summary is the product, the reasoning is
    never shown to anyone, and `summarise_model` defaults to a cloud model.
    Measured on `glm-5.2:cloud` 2026-08-06, three samples per state:

        omitted   27.7s   8994c thinking   2192c summary   2683 eval tok
        false      9.7s*     0c            3542c summary    817 eval tok

    \\* steady-state — the first `false` run was 45s on a cold cloud
    connection, the next two 9.6s and 9.8s.

    **3.3x fewer billed tokens, and a longer summary.** Quality did not suffer;
    if anything the non-thinking replies were more complete, and this task is
    condensation rather than analysis.

    ## Why it asks Ollama first

    ⚠️ **Sending `think` to a model that does not declare `thinking` is a hard
    error** (`OllamaClient.capabilities`), so this cannot be a flat `False`.
    `summarise_model` is deployment-configurable and the default may be swapped
    for a local model that cannot think — which would turn every summary into a
    failure, and `SummaryUnavailableError` is swallowed by design, so it would
    show up as summaries silently never appearing.

    A capability lookup that fails for any reason returns `None`: unknown means
    omit, never assume. That costs one `/api/show` per summary, against a call
    that already takes tens of seconds.
    """
    if not bool(_cfg(cfg).get("summary_no_thinking", True)):
        return None
    try:
        caps = await ollama.capabilities(model)
    except Exception:  # noqa: BLE001 — a probe failure must not fail a summary
        log.info("summarise: could not read %s capabilities, leaving think unset", model)
        return None
    return False if "thinking" in caps else None


def _cfg(cfg: Any) -> dict[str, Any]:
    raw = getattr(cfg, "raw", {}) or {}
    return ((raw.get("kb", {}) or {}).get("video", {}) or {})


__all__ = [
    "DEFAULT_INPUT_BUDGET",
    "SUMMARY_SYSTEM",
    "SummaryUnavailableError",
    "build_input",
    "summarise_video",
]
