"""Bounded media summary over transcripts and frame descriptions (Phase 37).

The smallest stage of media processing and the one that makes the rest legible.
A file list reading `jasonRetirement.mp4 · 288 MB · ready` tells you nothing
you did not already know.

## Why this runs in `audrey` and not in the worker

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

## Why the defaults now run locally

The original cloud default twice returned the writing instruction instead of a
description in live use. Rejecting that output prevented bad text from being
stored, but a second call to the same model left the video with no summary.

The current primary and fallback were checked with the production prompt and
the same synthetic video material. Both returned complete, natural
descriptions. They run through `FairLocalGate`, so summary work queues behind
interactive local chat rather than competing with it. That queue is accepted
because this stage is asynchronous and keeping private video-derived text on
the owner's machine is the safer default.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate

log = logging.getLogger(__name__)

SUMMARY_SYSTEM = (
    "Write a natural description for someone browsing their private video "
    "library. Use two or three concise, complete sentences. Explain the main "
    "subject, what happens or is demonstrated, and what a viewer can learn or "
    "take away. Use specific details from the video material and sound like a "
    "person who watched it. Start directly with the video; do not mention the "
    "user, the request, these instructions, sources, transcripts, frames, or "
    "being an AI. Do not write a title, heading, or bullets. Use only the "
    "material below; do not invent events between excerpts."
)

AUDIO_SUMMARY_SYSTEM = (
    "Write a natural description for someone browsing their private audio "
    "library. Use two or three concise, complete sentences. Explain the main "
    "subject, what is discussed or demonstrated, and what a listener can learn "
    "or take away. Use specific details from the recording and sound like a "
    "person who listened to it. Start directly with the recording; do not "
    "mention the user, the request, these instructions, sources, transcripts, "
    "or being an AI. Do not write a title, heading, or bullets. Use only the "
    "material below; do not invent events between excerpts."
)

DOCUMENT_SUMMARY_SYSTEM = (
    "Write a natural description for someone browsing their private document "
    "library. Use two or three concise, complete sentences. Explain the main "
    "subject, purpose or findings, and the most useful takeaway. Use specific "
    "details from the document and sound like a person who read it. Start "
    "directly with the document; do not mention the user, the request, these "
    "instructions, extraction, excerpts, source text, or being an AI. Do not "
    "write a title, heading, or bullets. Use only the material below; do not "
    "invent details that are not present."
)

SUMMARY_MAX_WORDS = 80
SUMMARY_MAX_CHARS = 560
SUMMARY_MAX_OUTPUT_TOKENS = 240

_PREAMBLE = re.compile(
    r"^(?:let me|i(?:'ll| will)) (?:analy[sz]e|summari[sz]e|review) "
    r"(?:this|the) (?:video|recording|audio|document|pdf)[.!:]?\s*|"
    r"^here(?:'s| is) (?:a |the )?(?:brief )?summary[.:]?\s*|"
    r"^(?:#+\s*)?(?:\*\*)?summary(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)

_INSTRUCTION_ECHO = re.compile(
    r"^(?:the user (?:wants|asks|asked|requested)|"
    r"the (?:task|request|prompt|instructions?) (?:asks?|requires?|is)|"
    r"(?:please )?write (?:a|the)|"
    r"(?:provide|create) (?:a|the) (?:brief|short|natural) |"
    r"(?:a|the) (?:brief|short) library description (?:should|must))",
    re.IGNORECASE,
)


def brief_video_summary(raw: str) -> str:
    """Keep only a short file description, never an echo of the assignment."""
    result = " ".join(raw.split())
    for _ in range(3):
        trimmed = _PREAMBLE.sub("", result, count=1).strip()
        if trimmed == result:
            break
        result = trimmed
    if _INSTRUCTION_ECHO.match(result):
        return ""

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", result)
    kept: list[str] = []
    for sentence in sentences[:3]:
        candidate = " ".join([*kept, sentence])
        if len(candidate.split()) <= SUMMARY_MAX_WORDS and len(candidate) <= SUMMARY_MAX_CHARS:
            kept.append(sentence)
        else:
            break
    result = " ".join(kept) if kept else sentences[0]
    if len(result.split()) > SUMMARY_MAX_WORDS:
        result = " ".join(result.split()[:SUMMARY_MAX_WORDS]).rstrip(".,;:") + "…"
    if len(result) > SUMMARY_MAX_CHARS:
        result = result[: SUMMARY_MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
    return result

#: Characters of transcript + descriptions handed to the model. A two-hour
#: transcript is ~100k characters and will not fit any context we want to pay
#: for, so the input is bounded here rather than discovered by a truncation
#: partway through a sentence. 24k is comfortably inside every model in the
#: pool and is roughly 45 minutes of speech.
DEFAULT_INPUT_BUDGET = 24_000

DEFAULT_MODEL = "qwen3.8:latest"
DEFAULT_FALLBACK_MODEL = "ornith-1.5:35b"
DEFAULT_TIMEOUT_S = 180.0


class SummaryUnavailableError(RuntimeError):
    """No usable summariser. A missing field, never a failed media file."""


def _thin(lines: list[str], budget: int) -> tuple[list[str], bool]:
    """Reduce `lines` to fit `budget` characters, spread evenly.

    Evenly rather than truncating the tail, for the same reason the keyframe
    cap spreads its losses: a recording summarised from its first fifteen
    minutes is confidently wrong about the other forty-five, and says nothing
    to indicate it. Sampling across the whole thing keeps the summary's
    coverage proportional to the media.

    Returns `(lines, was_thinned)` so the caller can tell the model it is
    reading excerpts — a model that thinks it has the whole transcript will
    happily assert what the video concluded.
    """
    total = sum(len(x) + 1 for x in lines)
    if total <= budget or not lines:
        return lines, False

    keep = max(1, int(len(lines) * budget / total))
    if keep == 1:
        return [lines[0]], True
    step = (len(lines) - 1) / (keep - 1)
    picked = [lines[round(i * step)] for i in range(keep)]
    return picked, True


def build_input(
    segments: list[dict],
    frames: list[dict],
    *,
    budget: int = DEFAULT_INPUT_BUDGET,
    media_kind: str = "video",
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
        if media_kind == "document":
            note = " (excerpts, evenly sampled across the document)" if spoken_cut else ""
            parts.append(f"DOCUMENT TEXT{note}:\n" + "\n".join(spoken))
        else:
            subject = "recording" if media_kind == "audio" else "video"
            note = f" (excerpts, evenly sampled across the {subject})" if spoken_cut else ""
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
    media_kind: str = "video",
) -> str:
    """Summarise both artifacts with one configured cross-model fallback.

    Raises SummaryUnavailableError when there is nothing to summarise or both
    models return unusable text. A primary OllamaError falls through to the
    fallback; a fallback OllamaError is left for the caller to swallow. By this
    point the transcript and descriptions are already ingested and useful, so a
    summary failure is a missing field and never a failed row.
    """
    if media_kind not in {"video", "audio", "document"}:
        raise ValueError(f"unsupported summary kind: {media_kind}")
    video_cfg = _cfg(cfg)
    material = build_input(
        segments,
        frames,
        budget=int(video_cfg.get("summary_input_chars", DEFAULT_INPUT_BUDGET)),
        media_kind=media_kind,
    )
    if not material:
        raise SummaryUnavailableError("no transcript or descriptions to summarise")

    primary_model = str(video_cfg.get("summarise_model") or DEFAULT_MODEL).strip()
    fallback_model = str(
        video_cfg.get("summarise_fallback_model") or DEFAULT_FALLBACK_MODEL
    ).strip()
    models = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models.append(fallback_model)

    minutes = duration_s / 60.0
    media_label = {
        "audio": "Audio recording",
        "document": "Document",
        "video": "Video file",
    }[media_kind]
    material_label = {
        "audio": "AUDIO MATERIAL",
        "document": "DOCUMENT MATERIAL",
        "video": "VIDEO MATERIAL",
    }[media_kind]
    header = (
        f"{media_label}: {filename}\n"
        f"Length: {minutes:.0f} minutes\n\n"
        if duration_s
        else f"{media_label}: {filename}\n\n"
    )
    user_content = (
        header + material_label + "\n\n" + material
        + f"\n\nEND {material_label}"
    )
    timeout_s = float(video_cfg.get("summary_timeout_s", DEFAULT_TIMEOUT_S))

    for attempt, model in enumerate(models):
        location = registry.location_of(model)
        think = await _think_flag(ollama, model, cfg)

        # Both defaults are local and must queue behind interactive chat in
        # the uploader's own fair-gate slice. A configured cloud replacement
        # makes this acquire a no-op.
        try:
            async with gate.acquire(model, location=location, user_id=user_id):
                resp = await ollama.chat(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                AUDIO_SUMMARY_SYSTEM
                                if media_kind == "audio"
                                else DOCUMENT_SUMMARY_SYSTEM
                                if media_kind == "document"
                                else SUMMARY_SYSTEM
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ],
                    timeout_s=timeout_s,
                    think=think,
                    options={"num_predict": SUMMARY_MAX_OUTPUT_TOKENS},
                )
        except OllamaError as exc:
            if attempt + 1 < len(models):
                log.warning(
                    "summarise: %s failed for %s; falling back to %s: %s",
                    model,
                    filename,
                    models[attempt + 1],
                    exc,
                )
                continue
            raise

        raw = str((resp.get("message") or {}).get("content") or "")
        text = brief_video_summary(raw)
        if not text:
            if attempt + 1 < len(models):
                log.warning(
                    "summarise: %s returned no usable description for %s; "
                    "falling back to %s",
                    model,
                    filename,
                    models[attempt + 1],
                )
                continue
            names = " and ".join(models)
            raise SummaryUnavailableError(names + " returned no usable summary")

        # Requested and returned thinking are different facts. A model can
        # declare the capability and still ignore the requested setting.
        thinking = str((resp.get("message") or {}).get("thinking") or "")
        log.info(
            "summarise: %s -> %d chars via %s (%d segments, %d descriptions) "
            "think=%s thinking=%dc eval=%s",
            filename,
            len(text),
            model,
            len(segments),
            len(frames),
            "unset" if think is None else think,
            len(thinking),
            resp.get("eval_count", "?"),
        )
        return text

    raise SummaryUnavailableError("no summary model was configured")


def _document_units(text: str, *, max_chars: int = 600) -> list[str]:
    """Split extracted prose into bounded units before even sampling it."""
    remaining = " ".join(text.split())
    units: list[str] = []
    while remaining:
        if len(remaining) <= max_chars:
            units.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        units.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    return units


async def summarise_document(
    text: str,
    *,
    filename: str,
    ollama: OllamaClient,
    registry: ModelRegistry,
    gate: FairLocalGate,
    cfg: Any,
    user_id: str | None = None,
) -> str:
    """Summarise extracted document text through the bounded summary path."""
    segments = [{"text": unit} for unit in _document_units(text)]
    return await summarise_video(
        segments,
        [],
        filename=filename,
        duration_s=0.0,
        ollama=ollama,
        registry=registry,
        gate=gate,
        cfg=cfg,
        user_id=user_id,
        media_kind="document",
    )


async def _think_flag(ollama: Any, model: str, cfg: Any) -> bool | None:
    """Whether to send `think`, and what. `None` means do not send the field.

    ## Why this role turns thinking off

    Summarising is the clearest case in the registry of reasoning that is
    generated and discarded: the summary is the product and the reasoning is
    never shown. The comparative measurement was made on the former GLM cloud
    summarizer in 2026-08-06:

        omitted   27.7s   8994c thinking   2192c summary   2683 eval tok
        false      9.7s*     0c            3542c summary    817 eval tok

    The current Qwen and Ornith probe did not repeat that comparison, but it
    confirmed that think=false is accepted by both, produces no thinking text,
    and still yields complete summaries. For local models the avoided work is
    GPU time rather than cloud billing.

    ## Why it asks Ollama first

    ⚠️ **Sending `think` to a model that does not declare `thinking` is a hard
    error** (`OllamaClient.capabilities`), so this cannot be a flat `False`.
    Either configured summary model may be swapped for one that cannot think,
    which would turn every summary into a failure. Because the caller swallows
    SummaryUnavailableError by design, the failure would surface as summaries
    silently never appearing.

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
    "AUDIO_SUMMARY_SYSTEM",
    "DOCUMENT_SUMMARY_SYSTEM",
    "SummaryUnavailableError",
    "brief_video_summary",
    "build_input",
    "summarise_document",
    "summarise_video",
]
