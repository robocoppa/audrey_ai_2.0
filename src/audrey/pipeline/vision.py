"""Vision sidecar — let text-only models answer about an attached image.

Only the models in the `vl` registry pool carry a vision encoder. Every
other model Audrey serves (`glm-5.2:cloud`, `qwen3.6:35b`, …) is
text-in/text-out: hand one an image and Ollama either rejects the request
or the model answers blind. That left two gaps.

  - A passthrough request to a text model with an attached image failed
    outright — the data URI became Ollama's `images: [...]` list and the
    model had nowhere to put it.
  - The pipeline dodged the problem by pinning *every* image turn to the
    `vl` pool, so picking `audrey_cloud` and uploading a screenshot got
    you the local vision model's answer, never the cloud model you asked
    for.

Both close the same way: split perception from reasoning. A local vision
model transcribes the image once, and the description is spliced into the
message list as ordinary text, in place of the `image_url` part. The
model the caller actually chose then answers from that text.

This is lossy next to real vision — a description is not the pixels — so
the transcription prompt leans hard on verbatim text capture, which is
what screenshots, diagrams, and error dumps actually need. Set
`vision.describe_for_text_models: false` to switch the whole thing off;
callers then fall back to the behaviour they had before.

Descriptions are cached per image because OWUI resends the entire history
— data URI included — on every turn. Without the cache, five turns about
one screenshot would pay for five vision calls.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate

log = logging.getLogger(__name__)

# The transcription prompt. Deliberately forbids answering: this model's
# only job is to be the eyes of the model downstream, and a vision model
# that starts reasoning produces a description contaminated with guesses
# the answering model can't tell apart from what was actually on screen.
DESCRIBE_SYSTEM = (
    "You are the eyes of another AI model that cannot see images. Describe the "
    "attached image so that model can answer questions about it without ever "
    "seeing it.\n"
    "- Transcribe every piece of visible text VERBATIM: preserve line breaks, "
    "code indentation, table rows, labels, legends, and axis values. Never "
    "paraphrase or summarize text that is legible.\n"
    "- Say what the image is (photo, screenshot, diagram, chart, scan, UI) and "
    "describe the layout — what sits where, and how the parts connect.\n"
    "- For charts, report the data values, or your best read of them.\n"
    "- For code, logs, or errors, reproduce them exactly, stack traces included.\n"
    "- Note what is visually distinctive: colours, arrows, highlighting, "
    "handwriting, redactions, damage, or anything that looks wrong.\n"
    "- Do NOT answer the user's question, offer opinions, or speculate about "
    "anything you cannot actually see. Describe only."
)

# The keyframe prompt (Phase 38). `DESCRIBE_SYSTEM` above is written for chat
# screenshots — error dumps, spreadsheets, diagrams — where "transcribe every
# legible character" is exactly right. Phase 36 reused it for video keyframes,
# and against real footage that turned out to be the single largest cost in
# video ingest.
#
# Measured on the box 2026-08-04. Generation was 87% of the visual pass, and
# the descriptions it produced looked like this:
#
#     **Layout and Key Elements:**
#     - **Two individuals** sit on chairs with red cushions and black metal
#       frames... Above the banner, a light beige wall displays a dark
#       brown/bronze metal decorative cross (star-shaped with scrollwork).
#
# Two separate wastes in one output. **Markdown**: the headings, bold markers
# and bullets cost tokens and buy nothing, because this text is stored as a
# retrieval chunk and never rendered — 1.9 characters per token against the ~4
# that plain prose runs at. **Inventory**: a decorative cross, a polo-shirt
# logo, two pieces of wood on a stand. Nobody searches for those, and the
# first two keyframes of a talking-head video opened with a byte-identical
# sentence about the backdrop.
#
# What a keyframe is actually *for* is the text on screen — the slide, the
# document, the whiteboard. That is what a transcript cannot capture and what
# someone will genuinely search. Everything else is a caption.
KEYFRAME_SYSTEM = (
    "You are describing one still frame from a video so that it can be found "
    "later by search. Another model will read your description and cannot see "
    "the frame.\n"
    "- Write PLAIN PROSE. No markdown, no headings, no bullet points, no bold "
    "or italic markers. This text is stored as a search chunk, never "
    "displayed.\n"
    "- Lead with any text visible in the frame, transcribed EXACTLY: slides, "
    "documents, whiteboards, captions, name plates, titles, signs, code, "
    "error messages. This is the most valuable thing you can record and often "
    "the only thing anyone will search for.\n"
    "- Then say briefly what is happening and who is present.\n"
    "- Do NOT inventory the scene. Furniture, clothing, decor, wall colours "
    "and background objects deserve a few words only when they identify the "
    "setting or carry meaning. A frame with no writing in it and nothing "
    "happening deserves one or two sentences, not a paragraph.\n"
    "- Describe only what is visible. Do not speculate, do not interpret the "
    "video's purpose, and do not comment on the footage."
)

# How a hint is framed to the model. The hint means something different on the
# two paths, and framing it wrongly is worse than omitting it.
#
# On the chat path it is the user's question, and the risk is the model
# answering it instead of describing.
QUESTION_HINT = (
    "\n\nFor context, the question asked about this image was: {hint}\n"
    "Describe the parts of the image that bear on it in extra detail — but "
    "still describe, do not answer."
)

# On the keyframe path there is no question — the description is written at
# ingest, and whatever anyone eventually asks arrives hours or days later. What
# *is* available is what was being said while the frame was on screen, which is
# the next best guide to what in the frame is worth the words.
#
# The final clause is load-bearing. Transcript and visual descriptions are
# ingested as separate chunks into the same collection, so a description that
# echoes the speech makes one query return the same sentences twice, under two
# artifacts, competing with each other for the same top_k.
TRANSCRIPT_HINT = (
    "\n\nFor context, this is what was being said in the video while this "
    "frame was on screen:\n{hint}\n"
    "Use it ONLY to judge what in the frame is worth describing. Do NOT "
    "repeat it back or summarise it — the spoken words are already stored "
    "separately, and repeating them here would return the same content twice "
    "for one search."
)

_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_MAX_IMAGES = 4
_DEFAULT_CACHE_SIZE = 64


def _sampling(conf: dict[str, Any]) -> tuple[dict[str, Any], bool | None]:
    """Build `(options, think)` for a vision call from the `vision:` block.

    Returns an empty options dict and `think=None` when nothing is configured,
    which reproduces the pre-phase-38 request byte for byte.

    **`think` is tri-state on purpose.** `None` omits the field; Ollama rejects
    it for a model without the `thinking` capability, so a plain boolean
    default would break any non-thinking member of the `vl` pool the moment
    someone added one. Absent config means absent field.
    """
    options: dict[str, Any] = {}
    if conf.get("num_predict") is not None:
        options["num_predict"] = int(conf["num_predict"])
    if conf.get("temperature") is not None:
        options["temperature"] = float(conf["temperature"])
    think = conf.get("think")
    return options, None if think is None else bool(think)

# Process-local LRU of image digest -> description. Bounded because the
# values are model prose, not the image bytes. Two concurrent requests
# carrying the same new image can both miss and both call the vision
# model; the work is idempotent and the race is rare enough not to be
# worth a per-key lock.
_cache: OrderedDict[tuple[str, str], str] = OrderedDict()


def vision_cfg(cfg: Any) -> dict[str, Any]:
    """Return the `vision:` config block (empty dict when absent)."""
    if cfg is None:
        return {}
    return (getattr(cfg, "raw", {}) or {}).get("vision", {}) or {}


def describe_enabled(cfg: Any) -> bool:
    """True when the sidecar may run. Defaults on."""
    return bool(vision_cfg(cfg).get("describe_for_text_models", True))


def vision_capable_models(cfg: Any) -> set[str]:
    """Model names that can read an image directly.

    The `vl` registry pool is the source of truth — it is already curated
    for exactly this property (see the comment above it in `config.yaml`,
    which is there because a text model once sat in that pool and answered
    image turns blind). `vision.also_capable` bolts on any model that can
    see but has no business in the vl pool.
    """
    names: set[str] = set()
    registry = (getattr(cfg, "model_registry", {}) or {}) if cfg is not None else {}
    for entry in registry.get("vl", []) or []:
        if isinstance(entry, dict) and entry.get("name"):
            names.add(str(entry["name"]))
    for extra in vision_cfg(cfg).get("also_capable", []) or []:
        names.add(str(extra))
    return names


def is_vision_capable(model: str, cfg: Any) -> bool:
    return model in vision_capable_models(cfg)


def has_any_image_part(messages: list[dict[str, Any]]) -> bool:
    """True if *any* message carries an image part.

    Broader on purpose than `messages.has_image_part`, which answers a
    different question — "is the turn being answered an image turn?" —
    and so only inspects the latest user message. Rewriting has to find
    every image in the history, including ones from earlier turns that
    the chosen text model still cannot read.
    """
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        if any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
            return True
    return False


def _image_url(part: dict[str, Any]) -> str:
    return str((part.get("image_url") or {}).get("url") or "")


def _digest(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()


def _cache_get(key: tuple[str, str]) -> str | None:
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
    return hit


def _cache_put(key: tuple[str, str], value: str, *, max_size: int) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > max(1, max_size):
        _cache.popitem(last=False)


def _described_block(idx: int, total: int, model: str, description: str) -> str:
    """Wrap a description so the answering model knows what it is holding.

    The framing matters as much as the text: without it, models either
    claim to have seen the image themselves or ask the user to attach it
    again — the image *is* attached, it just isn't readable by them.
    """
    return (
        f"[Attached image {idx} of {total} — you cannot see images, so the vision "
        f"model {model} transcribed this one for you. Treat the description below "
        f"as a faithful account of what the user attached; do not ask them to "
        f"send it again.]\n{description}"
    )


def _failed_block(idx: int, total: int, reason: str) -> str:
    return (
        f"[Attached image {idx} of {total} — could not be read ({reason}). Tell "
        f"the user their image could not be processed rather than guessing at "
        f"its contents.]"
    )


def _pick_vision_model(
    cfg: Any, registry: ModelRegistry, health: HealthTracker,
) -> tuple[str, str] | None:
    """Return (model_name, location) for the transcription call.

    `vision.model` pins an explicit choice; otherwise the highest-priority
    healthy member of the `vl` pool wins, which keeps the sidecar on the
    same health tracking and failover as every other dispatch.
    """
    pinned = vision_cfg(cfg).get("model")
    if pinned:
        return str(pinned), registry.location_of(str(pinned))
    spec = registry.first_healthy("vl", health.is_healthy)
    if spec is None:
        return None
    return spec.name, spec.location


#: Nanoseconds per second. Ollama reports every duration in nanoseconds, and
#: a factor of 1e9 read as 1e6 turns 62 seconds into 62 milliseconds — a
#: mistake that looks like good news rather than like a bug.
_NS = 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class VisionTiming:
    """Ollama's own account of where a vision call's time went (Phase 38).

    Every field is optional in the response — a backend that does not report
    them yields zeros rather than an exception, because a missing timing must
    never be able to fail a describe that otherwise worked.

    `queue_s` is the one field Ollama does not supply: it is the wall clock the
    caller measured minus the work Ollama says it did, which is time spent
    waiting on `FairLocalGate` and `UserInflightRegistry` rather than on the
    model. Keeping it separate is the whole point — 60 seconds of queue and 60
    seconds of generation are the same number on a stopwatch and want opposite
    fixes.
    """

    total_s: float = 0.0
    load_s: float = 0.0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    prompt_tokens: int = 0
    eval_tokens: int = 0

    @classmethod
    def from_response(cls, resp: dict[str, Any]) -> VisionTiming:
        def secs(key: str) -> float:
            try:
                return max(0.0, float(resp.get(key) or 0) / _NS)
            except (TypeError, ValueError):
                return 0.0

        def count(key: str) -> int:
            try:
                return max(0, int(resp.get(key) or 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            total_s=secs("total_duration"),
            load_s=secs("load_duration"),
            prompt_eval_s=secs("prompt_eval_duration"),
            eval_s=secs("eval_duration"),
            prompt_tokens=count("prompt_eval_count"),
            eval_tokens=count("eval_count"),
        )

    def queue_s(self, wall_s: float) -> float:
        """Wall clock not accounted for by Ollama, floored at zero.

        Returns 0.0 when the backend reported no `total_duration` at all:
        attributing the entire call to queueing because the timing fields were
        absent would invent the exact signal this exists to detect.
        """
        if self.total_s <= 0:
            return 0.0
        return max(0.0, wall_s - self.total_s)


async def _transcribe_one(
    ollama: OllamaClient,
    gate: FairLocalGate,
    *,
    url: str,
    model: str,
    location: str,
    user_question: str,
    timeout_s: float,
    user_id: str | None,
    options: dict[str, Any] | None = None,
    think: bool | None = None,
    system: str = DESCRIBE_SYSTEM,
    hint_template: str = QUESTION_HINT,
) -> tuple[str, VisionTiming]:
    """One vision call. Raises OllamaError; caller decides how to fail."""
    hint = (
        hint_template.format(hint=user_question.strip())
        if user_question.strip()
        else ""
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe this image.{hint}"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]
    async with gate.acquire(model, location=location, user_id=user_id):
        resp = await ollama.chat(
            model=model, messages=messages, timeout_s=timeout_s,
            options=options or None, think=think,
        )
    text = str((resp.get("message") or {}).get("content") or "").strip()
    return text, VisionTiming.from_response(resp)


class VisionUnavailableError(RuntimeError):
    """No healthy `vl` model. A deployment problem, not a problem with the image."""


async def describe_one_image(
    data_url: str,
    *,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    cfg: Any,
    user_question: str = "",
    user_id: str | None = None,
    system: str = KEYFRAME_SYSTEM,
    hint_template: str = TRANSCRIPT_HINT,
) -> tuple[str, str, VisionTiming]:
    """Describe one image through the `vl` pool.

    Returns `(description, model, timing)`.

    **Defaults to `KEYFRAME_SYSTEM`, not `DESCRIBE_SYSTEM`.** This entry point
    exists for phase 36's keyframe pass, and a video frame wants a different
    description from a pasted screenshot — see the comment on that constant.
    `describe_images` keeps the screenshot prompt, which is what a chat turn
    about an error dump actually needs.

    The single-image entry point, for callers that already hold an image and
    want prose back — Phase 36's keyframe pass. `describe_images` stays the
    entry point for chat, where images arrive buried in a message history that
    has to come out the other side intact.

    **Raises rather than degrading.** `describe_images` is fail-soft on
    purpose: a half-answered chat turn beats a 502, and a bracketed note tells
    the answering model to own the gap. An ingest has no user waiting and no
    reason to pretend — a frame that could not be described must be reported
    as a failure the worker can record, not ingested as searchable text saying
    a description was unavailable.

    **Deliberately does not use `_cache`.** The keyframe gate already removes
    the redundancy a digest cache could catch, and catches the near-identical
    frames a digest cannot. What is left is up to `keyframes_max` unique
    images per video, which would evict the entire 64-entry chat cache on
    every ingest — an interactive path paying for a background one.

    **Returns the timing, rather than recording it here.** This function does
    not know the wall clock its caller measured, and `queue_s` is defined as
    the difference between the two — so attributing the call is the caller's
    job and this only supplies the half Ollama knows.
    """
    picked = _pick_vision_model(cfg, registry, health)
    if picked is None:
        raise VisionUnavailableError("no healthy vl model is available")
    model, location = picked

    conf = vision_cfg(cfg)
    options, think = _sampling(conf)
    description, timing = await _transcribe_one(
        ollama, gate,
        url=data_url, model=model, location=location,
        user_question=user_question,
        timeout_s=float(conf.get("timeout_s", _DEFAULT_TIMEOUT_S)),
        user_id=user_id, options=options, think=think, system=system,
        hint_template=hint_template,
    )
    return description, model, timing


async def describe_images(
    messages: list[dict[str, Any]],
    *,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    cfg: Any,
    target_model: str = "",
    user_question: str = "",
    user_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Replace every `image_url` part with a text transcription of it.

    Returns `(messages, n_described)`. The input list is never mutated —
    only messages that actually carry an image are rebuilt, so the common
    text-only case returns the original list untouched and costs one
    scan.

    Fail-soft throughout: a vision model that is down, a non-inline URL,
    or a call that times out yields a bracketed note in place of the
    image rather than an exception. A half-answered image turn beats a
    502, and the note tells the answering model to own the gap instead of
    inventing contents.
    """
    if not messages or not has_any_image_part(messages):
        return messages, 0

    conf = vision_cfg(cfg)
    timeout_s = float(conf.get("timeout_s", _DEFAULT_TIMEOUT_S))
    max_images = int(conf.get("max_images_per_turn", _DEFAULT_MAX_IMAGES))
    cache_size = int(conf.get("cache_size", _DEFAULT_CACHE_SIZE))
    # Same settings as the keyframe path. The transcription prompt forbids
    # answering on both, so there is nothing here for thinking to contribute —
    # and unlike an ingest, a chat turn has someone waiting on the latency.
    options, think = _sampling(conf)

    picked = _pick_vision_model(cfg, registry, health)
    if picked is None:
        log.warning(
            "vision: no healthy vl model — images left undescribed for target=%s",
            target_model or "?",
        )
    vl_model, vl_location = picked if picked else ("", "")

    # Number images across the whole history so the labels the answering
    # model sees ("image 2 of 3") match what the user actually attached.
    total = sum(
        1
        for m in messages
        for p in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(p, dict) and p.get("type") == "image_url"
    )

    out: list[dict[str, Any]] = []
    seen = 0
    described = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        parts: list[str] = []
        touched = False
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
                continue
            if part.get("type") != "image_url":
                continue

            touched = True
            seen += 1
            url = _image_url(part)
            if not url.startswith("data:"):
                # `_to_ollama_messages` drops non-inline URLs, so a vision
                # call here would describe an image it never received.
                log.warning("vision: image %d is not an inline data URI — skipped", seen)
                parts.append(_failed_block(seen, total, "not an inline attachment"))
                continue
            if not vl_model:
                parts.append(_failed_block(seen, total, "no vision model available"))
                continue
            if seen > max_images:
                parts.append(
                    _failed_block(seen, total, f"over the {max_images}-image limit")
                )
                continue

            key = (vl_model, _digest(url))
            cached = _cache_get(key)
            if cached is not None:
                parts.append(_described_block(seen, total, vl_model, cached))
                described += 1
                continue
            try:
                # The chat path does not attribute its own cost: it is one
                # call inside a turn a user is waiting on, not a stage of a
                # background pipeline anyone is going to tune.
                text, _timing = await _transcribe_one(
                    ollama, gate,
                    url=url, model=vl_model, location=vl_location,
                    user_question=user_question, timeout_s=timeout_s, user_id=user_id,
                    options=options, think=think,
                )
            except OllamaError as e:
                health.record_failure(vl_model, str(e))
                log.warning("vision: transcription failed on %s: %s", vl_model, e)
                parts.append(_failed_block(seen, total, "the vision model errored"))
                continue
            health.record_success(vl_model)
            if not text:
                parts.append(_failed_block(seen, total, "the vision model returned nothing"))
                continue
            _cache_put(key, text, max_size=cache_size)
            parts.append(_described_block(seen, total, vl_model, text))
            described += 1

        out.append({**m, "content": "\n\n".join(p for p in parts if p)} if touched else m)

    if described or seen:
        log.info(
            "vision: described %d/%d image(s) with %s for target=%s user=%s",
            described, seen, vl_model or "none", target_model or "?", user_id or "?",
        )
    return out, described


async def describe_for_text_model(
    messages: list[dict[str, Any]],
    *,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    cfg: Any,
    target_model: str,
    user_question: str = "",
    user_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """`describe_images`, but a no-op when the target can already see.

    The guard callers want at a dispatch site: disabled config, a target
    that is itself a vision model, or a turn with no images all return
    the messages unchanged.
    """
    if not describe_enabled(cfg):
        return messages, 0
    if target_model and is_vision_capable(target_model, cfg):
        return messages, 0
    return await describe_images(
        messages,
        ollama=ollama, registry=registry, health=health, gate=gate, cfg=cfg,
        target_model=target_model, user_question=user_question, user_id=user_id,
    )


__all__ = [
    "DESCRIBE_SYSTEM",
    "KEYFRAME_SYSTEM",
    "VisionTiming",
    "describe_enabled",
    "describe_for_text_model",
    "describe_images",
    "has_any_image_part",
    "is_vision_capable",
    "vision_capable_models",
    "vision_cfg",
]
