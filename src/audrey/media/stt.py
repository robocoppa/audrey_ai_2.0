"""Whisper speech-to-text for the media worker (Phase 35).

faster-whisper rather than the reference implementation: it runs on CTranslate2
instead of torch, which is the difference between a ~1 GB image and a ~5 GB
one for a container that has no GPU to justify the weight.

The `faster_whisper` import is **deliberately lazy**. This module is imported
by the worker, and the worker's tests run on a laptop that has no whisper
installed — a module-level import would make every one of them fail on
something unrelated to what they test. It also keeps `audrey-ai` able to import
`audrey.media.*` without the package.

Model weights are baked into the image at build time and loaded from disk. The
worker's network is `internal: true` (Phase 34), so a runtime download would
not merely be slow, it would hang until the lease expired and then look like a
stuck job.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL = "small"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_DOWNLOAD_ROOT = "/opt/whisper"

# Whisper's most notorious failure is inventing speech in silence — a musical
# intro reliably produces "Thanks for watching!" and similar. The bundled VAD
# gates audio before it reaches the model, which is the cheapest fix and the
# reason `vad_filter` is on by default rather than offered as a knob.
DEFAULT_VAD = True


class WhisperUnavailableError(RuntimeError):
    """faster-whisper or its weights are missing — an image defect.

    Deliberately distinct from a transcription failure, and handled the same
    way `FFmpegMissingError` is: the job must NOT be failed, because every
    queued video would burn its attempts while the image is being fixed.
    """


class TranscriptionFailedError(RuntimeError):
    """Whisper ran and could not finish this file."""


@dataclass(frozen=True)
class Segment:
    """One timestamped span of speech."""

    t_start: float
    t_end: float
    text: str

    def as_payload(self) -> dict:
        return {"t_start": self.t_start, "t_end": self.t_end, "text": self.text}


_MODEL_CACHE: dict[tuple[str, str], object] = {}


def load_model(
    model_size: str = DEFAULT_MODEL,
    *,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    download_root: str = DEFAULT_DOWNLOAD_ROOT,
) -> object:
    """Load and cache a whisper model.

    Cached per (size, compute_type) because loading costs seconds and the
    worker is long-lived — paying it once per container rather than once per
    video matters most on a queue of short files, where the load would
    otherwise dominate the work.
    """
    key = (model_size, compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise WhisperUnavailableError(
            f"faster-whisper is not installed in this image: {e}",
        ) from e

    try:
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=True,  # the worker has no network; fail loudly here
        )
    except Exception as e:
        # `local_files_only` turns a missing bake into an immediate, explicit
        # error instead of a silent download attempt that hangs until the
        # lease expires.
        raise WhisperUnavailableError(
            f"whisper weights for {model_size!r} are not baked into this image "
            f"at {download_root}: {e}",
        ) from e

    log.info("stt: loaded whisper %s (%s) from %s", model_size, compute_type, download_root)
    _MODEL_CACHE[key] = model
    return model


def transcribe(
    wav: Path,
    *,
    model: object | None = None,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    budget_s: float | None = None,
    vad_filter: bool = DEFAULT_VAD,
) -> list[Segment]:
    """Transcribe a 16 kHz mono WAV into timestamped segments.

    `budget_s` bounds the whole run. faster-whisper yields segments lazily, so
    the clock is checked as they arrive and the run is abandoned mid-file if it
    overruns. That matters because the job holds a lease: a transcription that
    outlives `lease_minutes` gets swept, re-claimed, and runs again from the
    start — burning attempts on a file that was never going to finish in time.
    Failing at the budget turns that into one honest `failed` row.

    Raises rather than returning a partial transcript. Half a video ingested
    and reported `ready` is wrong in the way that never gets noticed: the row
    looks healthy and the missing half is only discovered by someone searching
    for something that was said in it.
    """
    engine = model if model is not None else load_model(model_size)
    started = time.monotonic()

    try:
        raw_segments, info = engine.transcribe(  # type: ignore[attr-defined]
            str(wav),
            language=language,
            vad_filter=vad_filter,
            # Whisper conditions each window on its own previous output, which
            # is how a single mis-decode becomes a repetition loop that eats
            # the rest of the file. Turning it off costs a little cross-window
            # coherence and removes that failure entirely.
            condition_on_previous_text=False,
        )
    except Exception as e:
        raise TranscriptionFailedError(f"whisper could not open {wav.name}: {e}") from e

    out: list[Segment] = []
    try:
        for seg in raw_segments:
            if budget_s is not None and time.monotonic() - started > budget_s:
                raise TranscriptionFailedError(
                    f"transcription exceeded its {budget_s:.0f}s budget at "
                    f"{seg.start:.0f}s of audio — raise kb.video.lease_minutes "
                    f"or use a smaller model",
                )
            text = (seg.text or "").strip()
            if not text:
                continue  # VAD gaps and pure-silence windows
            out.append(Segment(t_start=float(seg.start), t_end=float(seg.end), text=text))
    except TranscriptionFailedError:
        raise
    except Exception as e:
        raise TranscriptionFailedError(f"whisper failed mid-file: {e}") from e

    elapsed = time.monotonic() - started
    log.info(
        "stt: %s -> %d segments in %.1fs (language=%s, probability=%.2f)",
        wav.name, len(out), elapsed,
        getattr(info, "language", "?"), getattr(info, "language_probability", 0.0),
    )
    return collapse_repeats(out)


def collapse_repeats(segments: list[Segment], *, run_length: int = 3) -> list[Segment]:
    """Drop runs of the same line repeated back to back.

    Belt and braces alongside `condition_on_previous_text=False`. A stuck
    decode emits the same sentence dozens of times; left in, those near
    -identical chunks crowd out real content in every retrieval that touches
    the video. Two in a row can be genuine speech, so only runs longer than
    `run_length` are trimmed — and one instance is always kept.
    """
    if not segments:
        return segments

    out: list[Segment] = []
    run_start = 0
    for i in range(1, len(segments) + 1):
        same = i < len(segments) and segments[i].text == segments[run_start].text
        if same:
            continue
        run = segments[run_start:i]
        if len(run) > run_length:
            log.warning(
                "stt: collapsed %d repeats of %r at %.0fs",
                len(run), run[0].text[:60], run[0].t_start,
            )
            out.append(run[0])
        else:
            out.extend(run)
        run_start = i
    return out


__all__ = [
    "DEFAULT_COMPUTE_TYPE",
    "DEFAULT_DOWNLOAD_ROOT",
    "DEFAULT_MODEL",
    "Segment",
    "TranscriptionFailedError",
    "WhisperUnavailableError",
    "collapse_repeats",
    "load_model",
    "transcribe",
]
