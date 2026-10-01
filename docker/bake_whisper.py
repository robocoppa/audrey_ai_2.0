#!/usr/bin/env python3
"""Download whisper weights into the media-worker image at build time.

Runs once, during `docker build`. The worker's network is `internal: true`
(Phase 34), so weights it does not already have are weights it can never get —
a runtime download would hang until the lease expired and present as a stuck
job rather than a slow one.

This is a script rather than an inline `RUN python -c "..."` because the inline
form needed backslash continuations inside a double-quoted shell string, which
is three layers of quoting to get wrong and produced a build failure whose
error Docker truncated. A file has none of that, and can say what went wrong.
"""

from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path

MODEL = os.environ.get("WHISPER_BAKE", "small")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
ROOT = os.environ.get("WHISPER_DOWNLOAD_ROOT", "/opt/whisper")


def verify_audio_decoder() -> None:
    """Exercise the exact file decoder the worker uses before baking weights."""
    from faster_whisper.audio import decode_audio

    sample_count = 1600
    with tempfile.TemporaryDirectory(prefix="whisper-decoder-") as temp_dir:
        wav = Path(temp_dir) / "probe.wav"
        with wave.open(str(wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * sample_count)
        audio = decode_audio(str(wav))

    if getattr(audio, "ndim", None) != 1 or getattr(audio, "size", 0) != sample_count:
        raise RuntimeError(
            f"decoded probe has unexpected shape {getattr(audio, 'shape', None)!r}",
        )


def main() -> int:
    # flush: stdout is block-buffered when piped, stderr is not, so without
    # this the failure message below prints *before* the line saying what was
    # being attempted — in a build log that reads as an unrelated error.
    print(
        f"baking whisper model={MODEL!r} compute_type={COMPUTE_TYPE!r} -> {ROOT}",
        flush=True,
    )
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        # Name the module that is actually missing. This bit twice already:
        # faster-whisper imports `requests` without declaring it, so the
        # failure is an ImportError for a package nobody mentioned, raised
        # from inside a package that installed fine.
        print(
            f"FATAL: cannot import faster_whisper: {e}\n"
            f"  missing module: {getattr(e, 'name', 'unknown')!r}\n"
            "  if that is not 'faster_whisper' itself, it is an undeclared\n"
            "  transitive dependency — add it to the pip line explicitly.",
            file=sys.stderr,
        )
        return 1

    try:
        verify_audio_decoder()
    except Exception as e:  # noqa: BLE001 - image build gate reports exact dependency break
        print(
            f"FATAL: faster-whisper audio decoder is incompatible: "
            f"{type(e).__name__}: {e}\n"
            "  check the faster-whisper and PyAV pins before rebuilding.",
            file=sys.stderr,
        )
        return 1

    try:
        WhisperModel(
            MODEL, device="cpu", compute_type=COMPUTE_TYPE, download_root=ROOT,
        )
    except Exception as e:  # noqa: BLE001 - a build step: report anything, exit 1
        # The build host needs to reach huggingface.co for this one step. If it
        # cannot, say so plainly — the failure otherwise reads as a broken
        # Dockerfile rather than a network or proxy problem on the builder.
        print(
            f"FATAL: could not download whisper {MODEL!r}: {type(e).__name__}: {e}\n"
            "  - the BUILD host needs outbound access to huggingface.co\n"
            "  - valid sizes: tiny, base, small, medium, large-v3\n"
            "  - check free disk: 'small' needs ~500 MB, 'medium' ~1.5 GB",
            file=sys.stderr,
        )
        return 1

    print(f"baked {MODEL} into {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
