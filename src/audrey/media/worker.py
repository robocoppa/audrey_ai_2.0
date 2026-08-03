"""The media worker's claim loop (Phases 34-35).

Runs in its own container. Polls `POST /v1/files/jobs/claim`, demuxes the
audio, transcribes it, posts the segments back. Phase 36 adds frames.

Two invariants this file exists to hold:

**It never writes sqlite.** `UploadsDB` is one connection with no WAL behind
an in-process lock. A second container writing that file breaks the
single-writer contract quietly. Everything goes back through HTTP.

**It never speaks to a model.** No Ollama address, and compose keeps it off
the network that could reach one. Every fairness guarantee in
`scheduling.py` is bypassed the moment a worker calls Ollama directly, and
it is bypassed silently and only under load — so the invariant is
established now, while there is nothing here that would want to break it.

Configured by environment, not `config.yaml`: a sidecar that parses the app's
config file needs the file mounted and a YAML parser, and gains nothing — none
of the orchestrator's settings apply to it.

    AUDREY_ENDPOINT      default http://audrey-ai:8000
    KB_SERVICE_TOKEN     required
    POLL_SECONDS         default 10
    WORK_DIR             default /tmp/media-worker
    ONCE                 set to 1 to take a single job and exit
    WHISPER_MODEL        default "small" — must be baked into the image
    TRANSCRIBE_BUDGET_S  default 1440 (24 min); 0 disables the cap
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import FrameType

from audrey.media.audio import FFmpegFailedError, FFmpegMissingError, extract_audio, probe
from audrey.media.stt import (
    DEFAULT_MODEL,
    TranscriptionFailedError,
    WhisperUnavailableError,
    transcribe,
)

log = logging.getLogger("media-worker")

DEFAULT_ENDPOINT = "http://audrey-ai:8000"
DEFAULT_POLL_SECONDS = 10
HTTP_TIMEOUT_S = 60

# A transcript for a long video is a large POST. It goes to audrey-ai directly
# over the compose network, not through cloudflared, so the 100 MB edge cap
# that shaped Phase 32 does not apply here — but a slow read on a big body
# still shouldn't look like a hung worker.
RESULT_TIMEOUT_S = 300


class Stopping:
    """Flips on SIGTERM so a compose stop finishes the current job first.

    Killing mid-job is safe — Phase 33's lease sweep returns the row to the
    queue — but it costs a full lease expiry before anything picks it up
    again. Draining is the cheaper path when we are being asked politely.
    """

    def __init__(self) -> None:
        self.requested = False

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        log.info("worker: signal %d received, finishing current job then stopping", signum)
        self.requested = True

    def wait(self, seconds: float, *, slice_s: float = 0.5) -> None:
        """Sleep, but notice a stop request while doing it.

        A plain `time.sleep(poll_seconds)` makes an *idle* shutdown take up to
        a full poll interval, because the flag is only read at the top of the
        loop. That was measured at 7.3s against a 10s poll — and Docker's
        default `stop_grace_period` is 10s, so raising POLL_SECONDS to 30
        would have silently converted every graceful stop into a SIGKILL.
        Slicing the sleep decouples shutdown latency from poll frequency.
        """
        deadline = time.monotonic() + seconds
        while not self.requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(slice_s, remaining))


def post(
    endpoint: str, path: str, token: str, body: dict | None,
    *, timeout: int = HTTP_TIMEOUT_S,
) -> tuple[int, dict]:
    """POST JSON, returning `(status, parsed_body)`. Never raises on HTTP status."""
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must be http:// or https://, got {endpoint!r}")

    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        endpoint.rstrip("/") + path,
        data=json.dumps(body or {}).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Audrey-Service-Token": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - scheme checked above
            raw = response.read()
            if response.status == 204 or not raw:
                return response.status, {}
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Anything in front of Audrey answers in HTML, not JSON.
            return e.code, {"detail": raw.decode("utf-8", "replace")[:400]}


def handle_job(
    job: dict,
    *,
    endpoint: str,
    token: str,
    work_dir: Path,
    model_size: str = DEFAULT_MODEL,
    budget_s: float | None = None,
) -> None:
    """Do one job and report it. Any failure is reported, never swallowed.

    A worker that dies without reporting leaves the row in `processing` until
    the lease expires. That recovers, but it costs a full lease period and
    tells the user nothing in the meantime — so ffmpeg's and whisper's opinion
    of the file gets posted as a failure rather than raised.

    The two exceptions are `FFmpegMissingError` and `WhisperUnavailableError`,
    which say the *image* is broken rather than the file. Those propagate:
    failing the row would burn every queued video's attempts while the image
    is being fixed, and they would all be `failed` by the time it was.
    """
    file_id = job["file_id"]
    source = Path(job["path"])
    lease_id = job["lease_id"]

    if not source.exists():
        # The mount is wrong, or the file was deleted under us. Either way the
        # worker cannot fix it by retrying, so say so plainly.
        _report_failure(
            endpoint, token, file_id, lease_id,
            f"source not readable at {source} — check the media-worker mount",
        )
        return

    # Extract to the container's own work dir. The uploads mount is read-only
    # on purpose: the worker reads sources and posts results over HTTP, and
    # nothing about that requires write access to another service's data.
    wav = work_dir / f"{file_id}.wav"
    try:
        info = probe(source)
        duration = extract_audio(source, wav)

        segments: list[dict] = []
        if duration > 0 and wav.exists():
            segments = [
                s.as_payload()
                for s in transcribe(wav, model_size=model_size, budget_s=budget_s)
            ]
        else:
            # No audio stream at all. Not a failure — the file may still have
            # visual content Phase 36 will want, and the row completes with an
            # empty transcript rather than an error nobody can act on.
            log.info("worker: %s has no audio, nothing to transcribe", file_id)
    except (FFmpegMissingError, WhisperUnavailableError) as e:
        log.error("worker: %s", e)
        raise
    except (FFmpegFailedError, TranscriptionFailedError) as e:
        _report_failure(endpoint, token, file_id, lease_id, str(e))
        return
    finally:
        # The wav is an intermediate and can be a large fraction of the source
        # — 18 MB for nine minutes. Left behind these accumulate inside the
        # container, where nobody would think to look.
        wav.unlink(missing_ok=True)

    spoken = sum(len(s["text"]) for s in segments)
    log.info(
        "worker: %s container=%.1fs audio=%.1fs segments=%d chars=%d",
        job.get("filename", file_id), info.container_duration_s, duration,
        len(segments), spoken,
    )

    status, body = post(
        endpoint, f"/v1/files/{file_id}/ingest-result", token,
        {"lease_id": lease_id, "duration_s": duration, "segments": segments},
        timeout=RESULT_TIMEOUT_S,
    )
    if status != 200:
        log.warning("worker: result rejected for %s (%s): %s", file_id, status, body)


def _report_failure(
    endpoint: str, token: str, file_id: str, lease_id: str, reason: str,
) -> None:
    log.warning("worker: job %s failed: %s", file_id, reason)
    status, body = post(
        endpoint, f"/v1/files/{file_id}/ingest-failed", token,
        {"lease_id": lease_id, "reason": reason},
    )
    if status != 200:
        log.warning("worker: failure report rejected (%s): %s", status, body)


def run(
    *, endpoint: str, token: str, poll_seconds: int, work_dir: Path,
    once: bool = False, model_size: str = DEFAULT_MODEL,
    budget_s: float | None = None,
) -> int:
    stopping = Stopping()
    stopping.install()
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "worker: polling %s every %ds (whisper=%s, budget=%s)",
        endpoint, poll_seconds, model_size,
        f"{budget_s:.0f}s" if budget_s else "none",
    )

    idle_logged = False
    while not stopping.requested:
        try:
            status, job = post(endpoint, "/v1/files/jobs/claim", token, None)
        except OSError as e:
            # Audrey restarting is normal and must not kill the worker.
            log.warning("worker: claim failed (%s), retrying in %ds", e, poll_seconds)
            stopping.wait(poll_seconds)
            continue

        if status == 204 or not job:
            # An idle queue is the steady state. Log the transition into idle
            # once, not every poll — otherwise the log is useless at 10s.
            if not idle_logged:
                log.info("worker: queue empty, waiting")
                idle_logged = True
            if once:
                return 0
            stopping.wait(poll_seconds)
            continue

        if status != 200:
            log.error("worker: claim rejected (%s): %s", status, job)
            if once:
                return 1
            stopping.wait(poll_seconds)
            continue

        idle_logged = False
        log.info("worker: claimed %s (attempt %s)", job.get("filename"), job.get("attempts"))
        handle_job(
            job, endpoint=endpoint, token=token, work_dir=work_dir,
            model_size=model_size, budget_s=budget_s,
        )
        if once:
            return 0

    log.info("worker: stopped")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    token = os.environ.get("KB_SERVICE_TOKEN", "")
    if not token:
        log.error("KB_SERVICE_TOKEN is unset — every claim would 401. Refusing to start.")
        return 2

    # The budget must sit inside the lease, or a slow transcription is swept
    # and re-claimed while still running — the same file transcribed twice,
    # concurrently, until its attempts run out. Default 80% of a 30-minute
    # lease; set both together if you change either.
    budget = float(os.environ.get("TRANSCRIBE_BUDGET_S", 24 * 60))

    return run(
        endpoint=os.environ.get("AUDREY_ENDPOINT", DEFAULT_ENDPOINT),
        token=token,
        poll_seconds=int(os.environ.get("POLL_SECONDS", DEFAULT_POLL_SECONDS)),
        work_dir=Path(os.environ.get("WORK_DIR", Path(tempfile.gettempdir()) / "media-worker")),
        once=os.environ.get("ONCE", "") == "1",
        model_size=os.environ.get("WHISPER_MODEL", DEFAULT_MODEL),
        budget_s=budget if budget > 0 else None,
    )


if __name__ == "__main__":
    sys.exit(main())
