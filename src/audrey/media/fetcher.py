"""The media fetcher's claim loop (Phase 41, step 2).

Runs in its own container. Polls `POST /v1/files/fetch/claim`, downloads the
video, renames it into the user's upload directory, and posts the result — at
which point the row becomes `pending` and phases 33-38 take over without ever
learning a URL was involved.

Three invariants this file exists to hold:

**It never writes sqlite.** Same as the worker: `UploadsDB` is one connection
with no WAL behind an in-process lock, and a second container writing that file
breaks the single-writer contract quietly. Everything goes back through HTTP.

**It never writes to the final path.** Downloads land in a staging directory
and are renamed into place only when complete. A partial file at
`<user>/<file_id>.mp4` is a file the media worker will claim and transcribe,
and it would look like a successful ingest of a broken video rather than like
an interrupted download.

**It never speaks to a model.** It is the one sidecar with internet egress,
which is exactly why it is not on `media-net` or `ollama-net`. A downloader
that could reach the model server would make phase 34's isolation argument
false by topology rather than by intent.

Configured by environment, not `config.yaml` — the caps that matter arrive on
the claim, so `kb.fetch.*` stays the single source of truth:

    AUDREY_ENDPOINT   default http://audrey-ai:8000
    KB_SERVICE_TOKEN  required
    POLL_SECONDS      default 10
    ONCE              set to 1 to take a single job and exit
    PROBE_TIMEOUT_S   default 120 — the metadata pass, not the download
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

from audrey.media.fetch import (
    Downloaded,
    FetchFailedError,
    FetchRefusedError,
    UrlInfo,
    YtDlpMissingError,
    check_limits,
    download,
    probe_url,
)
from audrey.media.service import (
    DEFAULT_ENDPOINT,
    DEFAULT_POLL_SECONDS,
    Stopping,
    post,
)

log = logging.getLogger("media-fetcher")

DEFAULT_PROBE_TIMEOUT_S = 120

#: Seconds of lease held back from the download so the result post has time to
#: land. Same reasoning as the worker's `LEASE_RESERVE_S`, smaller number: this
#: post carries a caption track at most, not a transcript and 24 descriptions.
LEASE_RESERVE_S = 60.0

#: A downloaded title has to survive being a filename, a display string, and
#: the thing `_source_path` takes an extension from. Anything that is not
#: plainly safe in all three becomes an underscore.
_UNSAFE_NAME = re.compile(r"[^\w.\- ]+", re.UNICODE)
_MAX_TITLE_CHARS = 120


def _display_name(info: UrlInfo, path: Path, file_id: str) -> str:
    """A filename for the row: the video's real title, with the real extension.

    The extension is not decoration. `fetch/{id}/result` rebuilds the on-disk
    path from `file_id` plus this suffix, and fails the row when nothing is
    there — so it must be the suffix of the file actually written, never the
    one the metadata pass predicted. They differ whenever the format selector
    falls through to a different container.
    """
    stem = _UNSAFE_NAME.sub("_", info.title or "").strip(" ._")
    stem = re.sub(r"\s+", " ", stem)[:_MAX_TITLE_CHARS].strip()
    return f"{stem or file_id}{path.suffix}"


def _clear_staging(stage_dir: Path, file_id: str) -> None:
    """Remove this job's leftovers: caption files, fragments, a dead `.part`.

    Best effort, and never fatal. Audrey sweeps the staging directory against
    the database on every claim, so anything missed here is collected there —
    this only keeps the common case from needing that.
    """
    try:
        for child in stage_dir.glob(f"{file_id}*"):
            try:
                child.unlink()
            except OSError:
                continue
    except OSError as e:
        log.warning("fetcher: could not tidy staging for %s: %s", file_id, e)


def handle_job(job: dict, *, endpoint: str, token: str, probe_timeout_s: float) -> None:
    """Do one download and report it. Any failure is reported, never swallowed.

    A fetcher that dies without reporting leaves the row in `fetching` until
    the lease expires. That recovers — Audrey sweeps it back to `fetch_pending`
    — but it costs a full lease period and tells the user nothing meanwhile, so
    yt-dlp's opinion of the URL gets posted as a failure rather than raised.

    `YtDlpMissingError` is the exception, and propagates: it says the *image*
    is broken rather than the link, and failing rows for that would burn every
    queued URL's attempts while the image was being fixed.
    """
    file_id = job["file_id"]
    url = job["source_url"]
    lease_id = job["lease_id"]
    stage_dir = Path(job["stage_dir"])
    dest_dir = Path(job["dest_dir"])
    started = time.monotonic()

    lease_s = float(job.get("lease_seconds") or 0)
    max_bytes = int(job.get("max_bytes") or 0)
    max_duration_s = float(job.get("max_duration_s") or 0)

    try:
        info = probe_url(url, timeout_s=probe_timeout_s)
        check_limits(info, max_duration_s=max_duration_s, max_bytes=max_bytes)

        caption_source = info.caption_choice()
        budget = _remaining(lease_s, started)
        log.info(
            "fetcher: %s %r %.0fs captions=%s budget=%.0fs",
            file_id, info.title or url, info.duration_s, caption_source or "none", budget,
        )
        if budget <= 0:
            raise FetchFailedError(
                "the lease ran out during the metadata pass — the server is "
                "slower than this download was given time for",
            )

        got = download(
            url, stage_dir, file_id,
            timeout_s=budget, max_bytes=max_bytes,
            caption_source=caption_source,
        )
    except (FetchRefusedError, FetchFailedError) as e:
        _report_failure(endpoint, token, file_id, lease_id, str(e))
        _clear_staging(stage_dir, file_id)
        return
    except YtDlpMissingError as e:
        log.error("fetcher: %s", e)
        raise

    try:
        _finish(
            got, job, info,
            endpoint=endpoint, token=token, dest_dir=dest_dir, max_bytes=max_bytes,
        )
    finally:
        _clear_staging(stage_dir, file_id)


def _finish(
    got: Downloaded,
    job: dict,
    info: UrlInfo,
    *,
    endpoint: str,
    token: str,
    dest_dir: Path,
    max_bytes: int,
) -> None:
    """Move the finished file into place and post the result.

    The rename is the moment the download becomes real, and it is deliberately
    the *last* thing before the post: everything up to here can be abandoned by
    a crash with no consequence beyond a file in staging that the next sweep
    collects.
    """
    file_id = job["file_id"]
    lease_id = job["lease_id"]

    size = got.path.stat().st_size
    if size == 0:
        _report_failure(
            endpoint, token, file_id, lease_id,
            "the download produced an empty file",
        )
        return
    if max_bytes and size > max_bytes:
        # `--max-filesize` should have caught this. It does not apply to every
        # downloader path, and the check is cheap next to what it prevents:
        # handing Audrey a file that trips the quota check on the far side,
        # which fails the row with a message about quota rather than size.
        _report_failure(
            endpoint, token, file_id, lease_id,
            f"the download came to {size // (1024 * 1024)}MB, over this "
            f"server's {max_bytes // (1024 * 1024)}MB limit",
        )
        return

    filename = _display_name(info, got.path, file_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{file_id}{got.path.suffix}"
    # `Path.replace` rather than `shutil.move`: staging and the user
    # directories are both under the uploads root, so this is an atomic rename
    # within one filesystem. A cross-device move would be a copy, which for two
    # gigabytes is a second write and a window where a partial file exists at
    # the final path — the exact thing staging is for.
    got.path.replace(dest)

    status, body = post(
        endpoint, f"/v1/files/fetch/{file_id}/result", token,
        {
            "lease_id": lease_id,
            "filename": filename,
            "duration_s": info.duration_s,
            "segments": got.segments,
            "transcript_source": got.transcript_source,
        },
        timeout=300,
    )
    if status != 200:
        log.warning("fetcher: result rejected for %s (%s): %s", file_id, status, body)
        return
    log.info(
        "fetcher: %s -> %r %.1fMB%s",
        file_id, filename, size / (1024 * 1024),
        f" +{len(got.segments)} {got.transcript_source} segments" if got.segments else "",
    )


def _remaining(lease_s: float, started: float) -> float:
    """How long the download may run, given what the lease has left.

    Returns the configured default when the claim carried no lease length —
    an older Audrey, or a test — rather than zero, which would refuse every
    job on a field that is allowed to be absent.
    """
    if lease_s <= 0:
        return 3600.0
    return lease_s - (time.monotonic() - started) - LEASE_RESERVE_S


def _report_failure(
    endpoint: str, token: str, file_id: str, lease_id: str, reason: str,
) -> None:
    log.warning("fetcher: %s failed: %s", file_id, reason)
    status, body = post(
        endpoint, f"/v1/files/fetch/{file_id}/failed", token,
        {"lease_id": lease_id, "reason": reason},
    )
    if status != 200:
        log.warning("fetcher: failure report rejected (%s): %s", status, body)


def run(
    *, endpoint: str, token: str, poll_seconds: int, once: bool = False,
    probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
) -> int:
    stopping = Stopping()
    stopping.install()
    log.info("fetcher: polling %s every %ds", endpoint, poll_seconds)

    idle_logged = False
    while not stopping.requested:
        try:
            status, job = post(endpoint, "/v1/files/fetch/claim", token, None)
        except OSError as e:
            # Audrey restarting is normal and must not kill the fetcher.
            log.warning("fetcher: claim failed (%s), retrying in %ds", e, poll_seconds)
            stopping.wait(poll_seconds)
            continue

        if status == 204 or not job:
            if not idle_logged:
                log.info("fetcher: queue empty, waiting")
                idle_logged = True
            if once:
                return 0
            stopping.wait(poll_seconds)
            continue

        if status != 200:
            log.error("fetcher: claim rejected (%s): %s", status, job)
            if once:
                return 1
            stopping.wait(poll_seconds)
            continue

        idle_logged = False
        log.info(
            "fetcher: claimed %s (attempt %s) %r",
            job.get("file_id"), job.get("attempts"), job.get("source_url"),
        )
        handle_job(
            job, endpoint=endpoint, token=token, probe_timeout_s=probe_timeout_s,
        )
        if once:
            return 0

    log.info("fetcher: stopped")
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

    return run(
        endpoint=os.environ.get("AUDREY_ENDPOINT", DEFAULT_ENDPOINT),
        token=token,
        poll_seconds=int(os.environ.get("POLL_SECONDS", DEFAULT_POLL_SECONDS)),
        once=os.environ.get("ONCE", "") == "1",
        probe_timeout_s=float(os.environ.get("PROBE_TIMEOUT_S", DEFAULT_PROBE_TIMEOUT_S)),
    )


if __name__ == "__main__":
    sys.exit(main())
