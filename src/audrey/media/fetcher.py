"""The media fetcher's claim loop (Phase 41, step 2).

Runs in its own container. Polls `POST /v1/files/fetch/claim`, downloads the
video into a staging directory, and posts the result — at which point Audrey
verifies it, moves it into the user's directory, and the row becomes `pending`
for phases 33-38, which never learn a URL was involved.

Three invariants this file exists to hold:

**It never writes sqlite.** Same as the worker: `UploadsDB` is one connection
with no WAL behind an in-process lock, and a second container writing that file
breaks the single-writer contract quietly. Everything goes back through HTTP.

**It writes to the staging directory and nowhere else.** Not to the user's
directory — Audrey does that move, after checking the bytes. This end has one
writable path and no route to a stored file.

*Do not re-add the rename here.* It was here once, left over from an earlier
draft where the fetcher owned the move, and it is how `KeyError: 'dest_dir'`
shipped: the claim stopped carrying that field and this file went on reading
it. The claim's fields are the contract, and `FetchClaim` is where they live.

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

import contextlib
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

#: How often a download may report itself, in seconds.
#:
#: yt-dlp emits progress several times a second. The page polls every five, so
#: anything under a couple of seconds is a POST nobody will ever see — and each
#: one is a sqlite write on the connection every other request shares.
PROGRESS_INTERVAL_S = 2.0


class _ProgressReporter:
    """Forwards download progress to Audrey, throttled, and keeps it monotonic.

    **The monotonic part is not decoration.** Above a site's highest pre-muxed
    quality, video and audio arrive as two separate downloads, and yt-dlp
    reports each from zero — so a naive forward shows 100%, then 3%, then 100%
    again, which reads as a download that restarted. Folding a finished stream
    into a running total makes the numerator only ever climb.

    The denominator can still take one step *up*, when the second stream starts
    and its size becomes known. That is honest — the job did just get bigger —
    and it is visible as the "of 288.3 MB" figure changing rather than as a
    percentage mysteriously falling.
    """

    def __init__(
        self, endpoint: str, token: str, job: dict, title: str,
        *, interval_s: float = PROGRESS_INTERVAL_S,
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._file_id = job["file_id"]
        self._lease_id = job["lease_id"]
        self._title = title
        self._interval = interval_s
        self._last_post = 0.0
        self._last_done = 0
        self._last_total = 0
        self._base_done = 0
        self._base_total = 0
        #: False once any stream has declined to say how big it is. The total
        #: is then unknowable for the job as a whole, however much the other
        #: streams reported — see `_aggregate_total`.
        self._total_known = True

    def __call__(self, downloaded: int, total: int | None) -> None:
        if downloaded < self._last_done:
            # Counting restarted, so a new stream began. Bank the one that
            # just finished before its numbers are overwritten.
            self._base_done += self._last_done
            if self._last_total:
                self._base_total += self._last_total
            else:
                self._total_known = False
            self._last_done = 0
            self._last_total = 0
        self._last_done = downloaded
        self._last_total = total or 0

        now = time.monotonic()
        if now - self._last_post < self._interval:
            return
        self._last_post = now
        self.send(self._base_done + downloaded, self._aggregate_total(total))

    def _aggregate_total(self, total: int | None) -> int:
        """The job's total size, or 0 when any part of it is unknown.

        **One unknown stream makes the whole total unknown**, and summing the
        parts we do know is worse than admitting that. It produces a
        denominator smaller than the numerator will eventually be — a download
        that sits at "100% (130 MB of 120 MB)" and then keeps going, which is
        a more confusing display than the honest byte count with no percentage
        at all.
        """
        if not self._total_known or not total:
            return 0
        return self._base_total + total

    def send(self, downloaded: int, total: int) -> None:
        """Post one update. Failure is logged and swallowed.

        A progress report that cannot be delivered must not fail the download —
        the row simply keeps showing elapsed time, which is what it did before
        any of this existed.
        """
        status, body = post(
            self._endpoint, f"/v1/files/fetch/{self._file_id}/progress", self._token,
            {
                "lease_id": self._lease_id,
                "title": self._title,
                "downloaded_bytes": int(downloaded),
                "total_bytes": int(total),
            },
            timeout=15,
        )
        if status != 200:
            log.info(
                "fetcher: progress for %s not recorded (%s): %s",
                self._file_id, status, body,
            )


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

        # Send the title before a byte moves. It is known from the metadata
        # pass, and until it lands the row shows the video id — so "is it
        # fetching the one I meant?" is unanswerable for the whole download,
        # which is exactly when it is worth asking. The estimate goes with it
        # so the first poll has a denominator; the real totals replace it
        # within a couple of seconds.
        progress = _ProgressReporter(endpoint, token, job, info.title)
        progress.send(0, info.filesize_approx)

        got = download(
            url, stage_dir, file_id,
            timeout_s=budget, max_bytes=max_bytes,
            caption_source=caption_source,
            on_progress=progress,
        )
    except (FetchRefusedError, FetchFailedError) as e:
        _report_failure(endpoint, token, file_id, lease_id, str(e))
        _clear_staging(stage_dir, file_id)
        return
    except YtDlpMissingError as e:
        log.error("fetcher: %s", e)
        raise

    try:
        _finish(got, job, info, endpoint=endpoint, token=token, max_bytes=max_bytes)
    finally:
        _clear_staging(stage_dir, file_id)


def _finish(
    got: Downloaded,
    job: dict,
    info: UrlInfo,
    *,
    endpoint: str,
    token: str,
    max_bytes: int,
) -> None:
    """Check the staged file over and hand it to Audrey. **Never moves it.**

    Moving the finished download into the user's directory is Audrey's job, not
    this container's — see `_fetch_stage_dir` in `routes/files.py`. This end
    only reports; the result route rebuilds the staged path from the reported
    filename, re-derives everything it is told, and does the rename itself
    once every check has passed.

    The two checks here are not redundant with that. They exist so the *reason*
    a download is refused is the true one: an oversized file that reached
    Audrey unchecked trips the quota check instead, and tells the user their
    account is full when the actual problem is the video.
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

    # The naming contract, and the one place the two containers must agree:
    # the extension has to be the one on the file actually written, because
    # Audrey rebuilds `<stage_dir>/<file_id><ext>` from it to find what to
    # verify. The stem is the video's real title, which replaces the
    # video-id placeholder in the file list.
    filename = _display_name(info, got.path, file_id)

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
        try:
            handle_job(
                job, endpoint=endpoint, token=token, probe_timeout_s=probe_timeout_s,
            )
        except YtDlpMissingError:
            # The image is built wrong. Crash-looping is the right signal for
            # that — it shows in `docker ps` and it is fixed by rebuilding, not
            # by failing every queued URL on the way past.
            raise
        except Exception as e:
            # A bug in this container, not a fact about the video.
            #
            # **Reported rather than raised, which is a change made after one
            # cost an afternoon.** A `KeyError` here used to kill the process:
            # the row stayed 'fetching' with nobody holding it, the lease
            # expired twenty minutes later, the sweep re-queued it, and the
            # next claim crashed identically. From the page that is a download
            # that has been running for an hour. It is indistinguishable from a
            # slow one, and the only evidence is a traceback in a container log
            # nobody has reason to open.
            #
            # Saying so on the row costs a retry that was going to fail anyway
            # — `max_attempts` is 3 — and turns an hour of silence into a
            # sentence.
            log.exception("fetcher: unhandled error on %s", job.get("file_id"))
            with contextlib.suppress(Exception):
                _report_failure(
                    endpoint, token, str(job.get("file_id") or ""),
                    str(job.get("lease_id") or ""),
                    f"the fetcher hit an internal error and could not continue: {e}",
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
