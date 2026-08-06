"""The two things every media sidecar needs (Phase 41).

`media-worker` (phase 34) and `media-fetcher` (phase 41) are the same shape: a
container with no listening socket that polls audrey-ai for a lease, does one
long CPU-or-network-bound job, and posts the outcome back. They differ entirely
in what happens in the middle.

What they share is here rather than duplicated, and it is deliberately only two
things:

  - **`post`** — JSON over HTTP with a service token, which never raises on a
    status code. A sidecar that treats 409 as an exception cannot tell "my lease
    was swept" from "the network broke", and those need opposite responses.
  - **`Stopping`** — a SIGTERM flag, so `compose stop` drains the job in hand
    instead of abandoning it to a lease expiry.

Nothing here imports from `audrey` beyond this package, which is what lets both
images be `python:slim` plus one binary rather than the full app image.

`worker.py` re-exports both names, so `audrey.media.worker.post` still resolves
— several test modules patch it there, and moving a symbol out from under a
patch target is how a refactor breaks tests that had nothing to do with it.
"""

from __future__ import annotations

import json
import logging
import signal
import time
import urllib.error
import urllib.request
from types import FrameType

log = logging.getLogger("media-sidecar")

DEFAULT_ENDPOINT = "http://audrey-ai:8000"
DEFAULT_POLL_SECONDS = 10
HTTP_TIMEOUT_S = 60


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
        log.info("signal %d received, finishing current job then stopping", signum)
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
