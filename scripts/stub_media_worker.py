#!/usr/bin/env python3
"""A media worker that does no media work (Phase 33).

Claims a job, posts a fixed transcript, exits. It exists so the job lifecycle
can be exercised end to end before ffmpeg or whisper are anywhere near it —
when a real transcode fails in Phase 34/35, the question should be "why did
ffmpeg fail", not "is it ffmpeg or is it the lease logic".

It stays useful after those phases land: it is the fastest way to reproduce a
lease bug without waiting minutes for a real transcode.

    # one job, against the box
    python scripts/stub_media_worker.py --endpoint http://192.168.1.11:8000

    # take a job and never report, to watch the lease expire
    python scripts/stub_media_worker.py --abandon

    # report a failure instead of a transcript
    python scripts/stub_media_worker.py --fail "unreadable container"

    # put an already-processed video back in the queue, to run it again
    python scripts/stub_media_worker.py --requeue <file_id>

The service token comes from KB_SERVICE_TOKEN, the same one `custom-tools`
uses. Nothing here imports from `audrey` — a worker is an HTTP client, and
keeping it that way is what makes the container in Phase 34 a small one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "http://127.0.0.1:8000"


def _post(endpoint: str, path: str, token: str, body: dict | None) -> tuple[int, dict]:
    # The endpoint is a CLI argument, so the scheme is checked rather than
    # assumed — urlopen would otherwise happily accept `file:` and read a local
    # path, and the service token would go somewhere it was never meant to.
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must be http:// or https://, got {endpoint!r}")

    data = json.dumps(body).encode() if body is not None else b"{}"
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        endpoint.rstrip("/") + path,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Audrey-Service-Token": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - scheme checked above
            raw = response.read()
            # 204 is the empty queue, and it is the common case — an idle
            # Audrey must not look like an error to a polling worker.
            if response.status == 204 or not raw:
                return response.status, {}
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # An error body from something in front of Audrey — cloudflared or
            # the CDN edge — is HTML, not JSON. Report it rather than crashing.
            return e.code, {"detail": raw.decode("utf-8", "replace")[:400]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=os.environ.get("AUDREY_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token", default=os.environ.get("KB_SERVICE_TOKEN", ""))
    parser.add_argument("--abandon", action="store_true",
                        help="claim and never report, so the lease expires")
    parser.add_argument("--fail", metavar="REASON",
                        help="report a failure instead of a transcript")
    parser.add_argument("--lease", help="post against this lease id instead of the "
                                        "one just issued, to prove a stale lease is refused")
    parser.add_argument("--requeue", metavar="FILE_ID",
                        help="send a processed video back to the queue and exit, "
                             "instead of claiming")
    parser.add_argument("--force", action="store_true",
                        help="with --requeue, take the job back even if a worker "
                             "is mid-run — its work is discarded")
    args = parser.parse_args(argv)

    if not args.token:
        print("KB_SERVICE_TOKEN is unset — the routes will 401.", file=sys.stderr)
        return 2

    # Requeue is its own errand, not a step in the claim loop — the whole point
    # is to put work back so a *later* run can pick it up.
    if args.requeue:
        path = f"/v1/files/{args.requeue}/requeue"
        if args.force:
            path += "?force=true"
        status, body = _post(args.endpoint, path, args.token, None)
        print(f"requeue: {status} {body}")
        if status == 409:
            print("  (a worker is mid-run; re-run with --force to take it back)",
                  file=sys.stderr)
        return 0 if status == 200 else 1

    status, job = _post(args.endpoint, "/v1/files/jobs/claim", args.token, None)
    if status == 204 or not job:
        print("queue empty")
        return 0
    if status != 200:
        print(f"claim failed ({status}): {job.get('detail')}", file=sys.stderr)
        return 1

    print(f"claimed {job['filename']} "
          f"({job['bytes']} bytes, attempt {job['attempts']}) for {job['user']}")
    print(f"  path:  {job['path']}")
    print(f"  lease: {job['lease_id']}")

    if args.abandon:
        print("abandoning — the lease should expire and the row requeue")
        return 0

    lease = args.lease or job["lease_id"]
    file_id = job["file_id"]

    if args.fail:
        status, body = _post(
            args.endpoint, f"/v1/files/{file_id}/ingest-failed", args.token,
            {"lease_id": lease, "reason": args.fail},
        )
    else:
        status, body = _post(
            args.endpoint, f"/v1/files/{file_id}/ingest-result", args.token,
            {
                "lease_id": lease,
                "duration_s": 0.0,
                "segments": [{
                    "t_start": 0.0, "t_end": 1.0,
                    "text": f"Stub transcript for {job['filename']}. "
                            "Phase 35 replaces this with whisper output.",
                }],
            },
        )

    print(f"reported: {status} {body}")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
