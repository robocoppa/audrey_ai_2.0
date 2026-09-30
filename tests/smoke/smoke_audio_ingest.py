#!/usr/bin/env python3
"""Prove deployed MP3 ingestion from upload through transcript reading.

The laptop generates a short spoken MP3 with ffmpeg's local flite source. The
smoke uploads it, waits for Audrey's existing media worker to transcribe and
index it, reads the transcript through the native artifact route, verifies that
audio has no visual artifact, then deletes the temporary upload.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
AUDIO_TIMEOUT_SECONDS = float(os.getenv("AUDREY_AUDIO_SMOKE_TIMEOUT_SECONDS", "360"))
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin
_REQUIRED_WORDS = frozenset({"blue", "lantern", "audio", "ready"})
_SPOKEN_TEXT = (
    "The blue lantern is ready. "
    "This audio recording confirms the blue lantern is ready."
)


class SmokeError(RuntimeError):
    """A deployed MP3 ingestion contract was not satisfied."""


def _auth_headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        token,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


def _request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str = "",
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 60,
) -> tuple[int, bytes, Message]:
    headers = {"Accept": "application/json", **_auth_headers(token)}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = response.status
            content = response.read()
            response_headers = response.headers
    except HTTPError as exc:
        status = exc.code
        content = exc.read()
        response_headers = exc.headers
    if status not in expected:
        excerpt = content.decode(errors="replace")[:500]
        raise SmokeError(f"{method} {path}: HTTP {status}: {excerpt}")
    return status, content, response_headers


def _json_request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    expected: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, Any]]:
    status, content, _headers = _request(
        path,
        token=token,
        method=method,
        expected=expected,
    )
    try:
        value = json.loads(content) if content else {}
    except json.JSONDecodeError as exc:
        raise SmokeError(f"{method} {path}: response was not JSON") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"{method} {path}: response was not an object")
    return status, value


def _spoken_mp3() -> bytes:
    """Generate one real audio-only MP3 without network access."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SmokeError("ffmpeg is required on the laptop to generate the MP3 fixture")
    with tempfile.TemporaryDirectory(prefix="audrey-audio-smoke-") as raw_dir:
        path = Path(raw_dir) / "spoken.mp3"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"flite=text='{_SPOKEN_TEXT}':voice=slt",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            "-y",
            str(path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0 or not path.is_file():
            reason = completed.stderr.strip()[-500:] or "no output file"
            raise SmokeError(f"ffmpeg could not generate the MP3 fixture: {reason}")
        content = path.read_bytes()
    if len(content) < 1000:
        raise SmokeError("ffmpeg generated an unexpectedly small MP3 fixture")
    return content


def _upload_mp3(*, filename: str, content: bytes) -> tuple[int, dict[str, Any]]:
    boundary = f"audrey-{uuid.uuid4().hex}"
    delimiter = boundary.encode()
    body = b"\r\n".join(
        (
            b"--" + delimiter,
            f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
            b"Content-Type: audio/mpeg",
            b"",
            content,
            b"--" + delimiter + b"--",
            b"",
        )
    )
    status, response, _headers = _request(
        "/api/files",
        token=USER_TOKEN,
        method="POST",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
        timeout=120,
    )
    return status, json.loads(response)


def _wait_until_ready(file_id: str) -> tuple[dict[str, Any], int, float]:
    deadline = time.monotonic() + AUDIO_TIMEOUT_SECONDS
    polls = 0
    started = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        polls += 1
        _status, last = _json_request(f"/api/files/{file_id}", token=USER_TOKEN)
        status = str(last.get("status") or "")
        if status == "ready":
            return last, polls, time.monotonic() - started
        if status == "failed":
            reason = str(last.get("failure_reason") or "unspecified worker failure")
            raise SmokeError(f"audio job failed: {reason}")
        if status not in {"pending", "processing"}:
            raise SmokeError(f"audio job entered unexpected status {status!r}: {last}")
        time.sleep(1)
    raise SmokeError(
        f"audio job did not become ready within {AUDIO_TIMEOUT_SECONDS:g}s: {last}",
    )


def _repair_until_ready() -> str:
    _request(
        "/v1/admin/repair",
        token=ADMIN_TOKEN,
        method="POST",
        expected=frozenset({202}),
    )
    deadline = time.monotonic() + 180
    last_status = ""
    while time.monotonic() < deadline:
        _status, repair = _json_request("/v1/admin/repair-status", token=ADMIN_TOKEN)
        last_status = str(repair.get("status") or "")
        if last_status == "ready":
            return last_status
        time.sleep(0.5)
    raise SmokeError(f"repair queues did not become ready: {last_status or 'unknown'}")


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(MISSING_CREDENTIALS, file=sys.stderr)
        return 2
    if AUDIO_TIMEOUT_SECONDS <= 0:
        print("AUDREY_AUDIO_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1}
    file_id = ""
    primary_error: Exception | None = None
    try:
        filename = f"c3-audio-{uuid.uuid4().hex[:10]}.mp3"
        fixture = _spoken_mp3()
        upload_status, uploaded = _upload_mp3(filename=filename, content=fixture)
        file_id = str(uploaded.get("id") or "")
        if (
            not file_id
            or uploaded.get("filename") != filename
            or uploaded.get("mime") != "audio/mpeg"
            or uploaded.get("kind") != "audio"
            or uploaded.get("status") != "pending"
        ):
            raise SmokeError(f"MP3 did not enter the audio queue: {uploaded}")

        ready, polls, elapsed = _wait_until_ready(file_id)
        chunks = ready.get("chunks")
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
            raise SmokeError(f"Ready audio row has no indexed chunks: {ready}")
        transcript_status, transcript = _json_request(
            f"/api/files/{file_id}/artifacts/transcript",
            token=USER_TOKEN,
        )
        text = str(transcript.get("text") or "")
        words = set(re.findall(r"[a-z]+", text.lower()))
        missing = sorted(_REQUIRED_WORDS - words)
        if missing:
            raise SmokeError(
                f"audio transcript missed required words {missing}: {text[:500]!r}",
            )
        if transcript.get("next_offset") is not None:
            raise SmokeError("short audio fixture unexpectedly exceeded one transcript page")

        summary_status, summary = _json_request(
            f"/api/files/{file_id}/artifacts/summary",
            token=USER_TOKEN,
        )
        summary_text = str(summary.get("text") or "").strip()
        if not summary_text:
            raise SmokeError("Ready audio row produced no library summary")
        visual_status, _visual = _json_request(
            f"/api/files/{file_id}/artifacts/visual",
            token=USER_TOKEN,
            expected=frozenset({422}),
        )

        result["upload"] = {
            "http": upload_status,
            "filename": filename,
            "bytes": len(fixture),
            "kind": uploaded.get("kind"),
            "initial_status": uploaded.get("status"),
        }
        result["processing"] = {
            "final_status": ready.get("status"),
            "chunks": chunks,
            "polls": polls,
            "elapsed_seconds": round(elapsed, 3),
            "duration_seconds": ready.get("duration_s"),
            "transcript_source": ready.get("transcript_source"),
        }
        result["transcript"] = {
            "http": transcript_status,
            "chars": len(text),
            "required_words": sorted(_REQUIRED_WORDS),
        }
        result["artifacts"] = {
            "summary_http": summary_status,
            "summary_chars": len(summary_text),
            "visual_http": visual_status,
        }
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if file_id:
            try:
                delete_status, deleted = _json_request(
                    f"/api/files/{file_id}",
                    token=USER_TOKEN,
                    method="DELETE",
                    expected=frozenset({200, 404}),
                )
                was_deleted = bool(deleted.get("deleted"))
                if primary_error is None and not was_deleted:
                    raise SmokeError("successful audio smoke did not delete its upload")
                result["cleanup"] = {
                    "delete_http": delete_status,
                    "deleted": was_deleted,
                    "repair_status": _repair_until_ready(),
                }
            except Exception as exc:  # noqa: BLE001 - report cleanup separately
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

