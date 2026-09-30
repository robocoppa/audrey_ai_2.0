#!/usr/bin/env python3
"""Prove deployed scanned-PDF OCR from upload through the native text reader.

The fixture is generated as pixels and saved as a one-page PDF, so it has no
text layer for Audrey's synchronous PDF parser to read. The smoke expects the
upload to queue, waits for the media worker to finish OCR and indexing, reads
back the derived text, then deletes the temporary file.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import uuid
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OCR_TIMEOUT_SECONDS = float(os.getenv("AUDREY_OCR_SMOKE_TIMEOUT_SECONDS", "240"))
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin
_REQUIRED_WORDS = frozenset({"audrey", "scanned", "invoice", "forty", "dollars"})


class SmokeError(RuntimeError):
    """A deployed scanned-PDF OCR contract was not satisfied."""


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


def _scanned_pdf() -> bytes:
    """Build a high-contrast PDF whose words exist only as image pixels."""
    image = Image.new("RGB", (1800, 1200), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=72)
    except OSError as exc:
        raise SmokeError("DejaVu Sans is unavailable for the OCR fixture") from exc
    lines = (
        "Audrey scanned document",
        "The invoice total is forty dollars",
        "Process this page with OCR",
    )
    for index, line in enumerate(lines):
        draw.text((120, 170 + index * 180), line, fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PDF", resolution=150)
    return output.getvalue()


def _upload_pdf(*, filename: str, content: bytes) -> tuple[int, dict[str, Any]]:
    boundary = f"audrey-{uuid.uuid4().hex}"
    delimiter = boundary.encode()
    body = b"\r\n".join(
        (
            b"--" + delimiter,
            f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
            b"Content-Type: application/pdf",
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
    deadline = time.monotonic() + OCR_TIMEOUT_SECONDS
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
            raise SmokeError(f"OCR job failed: {reason}")
        if status not in {"pending", "processing"}:
            raise SmokeError(f"OCR job entered unexpected status {status!r}: {last}")
        time.sleep(1)
    raise SmokeError(
        f"OCR job did not become ready within {OCR_TIMEOUT_SECONDS:g}s: {last}",
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
    if OCR_TIMEOUT_SECONDS <= 0:
        print("AUDREY_OCR_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1}
    file_id = ""
    primary_error: Exception | None = None
    try:
        filename = f"c3-scanned-pdf-{uuid.uuid4().hex[:10]}.pdf"
        fixture = _scanned_pdf()
        upload_status, uploaded = _upload_pdf(filename=filename, content=fixture)
        file_id = str(uploaded.get("id") or "")
        if (
            not file_id
            or uploaded.get("filename") != filename
            or uploaded.get("mime") != "application/pdf"
            or uploaded.get("kind") != "text"
            or uploaded.get("status") != "pending"
        ):
            raise SmokeError(f"scanned PDF did not enter the OCR queue: {uploaded}")

        ready, polls, elapsed = _wait_until_ready(file_id)
        chunks = ready.get("chunks")
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
            raise SmokeError(f"Ready OCR row has no indexed chunks: {ready}")
        _reader_status, page = _json_request(f"/api/files/{file_id}/text", token=USER_TOKEN)
        text = str(page.get("text") or "")
        words = set(re.findall(r"[a-z]+", text.lower()))
        missing = sorted(_REQUIRED_WORDS - words)
        if missing:
            raise SmokeError(f"OCR text missed required words {missing}: {text[:500]!r}")
        if "--- Page 1 ---" not in text:
            raise SmokeError("OCR text omitted its page marker")
        if page.get("next_offset") is not None:
            raise SmokeError("one-page OCR fixture unexpectedly exceeded one reader page")

        result["upload"] = {
            "http": upload_status,
            "filename": filename,
            "bytes": len(fixture),
            "initial_status": uploaded.get("status"),
        }
        result["processing"] = {
            "final_status": ready.get("status"),
            "chunks": chunks,
            "polls": polls,
            "elapsed_seconds": round(elapsed, 3),
        }
        result["reader"] = {
            "http": _reader_status,
            "chars": len(text),
            "page_marker": True,
            "required_words": sorted(_REQUIRED_WORDS),
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
                    raise SmokeError("successful OCR smoke did not delete its upload")
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
