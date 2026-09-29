#!/usr/bin/env python3
"""Prove owner-scoped original-file downloads against deployed Audrey.

The smoke uploads one small text file, verifies full and ranged downloads,
proves a second account receives the same 404 as an unknown file, and removes
the file before exiting.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin


def _auth_headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        token,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


class SmokeError(RuntimeError):
    """A deployed original-file download contract was not satisfied."""


def _request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str = "",
    headers: dict[str, str] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 60,
) -> tuple[int, bytes, Message]:
    request_headers = {"Accept": "application/json", **_auth_headers(token)}
    if content_type:
        request_headers["Content-Type"] = content_type
    if headers:
        request_headers.update(headers)
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}",
        data=body,
        headers=request_headers,
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
    return status, json.loads(content) if content else {}


def _upload_text(*, filename: str, content: bytes) -> dict[str, Any]:
    boundary = f"audrey-{uuid.uuid4().hex}"
    delimiter = boundary.encode()
    body = b"\r\n".join(
        (
            b"--" + delimiter,
            f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
            b"Content-Type: text/plain",
            b"",
            content,
            b"--" + delimiter + b"--",
            b"",
        )
    )
    _status, response, _headers = _request(
        "/api/files",
        token=USER_TOKEN,
        method="POST",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    return json.loads(response)


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

    result: dict[str, Any] = {"schema": 1}
    file_id = ""
    primary_error: Exception | None = None
    try:
        _, user = _json_request("/api/me", token=USER_TOKEN)
        _, other_user = _json_request("/api/me", token=ADMIN_TOKEN)
        if not user.get("id") or user.get("id") == other_user.get("id"):
            raise SmokeError("smoke tokens must resolve to two different Audrey users")

        filename = f"c3-file-download-{uuid.uuid4().hex[:10]}.txt"
        content = f"Audrey original download {uuid.uuid4().hex}\n".encode()
        uploaded = _upload_text(filename=filename, content=content)
        file_id = str(uploaded.get("id") or "")
        if not file_id or uploaded.get("filename") != filename:
            raise SmokeError(f"native upload returned an invalid record: {uploaded}")

        path = f"/api/files/{file_id}/download"
        full_status, downloaded, full_headers = _request(path, token=USER_TOKEN)
        if downloaded != content:
            raise SmokeError("full download bytes did not match the uploaded original")
        if not str(full_headers.get("Content-Type") or "").startswith("text/plain"):
            raise SmokeError("download did not preserve the original media type")
        disposition = str(full_headers.get("Content-Disposition") or "")
        if "attachment" not in disposition or filename not in disposition:
            raise SmokeError(f"download filename was not preserved: {disposition}")
        if full_headers.get("Cache-Control") != "private, no-store":
            raise SmokeError("download omitted the private no-store cache policy")
        if full_headers.get("X-Content-Type-Options") != "nosniff":
            raise SmokeError("download omitted the nosniff policy")

        first = 7
        last = 19
        range_status, ranged, range_headers = _request(
            path,
            token=USER_TOKEN,
            headers={"Range": f"bytes={first}-{last}"},
            expected=frozenset({206}),
        )
        if ranged != content[first : last + 1]:
            raise SmokeError("ranged download bytes did not match the original")
        if range_headers.get("Content-Range") != f"bytes {first}-{last}/{len(content)}":
            raise SmokeError("ranged download returned an invalid Content-Range")

        cross_owner_status, _body, _headers = _request(
            path,
            token=ADMIN_TOKEN,
            expected=frozenset({404}),
        )
        result["identity"] = {
            "owner_id": user["id"],
            "other_user_id": other_user["id"],
            "cross_owner_http": cross_owner_status,
        }
        result["download"] = {
            "filename": filename,
            "bytes": len(content),
            "full_http": full_status,
            "range_http": range_status,
            "range": range_headers.get("Content-Range"),
            "attachment": True,
            "private_no_store": True,
            "nosniff": True,
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
                    expected=(
                        frozenset({200})
                        if primary_error is None
                        else frozenset({200, 404})
                    ),
                )
                if primary_error is None and not deleted.get("deleted"):
                    raise SmokeError("successful download smoke did not delete its upload")
                result["cleanup"] = {
                    "delete_http": delete_status,
                    "deleted": bool(deleted.get("deleted")),
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
