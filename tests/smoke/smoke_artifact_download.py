#!/usr/bin/env python3
"""Verify deployed downloads for the derived texts an existing video owns.

The smoke is read-only. It pages each artifact through the existing JSON reader,
downloads every non-empty artifact, checks exact UTF-8 bytes and filenames, and
proves missing sidecars are not advertised as downloads.
"""

from __future__ import annotations

import json
import os
import sys
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import SmokeCredentials
else:
    from smoke_native_auth import SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
FILE_ID = os.getenv("AUDREY_ARTIFACT_SMOKE_FILE_ID", "").strip()
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin
_ARTIFACT_SUFFIXES = {
    "summary": "summary",
    "transcript": "transcript",
    "visual": "visual-notes",
}


def _auth_headers() -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        USER_TOKEN,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


class SmokeError(RuntimeError):
    """A deployed artifact-download contract was not satisfied."""


def _request(
    path: str,
    *,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 60,
) -> tuple[int, bytes, Message]:
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}",
        headers={"Accept": "application/json", **_auth_headers()},
        method="GET",
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
        raise SmokeError(f"GET {path}: HTTP {status}: {excerpt}")
    return status, content, response_headers


def _json_request(path: str) -> dict[str, Any]:
    _status, content, _headers = _request(path)
    return json.loads(content)


def _candidate_videos(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if FILE_ID:
        selected = next((item for item in items if item.get("id") == FILE_ID), None)
        if selected is None:
            raise SmokeError(f"AUDREY_ARTIFACT_SMOKE_FILE_ID {FILE_ID!r} was not found")
        if selected.get("kind") != "video":
            raise SmokeError(f"AUDREY_ARTIFACT_SMOKE_FILE_ID {FILE_ID!r} is not a video")
        return [selected]

    candidates = [
        item
        for item in items
        if item.get("kind") == "video" and item.get("status") == "ready"
    ]
    if not candidates:
        raise SmokeError(
            "no ready video exists for the smoke user; upload or fetch a video "
            "and wait for it to reach Ready"
        )
    candidates.sort(
        key=lambda item: (
            not bool(item.get("source_freed_at")),
            str(item.get("uploaded_at") or ""),
        )
    )
    return candidates


def _read_artifact(file_id: str, artifact: str) -> str:
    parts: list[str] = []
    expected_total: int | None = None
    offset = 0
    seen: set[int] = set()
    encoded_file_id = quote(file_id, safe="")
    while True:
        page = _json_request(
            f"/api/files/{encoded_file_id}/artifacts/{artifact}?offset={offset}"
        )
        if page.get("id") != file_id or page.get("artifact") != artifact:
            raise SmokeError(f"{artifact} reader returned the wrong identity")
        if page.get("offset") != offset:
            raise SmokeError(f"{artifact} reader returned offset {page.get('offset')!r}")
        total = page.get("total_chars")
        if not isinstance(total, int) or total < 0:
            raise SmokeError(f"{artifact} reader returned an invalid total")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise SmokeError(f"{artifact} total changed while paging")
        parts.append(str(page.get("text") or ""))
        next_offset = page.get("next_offset")
        if next_offset is None:
            break
        if not isinstance(next_offset, int) or next_offset <= offset or next_offset in seen:
            raise SmokeError(f"{artifact} reader returned an invalid next offset")
        seen.add(next_offset)
        offset = next_offset

    text = "".join(parts)
    if len(text) != (expected_total or 0):
        raise SmokeError(
            f"{artifact} reader returned {len(text)} of {expected_total or 0} characters"
        )
    return text


def _download_filename(original: str, artifact: str) -> str:
    stem = Path(Path(original).name).stem or "video"
    return f"{stem}.{_ARTIFACT_SUFFIXES[artifact]}.txt"


def main() -> int:
    if not USER_TOKEN:
        print(
            "Set AUDREY_USER_JWT to an active Cloudflare Access application assertion.",
            file=sys.stderr,
        )
        return 2

    result: dict[str, Any] = {"schema": 1}
    try:
        listing = _json_request("/api/files")
        items = listing.get("items")
        if not isinstance(items, list):
            raise SmokeError("native file listing returned an invalid shape")
        candidates = _candidate_videos(items)
        video: dict[str, Any] | None = None
        artifact_texts: dict[str, str] = {}
        candidates_checked = 0
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "")
            if not candidate_id:
                continue
            candidates_checked += 1
            texts = {
                artifact: _read_artifact(candidate_id, artifact)
                for artifact in _ARTIFACT_SUFFIXES
            }
            if any(texts.values()):
                video = candidate
                artifact_texts = texts
                break

        if video is None:
            selected = f" {FILE_ID!r}" if FILE_ID else ""
            raise SmokeError(
                f"ready video{selected} has no transcript, visual notes, or summary; "
                "process a video with speech or visible content"
            )
        file_id = str(video.get("id") or "")
        filename = str(video.get("filename") or "")
        if not filename:
            raise SmokeError("selected video omitted its filename")

        artifact_results: dict[str, Any] = {}
        available = 0
        for artifact, text in artifact_texts.items():
            path = (
                f"/api/files/{quote(file_id, safe='')}/artifacts/{artifact}/download"
            )
            if not text:
                status, body, _headers = _request(
                    path,
                    expected=frozenset({404}),
                )
                detail = json.loads(body).get("detail") if body else ""
                if detail != "Artifact is unavailable.":
                    raise SmokeError(f"{artifact} absence returned an invalid detail")
                artifact_results[artifact] = {
                    "available": False,
                    "download_http": status,
                }
                continue

            status, body, headers = _request(path)
            expected_bytes = text.encode()
            if body != expected_bytes:
                raise SmokeError(f"{artifact} download bytes did not match its reader")
            if not str(headers.get("Content-Type") or "").startswith("text/plain"):
                raise SmokeError(f"{artifact} download did not use text/plain")
            expected_name = _download_filename(filename, artifact)
            if headers.get_filename() != expected_name:
                raise SmokeError(
                    f"{artifact} filename was {headers.get_filename()!r}, "
                    f"expected {expected_name!r}"
                )
            if headers.get("Cache-Control") != "private, no-store":
                raise SmokeError(f"{artifact} download omitted private no-store")
            if headers.get("X-Content-Type-Options") != "nosniff":
                raise SmokeError(f"{artifact} download omitted nosniff")
            available += 1
            artifact_results[artifact] = {
                "available": True,
                "bytes": len(body),
                "download_http": status,
                "filename": expected_name,
            }

        if available == 0:
            raise SmokeError(f"{filename!r} has no derived artifacts to download")
        result["file"] = {
            "id": file_id,
            "filename": filename,
            "original_reclaimed": bool(video.get("source_freed_at")),
        }
        result["artifacts"] = artifact_results
        result["available_count"] = available
        result["candidates_checked"] = candidates_checked
    except Exception as exc:  # noqa: BLE001 - emit one structured smoke result
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["status"] = "passed" if "error" not in result else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
