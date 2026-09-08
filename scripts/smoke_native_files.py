#!/usr/bin/env python3
"""Exercise native file attachment ownership against deployed Audrey.

Run this inside the Audrey container with a disposable ordinary-user token and
a different admin-account token. The script creates only random smoke data and
removes its file, canonical conversation, and search projection before exiting.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import Counter
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_TOKEN = os.getenv("TEST_OWUI_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_OWUI_TOKEN", "")
RUN_TIMEOUT_SECONDS = float(os.getenv("AUDREY_FILE_SMOKE_TIMEOUT_SECONDS", "300"))


class SmokeError(RuntimeError):
    """A deployed native file or attachment contract was not satisfied."""


def _request(
    path: str,
    *,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    body: bytes | None = None,
    content_type: str = "",
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 300,
) -> tuple[int, bytes, Message]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        if body is not None:
            raise SmokeError("request cannot contain both JSON and raw bytes")
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    elif content_type:
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
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 300,
) -> tuple[int, dict[str, Any]]:
    status, content, _headers = _request(
        path,
        token=token,
        method=method,
        payload=payload,
        expected=expected,
        timeout=timeout,
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


def _agent_turn(
    *,
    conversation_id: str,
    file_id: str,
    marker: str,
) -> tuple[str, list[dict[str, Any]], str]:
    prompt = (
        "Use an authorized file or knowledge tool to read the attached file. "
        "Reply with only the private marker stored inside it. Do not guess."
    )
    body = json.dumps(
        {
            "threadId": conversation_id,
            "runId": "native-file-smoke-browser-id",
            "attachmentIds": [file_id],
            "messages": [
                {
                    "id": "native-file-smoke-message",
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
    ).encode()
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/api/agent?mode=fast",
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {USER_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    events: list[dict[str, Any]] = []
    run_id = ""
    try:
        with urlopen(request, timeout=RUN_TIMEOUT_SECONDS) as response:  # noqa: S310
            if response.status != 200:
                raise SmokeError(f"POST /api/agent: HTTP {response.status}")
            run_id = str(response.headers.get("X-Audrey-Run-ID") or "")
            for raw_line in response:
                line = raw_line.decode(errors="replace").rstrip("\r\n")
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    except HTTPError as exc:
        excerpt = exc.read().decode(errors="replace")[:500]
        raise SmokeError(f"POST /api/agent: HTTP {exc.code}: {excerpt}") from exc
    if not run_id:
        raise SmokeError("POST /api/agent omitted X-Audrey-Run-ID")
    answer = "".join(
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "TEXT_MESSAGE_CONTENT"
    )
    terminal = events[-1] if events else {}
    if not (
        terminal.get("type") == "RUN_FINISHED"
        and terminal.get("outcome", {}).get("type") == "success"
    ):
        raise SmokeError(f"attached run did not finish successfully: {terminal}")
    if marker not in answer:
        raise SmokeError("attached answer did not contain the private file marker")
    tool_names = {
        str(event.get("toolCallName") or "")
        for event in events
        if event.get("type") == "TOOL_CALL_START"
    }
    if not tool_names.intersection({"get_file_text", "kb_search", "list_my_files"}):
        raise SmokeError(f"attached run did not expose a file-reading tool call: {tool_names}")
    return run_id, events, prompt


def _repair_until_ready() -> dict[str, Any]:
    _json_request(
        "/v1/admin/repair",
        token=ADMIN_TOKEN,
        method="POST",
        expected=frozenset({202}),
    )
    deadline = time.monotonic() + 180
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, last = _json_request("/v1/admin/repair-status", token=ADMIN_TOKEN)
        if last.get("status") == "ready":
            return last
        time.sleep(0.5)
    raise SmokeError(f"repair queues did not become ready: {last}")


def _cleanup(
    *,
    file_id: str,
    conversation_id: str,
    run_id: str,
) -> dict[str, Any]:
    if run_id:
        _request(
            f"/api/runs/{run_id}/cancel",
            token=USER_TOKEN,
            method="POST",
            expected=frozenset({200, 404}),
        )
    file_status = 0
    if file_id:
        file_status, _, _ = _request(
            f"/api/files/{file_id}",
            token=USER_TOKEN,
            method="DELETE",
            expected=frozenset({200, 404}),
        )
    canonical_status = 0
    archive_status = 0
    if conversation_id:
        canonical_status, _, _ = _request(
            f"/api/conversations/{conversation_id}",
            token=USER_TOKEN,
            method="DELETE",
            expected=frozenset({204, 404}),
        )
        archive_status, _, _ = _request(
            f"/v1/me/chat-history/{conversation_id}",
            token=USER_TOKEN,
            method="DELETE",
            expected=frozenset({202, 404}),
        )
    repair = _repair_until_ready()
    return {
        "file_delete_http": file_status,
        "canonical_delete_http": canonical_status,
        "archive_delete_http": archive_status,
        "repair_status": repair.get("status"),
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print("TEST_OWUI_TOKEN and ADMIN_OWUI_TOKEN must be set.", file=sys.stderr)
        return 2
    if RUN_TIMEOUT_SECONDS <= 0:
        print("AUDREY_FILE_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1}
    file_id = ""
    conversation_id = ""
    run_id = ""
    primary_error: Exception | None = None
    try:
        _, user = _json_request("/api/me", token=USER_TOKEN)
        _, admin = _json_request("/api/me", token=ADMIN_TOKEN)
        if not user.get("id") or user.get("id") == admin.get("id"):
            raise SmokeError("smoke tokens must resolve to two different Audrey users")

        marker = f"AUDREY-FILE-{uuid.uuid4().hex.upper()}"
        filename = f"c3-2d1-{uuid.uuid4().hex[:10]}.txt"
        uploaded = _upload_text(filename=filename, content=f"Private marker: {marker}\n".encode())
        file_id = str(uploaded.get("id") or "")
        if not file_id or uploaded.get("filename") != filename or uploaded.get("status") != "ready":
            raise SmokeError(f"native upload was not ready: {uploaded}")

        _, listing = _json_request("/api/files", token=USER_TOKEN)
        listed = [item for item in listing.get("items", []) if item.get("id") == file_id]
        if len(listed) != 1 or "collection" in listed[0] or "user" in listing:
            raise SmokeError("native file listing was incomplete or leaked compatibility fields")
        _, inspected = _json_request(f"/api/files/{file_id}", token=USER_TOKEN)
        if inspected.get("filename") != filename:
            raise SmokeError("native file inspection did not return the uploaded owner file")
        cross_file, _, _ = _request(
            f"/api/files/{file_id}",
            token=ADMIN_TOKEN,
            expected=frozenset({404}),
        )

        _, conversation = _json_request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={"title": "C3 2D1 FILE ATTACHMENT", "default_mode": "fast"},
            expected=frozenset({201}),
        )
        conversation_id = str(conversation.get("id") or "")
        cross_conversation, _, _ = _request(
            f"/api/conversations/{conversation_id}",
            token=ADMIN_TOKEN,
            expected=frozenset({404}),
        )
        run_id, events, prompt = _agent_turn(
            conversation_id=conversation_id,
            file_id=file_id,
            marker=marker,
        )
        _, page = _json_request(
            f"/api/conversations/{conversation_id}/messages?limit=100",
            token=USER_TOKEN,
        )
        messages = page.get("items", [])
        if len(messages) != 2 or messages[0].get("content") != prompt:
            raise SmokeError("canonical attachment turn did not preserve exact user text")
        attachments = messages[0].get("attachments", [])
        if attachments != [
            {
                "id": file_id,
                "filename": filename,
                "mime": "text/plain",
                "kind": "text",
                "bytes": len(f"Private marker: {marker}\n".encode()),
            }
        ]:
            raise SmokeError(f"canonical attachment snapshot was incorrect: {attachments}")

        file_delete, _, _ = _request(
            f"/api/files/{file_id}",
            token=USER_TOKEN,
            method="DELETE",
        )
        file_id = ""
        _, after_delete = _json_request(
            f"/api/conversations/{conversation_id}/messages?limit=100",
            token=USER_TOKEN,
        )
        if not after_delete.get("items", [{}])[0].get("attachments"):
            raise SmokeError("deleting source bytes removed the canonical attachment snapshot")

        counts = Counter(str(event.get("type")) for event in events)
        result.update(
            {
                "identity": {
                    "user_id": user.get("id"),
                    "admin_id": admin.get("id"),
                    "cross_file_http": cross_file,
                    "cross_conversation_http": cross_conversation,
                },
                "file": {
                    "filename": filename,
                    "bytes": uploaded.get("bytes"),
                    "status": uploaded.get("status"),
                    "delete_http": file_delete,
                    "snapshot_survived_delete": True,
                },
                "run": {
                    "run_id": run_id,
                    "event_counts": dict(counts),
                    "attachment_count": len(attachments),
                    "private_marker_retrieved": True,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            result["cleanup"] = _cleanup(
                file_id=file_id,
                conversation_id=conversation_id,
                run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001 - report cleanup separately
            result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if primary_error is None:
                primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
