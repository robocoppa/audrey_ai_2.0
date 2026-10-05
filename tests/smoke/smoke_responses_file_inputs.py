#!/usr/bin/env python3
"""Prove owned image/document references, streaming, and denial boundaries.

Creates a small red PNG and text document, uploads both as the smoke user,
makes two short model calls, and deletes only those temporary uploads.
No browser upload or existing library file is required.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
    from .smoke_responses_multimodal import _red_png_data_url
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
    from smoke_responses_multimodal import _red_png_data_url

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "").rstrip("/")
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin
MODEL = os.getenv("AUDREY_RESPONSES_MODEL", "audrey_fast")
TIMEOUT_SECONDS = float(os.getenv("AUDREY_RESPONSES_SMOKE_TIMEOUT_SECONDS", "300"))


class SmokeError(RuntimeError):
    """The deployed file-reference contract was not satisfied."""


def _request(
    path: str, *, token: str, method: str = "GET", body: bytes | None = None,
    content_type: str = "", expected: frozenset[int] = frozenset({200}),
) -> tuple[int, bytes, str]:
    headers = {"Accept": "application/json, text/event-stream", **_CREDENTIALS.headers_for(
        token, user_token=USER_TOKEN, admin_token=ADMIN_TOKEN,
    )}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}", data=body, headers=headers, method=method,
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            status, data = response.status, response.read()
            response_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        status, data = exc.code, exc.read()
        response_type = exc.headers.get("Content-Type", "")
    if status not in expected:
        raise SmokeError(f"{method} {path}: HTTP {status}: {data.decode(errors='replace')[:500]}")
    return status, data, response_type


def _json(path: str, *, token: str, method: str = "GET", payload=None, expected=frozenset({200})):
    status, data, _ = _request(
        path, token=token, method=method,
        body=json.dumps(payload).encode() if payload is not None else None,
        content_type="application/json" if payload is not None else "", expected=expected,
    )
    return status, json.loads(data) if data else {}


def _upload(filename: str, mime: str, content: bytes) -> dict[str, Any]:
    boundary = f"audrey-{uuid.uuid4().hex}"
    body = b"\r\n".join((
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
        f"Content-Type: {mime}".encode(), b"", content,
        f"--{boundary}--".encode(), b"",
    ))
    _, data, _ = _request(
        "/api/files", token=USER_TOKEN, method="POST", body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    return json.loads(data)


def _wait_ready(file_ids: list[str]) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        states = []
        for file_id in file_ids:
            _, record = _json(f"/api/files/{file_id}", token=USER_TOKEN)
            state = record.get("status")
            if state == "failed":
                raise SmokeError(f"temporary upload failed: {record.get('failure_reason')}")
            states.append(state)
        if all(state == "ready" for state in states):
            return
        time.sleep(0.5)
    raise SmokeError("temporary image and document did not become Ready within 180s")


def _payload(document_id: str, image_id: str, *, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL, "stream": stream, "max_output_tokens": 64,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": (
                "### Task:\nRead the launch code from the attached document and "
                "identify the dominant color in the attached image. Reply with "
                "only the launch code, a space, and the color in uppercase."
            )},
            {"type": "input_file", "file_id": document_id},
            {"type": "input_image", "file_id": image_id, "detail": "low"},
        ]}],
    }


def _validate_response(response: dict[str, Any], sentinel: str) -> dict[str, Any]:
    if response.get("status") != "completed" or not str(response.get("id", "")).startswith("resp_"):
        raise SmokeError(f"invalid completed response: {response}")
    text = str(response.get("output_text") or "")
    if sentinel not in text or not re.search(r"\bRED\b", text, re.IGNORECASE):
        raise SmokeError(f"answer did not use both file inputs: {text[-500:]!r}")
    output = response.get("output") or []
    if len(output) != 1 or output[0].get("content", [{}])[0].get("type") != "output_text":
        raise SmokeError("completed response omitted its typed output_text")
    usage = response.get("usage") or {}
    if not all(isinstance(usage.get(key), int) for key in ("input_tokens", "output_tokens", "total_tokens")):
        raise SmokeError(f"invalid usage: {usage}")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise SmokeError(f"token usage does not add up: {usage}")
    return {
        "http": 200, "id_prefix": "resp_", "document_sentinel": True,
        "image_color": "red", "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


def _validate_stream(body: bytes, content_type: str, sentinel: str) -> dict[str, Any]:
    if not content_type.startswith("text/event-stream"):
        raise SmokeError(f"unexpected stream content type: {content_type}")
    events = []
    for block in body.decode().split("\n\n"):
        lines = block.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "")
        if not data:
            continue
        event = json.loads(data)
        if name != event.get("type"):
            raise SmokeError("Responses SSE label did not match its typed event")
        events.append(event)
    if not events or events[0].get("type") != "response.created" or events[-1].get("type") != "response.completed":
        raise SmokeError("Responses stream did not start and complete normally")
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise SmokeError("Responses stream sequence numbers are not contiguous")
    deltas = [event["delta"] for event in events if event.get("type") == "response.output_text.delta"]
    response = events[-1]["response"]
    if not deltas or "".join(deltas) != response.get("output_text"):
        raise SmokeError("stream deltas do not match the completed answer")
    result = _validate_response(response, sentinel)
    result.update(terminal="response.completed", delta_count=len(deltas), event_count=len(events))
    return result


def main() -> int:
    if not BASE_URL or not USER_TOKEN or not ADMIN_TOKEN:
        print(f"Set AUDREY_SMOKE_BASE_URL. {MISSING_CREDENTIALS}", file=sys.stderr)
        return 2
    result: dict[str, Any] = {"schema": 1}
    file_ids: list[str] = []
    primary_error = None
    cleanup_errors = []
    try:
        _, user = _json("/api/me", token=USER_TOKEN)
        _, other = _json("/api/me", token=ADMIN_TOKEN)
        if not user.get("id") or not other.get("id") or user["id"] == other["id"]:
            raise SmokeError("the two smoke credentials must resolve to different Audrey accounts")
        sentinel = f"FILE_REF_{uuid.uuid4().hex[:12].upper()}"
        fixtures = [
            ("txt", "text/plain", f"The launch code is {sentinel}.\n".encode()),
            ("png", "image/png", base64.b64decode(_red_png_data_url().split(",", 1)[1])),
        ]
        for suffix, mime, content in fixtures:
            uploaded = _upload(f"c3-response-file-{uuid.uuid4().hex[:10]}.{suffix}", mime, content)
            file_id = str(uploaded.get("id") or "")
            if not file_id:
                raise SmokeError(f"upload omitted its native file id: {uploaded}")
            file_ids.append(file_id)
        document_id, image_id = file_ids
        _wait_ready(file_ids)
        result["uploads"] = {"count": 2, "ready": True, "document_id": document_id, "image_id": image_id}
        payload = _payload(document_id, image_id, stream=False)
        _, response = _json("/v1/responses", token=USER_TOKEN, method="POST", payload=payload)
        result["completed"] = _validate_response(response, sentinel)
        payload["stream"] = True
        _, body, content_type = _request(
            "/v1/responses", token=USER_TOKEN, method="POST", body=json.dumps(payload).encode(),
            content_type="application/json",
        )
        result["streamed"] = _validate_stream(body, content_type, sentinel)
        # Neither probe is allowed to start a model call or an SSE stream.
        for part_type, file_id in (("input_file", document_id), ("input_image", image_id)):
            probe = {"model": MODEL, "stream": True, "input": [{"role": "user", "content": [
                {"type": part_type, "file_id": file_id},
            ]}]}
            _, foreign = _json("/v1/responses", token=ADMIN_TOKEN, method="POST", payload=probe, expected=frozenset({404}))
            probe["input"][0]["content"][0]["file_id"] = "file_missing_smoke"
            _, missing = _json("/v1/responses", token=USER_TOKEN, method="POST", payload=probe, expected=frozenset({404}))
            if foreign != missing or foreign != {"detail": "File not found."}:
                raise SmokeError(f"{part_type}: foreign and missing references did not return the same not-found result")
        wrong_payload = _payload(image_id, document_id, stream=True)
        wrong_status, _ = _json("/v1/responses", token=USER_TOKEN, method="POST", payload=wrong_payload, expected=frozenset({422}))
        result["guards"] = {"foreign_http": 404, "missing_http": 404, "same_not_found": True, "wrong_kind_http": wrong_status}
    except Exception as exc:  # noqa: BLE001 - structured operator error
        primary_error = exc
    finally:
        deleted = 0
        for file_id in file_ids:
            try:
                _, removal = _json(f"/api/files/{file_id}", token=USER_TOKEN, method="DELETE")
                if removal.get("deleted") is not True:
                    raise SmokeError(f"temporary upload was not deleted: {file_id}")
                deleted += 1
            except Exception as exc:  # noqa: BLE001 - always attempt the other cleanup
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        result["cleanup"] = {"uploads_deleted": deleted}
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
    if primary_error is None and not cleanup_errors:
        try:
            deleted_payload = _payload(file_ids[0], file_ids[1], stream=True)
            status, _ = _json("/v1/responses", token=USER_TOKEN, method="POST", payload=deleted_payload, expected=frozenset({404}))
            result["guards"]["deleted_http"] = status
        except Exception as exc:  # noqa: BLE001 - structured operator error
            primary_error = exc
    if primary_error is not None:
        result["error"] = f"{type(primary_error).__name__}: {primary_error}"
    result["status"] = "passed" if primary_error is None and not cleanup_errors else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
