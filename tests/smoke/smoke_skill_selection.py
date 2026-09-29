#!/usr/bin/env python3
"""Prove one deployed native run selects and persists an explicit skill.

This targeted smoke makes one short Fast model call, verifies the streamed
AG-UI turn and durable skill provenance, then removes its conversation and
archive projection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin
SKILL_ID = "video-analysis"
SKILL_VERSION = 1
RUN_TIMEOUT_SECONDS = float(os.getenv("AUDREY_SKILL_SMOKE_TIMEOUT_SECONDS", "300"))


class SmokeError(RuntimeError):
    """The deployed explicit-skill contract was not satisfied."""


def _auth_headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        token,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


def _request(
    path: str,
    *,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 300,
) -> tuple[int, bytes, Message]:
    headers = {"Accept": "application/json"}
    if token:
        headers.update(_auth_headers(token))
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
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
    try:
        value = json.loads(content) if content else {}
    except json.JSONDecodeError as exc:
        raise SmokeError(f"{method} {path}: response was not JSON") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"{method} {path}: response was not an object")
    return status, value


def _agent_turn(*, conversation_id: str, run_ids: list[str]) -> dict[str, Any]:
    canary = "C3-3B-SKILL-READY"
    prompt = (
        f"Reply exactly: {canary}. "
        "This is a health check; do not call tools or add explanation."
    )
    body = json.dumps(
        {
            "threadId": conversation_id,
            "runId": "native-skill-selection-smoke",
            "messages": [
                {
                    "id": "native-skill-selection-message",
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
    ).encode()
    query = urlencode({"model": "fast", "skill": SKILL_ID})
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/api/agent?{query}",
        data=body,
        headers={
            "Accept": "text/event-stream",
            **_auth_headers(USER_TOKEN),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    events: list[dict[str, Any]] = []
    started_at = time.monotonic()
    try:
        with urlopen(request, timeout=RUN_TIMEOUT_SECONDS) as response:  # noqa: S310
            if response.status != 200:
                raise SmokeError(f"POST /api/agent: HTTP {response.status}")
            run_id = str(response.headers.get("X-Audrey-Run-ID") or "")
            if not run_id:
                raise SmokeError("explicit-skill run omitted X-Audrey-Run-ID")
            run_ids.append(run_id)
            for raw_line in response:
                line = raw_line.decode(errors="replace").rstrip("\r\n")
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    except HTTPError as exc:
        excerpt = exc.read().decode(errors="replace")[:500]
        raise SmokeError(f"POST /api/agent: HTTP {exc.code}: {excerpt}") from exc

    counts = Counter(str(event.get("type")) for event in events)
    terminal = events[-1] if events else {}
    answer = "".join(
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "TEXT_MESSAGE_CONTENT"
    )
    if counts["RUN_STARTED"] != 1 or counts["TEXT_MESSAGE_START"] != 1:
        raise SmokeError(f"explicit-skill run emitted invalid starts: {dict(counts)}")
    if not (
        terminal.get("type") == "RUN_FINISHED"
        and terminal.get("outcome", {}).get("type") == "success"
    ):
        raise SmokeError(f"explicit-skill run did not finish successfully: {terminal}")
    if canary not in answer:
        raise SmokeError(f"explicit-skill answer omitted {canary}: {answer!r}")

    run_id = run_ids[-1]
    _, persisted = _json_request(f"/api/runs/{run_id}", token=USER_TOKEN)
    expected = {
        "status": "succeeded",
        "mode": "fast",
        "virtual_model": "audrey_fast",
        "skill_id": SKILL_ID,
        "skill_version": SKILL_VERSION,
        "skill_reason": "request",
    }
    mismatches = {
        key: persisted.get(key)
        for key, value in expected.items()
        if persisted.get(key) != value
    }
    if mismatches:
        raise SmokeError(f"persisted explicit-skill run mismatch: {mismatches}")
    digest = str(persisted.get("skill_digest") or "")
    if len(digest) != 64:
        raise SmokeError("persisted explicit-skill run omitted its SHA-256 digest")

    _, page = _json_request(
        f"/api/conversations/{conversation_id}/messages?limit=100",
        token=USER_TOKEN,
    )
    messages = page.get("items", [])
    if len(messages) != 2:
        raise SmokeError(f"explicit-skill run did not persist one turn: {messages}")
    if messages[0].get("content") != prompt or messages[1].get("content") != answer:
        raise SmokeError("canonical messages did not match the streamed turn")

    return {
        "run_id": run_id,
        "virtual_model": persisted.get("virtual_model"),
        "concrete_model": persisted.get("concrete_model"),
        "skill_id": persisted.get("skill_id"),
        "skill_version": persisted.get("skill_version"),
        "skill_digest": digest,
        "skill_reason": persisted.get("skill_reason"),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "answer_chars": len(answer),
        "event_counts": dict(counts),
    }


def _repair_until_ready() -> str:
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
            return "ready"
        time.sleep(0.5)
    raise SmokeError(f"repair queues did not become ready: {last}")


def _cleanup(conversation_id: str, run_ids: list[str]) -> dict[str, Any]:
    for run_id in run_ids:
        _request(
            f"/api/runs/{run_id}/cancel",
            token=USER_TOKEN,
            method="POST",
            expected=frozenset({200, 404}),
        )
    canonical, _, _ = _request(
        f"/api/conversations/{conversation_id}",
        token=USER_TOKEN,
        method="DELETE",
        expected=frozenset({204, 404}),
    )
    archive, _, _ = _request(
        f"/v1/me/chat-history/{conversation_id}",
        token=USER_TOKEN,
        method="DELETE",
        expected=frozenset({202, 404}),
    )
    return {
        "canonical_delete_http": canonical,
        "archive_delete_http": archive,
        "repair_status": _repair_until_ready(),
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(MISSING_CREDENTIALS, file=sys.stderr)
        return 2
    if RUN_TIMEOUT_SECONDS <= 0:
        print("AUDREY_SKILL_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1}
    conversation_id = ""
    run_ids: list[str] = []
    primary_error: Exception | None = None
    try:
        _, user = _json_request("/api/me", token=USER_TOKEN)
        _, admin = _json_request("/api/me", token=ADMIN_TOKEN)
        if not user.get("id") or user.get("id") == admin.get("id"):
            raise SmokeError("smoke credentials must resolve to two Audrey users")

        _, conversation = _json_request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={
                "title": "C3 3B EXPLICIT SKILL SMOKE",
                "default_mode": "fast",
            },
            expected=frozenset({201}),
        )
        conversation_id = str(conversation["id"])
        result["identity"] = {"user_id": user["id"], "admin_id": admin["id"]}
        result["selection"] = _agent_turn(
            conversation_id=conversation_id,
            run_ids=run_ids,
        )
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conversation_id:
            try:
                result["cleanup"] = _cleanup(conversation_id, run_ids)
            except Exception as exc:  # noqa: BLE001 - report cleanup separately
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
