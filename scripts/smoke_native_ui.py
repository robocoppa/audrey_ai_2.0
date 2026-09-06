#!/usr/bin/env python3
"""Exercise the deployed native UI asset and authenticated AG-UI boundary.

Run this against the Audrey container with a disposable ordinary-user token and
a different admin-account token. The script creates one test-owned conversation,
proves both owner directions are hidden, and removes its canonical and projected
records before returning.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_TOKEN = os.getenv("TEST_OWUI_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_OWUI_TOKEN", "")
PROMPT = "Reply exactly: 2C1-NATIVE-UI-READY"
_SCRIPT_SOURCE = re.compile(rb'<script[^>]+src="(/assets/[^"?]+\.js)"')


class SmokeError(RuntimeError):
    """A deployed native UI contract was not satisfied."""


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
        headers["Authorization"] = f"Bearer {token}"
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
    return status, json.loads(content) if content else {}


def _agent_turn(conversation_id: str) -> tuple[str, list[dict[str, Any]]]:
    body = json.dumps(
        {
            "threadId": conversation_id,
            "runId": "browser-smoke-run",
            "messages": [
                {
                    "id": "browser-smoke-message",
                    "role": "user",
                    "content": PROMPT,
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
    with urlopen(request, timeout=300) as response:  # noqa: S310
        if response.status != 200:
            raise SmokeError(f"POST /api/agent: HTTP {response.status}")
        run_id = str(response.headers.get("X-Audrey-Run-ID") or "")
        for raw_line in response:
            line = raw_line.decode(errors="replace").rstrip("\r\n")
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    if not run_id:
        raise SmokeError("POST /api/agent omitted X-Audrey-Run-ID")
    return run_id, events


def _repair_until_ready() -> dict[str, Any]:
    _json_request(
        "/v1/admin/repair",
        token=ADMIN_TOKEN,
        method="POST",
        expected=frozenset({202}),
    )
    deadline = time.monotonic() + 120
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, last = _json_request("/v1/admin/repair-status", token=ADMIN_TOKEN)
        if last.get("status") == "ready":
            return last
        time.sleep(0.5)
    raise SmokeError(f"repair queues did not become ready: {last}")


def _listed_conversation_ids(*, archived: bool, search: str) -> list[str]:
    query = urlencode(
        {
            "archived": str(archived).lower(),
            "limit": 100,
            "q": search,
        }
    )
    _, page = _json_request(
        f"/api/conversations?{query}",
        token=USER_TOKEN,
    )
    return [str(item.get("id") or "") for item in page.get("items", [])]


def _cleanup(conversation_id: str, run_id: str) -> dict[str, Any]:
    if run_id:
        _request(
            f"/api/runs/{run_id}/cancel",
            token=USER_TOKEN,
            method="POST",
            expected=frozenset({200, 404}),
        )
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
        "canonical_delete_http": canonical_status,
        "archive_delete_http": archive_status,
        "repair_status": repair.get("status"),
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print("TEST_OWUI_TOKEN and ADMIN_OWUI_TOKEN must be set.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1}
    conversation_id = ""
    run_id = ""
    primary_error: Exception | None = None
    try:
        _, user = _json_request("/api/me", token=USER_TOKEN)
        _, admin = _json_request("/api/me", token=ADMIN_TOKEN)
        if not user.get("id") or user.get("id") == admin.get("id"):
            raise SmokeError("smoke tokens must resolve to two different Audrey users")

        _, html, headers = _request("/")
        script_match = _SCRIPT_SOURCE.search(html)
        if b'id="root"' not in html or script_match is None:
            raise SmokeError("/ did not return the built Audrey application shell")
        if "default-src 'self'" not in str(headers.get("Content-Security-Policy") or ""):
            raise SmokeError("/ omitted the native UI content security policy")
        script_path = script_match.group(1).decode()
        _, script, script_headers = _request(script_path)
        if not script or "javascript" not in str(script_headers.get_content_type()):
            raise SmokeError("native UI entry script was missing or had the wrong MIME type")
        result["assets"] = {
            "html_bytes": len(html),
            "entry_script": script_path,
            "entry_bytes": len(script),
            "content_security_policy": True,
        }

        _, conversation = _json_request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={"title": "C3 2C1 NATIVE UI", "default_mode": "fast"},
            expected=frozenset({201}),
        )
        conversation_id = str(conversation["id"])
        result["conversation_id"] = conversation_id

        cross_read, _, _ = _request(
            f"/api/conversations/{conversation_id}",
            token=ADMIN_TOKEN,
            expected=frozenset({404}),
        )
        cross_agent, _, _ = _request(
            "/api/agent",
            token=ADMIN_TOKEN,
            method="POST",
            payload={
                "threadId": conversation_id,
                "runId": "cross-owner-smoke",
                "messages": [
                    {"id": "cross-owner-message", "role": "user", "content": PROMPT}
                ],
            },
            expected=frozenset({404}),
        )

        run_id, events = _agent_turn(conversation_id)
        counts = Counter(str(event.get("type")) for event in events)
        terminal = events[-1] if events else {}
        answer = "".join(
            str(event.get("delta") or "")
            for event in events
            if event.get("type") == "TEXT_MESSAGE_CONTENT"
        )
        if counts["RUN_STARTED"] != 1 or counts["TEXT_MESSAGE_START"] != 1:
            raise SmokeError(f"AG-UI stream start events were invalid: {dict(counts)}")
        if not (
            terminal.get("type") == "RUN_FINISHED"
            and terminal.get("outcome", {}).get("type") == "success"
        ):
            raise SmokeError(f"AG-UI stream did not finish successfully: {terminal}")
        if "2C1-NATIVE-UI-READY" not in answer:
            raise SmokeError(f"AG-UI answer did not contain the canary: {answer!r}")

        _, message_page = _json_request(
            f"/api/conversations/{conversation_id}/messages?limit=100",
            token=USER_TOKEN,
        )
        messages = message_page.get("items", [])
        if len(messages) != 2:
            raise SmokeError(f"canonical conversation was not one turn: {messages}")
        if [item.get("role") for item in messages] != ["user", "assistant"]:
            raise SmokeError(f"canonical message roles were invalid: {messages}")
        if messages[0].get("content") != PROMPT or messages[1].get("content") != answer:
            raise SmokeError("canonical messages did not match the native AG-UI turn")

        managed_title = f"C3 2C2 100%_{conversation_id[-8:]}"
        _, renamed = _json_request(
            f"/api/conversations/{conversation_id}",
            token=USER_TOKEN,
            method="PATCH",
            payload={"title": managed_title},
        )
        if renamed.get("title") != managed_title:
            raise SmokeError("conversation rename did not persist")
        active_ids = _listed_conversation_ids(
            archived=False,
            search=managed_title,
        )
        if active_ids != [conversation_id]:
            raise SmokeError(f"literal active-title search was invalid: {active_ids}")

        _, archived = _json_request(
            f"/api/conversations/{conversation_id}",
            token=USER_TOKEN,
            method="PATCH",
            payload={"archived": True},
        )
        if not archived.get("archived_at"):
            raise SmokeError("conversation archive did not persist")
        active_after_archive = _listed_conversation_ids(
            archived=False,
            search=managed_title,
        )
        archived_ids = _listed_conversation_ids(
            archived=True,
            search=managed_title,
        )
        if active_after_archive or archived_ids != [conversation_id]:
            raise SmokeError(
                "archived conversation appeared in the wrong native list: "
                f"active={active_after_archive} archived={archived_ids}"
            )

        _, restored = _json_request(
            f"/api/conversations/{conversation_id}",
            token=USER_TOKEN,
            method="PATCH",
            payload={"archived": False},
        )
        if restored.get("archived_at") is not None:
            raise SmokeError("conversation restore did not persist")

        result["identity"] = {
            "user_id": user["id"],
            "admin_id": admin["id"],
            "cross_user_read_http": cross_read,
            "cross_user_agent_http": cross_agent,
        }
        result["run"] = {
            "run_id": run_id,
            "event_counts": dict(counts),
            "answer_chars": len(answer),
            "canonical_messages": len(messages),
        }
        result["management"] = {
            "renamed": True,
            "literal_search": True,
            "archived": True,
            "restored": True,
        }
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conversation_id:
            try:
                result["cleanup"] = _cleanup(conversation_id, run_id)
            except Exception as exc:  # noqa: BLE001 - report cleanup separately
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
