#!/usr/bin/env python3
"""Exercise canonical chat projection, rebuild, and native delete cleanup."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_TOKEN = os.getenv("TEST_OWUI_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_OWUI_TOKEN", "")
PROMPT = "Reply exactly: 2B5-PROJECTION-READY"


class SmokeError(RuntimeError):
    """A live projection contract was not satisfied."""


def _request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 300,
) -> tuple[int, bytes]:
    headers = {"Accept": "application/json"}
    body = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
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
    except HTTPError as exc:
        status = exc.code
        content = exc.read()
    if status not in expected:
        excerpt = content.decode(errors="replace")[:500]
        raise SmokeError(f"{method} {path}: HTTP {status}: {excerpt}")
    return status, content


def _json_request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 300,
) -> tuple[int, dict[str, Any]]:
    status, content = _request(
        path,
        token=token,
        method=method,
        payload=payload,
        expected=expected,
        timeout=timeout,
    )
    return status, json.loads(content) if content else {}


def _stream_events(path: str) -> list[dict[str, Any]]:
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}",
        headers={
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {USER_TOKEN}",
        },
    )
    events: list[dict[str, Any]] = []
    with urlopen(request, timeout=300) as response:  # noqa: S310
        for raw_line in response:
            line = raw_line.decode(errors="replace").rstrip("\r\n")
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    return events


def _export_conversation(conversation_id: str) -> list[dict[str, Any]]:
    cursor = ""
    matches: list[dict[str, Any]] = []
    for _ in range(100):
        query: dict[str, str | int] = {"limit": 200}
        if cursor:
            query["cursor"] = cursor
        _, page = _json_request(
            f"/v1/me/chat-history/export?{urlencode(query)}",
            token=USER_TOKEN,
        )
        matches.extend(
            item
            for item in page.get("items", [])
            if item.get("conversation_id") == conversation_id
        )
        cursor = str(page.get("next_cursor") or "")
        if not cursor:
            return matches
    raise SmokeError("archive export exceeded 100 pages")


def _canonical_turn(conversation_id: str) -> dict[str, str]:
    _, page = _json_request(
        f"/api/conversations/{conversation_id}/messages?limit=100",
        token=USER_TOKEN,
    )
    items = page.get("items", [])
    if page.get("next_cursor") is not None or len(items) != 2:
        raise SmokeError(f"canonical conversation was not exactly one turn: {items}")
    by_role = {str(item.get("role")): item for item in items}
    if set(by_role) != {"user", "assistant"}:
        raise SmokeError(f"canonical conversation roles were invalid: {items}")
    user_content = str(by_role["user"].get("content") or "")
    assistant_content = str(by_role["assistant"].get("content") or "")
    if user_content != PROMPT:
        raise SmokeError("canonical user content did not match the submitted prompt")
    if not assistant_content.strip():
        raise SmokeError("canonical assistant content was empty")
    return {
        "user": user_content,
        "assistant": assistant_content,
    }


def _wait_for_projection(
    conversation_id: str,
    *,
    present: bool,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 120
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last = _export_conversation(conversation_id)
        if bool(last) is present:
            return last
        time.sleep(0.5)
    state = "appear" if present else "disappear"
    raise SmokeError(f"archive projection did not {state}: {len(last)} message(s)")


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
        _, last = _json_request(
            "/v1/admin/repair-status",
            token=ADMIN_TOKEN,
        )
        if last.get("status") == "ready":
            return last
        time.sleep(0.5)
    raise SmokeError(f"repair queues did not become ready: {last}")


def _assert_one_turn(
    items: list[dict[str, Any]],
    canonical: dict[str, str],
) -> None:
    by_role = {str(item.get("role")): item for item in items}
    if len(items) != 2 or set(by_role) != {"user", "assistant"}:
        raise SmokeError(f"projection was not exactly one turn: {items}")
    if by_role["user"].get("content") != canonical["user"]:
        raise SmokeError("projected user content did not match canonical content")
    if by_role["assistant"].get("content") != canonical["assistant"]:
        raise SmokeError("projected assistant content did not match canonical content")


def _best_effort_cleanup(conversation_id: str, cancel_url: str) -> None:
    if cancel_url:
        _request(
            cancel_url,
            token=USER_TOKEN,
            method="POST",
            expected=frozenset({200, 404}),
        )
    _request(
        f"/api/conversations/{conversation_id}",
        token=USER_TOKEN,
        method="DELETE",
        expected=frozenset({204, 404, 409}),
    )
    _request(
        f"/v1/me/chat-history/{conversation_id}",
        token=USER_TOKEN,
        method="DELETE",
        expected=frozenset({202, 404}),
    )
    _repair_until_ready()


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(
            "TEST_OWUI_TOKEN and ADMIN_OWUI_TOKEN must be set.",
            file=sys.stderr,
        )
        return 2

    result: dict[str, Any] = {"schema": 1}
    conversation_id = ""
    cancel_url = ""
    deleted = False
    primary_error: Exception | None = None
    try:
        _, conversation = _json_request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={
                "title": "C3 2B5 CHAT PROJECTION",
                "default_mode": "fast",
            },
            expected=frozenset({201}),
        )
        conversation_id = str(conversation["id"])
        result["conversation_id"] = conversation_id
        _, run = _json_request(
            f"/api/conversations/{conversation_id}/runs",
            token=USER_TOKEN,
            method="POST",
            payload={"content": PROMPT, "mode": "fast"},
            expected=frozenset({202}),
        )
        cancel_url = str(run["cancel_url"])
        events = _stream_events(str(run["events_url"]))
        terminal = events[-1] if events else {}
        if not (
            terminal.get("type") == "run.finished"
            and terminal.get("status") == "succeeded"
        ):
            raise SmokeError(f"native run did not succeed: {terminal}")

        canonical = _canonical_turn(conversation_id)
        first = _wait_for_projection(conversation_id, present=True)
        _assert_one_turn(first, canonical)
        result["initial_projection_messages"] = len(first)
        result["assistant_chars"] = len(canonical["assistant"])

        _, rebuild = _json_request(
            "/v1/admin/chat_archive/rebuild-canonical",
            token=ADMIN_TOKEN,
            method="POST",
            expected=frozenset({202}),
        )
        _repair_until_ready()
        rebuilt = _wait_for_projection(conversation_id, present=True)
        _assert_one_turn(rebuilt, canonical)
        result["rebuild"] = {
            "projections_reset": rebuild.get("projections_reset"),
            "projection_messages": len(rebuilt),
        }

        delete_status, _ = _request(
            f"/api/conversations/{conversation_id}",
            token=USER_TOKEN,
            method="DELETE",
            expected=frozenset({204}),
        )
        deleted = True
        read_status, _ = _request(
            f"/api/conversations/{conversation_id}",
            token=USER_TOKEN,
            expected=frozenset({404}),
        )
        hidden = _wait_for_projection(conversation_id, present=False)
        repair = _repair_until_ready()
        result["deletion"] = {
            "canonical_delete_http": delete_status,
            "canonical_read_http": read_status,
            "projection_messages": len(hidden),
            "repair_status": repair.get("status"),
        }
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conversation_id and (primary_error is not None or not deleted):
            try:
                _best_effort_cleanup(conversation_id, cancel_url)
            except Exception as exc:  # noqa: BLE001 - report cleanup separately
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
