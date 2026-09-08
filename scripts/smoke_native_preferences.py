#!/usr/bin/env python3
"""Exercise owner-bound native preferences against deployed Audrey.

Run this inside the Audrey container with an ordinary-user provider token and
a different admin-account token. The script restores the ordinary user's exact
original preferences and removes its disposable conversation before exiting.
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

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_TOKEN = os.getenv("TEST_OWUI_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_OWUI_TOKEN", "")
RUN_TIMEOUT_SECONDS = float(os.getenv("AUDREY_PREFERENCES_SMOKE_TIMEOUT_SECONDS", "300"))

PREFERENCE_FIELDS = ("timezone", "persona", "detail", "tone", "show_progress")


class SmokeError(RuntimeError):
    """A deployed native preference contract was not satisfied."""


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


def _preference_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in PREFERENCE_FIELDS}


def _replace_preferences(payload: dict[str, Any]) -> dict[str, Any]:
    _status, updated = _json_request(
        "/api/me/preferences",
        token=USER_TOKEN,
        method="PUT",
        payload=payload,
    )
    return updated


def _agent_turn(
    *,
    conversation_id: str,
    marker: str,
) -> tuple[str, list[dict[str, Any]], str, str]:
    prompt = (
        "State the saved verification marker and IANA timezone from my Audrey "
        "account preferences. Use one short line."
    )
    body = json.dumps(
        {
            "threadId": conversation_id,
            "runId": "native-preferences-smoke-browser-id",
            "messages": [
                {
                    "id": "native-preferences-smoke-message",
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
        raise SmokeError(f"preference run did not finish successfully: {terminal}")
    if marker not in answer or "America/Denver" not in answer:
        raise SmokeError("model answer omitted its saved marker or timezone")
    return run_id, events, prompt, answer


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


def _cleanup(*, conversation_id: str, run_id: str) -> dict[str, Any]:
    if run_id:
        _request(
            f"/api/runs/{run_id}/cancel",
            token=USER_TOKEN,
            method="POST",
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
        "canonical_delete_http": canonical_status,
        "archive_delete_http": archive_status,
        "repair_status": repair.get("status"),
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print("TEST_OWUI_TOKEN and ADMIN_OWUI_TOKEN must be set.", file=sys.stderr)
        return 2
    if RUN_TIMEOUT_SECONDS <= 0:
        print("AUDREY_PREFERENCES_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1}
    original_user: dict[str, Any] | None = None
    original_admin: dict[str, Any] | None = None
    conversation_id = ""
    run_id = ""
    primary_error: Exception | None = None
    try:
        _, user = _json_request("/api/me", token=USER_TOKEN)
        _, admin = _json_request("/api/me", token=ADMIN_TOKEN)
        if not user.get("id") or user.get("id") == admin.get("id"):
            raise SmokeError("smoke tokens must resolve to two different Audrey users")

        _, original_user = _json_request("/api/me/preferences", token=USER_TOKEN)
        _, original_admin = _json_request("/api/me/preferences", token=ADMIN_TOKEN)
        invalid_status, invalid_content, _ = _request(
            "/api/me/preferences",
            token=USER_TOKEN,
            method="PUT",
            payload={
                **_preference_payload(original_user),
                "timezone": "Mountain Time",
            },
            expected=frozenset({422}),
        )
        invalid = json.loads(invalid_content)
        _, after_invalid = _json_request("/api/me/preferences", token=USER_TOKEN)
        if _preference_payload(after_invalid) != _preference_payload(original_user):
            raise SmokeError("invalid timezone changed the stored preferences")

        marker = f"AUDREY-PREF-{uuid.uuid4().hex.upper()}"
        persona = (
            f"The saved verification marker is {marker}. "
            + "Prefer direct, practical explanations with explicit assumptions. " * 45
        )
        desired = {
            "timezone": "America/Denver",
            "persona": persona,
            "detail": "detailed",
            "tone": "professional",
            "show_progress": False,
        }
        updated = _replace_preferences(desired)
        if _preference_payload(updated) != desired:
            raise SmokeError(f"preference update did not round-trip: {updated}")

        _, conversation = _json_request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={"title": "C3 2D2 PREFERENCES", "default_mode": "fast"},
            expected=frozenset({201}),
        )
        conversation_id = str(conversation.get("id") or "")
        run_id, events, prompt, answer = _agent_turn(
            conversation_id=conversation_id,
            marker=marker,
        )
        _, persisted = _json_request(f"/api/runs/{run_id}", token=USER_TOKEN)
        if persisted.get("status") != "succeeded":
            raise SmokeError(f"preference run did not persist success: {persisted}")
        _, messages = _json_request(
            f"/api/conversations/{conversation_id}/messages?limit=100",
            token=USER_TOKEN,
        )
        items = messages.get("items", [])
        if len(items) != 2 or items[0].get("content") != prompt:
            raise SmokeError("canonical preference turn did not preserve exact user text")
        if items[1].get("content") != answer:
            raise SmokeError("canonical assistant text differed from AG-UI output")

        _, admin_after = _json_request("/api/me/preferences", token=ADMIN_TOKEN)
        if admin_after != original_admin:
            raise SmokeError("ordinary-user preference update changed the admin owner")
        result.update(
            {
                "identity": {
                    "user_id": user.get("id"),
                    "admin_id": admin.get("id"),
                    "admin_unchanged": True,
                },
                "validation": {
                    "invalid_timezone_http": invalid_status,
                    "invalid_timezone_detail": invalid.get("detail"),
                    "invalid_write_preserved_state": True,
                },
                "preferences": {
                    "timezone": updated.get("timezone"),
                    "detail": updated.get("detail"),
                    "tone": updated.get("tone"),
                    "show_progress": updated.get("show_progress"),
                    "persona_chars": len(persona),
                },
                "run": {
                    "run_id": run_id,
                    "event_count": len(events),
                    "context_marker_retrieved": True,
                    "timezone_retrieved": True,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if original_user is not None:
            try:
                restored = _replace_preferences(_preference_payload(original_user))
                restored_exactly = _preference_payload(restored) == _preference_payload(
                    original_user
                )
                result["restore"] = {"original_preferences_restored": restored_exactly}
                if not restored_exactly:
                    raise SmokeError("original user preferences were not restored exactly")
            except Exception as exc:  # noqa: BLE001 - report restore separately
                result["restore_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc
        try:
            result["cleanup"] = _cleanup(
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
