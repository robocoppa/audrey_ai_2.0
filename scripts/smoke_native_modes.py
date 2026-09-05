#!/usr/bin/env python3
"""Exercise every published Audrey mode through native AG-UI resources.

Run this against the deployed Audrey container with a disposable ordinary-user
token and a different admin-account token. Each mode gets its own canonical
conversation. The script removes every conversation and its archive projection
before returning, including after a failed assertion.
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

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_TOKEN = os.getenv("TEST_OWUI_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_OWUI_TOKEN", "")
MODE_TIMEOUT_SECONDS = float(os.getenv("AUDREY_MODE_SMOKE_TIMEOUT_SECONDS", "900"))
MODES = ("auto", "fast", "deep", "cloud", "local", "video", "research")
EXPECTED_MODELS = {mode: f"audrey_{mode}" for mode in MODES}


class SmokeError(RuntimeError):
    """A deployed all-mode contract was not satisfied."""


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


def _agent_turn(
    *,
    conversation_id: str,
    mode: str,
    run_ids: list[str],
) -> dict[str, Any]:
    canary = f"2C3-{mode.upper()}-READY"
    prompt = (
        f"Reply exactly: {canary}. "
        "This is a routing health check; do not call tools or add explanation."
    )
    body = json.dumps(
        {
            "threadId": conversation_id,
            "runId": f"native-mode-smoke-{mode}",
            "messages": [
                {
                    "id": f"native-mode-message-{mode}",
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
    ).encode()
    query = urlencode({"mode": mode})
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/api/agent?{query}",
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {USER_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    events: list[dict[str, Any]] = []
    started_at = time.monotonic()
    try:
        with urlopen(request, timeout=MODE_TIMEOUT_SECONDS) as response:  # noqa: S310
            if response.status != 200:
                raise SmokeError(f"POST /api/agent?mode={mode}: HTTP {response.status}")
            run_id = str(response.headers.get("X-Audrey-Run-ID") or "")
            if not run_id:
                raise SmokeError(f"{mode} omitted X-Audrey-Run-ID")
            run_ids.append(run_id)
            for raw_line in response:
                line = raw_line.decode(errors="replace").rstrip("\r\n")
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    except HTTPError as exc:
        excerpt = exc.read().decode(errors="replace")[:500]
        raise SmokeError(
            f"POST /api/agent?mode={mode}: HTTP {exc.code}: {excerpt}"
        ) from exc

    counts = Counter(str(event.get("type")) for event in events)
    terminal = events[-1] if events else {}
    answer = "".join(
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "TEXT_MESSAGE_CONTENT"
    )
    if counts["RUN_STARTED"] != 1 or counts["TEXT_MESSAGE_START"] != 1:
        raise SmokeError(f"{mode} emitted invalid start events: {dict(counts)}")
    if not (
        terminal.get("type") == "RUN_FINISHED"
        and terminal.get("outcome", {}).get("type") == "success"
    ):
        raise SmokeError(f"{mode} did not finish successfully: {terminal}")
    if canary not in answer:
        raise SmokeError(f"{mode} answer omitted {canary}: {answer!r}")

    run_id = run_ids[-1]
    _, persisted = _json_request(f"/api/runs/{run_id}", token=USER_TOKEN)
    if persisted.get("status") != "succeeded":
        raise SmokeError(f"{mode} persisted non-success run: {persisted}")
    if persisted.get("mode") != mode:
        raise SmokeError(f"{mode} persisted as {persisted.get('mode')!r}")
    expected_model = EXPECTED_MODELS[mode]
    if persisted.get("virtual_model") != expected_model:
        raise SmokeError(
            f"{mode} launched {persisted.get('virtual_model')!r}, expected {expected_model!r}"
        )

    _, message_page = _json_request(
        f"/api/conversations/{conversation_id}/messages?limit=100",
        token=USER_TOKEN,
    )
    messages = message_page.get("items", [])
    if len(messages) != 2:
        raise SmokeError(f"{mode} did not persist exactly one turn: {messages}")
    if messages[0].get("content") != prompt or messages[1].get("content") != answer:
        raise SmokeError(f"{mode} canonical messages did not match its AG-UI turn")

    return {
        "run_id": run_id,
        "virtual_model": persisted.get("virtual_model"),
        "concrete_model": persisted.get("concrete_model"),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "answer_chars": len(answer),
        "event_counts": dict(counts),
    }


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


def _cleanup(conversation_ids: list[str], run_ids: list[str]) -> dict[str, Any]:
    for run_id in run_ids:
        _request(
            f"/api/runs/{run_id}/cancel",
            token=USER_TOKEN,
            method="POST",
            expected=frozenset({200, 404}),
        )

    canonical_statuses: list[int] = []
    archive_statuses: list[int] = []
    for conversation_id in conversation_ids:
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
        canonical_statuses.append(canonical)
        archive_statuses.append(archive)

    repair = _repair_until_ready()
    return {
        "conversations": len(conversation_ids),
        "canonical_delete_http": dict(Counter(canonical_statuses)),
        "archive_delete_http": dict(Counter(archive_statuses)),
        "repair_status": repair.get("status"),
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print("TEST_OWUI_TOKEN and ADMIN_OWUI_TOKEN must be set.", file=sys.stderr)
        return 2
    if MODE_TIMEOUT_SECONDS <= 0:
        print("AUDREY_MODE_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1, "modes": {}}
    conversation_ids: list[str] = []
    run_ids: list[str] = []
    primary_error: Exception | None = None
    try:
        _, user = _json_request("/api/me", token=USER_TOKEN)
        _, admin = _json_request("/api/me", token=ADMIN_TOKEN)
        if not user.get("id") or user.get("id") == admin.get("id"):
            raise SmokeError("smoke tokens must resolve to two different Audrey users")

        _, model_page = _json_request("/v1/models", token=USER_TOKEN)
        published = {
            str(item.get("id"))
            for item in model_page.get("data", [])
            if str(item.get("id", "")).startswith("audrey_")
        }
        missing = set(EXPECTED_MODELS.values()) - published
        if missing:
            raise SmokeError(f"published model list omitted: {sorted(missing)}")

        for mode in MODES:
            _, conversation = _json_request(
                "/api/conversations",
                token=USER_TOKEN,
                method="POST",
                payload={
                    "title": f"C3 2C3 {mode.upper()} MODE SMOKE",
                    "default_mode": mode,
                },
                expected=frozenset({201}),
            )
            conversation_id = str(conversation["id"])
            conversation_ids.append(conversation_id)
            result["modes"][mode] = _agent_turn(
                conversation_id=conversation_id,
                mode=mode,
                run_ids=run_ids,
            )
        result["identity"] = {
            "user_id": user["id"],
            "admin_id": admin["id"],
        }
        result["published_models"] = sorted(published)
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conversation_ids:
            try:
                result["cleanup"] = _cleanup(conversation_ids, run_ids)
            except Exception as exc:  # noqa: BLE001 - report cleanup separately
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
