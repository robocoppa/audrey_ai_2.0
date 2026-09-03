#!/usr/bin/env python3
"""Exercise native tool events, cancellation, `/v1`, and safe cleanup.

Run this against the deployed Audrey container. The script creates exactly one
disposable conversation owned by TEST_OWUI_TOKEN and removes both its archive
projection and canonical row before returning.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import Counter
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
USER_TOKEN = os.getenv("TEST_OWUI_TOKEN", "")
ADMIN_TOKEN = os.getenv("ADMIN_OWUI_TOKEN", "")
_WEB_SEARCH_METRIC = re.compile(
    r'^audrey_tool_calls_total\{[^}]*tool="web_search"[^}]*\}\s+([0-9.eE+-]+)$'
)


class SmokeError(RuntimeError):
    """A live smoke contract was not satisfied."""


def _request(
    path: str,
    *,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 180,
) -> tuple[int, bytes]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    request = Request(  # noqa: S310 - base URL is operator-controlled HTTP(S)
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
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 180,
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


def _stream_events(
    path: str,
    *,
    token: str,
    on_event: Any | None = None,
    timeout: float = 300,
) -> list[dict[str, Any]]:
    request = Request(  # noqa: S310 - base URL is operator-controlled HTTP(S)
        f"{BASE_URL}{path}",
        headers={
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {token}",
        },
    )
    events: list[dict[str, Any]] = []
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        if response.status != 200:
            raise SmokeError(f"GET {path}: HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode(errors="replace").rstrip("\r\n")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            events.append(event)
            if on_event is not None:
                on_event(event)
    return events


def _web_search_total() -> float:
    _, content = _request("/metrics")
    total = 0.0
    for line in content.decode().splitlines():
        match = _WEB_SEARCH_METRIC.fullmatch(line)
        if match:
            total += float(match.group(1))
    return total


def _repair_until_ready() -> dict[str, Any]:
    _json_request(
        "/v1/admin/repair",
        token=ADMIN_TOKEN,
        method="POST",
        expected=frozenset({202}),
    )
    deadline = time.monotonic() + 90
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, last = _json_request("/v1/admin/repair-status", token=ADMIN_TOKEN)
        if last.get("status") == "ready":
            return last
        time.sleep(0.5)
    raise SmokeError(f"repair queues did not become ready: {last}")


def _run_cancellation(conversation_id: str) -> dict[str, Any]:
    _, run = _json_request(
        f"/api/conversations/{conversation_id}/runs",
        token=USER_TOKEN,
        method="POST",
        payload={
            "content": (
                "Use web_search now. Compare several official sources about "
                "current Antarctic research programs and provide a detailed synthesis."
            ),
            "mode": "deep",
        },
    )

    tool_started = threading.Event()
    native_events: list[dict[str, Any]] = []
    stream_errors: list[str] = []

    def collect_native() -> None:
        try:
            native_events.extend(
                _stream_events(
                    str(run["events_url"]),
                    token=USER_TOKEN,
                    on_event=lambda event: (
                        tool_started.set()
                        if event.get("type") == "tool.started"
                        else None
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - report thread failure in main
            stream_errors.append(f"{type(exc).__name__}: {exc}")

    stream_thread = threading.Thread(target=collect_native, daemon=True)
    stream_thread.start()
    saw_active_tool = tool_started.wait(timeout=120)

    _, cancellation = _json_request(
        str(run["cancel_url"]),
        token=USER_TOKEN,
        method="POST",
    )
    stream_thread.join(timeout=60)
    if stream_thread.is_alive():
        raise SmokeError("native event stream remained open after cancellation")
    if stream_errors:
        raise SmokeError(f"native event stream failed: {stream_errors}")

    counts = Counter(str(event.get("type")) for event in native_events)
    finishes = [
        event for event in native_events if event.get("type") == "tool.finished"
    ]
    terminal = native_events[-1] if native_events else {}
    if not saw_active_tool:
        raise SmokeError("the run ended without observing an active tool")
    if not (
        counts["tool.started"]
        == counts["tool.arguments"]
        == counts["tool.finished"]
    ):
        raise SmokeError(f"unbalanced native tool lifecycle: {dict(counts)}")
    if not any(event.get("error") == "cancelled" for event in finishes):
        raise SmokeError(f"no interrupted tool was closed as cancelled: {finishes}")
    if not (
        terminal.get("type") == "run.finished"
        and terminal.get("status") == "cancelled"
        and terminal.get("error_code") == "cancelled_by_user"
    ):
        raise SmokeError(f"unexpected native terminal: {terminal}")

    agui_events = _stream_events(str(run["agui_events_url"]), token=USER_TOKEN)
    agui_terminal = agui_events[-1] if agui_events else {}
    if not (
        agui_terminal.get("type") == "RUN_ERROR"
        and agui_terminal.get("code") == "cancelled_by_user"
    ):
        raise SmokeError(f"unexpected AG-UI terminal: {agui_terminal}")

    return {
        "run_id": run["id"],
        "saw_active_tool": saw_active_tool,
        "native_tool_counts": {
            name: counts[name]
            for name in ("tool.started", "tool.arguments", "tool.finished")
        },
        "cancelled_tool_count": sum(
            event.get("error") == "cancelled" for event in finishes
        ),
        "native_terminal": {
            "type": terminal.get("type"),
            "status": terminal.get("status"),
            "error_code": terminal.get("error_code"),
        },
        "agui_terminal": {
            "type": agui_terminal.get("type"),
            "code": agui_terminal.get("code"),
        },
        "cancel_status": cancellation.get("status"),
    }


def _run_v1_compatibility(conversation_id: str) -> dict[str, Any]:
    before = _web_search_total()
    _, response = _json_request(
        "/v1/chat/completions",
        token=USER_TOKEN,
        method="POST",
        payload={
            "model": "audrey_fast",
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use web_search now. Find the official United States "
                        "Antarctic Program homepage and answer with its title and URL."
                    ),
                }
            ],
            "stream": False,
        },
        timeout=300,
    )
    after = _web_search_total()
    choice = response["choices"][0]
    answer = str(choice["message"].get("content") or "")
    forbidden = (
        "tool.started",
        "source.observed",
        "TOOL_CALL_START",
        "RUN_STARTED",
    )
    if after <= before:
        raise SmokeError("/v1 completed without a measured web_search dispatch")
    if not answer.strip():
        raise SmokeError("/v1 returned an empty answer")
    leaked = [marker for marker in forbidden if marker in answer]
    if leaked:
        raise SmokeError(f"typed native event names leaked into /v1: {leaked}")
    if choice.get("finish_reason") != "stop":
        raise SmokeError(f"unexpected /v1 finish reason: {choice.get('finish_reason')}")
    return {
        "web_search_delta": after - before,
        "finish_reason": choice.get("finish_reason"),
        "answer_chars": len(answer),
        "typed_event_leaks": leaked,
    }


def _cleanup(conversation_id: str) -> dict[str, Any]:
    _repair_until_ready()
    archive_status, _ = _request(
        f"/v1/me/chat-history/{conversation_id}",
        token=USER_TOKEN,
        method="DELETE",
        expected=frozenset({200, 404}),
    )
    canonical_status, _ = _request(
        f"/api/conversations/{conversation_id}",
        token=USER_TOKEN,
        method="DELETE",
        expected=frozenset({204, 404}),
    )
    final_repair = _repair_until_ready()
    return {
        "archive_delete_http": archive_status,
        "canonical_delete_http": canonical_status,
        "repair_status": final_repair.get("status"),
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(
            "TEST_OWUI_TOKEN and ADMIN_OWUI_TOKEN must be set.",
            file=sys.stderr,
        )
        return 2

    conversation_id = ""
    result: dict[str, Any] = {"schema": 1}
    primary_error: Exception | None = None
    try:
        _, conversation = _json_request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={"title": "C3 2B4 TOOL EVENTS", "default_mode": "deep"},
        )
        conversation_id = str(conversation["id"])
        result["conversation_id"] = conversation_id
        result["cancellation"] = _run_cancellation(conversation_id)
        result["v1_compatibility"] = _run_v1_compatibility(conversation_id)
    except Exception as exc:  # noqa: BLE001 - preserve failure across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conversation_id:
            try:
                result["cleanup"] = _cleanup(conversation_id)
            except Exception as exc:  # noqa: BLE001 - report cleanup separately
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
