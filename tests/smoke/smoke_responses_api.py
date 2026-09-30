#!/usr/bin/env python3
"""Prove the deployed text-only Responses API streaming contract."""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_raw_base = os.getenv("AUDREY_SMOKE_BASE_URL", "").rstrip("/")
if not _raw_base:
    _raw_base = os.getenv("AUDREY_EVAL_BASE_URL", "").rstrip("/")
if _raw_base.endswith("/v1"):
    _raw_base = _raw_base[:-3]
BASE_URL = _raw_base
API_KEY = os.getenv("AUDREY_EVAL_API_KEY", "")
TIMEOUT_SECONDS = float(os.getenv("AUDREY_RESPONSES_SMOKE_TIMEOUT_SECONDS", "300"))


class SmokeError(RuntimeError):
    """The deployed Responses API contract was not satisfied."""


def _request(
    payload: dict[str, Any],
    *,
    expected: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode()
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/v1/responses",
        data=body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            status = response.status
            content = response.read()
    except HTTPError as exc:
        status = exc.code
        content = exc.read()
    parsed = json.loads(content) if content else {}
    if status not in expected:
        raise SmokeError(f"POST /v1/responses: HTTP {status}: {parsed}")
    return status, parsed


def _stream_request(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]], str]:
    body = json.dumps(payload).encode()
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/v1/responses",
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            content = response.read().decode()
    except HTTPError as exc:
        parsed = json.loads(exc.read() or b"{}")
        raise SmokeError(f"stream POST /v1/responses: HTTP {exc.code}: {parsed}") from exc
    if status != 200:
        raise SmokeError(f"stream POST /v1/responses: HTTP {status}")
    if not content_type.lower().startswith("text/event-stream"):
        raise SmokeError(f"unexpected Responses stream content type: {content_type!r}")

    events: list[dict[str, Any]] = []
    for block in content.split("\n\n"):
        lines = block.splitlines()
        event_name = next(
            (line.removeprefix("event: ") for line in lines if line.startswith("event: ")),
            "",
        )
        data_line = next(
            (line.removeprefix("data: ") for line in lines if line.startswith("data: ")),
            "",
        )
        if not data_line:
            continue
        if data_line == "[DONE]":
            raise SmokeError("Responses stream used a Chat Completions [DONE] marker")
        event = json.loads(data_line)
        if event_name != event.get("type"):
            raise SmokeError(
                f"SSE event label {event_name!r} does not match data type {event.get('type')!r}"
            )
        events.append(event)
    if not events:
        raise SmokeError("Responses stream returned no typed events")
    return status, events, content


def _validate_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    types = [str(event.get("type") or "") for event in events]
    required_start = [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    required_end = [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    if types[:4] != required_start or types[-4:] != required_end:
        raise SmokeError(f"unexpected Responses event order: {types}")
    if any(event_type != "response.output_text.delta" for event_type in types[4:-4]):
        raise SmokeError(f"unexpected event inside text delta span: {types}")
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise SmokeError("Responses sequence_number values are not contiguous from zero")

    created = events[0].get("response") or {}
    response_id = str(created.get("id") or "")
    if not response_id.startswith("resp_"):
        raise SmokeError(f"invalid streamed response id: {response_id!r}")
    item = events[2].get("item") or {}
    message_id = str(item.get("id") or "")
    if not message_id.startswith("msg_"):
        raise SmokeError(f"invalid streamed message id: {message_id!r}")

    deltas = "".join(str(event.get("delta") or "") for event in events[4:-4])
    if "RESPONSES_STREAM_OK" not in deltas:
        raise SmokeError(f"streamed deltas omitted the sentinel: {deltas[-500:]!r}")
    completed = events[-1].get("response") or {}
    if completed.get("id") != response_id or completed.get("status") != "completed":
        raise SmokeError(f"invalid completed stream response: {completed}")
    if completed.get("output_text") != deltas:
        raise SmokeError("completed output_text does not match concatenated deltas")
    output = completed.get("output") or []
    if len(output) != 1 or output[0].get("id") != message_id:
        raise SmokeError(f"completed output item identity changed: {output}")
    if "choices" in completed:
        raise SmokeError("Responses stream terminal returned Chat Completions choices")

    usage = completed.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or usage.get("total_tokens") != input_tokens + output_tokens
    ):
        raise SmokeError(f"invalid streamed token usage: {usage}")
    return {
        "http": 200,
        "event_count": len(events),
        "delta_count": len(events) - 8,
        "id_prefix": "resp_",
        "message_id_prefix": "msg_",
        "terminal": "response.completed",
        "sentinel": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def main() -> int:
    if not BASE_URL or not API_KEY:
        print(
            "Set AUDREY_SMOKE_BASE_URL and AUDREY_EVAL_API_KEY "
            "(normally by sourcing .env.test.local).",
            file=sys.stderr,
        )
        return 2

    result: dict[str, Any] = {"schema": 1}
    try:
        _status, events, _raw = _stream_request({
            "model": "audrey_fast",
            "instructions": "Include the requested sentinel in a brief reply.",
            "input": "### Task:\nReply with exactly: RESPONSES_STREAM_OK",
            "max_output_tokens": 32,
            "stream": True,
        })
        result["streamed"] = _validate_stream(events)

        rejected_status, rejected = _request(
            {
                "model": "audrey_fast",
                "input": "This request must not start generation.",
                "background": True,
            },
            expected=frozenset({400}),
        )
        detail = rejected.get("detail") or {}
        if detail.get("error") != "responses_feature_unsupported":
            raise SmokeError(f"background rejection was not explicit: {rejected}")
        result["unsupported"] = {
            "background_http": rejected_status,
            "error": detail["error"],
        }
        result["status"] = "passed"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke must print structured failure
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
