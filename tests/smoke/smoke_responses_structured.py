#!/usr/bin/env python3
"""Prove deployed Responses JSON Schema output for completed and streamed calls."""

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
TIMEOUT_SECONDS = float(
    os.getenv("AUDREY_RESPONSES_STRUCTURED_SMOKE_TIMEOUT_SECONDS", "300")
)
SENTINEL = "RESPONSES_STRUCTURED_OK"
EXPECTED_COUNT = 3
SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": [SENTINEL]},
        "count": {"type": "integer", "enum": [EXPECTED_COUNT]},
    },
    "required": ["status", "count"],
    "additionalProperties": False,
}


class SmokeError(RuntimeError):
    """The deployed structured Responses contract was not satisfied."""


def _payload(*, stream: bool) -> dict[str, Any]:
    return {
        "model": "audrey_fast",
        "instructions": (
            "Return the requested status and count. Follow the response schema exactly."
        ),
        "input": (
            "Return status RESPONSES_STRUCTURED_OK and count 3. "
            "Do not add any other fields."
        ),
        "max_output_tokens": 64,
        "stream": stream,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "audrey_structured_smoke",
                "description": "A fixed deployment sentinel and count.",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    }


def _post(
    payload: dict[str, Any],
    *,
    accept: str,
    expected: frozenset[int] = frozenset({200}),
) -> tuple[int, bytes, str]:
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/v1/responses",
        data=json.dumps(payload).encode(),
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            status = response.status
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        status = exc.code
        body = exc.read()
        content_type = exc.headers.get("Content-Type", "")
    if status not in expected:
        rendered = body.decode("utf-8", "replace")
        raise SmokeError(f"POST /v1/responses: HTTP {status}: {rendered}")
    return status, body, content_type


def _validate_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SmokeError(f"output_text is not JSON: {exc}: {text!r}") from exc
    expected = {"status": SENTINEL, "count": EXPECTED_COUNT}
    if value != expected:
        raise SmokeError(f"structured object mismatch: expected {expected}, got {value}")
    return value


def _validate_response(response: dict[str, Any]) -> dict[str, Any]:
    response_id = str(response.get("id") or "")
    if not response_id.startswith("resp_"):
        raise SmokeError(f"invalid response id: {response_id!r}")
    if response.get("status") != "completed":
        raise SmokeError(f"response did not complete: {response}")
    _validate_object(str(response.get("output_text") or ""))
    text_format = (response.get("text") or {}).get("format") or {}
    if text_format.get("type") != "json_schema":
        raise SmokeError(f"response did not echo json_schema format: {text_format}")
    if text_format.get("name") != "audrey_structured_smoke":
        raise SmokeError(f"response changed schema name: {text_format}")
    if text_format.get("schema") != SCHEMA:
        raise SmokeError("response changed the requested JSON schema")
    output = response.get("output") or []
    if len(output) != 1:
        raise SmokeError(f"expected one output item: {output}")
    content = output[0].get("content") or []
    if len(content) != 1 or content[0].get("type") != "output_text":
        raise SmokeError(f"expected one output_text part: {content}")
    usage = response.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or usage.get("total_tokens") != input_tokens + output_tokens
    ):
        raise SmokeError(f"invalid usage: {usage}")
    return {
        "http": 200,
        "id_prefix": "resp_",
        "output_type": "output_text",
        "format": "json_schema",
        "sentinel": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _parse_stream(body: bytes, content_type: str) -> list[dict[str, Any]]:
    if not content_type.lower().startswith("text/event-stream"):
        raise SmokeError(f"unexpected stream content type: {content_type!r}")
    events = []
    for block in body.decode().split("\n\n"):
        lines = block.splitlines()
        event_name = next(
            (line[7:] for line in lines if line.startswith("event: ")),
            "",
        )
        data = next(
            (line[6:] for line in lines if line.startswith("data: ")),
            "",
        )
        if not data:
            continue
        if data == "[DONE]":
            raise SmokeError("Responses stream used a Chat Completions marker")
        event = json.loads(data)
        if event_name != event.get("type"):
            raise SmokeError(
                f"SSE label {event_name!r} differs from type {event.get('type')!r}"
            )
        events.append(event)
    if not events:
        raise SmokeError("structured stream returned no events")
    return events


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
        raise SmokeError(f"unexpected structured stream order: {types}")
    middle = types[4:-4]
    if not middle or any(item != "response.output_text.delta" for item in middle):
        raise SmokeError(f"non-text event appeared in structured delta span: {types}")
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise SmokeError("stream sequence numbers are not contiguous")

    deltas = "".join(str(event.get("delta") or "") for event in events[4:-4])
    _validate_object(deltas)
    if "Thinking" in deltas or "Planning" in deltas or "Synthesizing" in deltas:
        raise SmokeError(f"progress banner contaminated JSON deltas: {deltas!r}")
    completed = events[-1].get("response") or {}
    if completed.get("output_text") != deltas:
        raise SmokeError("stream terminal output_text differs from its deltas")
    result = _validate_response(completed)
    result.update({
        "event_count": len(events),
        "delta_count": len(middle),
        "terminal": "response.completed",
        "progress_hidden": True,
    })
    return result


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
        completed_status, completed_body, _content_type = _post(
            _payload(stream=False),
            accept="application/json",
        )
        completed = json.loads(completed_body)
        if completed_status != 200:
            raise SmokeError(f"completed request returned HTTP {completed_status}")
        result["completed"] = _validate_response(completed)

        streamed_status, streamed_body, streamed_type = _post(
            _payload(stream=True),
            accept="text/event-stream",
        )
        if streamed_status != 200:
            raise SmokeError(f"streamed request returned HTTP {streamed_status}")
        result["streamed"] = _validate_stream(
            _parse_stream(streamed_body, streamed_type)
        )

        legacy = {
            "model": "audrey_fast",
            "input": "This request must not start generation.",
            "text": {"format": {"type": "json_object"}},
        }
        rejected_status, rejected_body, _ = _post(
            legacy,
            accept="application/json",
            expected=frozenset({400}),
        )
        rejected = json.loads(rejected_body)
        detail = rejected.get("detail") or {}
        if detail.get("error") != "responses_feature_unsupported":
            raise SmokeError(f"legacy JSON mode rejection was not explicit: {rejected}")
        result["unsupported"] = {
            "json_object_http": rejected_status,
            "error": detail["error"],
        }

        result["status"] = "passed"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke prints a structured failure
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
