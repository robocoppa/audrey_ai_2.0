#!/usr/bin/env python3
"""Prove the deployed non-streaming, text-only Responses API contract."""

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


def _validate_completed(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("object") != "response" or response.get("status") != "completed":
        raise SmokeError(f"invalid response envelope: {response}")
    if not str(response.get("id") or "").startswith("resp_"):
        raise SmokeError("response id does not use the resp_ prefix")
    if response.get("model") != "audrey_fast":
        raise SmokeError(f"unexpected response model: {response.get('model')!r}")
    if "choices" in response:
        raise SmokeError("Responses endpoint returned a Chat Completions choices array")

    output = response.get("output") or []
    if len(output) != 1:
        raise SmokeError(f"expected one output item, got {len(output)}")
    message = output[0]
    parts = message.get("content") or []
    if (
        message.get("type") != "message"
        or message.get("role") != "assistant"
        or message.get("status") != "completed"
        or len(parts) != 1
        or parts[0].get("type") != "output_text"
    ):
        raise SmokeError(f"invalid output message: {message}")
    text = str(parts[0].get("text") or "")
    if response.get("output_text") != text:
        raise SmokeError("top-level output_text does not match the typed output item")
    if "RESPONSES_OK" not in text:
        raise SmokeError(f"model answer omitted the sentinel: {text[:300]!r}")

    usage = response.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or total_tokens != input_tokens + output_tokens
    ):
        raise SmokeError(f"invalid token usage: {usage}")
    return {
        "http": 200,
        "id_prefix": "resp_",
        "model": response["model"],
        "status": response["status"],
        "output_type": parts[0]["type"],
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
        _status, completed = _request({
            "model": "audrey_fast",
            "instructions": "Return the requested sentinel and no explanation.",
            "input": "### Task:\nReply with exactly: RESPONSES_OK",
            "max_output_tokens": 32,
        })
        result["completed"] = _validate_completed(completed)

        rejected_status, rejected = _request(
            {
                "model": "audrey_fast",
                "input": "This request must not start generation.",
                "stream": True,
            },
            expected=frozenset({400}),
        )
        detail = rejected.get("detail") or {}
        if detail.get("error") != "responses_feature_unsupported":
            raise SmokeError(f"streaming rejection was not explicit: {rejected}")
        result["unsupported"] = {
            "stream_http": rejected_status,
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
