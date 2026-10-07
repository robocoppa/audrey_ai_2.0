#!/usr/bin/env python3
"""Two model requests proving reasoning admission; no uploads or stored responses.

Installed metadata confirmed Kimi supports false and GLM supports low.
This checks protocol delivery, not a measured quality or token-saving effect.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

KIMI = "audrey_passthrough/kimi-k3:cloud"
GLM = "audrey_passthrough/glm-5.3:cloud"
BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", os.getenv("AUDREY_EVAL_BASE_URL", "")).rstrip("/").removesuffix("/v1")
API_KEY = os.getenv("AUDREY_EVAL_API_KEY", "")
TIMEOUT_SECONDS = float(os.getenv("AUDREY_RESPONSES_SMOKE_TIMEOUT_SECONDS", "300"))


class SmokeError(RuntimeError):
    """The deployed reasoning request contract failed."""


def request(path: str, payload: dict[str, Any]) -> tuple[int, str, str]:
    req = Request(  # noqa: S310 - operator-controlled base URL
        BASE_URL + path, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY,
                 "Accept": "text/event-stream" if payload.get("stream") else "application/json"},
    )
    try:
        with urlopen(req, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            return response.status, response.headers.get("Content-Type", ""), response.read().decode()
    except HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read().decode()


def run(send=request) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": 1, "status": "failed", "generation_requests": 0, "guards": {}}
    chat_base = {"model": KIMI, "messages": [{"role": "user", "content": "Reply OK."}]}
    guards = [
        ("conflicting_controls", "/v1/chat/completions", {**chat_base, "stream": True,
         "think": False, "reasoning_effort": "none"}, "reasoning_controls_conflict"),
        ("virtual_model", "/v1/responses", {"model": "audrey_fast", "input": "Reply OK.",
         "stream": True, "reasoning": {"effort": "none"}}, "reasoning_unsupported"),
        ("glm_off", "/v1/responses", {"model": GLM, "input": "Reply OK.",
         "stream": True, "reasoning": {"effort": "none"}}, "reasoning_unsupported"),
    ]
    try:
        for name, path, payload, code in guards:
            status, content_type, raw = send(path, payload)
            parsed = json.loads(raw)
            if status != 400 or not content_type.startswith("application/json") or parsed.get("detail", {}).get("error") != code:
                raise SmokeError(f"{name} must return JSON 400 before SSE; HTTP {status}: {raw[:300]}")
            result["guards"][name] = status

        marker = "REASONING_" + uuid.uuid4().hex[:12].upper()
        prompt = f"Reply with only {marker}."
        result["generation_requests"] += 1
        status, content_type, raw = send("/v1/chat/completions", {
            "model": KIMI, "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "none", "max_tokens": 4096,
        })
        if status != 200 or not content_type.startswith("application/json"):
            raise SmokeError(f"Kimi reasoning none: HTTP {status}: {raw[:300]}")
        response = json.loads(raw)
        choices = response.get("choices") or []
        if (not str(response.get("id", "")).startswith("chatcmpl-") or len(choices) != 1
                or choices[0].get("finish_reason") != "stop"
                or marker not in str(choices[0].get("message", {}).get("content", ""))):
            raise SmokeError("Kimi did not complete the requested short answer.")
        result["completed_chat"] = {"http": status, "model": KIMI, "effort": "none", "sentinel": True}

        result["generation_requests"] += 1
        status, content_type, raw = send("/v1/responses", {
            "model": GLM, "input": prompt, "stream": True,
            "reasoning": {"effort": "low"}, "max_output_tokens": 4096,
        })
        if status != 200 or not content_type.startswith("text/event-stream"):
            raise SmokeError(f"GLM reasoning low: HTTP {status}: {raw[:300]}")
        events = []
        for block in raw.split("\n\n"):
            lines = block.splitlines()
            data = next((line[6:] for line in lines if line.startswith("data: ")), None)
            if data is None:
                continue
            event = json.loads(data)
            label = next((line[7:] for line in lines if line.startswith("event: ")), "")
            if label != event.get("type"):
                raise SmokeError("GLM stream event label and data disagree.")
            events.append(event)
        if (not events or events[0].get("type") != "response.created"
                or events[-1].get("type") != "response.completed"
                or [event.get("sequence_number") for event in events] != list(range(len(events)))):
            raise SmokeError("GLM stream did not start and complete normally.")
        terminal = events[-1].get("response") or {}
        deltas = "".join(event["delta"] for event in events if event.get("type") == "response.output_text.delta")
        if (terminal.get("status") != "completed" or terminal.get("reasoning") != {"effort": "low"}
                or not str(terminal.get("id", "")).startswith("resp_")
                or terminal.get("id") != events[0].get("response", {}).get("id")
                or terminal.get("output_text") != deltas or marker not in deltas):
            raise SmokeError("GLM final response omitted the answer, identity, or requested effort.")
        result["streamed_responses"] = {"http": status, "model": GLM, "effort": "low",
                                        "sentinel": True, "terminal": "response.completed"}
        result["status"] = "passed"
    except Exception as error:  # noqa: BLE001 - return copyable bounded failure evidence
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> int:
    if not BASE_URL or not API_KEY:
        print(json.dumps({"schema": 1, "status": "failed", "error": "Set AUDREY_SMOKE_BASE_URL and AUDREY_EVAL_API_KEY (compat:full PAT)."}, indent=2))
        return 1
    result = run()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
