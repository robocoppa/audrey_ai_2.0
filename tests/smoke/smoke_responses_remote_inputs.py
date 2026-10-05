#!/usr/bin/env python3
"""Prove remote PDF/image Responses input and public-address denial.

Public W3C/Python fixtures support the full smoke or one selected generation
case. No uploads, library changes, admin credential, or browser action is required.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

if __package__:
    from . import smoke_responses_file_inputs as http
else:
    import smoke_responses_file_inputs as http

DOCUMENT_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
IMAGE_URL = "https://www.python.org/static/community_logos/python-logo.png"


def _payload(*, stream: bool, max_output_tokens: int = 4096) -> dict[str, Any]:
    return {
        "model": http.MODEL, "stream": stream, "max_output_tokens": max_output_tokens,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": (
                "### Task:\nRead the text in the attached PDF and identify the "
                "programming language whose logo is in the image. Reply with "
                "only the PDF text, a separator, and the language name."
            )},
            {"type": "input_file", "file_url": DOCUMENT_URL},
            {"type": "input_image", "image_url": IMAGE_URL, "detail": "low"},
        ]}],
    }


def _validate(response: dict[str, Any]) -> dict[str, Any]:
    diagnostics = http._response_diagnostics(response)
    answer = str(response.get("output_text") or "")
    if response.get("status") != "completed" or not str(response.get("id", "")).startswith("resp_"):
        raise http.SmokeError(f"remote response did not complete: {diagnostics}")
    if not re.search(r"dummy\s+pdf\s+file", answer, re.IGNORECASE) or not re.search(r"\bpython\b", answer, re.IGNORECASE):
        raise http.SmokeError(f"answer did not use the remote PDF and image: {diagnostics}")
    output = response.get("output") or []
    if len(output) != 1 or output[0].get("content", [{}])[0].get("type") != "output_text":
        raise http.SmokeError("remote response omitted its typed output_text")
    usage = response.get("usage") or {}
    if not all(isinstance(usage.get(key), int) for key in ("input_tokens", "output_tokens", "total_tokens")):
        raise http.SmokeError(f"invalid remote response usage: {usage}")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise http.SmokeError(f"remote response usage does not add up: {usage}")
    return {"http": 200, "id_prefix": "resp_", "pdf_text": True, "image_logo": "python", **diagnostics}


def _validate_stream(body: bytes, content_type: str) -> dict[str, Any]:
    if not content_type.startswith("text/event-stream"):
        raise http.SmokeError(f"unexpected remote stream type: {content_type}")
    events = []
    for block in body.decode().split("\n\n"):
        lines = block.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "")
        if data:
            event = json.loads(data)
            if event.get("type") != name:
                raise http.SmokeError("remote SSE labels do not match their typed events")
            events.append(event)
    if not events or events[0].get("type") != "response.created" or events[-1].get("type") != "response.completed":
        last = events[-1] if events else {}
        raise http.SmokeError(f"remote stream did not complete: {last.get('type')}; {http._response_diagnostics(last.get('response') or {})}")
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise http.SmokeError("remote stream sequence numbers are not contiguous")
    response = events[-1]["response"]
    deltas = [event["delta"] for event in events if event.get("type") == "response.output_text.delta"]
    if not deltas or "".join(deltas) != response.get("output_text"):
        raise http.SmokeError("remote stream deltas do not match the final answer")
    answer = "".join(deltas)
    if any(banner in answer for banner in (
        "> _Thinking", "> _Planning", "> _Dispatching panel", "> _Synthesizing",
    )):
        raise http.SmokeError("remote stream leaked a progress banner into output_text")
    result = _validate(response)
    result.update(terminal="response.completed", delta_count=len(deltas), event_count=len(events), progress_hidden=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "completed", "streamed"), default="all",
                        help="Run all checks, or only the selected answer case.")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Visible answer and reasoning token ceiling per call (default: 4096).")
    args = parser.parse_args(argv)
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if not http.BASE_URL or not http.USER_TOKEN:
        print("Set AUDREY_SMOKE_BASE_URL and AUDREY_USER_JWT in .env.test.local.", file=sys.stderr)
        return 2
    result: dict[str, Any] = {"schema": 1, "case": args.case, "uploads_created": 0}
    try:
        if args.case == "all":
            # Denials first: failures here must be ordinary JSON, even with stream=True.
            guards = {}
            for part_type, key in (("input_image", "image_url"), ("input_file", "file_url")):
                probe = {"model": http.MODEL, "stream": True, "input": [{"role": "user", "content": [
                    {"type": part_type, key: "http://127.0.0.1/private"},
                ]}]}
                status, body, content_type = http._request(
                    "/v1/responses", token=http.USER_TOKEN, method="POST", body=json.dumps(probe).encode(),
                    content_type="application/json", expected=frozenset({422}),
                )
                error = json.loads(body)
                if not content_type.startswith("application/json") or error.get("detail", {}).get("error") != "responses_remote_input_blocked":
                    raise http.SmokeError(f"private {part_type} denial changed: {error}")
                guards[part_type] = status
            result["guards"] = {**guards, "error": "responses_remote_input_blocked", "before_sse": True}
        if args.case in {"all", "completed"}:
            payload = _payload(stream=False, max_output_tokens=args.max_output_tokens)
            _, response = http._json("/v1/responses", token=http.USER_TOKEN, method="POST", payload=payload)
            result["completed"] = http._response_diagnostics(response)
            result["completed"].update(_validate(response))
        if args.case in {"all", "streamed"}:
            payload = _payload(stream=True, max_output_tokens=args.max_output_tokens)
            _, body, content_type = http._request(
                "/v1/responses", token=http.USER_TOKEN, method="POST", body=json.dumps(payload).encode(), content_type="application/json",
            )
            result["streamed"] = _validate_stream(body, content_type)
        result["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - structured operator evidence
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
