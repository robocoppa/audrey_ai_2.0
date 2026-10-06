#!/usr/bin/env python3
"""Prove caller-executed Responses functions and stateless tool-result continuation.

Makes three generation requests: completed call, streamed call, and a final
answer using the streamed call's locally computed result. Two JSON denial
probes run first. No uploads, server tool execution, or application writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from typing import Any

if __package__:
    from . import smoke_responses_file_inputs as http
else:
    import smoke_responses_file_inputs as http

MODEL = os.getenv("AUDREY_RESPONSES_TOOL_MODEL", "audrey_passthrough/qwen3.8:latest")
FUNCTION_NAME = "get_smoke_marker"
_PROGRESS_BANNERS = ("> _Thinking", "> _Planning", "> _Dispatching panel", "> _Synthesizing")
_ALLOWED_EVENTS = {
    "response.created", "response.in_progress", "response.completed",
    "response.output_item.added", "response.output_item.done",
    "response.function_call_arguments.delta", "response.function_call_arguments.done",
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done",
}


def _tools(nonce: str) -> list[dict[str, Any]]:
    return [{
        "type": "function", "name": FUNCTION_NAME,
        "description": "Read the smoke marker for the supplied nonce from the caller.",
        "strict": True,
        "parameters": {
            "type": "object", "properties": {"nonce": {"type": "string", "enum": [nonce]}},
            "required": ["nonce"], "additionalProperties": False,
        },
    }]


def _payload(nonce: str, *, stream: bool, max_output_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL, "stream": stream, "max_output_tokens": max_output_tokens,
        "tools": _tools(nonce), "tool_choice": "auto",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": (
                f"Call {FUNCTION_NAME} exactly once with nonce {nonce}. "
                "You do not know the marker: do not guess it. Stop after making "
                "the tool call. When its result is supplied, reply with only "
                "the marker value from that result."
            )},
        ]}],
    }


def _validate_common(response: dict[str, Any]) -> None:
    if response.get("status") != "completed" or not str(response.get("id", "")).startswith("resp_"):
        raise http.SmokeError(f"client-tool response did not complete: {http._response_diagnostics(response)}")
    usage = response.get("usage") or {}
    if not all(type(usage.get(key)) is int and usage[key] >= 0 for key in ("input_tokens", "output_tokens", "total_tokens")):
        raise http.SmokeError("client-tool response omitted integer token usage")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise http.SmokeError("client-tool response token usage does not add up")
    output = response.get("output")
    if not isinstance(output, list) or not output:
        raise http.SmokeError("client-tool response omitted typed output items")
    item_ids = []
    message_text = []
    for item in output:
        if not isinstance(item, dict) or item.get("status") != "completed":
            raise http.SmokeError("client-tool output item is not completed")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise http.SmokeError("client-tool output item omitted its ID")
        item_ids.append(item_id)
        if item.get("type") == "function_call":
            if not item_id.startswith("fc_"):
                raise http.SmokeError("function output omitted its fc_ item ID")
        elif item.get("type") == "message":
            content = item.get("content")
            if item.get("role") != "assistant" or not item_id.startswith("msg_") or not isinstance(content, list) or not content:
                raise http.SmokeError("client-tool assistant message has invalid typed content")
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                    raise http.SmokeError("client-tool assistant content is not output_text")
                message_text.append(part["text"])
        else:
            raise http.SmokeError("client-tool response included server-managed output")
    if len(set(item_ids)) != len(item_ids):
        raise http.SmokeError("client-tool output item IDs are duplicated")
    if "".join(message_text) != response.get("output_text", ""):
        raise http.SmokeError("client-tool output_text does not match assistant message items")
    if any(banner in "".join(message_text) for banner in _PROGRESS_BANNERS):
        raise http.SmokeError("client-tool response leaked a progress banner")


def _validate_call_response(response: dict[str, Any], nonce: str) -> dict[str, Any]:
    _validate_common(response)
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    if len(calls) != 1:
        raise http.SmokeError("client-tool response must request exactly one local function call")
    call = calls[0]
    if not str(call.get("call_id", "")).startswith("call_") or call.get("name") != FUNCTION_NAME:
        raise http.SmokeError("client-tool response has an unexpected call ID or function name")
    if not isinstance(call.get("arguments"), str):
        raise http.SmokeError("function arguments must be a JSON string")
    try:
        arguments = json.loads(call["arguments"])
    except (ValueError, TypeError) as exc:
        raise http.SmokeError("function arguments are not valid JSON") from exc
    if arguments != {"nonce": nonce}:
        raise http.SmokeError("function arguments do not match the declared schema and supplied nonce")
    return {
        "http": 200, "id_prefix": "resp_", "item_id_prefix": "fc_", "call_id_prefix": "call_",
        "function": FUNCTION_NAME, "function_calls": 1, "arguments_valid": True,
        "progress_hidden": True, **http._response_diagnostics(response),
    }


def _validate_stream(body: bytes, content_type: str, nonce: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not content_type.startswith("text/event-stream"):
        raise http.SmokeError(f"unexpected client-tool stream type: {content_type}")
    events = []
    for block in body.decode().split("\n\n"):
        lines = block.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "")
        if data:
            event = json.loads(data)
            if not isinstance(event, dict) or event.get("type") != name:
                raise http.SmokeError("client-tool SSE labels do not match typed events")
            events.append(event)
    if not events or events[0].get("type") != "response.created" or events[-1].get("type") != "response.completed":
        last = events[-1] if events else {}
        raise http.SmokeError(f"client-tool stream did not complete: {last.get('type')}; {http._response_diagnostics(last.get('response') or {})}")
    if [event.get("sequence_number") for event in events] != list(range(len(events))):
        raise http.SmokeError("client-tool stream sequence numbers are not contiguous")
    if any(event["type"] not in _ALLOWED_EVENTS for event in events):
        raise http.SmokeError("client-tool stream included server-managed tool or progress events")
    response = events[-1].get("response") or {}
    result = _validate_call_response(response, nonce)
    added = {}
    done = {}
    argument_deltas: dict[str, list[str]] = {}
    argument_done = {}
    text_deltas = []
    for event in events:
        kind = event["type"]
        if kind in {"response.output_item.added", "response.output_item.done"}:
            index = event.get("output_index")
            item = event.get("item")
            target = added if kind.endswith("added") else done
            if type(index) is not int or not isinstance(item, dict) or index in target:
                raise http.SmokeError("client-tool output item events have invalid indices")
            target[index] = item
        elif kind in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
            index = event.get("output_index")
            item_id = event.get("item_id")
            item = added.get(index)
            if not item or item.get("type") != "function_call" or item.get("id") != item_id:
                raise http.SmokeError("function argument event does not match an added item ID")
            key = "delta" if kind.endswith("delta") else "arguments"
            if not isinstance(event.get(key), str):
                raise http.SmokeError("function argument events require string arguments")
            if key == "delta":
                argument_deltas.setdefault(item_id, []).append(event[key])
            else:
                if item_id in argument_done:
                    raise http.SmokeError("function argument done event is duplicated")
                argument_done[item_id] = event[key]
        elif kind == "response.output_text.delta":
            if not isinstance(event.get("delta"), str):
                raise http.SmokeError("client-tool text delta is not a string")
            text_deltas.append(event["delta"])
    expected_indices = set(range(len(response["output"])))
    if set(added) != expected_indices or set(done) != expected_indices:
        raise http.SmokeError("client-tool stream omitted output item added/done events")
    if [done[index] for index in sorted(done)] != response["output"]:
        raise http.SmokeError("client-tool stream final output items do not match item done events")
    for index, item in enumerate(response["output"]):
        initial = added[index]
        if any(initial.get(key) != item.get(key) for key in ("type", "id", "call_id", "name")):
            raise http.SmokeError("client-tool stream changed an added item identity")
        if item["type"] == "function_call":
            item_id = item["id"]
            if not argument_deltas.get(item_id) or "".join(argument_deltas[item_id]) != item["arguments"] or argument_done.get(item_id) != item["arguments"]:
                raise http.SmokeError("function argument deltas do not match the final call arguments")
    if "".join(text_deltas) != response.get("output_text", ""):
        raise http.SmokeError("client-tool text deltas do not match final output_text")
    result.update(
        terminal="response.completed", event_count=len(events),
        argument_delta_count=sum(len(parts) for parts in argument_deltas.values()),
        argument_deltas_match=True, native_tool_events=0,
    )
    return response, result


def _execute_local(call: dict[str, Any], nonce: str) -> str:
    if call.get("name") != FUNCTION_NAME or json.loads(call["arguments"]) != {"nonce": nonce}:
        raise http.SmokeError("refusing to execute an undeclared local function")
    return "CLIENT_TOOL_" + hashlib.sha256(nonce.encode()).hexdigest()[:16].upper()


def _followup_payload(initial: dict[str, Any], response: dict[str, Any], marker: str) -> dict[str, Any]:
    call = next(item for item in response["output"] if item["type"] == "function_call")
    return {
        **initial, "stream": False, "tool_choice": "none",
        "input": [*initial["input"], *response["output"], {
            "type": "function_call_output", "call_id": call["call_id"],
            "output": json.dumps({"marker": marker}),
        }],
    }


def _validate_followup(response: dict[str, Any], marker: str) -> dict[str, Any]:
    _validate_common(response)
    if any(item["type"] == "function_call" for item in response["output"]):
        raise http.SmokeError("tool_choice none returned another function call")
    if response.get("output_text", "").strip() != marker:
        raise http.SmokeError(f"follow-up answer did not use the caller's tool result: {http._response_diagnostics(response)}")
    return {
        "http": 200, "id_prefix": "resp_", "sentinel": True, "tool_choice": "none",
        "caller_executed": True, "stateless_history": True, **http._response_diagnostics(response),
    }


def _denial_guard(payload: dict[str, Any]) -> int:
    status, body, content_type = http._request(
        "/v1/responses", token=http.USER_TOKEN, method="POST", body=json.dumps(payload).encode(),
        content_type="application/json", expected=frozenset({400}),
    )
    error = json.loads(body)
    if not content_type.startswith("application/json") or error.get("detail", {}).get("error") != "responses_feature_unsupported":
        raise http.SmokeError("client-tool unsupported-feature denial changed or began SSE")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "completed", "streamed"), default="all",
                        help="Run all checks, one completed call, or streamed call plus continuation.")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="Reasoning and answer token ceiling per call (default: 8192).")
    args = parser.parse_args(argv)
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if not http.BASE_URL or not http.USER_TOKEN:
        print("Set AUDREY_SMOKE_BASE_URL and AUDREY_USER_JWT in .env.test.local.", file=sys.stderr)
        return 2
    nonce = uuid.uuid4().hex
    result: dict[str, Any] = {"schema": 1, "case": args.case, "model": MODEL, "uploads_created": 0, "generation_calls": 0}
    try:
        if args.case == "all":
            initial = _payload(nonce, stream=True, max_output_tokens=args.max_output_tokens)
            virtual = _denial_guard({**initial, "model": "audrey_fast"})
            required = _denial_guard({**initial, "tool_choice": "required"})
            result["guards"] = {
                "virtual_model_http": virtual, "required_choice_http": required,
                "error": "responses_feature_unsupported", "before_sse": True,
            }
        if args.case in {"all", "completed"}:
            initial = _payload(nonce, stream=False, max_output_tokens=args.max_output_tokens)
            result["generation_calls"] += 1
            _, response = http._json("/v1/responses", token=http.USER_TOKEN, method="POST", payload=initial)
            result["completed"] = http._response_diagnostics(response)
            result["completed"].update(_validate_call_response(response, nonce))
        if args.case in {"all", "streamed"}:
            initial = _payload(nonce, stream=True, max_output_tokens=args.max_output_tokens)
            result["generation_calls"] += 1
            _, body, content_type = http._request(
                "/v1/responses", token=http.USER_TOKEN, method="POST", body=json.dumps(initial).encode(), content_type="application/json",
            )
            response, result["streamed"] = _validate_stream(body, content_type, nonce)
            call = next(item for item in response["output"] if item["type"] == "function_call")
            marker = _execute_local(call, nonce)
            result["client_functions_executed"] = 1
            followup = _followup_payload(initial, response, marker)
            result["generation_calls"] += 1
            _, response = http._json("/v1/responses", token=http.USER_TOKEN, method="POST", payload=followup)
            result["followup"] = http._response_diagnostics(response)
            result["followup"].update(_validate_followup(response, marker))
        result["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - structured operator evidence
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
