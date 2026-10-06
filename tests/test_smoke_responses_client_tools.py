"""The caller-tool smoke must verify calls, SSE identities, and result continuation."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "smoke"))
import smoke_responses_client_tools as smoke

NONCE = "nonce-test"


def _call_response(nonce=NONCE, *, suffix="test"):
    return {
        "id": "resp_" + suffix, "status": "completed", "output_text": "",
        "output": [{
            "type": "function_call", "id": "fc_" + suffix, "call_id": "call_" + suffix,
            "name": smoke.FUNCTION_NAME, "arguments": json.dumps({"nonce": nonce}), "status": "completed",
        }],
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
    }


def _answer_response(text):
    response = _call_response()
    response.update(output_text=text, output=[{
        "type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text}],
    }])
    return response


def _events(response):
    call = response["output"][0]
    args = call["arguments"]
    return [
        {"type": "response.created", "response": {"id": response["id"], "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**call, "status": "in_progress", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": call["id"], "delta": args[:8]},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": call["id"], "delta": args[8:]},
        {"type": "response.function_call_arguments.done", "output_index": 0, "item_id": call["id"], "arguments": args},
        {"type": "response.output_item.done", "output_index": 0, "item": call},
        {"type": "response.completed", "response": response},
    ]


def _sse(events):
    for index, event in enumerate(events):
        event.setdefault("sequence_number", index)
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def test_function_payload_has_strict_flat_definition_and_thinking_budget():
    payload = smoke._payload(NONCE, stream=True, max_output_tokens=8192)
    assert payload["tool_choice"] == "auto"
    assert payload["max_output_tokens"] == 8192
    assert payload["stream"] is True
    assert payload["tools"] == [{
        "type": "function", "name": "get_smoke_marker",
        "description": "Read the smoke marker for the supplied nonce from the caller.",
        "strict": True, "parameters": {
            "type": "object", "properties": {"nonce": {"type": "string", "enum": [NONCE]}},
            "required": ["nonce"], "additionalProperties": False,
        },
    }]
    assert "function" not in payload["tools"][0]
    assert NONCE in payload["input"][0]["content"][0]["text"]


@pytest.mark.parametrize("failure", [
    "empty_output", "no_call", "duplicate_calls", "item_id", "call_id", "name", "raw_args",
    "invalid_json", "wrong_nonce", "extra_arg", "item_status", "response_status",
    "usage_bool", "usage_sum", "server_output", "untyped_message", "progress",
])
def test_call_validation_rejects_missing_or_invalid_contract(failure):
    response = _call_response()
    call = response["output"][0]
    if failure == "empty_output":
        response["output"] = []
    elif failure == "no_call":
        response = _answer_response("guessed marker")
    elif failure == "duplicate_calls":
        response["output"].append({**call, "id": "fc_other", "call_id": "call_other"})
    elif failure == "item_id":
        call["id"] = "wrong"
    elif failure == "call_id":
        call["call_id"] = "wrong"
    elif failure == "name":
        call["name"] = "web_fetch"
    elif failure == "raw_args":
        call["arguments"] = {"nonce": NONCE}
    elif failure == "invalid_json":
        call["arguments"] = "{broken"
    elif failure == "wrong_nonce":
        call["arguments"] = json.dumps({"nonce": "other"})
    elif failure == "extra_arg":
        call["arguments"] = json.dumps({"nonce": NONCE, "extra": True})
    elif failure == "item_status":
        call["status"] = "in_progress"
    elif failure == "response_status":
        response["status"] = "incomplete"
    elif failure == "usage_bool":
        response["usage"]["output_tokens"] = True
    elif failure == "usage_sum":
        response["usage"]["total_tokens"] += 1
    elif failure == "server_output":
        response["output"].append({"type": "web_search_call", "id": "ws_test", "status": "completed"})
    elif failure == "untyped_message":
        response["output"].append({"id": "msg_test", "type": "message", "status": "completed", "content": "untyped"})
    elif failure == "progress":
        message = _answer_response("> _Thinking_ model")["output"][0]
        response["output"].append(message)
        response["output_text"] = message["content"][0]["text"]
    with pytest.raises(smoke.http.SmokeError):
        smoke._validate_call_response(response, NONCE)


@pytest.mark.parametrize("failure", [
    None, "terminal", "sequence", "label", "added_missing", "done_missing",
    "done_changed", "item_id", "arguments", "arguments_done", "identity", "server_event",
])
def test_stream_matches_arguments_item_ids_and_terminal_output(failure):
    events = _events(_call_response())
    if failure == "terminal":
        events[-1]["type"] = "response.incomplete"
        events[-1]["response"].update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    elif failure == "sequence":
        events[2]["sequence_number"] = 9
    elif failure == "label":
        data = _sse(events).replace(b"event: response.created", b"event: other.created", 1)
    elif failure == "added_missing":
        del events[1]
    elif failure == "done_missing":
        del events[-2]
    elif failure == "done_changed":
        events[-2]["item"] = {**events[-2]["item"], "arguments": "{}"}
    elif failure == "item_id":
        events[2]["item_id"] = "fc_wrong"
    elif failure == "arguments":
        events[2]["delta"] = "{}"
    elif failure == "arguments_done":
        events[4]["arguments"] = "{}"
    elif failure == "identity":
        events[1]["item"]["call_id"] = "call_wrong"
    elif failure == "server_event":
        events.insert(2, {"type": "response.web_search_call.completed"})
    if failure != "label":
        data = _sse(events)
    if failure:
        with pytest.raises(smoke.http.SmokeError):
            smoke._validate_stream(data, "text/event-stream", NONCE)
    else:
        response, result = smoke._validate_stream(data, "text/event-stream", NONCE)
        assert response["output"][0]["call_id"] == "call_test"
        assert result["terminal"] == "response.completed"
        assert result["argument_delta_count"] == 2
        assert result["argument_deltas_match"] is True
        assert result["native_tool_events"] == 0


def test_incomplete_stream_failure_retains_usage_and_stop_reason():
    events = _events(_call_response())
    events[-1].update(type="response.incomplete")
    events[-1]["response"].update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    with pytest.raises(smoke.http.SmokeError) as exc:
        smoke._validate_stream(_sse(events), "text/event-stream", NONCE)
    assert "max_output_tokens" in str(exc.value)
    assert "'output_tokens': 10" in str(exc.value)


def test_followup_preserves_typed_assistant_history_and_matches_call_result():
    response = _call_response(suffix="streamed")
    message = _answer_response("I will request the marker.")["output"][0]
    response["output"].insert(0, message)
    initial = smoke._payload(NONCE, stream=True, max_output_tokens=8192)
    original = copy.deepcopy(initial)
    marker = smoke._execute_local(response["output"][1], NONCE)
    followup = smoke._followup_payload(initial, response, marker)
    assert followup["input"][:-1] == initial["input"] + response["output"]
    assert followup["input"][-1] == {
        "type": "function_call_output", "call_id": "call_streamed", "output": json.dumps({"marker": marker}),
    }
    assert followup["tool_choice"] == "none"
    assert followup["stream"] is False
    assert followup["tools"] == initial["tools"]
    assert "previous_response_id" not in followup
    assert initial == original
    assert smoke._validate_followup(_answer_response(marker), marker)["sentinel"] is True


@pytest.mark.parametrize("wrong", ["", "invented", "CLIENT_TOOL_SIMILAR", "> _Thinking_ done"])
def test_followup_requires_the_actual_caller_result(wrong):
    with pytest.raises(smoke.http.SmokeError):
        smoke._validate_followup(_answer_response(wrong), "CLIENT_TOOL_EXPECTED")


@pytest.mark.parametrize("bad_guard", [None, "virtual", "required", "sse"])
def test_denial_guards_finish_before_generation(monkeypatch, capsys, bad_guard):
    monkeypatch.setattr(smoke.http, "BASE_URL", "http://example.test")
    monkeypatch.setattr(smoke.http, "USER_TOKEN", "user-test")
    payloads = []

    def request(path, **kw):
        assert path == "/v1/responses"
        payload = json.loads(kw["body"])
        payloads.append(payload)
        assert kw["expected"] == frozenset({400})
        assert payload["stream"] is True
        invalid = (bad_guard == "virtual" and len(payloads) == 1) or (bad_guard == "required" and len(payloads) == 2)
        return 400, json.dumps({"detail": {"error": "wrong" if invalid else "responses_feature_unsupported"}}).encode(), "text/event-stream" if bad_guard == "sse" else "application/json"

    monkeypatch.setattr(smoke.http, "_request", request)
    monkeypatch.setattr(smoke.http, "_json", lambda *_args, **_kw: (_ for _ in ()).throw(smoke.http.SmokeError("stop after guards")))
    assert smoke.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert len(payloads) == (1 if bad_guard in {"virtual", "sse"} else 2)
    assert payloads[0]["model"] == "audrey_fast"
    if len(payloads) == 2:
        assert payloads[1]["tool_choice"] == "required"
        assert payloads[1]["model"] == smoke.MODEL
    assert "completed" not in result
    assert result["generation_calls"] == (1 if bad_guard is None else 0)
    assert result["uploads_created"] == 0


@pytest.mark.parametrize("case,expected_calls", [("all", 3), ("completed", 1), ("streamed", 2)])
def test_targeted_smoke_has_no_uploads_and_executes_only_returned_local_call(monkeypatch, capsys, case, expected_calls):
    monkeypatch.setattr(smoke.http, "BASE_URL", "http://example.test")
    monkeypatch.setattr(smoke.http, "USER_TOKEN", "user-test")
    monkeypatch.setattr(smoke.http, "ADMIN_TOKEN", "")
    monkeypatch.setattr(smoke.uuid, "uuid4", lambda: type("UUID", (), {"hex": NONCE})())
    generated = []
    guards = []
    executed = []
    execute = smoke._execute_local

    def local(call, nonce):
        executed.append(call["call_id"])
        return execute(call, nonce)

    def request(path, **kw):
        assert path == "/v1/responses"
        assert kw["token"] == smoke.http.USER_TOKEN
        payload = json.loads(kw["body"])
        if kw.get("expected") == frozenset({400}):
            guards.append(payload)
            return 400, b'{"detail":{"error":"responses_feature_unsupported"}}', "application/json"
        generated.append(payload)
        assert payload["stream"] is True
        return 200, _sse(_events(_call_response(suffix="streamed"))), "text/event-stream"

    def request_json(path, **kw):
        assert path == "/v1/responses"
        assert kw["token"] == smoke.http.USER_TOKEN
        payload = kw["payload"]
        generated.append(payload)
        if payload["tool_choice"] == "none":
            assert payload["input"][1]["call_id"] == "call_streamed"
            assert payload["input"][2]["call_id"] == "call_streamed"
            marker = json.loads(payload["input"][2]["output"])["marker"]
            return 200, _answer_response(marker)
        return 200, _call_response(suffix="completed")

    monkeypatch.setattr(smoke, "_execute_local", local)
    monkeypatch.setattr(smoke.http, "_request", request)
    monkeypatch.setattr(smoke.http, "_json", request_json)
    assert smoke.main(["--case", case, "--max-output-tokens", "8192"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["generation_calls"] == expected_calls == len(generated)
    assert result["uploads_created"] == 0
    assert len(guards) == (2 if case == "all" else 0)
    assert all(payload["max_output_tokens"] == 8192 for payload in generated)
    if case != "completed":
        assert executed == ["call_streamed"]
        assert result["client_functions_executed"] == 1
        assert result["followup"]["sentinel"] is True
        assert result["streamed"]["argument_deltas_match"] is True
    else:
        assert executed == []
        assert "followup" not in result


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_nonpositive_budget_stops_before_network(monkeypatch, budget):
    monkeypatch.setattr(smoke.http, "_request", lambda *_a, **_kw: pytest.fail("network called"))
    monkeypatch.setattr(smoke.http, "_json", lambda *_a, **_kw: pytest.fail("network called"))
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--max-output-tokens", budget])
    assert exc.value.code == 2
