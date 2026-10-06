"""Client-owned Responses tool definitions, histories, and provider calls."""

from __future__ import annotations

import copy
import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from audrey.models.ollama import _to_ollama_messages
from audrey.routes.openai.client_tools import (
    MAX_CALL_ARGUMENT_BYTES,
    MAX_GENERATED_CALLS,
    MAX_REPLAY_ITEMS,
    MAX_TOOL_BYTES,
    adapt_tool_history_item,
    has_client_tools,
    merge_adjacent_tool_calls,
    provider_tools,
    validate_client_tool_request,
    validate_generated_calls,
)
from audrey.routes.openai.schemas import ResponseCreateRequest


def _tool(**changes):
    value = {
        "type": "function",
        "name": "lookup_weather",
        "description": "Read the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "minLength": 1}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    return {**value, **changes}


def _request(**changes):
    return ResponseCreateRequest(
        **{
            "model": "audrey_passthrough/test-model",
            "input": "Weather in Boston?",
            "tools": [_tool()],
            **changes,
        }
    )


def _call(call_id="call_one", *, name="lookup_weather", arguments='{"city":"Boston"}', **changes):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        **changes,
    }


def _output(call_id="call_one", output="Sunny", **changes):
    return {"type": "function_call_output", "call_id": call_id, "output": output, **changes}


def _generated(*, name="lookup_weather", arguments=None, **changes):
    return {
        "function": {"name": name, "arguments": {"city": "Boston"} if arguments is None else arguments},
        **changes,
    }


def _fails(function, *args, status=400):
    with pytest.raises(HTTPException) as caught:
        function(*args)
    assert caught.value.status_code == status
    return caught.value.detail


def test_plain_request_has_no_client_tool_surface():
    assert not has_client_tools(_request(tools=None))


@pytest.mark.parametrize("changes", [
    {"tools": []},
    {"tools": None, "tool_choice": "none"},
    {"tools": None, "parallel_tool_calls": False},
    {"tools": None, "input": [_call(), _output()]},
    {"tools": None, "input": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Old answer"}]}]},
])
def test_explicit_controls_and_replay_use_client_tool_admission(changes):
    assert has_client_tools(_request(**changes))


def test_flat_definitions_become_nested_provider_tools_without_strict_extension():
    payload = _request()
    before = payload.model_dump()
    validate_client_tool_request(payload)
    result = provider_tools(payload)
    assert result == [{"type": "function", "function": {
        "name": "lookup_weather", "description": "Read the weather for a city.",
        "parameters": _tool()["parameters"],
    }}]
    assert payload.model_dump() == before
    assert provider_tools(_request(tool_choice="none")) is None
    assert provider_tools(_request(tools=[])) is None


def test_function_without_parameters_is_an_empty_object_contract():
    payload = _request(tools=[{"type": "function", "name": "ping"}])
    validate_client_tool_request(payload)
    assert provider_tools(payload)[0]["function"]["parameters"] == {
        "type": "object", "properties": {}, "additionalProperties": False,
    }
    assert validate_generated_calls([_generated(name="ping", arguments={})], payload)[0]["arguments"] == "{}"
    _fails(validate_generated_calls, [_generated(name="ping", arguments={"unexpected": 1})], payload, status=502)


@pytest.mark.parametrize("tool", [
    _tool(type="web_search"),
    {"type": "function", "function": {"name": "nested"}},
    _tool(name=""),
    _tool(name="name.with.dots"),
    _tool(name="x" * 65),
    _tool(description=1),
    _tool(description="x" * 1025),
    _tool(strict="true"),
    _tool(parameters=None),
    _tool(parameters={"type": "array"}),
    _tool(parameters={"type": "object", "patternProperties": {}}),
    _tool(parameters={"type": "object", "$ref": "https://example.com/schema"}),
    _tool(parameters={"type": "object", "properties": {}, "additionalProperties": True}),
    _tool(extra="not silently dropped"),
])
def test_invalid_or_unsupported_definition_is_rejected(tool):
    _fails(validate_client_tool_request, _request(tools=[tool]))


def test_nonstrict_schema_is_admitted_but_arguments_are_still_checked():
    payload = _request(tools=[_tool(strict=False, parameters={
        "type": "object", "properties": {"city": {"type": "string"}},
    })])
    validate_client_tool_request(payload)
    assert validate_generated_calls([_generated(arguments={})], payload)
    _fails(validate_generated_calls, [_generated(arguments={"city": 12})], payload, status=502)


def test_definition_count_duplicates_and_aggregate_bytes_are_bounded():
    _fails(validate_client_tool_request, _request(tools=[_tool(name=f"function_{i}") for i in range(17)]))
    _fails(validate_client_tool_request, _request(tools=[_tool(), _tool()]))
    payload = _request(tools=[_tool(parameters={
        "type": "object", "description": "x" * MAX_TOOL_BYTES,
    })])
    _fails(validate_client_tool_request, payload, status=413)


@pytest.mark.parametrize("choice", ["required", {"type": "function", "name": "lookup_weather"}, "unknown"])
def test_forced_choices_are_explicitly_unsupported(choice):
    detail = _fails(validate_client_tool_request, _request(tool_choice=choice))
    assert detail["error"] == "responses_feature_unsupported"


@pytest.mark.parametrize("choice", [None, "auto", "none"])
def test_supported_choice_admitted(choice):
    validate_client_tool_request(_request(tool_choice=choice))


def test_replay_parallel_groups_preserve_call_links_in_real_ollama_adapter():
    payload = _request(input=[
        {"role": "user", "content": "Weather?"},
        _call("call_one"), _call("call_two", arguments='{"city":"Denver"}'),
        _output("call_two", "Snow"), _output("call_one", "Sun"),
        {"type": "message", "id": "msg_old", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": "One sun, one snow", "annotations": [], "logprobs": []},
        ]},
        {"role": "user", "content": "Compare them"},
    ])
    validate_client_tool_request(payload)
    messages = [
        {"role": item.role, "content": item.content} if getattr(item, "type", None) is None else adapt_tool_history_item(item)
        for item in payload.input
    ]
    before = copy.deepcopy(messages)
    merged = merge_adjacent_tool_calls(messages)
    assert messages == before
    assert len(merged[1]["tool_calls"]) == 2
    native = _to_ollama_messages(merged)
    assert native[1]["tool_calls"][1]["function"]["arguments"] == {"city": "Denver"}
    assert native[2] == {"role": "tool", "content": "Snow", "tool_name": "lookup_weather"}
    assert native[3] == {"role": "tool", "content": "Sun", "tool_name": "lookup_weather"}
    assert native[4]["content"] == "One sun, one snow"


def test_complete_history_does_not_require_readvertising_old_tools():
    payload = _request(tools=None, input=[_call(name="old_function"), _output()])
    validate_client_tool_request(payload)
    assert provider_tools(payload) is None


@pytest.mark.parametrize("history", [
    [_output()],
    [_call()],
    [_call(), _output("call_other")],
    [_call(), _output(), _output()],
    [_call(), _output(), _call(), _output()],
    [_call(), {"role": "user", "content": "Do something else"}, _output()],
    [_call(), {"role": "developer", "content": "Change instruction"}, _output()],
    [_call("one"), _call("two"), _output("one"), _call("three"), _output("two"), _output("three")],
    [_call(id="fc_same"), _output(id="fc_same")],
    [_call(arguments='{"city": 1}'), _output()],
])
def test_malformed_history_fails_before_generation(history):
    _fails(validate_client_tool_request, _request(input=history))


@pytest.mark.parametrize("arguments", ['[]', 'null', '{broken', '{"city":"A","city":"B"}', '{"city":NaN}'])
def test_history_requires_standard_object_json(arguments):
    _fails(validate_client_tool_request, _request(input=[_call(arguments=arguments), _output()]))


def test_pending_call_group_is_bounded():
    payload = _request(input=[
        *[_call(f"call_{i}") for i in range(9)],
        *[_output(f"call_{i}") for i in range(9)],
    ])
    _fails(validate_client_tool_request, payload)


def test_argument_limit_counts_utf8_bytes_not_characters():
    args = json.dumps({"city": "é" * (MAX_CALL_ARGUMENT_BYTES // 2)}, ensure_ascii=False)
    payload = _request(input=[_call(arguments=args), _output()])
    _fails(validate_client_tool_request, payload, status=413)


@pytest.mark.parametrize("item", [
    _call(status="in_progress"),
    _output(output=[{"type": "input_text", "text": "Not supported yet"}]),
    _output(name="forged_tool_name"),
    {"type": "message", "role": "user", "content": [{"type": "output_text", "text": "Old"}]},
    {"type": "message", "role": "assistant", "status": "incomplete", "content": [{"type": "output_text", "text": "Old"}]},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Old", "annotations": [{"type": "url_citation"}]}]},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Old", "logprobs": [{}]}]},
])
def test_replay_rejects_partial_items_and_unhandled_semantics(item):
    with pytest.raises(ValidationError):
        _request(input=[item])


def test_generated_calls_are_typed_valid_and_replayable():
    payload = _request()
    result = validate_generated_calls([_generated()], payload)
    assert result[0]["id"].startswith("fc_")
    assert result[0]["call_id"].startswith("call_")
    assert result[0]["type"] == "function_call"
    assert result[0]["status"] == "completed"
    assert json.loads(result[0]["arguments"]) == {"city": "Boston"}
    validate_client_tool_request(_request(input=[result[0], _output(result[0]["call_id"])]))


@pytest.mark.parametrize("raw", [
    "not an array",
    [None],
    [{"type": "custom", "function": {"name": "lookup_weather", "arguments": {"city": "Boston"}}}],
    [{"function": {"name": "unadvertised", "arguments": {}}}],
    [{"function": {"name": [], "arguments": {}}}],
    [{"function": {"name": {}, "arguments": {}}}],
    [{"function": {"name": "lookup_weather"}}],
    [_generated(arguments='{"city":"A","city":"B"}')],
    [_generated(arguments={"city": 2})],
    [_generated(arguments={"city": ""})],
    [_generated(arguments={"city": "Boston", "unexpected": True})],
    [_generated(arguments='{"city":NaN}')],
    [_generated(id="bad id")],
    [_generated(id="same"), _generated(id="same")],
    [_generated() for _ in range(MAX_GENERATED_CALLS + 1)],
])
def test_provider_failure_exposes_no_executable_calls(raw):
    _fails(validate_generated_calls, raw, _request(), status=502)


def test_generated_calls_cannot_reuse_history_ids():
    payload = _request(input=[_call(), _output()])
    _fails(validate_generated_calls, [_generated(id="call_one")], payload, status=502)


def test_tool_choice_none_and_parallel_false_are_enforced_before_exposure():
    _fails(validate_generated_calls, [_generated()], _request(tool_choice="none"), status=502)
    _fails(validate_generated_calls, [_generated(), _generated()], _request(parallel_tool_calls=False), status=502)
    assert validate_generated_calls([_generated()], _request(parallel_tool_calls=False))
    assert validate_generated_calls(None, _request()) == []
    assert validate_generated_calls([], _request(tools=[])) == []


def test_provider_argument_bytes_are_bounded():
    _fails(validate_generated_calls, [_generated(arguments={"city": "é" * MAX_CALL_ARGUMENT_BYTES})], _request(), status=502)


@pytest.mark.parametrize("arguments", ['{"city":"\ud800"}', '{"city":"\udfff"}', r'{"city":"\ud800"}'])
def test_provider_unicode_failure_is_reported_as_502(arguments):
    _fails(validate_generated_calls, [_generated(arguments=arguments)], _request(), status=502)


def test_definition_unicode_failure_is_a_request_error():
    _fails(validate_client_tool_request, _request(tools=[_tool(description="\ud800")]))



def test_mixed_assistant_text_and_parallel_calls_replay_as_one_provider_turn():
    payload = _request(input=[
        {"role": "user", "content": "Compare city weather"},
        {
            "type": "message", "id": "msg_previous", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "I will check both cities."}],
        },
        _call("call_one", id="fc_one"),
        _call("call_two", id="fc_two", arguments='{"city":"Denver"}'),
        _output("call_one", "Sun"),
        _output("call_two", "Snow"),
    ])
    validate_client_tool_request(payload)
    messages = [
        {"role": item.role, "content": item.content} if getattr(item, "type", None) is None else adapt_tool_history_item(item)
        for item in payload.input
    ]
    before = copy.deepcopy(messages)
    merged = merge_adjacent_tool_calls(messages)
    assert messages == before
    assert len(merged) == 4
    assert merged[1]["content"] == "I will check both cities."
    assert len(merged[1]["tool_calls"]) == 2
    native = _to_ollama_messages(merged)
    assert native[1]["role"] == "assistant"
    assert native[1]["content"] == "I will check both cities."
    assert [call["function"]["arguments"] for call in native[1]["tool_calls"]] == [
        {"city": "Boston"}, {"city": "Denver"},
    ]
    assert native[2] == {"role": "tool", "content": "Sun", "tool_name": "lookup_weather"}
    assert native[3] == {"role": "tool", "content": "Snow", "tool_name": "lookup_weather"}


def test_client_history_count_is_bounded_without_changing_plain_schema_history():
    history = [{"role": "user", "content": "Brief question"} for _ in range(MAX_REPLAY_ITEMS)]
    validate_client_tool_request(_request(input=history))
    too_many = _request(input=[*history, {"role": "user", "content": "Another question"}])
    _fails(validate_client_tool_request, too_many)
    assert len(_request(input=too_many.input, tools=None).input) == MAX_REPLAY_ITEMS + 1


def test_function_result_text_has_a_schema_limit():
    payload = _request(input=[_call(), _output(output="x" * 100_000)])
    validate_client_tool_request(payload)
    with pytest.raises(ValidationError):
        _request(input=[_call(), _output(output="x" * 100_001)])
