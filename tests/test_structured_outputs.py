"""Responses structured output contract and provider forwarding tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import audrey.routes.openai.routes as openai_routes
from audrey.models.ollama import OllamaClient
from audrey.routes.openai.routes import create_response
from audrey.routes.openai.schemas import ResponseCreateRequest
from audrey.routes.openai.streaming import ResponsesStreamSession, StreamOutcome
from audrey.routes.openai.structured_outputs import (
    StructuredOutputError,
    response_json_schema,
    validate_structured_output,
)


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }


def _payload(*, stream: bool = False) -> ResponseCreateRequest:
    return ResponseCreateRequest(
        model="audrey_fast",
        input="Return the sentinel answer.",
        stream=stream,
        text={
            "format": {
                "type": "json_schema",
                "name": "sentinel_result",
                "description": "A short answer and confidence score.",
                "schema": _schema(),
                "strict": True,
            }
        },
    )


def _chat_result(content: str) -> dict:
    return {
        "created": 1_800_000_000,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _request() -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace())


def _user() -> SimpleNamespace:
    return SimpleNamespace(email="alice@example.com", role="user")


def _events(*frames: str) -> list[dict]:
    result = []
    for block in "".join(frames).split("\n\n"):
        data = next(
            (line[6:] for line in block.splitlines() if line.startswith("data: ")),
            "",
        )
        if data:
            result.append(json.loads(data))
    return result


def test_named_schema_is_typed_admitted_and_dumped_with_schema_alias():
    payload = _payload()

    assert response_json_schema(payload) == _schema()
    assert payload.text is not None
    assert payload.text.model_dump(by_alias=True, exclude_none=True) == {
        "format": {
            "type": "json_schema",
            "name": "sentinel_result",
            "description": "A short answer and confidence score.",
            "schema": _schema(),
            "strict": True,
        }
    }


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object"},
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            "patternProperties": {},
        },
    ],
)
def test_unsupported_or_non_strict_schema_is_rejected_before_generation(schema):
    payload = ResponseCreateRequest(
        model="audrey_fast",
        input="Hello.",
        text={
            "format": {
                "type": "json_schema",
                "name": "bad_schema",
                "schema": schema,
                "strict": True,
            }
        },
    )

    with pytest.raises(StructuredOutputError):
        response_json_schema(payload)


def test_local_refs_and_any_of_validate_without_a_runtime_dependency():
    schema = {
        "type": "object",
        "$defs": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ]
            }
        },
        "properties": {"value": {"$ref": "#/$defs/value"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    payload = ResponseCreateRequest(
        model="audrey_fast",
        input="Hello.",
        text={
            "format": {
                "type": "json_schema",
                "name": "referenced",
                "schema": schema,
                "strict": True,
            }
        },
    )

    validate_structured_output('{"value":7}', payload)
    with pytest.raises(StructuredOutputError, match="matches no anyOf"):
        validate_structured_output('{"value":false}', payload)


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"ok","answer":"duplicate","confidence":1}',
        '{"answer":"ok","confidence":NaN}',
    ],
)
def test_nonstandard_or_ambiguous_json_is_rejected(text):
    with pytest.raises(StructuredOutputError):
        validate_structured_output(text, _payload())


@pytest.mark.asyncio
async def test_completed_route_forwards_schema_validates_and_echoes_format(monkeypatch):
    captured = {}

    async def generate(payload, request, me, *, response_format):
        captured["payload"] = payload
        captured["format"] = response_format
        return _chat_result('{"answer":"STRUCTURED_OK","confidence":1}')

    monkeypatch.setattr(openai_routes, "_create_chat_completion", generate)

    result = await create_response(_payload(), _request(), _user())

    assert captured["format"] == _schema()
    messages = [
        message.model_dump(exclude_none=True)
        for message in captured["payload"].messages
    ]
    assert messages[0]["role"] == "developer"
    assert "Return only one JSON object" in messages[0]["content"]
    assert result["status"] == "completed"
    assert json.loads(result["output_text"]) == {
        "answer": "STRUCTURED_OK",
        "confidence": 1,
    }
    assert result["text"]["format"]["type"] == "json_schema"
    assert result["text"]["format"]["schema"] == _schema()


@pytest.mark.asyncio
async def test_completed_route_returns_502_when_model_breaks_schema(monkeypatch):
    async def generate(*_args, **_kwargs):
        return _chat_result('{"answer":"missing confidence"}')

    monkeypatch.setattr(openai_routes, "_create_chat_completion", generate)

    with pytest.raises(HTTPException) as caught:
        await create_response(_payload(), _request(), _user())

    assert caught.value.status_code == 502
    assert "missing: confidence" in caught.value.detail


@pytest.mark.asyncio
async def test_bad_schema_returns_400_before_generation(monkeypatch):
    async def unexpected(*_args, **_kwargs):
        raise AssertionError("generation must not start")

    monkeypatch.setattr(openai_routes, "_create_chat_completion", unexpected)
    payload = ResponseCreateRequest(
        model="audrey_fast",
        input="Hello.",
        text={
            "format": {
                "type": "json_schema",
                "name": "bad",
                "schema": {"type": "array", "items": {"type": "string"}},
            }
        },
    )

    with pytest.raises(HTTPException) as caught:
        await create_response(payload, _request(), _user())

    assert caught.value.status_code == 400
    assert caught.value.detail["error"] == "responses_structured_output_invalid"


def test_structured_stream_hides_progress_and_completes_valid_json():
    session = ResponsesStreamSession(
        request=_payload(stream=True),
        virtual_model="audrey_fast",
        fingerprint_model="fast-model",
    )

    start = session.role_frame()
    session.stage_started("thinking", label="Thinking")
    assert session.status_frame("Thinking", stage="thinking") == ""
    session.stage_finished("thinking")
    delta = session.content_frame('{"answer":"STREAM_OK","confidence":1}')
    session.terminal.finish(StreamOutcome.OK, finish_reason="stop")
    terminal = session.terminal_frame()

    events = _events(start, delta, terminal)
    assert [event["type"] for event in events].count(
        "response.output_text.delta"
    ) == 1
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["text"]["format"]["type"] == "json_schema"


def test_structured_stream_finishes_failed_when_json_breaks_schema():
    session = ResponsesStreamSession(
        request=_payload(stream=True),
        virtual_model="audrey_fast",
        fingerprint_model="fast-model",
    )
    session.role_frame()
    session.content_frame('{"answer":"missing confidence"}')
    session.terminal.finish(StreamOutcome.OK, finish_reason="stop")

    events = _events(session.terminal_frame())

    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["code"] == "structured_output_invalid"


@pytest.mark.asyncio
async def test_ollama_stream_forwards_format_schema():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = (
            b'{"message":{"role":"assistant","content":"{}"},"done":false}\n'
            b'{"message":{"role":"assistant","content":""},"done":true}\n'
        )
        return httpx.Response(200, content=body)

    client = OllamaClient(
        "http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        chunks = [
            chunk
            async for chunk in client.chat_stream(
                model="test-model",
                messages=[{"role": "user", "content": "Return JSON."}],
                format=_schema(),
            )
        ]
    finally:
        await client.aclose()

    assert chunks[-1]["done"] is True
    assert seen["format"] == _schema()


@pytest.mark.parametrize("content", ["", '{"answer":'])
async def test_token_limited_json_is_incomplete_instead_of_schema_failure(monkeypatch, content):
    async def generate(*_args, **_kwargs):
        result = _chat_result(content)
        result["choices"][0]["finish_reason"] = "length"
        return result

    monkeypatch.setattr(openai_routes, "_create_chat_completion", generate)

    result = await create_response(_payload(), _request(), _user())

    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["output_text"] == content
    assert result["usage"]["output_tokens"] == 5
