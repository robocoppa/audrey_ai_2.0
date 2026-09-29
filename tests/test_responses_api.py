"""Contract tests for Audrey's first non-streaming Responses API slice."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import audrey.routes.openai.routes as openai_routes
from audrey.routes.openai.routes import create_response, router
from audrey.routes.openai.schemas import ResponseCreateRequest


def _request() -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace())


def _user() -> SimpleNamespace:
    return SimpleNamespace(email="alice@example.com", role="user")


def _chat_result(*, content: str = "A concise answer.") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_800_000_000,
        "model": "audrey_fast",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
        },
    }


@pytest.mark.asyncio
async def test_string_input_uses_shared_generation_and_returns_typed_output(monkeypatch):
    captured = {}

    async def generate(payload, request, me):
        captured["payload"] = payload
        captured["request"] = request
        captured["me"] = me
        return _chat_result()

    monkeypatch.setattr(openai_routes, "chat_completions", generate)
    payload = ResponseCreateRequest(
        model="audrey_fast",
        input="Explain the result.",
        instructions="Answer briefly.",
        max_output_tokens=77,
        temperature=0.2,
        top_p=0.8,
        metadata={"trace": "case-1"},
        user="ignored-client-user",
        skill="grounded-document-analysis",
    )

    result = await create_response(payload, _request(), _user())

    forwarded = captured["payload"]
    assert forwarded.model == "audrey_fast"
    assert forwarded.stream is False
    assert forwarded.max_tokens == 77
    assert forwarded.temperature == 0.2
    assert forwarded.top_p == 0.8
    assert forwarded.metadata == {"trace": "case-1"}
    assert forwarded.user == "ignored-client-user"
    assert forwarded.skill == "grounded-document-analysis"
    assert [message.model_dump() for message in forwarded.messages] == [
        {"role": "developer", "content": "Answer briefly.", "name": None, "metadata": None},
        {"role": "user", "content": "Explain the result.", "name": None, "metadata": None},
    ]

    assert result["id"].startswith("resp_")
    assert result["object"] == "response"
    assert result["status"] == "completed"
    assert result["model"] == "audrey_fast"
    assert result["instructions"] == "Answer briefly."
    assert result["output_text"] == "A concise answer."
    assert result["output"] == [
        {
            "id": result["output"][0]["id"],
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "A concise answer.",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    ]
    assert result["output"][0]["id"].startswith("msg_")
    assert result["usage"] == {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 16,
    }


@pytest.mark.asyncio
async def test_text_message_history_keeps_roles_and_order(monkeypatch):
    captured = {}

    async def generate(payload, _request, _me):
        captured["messages"] = [
            message.model_dump(exclude_none=True) for message in payload.messages
        ]
        return _chat_result(content="Paris.")

    monkeypatch.setattr(openai_routes, "chat_completions", generate)
    payload = ResponseCreateRequest(
        model="audrey_auto",
        input=[
            {"role": "system", "content": "Use one word."},
            {"role": "user", "content": "Capital of France?"},
        ],
    )

    result = await create_response(payload, _request(), _user())

    assert captured["messages"] == [
        {"role": "system", "content": "Use one word."},
        {"role": "user", "content": "Capital of France?"},
    ]
    assert result["output_text"] == "Paris."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream", True),
        ("background", True),
        ("store", False),
        ("previous_response_id", "resp_previous"),
        ("conversation", "conv_123"),
        ("tools", []),
        ("text", {"format": {"type": "json_object"}}),
    ],
)
async def test_unsupported_features_fail_before_generation(monkeypatch, field, value):
    async def unexpected(*_args, **_kwargs):
        raise AssertionError("generation must not start")

    monkeypatch.setattr(openai_routes, "chat_completions", unexpected)
    payload = ResponseCreateRequest(
        model="audrey_fast",
        input="Hello.",
        **{field: value},
    )

    with pytest.raises(HTTPException) as caught:
        await create_response(payload, _request(), _user())

    assert caught.value.status_code == 400
    assert caught.value.detail["error"] == "responses_feature_unsupported"
    assert field in caught.value.detail["message"]


@pytest.mark.parametrize(
    "input_value",
    ["", "   ", []],
)
def test_empty_input_is_rejected(input_value):
    with pytest.raises(ValidationError, match="input must contain"):
        ResponseCreateRequest(model="audrey_fast", input=input_value)


def test_multimodal_and_unknown_fields_are_not_silently_dropped():
    with pytest.raises(ValidationError):
        ResponseCreateRequest(
            model="audrey_fast",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello."}],
                }
            ],
        )
    with pytest.raises(ValidationError):
        ResponseCreateRequest(
            model="audrey_fast",
            input="Hello.",
            include=["message.output_text.logprobs"],
        )


def test_responses_route_is_registered():
    route = next(item for item in router.routes if item.path == "/v1/responses")
    assert "POST" in route.methods
