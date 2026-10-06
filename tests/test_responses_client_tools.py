"""Responses client tools through real Ollama payload and stream adaptation."""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from test_responses_file_inputs import library as library

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai import client_tool_generation as generation
from audrey.routes.openai import file_inputs, routes
from audrey.routes.openai.schemas import ResponseCreateRequest

MODEL = "tool-model"
TOOLS = [{
    "type": "function", "name": "get_marker", "strict": True,
    "parameters": {"type": "object", "properties": {"nonce": {"type": "string"}},
                   "required": ["nonce"], "additionalProperties": False},
}]
USER = SimpleNamespace(email="alice@example.com", role="user")


def call(nonce="one", *, name="get_marker"):
    return {"function": {"name": name, "arguments": {"nonce": nonce}}}


def reply(*calls, text="", reason="stop", **kwargs):
    return {"message": {"role": "assistant", "content": text, "tool_calls": list(calls)},
            "done": True, "done_reason": reason, "prompt_eval_count": 17, "eval_count": 11, **kwargs}


def payload(*, stream=False, **kwargs):
    return ResponseCreateRequest(model=f"audrey_passthrough/{MODEL}", input="Get the marker.",
                                 tools=copy.deepcopy(TOOLS), stream=stream, **kwargs)


@pytest.fixture
def provider():
    clients = []

    def build(*, raw=None, chunks=None, caps=None, show_status=200):
        captured = []
        paths = []
        raw = reply(call()) if raw is None else raw
        chunks = [reply(call())] if chunks is None else chunks

        def handler(request):
            paths.append(request.url.path)
            if request.url.path == "/api/show":
                return httpx.Response(show_status, json={"capabilities": ["completion", "tools"] if caps is None else caps})
            body = json.loads(request.content)
            captured.append(body)
            if body.get("stream"):
                return httpx.Response(200, content="".join(json.dumps(chunk) + "\n" for chunk in chunks).encode())
            return httpx.Response(200, json=raw)

        ollama = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))
        clients.append(ollama)
        cfg = SimpleNamespace(
            raw={"passthrough": {"enabled": True, "require_role": None, "allowed_models": [MODEL]},
                 "vision": {"describe_for_text_models": False}},
            timeouts={"medium": 3}, model_registry={},
        )
        registry = ModelRegistry(cfg)
        registry.location_of = lambda _name: "local"
        state = SimpleNamespace(cfg=cfg, ollama=ollama, registry=registry,
                                gate=FairLocalGate(concurrency=1),
                                inflight=UserInflightRegistry(max_inflight_per_user=1), health=HealthTracker())
        return SimpleNamespace(request=SimpleNamespace(app=SimpleNamespace(state=state)),
                               state=state, captured=captured, paths=paths)

    yield build
    # MockTransport has no socket resources. Tests exercise the real client
    # conversion and generators, without contacting a running provider.


async def events(response):
    assert isinstance(response, StreamingResponse)
    text = "".join([frame.decode() if isinstance(frame, bytes) else frame async for frame in response.body_iterator])
    result = []
    for block in text.split("\n\n"):
        if not block:
            continue
        name, data = block.splitlines()
        item = json.loads(data.removeprefix("data: "))
        assert name == "event: " + item["type"]
        result.append(item)
    assert [item["sequence_number"] for item in result] == list(range(len(result)))
    return result


@pytest.mark.parametrize("stream", [False, True])
async def test_valid_calls_preserve_payload_identity_limits_and_usage(provider, stream):
    fixture = provider()
    result = await routes.create_response(payload(stream=stream, user="spoofed", max_output_tokens=90,
                                                  temperature=0.2, top_p=0.8), fixture.request, USER)
    if stream:
        frames = await events(result)
        assert frames[0]["type"] == "response.created"
        assert frames[-1]["type"] == "response.completed"
        result = frames[-1]["response"]
        assert not any(frame["type"].startswith("response.output_text") for frame in frames)
        delta = next(frame for frame in frames if frame["type"] == "response.function_call_arguments.delta")
        done = next(frame for frame in frames if frame["type"] == "response.function_call_arguments.done")
        assert delta["item_id"] == done["item_id"] == result["output"][0]["id"]
        assert delta["delta"] == done["arguments"] == result["output"][0]["arguments"]
    assert result["status"] == "completed" and result["output_text"] == ""
    assert len(result["output"]) == 1
    item = result["output"][0]
    assert item["type"] == "function_call" and item["id"].startswith("fc_") and item["call_id"].startswith("call_")
    assert json.loads(item["arguments"]) == {"nonce": "one"}
    assert result["usage"]["input_tokens"] == 17 and result["usage"]["output_tokens"] == 11
    assert result["tools"] == TOOLS and result["parallel_tool_calls"] is True
    native = fixture.captured[0]
    assert native["model"] == MODEL
    assert native["options"] == {"num_predict": 90, "temperature": 0.2, "top_p": 0.8}
    assert native["tools"][0]["function"]["name"] == "get_marker"
    assert "strict" not in native["tools"][0]["function"] and "user" not in native


async def test_stream_collects_calls_from_early_chunks_and_keeps_text_order(provider):
    fixture = provider(chunks=[{"message": {"content": "I will check.", "tool_calls": [call("one")]}, "done": False},
                               reply(call("two"))])
    frames = await events(await routes.create_response(payload(stream=True), fixture.request, USER))
    result = frames[-1]["response"]
    assert [item["type"] for item in result["output"]] == ["message", "function_call", "function_call"]
    assert result["output_text"] == "I will check."
    assert [json.loads(item["arguments"])["nonce"] for item in result["output"][1:]] == ["one", "two"]
    items = [event for event in frames if event["type"] == "response.output_item.done"]
    assert [event["output_index"] for event in items] == [0, 1, 2]
    assert [event["item"] for event in items] == result["output"]


async def test_stateless_mixed_parallel_replay_reaches_real_ollama_format(provider):
    first = provider(raw=reply(call("one"), call("two"), text="I will check."))
    response = await routes.create_response(payload(), first.request, USER)
    history = [{"role": "user", "content": "Get the marker."}, *response["output"]]
    for index, item in enumerate(response["output"][1:]):
        history.append({"type": "function_call_output", "call_id": item["call_id"], "output": f"RESULT-{index}"})
    second = provider(raw=reply(text="RESULT-0 and RESULT-1"))
    continuation = payload()
    continuation.input = ResponseCreateRequest(model=continuation.model, input=history).input
    continuation.tool_choice = "none"
    result = await routes.create_response(continuation, second.request, USER)
    assert result["output_text"] == "RESULT-0 and RESULT-1"
    native = second.captured[0]
    assert "tools" not in native
    assert [message["role"] for message in native["messages"]] == ["user", "assistant", "tool", "tool"]
    assistant = native["messages"][1]
    assert assistant["content"] == "I will check." and len(assistant["tool_calls"]) == 2
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"nonce": "one"}
    assert all(message["tool_name"] == "get_marker" for message in native["messages"][2:])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["unknown", "bad_args", "mixed", "none", "parallel"])
async def test_invalid_call_batch_never_exposes_an_executable_item(provider, stream, mode):
    calls = [call()]
    kwargs = {}
    if mode == "unknown":
        calls = [call(name="not_advertised")]
    elif mode == "bad_args":
        calls[0]["function"]["arguments"] = {"nonce": 12}
    elif mode == "mixed":
        calls.append(call(name="not_advertised"))
    elif mode == "none":
        kwargs["tool_choice"] = "none"
    elif mode == "parallel":
        calls.append(call("two"))
        kwargs["parallel_tool_calls"] = False
    fixture = provider(raw=reply(*calls), chunks=[reply(*calls)])
    request = payload(stream=stream, **kwargs)
    if stream:
        frames = await events(await routes.create_response(request, fixture.request, USER))
        assert frames[-1]["type"] == "response.failed"
        assert frames[-1]["response"]["output"] == []
        assert not any("function_call_arguments" in frame["type"] or frame.get("item", {}).get("type") == "function_call" for frame in frames)
    else:
        with pytest.raises(HTTPException) as exc:
            await routes.create_response(request, fixture.request, USER)
        assert exc.value.status_code == 502


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["", "Partial answer."])
async def test_length_discards_buffered_calls_and_preserves_partial_text_usage(provider, stream, text):
    raw = reply(call(name="invalid"), text=text, reason="length")
    fixture = provider(raw=raw, chunks=[raw])
    result = await routes.create_response(payload(stream=stream), fixture.request, USER)
    if stream:
        frames = await events(result)
        assert frames[-1]["type"] == "response.incomplete"
        assert not any("function_call_arguments" in frame["type"] for frame in frames)
        result = frames[-1]["response"]
    assert result["status"] == "incomplete" and result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["output_text"] == text and result["usage"]["output_tokens"] == 11
    assert all(item["type"] == "message" for item in result["output"])


async def test_unconfirmed_eof_preserves_text_without_claiming_token_limit(provider):
    fixture = provider(chunks=[{"message": {"content": "Partial", "tool_calls": [call()]}, "done": False}])
    frames = await events(await routes.create_response(payload(stream=True), fixture.request, USER))
    result = frames[-1]["response"]
    assert frames[-1]["type"] == "response.failed"
    assert result["error"]["code"] == "responses_stream_interrupted"
    assert result["output_text"] == "Partial" and result["incomplete_details"] is None
    assert not any(item["type"] == "function_call" for item in result["output"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bad", [0, False, [], {}, 12])
async def test_malformed_provider_content_rejected(provider, stream, bad):
    raw = reply(call(), text=bad)
    fixture = provider(raw=raw, chunks=[raw])
    if stream:
        frames = await events(await routes.create_response(payload(stream=True), fixture.request, USER))
        assert frames[-1]["type"] == "response.failed"
        assert frames[-1]["response"]["output"] == []
    else:
        with pytest.raises(HTTPException) as exc:
            await routes.create_response(payload(), fixture.request, USER)
        assert exc.value.status_code == 502


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bad", [-1, True, "5", 2.5])
async def test_malformed_usage_rejected_before_exposing_calls(provider, stream, bad):
    raw = reply(call(), eval_count=bad)
    fixture = provider(raw=raw, chunks=[raw])
    if stream:
        frames = await events(await routes.create_response(payload(stream=True), fixture.request, USER))
        assert frames[-1]["type"] == "response.failed" and frames[-1]["response"]["output"] == []
    else:
        with pytest.raises(HTTPException) as exc:
            await routes.create_response(payload(), fixture.request, USER)
        assert exc.value.status_code == 502


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode,status", [("virtual", 400), ("skill", 400), ("json_schema", 400),
                                          ("required", 400), ("builtin", 400), ("disabled", 403),
                                          ("role", 403), ("allowlist", 403), ("capability", 400), ("lookup", 502)])
async def test_model_and_control_denials_precede_any_file_fetch_or_sse(provider, monkeypatch, stream, mode, status):
    fixture = provider(caps=[] if mode == "capability" else None, show_status=503 if mode == "lookup" else 200)
    request = payload(stream=stream)
    request.input = ResponseCreateRequest(model=request.model, input=[{"role": "user", "content": [
        {"type": "input_file", "file_url": "https://example.org/document"},
    ]}]).input
    if mode == "virtual":
        request.model = "audrey_fast"
    elif mode == "skill":
        request.skill = "grounded-document-analysis"
    elif mode == "json_schema":
        request.text = ResponseCreateRequest(model=request.model, input="x", text={"format": {
            "type": "json_schema", "name": "value", "schema": TOOLS[0]["parameters"], "strict": True,
        }}).text
    elif mode == "required":
        request.tool_choice = "required"
    elif mode == "builtin":
        request.tools = [{"type": "web_search"}]
    elif mode == "disabled":
        fixture.state.cfg.raw["passthrough"]["enabled"] = False
    elif mode == "role":
        fixture.state.cfg.raw["passthrough"]["require_role"] = "admin"
    elif mode == "allowlist":
        fixture.state.cfg.raw["passthrough"]["allowed_models"] = ["other"]

    async def no_io(*args, **kwargs):
        pytest.fail("file input admitted before tool request denial")

    monkeypatch.setattr(routes, "response_input_messages", no_io)
    with pytest.raises(HTTPException) as exc:
        await routes.create_response(request, fixture.request, USER)
    assert exc.value.status_code == status and fixture.captured == []


async def test_result_and_schema_text_share_pre_fetch_budget(provider, monkeypatch):
    fixture = provider()
    request = payload()
    request.input = ResponseCreateRequest(model=request.model, input=[
        {"role": "user", "content": "Please check."},
        {"type": "function_call", "call_id": "call_old", "name": "get_marker", "arguments": '{"nonce":"one"}'},
        {"type": "function_call_output", "call_id": "call_old", "output": "long result " * 100},
        {"role": "user", "content": [{"type": "input_file", "file_url": "https://example.org/doc"}]},
    ]).input
    monkeypatch.setattr(file_inputs, "MAX_PROMPT_CHARS", 500)

    async def no_io(*args, **kwargs):
        pytest.fail("oversized caller history opened file inputs")

    monkeypatch.setattr(routes, "response_input_messages", no_io)
    with pytest.raises(HTTPException) as exc:
        await routes.create_response(request, fixture.request, USER)
    assert exc.value.status_code == 413 and fixture.paths == []


async def test_vision_description_is_checked_before_final_generation(provider, monkeypatch):
    fixture = provider()
    monkeypatch.setattr(file_inputs, "MAX_PROMPT_CHARS", 1000)

    async def describe(messages, **kwargs):
        return [{"role": "user", "content": "vision text " * 100}], 1

    monkeypatch.setattr(generation, "describe_for_text_model", describe)
    with pytest.raises(HTTPException) as exc:
        await routes.create_response(payload(stream=True), fixture.request, USER)
    assert exc.value.status_code == 413 and fixture.captured == []


async def test_owned_document_with_tool_replay_keeps_principal_evidence_scope(library, provider):
    library.add(text="Evidence ORCHID-42")
    fixture = provider(raw=reply(text="ORCHID-42"))
    fixture.state.cfg.raw["vision"]["max_images_per_turn"] = 4
    request = payload()
    request.input = ResponseCreateRequest(model=request.model, input=[
        {"role": "user", "content": [{"type": "input_file", "file_id": "file_doc"}]},
        {"type": "function_call", "call_id": "call_old", "name": "get_marker", "arguments": '{"nonce":"one"}'},
        {"type": "function_call_output", "call_id": "call_old", "output": "CLIENT_RESULT"},
    ]).input
    request.tool_choice = "none"
    result = await routes.create_response(request, fixture.request, library.user)
    assert result["output_text"] == "ORCHID-42" and library.listing_calls == ["owner-store"]
    assert "Attached document evidence (quoted JSON)" in fixture.captured[0]["messages"][0]["content"]
    assert "Evidence ORCHID-42" in fixture.captured[0]["messages"][0]["content"]


async def test_stream_close_records_cancellation_once_and_does_not_start_provider(provider, monkeypatch):
    fixture = provider()
    observations = []
    monkeypatch.setattr(generation, "_observe", lambda started, outcome: observations.append(outcome))
    result = await routes.create_response(payload(stream=True), fixture.request, USER)
    await anext(result.body_iterator)
    await result.body_iterator.aclose()
    assert fixture.captured == [] and observations == ["cancelled"]
    # A following request can still acquire both user and GPU slots.
    result = await routes.create_response(payload(), fixture.request, USER)
    assert result["status"] == "completed"


async def test_stream_cancellation_releases_provider_gate_and_user_slot(provider, monkeypatch):
    fixture = provider()
    closed = asyncio.Event()
    started = asyncio.Event()
    observations = []
    monkeypatch.setattr(generation, "_observe", lambda start, outcome: observations.append(outcome))

    async def native(**kwargs):
        try:
            started.set()
            await asyncio.Event().wait()
            yield {}
        finally:
            closed.set()

    monkeypatch.setattr(fixture.state.ollama, "chat_stream", native)
    result = await routes.create_response(payload(stream=True), fixture.request, USER)
    await anext(result.body_iterator)
    task = asyncio.create_task(anext(result.body_iterator))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set() and observations == ["cancelled"]
    followup = await routes.create_response(payload(), fixture.request, USER)
    assert followup["status"] == "completed"
