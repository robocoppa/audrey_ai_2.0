"""Stored Responses through ASGI, real SQLite, and native Ollama adaptation."""
from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import AuthedUser, require_user
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.inflight import UserInflightRegistry
from audrey.routes.openai import file_inputs, routes
from audrey.routes.openai.schemas import ResponseCreateRequest

MODEL = "storage-model"
TOOLS = [{
    "type": "function", "name": "get_marker", "strict": True,
    "parameters": {"type": "object", "properties": {"nonce": {"type": "string"}},
                   "required": ["nonce"], "additionalProperties": False},
}]


def reply(*, text="A first answer.", calls=None, reason="stop"):
    return {"message": {"role": "assistant", "content": text, "tool_calls": calls or []},
            "done": True, "done_reason": reason, "prompt_eval_count": 17, "eval_count": 11}


def call(nonce="one"):
    return {"function": {"name": "get_marker", "arguments": {"nonce": nonce}}}


def request_body(**updates):
    return {"model": f"audrey_passthrough/{MODEL}", "input": "First question.", **updates}


def parse_events(text):
    events = []
    for block in text.split("\n\n"):
        if not block:
            continue
        lines = block.splitlines()
        event = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        assert next(line[7:] for line in lines if line.startswith("event: ")) == event["type"]
        events.append(event)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    return events


@pytest.fixture
async def service(tmp_path):
    store = ApplicationStore(tmp_path / "application.sqlite")
    alice = await store.resolve_external_identity(
        provider="owui", subject="storage-alice", email="alice@example.com",
        display_name="Alice", role="user", auth_method="owui_bearer",
        legacy_storage_namespace="alice@example.com",
    )
    bob = await store.resolve_external_identity(
        provider="owui", subject="storage-bob", email="bob@example.com",
        display_name="Bob", role="user", auth_method="owui_bearer",
        legacy_storage_namespace="bob@example.com",
    )
    current = [AuthedUser(email=alice.email, role=alice.role, owui_id="alice", principal=alice)]
    state = SimpleNamespace(raw=reply(), chunks=None, captured=[], paths=[])

    def handler(request):
        state.paths.append(request.url.path)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        native = json.loads(request.content)
        state.captured.append(native)
        if native.get("stream"):
            chunks = state.chunks if state.chunks is not None else [state.raw]
            return httpx.Response(200, content="".join(json.dumps(chunk) + "\n" for chunk in chunks).encode())
        return httpx.Response(200, json=state.raw)

    ollama = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))
    cfg = SimpleNamespace(
        raw={"passthrough": {"enabled": True, "require_role": None, "allowed_models": [MODEL, "second-model"]},
             "vision": {"describe_for_text_models": False}},
        timeouts={"medium": 3}, model_registry={},
    )
    registry = ModelRegistry(cfg)
    registry.location_of = lambda _name: "local"
    app = FastAPI()
    app.include_router(routes.router)
    app.state.application_store = store
    app.state.cfg = cfg
    app.state.ollama = ollama
    app.state.registry = registry
    app.state.gate = FairLocalGate(concurrency=1)
    app.state.inflight = UserInflightRegistry(max_inflight_per_user=1)
    app.state.health = HealthTracker()
    app.dependency_overrides[require_user] = lambda: current[0]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://audrey") as client:
        yield SimpleNamespace(app=app, client=client, store=store, alice=alice, bob=bob,
                              current=current, provider=state, cfg=cfg)
    await ollama.aclose()
    store.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_store_true_saves_exact_terminal_object_and_get_uses_no_provider(service, stream):
    result = await service.client.post("/v1/responses", json=request_body(store=True, stream=stream, tools=[]))
    assert result.status_code == 200
    if stream:
        frames = parse_events(result.text)
        assert frames[-1]["type"] == "response.completed"
        response = frames[-1]["response"]
        assert all(frame["response"]["store"] is True for frame in frames if "response" in frame)
    else:
        response = result.json()
    assert response["store"] is True and response["previous_response_id"] is None
    before = len(service.provider.paths)
    retrieved = await service.client.get("/v1/responses/" + response["id"])
    assert retrieved.status_code == 200 and retrieved.json() == response
    assert len(service.provider.paths) == before


@pytest.mark.parametrize("store_value", [None, False])
@pytest.mark.parametrize("stream", [False, True])
async def test_default_and_false_remain_unstored(service, store_value, stream):
    body = request_body(stream=stream, tools=[])
    if store_value is not None:
        body["store"] = store_value
    result = await service.client.post("/v1/responses", json=body)
    assert result.status_code == 200
    response = parse_events(result.text)[-1]["response"] if stream else result.json()
    assert response["store"] is False
    assert (await service.client.get("/v1/responses/" + response["id"])).status_code == 404


@pytest.mark.parametrize("stream", [False, True])
async def test_chain_replays_full_text_history_without_inheriting_instructions_or_model(service, stream):
    first = (await service.client.post("/v1/responses", json=request_body(
        store=True, instructions="Old instruction.", tools=[], temperature=0.2,
        metadata={"old": "value"},
    ))).json()
    service.provider.raw = reply(text="A second answer.")
    second = await service.client.post("/v1/responses", json=request_body(
        input="Second question.", previous_response_id=first["id"], store=True,
        instructions="Current instruction.", model="audrey_passthrough/second-model", stream=stream,
    ))
    assert second.status_code == 200
    response = parse_events(second.text)[-1]["response"] if stream else second.json()
    assert response["previous_response_id"] == first["id"]
    assert response["model"] == "audrey_passthrough/second-model"
    assert response["instructions"] == "Current instruction." and response["metadata"] == {}
    native = service.provider.captured[-1]
    assert native["model"] == "second-model" and "temperature" not in native.get("options", {})
    assert native["messages"] == [
        {"role": "system", "content": "Current instruction."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "A first answer."},
        {"role": "user", "content": "Second question."},
    ]
    assert (await service.client.get("/v1/responses/" + response["id"])).json() == response
    service.provider.raw = reply(text="Third answer.")
    third = await service.client.post("/v1/responses", json=request_body(
        input="Third question.", previous_response_id=response["id"], tools=[],
    ))
    assert third.status_code == 200 and third.json()["store"] is False
    native = service.provider.captured[-1]
    assert [message["content"] for message in native["messages"]] == [
        "First question.", "A first answer.", "Second question.", "A second answer.", "Third question.",
    ]
    assert (await service.client.get("/v1/responses/" + third.json()["id"])).status_code == 404


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
async def test_chain_resolves_saved_function_calls_with_only_new_client_results(service, stream, parallel):
    service.provider.raw = reply(text="Checking." if parallel else "", calls=[call("one"), call("two")] if parallel else [call()])
    root_result = await service.client.post("/v1/responses", json=request_body(store=True, tools=copy.deepcopy(TOOLS)))
    assert root_result.status_code == 200
    root = root_result.json()
    calls = [item for item in root["output"] if item["type"] == "function_call"]
    assert len(calls) == (2 if parallel else 1)
    service.provider.raw = reply(text="CLIENT RESULT")
    result = await service.client.post("/v1/responses", json=request_body(
        input=[{"type": "function_call_output", "call_id": item["call_id"], "output": "RESULT-" + str(index)}
               for index, item in enumerate(calls)],
        previous_response_id=root["id"], store=True, stream=stream, tool_choice="none",
    ))
    assert result.status_code == 200
    response = parse_events(result.text)[-1]["response"] if stream else result.json()
    assert response["output_text"] == "CLIENT RESULT" and response["tools"] == []
    assert response["tool_choice"] == "none" and response["previous_response_id"] == root["id"]
    native = service.provider.captured[-1]
    assert "tools" not in native
    assert [message["role"] for message in native["messages"]] == ["user", "assistant", *(["tool"] * len(calls))]
    assert len(native["messages"][1]["tool_calls"]) == len(calls)
    assert native["messages"][1].get("content", "") == ("Checking." if parallel else "")
    assert [message["content"] for message in native["messages"][2:]] == ["RESULT-" + str(index) for index in range(len(calls))]
    assert (await service.client.get("/v1/responses/" + response["id"])).json() == response


@pytest.mark.parametrize("kind", ["foreign", "missing", "deleted"])
async def test_parent_lookup_is_uniform_owner_scoped_and_precedes_provider(service, kind):
    root = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()
    response_id = root["id"]
    if kind == "foreign":
        service.current[0] = AuthedUser(email=service.bob.email, role="user", owui_id="bob", principal=service.bob)
    elif kind == "deleted":
        deleted = await service.client.delete("/v1/responses/" + response_id)
        assert deleted.status_code == 200 and deleted.json() == {"id": response_id, "object": "response.deleted", "deleted": True}
    else:
        response_id = "resp_missing"
    before = len(service.provider.paths)
    get = await service.client.get("/v1/responses/" + response_id)
    child = await service.client.post("/v1/responses", json=request_body(previous_response_id=response_id, tools=[]))
    delete = await service.client.delete("/v1/responses/" + response_id)
    assert get.status_code == child.status_code == delete.status_code == 404
    assert get.json() == child.json() == delete.json()
    assert get.json()["detail"]["error"] == "responses_not_found"
    assert len(service.provider.paths) == before


@pytest.mark.parametrize("part", [
    {"type": "input_file", "file_url": "https://example.org/document.pdf"},
    {"type": "input_image", "image_url": "https://example.org/image.png"},
    {"type": "input_file", "file_id": "owned-document"},
    {"type": "input_image", "image_url": "data:image/png;base64,eA=="},
])
@pytest.mark.parametrize("chained", [False, True])
async def test_storage_rejects_every_file_image_before_hydration(service, monkeypatch, part, chained):
    parent = None
    if chained:
        parent = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()["id"]
    before = len(service.provider.paths)

    async def forbidden(*args, **kwargs):
        pytest.fail("stored file/image reached hydration")

    monkeypatch.setattr(routes, "response_input_messages", forbidden)
    body = request_body(input=[{"role": "user", "content": [part]}], tools=[])
    if chained:
        body["previous_response_id"] = parent
    else:
        body["store"] = True
    result = await service.client.post("/v1/responses", json=body)
    assert result.status_code == 400 and result.json()["detail"]["error"] == "responses_feature_unsupported"
    assert len(service.provider.paths) == before


async def test_chained_calls_validate_against_current_schema_and_require_results(service):
    service.provider.raw = reply(text="", calls=[call()])
    root = (await service.client.post("/v1/responses", json=request_body(store=True, tools=copy.deepcopy(TOOLS)))).json()
    item = root["output"][0]
    before = len(service.provider.paths)
    missing = await service.client.post("/v1/responses", json=request_body(previous_response_id=root["id"]))
    changed = copy.deepcopy(TOOLS)
    changed[0]["parameters"]["properties"]["nonce"] = {"type": "integer"}
    invalid = await service.client.post("/v1/responses", json=request_body(
        previous_response_id=root["id"], tools=changed,
        input=[{"type": "function_call_output", "call_id": item["call_id"], "output": "RESULT"}],
    ))
    assert missing.status_code == invalid.status_code == 400
    assert len(service.provider.paths) == before


async def test_chain_combined_prompt_revalidated_before_provider(service, monkeypatch):
    root = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()
    before = len(service.provider.paths)
    monkeypatch.setattr(file_inputs, "MAX_PROMPT_CHARS", 30)
    result = await service.client.post("/v1/responses", json=request_body(
        input="Second question.", previous_response_id=root["id"], tools=[],
    ))
    assert result.status_code == 413 and len(service.provider.paths) == before


async def test_chain_rechecks_current_model_access(service):
    root = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()
    before = len(service.provider.paths)
    service.cfg.raw["passthrough"]["allowed_models"] = ["second-model"]
    denied = await service.client.post("/v1/responses", json=request_body(previous_response_id=root["id"], tools=[]))
    assert denied.status_code == 403 and len(service.provider.paths) == before


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("explicit_tools", [False, True])
async def test_truncated_output_is_never_saved(service, stream, explicit_tools):
    service.provider.raw = reply(text="Partial answer.", reason="length")
    body = request_body(store=True, stream=stream)
    if explicit_tools:
        body["tools"] = []
    result = await service.client.post("/v1/responses", json=body)
    assert result.status_code == 200
    if stream:
        frames = parse_events(result.text)
        assert frames[-1]["type"] == "response.incomplete"
        response = frames[-1]["response"]
    else:
        response = result.json()
    assert response["status"] == "incomplete" and response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert (await service.client.get("/v1/responses/" + response["id"])).status_code == 404


@pytest.mark.parametrize("stream", [False, True])
async def test_storage_error_cannot_return_completed_or_executable_call(service, monkeypatch, stream):
    service.provider.raw = reply(text="", calls=[call()])

    async def disk_failed(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(service.store.responses, "save", disk_failed)
    result = await service.client.post("/v1/responses", json=request_body(store=True, tools=copy.deepcopy(TOOLS), stream=stream))
    if stream:
        assert result.status_code == 200
        frames = parse_events(result.text)
        assert frames[-1]["type"] == "response.failed"
        assert frames[-1]["response"]["error"]["code"] == "responses_storage_failed"
        assert all(frame["type"] != "response.completed" for frame in frames)
        assert not any(frame.get("item", {}).get("type") == "function_call" for frame in frames)
        assert frames[-1]["response"]["output"] == []
    else:
        assert result.status_code == 503 and result.json()["detail"]["error"] == "responses_storage_failed"


async def test_stream_saves_before_terminal_is_exposed(service):
    user = service.current[0]
    response = await routes.create_response(
        ResponseCreateRequest(**request_body(store=True, stream=True, tools=copy.deepcopy(TOOLS))),
        SimpleNamespace(app=service.app), user,
    )
    async for chunk in response.body_iterator:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        for block in text.split("\n\n"):
            if "event: response.completed\n" not in block:
                continue
            terminal = json.loads(next(line[6:] for line in block.splitlines() if line.startswith("data: ")))["response"]
            record = await service.store.responses.get(user.principal.user_id, terminal["id"])
            assert record is not None and record["response"] == terminal


async def test_cancelled_stream_leaves_no_saved_response_or_busy_user_slot(service):
    response = await routes.create_response(
        ResponseCreateRequest(**request_body(store=True, stream=True, tools=[])),
        SimpleNamespace(app=service.app), service.current[0],
    )
    first = await anext(response.body_iterator)
    first = first.decode() if isinstance(first, bytes) else first
    response_id = parse_events(first)[0]["response"]["id"]
    await response.body_iterator.aclose()
    assert (await service.client.get("/v1/responses/" + response_id)).status_code == 404
    followup = await service.client.post("/v1/responses", json=request_body(tools=[]))
    assert followup.status_code == 200


async def test_delete_parent_cascades_saved_descendants_without_affecting_other_roots(service):
    first = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()
    second = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[], previous_response_id=first["id"]))).json()
    separate = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()
    deleted = await service.client.delete("/v1/responses/" + first["id"])
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    for item in (first, second):
        assert (await service.client.get("/v1/responses/" + item["id"])).status_code == 404
    assert (await service.client.get("/v1/responses/" + separate["id"])).json() == separate


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
async def test_cancellation_during_sqlite_commit_undoes_saved_record(service, monkeypatch, stream, repeated):
    entered = asyncio.Event()
    release = asyncio.Event()
    identities = []
    saved = asyncio.Event()
    original = service.store.responses.save

    async def delayed_save(owner_id, response, replay, previous_response_id=None, now=None):
        identities.append(response["id"])
        entered.set()
        await release.wait()
        await original(owner_id, response, replay, previous_response_id, now)
        saved.set()

    monkeypatch.setattr(service.store.responses, "save", delayed_save)
    payload = ResponseCreateRequest(**request_body(store=True, stream=stream, tools=[]))

    async def consume():
        result = await routes.create_response(payload, SimpleNamespace(app=service.app), service.current[0])
        if stream:
            async for _ in result.body_iterator:
                pass
        return result

    operation = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), timeout=2)
    operation.cancel()
    await asyncio.sleep(0)
    waits_for_commit = not operation.done()
    if repeated:
        operation.cancel()
        await asyncio.sleep(0)
        waits_for_commit = waits_for_commit and not operation.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await asyncio.wait_for(saved.wait(), timeout=2)
    assert waits_for_commit, "cancelled persistence must finish and undo its write despite repeated cancellation"
    assert len(identities) == 1
    assert (await service.client.get("/v1/responses/" + identities[0])).status_code == 404
    assert (await service.client.post("/v1/responses", json=request_body(tools=[]))).status_code == 200


@pytest.mark.parametrize("missing", ["principal", "store"])
@pytest.mark.parametrize("operation", ["post", "get", "delete"])
async def test_storage_requires_canonical_owner_and_available_repository(service, missing, operation):
    if missing == "principal":
        service.current[0] = AuthedUser(email="alice@example.com", role="user", owui_id="alice")
    else:
        del service.app.state.application_store
    before = len(service.provider.paths)
    if operation == "post":
        result = await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))
    else:
        result = await getattr(service.client, operation)("/v1/responses/resp_missing")
    assert result.status_code == (401 if missing == "principal" else 503)
    assert len(service.provider.paths) == before


async def test_retained_parent_expiry_is_uniform_not_found_before_generation(service):
    root = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()
    with sqlite3.connect(service.store.path) as connection:
        connection.execute("UPDATE app_responses SET created_at = 0, expires_at = 1 WHERE response_id = ?", (root["id"],))
    before = len(service.provider.paths)
    result = await service.client.post("/v1/responses", json=request_body(previous_response_id=root["id"], tools=[]))
    assert result.status_code == 404 and result.json()["detail"]["error"] == "responses_not_found"
    assert len(service.provider.paths) == before


@pytest.mark.parametrize("stream", [False, True])
async def test_parent_deleted_during_generation_cannot_save_successful_child(service, monkeypatch, stream):
    root = (await service.client.post("/v1/responses", json=request_body(store=True, tools=[]))).json()
    original = service.store.responses.save

    async def remove_parent_before_commit(owner_id, response, replay, previous_response_id=None, now=None):
        await service.store.responses.delete(owner_id, previous_response_id)
        await original(owner_id, response, replay, previous_response_id, now)

    monkeypatch.setattr(service.store.responses, "save", remove_parent_before_commit)
    result = await service.client.post("/v1/responses", json=request_body(
        store=True, stream=stream, tools=[], previous_response_id=root["id"], input="A followup.",
    ))
    if stream:
        frames = parse_events(result.text)
        assert frames[-1]["type"] == "response.failed"
        assert frames[-1]["response"]["error"]["code"] == "responses_not_found"
        assert all(frame["type"] != "response.completed" for frame in frames)
    else:
        assert result.status_code == 404 and result.json()["detail"]["error"] == "responses_not_found"


@pytest.mark.parametrize("explicit_tools", [False, True])
async def test_unconfirmed_provider_eof_cannot_save_completed_response(service, explicit_tools):
    service.provider.chunks = [{"message": {"role": "assistant", "content": "Partial answer."}, "done": False}]
    body = request_body(store=True, stream=True)
    if explicit_tools:
        body["tools"] = []
    result = await service.client.post("/v1/responses", json=body)
    assert result.status_code == 200
    frames = parse_events(result.text)
    assert frames[-1]["type"] == "response.failed"
    response = frames[-1]["response"]
    assert response["status"] == "failed" and response["incomplete_details"] is None
    assert response["output_text"] == "Partial answer."
    assert (await service.client.get("/v1/responses/" + response["id"])).status_code == 404


async def test_plain_virtual_model_chain_does_not_acquire_client_tool_requirements(service, monkeypatch):
    forwarded = []

    async def generate(payload, _request, _user):
        forwarded.append(payload)
        return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1_800_000_000,
                "model": payload.model, "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "Virtual answer."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 11, "total_tokens": 28}}

    monkeypatch.setattr(routes, "chat_completions", generate)
    root = await service.client.post("/v1/responses", json=request_body(model="audrey_fast", store=True))
    assert root.status_code == 200
    result = await service.client.post("/v1/responses", json=request_body(
        model="audrey_auto", input="Followup.", previous_response_id=root.json()["id"],
    ))
    assert result.status_code == 200 and result.json()["output_text"] == "Virtual answer."
    assert len(forwarded) == 2
    assert [message.role for message in forwarded[1].messages] == ["user", "assistant", "user"]
    assert forwarded[1].model == "audrey_auto" and service.provider.paths == []
