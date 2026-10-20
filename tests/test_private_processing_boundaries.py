"""Private-context transport refusals at the real model and tool boundaries."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from audrey.models.ollama import OllamaClient, observe_model_calls
from audrey.processing_policy import ProcessingPolicy, ProcessingPolicyError, processing_scope
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec
from audrey.tools.dispatch import dispatch_one
from audrey.tools.private_dispatch import private_tool_scope

_MODEL = "local-chat:latest"
_DIGEST = "a" * 64
_PRIVATE_TEXT = "PRIVATE_TEST_CONTENT_17804"
_PATHS = ("chat", "stream", "embed")


def _policy() -> ProcessingPolicy:
    return ProcessingPolicy(
        local_models={_MODEL: _DIGEST}, ollama_base_url="http://ollama:11434",
        allowed_tools=frozenset({"calculate", "get_file_text"}),
        tool_servers=frozenset({"http://custom-tools:8001"}),
    )


def _metadata():
    return {
        "details": {"format": "gguf"},
        "model_info": {"general.architecture": "qwen", "general.parameter_count": 100},
    }


def _installed():
    return [{"name": _MODEL, "model": _MODEL, "digest": _DIGEST}]


async def _invoke(client, path, model=_MODEL):
    if path == "embed":
        return await client.embed(model=model, texts=[_PRIVATE_TEXT])
    messages = [{"role": "user", "content": _PRIVATE_TEXT}]
    if path == "stream":
        return [chunk async for chunk in client.chat_stream(model=model, messages=messages)]
    return await client.chat(model=model, messages=messages)


def _reply(request):
    payload = json.loads(request.content)
    if request.url.path == "/api/embed":
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
    body = {"message": {"content": "complete"}, "done": True}
    if payload.get("stream"):
        return httpx.Response(200, content=json.dumps(body) + "\n")
    return httpx.Response(200, json=body)


@pytest.mark.parametrize("path", _PATHS)
async def test_verified_local_calls_send_content_only_after_metadata_checks(path):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": _installed()})
        if request.url.path == "/api/show":
            assert json.loads(request.content) == {"model": _MODEL}
            return httpx.Response(200, json=_metadata())
        return _reply(request)

    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler), trust_env=False)
    try:
        with processing_scope(_policy()):
            await _invoke(client, path)
    finally:
        await client.aclose()
    assert [r.url.path for r in seen] == ["/api/tags", "/api/show", "/api/embed" if path == "embed" else "/api/chat"]
    assert all(_PRIVATE_TEXT.encode() not in r.content for r in seen[:2])
    assert _PRIVATE_TEXT.encode() in seen[-1].content


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("model", ["kimi-k3:cloud", "qwen:397b-cloud", "unapproved:latest"])
async def test_unapproved_models_never_send_any_request_or_observation(path, model):
    seen, observed = [], []

    def handler(request):
        seen.append(request)
        return _reply(request)

    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler), trust_env=False)
    try:
        with processing_scope(_policy()), observe_model_calls(observed.append):
            with pytest.raises(ProcessingPolicyError):
                await _invoke(client, path, model)
    finally:
        await client.aclose()
    assert seen == []
    assert observed == []


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("issue", ["digest", "remote_alias", "missing_weights", "missing_model", "outage", "malformed"])
async def test_ambiguous_or_unavailable_metadata_prevents_payload_egress(path, issue, caplog):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/api/tags":
            models = _installed()
            if issue == "digest":
                models[0]["digest"] = "b" * 64
            if issue == "missing_model":
                models = []
            if issue == "outage":
                return httpx.Response(503, text=_PRIVATE_TEXT)
            if issue == "malformed":
                return httpx.Response(200, text=_PRIVATE_TEXT)
            return httpx.Response(200, json={"models": models})
        assert request.url.path == "/api/show"
        metadata = _metadata()
        if issue == "remote_alias":
            metadata["remote_host"] = "https://remote.example"
            metadata["remote_model"] = "a-cloud-model"
        if issue == "missing_weights":
            metadata["model_info"] = {}
        return httpx.Response(200, json=metadata)

    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler), trust_env=False)
    try:
        with caplog.at_level(logging.DEBUG), processing_scope(_policy()):
            with pytest.raises(ProcessingPolicyError) as exc:
                await _invoke(client, path)
    finally:
        await client.aclose()
    assert all(r.url.path in {"/api/tags", "/api/show"} for r in seen)
    assert all(_PRIVATE_TEXT.encode() not in r.content for r in seen)
    assert _PRIVATE_TEXT not in str(exc.value) + caplog.text


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("issue", ["endpoint", "environment_proxy"])
async def test_untrusted_provider_or_proxy_transport_is_refused_before_metadata(path, issue):
    seen = []
    client = OllamaClient(
        "https://remote.example" if issue == "endpoint" else "http://ollama:11434",
        transport=httpx.MockTransport(lambda request: seen.append(request)),
        trust_env=issue == "environment_proxy",
    )
    try:
        with processing_scope(_policy()):
            with pytest.raises(ProcessingPolicyError):
                await _invoke(client, path)
    finally:
        await client.aclose()
    assert seen == []


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("failure", ["http", "transport", "body"])
async def test_private_provider_failures_do_not_echo_content_into_errors_or_logs(path, failure, caplog):
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": _installed()})
        if request.url.path == "/api/show":
            return httpx.Response(200, json=_metadata())
        if failure == "http":
            return httpx.Response(503, text=_PRIVATE_TEXT)
        if failure == "transport":
            raise httpx.ConnectError(_PRIVATE_TEXT, request=request)
        if path == "stream":
            return httpx.Response(200, content=_PRIVATE_TEXT + "\n")
        return httpx.Response(200, text=_PRIVATE_TEXT)

    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler), trust_env=False)
    try:
        with caplog.at_level(logging.DEBUG), processing_scope(_policy()):
            with pytest.raises(ProcessingPolicyError) as exc:
                await _invoke(client, path)
    finally:
        await client.aclose()
    assert _PRIVATE_TEXT not in str(exc.value) + caplog.text
    assert exc.value.__suppress_context__ or exc.value.__context__ is None


async def test_second_call_rechecks_changed_alias_instead_of_reusing_old_approval():
    generations = []
    show_count = 0

    def handler(request):
        nonlocal show_count
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": _installed()})
        if request.url.path == "/api/show":
            show_count += 1
            metadata = _metadata()
            if show_count == 2:
                metadata["remote_host"] = "https://remote.example"
            return httpx.Response(200, json=metadata)
        generations.append(request)
        return _reply(request)

    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler), trust_env=False)
    try:
        with processing_scope(_policy()):
            await _invoke(client, "chat")
            with pytest.raises(ProcessingPolicyError):
                await _invoke(client, "chat")
    finally:
        await client.aclose()
    assert show_count == 2
    assert len(generations) == 1


@pytest.mark.parametrize("path", _PATHS)
async def test_ordinary_requests_preserve_existing_cloud_calls_without_metadata_preflight(path):
    seen = []

    def handler(request):
        seen.append(request)
        return _reply(request)

    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))
    try:
        await _invoke(client, path, "kimi-k3:cloud")
    finally:
        await client.aclose()
    assert len(seen) == 1
    assert seen[0].url.path == ("/api/embed" if path == "embed" else "/api/chat")


def _registry(name="calculate", *, endpoint="http://custom-tools:8001", path=None):
    declaration = TOOL_DECLARATIONS.get(name)
    spec = ToolSpec(
        name=name, description=name, parameters={"type": "object", "properties": {}},
        server_url=endpoint, path=path or f"/{name}",
    )
    if declaration:
        spec.user_scope = declaration.user_scope
    return ToolRegistry(by_name={name: spec})


async def _dispatch(client, registry, name="calculate", args=None):
    return await dispatch_one(
        client, registry,
        {"id": "call_private", "function": {"name": name, "arguments": args if args is not None else {"expression": _PRIVATE_TEXT}}},
        max_result_chars=2000, timeout_s=5, user_id="owner",
    )


@pytest.mark.parametrize("name", ["web_search", "web_fetch", "unknown_tool", "memory_store", "reference_search"])
async def test_blocked_tools_cannot_parse_log_or_send_private_arguments(name, caplog):
    seen = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: seen.append(r)), trust_env=False) as client:
        with caplog.at_level(logging.DEBUG), processing_scope(_policy()):
            result = await _dispatch(client, _registry(name), name, _PRIVATE_TEXT)
    assert result.is_error
    assert seen == []
    assert _PRIVATE_TEXT not in result.content + caplog.text


@pytest.mark.parametrize("issue", ["endpoint", "path", "proxy", "missing_spec", "missing_endpoint"])
async def test_tool_dispatch_refuses_url_path_proxy_or_registry_bypasses(issue):
    seen = []
    registry = _registry(
        endpoint="https://remote.example" if issue == "endpoint" else "http://custom-tools:8001",
        path="/web_fetch" if issue == "path" else None,
    )
    if issue == "missing_spec":
        registry = ToolRegistry()
    if issue == "missing_endpoint":
        registry.by_name["calculate"].server_url = None
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: seen.append(r)), trust_env=issue == "proxy") as client:
        with processing_scope(_policy()):
            result = await _dispatch(client, registry)
    assert result.is_error
    assert seen == []


async def test_allowed_readonly_tool_still_binds_authenticated_owner():
    seen = []

    async def handler(args):
        seen.append(args)
        return {"text": "owned text"}

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: pytest.fail("No private HTTP")), trust_env=False) as client:
        with processing_scope(_policy()), private_tool_scope(user_id="owner", handlers={"get_file_text": handler}):
            result = await _dispatch(client, _registry("get_file_text"), "get_file_text", {"user": "foreign", "file_id": "file"})
    assert not result.is_error
    assert seen[0]["user"] == "owner"


async def test_private_tool_redirect_cannot_escape_even_with_redirecting_client():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(307, headers={"Location": "https://remote.example/steal"}, text=_PRIVATE_TEXT)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=True) as client:
        with processing_scope(_policy()):
            result = await _dispatch(client, _registry())
    assert result.is_error
    assert seen == []
    assert _PRIVATE_TEXT not in result.content


@pytest.mark.parametrize("failure", ["http", "timeout", "transport", "arguments"])
async def test_private_tool_failure_diagnostics_do_not_contain_content(failure, caplog):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout(_PRIVATE_TEXT, request=request)
        if failure == "transport":
            raise httpx.ConnectError(_PRIVATE_TEXT, request=request)
        return httpx.Response(500, text=_PRIVATE_TEXT)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        with caplog.at_level(logging.DEBUG), processing_scope(_policy()):
            result = await _dispatch(client, _registry(), args=_PRIVATE_TEXT if failure == "arguments" else None)
    assert result.is_error
    assert _PRIVATE_TEXT not in result.content + caplog.text
