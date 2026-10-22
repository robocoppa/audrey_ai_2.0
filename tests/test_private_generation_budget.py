"""Real Ollama transport enforces opted-in Private task generation budgets."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing
from unittest.mock import AsyncMock

import httpx
import pytest

from audrey.models.ollama import (
    GenerationUsageError,
    OllamaClient,
    OllamaError,
    generation_budget,
    observe_model_calls,
)
from audrey.processing_policy import ProcessingPolicy, processing_scope

_MODEL = "local-chat:latest"
_MESSAGES = [{"role": "user", "content": "private test input"}]


def _client(body=None, *, chunks=None, requests=None):
    seen = requests if requests is not None else []

    def handle(request):
        seen.append(request)
        if chunks is not None:
            return httpx.Response(200, content="".join(json.dumps(c) + "\n" for c in chunks))
        return httpx.Response(200, json=body or {"done": True, "eval_count": 7})

    return OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handle), trust_env=False)


async def _invoke(client, stream, **kwargs):
    if stream:
        return [chunk async for chunk in client.chat_stream(model=_MODEL, messages=_MESSAGES, **kwargs)]
    return await client.chat(model=_MODEL, messages=_MESSAGES, **kwargs)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("requested, expected", [(None, 40), (100, 40), (9, 9), (-1, 40), (0, 40), (True, 40)])
async def test_cap_copies_options_and_records_exact_terminal(stream, requested, expected):
    seen = []
    options = {"temperature": .2}
    if requested is not None:
        options["num_predict"] = requested
    original = dict(options)
    check, record = AsyncMock(return_value=40), AsyncMock()
    terminal = {"done": True, "eval_count": 7, "message": {"content": "answer"}}
    client = _client(terminal, chunks=[terminal] if stream else None, requests=seen)
    try:
        with generation_budget(check, record):
            await _invoke(client, stream, options=options)
    finally:
        await client.aclose()
    payload = json.loads(seen[0].content)
    assert payload["options"] == {"temperature": .2, "num_predict": expected}
    assert options == original
    check.assert_awaited_once_with()
    record.assert_awaited_once_with(7)


@pytest.mark.parametrize("stream", [False, True])
async def test_live_guard_runs_after_private_metadata_and_prevents_content_post(stream):
    paths, calls = [], []
    digest = "a" * 64
    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": _MODEL, "digest": digest}]})
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
            "general.architecture": "qwen", "general.parameter_count": 100,
        }})

    async def check():
        assert paths == ["/api/tags", "/api/show"]
        raise PermissionError("revoked")

    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handle), trust_env=False)
    policy = ProcessingPolicy(local_models={_MODEL: digest}, ollama_base_url="http://ollama:11434",
                              allowed_tools=frozenset(), tool_servers=frozenset())
    try:
        with processing_scope(policy), generation_budget(check, AsyncMock()), observe_model_calls(calls.append):
            with pytest.raises(PermissionError, match="revoked"):
                await _invoke(client, stream)
    finally:
        await client.aclose()
    assert paths == ["/api/tags", "/api/show"]
    assert calls == []


@pytest.mark.parametrize("remaining", [0, -1, True, None, "40"])
async def test_invalid_remaining_budget_never_dispatches(remaining):
    seen = []
    client = _client(requests=seen)
    try:
        with pytest.raises(GenerationUsageError, match="generation_budget_exhausted"):
            with generation_budget(AsyncMock(return_value=remaining), AsyncMock()):
                await _invoke(client, False)
    finally:
        await client.aclose()
    assert seen == []


async def test_stream_usage_saved_before_terminal_is_yielded_and_not_counted_on_close_twice():
    record = AsyncMock()
    terminal = {"done": True, "eval_count": 8}
    client = _client(chunks=[{"done": False, "message": {"content": "answer"}}, terminal, terminal])
    try:
        with generation_budget(AsyncMock(return_value=40), record):
            async with aclosing(client.chat_stream(model=_MODEL, messages=_MESSAGES)) as stream:
                await anext(stream)
                record.assert_not_awaited()
                assert await anext(stream) == terminal
                record.assert_awaited_once_with(8)
    finally:
        await client.aclose()
    record.assert_awaited_once_with(8)


async def test_duplicate_terminal_records_only_once():
    record = AsyncMock()
    terminal = {"done": True, "eval_count": 8}
    client = _client(chunks=[terminal, terminal])
    try:
        with generation_budget(AsyncMock(return_value=40), record):
            await _invoke(client, True)
    finally:
        await client.aclose()
    record.assert_awaited_once_with(8)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("count", [None, True, -1, "7", 1.5, 2**63])
async def test_unknown_or_malformed_usage_fails_only_scoped_generation(stream, count):
    terminal = {"done": True, "eval_count": count}
    record = AsyncMock()
    client = _client(terminal, chunks=[terminal] if stream else None)
    try:
        with pytest.raises(GenerationUsageError, match="generation_usage_unavailable"):
            with generation_budget(AsyncMock(return_value=40), record):
                await _invoke(client, stream)
        await _invoke(client, stream)
    finally:
        await client.aclose()
    record.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
async def test_measured_zero_output_is_valid(stream):
    terminal = {"done": True, "eval_count": 0}
    record = AsyncMock()
    client = _client(terminal, chunks=[terminal] if stream else None)
    try:
        with generation_budget(AsyncMock(return_value=40), record):
            await _invoke(client, stream)
    finally:
        await client.aclose()
    record.assert_awaited_once_with(0)


@pytest.mark.parametrize("stream", [False, True])
async def test_missing_terminal_never_becomes_measured_zero(stream):
    body = {"done": False, "message": {"content": "partial"}}
    record = AsyncMock()
    client = _client(body, chunks=[body] if stream else None)
    try:
        with pytest.raises(GenerationUsageError, match="generation_usage_unavailable"):
            with generation_budget(AsyncMock(return_value=40), record):
                await _invoke(client, stream)
    finally:
        await client.aclose()
    record.assert_not_awaited()


async def test_closed_scope_refuses_inherited_child_before_content_dispatch():
    gate = asyncio.Event()
    seen = []
    client = _client(requests=seen)

    async def child():
        await gate.wait()
        return await _invoke(client, False)

    try:
        with generation_budget(AsyncMock(return_value=40), AsyncMock()):
            inherited = asyncio.create_task(child())
        gate.set()
        with pytest.raises(GenerationUsageError, match="generation_budget_scope_expired"):
            await inherited
        # Scope reset in the parent leaves ordinary callers unchanged.
        await _invoke(client, False)
    finally:
        await client.aclose()
    assert len(seen) == 1
    assert "options" not in json.loads(seen[0].content)


async def test_scope_closed_while_guard_waits_cannot_send_content():
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []
    client = _client(requests=seen)

    async def check():
        entered.set()
        await release.wait()
        return 40

    try:
        with generation_budget(check, AsyncMock()):
            call = asyncio.create_task(_invoke(client, False))
            await entered.wait()
        release.set()
        with pytest.raises(GenerationUsageError, match="generation_budget_scope_expired"):
            await call
    finally:
        await client.aclose()
    assert seen == []


async def test_real_stream_progress_is_throttled_but_first_and_terminal_always_report(monkeypatch):
    monkeypatch.setattr("audrey.models.ollama.time.monotonic", lambda: 10)
    progress, record = AsyncMock(), AsyncMock()
    chunks = [{"done": False, "message": {"content": ""}},
              {"done": False, "message": {"thinking": "work"}},
              {"done": False, "message": {"content": "answer"}},
              {"done": False, "message": {"tool_calls": [{"function": {"name": "calculate"}}]}},
              {"done": True, "eval_count": 8}]
    client = _client(chunks=chunks)
    try:
        with generation_budget(AsyncMock(return_value=40), record, progress=progress):
            await _invoke(client, True)
    finally:
        await client.aclose()
    assert progress.await_count == 2
    record.assert_awaited_once_with(8)


async def test_embedding_does_not_consume_generation_budget():
    check, record = AsyncMock(return_value=0), AsyncMock()
    client = _client({"embeddings": [[1.0, 0.0]]})
    try:
        with generation_budget(check, record):
            assert await client.embed(model=_MODEL, texts=["document"]) == [[1.0, 0.0]]
    finally:
        await client.aclose()
    check.assert_not_awaited()
    record.assert_not_awaited()


async def test_swallowed_usage_error_refuses_fallback_dispatch_and_scope_success():
    seen = []
    client = _client({"done": True, "message": {"content": "uncounted"}}, requests=seen)
    try:
        with pytest.raises(GenerationUsageError, match="generation_usage_unavailable"):
            with generation_budget(AsyncMock(return_value=40), AsyncMock()):
                for _ in range(2):
                    try:
                        await _invoke(client, False)
                    except GenerationUsageError:
                        pass
        # The failed scope cannot poison ordinary requests afterward.
        await _invoke(client, False)
    finally:
        await client.aclose()
    assert len(seen) == 2


async def test_latched_usage_does_not_replace_an_active_cancellation():
    client = _client({"done": True, "message": {"content": "uncounted"}})
    try:
        with pytest.raises(asyncio.CancelledError):
            with generation_budget(AsyncMock(return_value=40), AsyncMock()):
                try:
                    await _invoke(client, False)
                except GenerationUsageError:
                    pass
                raise asyncio.CancelledError()
    finally:
        await client.aclose()


async def test_early_close_before_terminal_refuses_success_without_inventing_usage():
    record = AsyncMock()
    client = _client(chunks=[{"done": False, "message": {"content": "partial"}},
                            {"done": True, "eval_count": 8}])
    try:
        with pytest.raises(GenerationUsageError, match="generation_usage_unavailable"):
            with generation_budget(AsyncMock(return_value=40), record):
                async with aclosing(client.chat_stream(model=_MODEL, messages=_MESSAGES)) as stream:
                    await anext(stream)
    finally:
        await client.aclose()
    record.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
async def test_swallowed_transport_error_cannot_finish_with_unknown_usage(stream):
    def fail(request):
        raise httpx.ReadError("provider disconnected", request=request)

    record = AsyncMock()
    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(fail), trust_env=False)
    try:
        with pytest.raises(GenerationUsageError, match="generation_usage_unavailable"):
            with generation_budget(AsyncMock(return_value=40), record):
                try:
                    await _invoke(client, stream)
                except OllamaError:
                    pass
    finally:
        await client.aclose()
    record.assert_not_awaited()


async def test_swallowed_usage_persistence_failure_refuses_scope_success():
    client = _client()
    try:
        with pytest.raises(GenerationUsageError, match="generation_usage_unavailable"):
            with generation_budget(AsyncMock(return_value=40), AsyncMock(side_effect=RuntimeError("storage"))):
                try:
                    await _invoke(client, False)
                except RuntimeError:
                    pass
    finally:
        await client.aclose()
