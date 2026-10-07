"""Provider outcomes and reported usage from the real HTTP client, without live calls."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram

from audrey.models import ollama
from audrey.models.ollama import OllamaClient, OllamaError, observe_model_calls

_MODEL = "writer:latest"
_MESSAGES = [{"role": "user", "content": "Reply briefly."}]
_KINDS = ("input", "output", "cached_input")
_MISSING = object()


@pytest.fixture
def model_metrics(monkeypatch: pytest.MonkeyPatch) -> CollectorRegistry:
    """Avoid process-global samples leaking between tests or into other suites."""
    registry = CollectorRegistry()
    monkeypatch.setattr(
        ollama,
        "model_seconds",
        Histogram(
            "audrey_model_seconds", "Model duration", ("model", "outcome"), registry=registry,
        ),
    )
    for name in ("model_tokens_total", "model_usage_observations_total"):
        monkeypatch.setattr(
            ollama, name, Counter(f"audrey_{name}", "Provider usage", ("model", "kind"), registry=registry),
        )
    return registry


def _assert_outcome(registry: CollectorRegistry, expected: str, *, model: str = _MODEL, count: int = 1) -> None:
    assert {
        outcome: registry.get_sample_value(
            "audrey_model_seconds_count", {"model": model, "outcome": outcome},
        ) or 0
        for outcome in ("ok", "error", "cancelled")
    } == {outcome: count if outcome == expected else 0 for outcome in ("ok", "error", "cancelled")}


def _assert_usage(
    registry: CollectorRegistry,
    expected: dict[str, int],
    *,
    model: str = _MODEL,
    observations: int = 1,
) -> None:
    for kind in _KINDS:
        labels = {"model": model, "kind": kind}
        tokens = registry.get_sample_value("audrey_model_tokens_total", labels)
        seen = registry.get_sample_value("audrey_model_usage_observations_total", labels)
        if kind in expected:
            assert tokens == float(expected[kind]), kind
            assert seen == observations, kind
        else:
            assert tokens is None, kind
            assert seen is None, kind


def _wire_frame(body: Any) -> bytes:
    return (json.dumps(body) + "\n").encode()


def _response(body: dict[str, Any], path: str) -> httpx.Response:
    if path == "stream":
        return httpx.Response(200, content=_wire_frame(body))
    return httpx.Response(200, json=body)


@asynccontextmanager
async def _client(handler: Callable[..., Any]) -> AsyncIterator[OllamaClient]:
    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handler))
    try:
        yield client
    finally:
        await client.aclose()


async def _complete(body: dict[str, Any], path: str, *, model: str = _MODEL) -> Any:
    async with _client(lambda _request: _response(body, path)) as client:
        if path == "stream":
            return [chunk async for chunk in client.chat_stream(model=model, messages=_MESSAGES)]
        return await client.chat(model=model, messages=_MESSAGES)


class _WaitingBody(httpx.AsyncByteStream):
    """Expose one frame, then block in the provider read until cancellation."""

    def __init__(self, frame: dict[str, Any], waiting: asyncio.Event) -> None:
        self.frame = frame
        self.waiting = waiting

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _wire_frame(self.frame)
        self.waiting.set()
        await asyncio.Event().wait()


@pytest.mark.parametrize("path", ["chat", "stream"])
async def test_completion_reports_provider_usage(model_metrics: CollectorRegistry, path: str):
    body = {"done": True, "prompt_eval_count": 12, "eval_count": 4, "prompt_eval_cached_count": 7}
    result = await _complete(body, path)
    assert result == ([body] if path == "stream" else body)
    _assert_outcome(model_metrics, "ok")
    _assert_usage(model_metrics, {"input": 12, "output": 4, "cached_input": 7})


@pytest.mark.parametrize("path", ["chat", "stream"])
async def test_reported_zero_is_an_observation(model_metrics: CollectorRegistry, path: str):
    await _complete(
        {"done": True, "prompt_eval_count": 0, "eval_count": 0, "prompt_eval_cached_count": 0}, path,
    )
    _assert_outcome(model_metrics, "ok")
    _assert_usage(model_metrics, dict.fromkeys(_KINDS, 0))


@pytest.mark.parametrize("path", ["chat", "stream"])
@pytest.mark.parametrize("invalid", [None, True, False, "4", -1, 1.0, 2**63])
async def test_invalid_usage_stays_unknown(model_metrics: CollectorRegistry, path: str, invalid: Any):
    await _complete(
        {"done": True, "prompt_eval_count": invalid, "eval_count": invalid, "prompt_eval_cached_count": invalid},
        path,
    )
    _assert_outcome(model_metrics, "ok")
    _assert_usage(model_metrics, {})


@pytest.mark.parametrize("path", ["chat", "stream"])
async def test_missing_usage_is_not_a_measured_zero(model_metrics: CollectorRegistry, path: str):
    await _complete({"done": True}, path)
    _assert_outcome(model_metrics, "ok")
    _assert_usage(model_metrics, {})


@pytest.mark.parametrize("path", ["chat", "stream"])
async def test_valid_usage_fields_survive_an_invalid_sibling(model_metrics: CollectorRegistry, path: str):
    await _complete({"done": True, "prompt_eval_count": True, "eval_count": 3, "prompt_eval_cached_count": 2}, path)
    _assert_usage(model_metrics, {"output": 3, "cached_input": 2})


@pytest.mark.parametrize("path", ["chat", "stream"])
async def test_largest_supported_integer_is_observed(model_metrics: CollectorRegistry, path: str):
    largest = 2**63 - 1
    await _complete({"done": True, "prompt_eval_count": largest, "eval_count": 0, "prompt_eval_cached_count": largest}, path)
    _assert_usage(model_metrics, {"input": largest, "output": 0, "cached_input": largest})


@pytest.mark.parametrize("path", ["chat", "stream"])
async def test_cached_count_cannot_exceed_known_input(model_metrics: CollectorRegistry, path: str):
    await _complete({"done": True, "prompt_eval_count": 2, "eval_count": 3, "prompt_eval_cached_count": 4}, path)
    _assert_usage(model_metrics, {"input": 2, "output": 3})


async def test_usage_aggregates_by_model_and_kind_without_requiring_an_observer(model_metrics: CollectorRegistry):
    observed: list[str] = []
    await _complete({"done": True, "prompt_eval_count": 5, "eval_count": 2}, "chat")
    with observe_model_calls(observed.append):
        await _complete({"done": True, "prompt_eval_count": 7, "eval_count": 0}, "stream")
        await _complete({"done": True, "eval_count": 3}, "chat", model="router:latest")
    assert observed == [_MODEL, "router:latest"]
    _assert_outcome(model_metrics, "ok", count=2)
    _assert_outcome(model_metrics, "ok", model="router:latest")
    _assert_usage(model_metrics, {"input": 12, "output": 2}, observations=2)
    _assert_usage(model_metrics, {"output": 3}, model="router:latest")


@pytest.mark.parametrize("path", ["chat", "stream"])
@pytest.mark.parametrize("done", [_MISSING, None, False, 0, 1, "true"], ids=["missing", "null", "false", "zero", "one", "string"])
async def test_unconfirmed_completion_preserves_wire_result_but_is_error(
    model_metrics: CollectorRegistry, path: str, done: Any,
):
    body = {"message": {"content": "partial"}, "prompt_eval_count": 10, "eval_count": 3}
    if done is not _MISSING:
        body["done"] = done
    result = await _complete(body, path)
    assert result == ([body] if path == "stream" else body)
    _assert_outcome(model_metrics, "error")
    _assert_usage(model_metrics, {})


@pytest.mark.parametrize("path", ["chat", "stream"])
@pytest.mark.parametrize("done", [True, False])
async def test_provider_error_counts_once_and_only_terminal_usage_is_reported(
    model_metrics: CollectorRegistry, path: str, done: bool,
):
    body = {"error": "provider failed", "done": done, "prompt_eval_count": 8, "eval_count": 2}
    assert await _complete(body, path) == ([body] if path == "stream" else body)
    _assert_outcome(model_metrics, "error")
    _assert_usage(model_metrics, {"input": 8, "output": 2} if done else {})


async def test_duplicate_terminal_frames_do_not_double_count_or_include_partial_usage(model_metrics: CollectorRegistry):
    frames = [
        {"done": False, "eval_count": 100},
        {"done": True, "prompt_eval_count": 7, "eval_count": 4},
        {"done": True, "prompt_eval_count": 99, "eval_count": 99},
        {"error": "late frame"},
    ]
    async with _client(lambda _request: httpx.Response(200, content=b"".join(map(_wire_frame, frames)))) as client:
        assert [chunk async for chunk in client.chat_stream(model=_MODEL, messages=_MESSAGES)] == frames
    _assert_outcome(model_metrics, "ok")
    _assert_usage(model_metrics, {"input": 7, "output": 4})


@pytest.mark.parametrize("reason", ["stop", "length"])
@pytest.mark.parametrize("cleanup", ["close", "cancel"])
async def test_terminal_is_recorded_before_yield_and_consumer_cleanup_cannot_change_it(
    model_metrics: CollectorRegistry, monkeypatch: pytest.MonkeyPatch, reason: str, cleanup: str,
):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(ollama, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    waiting = asyncio.Event()
    terminal = {"done": True, "done_reason": reason, "eval_count": 3}

    def handler(_request: httpx.Request) -> httpx.Response:
        clock.now = 12.0
        return httpx.Response(200, stream=_WaitingBody(terminal, waiting))

    async with _client(handler) as client:
        stream = client.chat_stream(model=_MODEL, messages=_MESSAGES)
        assert await anext(stream) == terminal
        _assert_outcome(model_metrics, "ok")
        _assert_usage(model_metrics, {"output": 3})
        labels = {"model": _MODEL, "outcome": "ok"}
        assert model_metrics.get_sample_value("audrey_model_seconds_sum", labels) == 2.0
        clock.now = 900.0
        if cleanup == "cancel":
            task = asyncio.create_task(anext(stream))
            await waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await stream.aclose()
    _assert_outcome(model_metrics, "ok")
    _assert_usage(model_metrics, {"output": 3})
    assert model_metrics.get_sample_value("audrey_model_seconds_sum", labels) == 2.0


@pytest.mark.parametrize("path", ["chat", "stream"])
async def test_cancelling_a_pending_request_records_cancelled_once(model_metrics: CollectorRegistry, path: str):
    waiting = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("Cancelled request resumed")

    async with _client(handler) as client:
        stream = client.chat_stream(model=_MODEL, messages=_MESSAGES) if path == "stream" else None
        task = asyncio.create_task(anext(stream) if stream is not None else client.chat(model=_MODEL, messages=_MESSAGES))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if stream is not None:
            await stream.aclose()
    _assert_outcome(model_metrics, "cancelled")
    _assert_usage(model_metrics, {})


@pytest.mark.parametrize("cleanup", ["close", "cancel"])
async def test_partial_stream_close_or_cancellation_is_cancelled_and_usage_stays_unknown(
    model_metrics: CollectorRegistry, cleanup: str,
):
    waiting = asyncio.Event()
    partial = {"done": False, "message": {"content": "partial"}, "eval_count": 100}
    async with _client(lambda _request: httpx.Response(200, stream=_WaitingBody(partial, waiting))) as client:
        stream = client.chat_stream(model=_MODEL, messages=_MESSAGES)
        assert await anext(stream) == partial
        assert not any(metric.samples for metric in model_metrics.collect())
        if cleanup == "cancel":
            task = asyncio.create_task(anext(stream))
            await waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await stream.aclose()
    _assert_outcome(model_metrics, "cancelled")
    _assert_usage(model_metrics, {})


@pytest.mark.parametrize("path", ["chat", "stream"])
@pytest.mark.parametrize("failure", ["http", "transport", "json"])
async def test_http_transport_and_json_failures_count_once_without_inventing_usage(
    model_metrics: CollectorRegistry, path: str, failure: str,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "transport":
            raise httpx.ConnectError("unreachable provider", request=request)
        if failure == "http":
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, content=b"invalid JSON\n")

    async with _client(handler) as client:
        if path == "stream" and failure == "json":
            # Invalid NDJSON lines remain skipped; EOF still ends the iterator.
            assert [chunk async for chunk in client.chat_stream(model=_MODEL, messages=_MESSAGES)] == []
        else:
            with pytest.raises(OllamaError):
                if path == "stream":
                    _ = [chunk async for chunk in client.chat_stream(model=_MODEL, messages=_MESSAGES)]
                else:
                    await client.chat(model=_MODEL, messages=_MESSAGES)
    _assert_outcome(model_metrics, "error")
    _assert_usage(model_metrics, {})


async def test_midstream_transport_error_keeps_partial_frame_and_counts_error_once(model_metrics: CollectorRegistry):
    partial = {"done": False, "message": {"content": "partial"}, "eval_count": 100}

    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield _wire_frame(partial)
            raise httpx.ReadError("provider disconnected")

    async with _client(lambda _request: httpx.Response(200, stream=BrokenBody())) as client:
        stream = client.chat_stream(model=_MODEL, messages=_MESSAGES)
        assert await anext(stream) == partial
        with pytest.raises(OllamaError, match="transport error"):
            await anext(stream)
        await stream.aclose()
    _assert_outcome(model_metrics, "error")
    _assert_usage(model_metrics, {})


async def test_http_failure_is_recorded_before_error_body_read_and_survives_cancellation(
    model_metrics: CollectorRegistry, monkeypatch: pytest.MonkeyPatch,
):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(ollama, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    waiting = asyncio.Event()

    class WaitingErrorBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            waiting.set()
            await asyncio.Event().wait()
            yield b"unavailable"

    def handler(_request: httpx.Request) -> httpx.Response:
        clock.now = 12.0
        return httpx.Response(503, stream=WaitingErrorBody())

    async with _client(handler) as client:
        stream = client.chat_stream(model=_MODEL, messages=_MESSAGES)
        task = asyncio.create_task(anext(stream))
        await waiting.wait()
        _assert_outcome(model_metrics, "error")
        labels = {"model": _MODEL, "outcome": "error"}
        assert model_metrics.get_sample_value("audrey_model_seconds_sum", labels) == 2.0
        clock.now = 900.0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stream.aclose()
    _assert_outcome(model_metrics, "error")
    _assert_usage(model_metrics, {})
    assert model_metrics.get_sample_value("audrey_model_seconds_sum", labels) == 2.0
