"""Hermetic tests for `pipeline/memory.py`.

The module has two public surfaces: `recall_for_request` (does the
search) and `memory_system_message` (formats the system-message body).
Both have small surface areas with a handful of skip cases each — these
tests pin each branch.

Recall tests monkeypatch `dispatch_one` rather than wire up an httpx
mock; the dispatcher's own tests live in test_dispatch.py.
"""

from __future__ import annotations

import json

import pytest

from audrey.pipeline import memory as memory_mod
from audrey.pipeline.memory import (
    MAX_QUERY_CHARS,
    memory_system_message,
    recall_for_request,
)
from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.dispatch import ToolResult

# ─── Fixtures ─────────────────────────────────────────────────────────


def _registry_with_memory_search() -> ToolRegistry:
    r = ToolRegistry()
    r.by_name["memory_search"] = ToolSpec(
        name="memory_search",
        description="search memory",
        parameters={"type": "object", "properties": {}},
        server_url="http://server.test",
        path="/memory_search",
    )
    return r


def _result(content: str, *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        name="memory_search",
        call_id=None,
        content=content,
        elapsed_s=0.001,
        is_error=is_error,
    )


def _user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


# ─── recall_for_request skip cases ────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_skipped_when_user_id_empty():
    """No user → no recall. Direct-curl callers without auth land here."""
    out = await recall_for_request(
        _registry_with_memory_search(),
        user_id="",
        messages=[_user_msg("hello")],
    )
    assert out == []


@pytest.mark.asyncio
async def test_recall_skipped_when_registry_is_none():
    """custom-tools unreachable at boot → registry is None → no recall."""
    out = await recall_for_request(
        None,
        user_id="alice@example.com",
        messages=[_user_msg("hello")],
    )
    assert out == []


@pytest.mark.asyncio
async def test_recall_skipped_when_memory_search_not_in_registry():
    """Tools came up but memory_search isn't one of them → no recall."""
    out = await recall_for_request(
        ToolRegistry(),  # empty registry
        user_id="alice@example.com",
        messages=[_user_msg("hello")],
    )
    assert out == []


@pytest.mark.asyncio
async def test_recall_skipped_when_last_user_text_empty():
    """Empty user turn → no recall query to send."""
    out = await recall_for_request(
        _registry_with_memory_search(),
        user_id="alice@example.com",
        messages=[{"role": "user", "content": "   "}],
    )
    assert out == []


# ─── recall_for_request dispatch behavior ─────────────────────────────


@pytest.mark.asyncio
async def test_recall_returns_hits_on_success(monkeypatch):
    captured = {}

    async def fake_dispatch(http, registry, call, **kwargs):
        captured["args"] = call["function"]["arguments"]
        return _result(json.dumps({"results": [
            {"key": "favorite_color", "value": "blue"},
            {"key": "city", "value": "Portland"},
        ]}))

    monkeypatch.setattr(memory_mod, "dispatch_one", fake_dispatch)
    out = await recall_for_request(
        _registry_with_memory_search(),
        user_id="alice@example.com",
        messages=[_user_msg("what's my favorite color?")],
    )
    assert len(out) == 2
    assert out[0]["key"] == "favorite_color"
    # Verify the dispatcher was handed the user id, query, and top_k.
    assert captured["args"]["user"] == "alice@example.com"
    assert captured["args"]["query"] == "what's my favorite color?"
    assert captured["args"]["top_k"] == 3


@pytest.mark.asyncio
async def test_recall_clamps_long_query(monkeypatch):
    """Queries longer than MAX_QUERY_CHARS get clipped before dispatch.

    Long prompts dilute the embedding's signal, so the recall path
    truncates rather than sending the whole thing.
    """
    captured = {}

    async def fake_dispatch(http, registry, call, **kwargs):
        captured["query"] = call["function"]["arguments"]["query"]
        return _result(json.dumps({"results": []}))

    monkeypatch.setattr(memory_mod, "dispatch_one", fake_dispatch)
    long_text = "x" * (MAX_QUERY_CHARS + 100)
    await recall_for_request(
        _registry_with_memory_search(),
        user_id="alice@example.com",
        messages=[_user_msg(long_text)],
    )
    assert len(captured["query"]) == MAX_QUERY_CHARS


@pytest.mark.asyncio
async def test_recall_returns_empty_on_dispatch_error(monkeypatch):
    """Dispatcher returned an error result → no recall, no raise."""
    async def fake_dispatch(http, registry, call, **kwargs):
        return _result("upstream 500", is_error=True)

    monkeypatch.setattr(memory_mod, "dispatch_one", fake_dispatch)
    out = await recall_for_request(
        _registry_with_memory_search(),
        user_id="alice@example.com",
        messages=[_user_msg("hello")],
    )
    assert out == []


@pytest.mark.asyncio
async def test_recall_error_is_logged_at_warning_with_its_cost(monkeypatch, caplog):
    """A skip is best-effort, not free — it spends the whole 5s recall budget on
    the hot path of the request. It used to log at INFO with no timing, which is
    how a stalled embedder went a full eval campaign without being noticed.
    """
    async def fake_dispatch(http, registry, call, **kwargs):
        return _result("timeout in 5.00s", is_error=True)

    monkeypatch.setattr(memory_mod, "dispatch_one", fake_dispatch)
    with caplog.at_level("WARNING", logger="audrey.pipeline.memory"):
        out = await recall_for_request(
            _registry_with_memory_search(),
            user_id="alice@example.com",
            messages=[_user_msg("hello")],
        )
    assert out == []
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "recall skipped in " in msg
    assert "timeout in 5.00s" in msg


@pytest.mark.asyncio
async def test_recall_returns_empty_on_non_json_body(monkeypatch):
    """Tools-server returned 200 with garbage → degrade silently."""
    async def fake_dispatch(http, registry, call, **kwargs):
        return _result("<html>oops</html>")

    monkeypatch.setattr(memory_mod, "dispatch_one", fake_dispatch)
    out = await recall_for_request(
        _registry_with_memory_search(),
        user_id="alice@example.com",
        messages=[_user_msg("hello")],
    )
    assert out == []


@pytest.mark.asyncio
async def test_recall_returns_empty_when_results_not_list(monkeypatch):
    """Body parsed but `results` isn't a list → treat as no hits."""
    async def fake_dispatch(http, registry, call, **kwargs):
        return _result(json.dumps({"results": "oops, a string"}))

    monkeypatch.setattr(memory_mod, "dispatch_one", fake_dispatch)
    out = await recall_for_request(
        _registry_with_memory_search(),
        user_id="alice@example.com",
        messages=[_user_msg("hello")],
    )
    assert out == []


# ─── memory_system_message formatting ─────────────────────────────────


def test_memory_system_message_returns_none_when_empty():
    """No hits and no store-hint request → nothing to inject."""
    assert memory_system_message([], user_id="alice@example.com") is None
    assert memory_system_message([]) is None


def test_memory_system_message_with_hits_only():
    msg = memory_system_message(
        [{"key": "city", "value": "Portland"}],
        user_id="",
    )
    assert msg is not None
    assert msg["role"] == "system"
    assert "Portland" in msg["content"]
    assert "(city)" in msg["content"]


def test_memory_system_message_truncates_long_values():
    """Per-hit value is capped at 400 chars + ellipsis so a single
    runaway memory can't crowd out the prompt."""
    long_value = "x" * 1000
    msg = memory_system_message(
        [{"key": "k", "value": long_value}],
        user_id="",
    )
    assert msg is not None
    # Original value gone; truncated form present.
    assert long_value not in msg["content"]
    assert "…" in msg["content"]


def test_memory_system_message_substitutes_user_id_in_store_hint():
    """The {user_id} placeholder must be replaced with the real id —
    otherwise the model writes back with a literal '{user_id}' tag."""
    msg = memory_system_message(
        [],
        user_id="alice@example.com",
        include_store_hint=True,
    )
    assert msg is not None
    assert "alice@example.com" in msg["content"]
    assert "{user_id}" not in msg["content"]


def test_memory_system_message_drops_store_hint_when_no_user_id():
    """No user → no point telling the model to call memory_store; the
    dispatcher would reject the call anyway."""
    msg = memory_system_message(
        [],
        user_id="",
        include_store_hint=True,
    )
    assert msg is None


def test_memory_system_message_combines_hits_and_store_hint():
    """Hits and store-hint compose into one system message, hits first."""
    msg = memory_system_message(
        [{"key": "city", "value": "Portland"}],
        user_id="alice@example.com",
        include_store_hint=True,
    )
    assert msg is not None
    body = msg["content"]
    # Hits block lands before the store-hint block.
    assert body.index("Portland") < body.index("alice@example.com")
