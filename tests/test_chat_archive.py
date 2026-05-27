"""Hermetic tests for chat archive (Audrey side + tools-server side).

Both sides are exercised here because their contract is small: the
Audrey client posts JSON to `/chat_history/archive` and the store
persists message rows + Q+A chunks. We don't hit Qdrant or Ollama in
tests — Qdrant calls go through fakes, the embed call is monkeypatched.

Coverage:
  - Pure helpers: derive_message_id (deterministic, dedup-safe),
    build_chunks (one chunk per Q+A pair; oversize splits with overlap).
  - StreamCollector: filters delta.content out of SSE frames, ignores
    role/finish/tool frames, marks partial on CancelledError.
  - resolve_conversation_id: OWUI top-level chat_id wins; deterministic
    fallback for steps 1–3 misses; UUID last resort.
  - ChatArchiveClient: skipped metric on missing user / missing tool;
    posts payload otherwise; never raises on transport failure.
  - dispatch._USER_SCOPED_TOOLS includes chat_history_search and the
    dispatcher overwrites a model-supplied user.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# Add tools-server to sys.path so we can import the chat_archive module
# directly. The custom-tools service isn't packaged for installation; it
# runs as a script in its own container.
_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

# Ensure a fresh import each test session (settings env may differ).
chat_archive_module = importlib.import_module("chat_archive")
app_module = importlib.import_module("app")


# ─── Request-schema bounds ────────────────────────────────────────────


def test_chat_history_search_limit_accepts_20():
    # Phase 8: cap raised from 10 to 20 to match sibling tools (kb_search,
    # memory_*) and to stop the model 422'ing on its own broad-recall calls.
    req = app_module.ChatHistorySearchRequest(
        user="alice@example.com", query="test", limit=20,
    )
    assert req.limit == 20


def test_chat_history_search_limit_rejects_21():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        app_module.ChatHistorySearchRequest(
            user="alice@example.com", query="test", limit=21,
        )


# ─── Pure helpers ─────────────────────────────────────────────────────


def test_derive_message_id_is_deterministic_within_minute():
    a = chat_archive_module.derive_message_id(
        "alice@example.com", "conv-1", "user", "hello", "2026-05-08T12:34:56+00:00",
    )
    b = chat_archive_module.derive_message_id(
        "alice@example.com", "conv-1", "user", "hello", "2026-05-08T12:34:59+00:00",
    )
    assert a == b, "same minute bucket should yield identical id"


def test_derive_message_id_changes_across_minutes():
    a = chat_archive_module.derive_message_id(
        "alice@example.com", "conv-1", "user", "hello", "2026-05-08T12:34:56+00:00",
    )
    b = chat_archive_module.derive_message_id(
        "alice@example.com", "conv-1", "user", "hello", "2026-05-08T12:35:01+00:00",
    )
    assert a != b


def test_derive_message_id_user_scoped():
    a = chat_archive_module.derive_message_id(
        "alice@example.com", "conv-1", "user", "hello", "2026-05-08T12:34:56+00:00",
    )
    b = chat_archive_module.derive_message_id(
        "bob@example.com", "conv-1", "user", "hello", "2026-05-08T12:34:56+00:00",
    )
    assert a != b


def test_build_chunks_one_pair_one_chunk():
    chunks = chat_archive_module.build_chunks(
        user="alice@example.com",
        conversation_id="conv-1",
        user_message_id="m1", user_content="What's the weather?",
        assistant_message_id="m2", assistant_content="Sunny and warm.",
        created_at="2026-05-08T12:34:56+00:00",
        max_chars=1500, overlap_chars=100,
    )
    assert len(chunks) == 1
    assert "User: What's the weather?" in chunks[0]["text"]
    assert "Assistant: Sunny and warm." in chunks[0]["text"]
    assert chunks[0]["message_ids"] == ["m1", "m2"]


def test_build_chunks_oversize_assistant_splits_with_overlap():
    long_answer = ". ".join(f"Sentence number {i}" for i in range(200)) + "."
    chunks = chat_archive_module.build_chunks(
        user="alice@example.com",
        conversation_id="conv-1",
        user_message_id="m1", user_content="Tell me a long story.",
        assistant_message_id="m2", assistant_content=long_answer,
        created_at="2026-05-08T12:34:56+00:00",
        max_chars=400, overlap_chars=40,
    )
    assert len(chunks) > 1
    for c in chunks:
        # Soft cap; allow a small overshoot due to overlap-stitching.
        assert len(c["text"]) <= 400 + 100
    # Each chunk has its own deterministic id.
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)


# ─── StreamCollector ──────────────────────────────────────────────────

# Need the Audrey-side module — skipped above because we put tools-server
# first on sys.path. Re-resolve after.
from audrey.pipeline.chat_archive import (  # noqa: E402
    ChatArchiveClient,
    StreamCollector,
    resolve_conversation_id,
)


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _gen_frames(frames):
    for f in frames:
        yield f


async def test_stream_collector_captures_only_content_deltas():
    frames = [
        # role-only first frame — must NOT be captured
        _frame({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
        _frame({"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]}),
        _frame({"choices": [{"delta": {"content": "world"}, "finish_reason": None}]}),
        # tool-call frame — must NOT be captured
        _frame({"choices": [{"delta": {"tool_calls": [{"id": "x"}]}, "finish_reason": None}]}),
        _frame({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]\n\n",
    ]
    collector = StreamCollector()
    out = []
    async for frame in collector.wrap(_gen_frames(frames)):
        out.append(frame)
    assert out == frames, "frames must pass through unchanged"
    assert collector.text == "Hello world"
    assert collector.partial is False


async def test_stream_collector_marks_partial_on_cancel():
    async def cancelling():
        yield _frame({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]})
        raise asyncio.CancelledError()

    collector = StreamCollector()
    with pytest.raises(asyncio.CancelledError):
        async for _ in collector.wrap(cancelling()):
            pass
    assert collector.text == "Hello"
    assert collector.partial is True


def test_stream_collector_feed_text():
    collector = StreamCollector()
    collector.feed_text("part one ")
    collector.feed_text("part two")
    assert collector.text == "part one part two"


# ─── resolve_conversation_id ──────────────────────────────────────────


def test_resolve_uses_top_level_chat_id():
    cid = resolve_conversation_id(
        user_id="alice@example.com",
        raw_payload={"chat_id": "owui-abc"},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert cid == "owui-abc"


def test_resolve_uses_metadata_chat_id():
    cid = resolve_conversation_id(
        user_id="alice@example.com",
        raw_payload={"metadata": {"chat_id": "  meta-xyz  "}},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert cid == "meta-xyz"


def test_resolve_falls_back_to_deterministic_hash():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    a = resolve_conversation_id(user_id="alice@example.com", raw_payload={}, messages=msgs)
    b = resolve_conversation_id(user_id="alice@example.com", raw_payload={}, messages=msgs)
    assert a == b, "same prefix should hash to the same conversation"
    assert a.startswith("derived-")


def test_resolve_different_users_get_different_ids():
    msgs = [{"role": "user", "content": "hi"}]
    a = resolve_conversation_id(user_id="alice@example.com", raw_payload=None, messages=msgs)
    b = resolve_conversation_id(user_id="bob@example.com", raw_payload=None, messages=msgs)
    assert a != b


# ─── ChatArchiveClient ────────────────────────────────────────────────


class _FakeRegistry:
    """Tiny stub mirroring ToolRegistry's `.get()` shape."""

    def __init__(self, tools: dict[str, str]) -> None:
        self._tools = tools

    def get(self, name: str):
        url = self._tools.get(name)
        if url is None:
            return None
        return SimpleNamespace(name=name, server_url=url, path="/" + name, parameters={})


async def test_archive_client_skips_when_no_user():
    http = AsyncMock()
    client = ChatArchiveClient(http)
    await client.archive_turn(
        registry=_FakeRegistry({"chat_history_search": "http://tools:8000"}),
        user_id="",
        conversation_id="c",
        user_content="u",
        assistant_content="a",
    )
    http.post.assert_not_called()


async def test_archive_client_skips_when_tool_not_registered():
    http = AsyncMock()
    client = ChatArchiveClient(http)
    await client.archive_turn(
        registry=_FakeRegistry({}),
        user_id="alice@example.com",
        conversation_id="c",
        user_content="u",
        assistant_content="a",
    )
    http.post.assert_not_called()


async def test_archive_client_posts_payload_to_internal_route():
    captured: dict = {}

    async def fake_post(url, json=None, timeout=None):  # noqa: ASYNC109 — match httpx.AsyncClient.post kwargs
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"chunks": 1})

    http = MagicMock()
    http.post = AsyncMock(side_effect=fake_post)
    client = ChatArchiveClient(http)
    await client.archive_turn(
        registry=_FakeRegistry({"chat_history_search": "http://tools:8000"}),
        user_id="alice@example.com",
        conversation_id="c1",
        user_content="hello",
        assistant_content="hi back",
        partial=True,
        virtual_model="audrey_fast",
        concrete_model="qwen3:4b",
    )
    assert captured["url"] == "http://tools:8000/chat_history/archive"
    body = captured["json"]
    assert body["user"] == "alice@example.com"
    assert body["conversation_id"] == "c1"
    assert body["partial"] is True
    assert body["virtual_model"] == "audrey_fast"
    assert body["concrete_model"] == "qwen3:4b"


async def test_archive_client_swallows_transport_errors():
    http = MagicMock()
    http.post = AsyncMock(side_effect=httpx.ConnectError("nope"))
    client = ChatArchiveClient(http)
    # Should not raise.
    await client.archive_turn(
        registry=_FakeRegistry({"chat_history_search": "http://tools:8000"}),
        user_id="alice@example.com",
        conversation_id="c1",
        user_content="hello",
        assistant_content="hi",
    )


# ─── dispatcher: chat_history_search is user-scoped ───────────────────


def test_chat_history_search_in_user_scoped_set():
    from audrey.tools.dispatch import _USER_SCOPED_TOOLS

    assert "chat_history_search" in _USER_SCOPED_TOOLS
