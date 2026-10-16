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
  - the chat_history_search capability policy binds the authenticated user and
    the dispatcher overwrites a model-supplied user.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest
from fastapi.testclient import TestClient

# Add tools-server to sys.path so we can import the chat_archive module
# directly. The custom-tools service isn't packaged for installation; it
# runs as a script in its own container.
_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)
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


def test_derive_message_id_uses_stable_archive_id_across_retry_minutes():
    a = chat_archive_module.derive_message_id(
        "alice@example.com",
        "conv-1",
        "user",
        "hello",
        "2026-05-08T12:34:56+00:00",
        "delivery-1",
    )
    b = chat_archive_module.derive_message_id(
        "alice@example.com",
        "conv-1",
        "user",
        "hello",
        "2026-05-08T13:45:01+00:00",
        "delivery-1",
    )
    assert a == b


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


def test_resolve_uses_explicit_client_conversation_id():
    cid = resolve_conversation_id(
        user_id="alice@example.com",
        raw_payload={"conversation_id": " client-thread-7 "},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert cid == "client-thread-7"


def test_resolve_uses_validated_message_metadata():
    cid = resolve_conversation_id(
        user_id="alice@example.com",
        raw_payload={},
        messages=[{
            "role": "user",
            "content": "hi",
            "metadata": {"conversation_id": "message-thread-3"},
        }],
    )
    assert cid == "message-thread-3"


def test_resolve_falls_back_to_deterministic_hash():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    a = resolve_conversation_id(user_id="alice@example.com", raw_payload={}, messages=msgs)
    b = resolve_conversation_id(user_id="alice@example.com", raw_payload={}, messages=msgs)
    assert a == b, "same prefix should hash to the same conversation"
    assert a.startswith("derived-")


def test_resolve_fallback_stays_stable_as_history_grows():
    first_turn = [{"role": "user", "content": "start this thread"}]
    growing = [
        *first_turn,
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "second answer"},
    ]
    initial = resolve_conversation_id(
        user_id="alice@example.com",
        raw_payload={},
        messages=first_turn,
    )
    later = resolve_conversation_id(
        user_id="alice@example.com",
        raw_payload={},
        messages=growing,
    )
    assert initial == later


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

    async def fake_post(url, json=None, timeout=None, headers=None):  # noqa: ASYNC109 — match httpx.AsyncClient.post kwargs
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return httpx.Response(200, json={"chunks": 1})

    http = MagicMock()
    http.post = AsyncMock(side_effect=fake_post)
    client = ChatArchiveClient(http, service_token=SECRET)
    await client.archive_turn(
        registry=_FakeRegistry({"chat_history_search": "http://tools:8000"}),
        user_id="alice@example.com",
        conversation_id="c1",
        user_content="hello",
        assistant_content="hi back",
        partial=True,
        virtual_model="audrey_fast",
        concrete_model="qwen3:4b",
        archive_id="delivery-1",
        created_at="2026-08-28T12:00:00.000000+00:00",
    )
    assert captured["url"] == "http://tools:8000/chat_history/archive"
    body = captured["json"]
    assert body["user"] == "alice@example.com"
    assert body["conversation_id"] == "c1"
    assert body["partial"] is True
    assert body["virtual_model"] == "audrey_fast"
    assert body["concrete_model"] == "qwen3:4b"
    assert body["archive_id"] == "delivery-1"
    assert captured["headers"] == {
        "X-Audrey-Service-Token": SECRET,
    }
    assert body["created_at"] == "2026-08-28T12:00:00.000000+00:00"


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


def test_chat_history_search_has_argument_scope_policy():
    from audrey.tools.discovery import TOOL_DECLARATIONS, ToolUserScope

    policy = TOOL_DECLARATIONS["chat_history_search"]
    assert policy.user_scope is ToolUserScope.ARGUMENT


# A failed embed must never look like a successful search with no matches.
async def test_history_embedding_failure_is_not_an_empty_result(monkeypatch):
    store = chat_archive_module.ChatArchiveStore.__new__(chat_archive_module.ChatArchiveStore)
    store.user_purge_cutoff = AsyncMock(return_value="")
    store._keyword_hits = AsyncMock(return_value=[])
    store._http = MagicMock()
    store._qdrant = SimpleNamespace(query_points=AsyncMock())
    store._embed_model = "test-embedding-model"
    store._embed_keep_alive = "24h"
    cause = chat_archive_module._EmbedError("Embedding request failed (TimeoutError)")
    monkeypatch.setattr(chat_archive_module, "_embed", AsyncMock(side_effect=cause))

    with pytest.raises(chat_archive_module.ChatHistorySearchUnavailableError) as caught:
        await store.search(user="alice@example.com", query="asked to calculate", limit=5)

    assert caught.value.__cause__ is cause
    store._qdrant.query_points.assert_not_awaited()


@pytest.fixture
def post_history(monkeypatch):
    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    monkeypatch.setattr(app_module.app.router, "lifespan_context", noop_lifespan)
    monkeypatch.setattr(
        app_module.app.state, "capabilities",
        app_module.CapabilityRegistry.all_available(), raising=False,
    )

    def post(archive, body):
        monkeypatch.setattr(app_module.app.state, "chat_archive", archive, raising=False)
        with TestClient(app_module.app) as client:
            return client.post("/chat_history_search", json=body)

    return post


def test_history_search_unavailable_returns_safe_503(post_history):
    archive = SimpleNamespace(search=AsyncMock(side_effect=(
        chat_archive_module.ChatHistorySearchUnavailableError("private-provider-response")
    )))
    response = post_history(archive, {"user": "alice@example.com", "query": "calculation"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Chat history search is temporarily unavailable (embedder). Try again later.",
    }
    assert "private-provider-response" not in response.text
    assert "results" not in response.json()


def test_history_search_genuine_no_match_remains_success(post_history):
    archive = SimpleNamespace(search=AsyncMock(return_value=[]))
    response = post_history(archive, {"user": "alice@example.com", "query": "calculation"})

    assert response.status_code == 200
    assert response.json() == {"query": "calculation", "results": []}
    archive.search.assert_awaited_once_with(
        user="alice@example.com", query="calculation", limit=5,
        date_from=None, date_to=None,
    )


def test_history_search_success_preserves_hits_and_requested_scope(post_history):
    archive = SimpleNamespace(search=AsyncMock(return_value=[
        chat_archive_module.SearchHit(
            conversation_id="conversation-1", chunk_id="chunk-1",
            created_at="2026-10-15T12:00:00+00:00",
            snippet="User: Calculate 1250 × 1.08 using the calculator", score=0.8,
        ),
    ]))
    response = post_history(archive, {
        "user": "alice@example.com", "query": "calculation", "limit": 2,
        "date_from": "2026-10-15T00:00:00+00:00",
    })

    assert response.status_code == 200
    assert response.json()["results"][0]["conversation_id"] == "conversation-1"
    assert response.json()["results"][0]["snippet"].startswith("User: Calculate 1250")
    archive.search.assert_awaited_once_with(
        user="alice@example.com", query="calculation", limit=2,
        date_from="2026-10-15T00:00:00+00:00", date_to=None,
    )


@pytest.fixture
async def history_store(tmp_path, monkeypatch):
    """Real archive schema with mocked vector and embedding transports."""
    store = chat_archive_module.ChatArchiveStore.__new__(chat_archive_module.ChatArchiveStore)
    store._db = await aiosqlite.connect(tmp_path / "history.db")
    await store._db.executescript(chat_archive_module._SCHEMA)
    store._db_lock = asyncio.Lock()
    store._retention_days = 0
    store._http = None
    store._embed_model = "test-embedding-model"
    store._embed_keep_alive = "24h"
    store._collection = "history"
    store._threshold = 0.62
    store._qdrant = SimpleNamespace(query_points=AsyncMock(
        return_value=SimpleNamespace(points=[]),
    ))
    monkeypatch.setattr(chat_archive_module, "_embed", AsyncMock(return_value=[1.0, 0.0]))
    try:
        yield store
    finally:
        await store._db.close()


async def save_history_chunk(
    store, chunk_id, text, *, user="alice@example.com", conversation_id="calculation",
    created_at=None, indexed_at=None,
):
    await store._db.execute(
        "INSERT INTO archive_chunks "
        "(chunk_id, conversation_id, user, message_ids_json, text, created_at, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chunk_id, conversation_id, user, f"message-{chunk_id}", text,
         created_at or chat_archive_module._now_iso(), indexed_at),
    )
    await store._db.commit()


@pytest.mark.parametrize("query", [
    "calculate", "asked to calculate", "what did I ask you to calculate",
])
async def test_literal_history_finds_actual_unindexed_calculation_below_dense_cutoff(history_store, query):
    await save_history_chunk(
        history_store, "actual-calculation",
        "User: Calculate 1250 × 1.08 using the calculator\nAssistant: 1350",
    )
    await save_history_chunk(
        history_store, "later-question",
        "User: Use chat history search to tell me what I asked you to calculate\n"
        "Assistant: I could not find the calculation.",
    )
    hits = await history_store.search(user="alice@example.com", query=query, limit=5)

    assert {hit.chunk_id for hit in hits} == {"actual-calculation", "later-question"}
    actual = next(hit for hit in hits if hit.chunk_id == "actual-calculation")
    assert "1250 × 1.08" in actual.snippet
    assert actual.score is None and actual.match_type == "keyword"
    assert history_store._qdrant.query_points.await_args.kwargs["score_threshold"] == 0.62


@pytest.mark.parametrize("query", ["foo", "please tell me what I asked before", "unrelated"])
async def test_literal_history_ignores_substrings_fillers_and_unrelated_queries(history_store, query):
    await save_history_chunk(history_store, "foo-substring", "User: Discuss foobar\nAssistant: Details")
    hits = await history_store.search(user="alice@example.com", query=query, limit=5)
    assert hits == []


async def test_literal_history_requires_all_informative_terms_and_bounds_results(history_store):
    await save_history_chunk(history_store, "only-one", "User: calculate\nAssistant: Done")
    await save_history_chunk(history_store, "both-a", "User: calculate invoice\nAssistant: Done")
    await save_history_chunk(history_store, "both-b", "User: calculate invoice\nAssistant: Done")
    hits = await history_store.search(user="alice@example.com", query="calculate invoice", limit=1)
    assert len(hits) == 1
    assert hits[0].chunk_id in {"both-a", "both-b"}


@pytest.mark.parametrize(
    "gate", ["foreign", "date_from", "date_to", "purge", "conversation_delete", "outbox", "retention"],
)
async def test_history_visibility_gates_keyword_and_semantic_hits_independently(history_store, gate):
    now = chat_archive_module._dt.datetime.now(chat_archive_module._dt.UTC)
    hour = chat_archive_module._dt.timedelta(hours=1)
    visible_at = now - hour
    blocked_at = now - 2 * hour
    kwargs = {}
    if gate == "retention":
        blocked_at = now - chat_archive_module._dt.timedelta(days=2)
        history_store._retention_days = 1
    if gate == "date_from":
        kwargs["date_from"] = (now - 1.5 * hour).isoformat()
    if gate == "date_to":
        blocked_at = now
        kwargs["date_to"] = (now - 0.5 * hour).isoformat()
    await save_history_chunk(
        history_store, "visible", "User: calculate invoice\nAssistant: 1350",
        conversation_id="visible-conversation", created_at=visible_at.isoformat(),
    )
    await save_history_chunk(
        history_store, "blocked", "User: calculate invoice\nAssistant: PRIVATE_BLOCKED",
        user="bob@example.com" if gate == "foreign" else "alice@example.com",
        conversation_id="blocked-conversation", created_at=blocked_at.isoformat(),
    )
    cutoff = (now - 1.5 * hour).isoformat()
    if gate == "purge":
        await history_store._db.execute(
            "INSERT INTO archive_user_purges "
            "(purge_id, user, cutoff_at, requested_at) VALUES (?, ?, ?, ?)",
            ("purge", "alice@example.com", cutoff, now.isoformat()),
        )
    if gate == "conversation_delete":
        await history_store._db.execute(
            "INSERT INTO archive_conversation_deletions "
            "(user, conversation_id, cutoff_at, requested_at) VALUES (?, ?, ?, ?)",
            ("alice@example.com", "blocked-conversation", cutoff, now.isoformat()),
        )
    if gate == "outbox":
        await history_store._db.execute(
            "INSERT INTO archive_deletion_outbox "
            "(chunk_id, conversation_id, message_ids_json, point_id, requested_at) VALUES (?, ?, ?, ?, ?)",
            ("blocked", "blocked-conversation", "message-blocked", "point", now.isoformat()),
        )
    await history_store._db.commit()
    # Even stale vector results cannot bypass the authoritative source gates.
    history_store._qdrant.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(payload={"chunk_id": "blocked"}, score=0.95),
        SimpleNamespace(payload={"chunk_id": "visible"}, score=0.9),
    ])
    hits = await history_store.search(
        user="alice@example.com", query="calculate invoice", limit=20, **kwargs,
    )
    assert [hit.chunk_id for hit in hits] == ["visible"]
    assert all("PRIVATE_BLOCKED" not in hit.snippet for hit in hits)


async def test_literal_history_recovers_matches_when_embedding_is_unavailable(history_store, monkeypatch):
    await save_history_chunk(history_store, "calculation", "User: calculate 1250 × 1.08\nAssistant: 1350")
    monkeypatch.setattr(chat_archive_module, "_embed", AsyncMock(
        side_effect=chat_archive_module._EmbedError("timeout"),
    ))
    hits = await history_store.search(user="alice@example.com", query="calculate", limit=5)
    assert [hit.chunk_id for hit in hits] == ["calculation"]
    assert hits[0].score is None
    with pytest.raises(chat_archive_module.ChatHistorySearchUnavailableError):
        await history_store.search(user="alice@example.com", query="unrelated", limit=5)


async def test_history_dense_fallback_preserves_cosine_and_deduplicates_literal_matches(history_store):
    await save_history_chunk(history_store, "semantic", "User: invoice total\nAssistant: 1350")
    await save_history_chunk(history_store, "literal", "User: calculate\nAssistant: 1350")
    history_store._qdrant.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(payload={"chunk_id": "literal"}, score=0.9),
        SimpleNamespace(payload={"chunk_id": "semantic"}, score=0.8),
        SimpleNamespace(payload={"chunk_id": "unknown"}, score=0.9),
    ])
    hits = await history_store.search(user="alice@example.com", query="calculate", limit=5)
    assert [(hit.chunk_id, hit.match_type, hit.score) for hit in hits] == [
        ("literal", "keyword", None), ("semantic", "semantic", 0.8),
    ]
    # A different query with no literal matches retains semantic fallback.
    hits = await history_store.search(user="alice@example.com", query="arithmetic", limit=5)
    assert [hit.match_type for hit in hits] == ["semantic", "semantic"]


async def test_literal_history_snippet_shows_a_late_matching_passage(history_store):
    await save_history_chunk(
        history_store, "late-match", "Unrelated setup. " * 70 + "Calculate 1250 × 1.08 = 1350.",
    )
    hits = await history_store.search(user="alice@example.com", query="calculate", limit=5, snippet_chars=100)
    assert "Calculate 1250" in hits[0].snippet
    assert len(hits[0].snippet) <= 100


@pytest.mark.parametrize("embedding_failure", [False, True])
async def test_literal_history_rechecks_visibility_after_embedding_await(history_store, monkeypatch, embedding_failure):
    await save_history_chunk(history_store, "deleted-during-search", "User: calculate\nAssistant: 1350")

    async def embed_then_delete(*_args, **_kwargs):
        await history_store._db.execute(
            "INSERT INTO archive_conversation_deletions "
            "(user, conversation_id, cutoff_at, requested_at) VALUES (?, ?, ?, ?)",
            ("alice@example.com", "calculation", chat_archive_module._now_iso(), chat_archive_module._now_iso()),
        )
        await history_store._db.commit()
        if embedding_failure:
            raise chat_archive_module._EmbedError("timeout")
        return [1.0, 0.0]

    monkeypatch.setattr(chat_archive_module, "_embed", embed_then_delete)
    if embedding_failure:
        # The only literal hit disappeared; do not claim the unavailable
        # semantic search found nothing either.
        with pytest.raises(chat_archive_module.ChatHistorySearchUnavailableError):
            await history_store.search(user="alice@example.com", query="calculate", limit=5)
    else:
        hits = await history_store.search(user="alice@example.com", query="calculate", limit=5)
        assert hits == []


def test_history_route_keyword_match_has_null_score(post_history):
    archive = SimpleNamespace(search=AsyncMock(return_value=[chat_archive_module.SearchHit(
        conversation_id="calculation", chunk_id="chunk", created_at="2026-10-15T12:00:00+00:00",
        snippet="Calculate 1250 × 1.08", score=None, match_type="keyword",
    )]))
    response = post_history(archive, {"user": "alice@example.com", "query": "calculate"})
    assert response.status_code == 200
    assert response.json()["results"][0]["score"] is None
    assert response.json()["results"][0]["match_type"] == "keyword"
