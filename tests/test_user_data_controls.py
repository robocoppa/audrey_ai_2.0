"""Authenticated inventory/export contracts for Campaign 3 Wave 1C.5."""

from __future__ import annotations

import importlib
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from audrey.auth import AuthedUser, require_user
from audrey.routes.user_data import export_chat_history, list_memories, router

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

archive_module = importlib.import_module("chat_archive")
memory_module = importlib.import_module("db")
tools_app_module = importlib.import_module("app")


class _MemoryQdrant:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.next_offset = uuid.uuid4()

    async def scroll(self, **kwargs):
        self.calls.append(kwargs)
        return (
            [
                SimpleNamespace(payload={
                    "key": "theme",
                    "value": "dark",
                    "tags": "user:alice@example.com,topic:preferences",
                    "created_at": "created",
                    "updated_at": "updated",
                })
            ],
            self.next_offset,
        )


async def test_memory_inventory_is_user_filtered_and_hides_scope_tag():
    qdrant = _MemoryQdrant()
    store = memory_module.MemoryStore.__new__(memory_module.MemoryStore)
    store._qdrant = qdrant
    store._collection = "kb_memory"

    items, cursor = await store.list_user(user="alice@example.com", limit=25)

    assert [item.key for item in items] == ["theme"]
    assert items[0].tags == "topic:preferences"
    assert cursor == str(qdrant.next_offset)
    call = qdrant.calls[0]
    condition = call["scroll_filter"].must[0]
    assert condition.key == "user"
    assert condition.match.value == "alice@example.com"
    assert call["limit"] == 25


async def test_memory_inventory_rejects_non_uuid_cursor():
    store = memory_module.MemoryStore.__new__(memory_module.MemoryStore)
    store._qdrant = _MemoryQdrant()
    store._collection = "kb_memory"

    with pytest.raises(ValueError, match="invalid memory cursor"):
        await store.list_user(user="alice@example.com", cursor="not-a-cursor")


class _NoopQdrant:
    async def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name="archive-test")],
        )

    async def close(self) -> None:
        pass


class _NoopHttp:
    async def aclose(self) -> None:
        pass


async def _archive_store(path: Path):
    store = archive_module.ChatArchiveStore.__new__(archive_module.ChatArchiveStore)
    store._sqlite_path = path
    store._qdrant = _NoopQdrant()
    store._http = _NoopHttp()
    store._collection = "archive-test"
    store._embed_model = "unused"
    store._embed_dim = 3
    store._chunk_max = 2500
    store._chunk_overlap = 100
    store._threshold = 0.0
    store._retention_days = 0
    store._embed_keep_alive = ""
    store._max_bytes = 0
    store._repair_batch_size = 20
    store._max_retry_attempts = 3
    import asyncio

    store._db = None
    store._db_lock = asyncio.Lock()
    store._qdrant_write_lock = asyncio.Lock()
    store._maintenance_lock = asyncio.Lock()
    await store.init()
    return store


async def _insert_archive_rows(store) -> None:
    assert store._db is not None
    async with store._db_lock:
        await store._db.executemany(
            """
            INSERT INTO conversations
                (conversation_id, user, title, created_at, updated_at, last_message_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("a", "alice@example.com", "Alice chat", "1", "2", "2"),
                ("b", "bob@example.com", "Bob chat", "1", "2", "2"),
            ],
        )
        await store._db.executemany(
            """
            INSERT INTO messages
                (message_id, conversation_id, user, role, content,
                 created_at, archived_at, partial)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("a1", "a", "alice@example.com", "user", "first", "1", "1", 0),
                ("a2", "a", "alice@example.com", "assistant", "second", "2", "2", 0),
                ("a3", "a", "alice@example.com", "user", "pending", "3", "3", 0),
                ("b1", "b", "bob@example.com", "user", "private", "1", "1", 0),
            ],
        )
        await store._db.execute(
            """
            INSERT INTO archive_deletion_outbox
                (chunk_id, conversation_id, message_ids_json, point_id, requested_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("pending-chunk", "a", "a3", "point", "now"),
        )
        await store._db.commit()


async def test_chat_export_pages_one_user_and_hides_pending_deletion(tmp_path: Path):
    store = await _archive_store(tmp_path / "archive.db")
    try:
        await _insert_archive_rows(store)
        first, cursor = await store.export_user_messages(
            user="alice@example.com", limit=1,
        )
        second, final_cursor = await store.export_user_messages(
            user="alice@example.com", limit=1, cursor=cursor,
        )

        assert [item.message_id for item in first] == ["a1"]
        assert [item.message_id for item in second] == ["a2"]
        assert first[0].conversation_title == "Alice chat"
        assert final_cursor is None
        assert "pending" not in [item.content for item in [*first, *second]]
        assert "private" not in [item.content for item in [*first, *second]]
    finally:
        await store.aclose()


def _request(*, tool: str, response: dict | None = None):
    http = SimpleNamespace(post=AsyncMock(return_value=httpx.Response(200, json=response)))
    registry = SimpleNamespace(
        get=lambda name: SimpleNamespace(server_url="http://custom-tools:8001")
        if name == tool
        else None,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                tools=registry, archive_http=http, kb_service_token=SECRET,
            ),
        ),
    )
    return request, http


async def test_memory_route_injects_authenticated_user_not_browser_identity():
    request, http = _request(
        tool="memory_search",
        response={"items": [], "next_cursor": None},
    )
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    await list_memories(request, limit=10, cursor=None, me=me)

    assert http.post.await_args.kwargs["json"] == {
        "user": "alice@example.com",
        "limit": 10,
    }
    assert http.post.await_args.kwargs["headers"] == {
        "X-Audrey-Service-Token": SECRET,
    }


async def test_chat_route_injects_authenticated_user():
    request, http = _request(
        tool="chat_history_search",
        response={"schema_version": 1, "items": [], "next_cursor": None},
    )
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    await export_chat_history(request, limit=50, cursor="next", me=me)

    assert http.post.await_args.kwargs["json"] == {
        "user": "alice@example.com",
        "limit": 50,
        "cursor": "next",
    }


async def test_sidecar_auth_failure_is_backend_failure_not_browser_401():
    request, _ = _request(tool="memory_search", response={})
    request.app.state.archive_http.post.return_value = httpx.Response(401)
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    with pytest.raises(HTTPException) as exc:
        await list_memories(request, limit=10, cursor=None, me=me)
    assert exc.value.status_code == 503


async def test_invalid_cursor_stays_a_client_error():
    request, _ = _request(tool="memory_search", response={})
    request.app.state.archive_http.post.return_value = httpx.Response(422)
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    with pytest.raises(HTTPException) as exc:
        await list_memories(request, limit=10, cursor="bad", me=me)
    assert exc.value.status_code == 422
    assert exc.value.detail == "invalid_pagination_cursor"


def test_public_routes_do_not_accept_a_user_selector():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user] = lambda: AuthedUser(
        email="alice@example.com", role="user", owui_id="a",
    )
    schema = app.openapi()

    for path in ("/v1/me/memories", "/v1/me/chat-history/export"):
        names = {
            parameter["name"]
            for parameter in schema["paths"][path]["get"]["parameters"]
        }
        assert "user" not in names


async def test_internal_routes_require_service_token(monkeypatch):
    monkeypatch.setattr(
        tools_app_module.settings, "kb_service_token", SECRET,
    )

    with pytest.raises(HTTPException) as exc:
        await tools_app_module._require_internal_service(None)
    assert exc.value.status_code == 401

    assert (
        await tools_app_module._require_internal_service(SECRET)
        is None
    )


def test_internal_routes_are_hidden_from_model_tool_discovery():
    paths = tools_app_module.app.openapi()["paths"]

    assert "/user_data/memories/list" not in paths
    assert "/user_data/chat_history/export" not in paths
