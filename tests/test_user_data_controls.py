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
from pydantic import ValidationError

from audrey.auth import AuthedUser, require_user
from audrey.routes.user_data import (
    AccountPurgeRequest,
    MemoryCorrection,
    correct_memory,
    delete_chat_history,
    delete_memory,
    export_chat_history,
    get_account_purge,
    get_repair_status,
    list_memories,
    request_account_purge,
    router,
)
from audrey.user_data_visibility import (
    block_remote_personal_reads,
    unblock_remote_personal_reads,
)

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


class _MemoryMutationQdrant:
    def __init__(self, *, user: str, key: str) -> None:
        self.point_id = memory_module._point_id(user, key)
        self.points = {
            self.point_id: {
                "key": key,
                "value": "dark",
                "tags": f"user:{user},topic:preferences",
                "user": user,
                "created_at": "created",
                "updated_at": "updated",
            }
        }
        self.upsert_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    async def retrieve(self, *, ids, **_kwargs):
        return [
            SimpleNamespace(payload=self.points[point_id])
            for point_id in ids
            if point_id in self.points
        ]

    async def upsert(self, *, points, **kwargs):
        self.upsert_calls.append(kwargs)
        for point in points:
            self.points[str(point.id)] = dict(point.payload or {})

    async def delete(self, *, points_selector, **kwargs):
        self.delete_calls.append(kwargs)
        for point_id in points_selector.points:
            self.points.pop(str(point_id), None)


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


async def test_memory_correction_and_delete_are_exact_user_scoped():
    qdrant = _MemoryMutationQdrant(user="alice@example.com", key="theme")
    store = memory_module.MemoryStore.__new__(memory_module.MemoryStore)
    store._qdrant = qdrant
    store._collection = "kb_memory"
    store._embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

    corrected = await store.update_user(
        user="alice@example.com",
        key="theme",
        value="light",
    )

    assert corrected is not None
    assert corrected.value == "light"
    assert corrected.tags == "topic:preferences"
    payload = qdrant.points[qdrant.point_id]
    assert payload["tags"] == "user:alice@example.com,topic:preferences"

    corrected = await store.update_user(
        user="alice@example.com",
        key="theme",
        value="blue",
        tags="topic:display,user:bob@example.com",
    )

    assert corrected is not None
    assert corrected.value == "blue"
    assert corrected.tags == "topic:display"
    payload = qdrant.points[qdrant.point_id]
    assert payload["user"] == "alice@example.com"
    assert payload["tags"] == "user:alice@example.com,topic:display"
    assert qdrant.upsert_calls == [
        {"collection_name": "kb_memory", "wait": True},
        {"collection_name": "kb_memory", "wait": True},
    ]

    assert not await store.delete_user(
        user="bob@example.com",
        key="theme",
    )
    assert qdrant.point_id in qdrant.points

    assert await store.delete_user(
        user="alice@example.com",
        key="theme",
    )
    assert qdrant.point_id not in qdrant.points
    assert qdrant.delete_calls == [{"collection_name": "kb_memory", "wait": True}]


class _MemoryPurgeQdrant:
    def __init__(self) -> None:
        self.delete_call: dict | None = None

    async def delete(self, **kwargs) -> None:
        self.delete_call = kwargs


async def test_memory_purge_filter_is_exact_user_cutoff_and_acknowledged():
    qdrant = _MemoryPurgeQdrant()
    store = memory_module.MemoryStore.__new__(memory_module.MemoryStore)
    store._qdrant = qdrant
    store._collection = "kb_memory"
    cutoff = "2026-01-01T00:00:00.000000+00:00"

    await store.delete_user_before(
        user="alice@example.com",
        cutoff_at=cutoff,
    )

    assert qdrant.delete_call is not None
    assert qdrant.delete_call["collection_name"] == "kb_memory"
    assert qdrant.delete_call["wait"] is True
    selector = qdrant.delete_call["points_selector"].filter
    assert selector.must[0].key == "user"
    assert selector.must[0].match.value == "alice@example.com"
    assert selector.must_not[0].key == "updated_at"
    assert selector.must_not[0].range.gt.isoformat() == "2026-01-01T00:00:00+00:00"


async def test_memory_inventory_applies_post_purge_visibility_cutoff():
    qdrant = _MemoryQdrant()
    store = memory_module.MemoryStore.__new__(memory_module.MemoryStore)
    store._qdrant = qdrant
    store._collection = "kb_memory"
    cutoff = "2026-01-01T00:00:00.000000+00:00"

    await store.list_user(
        user="alice@example.com",
        visible_after=cutoff,
    )

    conditions = qdrant.calls[0]["scroll_filter"].must
    assert conditions[0].key == "user"
    assert conditions[0].match.value == "alice@example.com"
    assert conditions[1].key == "updated_at"
    assert conditions[1].range.gt.isoformat() == "2026-01-01T00:00:00+00:00"


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


async def test_chat_delete_is_owned_hidden_immediately_and_blocks_late_delivery(
    tmp_path: Path,
):
    store = await _archive_store(tmp_path / "archive.db")
    try:
        await _insert_archive_rows(store)
        assert store._db is not None
        async with store._db_lock:
            await store._db.execute(
                """
                INSERT INTO messages
                    (message_id, conversation_id, user, role, content,
                     created_at, archived_at, partial)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "b2",
                    "a",
                    "bob@example.com",
                    "user",
                    "colliding id",
                    "2",
                    "2",
                    0,
                ),
            )
            await store._db.commit()

        assert (
            await store.request_conversation_deletion(
                user="charlie@example.com",
                conversation_id="a",
            )
            is None
        )
        bob_result = await store.request_conversation_deletion(
            user="bob@example.com",
            conversation_id="a",
        )
        assert bob_result is not None
        result = await store.request_conversation_deletion(
            user="alice@example.com",
            conversation_id="a",
        )
        assert result is not None
        assert result["status"] == "pending"

        alice_items, _ = await store.export_user_messages(
            user="alice@example.com",
        )
        bob_items, _ = await store.export_user_messages(
            user="bob@example.com",
        )
        assert alice_items == []
        assert [item.message_id for item in bob_items] == ["b1"]

        late = await store.archive_turn(
            user="alice@example.com",
            conversation_id="a",
            user_content="late retry",
            assistant_content="must stay deleted",
            created_at="2020-01-01T00:00:00+00:00",
        )
        assert late["skipped_deleted"] is True
    finally:
        await store.aclose()


async def test_chat_repair_status_is_exact_user_scoped(tmp_path: Path):
    store = await _archive_store(tmp_path / "archive.db")
    try:
        assert store._db is not None
        async with store._db_lock:
            await store._db.executemany(
                """
                INSERT INTO archive_chunks
                    (chunk_id, conversation_id, user, message_ids_json, text,
                     created_at, indexed_at, index_attempts,
                     index_last_attempt_at, index_last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "alice-index", "alice-index-conv", "alice", "a1", "text",
                        "1", None, 3, "2", "index failed",
                    ),
                    (
                        "alice-delete", "alice-delete-conv", "alice", "a2", "text",
                        "1", "1", 0, None, None,
                    ),
                    (
                        "bob-index", "bob-index-conv", "bob", "b1", "text",
                        "1", None, 1, "2", None,
                    ),
                ],
            )
            await store._db.execute(
                """
                INSERT INTO archive_deletion_outbox
                    (chunk_id, conversation_id, message_ids_json, point_id,
                     requested_at, attempts, last_attempt_at, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "alice-delete", "alice-delete-conv", "a2", "point-a",
                    "1", 2, "2", "delete failed",
                ),
            )
            await store._db.executemany(
                """
                INSERT INTO archive_conversation_deletions
                    (user, conversation_id, cutoff_at, requested_at, completed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("alice", "pending", "1", "1", None),
                    ("alice", "completed", "1", "1", "2"),
                    ("bob", "pending", "1", "1", None),
                ],
            )
            await store._db.commit()

        assert await store.user_stats(user="alice") == {
            "indexing": {
                "pending": 0,
                "attempts": 3,
                "with_error": 1,
                "exhausted": 1,
                "completed": 0,
            },
            "deletions": {
                "pending": 1,
                "attempts": 2,
                "with_error": 1,
                "exhausted": 0,
                "completed": 0,
            },
            "conversation_deletions": {
                "pending": 1,
                "attempts": 0,
                "with_error": 0,
                "exhausted": 0,
                "completed": 1,
            },
        }
        bob = await store.user_stats(user="bob")
        assert bob["indexing"]["pending"] == 1
        assert bob["deletions"]["pending"] == 0
        assert bob["conversation_deletions"]["pending"] == 1
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


async def test_remote_inventory_is_hidden_until_purge_cutoff_is_acknowledged():
    request, http = _request(
        tool="memory_search",
        response={"items": [{"value": "old data"}], "next_cursor": None},
    )
    me = AuthedUser(email="purge-gate@example.com", role="user", owui_id="a")
    block_remote_personal_reads(user=me.email, purge_id="purge-gate")
    try:
        with pytest.raises(HTTPException) as exc:
            await list_memories(request, limit=10, cursor=None, me=me)
        assert exc.value.status_code == 409
        assert exc.value.detail == "personal_data_purge_in_progress"
        http.post.assert_not_awaited()
    finally:
        unblock_remote_personal_reads(user=me.email, purge_id="purge-gate")



async def test_mutation_routes_inject_authenticated_user():
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    update_request, update_http = _request(
        tool="memory_search",
        response={
            "key": "theme",
            "value": "light",
            "tags": "topic:display",
            "created_at": "created",
            "updated_at": "updated",
        },
    )
    await correct_memory(
        update_request,
        MemoryCorrection(value="light", tags="topic:display"),
        key="theme",
        me=me,
    )
    assert update_http.post.await_args.kwargs["json"] == {
        "user": "alice@example.com",
        "key": "theme",
        "value": "light",
        "tags": "topic:display",
    }
    assert update_http.post.await_args.args[0].endswith("/user_data/memories/update")

    memory_request, memory_http = _request(
        tool="memory_search",
        response={"key": "theme", "deleted": True},
    )
    await delete_memory(memory_request, key="theme", me=me)
    assert memory_http.post.await_args.kwargs["json"] == {
        "user": "alice@example.com",
        "key": "theme",
    }

    chat_request, chat_http = _request(
        tool="chat_history_search",
        response={
            "conversation_id": "conversation-a",
            "requested_at": "now",
            "status": "pending",
            "chunks_queued": 1,
            "deletions_pending": 1,
        },
    )
    await delete_chat_history(
        chat_request,
        conversation_id="conversation-a",
        me=me,
    )
    assert chat_http.post.await_args.kwargs["json"] == {
        "user": "alice@example.com",
        "conversation_id": "conversation-a",
    }


async def test_repair_status_composes_current_user_queues_without_raw_errors():
    remote = {
        "indexing": {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 0,
        },
        "deletions": {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 0,
        },
        "conversation_deletions": {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 1,
        },
    }
    request, http = _request(tool="chat_history_search", response=remote)
    file_stats = AsyncMock(return_value={
        "pending": 1,
        "attempts": 2,
        "with_error": 1,
        "exhausted": 0,
        "completed": 3,
    })
    delivery_stats = AsyncMock(return_value={
        "pending": 0,
        "attempts": 0,
        "with_error": 0,
        "exhausted": 0,
        "completed": 0,
    })
    request.app.state.uploads_db = SimpleNamespace(
        user_file_deletion_stats=file_stats,
        user_data_purge_stats=AsyncMock(return_value={
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 1,
        }),
    )
    request.app.state.archive_client = SimpleNamespace(user_stats=delivery_stats)
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    result = await get_repair_status(request, me=me)

    assert result.status == "repairing"
    assert result.file_deletions.pending == 1
    assert result.conversation_deletions.completed == 1
    assert result.account_purges.completed == 1
    file_stats.assert_awaited_once_with(me.email)
    delivery_stats.assert_awaited_once_with(me.email)
    assert http.post.await_args.kwargs["json"] == {"user": me.email}
    assert http.post.await_args.args[0].endswith(
        "/user_data/chat_history/status"
    )
    rendered = result.model_dump_json()
    assert "private backend detail" not in rendered
    assert me.email not in rendered

    attention_remote = {
        **remote,
        "indexing": {**remote["indexing"], "exhausted": 1},
    }
    http.post.return_value = httpx.Response(200, json=attention_remote)
    attention = await get_repair_status(request, me=me)
    assert attention.status == "attention_required"


async def test_repair_status_reports_remote_backend_degraded():
    request, _ = _request(tool="chat_history_search", response={})
    request.app.state.archive_http.post.return_value = httpx.Response(503)
    empty = {
        "pending": 0,
        "attempts": 0,
        "with_error": 0,
        "exhausted": 0,
        "completed": 0,
    }
    request.app.state.uploads_db = SimpleNamespace(
        user_file_deletion_stats=AsyncMock(return_value=empty),
        user_data_purge_stats=AsyncMock(return_value=empty),
    )
    request.app.state.archive_client = SimpleNamespace(
        user_stats=AsyncMock(return_value=empty),
    )
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    result = await get_repair_status(request, me=me)

    assert result.status == "degraded"
    assert result.file_deletions.available is True
    assert result.chat_delivery.available is True
    assert result.chat_indexing.available is False
    assert result.chat_deletions.available is False
    assert result.conversation_deletions.available is False
    assert result.account_purges.available is True


def test_account_purge_requires_exact_confirmation_phrase():
    with pytest.raises(ValidationError):
        AccountPurgeRequest(confirmation="delete my data")  # type: ignore[arg-type]



def _purge_status(*, purge_id: str = "purge-a") -> dict:
    return {
        "schema_version": 1,
        "purge_id": purge_id,
        "cutoff_at": "2026-01-01T00:00:00.000000+00:00",
        "requested_at": "2026-01-01T00:00:00.000000+00:00",
        "status": "pending",
        "completed_at": "",
        "files": {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "completed": 0,
        },
        "paths": {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "completed": 0,
        },
        "local_delivery": {
            "completed": True,
            "attempts": 1,
            "with_error": False,
        },
        "sidecar": {
            "acknowledged": True,
            "status": "pending",
            "completed": False,
            "attempts": 1,
            "with_error": False,
        },
    }


async def test_account_purge_uses_verified_owner_and_scoped_idempotency_key():
    coordinator = SimpleNamespace(request=AsyncMock(return_value=_purge_status()))
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(user_data_purges=coordinator)),
    )
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    result = await request_account_purge(
        request,
        AccountPurgeRequest(confirmation="DELETE ALL MY AUDREY DATA"),
        idempotency_key="same-operation",
        me=me,
    )

    expected_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        "audrey-purge|alice@example.com|same-operation",
    ))
    assert result.purge_id == "purge-a"
    coordinator.request.assert_awaited_once_with(
        user="alice@example.com",
        purge_id=expected_id,
    )


async def test_account_purge_status_is_exact_owner_and_not_found_is_404():
    coordinator = SimpleNamespace(
        status=AsyncMock(side_effect=[_purge_status(), None]),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(user_data_purges=coordinator)),
    )
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    found = await get_account_purge(request, purge_id="purge-a", me=me)
    assert found.purge_id == "purge-a"

    with pytest.raises(HTTPException) as exc:
        await get_account_purge(request, purge_id="missing", me=me)
    assert exc.value.status_code == 404
    assert exc.value.detail == "purge_not_found"
    assert coordinator.status.await_args_list[0].kwargs == {
        "user": "alice@example.com",
        "purge_id": "purge-a",
    }


async def test_mutation_not_found_is_a_current_user_404():
    request, _ = _request(tool="memory_search", response={})
    request.app.state.archive_http.post.return_value = httpx.Response(404)
    me = AuthedUser(email="alice@example.com", role="user", owui_id="a")

    with pytest.raises(HTTPException) as exc:
        await delete_memory(request, key="missing", me=me)
    assert exc.value.status_code == 404
    assert exc.value.detail == "memory_not_found"


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

    operations = (
        ("/v1/me/memories", "get"),
        ("/v1/me/memories/{key}", "put"),
        ("/v1/me/memories/{key}", "delete"),
        ("/v1/me/chat-history/export", "get"),
        ("/v1/me/chat-history/{conversation_id}", "delete"),
        ("/v1/me/repair-status", "get"),
        ("/v1/me/data-purge", "post"),
        ("/v1/me/data-purge/{purge_id}", "get"),
    )
    for path, method in operations:
        operation = schema["paths"][path][method]
        names = {
            parameter["name"]
            for parameter in operation.get("parameters", [])
        }
        assert "user" not in names
        assert "user" not in str(operation.get("requestBody", {})).lower()


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
    assert "/user_data/memories/update" not in paths
    assert "/user_data/memories/delete" not in paths
    assert "/user_data/chat_history/export" not in paths
    assert "/user_data/chat_history/delete" not in paths
    assert "/user_data/chat_history/status" not in paths
    assert "/user_data/purge" not in paths
    assert "/user_data/purge/status" not in paths
