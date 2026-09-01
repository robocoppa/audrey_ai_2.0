"""Failure-injection tests for durable chat-archive repair."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest
from pydantic import ValidationError

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

archive_module = importlib.import_module("chat_archive")
settings_module = importlib.import_module("settings")


class _FakeHttp:
    async def aclose(self) -> None:
        pass


class _UnusedQdrant:
    def __init__(self, **_kwargs) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _avoid_real_qdrant_client(monkeypatch):
    monkeypatch.setattr(archive_module, "AsyncQdrantClient", _UnusedQdrant)


class _FakeQdrant:
    def __init__(
        self,
        *,
        upsert_failures: int = 0,
        delete_failures: int = 0,
    ) -> None:
        self.upsert_failures = upsert_failures
        self.delete_failures = delete_failures
        self.upsert_calls = 0
        self.delete_calls = 0
        self.points: dict[str, dict] = {}

    async def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="archive-test")])

    async def create_collection(self, **_kwargs) -> None:
        raise AssertionError("the fake collection already exists")

    async def create_payload_index(self, **_kwargs) -> None:
        raise AssertionError("the fake collection already exists")

    async def upsert(self, *, points, **_kwargs) -> None:
        self.upsert_calls += 1
        if self.upsert_failures:
            self.upsert_failures -= 1
            raise RuntimeError("injected upsert failure")
        for point in points:
            self.points[str(point.id)] = dict(point.payload or {})

    async def delete(self, *, points_selector, wait, **_kwargs) -> None:
        assert wait is True
        self.delete_calls += 1
        if self.delete_failures:
            self.delete_failures -= 1
            raise RuntimeError("injected delete failure")
        for point_id in points_selector.points:
            self.points.pop(str(point_id), None)

    async def query_points(self, **_kwargs):
        return SimpleNamespace(
            points=[SimpleNamespace(payload=payload, score=0.9) for payload in self.points.values()]
        )

    async def close(self) -> None:
        pass


async def _good_embed(_http, _model, _text, _keep_alive="") -> list[float]:
    return [0.1, 0.2, 0.3]


async def _make_store(
    sqlite_path: Path,
    qdrant: _FakeQdrant,
    *,
    retention_days: int = 0,
    max_retry_attempts: int = 3,
) -> archive_module.ChatArchiveStore:
    store = archive_module.ChatArchiveStore(
        sqlite_path=sqlite_path,
        qdrant_url="http://qdrant.invalid",
        ollama_url="http://ollama.invalid",
        collection="archive-test",
        embed_model="embed-test",
        embed_dim=3,
        embed_timeout_s=1.0,
        chunk_max_chars=2500,
        chunk_overlap_chars=100,
        search_threshold=0.0,
        retention_days=retention_days,
        max_bytes=0,
        repair_batch_size=20,
        max_retry_attempts=max_retry_attempts,
    )
    await store._qdrant.close()
    await store._http.aclose()
    store._qdrant = qdrant
    store._http = _FakeHttp()
    await store.init()
    return store


async def _archive_one(store: archive_module.ChatArchiveStore) -> dict:
    return await store.archive_turn(
        user="alice@example.com",
        conversation_id="conv-1",
        user_content="What did we decide?",
        assistant_content="Keep the durable source until deletion is confirmed.",
    )


async def _age_all_rows(store: archive_module.ChatArchiveStore) -> None:
    assert store._db is not None
    old = "2020-01-01T00:00:00+00:00"
    async with store._db_lock:
        await store._db.execute("UPDATE messages SET created_at = ?", (old,))
        await store._db.execute("UPDATE archive_chunks SET created_at = ?", (old,))
        await store._db.commit()


class _PurgeMemory:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[tuple[str, str]] = []

    async def delete_user_before(self, *, user: str, cutoff_at: str) -> None:
        self.calls.append((user, cutoff_at))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("injected memory delete failure")


async def test_init_migrates_pre_repair_archive_schema(tmp_path: Path):
    db_path = tmp_path / "archive.db"
    db = await aiosqlite.connect(db_path)
    try:
        await db.execute(
            """
            CREATE TABLE archive_chunks (
                chunk_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user TEXT NOT NULL,
                message_ids_json TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                indexed_at TEXT
            )
            """
        )
        await db.commit()
    finally:
        await db.close()

    store = await _make_store(db_path, _FakeQdrant())
    try:
        assert store._db is not None
        cursor = await store._db.execute("PRAGMA table_info(archive_chunks)")
        columns = {str(row[1]) for row in await cursor.fetchall()}
        await cursor.close()
        assert {
            "index_attempts",
            "index_last_attempt_at",
            "index_last_error",
        }.issubset(columns)

        cursor = await store._db.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'archive_deletion_outbox'
            """
        )
        assert await cursor.fetchone() is not None
        await cursor.close()
        cursor = await store._db.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'archive_conversation_deletions'
            """
        )
        assert await cursor.fetchone() is not None
        await cursor.close()
        cursor = await store._db.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'archive_maintenance_state'
            """
        )
        assert await cursor.fetchone() is not None
        await cursor.close()
    finally:
        await store.aclose()


async def test_embed_failure_reindexes_after_restart(tmp_path: Path, monkeypatch):
    calls = 0

    async def fail_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise archive_module._EmbedError("injected embed failure")
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(archive_module, "_embed", fail_once)
    db_path = tmp_path / "archive.db"
    qdrant = _FakeQdrant()
    store = await _make_store(db_path, qdrant)
    result = await _archive_one(store)
    assert result["index_failed"] == 1
    stats = await store.stats()
    assert stats["chunks_reindex_pending"] == 1
    assert stats["chunks_reindex_attempts"] == 1
    assert stats["chunks_reindex_with_error"] == 1
    repair_status = await store.repair_stats()
    assert repair_status["indexing"] == {
        "pending": 1,
        "attempts": 1,
        "with_error": 1,
        "exhausted": 0,
        "completed": 0,
    }
    assert "last_error" not in str(repair_status)
    assert "injected embed failure" in stats["index_last_error"]
    await store.aclose()

    restarted = await _make_store(db_path, qdrant)
    try:
        repair = await restarted.maintain()
        assert repair["reindex"] == {
            "attempted": 1,
            "indexed": 1,
            "failed": 0,
        }
        assert len(qdrant.points) == 1
        stats = await restarted.stats()
        assert stats["chunks_unindexed"] == 0
        assert stats["index_last_attempt_at"]
        assert stats["index_last_error"] == ""
    finally:
        await restarted.aclose()


async def test_upsert_retry_is_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    qdrant = _FakeQdrant(upsert_failures=1)
    store = await _make_store(tmp_path / "archive.db", qdrant)
    try:
        result = await _archive_one(store)
        assert result["index_failed"] == 1

        first = await store.maintain()
        assert first["reindex"]["indexed"] == 1
        assert len(qdrant.points) == 1

        second = await store.maintain()
        assert second["reindex"]["attempted"] == 0
        assert len(qdrant.points) == 1
        assert qdrant.upsert_calls == 2
    finally:
        await store.aclose()


async def test_repeated_delivery_id_does_not_duplicate_source_rows(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    qdrant = _FakeQdrant()
    store = await _make_store(tmp_path / "archive.db", qdrant)
    try:
        kwargs = {
            "user": "alice@example.com",
            "conversation_id": "conv-1",
            "user_content": "same question",
            "assistant_content": "same answer",
            "archive_id": "stable-delivery-id",
        }
        first = await store.archive_turn(
            **kwargs,
            created_at="2026-08-28T12:00:00+00:00",
        )
        second = await store.archive_turn(
            **kwargs,
            created_at="2026-08-28T13:00:00+00:00",
        )

        assert first["user_message_id"] == second["user_message_id"]
        assert first["assistant_message_id"] == second["assistant_message_id"]
        assert store._db is not None
        cursor = await store._db.execute("SELECT COUNT(*) FROM messages")
        assert int((await cursor.fetchone())[0]) == 2
        await cursor.close()
        cursor = await store._db.execute("SELECT COUNT(*) FROM archive_chunks")
        assert int((await cursor.fetchone())[0]) == 1
        await cursor.close()
        assert len(qdrant.points) == 1
    finally:
        await store.aclose()


async def test_index_retries_are_bounded_and_manual_prune_resets_them(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    qdrant = _FakeQdrant(upsert_failures=99)
    store = await _make_store(
        tmp_path / "archive.db",
        qdrant,
        max_retry_attempts=2,
    )
    try:
        result = await _archive_one(store)
        assert result["index_failed"] == 1
        await store.maintain()
        await store.maintain()
        assert qdrant.upsert_calls == 2
        assert (await store.stats())["chunks_reindex_exhausted"] == 1

        qdrant.upsert_failures = 0
        no_reset = await store.prune()
        assert no_reset["reindex_attempted"] == 0
        assert qdrant.upsert_calls == 2

        reset = await store.prune(retry_exhausted=True)
        assert reset["reindexed"] == 1
        assert qdrant.upsert_calls == 3
        assert (await store.stats())["chunks_unindexed"] == 0
    finally:
        await store.aclose()


async def test_failed_delete_survives_restart_and_is_hidden_from_search(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    db_path = tmp_path / "archive.db"
    qdrant = _FakeQdrant(delete_failures=1)
    store = await _make_store(db_path, qdrant, retention_days=1)
    await _archive_one(store)
    await _age_all_rows(store)

    result = await store.prune()
    assert result == {
        "deletions_queued": 1,
        "messages_deleted": 0,
        "chunks_deleted": 0,
        "qdrant_deleted": 0,
        "delete_failed": 1,
        "deletions_pending": 1,
        "reindex_attempted": 0,
        "reindexed": 0,
        "reindex_failed": 0,
    }
    stats = await store.stats()
    assert stats["messages"] == 2
    assert stats["chunks"] == 1
    assert stats["deletions_pending"] == 1
    assert stats["deletion_attempts"] == 1
    assert stats["deletions_with_error"] == 1
    repair_status = await store.repair_stats()
    assert repair_status["deletions"]["pending"] == 1
    assert repair_status["deletions"]["attempts"] == 1
    assert repair_status["deletions"]["with_error"] == 1
    assert "injected delete failure" in stats["delete_last_error"]
    assert (
        await store.search(
            user="alice@example.com",
            query="durable source",
            limit=5,
        )
        == []
    )
    await store.aclose()

    restarted = await _make_store(
        db_path,
        qdrant,
        retention_days=1,
    )
    try:
        repair = await restarted.maintain()
        assert repair["delete"]["chunks_deleted"] == 1
        assert repair["delete"]["messages_deleted"] == 2
        assert repair["deletions_pending"] == 0
        stats = await restarted.stats()
        assert stats["messages"] == 0
        assert stats["chunks"] == 0
        assert stats["delete_last_attempt_at"]
        assert stats["delete_last_error"] == ""
        assert qdrant.points == {}

        await restarted.maintain()
        assert qdrant.delete_calls == 2
    finally:
        await restarted.aclose()


async def test_user_conversation_delete_converges_after_restart_and_blocks_late_retry(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    db_path = tmp_path / "archive.db"
    qdrant = _FakeQdrant(delete_failures=1)
    store = await _make_store(db_path, qdrant)
    await _archive_one(store)

    assert (
        await store.request_conversation_deletion(
            user="bob@example.com",
            conversation_id="conv-1",
        )
        is None
    )
    receipt = await store.request_conversation_deletion(
        user="alice@example.com",
        conversation_id="conv-1",
    )
    assert receipt is not None
    assert receipt["chunks_queued"] == 1
    assert receipt["deletions_pending"] == 1
    assert await store.export_user_messages(user="alice@example.com") == ([], None)

    first = await store.maintain()
    assert first["delete"]["failed"] == 1
    stats = await store.stats()
    assert stats["conversation_deletions_pending"] == 1
    await store.aclose()

    restarted = await _make_store(db_path, qdrant)
    try:
        repaired = await restarted.maintain()
        assert repaired["delete"]["chunks_deleted"] == 1
        stats = await restarted.stats()
        assert stats["messages"] == 0
        assert stats["chunks"] == 0
        assert stats["conversation_deletions_pending"] == 0
        assert stats["conversation_deletions_completed"] == 1

        late = await restarted.archive_turn(
            user="alice@example.com",
            conversation_id="conv-1",
            user_content="late retry",
            assistant_content="must stay deleted",
            archive_id="late-delivery",
            created_at="2020-01-01T00:00:00+00:00",
        )
        assert late["skipped_deleted"] is True
        assert (await restarted.stats())["messages"] == 0

        fresh = await restarted.archive_turn(
            user="alice@example.com",
            conversation_id="conv-1",
            user_content="new turn",
            assistant_content="allowed after deletion",
            archive_id="fresh-delivery",
            created_at="2099-01-01T00:00:00+00:00",
        )
        assert "skipped_deleted" not in fresh
        items, _ = await restarted.export_user_messages(
            user="alice@example.com",
        )
        assert [item.content for item in items] == [
            "new turn",
            "allowed after deletion",
        ]
    finally:
        await restarted.aclose()


async def test_delete_retries_are_bounded_and_manual_prune_resets_them(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    qdrant = _FakeQdrant(delete_failures=99)
    store = await _make_store(
        tmp_path / "archive.db",
        qdrant,
        retention_days=1,
        max_retry_attempts=2,
    )
    try:
        await _archive_one(store)
        await _age_all_rows(store)

        await store.maintain()
        await store.maintain()
        await store.maintain()
        assert qdrant.delete_calls == 2
        stats = await store.stats()
        assert stats["deletions_exhausted"] == 1

        qdrant.delete_failures = 0
        no_reset = await store.prune()
        assert no_reset["deletions_pending"] == 1
        assert qdrant.delete_calls == 2

        reset = await store.prune(retry_exhausted=True)
        assert reset["chunks_deleted"] == 1
        assert reset["deletions_pending"] == 0
        assert qdrant.delete_calls == 3
    finally:
        await store.aclose()


async def test_maintainer_runs_immediately_and_stops_cleanly():
    ran = asyncio.Event()
    fake_store = SimpleNamespace(maintain=AsyncMock(side_effect=lambda: ran.set()))
    maintainer = archive_module.ChatArchiveMaintainer(
        fake_store,
        interval_s=3600,
    )

    await maintainer.start()
    await asyncio.wait_for(ran.wait(), timeout=1.0)
    assert fake_store.maintain.await_count == 1
    assert maintainer._task is not None
    assert maintainer._task.get_name() == "chat-archive-maintenance"

    await maintainer.stop()
    assert maintainer._task is None


def test_nonzero_archive_max_bytes_is_rejected():
    with pytest.raises(ValidationError, match="CHAT_ARCHIVE_MAX_BYTES"):
        settings_module.Settings(
            _env_file=None,
            CHAT_ARCHIVE_MAX_BYTES=1,
        )


async def test_account_purge_is_owned_cutoff_based_and_idempotent(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    qdrant = _FakeQdrant()
    store = await _make_store(tmp_path / "purge.db", qdrant)
    memory = _PurgeMemory()
    cutoff = "2026-01-01T00:00:00.000000+00:00"
    try:
        await _archive_one(store)
        await store.archive_turn(
            user="bob@example.com",
            conversation_id="bob-conv",
            user_content="Bob old question",
            assistant_content="Bob old answer",
        )
        await _age_all_rows(store)

        receipt = await store.request_user_purge(
            user="alice@example.com",
            purge_id="purge-alice-1",
            cutoff_at=cutoff,
            memory_store=memory,
        )

        assert receipt["status"] == "pending"
        assert receipt["memory"] == {
            "completed": True,
            "attempts": 1,
            "with_error": False,
            "exhausted": False,
        }
        assert receipt["chat"]["pending"] == 1
        assert memory.calls == [("alice@example.com", cutoff)]
        assert await store.user_purge_cutoff(user="alice@example.com") == cutoff
        assert await store.user_purge_cutoff(user="bob@example.com") == ""

        alice_hidden, _ = await store.export_user_messages(user="alice@example.com")
        bob_visible, _ = await store.export_user_messages(user="bob@example.com")
        assert alice_hidden == []
        assert len(bob_visible) == 2

        late = await store.archive_turn(
            user="alice@example.com",
            conversation_id="conv-late",
            user_content="late old question",
            assistant_content="late old answer",
            created_at="2025-12-01T00:00:00.000000+00:00",
        )
        assert late["skipped_deleted"] is True

        fresh = await store.archive_turn(
            user="alice@example.com",
            conversation_id="conv-fresh",
            user_content="fresh question",
            assistant_content="fresh answer",
            created_at="2027-01-01T00:00:00.000000+00:00",
        )
        assert "skipped_deleted" not in fresh

        repeated = await store.request_user_purge(
            user="alice@example.com",
            purge_id="purge-alice-1",
            cutoff_at=cutoff,
            memory_store=memory,
        )
        assert repeated["purge_id"] == "purge-alice-1"
        assert memory.calls == [("alice@example.com", cutoff)]
        with pytest.raises(ValueError, match="already bound"):
            await store.request_user_purge(
                user="bob@example.com",
                purge_id="purge-alice-1",
                cutoff_at=cutoff,
                memory_store=memory,
            )

        await store.maintain(memory_store=memory)
        completed = await store.user_purge_status(
            user="alice@example.com",
            purge_id="purge-alice-1",
        )
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["chat"]["pending"] == 0
        fresh_only, _ = await store.export_user_messages(user="alice@example.com")
        assert {item.content for item in fresh_only} == {
            "fresh question",
            "fresh answer",
        }
    finally:
        await store.aclose()


async def test_account_purge_recovers_after_restart_and_backend_outage(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(archive_module, "_embed", _good_embed)
    db_path = tmp_path / "purge-restart.db"
    cutoff = "2026-01-01T00:00:00.000000+00:00"
    failing_memory = _PurgeMemory(failures=1)
    first = await _make_store(db_path, _FakeQdrant(delete_failures=1))
    try:
        await _archive_one(first)
        await _age_all_rows(first)
        receipt = await first.request_user_purge(
            user="alice@example.com",
            purge_id="purge-restart",
            cutoff_at=cutoff,
            memory_store=failing_memory,
        )
        assert receipt["status"] == "pending"
        assert receipt["memory"]["with_error"] is True
        hidden, _ = await first.export_user_messages(user="alice@example.com")
        assert hidden == []
    finally:
        await first.aclose()

    recovered_memory = _PurgeMemory()
    restarted = await _make_store(db_path, _FakeQdrant())
    try:
        hidden, _ = await restarted.export_user_messages(user="alice@example.com")
        assert hidden == []
        before = await restarted.user_purge_status(
            user="alice@example.com",
            purge_id="purge-restart",
        )
        assert before is not None
        assert before["memory"]["attempts"] == 1
        assert before["memory"]["with_error"] is True

        await restarted.maintain(memory_store=recovered_memory)

        after = await restarted.user_purge_status(
            user="alice@example.com",
            purge_id="purge-restart",
        )
        assert after is not None
        assert after["status"] == "completed"
        assert after["memory"]["attempts"] == 2
        assert after["memory"]["with_error"] is False
        assert recovered_memory.calls == [("alice@example.com", cutoff)]
    finally:
        await restarted.aclose()
