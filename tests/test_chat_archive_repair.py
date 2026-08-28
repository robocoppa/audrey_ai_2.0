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
        assert (await restarted.stats())["chunks_unindexed"] == 0
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
        assert qdrant.points == {}

        await restarted.maintain()
        assert qdrant.delete_calls == 2
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
