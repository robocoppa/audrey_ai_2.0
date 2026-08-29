"""Legacy SQLite-to-Qdrant memory migration failure semantics."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from db import EmbedError, MemoryStore, _point_id  # noqa: E402


class _FakeQdrant:
    def __init__(self) -> None:
        self.points: dict[str, object] = {}
        self.fail_keys: set[str] = set()

    async def upsert(self, *, collection_name: str, points: list) -> None:
        assert collection_name == "kb_memory"
        point = points[0]
        key = str(point.payload["key"])
        if key in self.fail_keys:
            raise RuntimeError(f"injected upsert failure for {key}")
        self.points[str(point.id)] = point


def _legacy_db(path: Path, rows: list[tuple[str, str, str]]) -> None:
    db = sqlite3.connect(path)
    try:
        db.execute(
            "CREATE TABLE memory ("
            "key TEXT, value TEXT, tags TEXT, created_at TEXT, updated_at TEXT)"
        )
        db.executemany(
            "INSERT INTO memory VALUES (?, ?, ?, 'created', 'updated')",
            rows,
        )
        db.commit()
    finally:
        db.close()


def _store(path: Path, qdrant: _FakeQdrant) -> MemoryStore:
    store = MemoryStore.__new__(MemoryStore)
    store._legacy_sqlite_path = path
    store._collection = "kb_memory"
    store._qdrant = qdrant
    return store


async def test_embed_failure_retains_source_and_retry_is_idempotent(tmp_path: Path):
    path = tmp_path / "memory.db"
    _legacy_db(
        path,
        [
            ("good", "first value", "user:alice@example.com"),
            ("retry", "second value", "user:alice@example.com"),
        ],
    )
    qdrant = _FakeQdrant()
    store = _store(path, qdrant)
    fail_retry = True

    async def embed(text: str):
        if fail_retry and text.startswith("retry:"):
            raise EmbedError("injected embed failure")
        return [0.1, 0.2, 0.3]

    store._embed = embed
    await store._migrate_sqlite_if_present()

    assert path.exists()
    assert not path.with_suffix(".db.migrated").exists()
    assert set(qdrant.points) == {_point_id("alice@example.com", "good")}

    fail_retry = False
    await store._migrate_sqlite_if_present()

    assert not path.exists()
    assert path.with_suffix(".db.migrated").exists()
    assert set(qdrant.points) == {
        _point_id("alice@example.com", "good"),
        _point_id("alice@example.com", "retry"),
    }


async def test_upsert_failure_retains_source_for_restart(tmp_path: Path):
    path = tmp_path / "memory.db"
    _legacy_db(path, [("retry", "value", "user:alice@example.com")])
    qdrant = _FakeQdrant()
    qdrant.fail_keys.add("retry")
    store = _store(path, qdrant)

    async def embed(_text: str):
        return [0.1, 0.2, 0.3]

    store._embed = embed
    await store._migrate_sqlite_if_present()
    assert path.exists()

    qdrant.fail_keys.clear()
    await store._migrate_sqlite_if_present()
    assert not path.exists()
    assert path.with_suffix(".db.migrated").exists()


async def test_unscoped_row_prevents_source_from_being_renamed(tmp_path: Path):
    path = tmp_path / "memory.db"
    _legacy_db(path, [("unsafe", "value", "topic:notes")])
    qdrant = _FakeQdrant()
    store = _store(path, qdrant)

    async def embed(_text: str):
        raise AssertionError("unscoped rows must not be embedded")

    store._embed = embed
    await store._migrate_sqlite_if_present()

    assert path.exists()
    assert qdrant.points == {}


async def test_existing_migrated_backup_is_not_overwritten(tmp_path: Path):
    path = tmp_path / "memory.db"
    migrated = path.with_suffix(".db.migrated")
    migrated.write_text("older backup", encoding="utf-8")
    _legacy_db(path, [("new", "value", "user:alice@example.com")])
    store = _store(path, _FakeQdrant())

    async def embed(_text: str):
        return [0.1, 0.2, 0.3]

    store._embed = embed
    await store._migrate_sqlite_if_present()

    assert migrated.read_text(encoding="utf-8") == "older backup"
    assert (tmp_path / "memory.db.migrated.1").exists()
