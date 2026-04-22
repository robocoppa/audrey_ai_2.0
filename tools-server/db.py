"""SQLite-backed memory store for the custom-tools server.

Schema is intentionally tiny: (key, value, tags, created_at, updated_at).
The models decide what shape to use inside `value` (JSON, plain text, etc).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass(slots=True, frozen=True)
class MemoryEntry:
    key: str
    value: str
    tags: str
    created_at: str
    updated_at: str


class MemoryStore:
    """Thin async wrapper around a SQLite file."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def init(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_SCHEMA)
            await db.commit()

    async def store(self, key: str, value: str, tags: str = "") -> MemoryEntry:
        now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO memory (key, value, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    tags = excluded.tags,
                    updated_at = excluded.updated_at
                """,
                (key, value, tags, now, now),
            )
            await db.commit()
            row = await db.execute_fetchall(
                "SELECT key, value, tags, created_at, updated_at FROM memory WHERE key = ?",
                (key,),
            )
        (k, v, t, c, u) = row[0]
        return MemoryEntry(key=k, value=v, tags=t, created_at=c, updated_at=u)

    async def recall(self, key: str) -> MemoryEntry | None:
        async with aiosqlite.connect(self._db_path) as db:
            rows = await db.execute_fetchall(
                "SELECT key, value, tags, created_at, updated_at FROM memory WHERE key = ?",
                (key,),
            )
        if not rows:
            return None
        (k, v, t, c, u) = rows[0]
        return MemoryEntry(key=k, value=v, tags=t, created_at=c, updated_at=u)
