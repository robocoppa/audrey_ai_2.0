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

    async def search(
        self, *, user: str, query: str, top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Keyword-match search scoped to a user.

        Matches rows whose tags contain `user:<user>` AND whose key, value,
        or tags contain any whitespace-separated token from `query`. Ordered
        by `updated_at DESC` so recent notes win ties.
        """
        tokens = [t for t in query.split() if t]
        if not tokens:
            return []
        user_tag = f"user:{user}"
        like_clauses = []
        params: list[str] = [f"%{user_tag}%"]
        for tok in tokens:
            like_clauses.append("(key LIKE ? OR value LIKE ? OR tags LIKE ?)")
            like = f"%{tok}%"
            params.extend([like, like, like])
        sql = (
            "SELECT key, value, tags, created_at, updated_at FROM memory "
            "WHERE tags LIKE ? AND (" + " OR ".join(like_clauses) + ") "
            "ORDER BY updated_at DESC LIMIT ?"
        )
        params.append(str(top_k))
        async with aiosqlite.connect(self._db_path) as db:
            rows = await db.execute_fetchall(sql, tuple(params))
        return [
            MemoryEntry(key=k, value=v, tags=t, created_at=c, updated_at=u)
            for (k, v, t, c, u) in rows
        ]
