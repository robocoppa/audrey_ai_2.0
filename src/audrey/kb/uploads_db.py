"""SQLite index over per-user upload metadata (Phase 15).

Qdrant remains authoritative for *file content* (text chunks, image
embeddings). This module is a fast index over the per-file metadata so
`GET /v1/files` and the quota gate don't have to scroll a whole user
collection.

Source-of-truth contract:
  - Qdrant has the file's content → sqlite has a row for it.
  - Qdrant has nothing for a file_id → sqlite must not either.

The startup `reconcile_with_qdrant` enforces (b) by deleting sqlite rows
whose file_id has zero points in the user's Qdrant collections, and
backfills (a) on first run by scrolling each `kb_user_*` collection.
After reconciliation, normal request flow keeps both stores in step:

  upload   → qdrant upsert succeeds → sqlite insert
             (insert fails ⇒ delete the just-upserted points to avoid
              a phantom file no one can see in their list)
  delete   → sqlite delete first (so the file vanishes from list even
             if Qdrant delete is slow), then qdrant delete-by-filter
  list     → sqlite only
  quota    → sqlite only

The whole table is a single connection guarded by a per-instance lock,
because sqlite3's default `check_same_thread=True` and our async wrapper
runs every call in a thread. Concurrent writers serialize through the
lock, which is fine — uploads are bursty per user, not high QPS.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
  file_id     TEXT PRIMARY KEY,
  user        TEXT NOT NULL,
  filename    TEXT NOT NULL,
  mime        TEXT NOT NULL,
  bytes       INTEGER NOT NULL,
  kind        TEXT NOT NULL,
  collection  TEXT NOT NULL,
  chunks      INTEGER NOT NULL,
  uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads(user);
"""


class UploadsDB:
    """Thin async wrapper over a single sqlite connection.

    All public methods run their sqlite work in a thread to keep the
    event loop free; the per-instance lock serializes writes.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False` because asyncio.to_thread can hand the
        # connection to any worker thread. The lock makes that safe.
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def record_upload(
        self,
        *,
        file_id: str,
        user: str,
        filename: str,
        mime: str,
        bytes_: int,
        kind: str,
        collection: str,
        chunks: int,
        uploaded_at: str,
    ) -> None:
        await asyncio.to_thread(
            self._record_sync, file_id, user, filename, mime, bytes_,
            kind, collection, chunks, uploaded_at,
        )

    def _record_sync(
        self, file_id: str, user: str, filename: str, mime: str,
        bytes_: int, kind: str, collection: str, chunks: int, uploaded_at: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO uploads "
                "(file_id, user, filename, mime, bytes, kind, collection, chunks, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (file_id, user, filename, mime, bytes_, kind, collection, chunks, uploaded_at),
            )

    async def delete_upload(self, file_id: str, *, user: str) -> bool:
        """Delete the row for (file_id, user). Returns True if a row was removed.

        Like the Qdrant delete, this is double-keyed on user — a leaked
        file_id from another user can't drop the row.
        """
        return await asyncio.to_thread(self._delete_sync, file_id, user)

    def _delete_sync(self, file_id: str, user: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM uploads WHERE file_id = ? AND user = ?",
                (file_id, user),
            )
            return cur.rowcount > 0

    async def list_user(self, user: str) -> list[dict]:
        return await asyncio.to_thread(self._list_user_sync, user)

    def _list_user_sync(self, user: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT file_id, filename, mime, bytes, kind, collection, "
                "       chunks, uploaded_at "
                "FROM uploads WHERE user = ? ORDER BY uploaded_at DESC",
                (user,),
            )
            return [dict(r) for r in cur.fetchall()]

    async def user_total_bytes(self, user: str) -> int:
        return await asyncio.to_thread(self._user_total_sync, user)

    def _user_total_sync(self, user: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS total FROM uploads WHERE user = ?",
                (user,),
            )
            return int(cur.fetchone()["total"])

    async def all_users(self) -> list[str]:
        return await asyncio.to_thread(self._all_users_sync)

    def _all_users_sync(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute("SELECT DISTINCT user FROM uploads")
            return [r["user"] for r in cur.fetchall()]

    async def file_ids_for_user(self, user: str) -> set[str]:
        return await asyncio.to_thread(self._file_ids_sync, user)

    def _file_ids_sync(self, user: str) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT file_id FROM uploads WHERE user = ?", (user,),
            )
            return {r["file_id"] for r in cur.fetchall()}


async def reconcile_with_qdrant(db: "UploadsDB", qdrant) -> dict[str, int]:
    """Make sqlite agree with Qdrant.

    Strategy:
      1. For every user collection that exists in Qdrant, scroll it,
         build the same `(file_id → row)` map as `qdrant.list_user_files`,
         and INSERT OR REPLACE every row. Brings sqlite up to whatever
         is actually in vector storage right now.
      2. For every user already in sqlite, delete sqlite rows whose
         file_id no longer appears in either of that user's Qdrant
         collections. Catches files that were deleted in Qdrant out of
         band (manual purge, migration, etc.).

    Returns a small stats dict for the boot log.
    """
    from audrey.kb.user_store import (
        USER_IMAGE_PREFIX,
        USER_TEXT_PREFIX,
        user_image_collection,
        user_text_collection,
    )

    # 1. Pull every kb_user_* collection from qdrant and backfill sqlite.
    all_collections = await asyncio.to_thread(_list_collection_names, qdrant)
    user_to_collections: dict[str, list[str]] = {}
    for col in all_collections:
        if not (col.startswith(USER_TEXT_PREFIX) or col.startswith(USER_IMAGE_PREFIX)):
            continue
        # We don't know the raw user id from the sanitized suffix, so we
        # scroll and read it from each point's payload (which preserves
        # the raw id). Group collections by raw user as we go.
        rows = await _scroll_user_rows(qdrant, col)
        for raw_user, files in rows.items():
            user_to_collections.setdefault(raw_user, [])
            if col not in user_to_collections[raw_user]:
                user_to_collections[raw_user].append(col)
            for f in files:
                await db.record_upload(
                    file_id=f["file_id"],
                    user=raw_user,
                    filename=f["filename"],
                    mime=f["mime"],
                    bytes_=f["bytes"],
                    kind=f["kind"],
                    collection=col,
                    chunks=f["chunks"],
                    uploaded_at=f["uploaded_at"],
                )

    # 2. Drop sqlite rows for files that no longer exist anywhere in qdrant.
    pruned = 0
    for user in await db.all_users():
        text_col = user_text_collection(user)
        image_col = user_image_collection(user)
        live_ids: set[str] = set()
        for col in (text_col, image_col):
            if not await qdrant.collection_exists(col):
                continue
            for row in await qdrant.list_user_files(user=user, collection=col):
                live_ids.add(row["file_id"])
        sqlite_ids = await db.file_ids_for_user(user)
        for fid in sqlite_ids - live_ids:
            if await db.delete_upload(fid, user=user):
                pruned += 1

    backfilled = sum(len(v) for v in user_to_collections.values())
    log.info(
        "uploads_db: reconcile complete (backfilled=%d collections, pruned=%d ghost rows)",
        backfilled, pruned,
    )
    return {"backfilled_collections": backfilled, "pruned_rows": pruned}


def _list_collection_names(qdrant) -> list[str]:
    return [c.name for c in qdrant._client.get_collections().collections]


async def _scroll_user_rows(qdrant, collection: str) -> dict[str, list[dict]]:
    """Scroll a single collection and group `list_user_files`-shaped rows by raw user.

    Reuses the same payload-shape contract as QdrantKB._list_user_files_sync,
    but doesn't pre-filter by user — we don't know the user yet.
    """
    return await asyncio.to_thread(_scroll_user_rows_sync, qdrant, collection)


def _scroll_user_rows_sync(qdrant, collection: str) -> dict[str, list[dict]]:
    by_user_file: dict[tuple[str, str], dict] = {}
    next_page = None
    while True:
        points, next_page = qdrant._client.scroll(
            collection_name=collection,
            limit=256,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            raw_user = str(payload.get("user") or "")
            fid = str(payload.get("file_id") or "")
            if not raw_user or not fid:
                continue
            key = (raw_user, fid)
            row = by_user_file.setdefault(key, {
                "file_id": fid,
                "filename": str(payload.get("filename") or ""),
                "mime": str(payload.get("mime") or ""),
                "bytes": int(payload.get("bytes") or 0),
                "kind": str(payload.get("kind") or "text"),
                "uploaded_at": str(payload.get("uploaded_at") or ""),
                "chunks": 0,
            })
            row["chunks"] += 1
        if next_page is None:
            break
    grouped: dict[str, list[dict]] = {}
    for (user, _fid), row in by_user_file.items():
        grouped.setdefault(user, []).append(row)
    return grouped


__all__ = ["UploadsDB", "reconcile_with_qdrant"]
