"""Per-user chat archive: SQLite source of truth + Qdrant search index.

Audrey calls `archive_turn()` after every chat completion (streaming or
not). The store writes raw user/assistant messages to SQLite, derives
Q+A-pair chunks, embeds them with `nomic-embed-text`, and upserts the
embeddings to Qdrant. The model can later look conversations back up via
the user-scoped `chat_history_search` tool — search filters by `user`
payload field, never by tag substring.

Why SQLite plus Qdrant:
  - SQLite is durable, exportable, prunable, and re-indexable.
  - Qdrant is good at semantic search.
  - If the vector index ever needs rebuilding, SQLite has the source text.

Idempotency / dedup:
  - `message_id` is `sha256(user|conversation_id|role|content|minute_bucket)[:32]`.
  - `INSERT OR IGNORE` on `messages` collapses retries within the same minute
    and full-history re-sends.

Chunking:
  - One chunk per (user_turn, following_assistant_turn) pair.
  - Soft cap: `CHAT_ARCHIVE_CHUNK_MAX_CHARS`. Oversize chunks split at
    sentence boundaries with `CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS` overlap.
  - Single-message chunks lose all "what was I asking" context at search
    time, so we deliberately don't ship them.

Best-effort writes: the archive must never break chat. The Audrey-side
client logs and continues on any HTTP failure; this module raises only
on programmer errors (missing user, etc.) — write-time embedding/Qdrant
failures are logged and the chunk is left in SQLite with `indexed_at IS
NULL`. The scheduled maintainer retries those rows and durable deletion
outboxes without putting either repair path on the chat request lifecycle.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

log = logging.getLogger(__name__)


# ─── Public dataclasses ───────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class ArchiveMessage:
    """One turn — either user or assistant — as written to SQLite."""
    message_id: str
    conversation_id: str
    user: str
    role: str       # "user" | "assistant"
    content: str
    created_at: str
    archived_at: str
    partial: bool   # True iff stream was cut short before completion
    virtual_model: str = ""
    concrete_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(slots=True, frozen=True)
class ChatExportMessage:
    """One current-user message in a portable archive export page."""
    message_id: str
    conversation_id: str
    conversation_title: str
    conversation_created_at: str
    conversation_updated_at: str
    role: str
    content: str
    created_at: str
    archived_at: str
    partial: bool
    virtual_model: str
    concrete_model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True, frozen=True)
class SearchHit:
    """One hit from a `chat_history_search` call — snippet-first."""
    conversation_id: str
    chunk_id: str
    created_at: str
    snippet: str
    score: float


# ─── Helpers ──────────────────────────────────────────────────────────

_DEFAULT_SNIPPET_CHARS = 600
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds")


def _canonical_cutoff(value: str) -> str:
    """Require an aware, non-future cutoff and normalize it to UTC."""
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError("cutoff_at must be an ISO-8601 timestamp") from e
    if parsed.tzinfo is None:
        raise ValueError("cutoff_at must include a timezone")
    parsed = parsed.astimezone(_dt.UTC)
    if parsed > _dt.datetime.now(_dt.UTC):
        raise ValueError("cutoff_at cannot be in the future")
    return parsed.isoformat(timespec="microseconds")


def _encode_export_cursor(created_at: str, message_id: str) -> str:
    raw = json.dumps([created_at, message_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_export_cursor(cursor: str) -> tuple[str, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(cursor + padding))
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ValueError("invalid chat export cursor") from e
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("invalid chat export cursor")
    return value[0], value[1]


def _minute_bucket(iso_ts: str) -> str:
    """Round an ISO timestamp down to the minute. Retries within the same
    minute hash to the same `message_id` so dedup actually fires."""
    # 2026-05-08T12:34:56+00:00 -> 2026-05-08T12:34
    return iso_ts[:16]


def derive_message_id(
    user: str,
    conversation_id: str,
    role: str,
    content: str,
    created_at: str,
    archive_id: str = "",
) -> str:
    """Deterministic id so local-outbox retries collapse across restarts."""
    h = hashlib.sha256()
    h.update(user.encode("utf-8"))
    h.update(b"|")
    h.update(conversation_id.encode("utf-8"))
    h.update(b"|")
    h.update(role.encode("utf-8"))
    h.update(b"|")
    if archive_id:
        h.update(b"archive|")
        h.update(archive_id.encode("utf-8"))
    else:
        # Backward-compatible id for older callers without a delivery id.
        h.update(content.encode("utf-8"))
        h.update(b"|")
        h.update(_minute_bucket(created_at).encode("utf-8"))
    return h.hexdigest()[:32]


def _chunk_id(conversation_id: str, message_ids: list[str]) -> str:
    """Stable chunk id from its constituent messages, so re-archiving
    the same Q+A pair upserts in place rather than duplicating."""
    h = hashlib.sha256()
    h.update(conversation_id.encode("utf-8"))
    for mid in message_ids:
        h.update(b"|")
        h.update(mid.encode("utf-8"))
    return h.hexdigest()[:32]


def _qdrant_point_id(chunk_id: str) -> str:
    """Qdrant requires UUID or unsigned int point ids. Map our hex-32
    chunk id to a deterministic UUID5 so upserts collide correctly."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chat_archive|{chunk_id}"))


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split `text` at sentence boundaries, honoring the soft cap with
    `overlap` chars of context bleed between adjacent splits."""
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        candidate = f"{buf} {s}".strip() if buf else s
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        # buf is full — emit it, seed next buf with the overlap tail.
        if buf:
            out.append(buf)
            buf = (buf[-overlap:] + " " + s).strip() if overlap > 0 else s
        else:
            # `s` alone exceeds max_chars — hard split.
            for i in range(0, len(s), max_chars - overlap):
                out.append(s[i:i + max_chars])
            buf = ""
    if buf:
        out.append(buf)
    return out


def build_chunks(
    *,
    user: str,
    conversation_id: str,
    user_message_id: str,
    user_content: str,
    assistant_message_id: str,
    assistant_content: str,
    created_at: str,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    """Produce one or more Q+A-pair chunks ready for SQLite + Qdrant.

    The chunk text itself includes both turns so search snippets carry
    the question alongside the answer — matching just an assistant turn
    loses the "what was I asking" context that makes archive search
    useful.
    """
    pair_text = f"User: {user_content}\nAssistant: {assistant_content}"
    pieces = _split_long(pair_text, max_chars, overlap_chars)
    out: list[dict[str, Any]] = []
    message_ids = [user_message_id, assistant_message_id]
    if len(pieces) == 1:
        out.append({
            "chunk_id": _chunk_id(conversation_id, [*message_ids, "0"]),
            "user": user,
            "conversation_id": conversation_id,
            "message_ids": message_ids,
            "text": pieces[0],
            "created_at": created_at,
        })
        return out
    for idx, piece in enumerate(pieces):
        out.append({
            "chunk_id": _chunk_id(conversation_id, [*message_ids, str(idx)]),
            "user": user,
            "conversation_id": conversation_id,
            "message_ids": message_ids,
            "text": piece,
            "created_at": created_at,
        })
    return out


def _snippet(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# ─── Embedding (mirrors db.MemoryStore._embed) ────────────────────────

class _EmbedError(RuntimeError):
    pass


async def _embed(
    http: httpx.AsyncClient, model: str, text: str, keep_alive: str = "",
) -> list[float]:
    payload: dict[str, Any] = {"model": model, "input": [text]}
    if keep_alive:
        # Same reasoning as MemoryStore._embed: Ollama's 5-minute default
        # evicts the embedder between bursts, and the cold reload costs ~70x a
        # warm call. Both stores share one embedder, so both must pin it —
        # otherwise whichever ran last decides how long it stays.
        payload["keep_alive"] = keep_alive
    try:
        r = await http.post("/api/embed", json=payload)
    except httpx.HTTPError as e:
        raise _EmbedError(f"transport: {type(e).__name__}: {e}") from e
    if r.status_code >= 400:
        raise _EmbedError(f"/api/embed -> {r.status_code}: {r.text[:200]}")
    body = r.json()
    vecs = body.get("embeddings") or []
    if not vecs or not isinstance(vecs[0], list):
        raise _EmbedError(f"unexpected shape: {body!r}")
    return vecs[0]


# ─── Schema ───────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user            TEXT NOT NULL,
    title           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_message_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    message_id        TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL,
    user              TEXT NOT NULL,
    role              TEXT NOT NULL,
    content           TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    archived_at       TEXT NOT NULL,
    partial           INTEGER NOT NULL DEFAULT 0,
    virtual_model     TEXT,
    concrete_model    TEXT,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS archive_chunks (
    chunk_id         TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL,
    user             TEXT NOT NULL,
    message_ids_json TEXT NOT NULL,
    text             TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    indexed_at       TEXT,
    index_attempts   INTEGER NOT NULL DEFAULT 0,
    index_last_attempt_at TEXT,
    index_last_error TEXT
);
CREATE TABLE IF NOT EXISTS archive_deletion_outbox (
    chunk_id         TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL,
    message_ids_json TEXT NOT NULL,
    point_id         TEXT NOT NULL,
    requested_at     TEXT NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_attempt_at  TEXT,
    last_error       TEXT
);
CREATE TABLE IF NOT EXISTS archive_conversation_deletions (
    user             TEXT NOT NULL,
    conversation_id  TEXT NOT NULL,
    cutoff_at        TEXT NOT NULL,
    requested_at     TEXT NOT NULL,
    completed_at     TEXT,
    PRIMARY KEY (user, conversation_id)
);
CREATE TABLE IF NOT EXISTS archive_user_purges (
    purge_id               TEXT PRIMARY KEY,
    user                   TEXT NOT NULL,
    cutoff_at              TEXT NOT NULL,
    requested_at           TEXT NOT NULL,
    memory_completed_at    TEXT,
    memory_attempts        INTEGER NOT NULL DEFAULT 0,
    memory_last_attempt_at TEXT,
    memory_last_error      TEXT,
    completed_at           TEXT
);
CREATE TABLE IF NOT EXISTS archive_maintenance_state (
    operation       TEXT PRIMARY KEY,
    last_attempt_at TEXT NOT NULL,
    last_error      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_messages_user_conv ON messages(user, conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_created ON messages(user, created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON archive_chunks(user);
"""

_REPAIR_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chunks_repair
    ON archive_chunks(indexed_at, index_attempts);
CREATE INDEX IF NOT EXISTS idx_deletion_repair
    ON archive_deletion_outbox(attempts, requested_at);
CREATE INDEX IF NOT EXISTS idx_conversation_deletion_pending
    ON archive_conversation_deletions(completed_at, requested_at);
CREATE INDEX IF NOT EXISTS idx_user_purge_owner
    ON archive_user_purges(user, cutoff_at);
CREATE INDEX IF NOT EXISTS idx_user_purge_pending
    ON archive_user_purges(completed_at, requested_at);
"""


# ─── Store ────────────────────────────────────────────────────────────

class ChatArchiveStore:
    """Async SQLite + Qdrant chat archive.

    Lifecycle:
      `init()` opens the SQLite file, creates the schema, ensures the
      Qdrant collection. `aclose()` closes both connections.

    Concurrency:
      One `aiosqlite.Connection` is held open for the lifetime of the
      store. SQLite's WAL mode + `aiosqlite`'s thread-pool serialization
      make this safe for the request volumes Audrey actually sees.
    """

    def __init__(
        self,
        *,
        sqlite_path: Path,
        qdrant_url: str,
        ollama_url: str,
        collection: str,
        embed_model: str,
        embed_dim: int,
        embed_timeout_s: float,
        chunk_max_chars: int,
        chunk_overlap_chars: int,
        search_threshold: float,
        retention_days: int,
        max_bytes: int,
        embed_keep_alive: str = "",
        repair_batch_size: int = 50,
        max_retry_attempts: int = 5,
    ) -> None:
        if max_bytes:
            raise ValueError("CHAT_ARCHIVE_MAX_BYTES is not implemented; it must be 0")
        self._sqlite_path = sqlite_path
        self._qdrant = AsyncQdrantClient(url=qdrant_url)
        self._http = httpx.AsyncClient(base_url=ollama_url, timeout=embed_timeout_s)
        self._collection = collection
        self._embed_model = embed_model
        self._embed_dim = embed_dim
        self._chunk_max = chunk_max_chars
        self._chunk_overlap = chunk_overlap_chars
        self._threshold = search_threshold
        self._retention_days = retention_days
        self._embed_keep_alive = embed_keep_alive
        self._max_bytes = max_bytes
        self._repair_batch_size = max(1, repair_batch_size)
        self._max_retry_attempts = max(1, max_retry_attempts)
        self._db: aiosqlite.Connection | None = None
        self._db_lock = asyncio.Lock()
        self._qdrant_write_lock = asyncio.Lock()
        self._maintenance_lock = asyncio.Lock()

    async def init(self) -> None:
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._sqlite_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        # `executescript` runs the multi-statement schema in one shot.
        await self._db.executescript(_SCHEMA)
        await self._migrate_repair_schema()
        await self._db.executescript(_REPAIR_INDEXES)
        await self._db.commit()
        await self._ensure_collection()
        log.info(
            "chat_archive: ready sqlite=%s qdrant_collection=%s dim=%d retention_days=%d",
            self._sqlite_path, self._collection, self._embed_dim, self._retention_days,
        )

    async def _migrate_repair_schema(self) -> None:
        """Add repair metadata to databases created before Campaign 3."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        cursor = await self._db.execute("PRAGMA table_info(archive_chunks)")
        columns = {str(row[1]) for row in await cursor.fetchall()}
        await cursor.close()
        additions = {
            "index_attempts": (
                "ALTER TABLE archive_chunks ADD COLUMN "
                "index_attempts INTEGER NOT NULL DEFAULT 0"
            ),
            "index_last_attempt_at": (
                "ALTER TABLE archive_chunks ADD COLUMN "
                "index_last_attempt_at TEXT"
            ),
            "index_last_error": (
                "ALTER TABLE archive_chunks ADD COLUMN index_last_error TEXT"
            ),
        }
        for name, statement in additions.items():
            if name not in columns:
                await self._db.execute(statement)

    async def aclose(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
        await self._qdrant.close()
        await self._http.aclose()

    async def _ensure_collection(self) -> None:
        existing = {c.name for c in (await self._qdrant.get_collections()).collections}
        if self._collection in existing:
            return
        await self._qdrant.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(
                size=self._embed_dim,
                distance=qm.Distance.COSINE,
            ),
        )
        await self._qdrant.create_payload_index(
            collection_name=self._collection,
            field_name="user",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        log.info("chat_archive: created Qdrant collection %r", self._collection)

    # ─── Writes ────────────────────────────────────────────────────────

    async def user_purge_cutoff(self, *, user: str) -> str:
        """Return the newest durable account-purge cutoff for one user."""
        if not user:
            raise ValueError("user_purge_cutoff requires a non-empty user")
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        async with self._db_lock:
            cursor = await self._db.execute(
                "SELECT MAX(cutoff_at) FROM archive_user_purges WHERE user = ?",
                (user,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return str(row[0]) if row and row[0] else ""

    async def archive_turn(
        self,
        *,
        user: str,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        partial: bool = False,
        virtual_model: str = "",
        concrete_model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        archive_id: str = "",
        created_at: str = "",
    ) -> dict[str, Any]:
        """Archive one user-turn + assistant-turn pair.

        Returns a dict with the assigned ids and write counts, suitable
        for logging/metrics on the caller side. Never raises on Qdrant /
        embedding errors — those leave the SQLite chunk row with
        `indexed_at IS NULL` for later reconcile.
        """
        if not user:
            raise ValueError("archive_turn requires a non-empty user")
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        now = created_at or _now_iso()
        archived_at = _now_iso()
        user_msg_id = derive_message_id(
            user, conversation_id, "user", user_content, now, archive_id,
        )
        asst_msg_id = derive_message_id(
            user, conversation_id, "assistant", assistant_content, now, archive_id,
        )

        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT MAX(cutoff_at)
                FROM (
                    SELECT cutoff_at
                    FROM archive_conversation_deletions
                    WHERE user = ? AND conversation_id = ?
                    UNION ALL
                    SELECT cutoff_at
                    FROM archive_user_purges
                    WHERE user = ?
                )
                """,
                (user, conversation_id, user),
            )
            tombstone = await cursor.fetchone()
            await cursor.close()
            if tombstone is not None and tombstone[0] and now <= str(tombstone[0]):
                log.info(
                    "chat_archive: skipped pre-purge delivery user=%s conversation=%s",
                    user,
                    conversation_id,
                )
                return {
                    "conversation_id": conversation_id,
                    "user_message_id": user_msg_id,
                    "assistant_message_id": asst_msg_id,
                    "chunks": 0,
                    "indexed": 0,
                    "index_failed": 0,
                    "partial": partial,
                    "skipped_deleted": True,
                }

            # Conversation upsert: create on first sight, otherwise bump
            # last_message_at. We don't fight OWUI for `title` ownership.
            await self._db.execute(
                """
                INSERT INTO conversations (conversation_id, user, title, created_at, updated_at, last_message_at)
                VALUES (?, ?, NULL, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    last_message_at = excluded.last_message_at,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, user, now, now, now),
            )

            # Both messages: INSERT OR IGNORE so retries collapse.
            msg_rows = [
                (user_msg_id, conversation_id, user, "user", user_content,
                 now, archived_at, 0, virtual_model, concrete_model,
                 prompt_tokens, completion_tokens),
                (asst_msg_id, conversation_id, user, "assistant", assistant_content,
                 now, archived_at, 1 if partial else 0, virtual_model, concrete_model,
                 prompt_tokens, completion_tokens),
            ]
            await self._db.executemany(
                """
                INSERT OR IGNORE INTO messages
                (message_id, conversation_id, user, role, content,
                 created_at, archived_at, partial,
                 virtual_model, concrete_model, prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                msg_rows,
            )

            chunks = build_chunks(
                user=user,
                conversation_id=conversation_id,
                user_message_id=user_msg_id,
                user_content=user_content,
                assistant_message_id=asst_msg_id,
                assistant_content=assistant_content,
                created_at=now,
                max_chars=self._chunk_max,
                overlap_chars=self._chunk_overlap,
            )

            chunk_rows = [
                (c["chunk_id"], c["conversation_id"], c["user"],
                 ",".join(c["message_ids"]), c["text"], c["created_at"], None)
                for c in chunks
            ]
            await self._db.executemany(
                """
                INSERT OR REPLACE INTO archive_chunks
                (chunk_id, conversation_id, user, message_ids_json, text, created_at, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )
            await self._db.commit()

        indexed = 0
        index_failed = 0
        for chunk in chunks:
            if await self._index_chunk(chunk["chunk_id"]):
                indexed += 1
            else:
                index_failed += 1

        return {
            "conversation_id": conversation_id,
            "user_message_id": user_msg_id,
            "assistant_message_id": asst_msg_id,
            "chunks": len(chunks),
            "indexed": indexed,
            "index_failed": index_failed,
            "partial": partial,
        }

    async def _index_chunk(self, chunk_id: str) -> bool:
        """Index one SQLite chunk.

        The point id is deterministic, so a crash after Qdrant accepts the
        upsert but before SQLite records `indexed_at` is safe: the next pass
        overwrites the same point rather than creating a duplicate.
        """
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        attempted_at = _now_iso()
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT c.conversation_id, c.user, c.message_ids_json,
                       c.text, c.created_at, c.indexed_at, c.index_attempts
                FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.chunk_id = ? AND d.chunk_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM archive_user_purges AS p
                      WHERE p.user = c.user AND c.created_at <= p.cutoff_at
                  )
                """,
                (chunk_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if (
                row is None
                or row[5] is not None
                or int(row[6]) >= self._max_retry_attempts
            ):
                return False
            await self._db.execute(
                """
                UPDATE archive_chunks
                SET index_attempts = index_attempts + 1,
                    index_last_attempt_at = ?
                WHERE chunk_id = ?
                """,
                (attempted_at, chunk_id),
            )
            await self._record_attempt_locked("index", attempted_at)
            await self._db.commit()

        conversation_id, user, message_ids_json, chunk_text, created_at = row[:5]
        try:
            vec = await _embed(
                self._http, self._embed_model, str(chunk_text),
                self._embed_keep_alive,
            )
            async with self._qdrant_write_lock:
                # A retention sweep can tombstone the row while embedding is
                # in flight. Re-check before upserting so a confirmed delete
                # can never be followed by a stale-vector resurrection.
                async with self._db_lock:
                    cursor = await self._db.execute(
                        """
                        SELECT 1 FROM archive_chunks AS c
                        LEFT JOIN archive_deletion_outbox AS d
                          ON d.chunk_id = c.chunk_id
                        WHERE c.chunk_id = ? AND d.chunk_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM archive_user_purges AS p
                      WHERE p.user = c.user AND c.created_at <= p.cutoff_at
                  )
                        """,
                        (chunk_id,),
                    )
                    still_live = await cursor.fetchone()
                    await cursor.close()
                if still_live is None:
                    return False
                await self._qdrant.upsert(
                    collection_name=self._collection,
                    points=[
                        qm.PointStruct(
                            id=_qdrant_point_id(chunk_id),
                            vector=vec,
                            payload={
                                "user": str(user),
                                "conversation_id": str(conversation_id),
                                "chunk_id": chunk_id,
                                "message_ids": str(message_ids_json).split(","),
                                "created_at": str(created_at),
                                "text": str(chunk_text),
                            },
                        )
                    ],
                )
            async with self._db_lock:
                await self._db.execute(
                    """
                    UPDATE archive_chunks
                    SET indexed_at = ?, index_last_error = NULL
                    WHERE chunk_id = ?
                    """,
                    (_now_iso(), chunk_id),
                )
                await self._db.commit()
            return True
        except Exception as e:  # noqa: BLE001 — embed + Qdrant raise wide trees
            error = f"{type(e).__name__}: {e}"[:500]
            async with self._db_lock:
                await self._db.execute(
                    "UPDATE archive_chunks SET index_last_error = ? WHERE chunk_id = ?",
                    (error, chunk_id),
                )
                await self._record_attempt_locked("index", attempted_at, error)
                await self._db.commit()
            log.warning("chat_archive: index failed for chunk %s: %s", chunk_id, error)
            return False

    async def reindex_pending(
        self,
        *,
        limit: int | None = None,
        reset_exhausted: bool = False,
    ) -> dict[str, int]:
        """Retry a bounded batch of unindexed, non-tombstoned chunks."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        batch = self._repair_batch_size if limit is None else max(1, limit)
        async with self._db_lock:
            if reset_exhausted:
                await self._db.execute(
                    """
                    UPDATE archive_chunks
                    SET index_attempts = 0
                    WHERE indexed_at IS NULL
                      AND index_attempts >= ?
                      AND NOT EXISTS (
                          SELECT 1 FROM archive_deletion_outbox
                          WHERE archive_deletion_outbox.chunk_id = archive_chunks.chunk_id
                      )
                    """,
                    (self._max_retry_attempts,),
                )
                await self._db.commit()
            cursor = await self._db.execute(
                """
                SELECT c.chunk_id
                FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.indexed_at IS NULL
                  AND c.index_attempts < ?
                  AND d.chunk_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM archive_user_purges AS p
                      WHERE p.user = c.user AND c.created_at <= p.cutoff_at
                  )
                ORDER BY c.created_at, c.chunk_id
                LIMIT ?
                """,
                (self._max_retry_attempts, batch),
            )
            chunk_ids = [str(row[0]) for row in await cursor.fetchall()]
            await cursor.close()
        indexed = 0
        for chunk_id in chunk_ids:
            indexed += int(await self._index_chunk(chunk_id))
        return {
            "attempted": len(chunk_ids),
            "indexed": indexed,
            "failed": len(chunk_ids) - indexed,
        }

    # ─── Search ────────────────────────────────────────────────────────

    async def search(
        self,
        *,
        user: str,
        query: str,
        limit: int,
        snippet_chars: int = _DEFAULT_SNIPPET_CHARS,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SearchHit]:
        """Vector search over a single user's chat archive.

        Filter is `user == <user>` plus optional `created_at` range. The
        threshold is tuned looser than durable memory's because chat-archive
        recall is human-triggered ("did we talk about X?"), so a few low-
        relevance hits are tolerable when memory_search false-positives
        would not be.
        """
        if not user:
            raise ValueError("search requires a non-empty user")
        if not query.strip():
            return []
        cutoff_at = await self.user_purge_cutoff(user=user)
        try:
            qvec = await _embed(self._http, self._embed_model, query, self._embed_keep_alive)
        except _EmbedError as e:
            log.warning("chat_archive: embed failed for search query: %s", e)
            return []

        must: list[qm.FieldCondition] = [
            qm.FieldCondition(key="user", match=qm.MatchValue(value=user)),
        ]
        if date_from or date_to:
            must.append(qm.FieldCondition(
                key="created_at",
                range=qm.DatetimeRange(gte=date_from, lte=date_to),
            ))
        if cutoff_at:
            must.append(qm.FieldCondition(
                key="created_at",
                range=qm.DatetimeRange(gt=cutoff_at),
            ))

        result = await self._qdrant.query_points(
            collection_name=self._collection,
            query=qvec,
            limit=limit,
            score_threshold=self._threshold,
            query_filter=qm.Filter(must=must),
            with_payload=True,
        )
        payloads = [point.payload or {} for point in result.points]
        candidate_ids = [
            str(payload.get("chunk_id", ""))
            for payload in payloads
            if payload.get("chunk_id")
        ]
        pending_ids: set[str] = set()
        if candidate_ids:
            async with self._db_lock:
                for chunk_id in candidate_ids:
                    cursor = await self._db.execute(
                        """
                        SELECT 1 FROM archive_deletion_outbox
                        WHERE chunk_id = ?
                        """,
                        (chunk_id,),
                    )
                    if await cursor.fetchone() is not None:
                        pending_ids.add(chunk_id)
                    await cursor.close()

        out: list[SearchHit] = []
        for point, payload in zip(result.points, payloads, strict=True):
            chunk_id = str(payload.get("chunk_id", ""))
            created_at = str(payload.get("created_at", ""))
            if chunk_id in pending_ids or (cutoff_at and not created_at > cutoff_at):
                continue
            out.append(SearchHit(
                conversation_id=str(payload.get("conversation_id", "")),
                chunk_id=chunk_id,
                created_at=created_at,
                snippet=_snippet(str(payload.get("text", "")), snippet_chars),
                score=float(point.score or 0.0),
            ))
        return out

    async def export_user_messages(
        self,
        *,
        user: str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[ChatExportMessage], str | None]:
        """Return a stable, current-user-only page from the SQLite source.

        Rows already tombstoned in either deletion table are logically deleted
        and stay hidden even while Qdrant is unavailable. The cursor is an
        opaque encoding of the last created_at/message_id keyset pair.
        """
        if not user:
            raise ValueError("export_user_messages requires a non-empty user")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        after_created_at = ""
        after_message_id = ""
        if cursor:
            after_created_at, after_message_id = _decode_export_cursor(cursor)

        sql = """
            SELECT m.message_id, m.conversation_id, m.role, m.content,
                   m.created_at, m.archived_at, m.partial, m.virtual_model,
                   m.concrete_model, m.prompt_tokens, m.completion_tokens,
                   c.title, c.created_at, c.updated_at
            FROM messages AS m
            LEFT JOIN conversations AS c
              ON c.conversation_id = m.conversation_id AND c.user = m.user
            WHERE m.user = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM archive_deletion_outbox AS d
                  WHERE instr(
                      "," || d.message_ids_json || ",",
                      "," || m.message_id || ","
                  ) > 0
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM archive_conversation_deletions AS cd
                  WHERE cd.user = m.user
                    AND cd.conversation_id = m.conversation_id
                    AND m.created_at <= cd.cutoff_at
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM archive_user_purges AS p
                  WHERE p.user = m.user
                    AND m.created_at <= p.cutoff_at
              )
        """
        params: list[Any] = [user]
        if cursor:
            sql += """
              AND (m.created_at > ? OR (m.created_at = ? AND m.message_id > ?))
            """
            params.extend([after_created_at, after_created_at, after_message_id])
        sql += " ORDER BY m.created_at, m.message_id LIMIT ?"
        params.append(limit + 1)

        async with self._db_lock:
            db_cursor = await self._db.execute(sql, tuple(params))
            rows = await db_cursor.fetchall()
            await db_cursor.close()

        page_rows = rows[:limit]
        items = [
            ChatExportMessage(
                message_id=str(row[0]),
                conversation_id=str(row[1]),
                role=str(row[2]),
                content=str(row[3]),
                created_at=str(row[4]),
                archived_at=str(row[5]),
                partial=bool(row[6]),
                virtual_model=str(row[7] or ""),
                concrete_model=str(row[8] or ""),
                prompt_tokens=int(row[9] or 0),
                completion_tokens=int(row[10] or 0),
                conversation_title=str(row[11] or ""),
                conversation_created_at=str(row[12] or ""),
                conversation_updated_at=str(row[13] or ""),
            )
            for row in page_rows
        ]
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = _encode_export_cursor(last.created_at, last.message_id)
        return items, next_cursor

    async def request_conversation_deletion(
        self,
        *,
        user: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        """Durably hide and queue one current-user conversation for deletion."""
        if not user:
            raise ValueError("request_conversation_deletion requires a non-empty user")
        if not conversation_id:
            raise ValueError("request_conversation_deletion requires a non-empty conversation_id")
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        requested_at = _now_iso()
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT 1
                WHERE EXISTS (
                    SELECT 1 FROM conversations
                    WHERE user = ? AND conversation_id = ?
                )
                   OR EXISTS (
                    SELECT 1 FROM messages
                    WHERE user = ? AND conversation_id = ?
                )
                   OR EXISTS (
                    SELECT 1 FROM archive_chunks
                    WHERE user = ? AND conversation_id = ?
                )
                """,
                (
                    user,
                    conversation_id,
                    user,
                    conversation_id,
                    user,
                    conversation_id,
                ),
            )
            known = await cursor.fetchone() is not None
            await cursor.close()
            if not known:
                cursor = await self._db.execute(
                    """
                    SELECT 1 FROM archive_conversation_deletions
                    WHERE user = ? AND conversation_id = ?
                    """,
                    (user, conversation_id),
                )
                known = await cursor.fetchone() is not None
                await cursor.close()
            if not known:
                return None

            await self._db.execute(
                """
                INSERT INTO archive_conversation_deletions
                    (user, conversation_id, cutoff_at, requested_at, completed_at)
                VALUES (?, ?, ?, ?, NULL)
                ON CONFLICT(user, conversation_id) DO UPDATE SET
                    cutoff_at = CASE
                        WHEN excluded.cutoff_at >
                             archive_conversation_deletions.cutoff_at
                        THEN excluded.cutoff_at
                        ELSE archive_conversation_deletions.cutoff_at
                    END,
                    requested_at = excluded.requested_at,
                    completed_at = NULL
                """,
                (user, conversation_id, requested_at, requested_at),
            )
            cursor = await self._db.execute(
                """
                SELECT c.chunk_id, c.conversation_id, c.message_ids_json
                FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.user = ?
                  AND c.conversation_id = ?
                  AND c.created_at <= ?
                  AND d.chunk_id IS NULL
                ORDER BY c.created_at, c.chunk_id
                """,
                (user, conversation_id, requested_at),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            await self._db.executemany(
                """
                INSERT OR IGNORE INTO archive_deletion_outbox
                    (chunk_id, conversation_id, message_ids_json, point_id,
                     requested_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(row[0]),
                        str(row[1]),
                        str(row[2]),
                        _qdrant_point_id(str(row[0])),
                        requested_at,
                    )
                    for row in rows
                ],
            )
            cursor = await self._db.execute(
                """
                SELECT COUNT(*) FROM archive_deletion_outbox AS d
                JOIN archive_chunks AS c ON c.chunk_id = d.chunk_id
                WHERE c.user = ? AND c.conversation_id = ?
                """,
                (user, conversation_id),
            )
            pending = int((await cursor.fetchone())[0])
            await cursor.close()
            await self._db.commit()

        log.info(
            "chat_archive: conversation deletion queued user=%s conversation=%s "
            "chunks_queued=%d pending=%d",
            user,
            conversation_id,
            len(rows),
            pending,
        )
        return {
            "conversation_id": conversation_id,
            "requested_at": requested_at,
            "status": "pending",
            "chunks_queued": len(rows),
            "deletions_pending": pending,
        }

    async def request_user_purge(
        self,
        *,
        user: str,
        purge_id: str,
        cutoff_at: str,
        memory_store: Any,
    ) -> dict[str, Any]:
        """Create one idempotent account tombstone and begin physical cleanup."""
        if not user:
            raise ValueError("request_user_purge requires a non-empty user")
        if not purge_id:
            raise ValueError("request_user_purge requires a non-empty purge_id")
        if not cutoff_at:
            raise ValueError("request_user_purge requires a non-empty cutoff_at")
        cutoff_at = _canonical_cutoff(cutoff_at)
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        requested_at = _now_iso()
        async with self._db_lock:
            cursor = await self._db.execute(
                "SELECT user, cutoff_at FROM archive_user_purges WHERE purge_id = ?",
                (purge_id,),
            )
            existing = await cursor.fetchone()
            await cursor.close()
            if existing is not None:
                if str(existing[0]) != user or str(existing[1]) != cutoff_at:
                    raise ValueError("purge_id is already bound to another request")
            else:
                await self._db.execute(
                    """
                    INSERT INTO archive_user_purges
                        (purge_id, user, cutoff_at, requested_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (purge_id, user, cutoff_at, requested_at),
                )
                await self._db.commit()

        await self._queue_user_purge_chunks(purge_id=purge_id)
        await self._repair_user_purge_memories(
            memory_store=memory_store,
            purge_id=purge_id,
        )
        await self._finalize_user_purges()
        result = await self.user_purge_status(user=user, purge_id=purge_id)
        if result is None:
            raise RuntimeError("account purge disappeared after creation")
        return result

    async def _queue_user_purge_chunks(self, *, purge_id: str = "") -> int:
        """Queue one bounded batch of pre-cutoff chat vectors for deletion."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT DISTINCT c.chunk_id, c.conversation_id, c.message_ids_json
                FROM archive_user_purges AS p
                JOIN archive_chunks AS c
                  ON c.user = p.user AND c.created_at <= p.cutoff_at
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE p.completed_at IS NULL
                  AND (? = '' OR p.purge_id = ?)
                  AND d.chunk_id IS NULL
                ORDER BY c.created_at, c.chunk_id
                LIMIT ?
                """,
                (purge_id, purge_id, self._repair_batch_size),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            requested_at = _now_iso()
            await self._db.executemany(
                """
                INSERT OR IGNORE INTO archive_deletion_outbox
                    (chunk_id, conversation_id, message_ids_json, point_id,
                     requested_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(row[0]),
                        str(row[1]),
                        str(row[2]),
                        _qdrant_point_id(str(row[0])),
                        requested_at,
                    )
                    for row in rows
                ],
            )
            await self._db.commit()
        return len(rows)

    async def _repair_user_purge_memories(
        self,
        *,
        memory_store: Any,
        purge_id: str = "",
    ) -> dict[str, int]:
        """Retry acknowledged Qdrant memory deletion for pending purges."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        if memory_store is None:
            return {"attempted": 0, "completed": 0, "failed": 0}

        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT purge_id, user, cutoff_at
                FROM archive_user_purges
                WHERE memory_completed_at IS NULL
                  AND memory_attempts < ?
                  AND (? = '' OR purge_id = ?)
                ORDER BY requested_at, purge_id
                LIMIT ?
                """,
                (
                    self._max_retry_attempts,
                    purge_id,
                    purge_id,
                    self._repair_batch_size,
                ),
            )
            rows = await cursor.fetchall()
            await cursor.close()

        completed = 0
        failed = 0
        for row_purge_id, user, cutoff_at in rows:
            attempted_at = _now_iso()
            async with self._db_lock:
                await self._db.execute(
                    """
                    UPDATE archive_user_purges
                    SET memory_attempts = memory_attempts + 1,
                        memory_last_attempt_at = ?
                    WHERE purge_id = ? AND memory_completed_at IS NULL
                    """,
                    (attempted_at, row_purge_id),
                )
                await self._db.commit()
            try:
                await memory_store.delete_user_before(
                    user=str(user),
                    cutoff_at=str(cutoff_at),
                )
            except Exception as e:  # noqa: BLE001 — Qdrant raises a wide tree
                error = f"{type(e).__name__}: {e}"[:500]
                async with self._db_lock:
                    await self._db.execute(
                        """
                        UPDATE archive_user_purges
                        SET memory_last_error = ?
                        WHERE purge_id = ?
                        """,
                        (error, row_purge_id),
                    )
                    await self._db.commit()
                failed += 1
                log.warning(
                    "chat_archive: account purge memory delete deferred purge=%s",
                    row_purge_id,
                )
                continue

            async with self._db_lock:
                await self._db.execute(
                    """
                    UPDATE archive_user_purges
                    SET memory_completed_at = ?, memory_last_error = NULL
                    WHERE purge_id = ?
                    """,
                    (_now_iso(), row_purge_id),
                )
                await self._db.commit()
            completed += 1

        return {
            "attempted": len(rows),
            "completed": completed,
            "failed": failed,
        }

    async def _finalize_user_purges(self) -> int:
        """Finalize SQLite source rows after memory and chat vector acks."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        completed = 0
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT purge_id, user, cutoff_at
                FROM archive_user_purges
                WHERE completed_at IS NULL AND memory_completed_at IS NOT NULL
                ORDER BY requested_at, purge_id
                LIMIT ?
                """,
                (self._repair_batch_size,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            for purge_id, user, cutoff_at in rows:
                cursor = await self._db.execute(
                    """
                    SELECT 1 FROM archive_chunks
                    WHERE user = ? AND created_at <= ?
                    LIMIT 1
                    """,
                    (user, cutoff_at),
                )
                old_chunk_remains = await cursor.fetchone() is not None
                await cursor.close()
                if old_chunk_remains:
                    continue

                await self._db.execute(
                    "DELETE FROM messages WHERE user = ? AND created_at <= ?",
                    (user, cutoff_at),
                )
                await self._db.execute(
                    """
                    DELETE FROM conversations
                    WHERE user = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM messages
                          WHERE messages.user = conversations.user
                            AND messages.conversation_id = conversations.conversation_id
                      )
                    """,
                    (user,),
                )
                await self._db.execute(
                    "UPDATE archive_user_purges SET completed_at = ? WHERE purge_id = ?",
                    (_now_iso(), purge_id),
                )
                completed += 1
            await self._db.commit()
        return completed

    async def user_purge_status(
        self,
        *,
        user: str,
        purge_id: str,
    ) -> dict[str, Any] | None:
        """Return sanitized progress for one exact-owner purge receipt."""
        if not user:
            raise ValueError("user_purge_status requires a non-empty user")
        if not purge_id:
            raise ValueError("user_purge_status requires a non-empty purge_id")
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT cutoff_at, requested_at, memory_completed_at,
                       memory_attempts, memory_last_error, completed_at
                FROM archive_user_purges
                WHERE user = ? AND purge_id = ?
                """,
                (user, purge_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            cutoff_at = str(row[0])

            cursor = await self._db.execute(
                """
                SELECT COUNT(*)
                FROM archive_chunks
                WHERE user = ? AND created_at <= ?
                """,
                (user, cutoff_at),
            )
            chat_pending = int((await cursor.fetchone())[0])
            await cursor.close()
            cursor = await self._db.execute(
                """
                SELECT COALESCE(SUM(d.attempts), 0),
                       COALESCE(SUM(CASE WHEN d.last_error IS NOT NULL
                                           AND length(d.last_error) > 0
                                         THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN d.attempts >= ? THEN 1 ELSE 0 END), 0)
                FROM archive_deletion_outbox AS d
                JOIN archive_chunks AS c ON c.chunk_id = d.chunk_id
                WHERE c.user = ? AND c.created_at <= ?
                """,
                (self._max_retry_attempts, user, cutoff_at),
            )
            delete_row = await cursor.fetchone()
            await cursor.close()

        memory_completed = bool(row[2])
        memory_attempts = int(row[3] or 0)
        memory_with_error = bool(row[4])
        memory_exhausted = (
            not memory_completed and memory_attempts >= self._max_retry_attempts
        )
        chat_attempts = int(delete_row[0] or 0)
        chat_with_error = int(delete_row[1] or 0)
        chat_exhausted = int(delete_row[2] or 0)
        completed_at = str(row[5] or "")
        if completed_at:
            purge_status = "completed"
        elif memory_exhausted or chat_exhausted:
            purge_status = "attention_required"
        else:
            purge_status = "pending"
        return {
            "purge_id": purge_id,
            "cutoff_at": cutoff_at,
            "requested_at": str(row[1]),
            "status": purge_status,
            "completed_at": completed_at,
            "memory": {
                "completed": memory_completed,
                "attempts": memory_attempts,
                "with_error": memory_with_error,
                "exhausted": memory_exhausted,
            },
            "chat": {
                "pending": chat_pending,
                "attempts": chat_attempts,
                "with_error": chat_with_error,
                "exhausted": chat_exhausted,
            },
        }

    # ─── Durable repair and retention ─────────────────────────────────

    async def _queue_retention(self) -> int:
        """Tombstone one bounded batch without deleting its source rows."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        if self._retention_days <= 0:
            return 0

        cutoff = (
            _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=self._retention_days)
        ).isoformat(timespec="seconds")
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT c.chunk_id, c.conversation_id, c.message_ids_json
                FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.created_at < ? AND d.chunk_id IS NULL
                ORDER BY c.created_at, c.chunk_id
                LIMIT ?
                """,
                (cutoff, self._repair_batch_size),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            requested_at = _now_iso()
            await self._db.executemany(
                """
                INSERT OR IGNORE INTO archive_deletion_outbox
                (chunk_id, conversation_id, message_ids_json, point_id, requested_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(row[0]), str(row[1]), str(row[2]),
                        _qdrant_point_id(str(row[0])), requested_at,
                    )
                    for row in rows
                ],
            )
            await self._db.commit()
        return len(rows)

    async def _delete_pending(
        self,
        *,
        reset_exhausted: bool = False,
    ) -> dict[str, int]:
        """Delete one bounded outbox batch, finalizing SQLite only on ack."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        async with self._db_lock:
            if reset_exhausted:
                await self._db.execute(
                    """
                    UPDATE archive_deletion_outbox SET attempts = 0
                    WHERE attempts >= ?
                    """,
                    (self._max_retry_attempts,),
                )
                await self._db.commit()
            cursor = await self._db.execute(
                """
                SELECT chunk_id, conversation_id, message_ids_json, point_id
                FROM archive_deletion_outbox
                WHERE attempts < ?
                ORDER BY requested_at, chunk_id
                LIMIT ?
                """,
                (self._max_retry_attempts, self._repair_batch_size),
            )
            rows = await cursor.fetchall()
            await cursor.close()

        chunks_deleted = 0
        messages_deleted = 0
        qdrant_deleted = 0
        failed = 0
        for chunk_id, conversation_id, message_ids_json, point_id in rows:
            attempted_at = _now_iso()
            async with self._db_lock:
                await self._db.execute(
                    """
                    UPDATE archive_deletion_outbox
                    SET attempts = attempts + 1, last_attempt_at = ?
                    WHERE chunk_id = ?
                    """,
                    (attempted_at, chunk_id),
                )
                await self._record_attempt_locked("delete", attempted_at)
                await self._db.commit()
            try:
                async with self._qdrant_write_lock:
                    await self._qdrant.delete(
                        collection_name=self._collection,
                        points_selector=qm.PointIdsList(points=[str(point_id)]),
                        wait=True,
                    )
                    async with self._db_lock:
                        await self._db.execute(
                            "DELETE FROM archive_chunks WHERE chunk_id = ?",
                            (chunk_id,),
                        )
                        cursor = await self._db.execute("SELECT changes()")
                        chunks_deleted += int((await cursor.fetchone())[0])
                        await cursor.close()

                        cursor = await self._db.execute(
                            """
                            SELECT 1 FROM archive_chunks
                            WHERE message_ids_json = ? LIMIT 1
                            """,
                            (message_ids_json,),
                        )
                        messages_still_referenced = await cursor.fetchone()
                        await cursor.close()
                        if messages_still_referenced is None:
                            message_ids = [
                                value for value in str(message_ids_json).split(",")
                                if value
                            ]
                            for message_id in message_ids:
                                cursor = await self._db.execute(
                                    "DELETE FROM messages WHERE message_id = ?",
                                    (message_id,),
                                )
                                messages_deleted += max(0, int(cursor.rowcount))
                                await cursor.close()

                        await self._db.execute(
                            """
                            DELETE FROM conversations
                            WHERE conversation_id = ?
                              AND NOT EXISTS (
                                  SELECT 1 FROM messages
                                  WHERE conversation_id = ?
                              )
                            """,
                            (conversation_id, conversation_id),
                        )
                        await self._db.execute(
                            "DELETE FROM archive_deletion_outbox WHERE chunk_id = ?",
                            (chunk_id,),
                        )
                        await self._db.commit()
                qdrant_deleted += 1
            except Exception as e:  # noqa: BLE001 — SQLite + Qdrant raise wide trees
                error = f"{type(e).__name__}: {e}"[:500]
                async with self._db_lock:
                    await self._db.rollback()
                    await self._db.execute(
                        """
                        UPDATE archive_deletion_outbox
                        SET last_error = ? WHERE chunk_id = ?
                        """,
                        (error, chunk_id),
                    )
                    await self._record_attempt_locked("delete", attempted_at, error)
                    await self._db.commit()
                failed += 1
                log.warning(
                    "chat_archive: delete failed for chunk %s: %s",
                    chunk_id, error,
                )

        return {
            "attempted": len(rows),
            "chunks_deleted": chunks_deleted,
            "messages_deleted": messages_deleted,
            "qdrant_deleted": qdrant_deleted,
            "failed": failed,
        }

    async def _deletions_pending(self) -> int:
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        async with self._db_lock:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM archive_deletion_outbox"
            )
            count = int((await cursor.fetchone())[0])
            await cursor.close()
        return count

    async def _finalize_conversation_deletions(self) -> int:
        """Finalize source rows only after every old chunk has been acknowledged."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        completed = 0
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT user, conversation_id, cutoff_at
                FROM archive_conversation_deletions
                WHERE completed_at IS NULL
                ORDER BY requested_at, user, conversation_id
                LIMIT ?
                """,
                (self._repair_batch_size,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            for user, conversation_id, cutoff_at in rows:
                cursor = await self._db.execute(
                    """
                    SELECT 1 FROM archive_chunks
                    WHERE user = ?
                      AND conversation_id = ?
                      AND created_at <= ?
                    LIMIT 1
                    """,
                    (user, conversation_id, cutoff_at),
                )
                old_chunk_remains = await cursor.fetchone() is not None
                await cursor.close()
                if old_chunk_remains:
                    continue

                await self._db.execute(
                    """
                    DELETE FROM messages
                    WHERE user = ?
                      AND conversation_id = ?
                      AND created_at <= ?
                    """,
                    (user, conversation_id, cutoff_at),
                )
                await self._db.execute(
                    """
                    DELETE FROM conversations
                    WHERE user = ?
                      AND conversation_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM messages
                          WHERE user = ? AND conversation_id = ?
                      )
                    """,
                    (user, conversation_id, user, conversation_id),
                )
                await self._db.execute(
                    """
                    UPDATE archive_conversation_deletions
                    SET completed_at = ?
                    WHERE user = ? AND conversation_id = ?
                    """,
                    (_now_iso(), user, conversation_id),
                )
                completed += 1
            await self._db.commit()
        return completed

    async def prune(
        self,
        *,
        retry_exhausted: bool = False,
        memory_store: Any = None,
    ) -> dict[str, int]:
        """Queue expired chunks and process a bounded deletion-outbox batch."""
        async with self._maintenance_lock:
            await self._queue_user_purge_chunks()
            await self._repair_user_purge_memories(memory_store=memory_store)
            queued = await self._queue_retention()
            deleted = await self._delete_pending(
                reset_exhausted=retry_exhausted,
            )
            await self._finalize_conversation_deletions()
            await self._finalize_user_purges()
            reindexed = await self.reindex_pending(
                reset_exhausted=retry_exhausted,
            )
            pending = await self._deletions_pending()
        return {
            "deletions_queued": queued,
            "messages_deleted": deleted["messages_deleted"],
            "chunks_deleted": deleted["chunks_deleted"],
            "qdrant_deleted": deleted["qdrant_deleted"],
            "delete_failed": deleted["failed"],
            "deletions_pending": pending,
            "reindex_attempted": reindexed["attempted"],
            "reindexed": reindexed["indexed"],
            "reindex_failed": reindexed["failed"],
        }

    async def maintain(self, *, memory_store: Any = None) -> dict[str, Any]:
        """Run one scheduled retention, deletion, and reindex repair pass."""
        async with self._maintenance_lock:
            purge_queued = await self._queue_user_purge_chunks()
            purge_memories = await self._repair_user_purge_memories(
                memory_store=memory_store,
            )
            queued = await self._queue_retention()
            deleted = await self._delete_pending()
            conversation_deletions_completed = await self._finalize_conversation_deletions()
            user_purges_completed = await self._finalize_user_purges()
            reindexed = await self.reindex_pending()
            pending = await self._deletions_pending()
        result: dict[str, Any] = {
            "deletions_queued": queued,
            "deletions_pending": pending,
            "delete": deleted,
            "reindex": reindexed,
        }
        if (
            queued
            or purge_queued
            or purge_memories["attempted"]
            or deleted["attempted"]
            or conversation_deletions_completed
            or user_purges_completed
            or reindexed["attempted"]
        ):
            log.info("chat_archive: maintenance result=%s", result)
        return result

    # ─── Stats (for admin visibility) ─────────────────────────────────

    async def _record_attempt_locked(
        self,
        operation: str,
        attempted_at: str,
        error: str = "",
    ) -> None:
        """Persist latest operation state while the caller holds `_db_lock`."""
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        await self._db.execute(
            """
            INSERT INTO archive_maintenance_state
                (operation, last_attempt_at, last_error)
            VALUES (?, ?, ?)
            ON CONFLICT(operation) DO UPDATE SET
                last_attempt_at = excluded.last_attempt_at,
                last_error = excluded.last_error
            WHERE excluded.last_attempt_at >=
                  archive_maintenance_state.last_attempt_at
            """,
            (operation, attempted_at, error),
        )

    async def user_stats(self, *, user: str) -> dict[str, dict[str, int]]:
        """Repair counts for exactly one authenticated archive owner."""
        if not user:
            raise ValueError("user_stats requires a non-empty user")
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        async with self._db_lock:
            async def scalar(sql: str, params: tuple[Any, ...] = ()) -> int:
                cursor = await self._db.execute(sql, params)
                value = int((await cursor.fetchone())[0])
                await cursor.close()
                return value

            indexing_pending = await scalar(
                """
                SELECT COUNT(*) FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.user = ?
                  AND c.indexed_at IS NULL
                  AND c.index_attempts < ?
                  AND d.chunk_id IS NULL
                """,
                (user, self._max_retry_attempts),
            )
            indexing_exhausted = await scalar(
                """
                SELECT COUNT(*) FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.user = ?
                  AND c.indexed_at IS NULL
                  AND c.index_attempts >= ?
                  AND d.chunk_id IS NULL
                """,
                (user, self._max_retry_attempts),
            )
            indexing_attempts = await scalar(
                """
                SELECT COALESCE(SUM(c.index_attempts), 0)
                FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.user = ?
                  AND c.indexed_at IS NULL
                  AND d.chunk_id IS NULL
                """,
                (user,),
            )
            indexing_with_error = await scalar(
                """
                SELECT COUNT(*) FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.user = ?
                  AND c.indexed_at IS NULL
                  AND c.index_last_error IS NOT NULL
                  AND length(c.index_last_error) > 0
                  AND d.chunk_id IS NULL
                """,
                (user,),
            )
            deletions_pending = await scalar(
                """
                SELECT COUNT(*) FROM archive_deletion_outbox AS d
                JOIN archive_chunks AS c ON c.chunk_id = d.chunk_id
                WHERE c.user = ?
                """,
                (user,),
            )
            deletions_exhausted = await scalar(
                """
                SELECT COUNT(*) FROM archive_deletion_outbox AS d
                JOIN archive_chunks AS c ON c.chunk_id = d.chunk_id
                WHERE c.user = ? AND d.attempts >= ?
                """,
                (user, self._max_retry_attempts),
            )
            deletion_attempts = await scalar(
                """
                SELECT COALESCE(SUM(d.attempts), 0)
                FROM archive_deletion_outbox AS d
                JOIN archive_chunks AS c ON c.chunk_id = d.chunk_id
                WHERE c.user = ?
                """,
                (user,),
            )
            deletion_with_error = await scalar(
                """
                SELECT COUNT(*) FROM archive_deletion_outbox AS d
                JOIN archive_chunks AS c ON c.chunk_id = d.chunk_id
                WHERE c.user = ?
                  AND d.last_error IS NOT NULL
                  AND length(d.last_error) > 0
                """,
                (user,),
            )
            conversation_pending = await scalar(
                """
                SELECT COUNT(*) FROM archive_conversation_deletions
                WHERE user = ? AND completed_at IS NULL
                """,
                (user,),
            )
            conversation_completed = await scalar(
                """
                SELECT COUNT(*) FROM archive_conversation_deletions
                WHERE user = ? AND completed_at IS NOT NULL
                """,
                (user,),
            )

        return {
            "indexing": {
                "pending": indexing_pending,
                "attempts": indexing_attempts,
                "with_error": indexing_with_error,
                "exhausted": indexing_exhausted,
                "completed": 0,
            },
            "deletions": {
                "pending": deletions_pending,
                "attempts": deletion_attempts,
                "with_error": deletion_with_error,
                "exhausted": deletions_exhausted,
                "completed": 0,
            },
            "conversation_deletions": {
                "pending": conversation_pending,
                "attempts": 0,
                "with_error": 0,
                "exhausted": 0,
                "completed": conversation_completed,
            },
        }

    async def stats(self) -> dict[str, Any]:
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")

        async with self._db_lock:
            async def scalar(sql: str, params: tuple[Any, ...] = ()) -> int:
                cursor = await self._db.execute(sql, params)
                value = int((await cursor.fetchone())[0])
                await cursor.close()
                return value

            msgs = await scalar("SELECT COUNT(*) FROM messages")
            chunks = await scalar("SELECT COUNT(*) FROM archive_chunks")
            unindexed = await scalar(
                "SELECT COUNT(*) FROM archive_chunks WHERE indexed_at IS NULL"
            )
            reindex_pending = await scalar(
                """
                SELECT COUNT(*) FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.indexed_at IS NULL
                  AND c.index_attempts < ?
                  AND d.chunk_id IS NULL
                """,
                (self._max_retry_attempts,),
            )
            reindex_exhausted = await scalar(
                """
                SELECT COUNT(*) FROM archive_chunks AS c
                LEFT JOIN archive_deletion_outbox AS d ON d.chunk_id = c.chunk_id
                WHERE c.indexed_at IS NULL
                  AND c.index_attempts >= ?
                  AND d.chunk_id IS NULL
                """,
                (self._max_retry_attempts,),
            )
            deletions_pending = await scalar(
                "SELECT COUNT(*) FROM archive_deletion_outbox"
            )
            deletions_exhausted = await scalar(
                """
                SELECT COUNT(*) FROM archive_deletion_outbox
                WHERE attempts >= ?
                """,
                (self._max_retry_attempts,),
            )
            conversation_deletions_pending = await scalar(
                """
                SELECT COUNT(*) FROM archive_conversation_deletions
                WHERE completed_at IS NULL
                """
            )
            conversation_deletions_completed = await scalar(
                """
                SELECT COUNT(*) FROM archive_conversation_deletions
                WHERE completed_at IS NOT NULL
                """
            )

            cursor = await self._db.execute(
                """
                SELECT last_attempt_at, last_error
                FROM archive_maintenance_state
                WHERE operation = 'index'
                """
            )
            index_last = await cursor.fetchone()
            await cursor.close()
            cursor = await self._db.execute(
                """
                SELECT last_attempt_at, last_error
                FROM archive_maintenance_state
                WHERE operation = 'delete'
                """
            )
            delete_last = await cursor.fetchone()
            await cursor.close()

        return {
            "messages": msgs,
            "chunks": chunks,
            "chunks_unindexed": unindexed,
            "chunks_reindex_pending": reindex_pending,
            "chunks_reindex_exhausted": reindex_exhausted,
            "deletions_pending": deletions_pending,
            "deletions_exhausted": deletions_exhausted,
            "conversation_deletions_pending": conversation_deletions_pending,
            "conversation_deletions_completed": conversation_deletions_completed,
            "index_last_attempt_at": str(index_last[0]) if index_last else "",
            "index_last_error": str(index_last[1] or "") if index_last else "",
            "delete_last_attempt_at": str(delete_last[0]) if delete_last else "",
            "delete_last_error": str(delete_last[1] or "") if delete_last else "",
        }


class ChatArchiveMaintainer:
    """Own the scheduled archive repair task and cancel it cleanly."""

    def __init__(
        self,
        store: ChatArchiveStore,
        *,
        interval_s: float,
        memory_store: Any = None,
    ) -> None:
        self._store = store
        self._memory_store = memory_store
        self._interval_s = max(0.0, interval_s)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._interval_s <= 0:
            log.info("chat_archive: scheduled maintenance disabled")
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(), name="chat-archive-maintenance"
        )
        log.info(
            "chat_archive: scheduled maintenance every %.0fs",
            self._interval_s,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                try:
                    if self._memory_store is None:
                        await self._store.maintain()
                    else:
                        await self._store.maintain(memory_store=self._memory_store)
                except Exception as e:  # noqa: BLE001 — loop must survive a pass
                    log.warning("chat_archive: maintenance pass failed: %s", e)
                await asyncio.sleep(self._interval_s)
        except asyncio.CancelledError:
            return


__all__ = [
    "ArchiveMessage",
    "ChatArchiveMaintainer",
    "SearchHit",
    "ChatArchiveStore",
    "build_chunks",
    "derive_message_id",
]
