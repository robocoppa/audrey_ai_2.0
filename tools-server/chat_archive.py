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
NULL` so a future reconcile can pick it up.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
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
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _minute_bucket(iso_ts: str) -> str:
    """Round an ISO timestamp down to the minute. Retries within the same
    minute hash to the same `message_id` so dedup actually fires."""
    # 2026-05-08T12:34:56+00:00 -> 2026-05-08T12:34
    return iso_ts[:16]


def derive_message_id(user: str, conversation_id: str, role: str, content: str, created_at: str) -> str:
    """Deterministic id so retries and full-history re-sends collapse."""
    h = hashlib.sha256()
    h.update(user.encode("utf-8"))
    h.update(b"|")
    h.update(conversation_id.encode("utf-8"))
    h.update(b"|")
    h.update(role.encode("utf-8"))
    h.update(b"|")
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


async def _embed(http: httpx.AsyncClient, model: str, text: str) -> list[float]:
    try:
        r = await http.post("/api/embed", json={"model": model, "input": [text]})
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
    indexed_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_user_conv ON messages(user, conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_created ON messages(user, created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON archive_chunks(user);
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
    ) -> None:
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
        self._max_bytes = max_bytes
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._sqlite_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        # `executescript` runs the multi-statement schema in one shot.
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._ensure_collection()
        log.info(
            "chat_archive: ready sqlite=%s qdrant_collection=%s dim=%d retention_days=%d",
            self._sqlite_path, self._collection, self._embed_dim, self._retention_days,
        )

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

        now = _now_iso()
        user_msg_id = derive_message_id(user, conversation_id, "user", user_content, now)
        asst_msg_id = derive_message_id(user, conversation_id, "assistant", assistant_content, now)

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
             now, now, 0, virtual_model, concrete_model, prompt_tokens, completion_tokens),
            (asst_msg_id, conversation_id, user, "assistant", assistant_content,
             now, now, 1 if partial else 0, virtual_model, concrete_model,
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
            try:
                vec = await _embed(self._http, self._embed_model, chunk["text"])
                await self._qdrant.upsert(
                    collection_name=self._collection,
                    points=[
                        qm.PointStruct(
                            id=_qdrant_point_id(chunk["chunk_id"]),
                            vector=vec,
                            payload={
                                "user": chunk["user"],
                                "conversation_id": chunk["conversation_id"],
                                "chunk_id": chunk["chunk_id"],
                                "message_ids": chunk["message_ids"],
                                "created_at": chunk["created_at"],
                                "text": chunk["text"],
                            },
                        )
                    ],
                )
                await self._db.execute(
                    "UPDATE archive_chunks SET indexed_at = ? WHERE chunk_id = ?",
                    (_now_iso(), chunk["chunk_id"]),
                )
                indexed += 1
            except _EmbedError as e:
                log.warning("chat_archive: embed failed for chunk %s: %s", chunk["chunk_id"], e)
                index_failed += 1
            except Exception as e:  # noqa: BLE001 — Qdrant client raises a wide tree
                log.warning("chat_archive: qdrant upsert failed for chunk %s: %s", chunk["chunk_id"], e)
                index_failed += 1
        await self._db.commit()

        return {
            "conversation_id": conversation_id,
            "user_message_id": user_msg_id,
            "assistant_message_id": asst_msg_id,
            "chunks": len(chunks),
            "indexed": indexed,
            "index_failed": index_failed,
            "partial": partial,
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
        try:
            qvec = await _embed(self._http, self._embed_model, query)
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

        result = await self._qdrant.query_points(
            collection_name=self._collection,
            query=qvec,
            limit=limit,
            score_threshold=self._threshold,
            query_filter=qm.Filter(must=must),
            with_payload=True,
        )
        out: list[SearchHit] = []
        for point in result.points:
            p = point.payload or {}
            out.append(SearchHit(
                conversation_id=str(p.get("conversation_id", "")),
                chunk_id=str(p.get("chunk_id", "")),
                created_at=str(p.get("created_at", "")),
                snippet=_snippet(str(p.get("text", "")), snippet_chars),
                score=float(point.score or 0.0),
            ))
        return out

    # ─── Prune ────────────────────────────────────────────────────────

    async def prune(self) -> dict[str, int]:
        """Apply retention rules. Returns counts deleted.

        Two rules, both opt-in via env:
          - `retention_days > 0`: delete messages and chunks older than
            cutoff, drop their Qdrant points.
          - `max_bytes > 0`: not implemented yet — would require a size
            estimate. Reserved knob; emits a zero count for now.
        """
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        if self._retention_days <= 0:
            return {"messages_deleted": 0, "chunks_deleted": 0, "qdrant_deleted": 0}

        cutoff = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=self._retention_days)).isoformat(timespec="seconds")
        # Get chunk ids first so we can delete from Qdrant before SQLite.
        cursor = await self._db.execute(
            "SELECT chunk_id FROM archive_chunks WHERE created_at < ?", (cutoff,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        chunk_ids = [r[0] for r in rows]

        qdrant_deleted = 0
        if chunk_ids:
            point_ids = [_qdrant_point_id(cid) for cid in chunk_ids]
            try:
                await self._qdrant.delete(
                    collection_name=self._collection,
                    points_selector=qm.PointIdsList(points=point_ids),
                )
                qdrant_deleted = len(point_ids)
            except Exception as e:  # noqa: BLE001
                log.warning("chat_archive: prune qdrant delete failed: %s", e)

        await self._db.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
        msg_count_cursor = await self._db.execute("SELECT changes()")
        msgs = (await msg_count_cursor.fetchone())[0]
        await msg_count_cursor.close()

        await self._db.execute("DELETE FROM archive_chunks WHERE created_at < ?", (cutoff,))
        chunk_count_cursor = await self._db.execute("SELECT changes()")
        chunks = (await chunk_count_cursor.fetchone())[0]
        await chunk_count_cursor.close()
        await self._db.commit()

        log.info(
            "chat_archive: prune cutoff=%s messages=%d chunks=%d qdrant=%d",
            cutoff, msgs, chunks, qdrant_deleted,
        )
        return {"messages_deleted": int(msgs), "chunks_deleted": int(chunks), "qdrant_deleted": qdrant_deleted}

    # ─── Stats (for /metrics gauge or admin debug) ─────────────────────

    async def stats(self) -> dict[str, int]:
        if self._db is None:
            raise RuntimeError("ChatArchiveStore.init() not called")
        cursor = await self._db.execute("SELECT COUNT(*) FROM messages")
        msgs = (await cursor.fetchone())[0]
        await cursor.close()
        cursor = await self._db.execute("SELECT COUNT(*) FROM archive_chunks")
        chunks = (await cursor.fetchone())[0]
        await cursor.close()
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM archive_chunks WHERE indexed_at IS NULL"
        )
        unindexed = (await cursor.fetchone())[0]
        await cursor.close()
        return {"messages": int(msgs), "chunks": int(chunks), "chunks_unindexed": int(unindexed)}


__all__ = [
    "ArchiveMessage",
    "SearchHit",
    "ChatArchiveStore",
    "build_chunks",
    "derive_message_id",
]
