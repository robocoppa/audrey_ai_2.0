"""Chat-archive capture (Audrey side).

Three responsibilities:

  ChatArchiveClient
    Performs one best-effort delivery to custom-tools' internal route.

  ChatArchiveQueue
    Commits response-time source rows to local SQLite, wakes one bounded
    lifecycle-owned delivery worker, and retries after failure/restart.
    Queue saturation is visible but cannot discard the durable row.

  StreamCollector
    Wraps an SSE async-generator and accumulates only the assistant
    content deltas while passing every frame through unchanged. Used by
    every streaming branch in routes/openai.py so capture lives in one
    place and banner / tool-call frames are filtered out consistently.

Conversation id resolution is here too, because both the streaming and
non-streaming paths need it. Order is documented in
`docs/campaign-2/phase-01-chat-archive-plan.md`.

Request-time enqueue never raises out of the chat path. Persistence,
overflow, delivery, and retry each have bounded metrics and logs.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiosqlite
import httpx

from audrey.metrics import (
    chat_archive_enqueue_seconds,
    chat_archive_queue_depth,
    chat_archive_queue_events_total,
    chat_archive_write_seconds,
    chat_archive_writes_total,
)
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)

# The model-callable search tool whose registry entry tells us which
# custom-tools server hosts the archive. Internal write/admin routes
# live on the same server but aren't in the OpenAPI tool surface.
_ARCHIVE_HOST_TOOL = "chat_history_search"
_ARCHIVE_WRITE_PATH = "/chat_history/archive"
_USER_PURGE_PATH = "/user_data/purge"
_REPAIR_STATUS_PATH = "/user_data/repair/status"
_REPAIR_RUN_PATH = "/user_data/repair/run"

# Cap stored content so a single runaway response can't blow up the
# archive row size. Long answers still get archived — just clipped at
# the boundary the search snippet would never use anyway.
_MAX_CONTENT_CHARS = 32_000


# ─── Conversation id resolution ───────────────────────────────────────

def resolve_conversation_id(
    *,
    user_id: str,
    raw_payload: dict[str, Any] | None,
    messages: list[dict[str, Any]],
) -> str:
    """Pick a conversation id, in this order:

      1. Top-level `chat_id` or `conversation_id`.
      2. Either id nested in top-level `metadata`.
      3. Either id in validated message metadata, newest first.
      4. Deterministic hash of the authenticated user and first user turn.
      5. Fresh UUID when no stable material exists.

    The fallback deliberately ignores later history. A client that resends
    the whole growing thread therefore keeps one id from turn one onward.
    """
    def explicit_id(source: dict[str, Any] | None) -> str | None:
        if not isinstance(source, dict):
            return None
        for key in ("chat_id", "conversation_id"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip()
                if len(cleaned) <= 200:
                    return cleaned
                digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:32]
                return f"client-{digest}"
        return None

    if raw_payload:
        if cid := explicit_id(raw_payload):
            return cid
        if cid := explicit_id(raw_payload.get("metadata")):
            return cid
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if cid := explicit_id(message.get("metadata")):
            return cid

    # Step 4: only stable first-turn material participates. Hashing a fixed
    # slice of the *current* history changed the id while the first six
    # messages were still accumulating.
    if user_id and messages:
        first_user = next(
            (
                message for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            None,
        )
        if first_user is not None:
            content = first_user.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True, separators=(",", ":"))
            h = hashlib.sha256()
            h.update(user_id.encode("utf-8"))
            h.update(b"|first-user|")
            h.update(content.encode("utf-8"))
            return f"derived-{h.hexdigest()[:24]}"

    # Step 5.
    return f"fresh-{uuid.uuid4().hex}"


# ─── StreamCollector ──────────────────────────────────────────────────

class StreamCollector:
    """Accumulates assistant `content` deltas from an SSE stream.

    Wrap any streaming generator that emits OpenAI-shaped SSE frames in
    `wrap()`. The wrapper passes every frame through unchanged and
    appends only `choices[0].delta.content` strings to its buffer.
    Banner-text frames are also `delta.content`-shaped, so the caller is
    responsible for *not* wrapping inside the banner-only sections —
    `_stream_deep_with_banners` calls `wrap()` only around the synth
    deltas to keep banners out of the archive.

    Usage:
        collector = StreamCollector()
        async for frame in collector.wrap(generator):
            yield frame
        # collector.text and collector.partial are valid here.
    """

    __slots__ = ("_finalized", "partial", "text")

    def __init__(self) -> None:
        self.text: str = ""
        self.partial: bool = False
        self._finalized: bool = False

    async def wrap(self, source):
        """Pass frames through; accumulate deltas. Marks `partial=True`
        when CancelledError is observed (client disconnect)."""
        try:
            async for frame in source:
                self._absorb(frame)
                yield frame
        except asyncio.CancelledError:
            self.partial = True
            self._finalized = True
            raise
        finally:
            self._finalized = True

    def feed_text(self, text: str) -> None:
        """For non-streaming or already-collected text. No frame parsing."""
        if not text:
            return
        if len(self.text) + len(text) > _MAX_CONTENT_CHARS:
            self.text = (self.text + text)[:_MAX_CONTENT_CHARS]
        else:
            self.text += text

    def mark_partial(self) -> None:
        self.partial = True

    def _absorb(self, frame: Any) -> None:
        # Frames are SSE strings: "data: {json}\n\n". We only parse to
        # pull `delta.content` out; anything that doesn't fit the shape
        # is skipped silently.
        if not isinstance(frame, str):
            return
        if not frame.startswith("data: "):
            return
        body = frame[6:].strip()
        if not body or body == "[DONE]":
            return
        try:
            obj = json.loads(body)
        except (ValueError, TypeError):
            return
        choices = obj.get("choices") or []
        if not choices:
            return
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.feed_text(content)


# ─── Durable response-time queue ──────────────────────────────────────

class ArchiveDelivery(StrEnum):
    """The queue either finalizes or retains one durable source row."""

    DELIVERED = "delivered"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class ArchiveJob:
    """Serializable archive request with a stable retry identity."""

    archive_id: str
    created_at: str
    user_id: str
    conversation_id: str
    user_content: str
    assistant_content: str
    partial: bool = False
    virtual_model: str = ""
    concrete_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
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
    ) -> ArchiveJob:
        return cls(
            archive_id=archive_id or uuid.uuid4().hex,
            created_at=created_at or _now_iso(),
            user_id=user_id,
            conversation_id=conversation_id,
            user_content=user_content[:_MAX_CONTENT_CHARS],
            assistant_content=assistant_content[:_MAX_CONTENT_CHARS],
            partial=partial,
            virtual_model=virtual_model,
            concrete_model=concrete_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "archive_id": self.archive_id,
            "created_at": self.created_at,
            "user": self.user_id,
            "conversation_id": self.conversation_id,
            "user_content": self.user_content,
            "assistant_content": self.assistant_content,
            "partial": self.partial,
            "virtual_model": self.virtual_model,
            "concrete_model": self.concrete_model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _retry_at(seconds: float) -> str:
    return (
        dt.datetime.now(dt.UTC) + dt.timedelta(seconds=seconds)
    ).isoformat(timespec="microseconds")


class ChatArchiveClient:
    """Best-effort transport to custom-tools' internal archive route."""

    __slots__ = ("_http", "_service_headers", "_timeout_s")

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        timeout_s: float = 5.0,
        service_token: str = "",
    ) -> None:
        self._http = http
        self._timeout_s = timeout_s
        self._service_headers = (
            {"X-Audrey-Service-Token": service_token} if service_token else {}
        )

    def host_url(self, registry: ToolRegistry | None) -> str | None:
        if registry is None:
            return None
        spec = registry.get(_ARCHIVE_HOST_TOOL)
        if spec is None:
            return None
        return spec.server_url

    async def deliver_job(
        self,
        *,
        registry: ToolRegistry | None,
        job: ArchiveJob,
    ) -> tuple[ArchiveDelivery, str]:
        """Attempt one delivery and return whether its source row may clear."""
        host = self.host_url(registry)
        if host is None:
            chat_archive_writes_total.labels(result="deferred").inc()
            return ArchiveDelivery.RETRY, "chat_history_search is not registered"

        url = f"{host}{_ARCHIVE_WRITE_PATH}"
        t0 = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "json": job.payload(),
                "timeout": self._timeout_s,
            }
            if self._service_headers:
                kwargs["headers"] = self._service_headers
            response = await self._http.post(url, **kwargs)
            chat_archive_write_seconds.observe(time.perf_counter() - t0)
            if response.status_code >= 400:
                chat_archive_writes_total.labels(result="fail").inc()
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                log.warning("chat_archive: write failed %s", error)
                return ArchiveDelivery.RETRY, error
            chat_archive_writes_total.labels(
                result="partial" if job.partial else "ok",
            ).inc()
            return ArchiveDelivery.DELIVERED, ""
        except (httpx.HTTPError, TimeoutError) as exc:
            chat_archive_write_seconds.observe(time.perf_counter() - t0)
            chat_archive_writes_total.labels(result="fail").inc()
            error = f"{type(exc).__name__}: {exc}"[:500]
            log.warning("chat_archive: write transport error: %s", error)
            return ArchiveDelivery.RETRY, error

    async def request_user_purge(
        self,
        *,
        registry: ToolRegistry | None,
        user: str,
        purge_id: str,
        cutoff_at: str,
    ) -> dict[str, Any]:
        """Create or poll one idempotent sidecar purge receipt."""
        host = self.host_url(registry)
        if host is None:
            raise RuntimeError("chat_history_search is not registered")
        kwargs: dict[str, Any] = {
            "json": {
                "user": user,
                "purge_id": purge_id,
                "cutoff_at": cutoff_at,
            },
            "timeout": self._timeout_s,
        }
        if self._service_headers:
            kwargs["headers"] = self._service_headers
        response = await self._http.post(f"{host}{_USER_PURGE_PATH}", **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(
                f"sidecar purge returned HTTP {response.status_code}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError("sidecar purge returned an invalid response")
        return value

    async def _request_repair_control(
        self,
        *,
        registry: ToolRegistry | None,
        path: str,
        operation: str,
    ) -> dict[str, Any]:
        host = self.host_url(registry)
        if host is None:
            raise RuntimeError("chat_history_search is not registered")
        kwargs: dict[str, Any] = {
            "json": {},
            "timeout": self._timeout_s,
        }
        if self._service_headers:
            kwargs["headers"] = self._service_headers
        response = await self._http.post(f"{host}{path}", **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(
                f"sidecar {operation} returned HTTP {response.status_code}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError(f"sidecar {operation} returned an invalid response")
        return value

    async def repair_status(
        self,
        *,
        registry: ToolRegistry | None,
    ) -> dict[str, Any]:
        """Return global sidecar repair counts through service authentication."""
        return await self._request_repair_control(
            registry=registry,
            path=_REPAIR_STATUS_PATH,
            operation="repair status",
        )

    async def repair(
        self,
        *,
        registry: ToolRegistry | None,
    ) -> dict[str, Any]:
        """Run one bounded sidecar repair pass, including exhausted work."""
        return await self._request_repair_control(
            registry=registry,
            path=_REPAIR_RUN_PATH,
            operation="repair",
        )

    async def archive_turn(
        self,
        *,
        registry: ToolRegistry | None,
        user_id: str,
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
    ) -> None:
        """Compatibility direct write. The application uses the queue below."""
        if not user_id or (not user_content and not assistant_content):
            chat_archive_writes_total.labels(result="skipped").inc()
            return
        job = ArchiveJob.create(
            user_id=user_id,
            conversation_id=conversation_id,
            user_content=user_content,
            assistant_content=assistant_content,
            partial=partial,
            virtual_model=virtual_model,
            concrete_model=concrete_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            archive_id=archive_id,
            created_at=created_at,
        )
        await self.deliver_job(registry=registry, job=job)


_OUTBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive_write_outbox (
    archive_id       TEXT PRIMARY KEY,
    payload_json     TEXT NOT NULL,
    user_id          TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_attempt_at  TEXT,
    last_error       TEXT,
    next_attempt_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_archive_write_due
    ON archive_write_outbox(next_attempt_at, created_at);
CREATE TABLE IF NOT EXISTS archive_user_purges (
    user       TEXT PRIMARY KEY,
    cutoff_at  TEXT NOT NULL,
    purge_id   TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class ChatArchiveQueue:
    """Durable outbox with a bounded in-process wake channel.

    Every response first commits a compact source row to local SQLite. The
    bounded asyncio queue carries only wake signals, so a full channel never
    loses the turn: overflow is logged/counted and the worker discovers the
    durable row on its next scan. Remote HTTP, embedding, and Qdrant work all
    happen in the lifecycle-owned worker rather than the response path.
    """

    def __init__(
        self,
        *,
        client: ChatArchiveClient,
        registry: ToolRegistry,
        sqlite_path: Path,
        maxsize: int = 128,
        retry_interval_s: float = 30.0,
    ) -> None:
        if maxsize < 1:
            raise ValueError("chat archive queue maxsize must be positive")
        if retry_interval_s <= 0:
            raise ValueError("chat archive retry interval must be positive")
        self._client = client
        self._registry = registry
        self._sqlite_path = sqlite_path
        self._retry_interval_s = retry_interval_s
        self._signals: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self._db: aiosqlite.Connection | None = None
        self._db_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False

    def host_url(self, registry: ToolRegistry | None) -> str | None:
        """Preserve the admin routes' host lookup contract."""
        return self._client.host_url(registry)

    async def start(self, *, run_worker: bool = True) -> None:
        if self._db is not None:
            raise RuntimeError("chat archive queue already started")
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._sqlite_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA secure_delete=ON")
        await self._db.executescript(_OUTBOX_SCHEMA)
        await self._migrate_schema()
        # A restart is an operator-visible retry boundary: the upstream may
        # have been repaired while Audrey was down, so do not preserve an old
        # in-process backoff across the new lifecycle.
        await self._db.execute(
            "UPDATE archive_write_outbox SET next_attempt_at = ?",
            (_now_iso(),),
        )
        await self._db.commit()
        pending = await self.pending_count()
        chat_archive_queue_depth.set(pending)
        self._accepting = run_worker
        if run_worker:
            self._worker = asyncio.create_task(
                self._run(),
                name="chat-archive-outbox",
            )
            self._signal("startup")
        log.info(
            "chat_archive: queue ready sqlite=%s maxsize=%d pending=%d worker=%s",
            self._sqlite_path,
            self._signals.maxsize,
            pending,
            "on" if run_worker else "off",
        )

    async def _migrate_schema(self) -> None:
        """Add exact-user projection to outboxes created before account purge."""
        if self._db is None:
            raise RuntimeError("chat archive queue is not started")
        cursor = await self._db.execute("PRAGMA table_info(archive_write_outbox)")
        columns = {str(row[1]) for row in await cursor.fetchall()}
        await cursor.close()
        if "user_id" not in columns:
            await self._db.execute(
                "ALTER TABLE archive_write_outbox "
                "ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
        cursor = await self._db.execute(
            "SELECT archive_id, payload_json FROM archive_write_outbox "
            "WHERE user_id = ''"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        for archive_id, payload_json in rows:
            try:
                user_id = str(json.loads(str(payload_json)).get("user_id") or "")
            except (TypeError, ValueError, json.JSONDecodeError):
                user_id = ""
            if user_id:
                await self._db.execute(
                    "UPDATE archive_write_outbox SET user_id = ? WHERE archive_id = ?",
                    (user_id, archive_id),
                )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_write_user "
            "ON archive_write_outbox(user_id, created_at)"
        )
        await self._db.commit()

    async def stop(self) -> None:
        self._accepting = False
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        if self._db is not None:
            try:
                await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                await self._db.close()
            self._db = None

    async def archive_turn(
        self,
        *,
        registry: ToolRegistry | None,
        user_id: str,
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
    ) -> None:
        """Commit one retryable source row, then wake the delivery worker."""
        del registry  # the queue owns the live, in-place ToolRegistry
        if not user_id or (not user_content and not assistant_content):
            chat_archive_writes_total.labels(result="skipped").inc()
            return
        if not self._accepting or self._db is None:
            chat_archive_queue_events_total.labels(result="enqueue_fail").inc()
            log.error("chat_archive: enqueue rejected while queue is stopped")
            return

        job = ArchiveJob.create(
            user_id=user_id,
            conversation_id=conversation_id,
            user_content=user_content,
            assistant_content=assistant_content,
            partial=partial,
            virtual_model=virtual_model,
            concrete_model=concrete_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            archive_id=archive_id,
            created_at=created_at,
        )
        encoded = json.dumps(asdict(job), sort_keys=True, separators=(",", ":"))
        started = time.perf_counter()
        try:
            async with self._db_lock:
                cursor = await self._db.execute(
                    "SELECT cutoff_at FROM archive_user_purges WHERE user = ?",
                    (job.user_id,),
                )
                purge = await cursor.fetchone()
                await cursor.close()
                if purge is not None and job.created_at <= str(purge[0]):
                    chat_archive_queue_events_total.labels(result="purged").inc()
                    return
                cursor = await self._db.execute(
                    """
                    INSERT OR IGNORE INTO archive_write_outbox
                    (archive_id, payload_json, user_id, created_at, next_attempt_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job.archive_id,
                        encoded,
                        job.user_id,
                        job.created_at,
                        job.created_at,
                    ),
                )
                inserted = cursor.rowcount > 0
                await cursor.close()
                await self._db.commit()
        except Exception as exc:  # noqa: BLE001 — persistence must fail soft
            try:
                await self._db.rollback()
            except Exception:  # noqa: BLE001, S110 — preserve original failure
                pass
            chat_archive_enqueue_seconds.observe(time.perf_counter() - started)
            chat_archive_queue_events_total.labels(result="enqueue_fail").inc()
            log.error(
                "chat_archive: durable enqueue failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return
        chat_archive_enqueue_seconds.observe(time.perf_counter() - started)
        if inserted:
            chat_archive_queue_depth.inc()

        if self._signal(job.archive_id):
            chat_archive_queue_events_total.labels(result="enqueued").inc()
        else:
            chat_archive_queue_events_total.labels(result="overflow").inc()
            log.warning(
                "chat_archive: wake queue full; durable source retained archive_id=%s",
                job.archive_id,
            )

    async def purge_user_before(
        self,
        *,
        user_id: str,
        cutoff_at: str,
        purge_id: str,
    ) -> int:
        """Durably block and remove pre-cutoff local archive deliveries."""
        if not user_id or not cutoff_at or not purge_id:
            raise ValueError("local archive purge requires user, cutoff, and purge id")
        if self._db is None:
            raise RuntimeError("chat archive queue is not started")
        async with self._db_lock:
            await self._db.execute(
                """
                INSERT INTO archive_user_purges (user, cutoff_at, purge_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user) DO UPDATE SET
                    cutoff_at = CASE
                        WHEN excluded.cutoff_at > archive_user_purges.cutoff_at
                        THEN excluded.cutoff_at ELSE archive_user_purges.cutoff_at
                    END,
                    purge_id = CASE
                        WHEN excluded.cutoff_at >= archive_user_purges.cutoff_at
                        THEN excluded.purge_id ELSE archive_user_purges.purge_id
                    END,
                    updated_at = excluded.updated_at
                """,
                (user_id, cutoff_at, purge_id, _now_iso()),
            )
            cursor = await self._db.execute(
                """
                DELETE FROM archive_write_outbox
                WHERE user_id = ? AND created_at <= ?
                """,
                (user_id, cutoff_at),
            )
            deleted = max(0, cursor.rowcount)
            await cursor.close()
            await self._db.commit()
        if deleted:
            chat_archive_queue_depth.dec(deleted)
        return deleted

    async def pending_count(self) -> int:
        if self._db is None:
            return 0
        async with self._db_lock:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM archive_write_outbox"
            )
            count = int((await cursor.fetchone())[0])
            await cursor.close()
        return count

    async def retry_now(self, *, limit: int = 50) -> int:
        """Make one bounded delivery batch due now and wake its sole worker."""
        if self._db is None or not self._accepting:
            return 0
        async with self._db_lock:
            cursor = await self._db.execute(
                "SELECT archive_id FROM archive_write_outbox "
                "ORDER BY next_attempt_at, created_at LIMIT ?",
                (max(1, limit),),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            if rows:
                await self._db.executemany(
                    "UPDATE archive_write_outbox SET next_attempt_at = ? "
                    "WHERE archive_id = ?",
                    [(_now_iso(), str(row[0])) for row in rows],
                )
                await self._db.commit()
        if rows:
            self._signal("admin-repair")
        return len(rows)

    async def repair_stats(self) -> dict[str, int]:
        """Global delivery repair counts without payloads or raw errors."""
        empty = {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 0,
        }
        if self._db is None:
            return empty
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(attempts), 0),
                       COALESCE(SUM(CASE WHEN length(last_error) > 0
                                         THEN 1 ELSE 0 END), 0)
                FROM archive_write_outbox
                """
            )
            pending, attempts, with_error = await cursor.fetchone()
            await cursor.close()
        return {
            **empty,
            "pending": int(pending),
            "attempts": int(attempts),
            "with_error": int(with_error),
        }

    async def user_stats(self, user_id: str) -> dict[str, int]:
        """Current-user delivery backlog without returning payloads or errors."""
        empty = {
            "pending": 0,
            "attempts": 0,
            "with_error": 0,
            "exhausted": 0,
            "completed": 0,
        }
        if self._db is None:
            return empty
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT attempts, last_error
                FROM archive_write_outbox
                WHERE user_id = ?
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()

        result = dict(empty)
        for attempts, last_error in rows:
            result["pending"] += 1
            result["attempts"] += int(attempts)
            if last_error:
                result["with_error"] += 1
        return result

    async def stats(self) -> dict[str, Any]:
        """Small operational/test view of the durable source backlog."""
        if self._db is None:
            return {
                "pending": 0,
                "attempts": 0,
                "oldest_created_at": "",
                "last_attempt_at": "",
                "last_error": "",
            }
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(attempts), 0), MIN(created_at)
                FROM archive_write_outbox
                """
            )
            pending, attempts, oldest_created_at = await cursor.fetchone()
            await cursor.close()
            cursor = await self._db.execute(
                """
                SELECT last_attempt_at, last_error FROM archive_write_outbox
                WHERE last_attempt_at IS NOT NULL
                ORDER BY last_attempt_at DESC LIMIT 1
                """
            )
            row = await cursor.fetchone()
            await cursor.close()
        return {
            "pending": int(pending),
            "attempts": int(attempts),
            "oldest_created_at": str(oldest_created_at or ""),
            "last_attempt_at": str(row[0]) if row else "",
            "last_error": str(row[1] or "") if row else "",
        }

    def _signal(self, archive_id: str) -> bool:
        try:
            self._signals.put_nowait(archive_id)
        except asyncio.QueueFull:
            return False
        return True

    async def _next_due(self) -> ArchiveJob | None:
        if self._db is None:
            return None
        now = _now_iso()
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT archive_id, payload_json
                FROM archive_write_outbox
                WHERE next_attempt_at <= ?
                ORDER BY next_attempt_at, created_at
                LIMIT 1
                """,
                (now,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            archive_id, payload_json = str(row[0]), str(row[1])
            await self._db.execute(
                """
                UPDATE archive_write_outbox
                SET attempts = attempts + 1, last_attempt_at = ?
                WHERE archive_id = ?
                """,
                (now, archive_id),
            )
            await self._db.commit()
        try:
            return ArchiveJob(**json.loads(payload_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            await self._defer(archive_id, f"invalid durable payload: {exc}")
            return None

    async def _finalize(self, archive_id: str) -> None:
        if self._db is None:
            return
        async with self._db_lock:
            cursor = await self._db.execute(
                "DELETE FROM archive_write_outbox WHERE archive_id = ?",
                (archive_id,),
            )
            deleted = max(0, cursor.rowcount)
            await cursor.close()
            await self._db.commit()
        if deleted:
            chat_archive_queue_depth.dec(deleted)

    async def _defer(self, archive_id: str, error: str) -> None:
        if self._db is None:
            return
        async with self._db_lock:
            await self._db.execute(
                """
                UPDATE archive_write_outbox
                SET last_error = ?, next_attempt_at = ?
                WHERE archive_id = ?
                """,
                (error[:500], _retry_at(self._retry_interval_s), archive_id),
            )
            await self._db.commit()

    async def _has_due(self) -> bool:
        if self._db is None:
            return False
        async with self._db_lock:
            cursor = await self._db.execute(
                """
                SELECT 1 FROM archive_write_outbox
                WHERE next_attempt_at <= ? LIMIT 1
                """,
                (_now_iso(),),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row is not None

    async def _deliver_one(self) -> None:
        job = await self._next_due()
        if job is None:
            return
        try:
            outcome, error = await self._client.deliver_job(
                registry=self._registry,
                job=job,
            )
        except Exception as exc:  # noqa: BLE001 — worker must survive one job
            outcome = ArchiveDelivery.RETRY
            error = f"{type(exc).__name__}: {exc}"[:500]
        if outcome is ArchiveDelivery.DELIVERED:
            await self._finalize(job.archive_id)
            chat_archive_queue_events_total.labels(result="delivered").inc()
        else:
            await self._defer(job.archive_id, error)
            chat_archive_queue_events_total.labels(result="retry").inc()
            log.warning(
                "chat_archive: durable delivery deferred archive_id=%s error=%s",
                job.archive_id,
                error,
            )

    async def _run(self) -> None:
        while True:
            received = False
            try:
                try:
                    await asyncio.wait_for(
                        self._signals.get(),
                        timeout=self._retry_interval_s,
                    )
                    received = True
                except TimeoutError:
                    pass
                await self._deliver_one()
                # One signal is enough to begin draining any startup/overflow
                # backlog. Extra signals remain bounded and harmless.
                if await self._has_due():
                    self._signal("backlog")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep the owner alive
                log.error(
                    "chat_archive: queue worker iteration failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(min(self._retry_interval_s, 5.0))
            finally:
                if received:
                    self._signals.task_done()


__all__ = [
    "ArchiveDelivery",
    "ArchiveJob",
    "ChatArchiveClient",
    "ChatArchiveQueue",
    "StreamCollector",
    "resolve_conversation_id",
]
