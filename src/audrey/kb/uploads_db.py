"""SQLite index over per-user upload metadata.

Qdrant remains authoritative for *file content* (text chunks, image
embeddings). This module is a fast index over the per-file metadata so
`GET /v1/files` and the quota gate don't have to scroll a whole user
collection.

Source-of-truth contract:
  - Qdrant has the file's content → sqlite has a row for it.
  - Qdrant has nothing for a file_id → sqlite must not either, *unless
    the row never claimed Qdrant content in the first place*. A video
    waiting on the media worker is the case that carve-out exists for:
    it holds bytes on disk and no points anywhere, on purpose.

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

Each `UploadsDB` instance owns one connection guarded by a lock because the
async wrapper runs calls in worker threads. Atomic quota decisions also use
`BEGIN IMMEDIATE`: the lock protects one connection, while SQLite's write
transaction serializes decisions made through separate app/process connections.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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
  uploaded_at TEXT NOT NULL,
  -- 'ready'         — ingested, searchable now.
  -- 'fetch_pending' — a URL was accepted and no bytes exist yet (Phase 41).
  --                   Waiting for a media fetcher.
  -- 'fetching'      — leased by a media fetcher, download in progress.
  -- 'pending'       — bytes stored, nothing extracted yet, waiting for a
  --                   worker. Video lands here on upload, and a fetched video
  --                   arrives here from 'fetching'.
  -- 'processing'    — leased by a media worker (Phase 33). `lease_id` says
  --                   which lease, `leased_at` says when it expires from.
  -- 'failed'        — gave up. `failure_reason` says why, and it is shown to
  --                   the user: a row that sits at 'pending' forever with no
  --                   explanation is the failure mode this state exists to
  --                   avoid.
  --
  -- The two queues have the same shape and it is deliberate: a *_pending
  -- status means "this stage's work is waiting", and the claim transitions it
  -- to the leased status under one lock hold. That transition IS the mutual
  -- exclusion, which is why neither claim needs a `lease_id = ''` predicate to
  -- avoid handing one row to two holders.
  status      TEXT NOT NULL DEFAULT 'ready',
  -- Empty unless status is 'processing' or 'fetching' — the two stages share
  -- these columns because a row is only ever in one of them. Presented back on
  -- completion and checked, so a holder that stalled past its lease and woke up
  -- after the job was re-leased cannot overwrite the newer run's result.
  lease_id       TEXT NOT NULL DEFAULT '',
  leased_at      TEXT NOT NULL DEFAULT '',
  -- Counts leases, not failures — a worker that dies without reporting never
  -- gets to increment anything, so the lease is the only reliable tally.
  attempts       INTEGER NOT NULL DEFAULT 0,
  failure_reason TEXT NOT NULL DEFAULT '',
  -- Seconds of audio, reported by the worker (Phase 35). 0 for everything
  -- that is not a video, and for a video with no audio stream — those are
  -- the same number for different reasons, so read it with `kind`.
  duration_s     REAL NOT NULL DEFAULT 0,
  -- One-paragraph summary of a processed video (Phase 37), shown in the file
  -- list. Empty for everything else, and empty for a video whose summary call
  -- failed — that is a missing field, never a failed row, because by the time
  -- it is written the transcript and descriptions are already ingested and
  -- already useful.
  summary        TEXT NOT NULL DEFAULT '',
  -- When `ingest_result` finished with this row (Phase 38). Empty until then.
  -- Source reclamation measures its retention window from HERE and not from
  -- `uploaded_at`, because they are only the same number when the queue is
  -- moving: a video that sat pending for two days while the worker was down
  -- would, measured from upload, become eligible for deletion the instant it
  -- finished — a retention window of zero, exactly for the file whose
  -- processing was most likely to want a second look.
  completed_at   TEXT NOT NULL DEFAULT '',
  -- When the uploaded bytes were unlinked (Phase 38). Empty while the source
  -- is still on disk. `bytes` is deliberately left alone: it is the honest
  -- record of what was uploaded and what the file list shows, so the quota
  -- reads this column instead of zeroing that one.
  source_freed_at TEXT NOT NULL DEFAULT '',
  -- Where a fetched video came from (Phase 41). Empty for an ordinary upload,
  -- and that emptiness is the only thing distinguishing the two afterwards —
  -- everything downstream of 'pending' treats them identically on purpose.
  --
  -- Kept after the fetch succeeds rather than cleared, for three reasons: the
  -- page links back to it, a duplicate paste is refused against it, and it is
  -- the answer to "can these bytes be re-obtained?" — for a URL-sourced video
  -- the source is re-downloadable, which is exactly the argument phase 38 found
  -- did not hold for an upload.
  source_url  TEXT NOT NULL DEFAULT '',
  -- A caption track the fetcher got from the site (Phase 41 step 4), as JSON:
  -- `{"source": ..., "segments": [{t_start, t_end, text}, ...]}`. Handed to the
  -- media worker on its claim, which is the whole reason it is stored rather
  -- than posted straight through — the two stages are separated by a queue and
  -- a container, and there is no moment when both are holding the row.
  --
  -- **Cleared by `complete_job`.** By then the transcript is chunked, indexed
  -- and written to a `.transcript.txt` sidecar, so keeping it would be a third
  -- copy of the same text living in a metadata table — a two-hour video's
  -- captions are ~120 KB, and this column is on the row `SELECT *` reads on
  -- every claim.
  --
  -- Deliberately NOT cleared by `fail_job` or the lease sweep: a retry should
  -- reuse the captions it already has rather than fetching them again or
  -- falling back to whisper for a video that plainly has them.
  fetched_transcript TEXT NOT NULL DEFAULT '',
  -- What produced the transcript that was actually ingested: 'subtitles',
  -- 'auto_captions', or 'whisper'. Empty for anything that is not a processed
  -- video, and for a video with no speech at all.
  --
  -- Written by `complete_job` from what the worker reports, NOT from what the
  -- fetch offered. Those differ whenever a claim carries captions and the
  -- worker transcribes anyway, and the point of the column is to be right
  -- about which text is on the row — "the transcript is wrong" has a different
  -- answer for each of the three.
  transcript_source  TEXT NOT NULL DEFAULT '',
  -- How far a download has got, in bytes, and how far it has to go (Phase 41).
  -- Both 0 outside the 'fetching' state; `fetch_total_bytes` stays 0 for a
  -- site that will not say how big the file is, which the page renders as a
  -- running byte count with no percentage rather than as a fake one.
  --
  -- `bytes` is deliberately NOT reused for the first of these. That column is
  -- what the user is billed for and what the file list shows, and a partial
  -- download is neither — writing progress there would put a growing number
  -- into the quota for a file that may never exist.
  fetch_downloaded_bytes INTEGER NOT NULL DEFAULT 0,
  fetch_total_bytes      INTEGER NOT NULL DEFAULT 0,
  -- Capacity held while a URL is waiting for or actively being downloaded.
  -- The real size replaces this ceiling atomically in `complete_fetch`.
  quota_reserved_bytes   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads(user);
CREATE TABLE IF NOT EXISTS file_deletions (
  file_id          TEXT NOT NULL,
  user             TEXT NOT NULL,
  requested_at     TEXT NOT NULL,
  attempts         INTEGER NOT NULL DEFAULT 0,
  last_attempt_at  TEXT NOT NULL DEFAULT '',
  last_error       TEXT NOT NULL DEFAULT '',
  completed_at     TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (file_id, user)
);
CREATE INDEX IF NOT EXISTS idx_file_deletions_pending
  ON file_deletions(completed_at, requested_at);
-- No index on `status` here. `_SCHEMA` runs before `_migrate`, and on the
-- deployed database the column does not exist yet — indexing it at this point
-- fails at boot with "no such column". It is created in `_migrate` instead,
-- after the columns are in place.

-- Chunked-upload sessions (Phase 32). A row lives only while an upload is
-- in flight: created by `open_session`, dropped when `/complete` succeeds or
-- the TTL sweep collects it. Nothing here is authoritative about stored
-- files — that stays `uploads` above, written only after assembly lands.
CREATE TABLE IF NOT EXISTS upload_sessions (
  upload_id    TEXT PRIMARY KEY,
  user         TEXT NOT NULL,
  filename     TEXT NOT NULL,
  total_bytes  INTEGER NOT NULL,
  part_size    INTEGER NOT NULL,
  parts_total  INTEGER NOT NULL,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON upload_sessions(user);

-- One row per part that has actually landed on disk. The PRIMARY KEY makes
-- a retried part idempotent rather than double-counted, which matters
-- because a flaky connection retrying part 7 is the normal case, not an
-- edge case.
CREATE TABLE IF NOT EXISTS upload_parts (
  upload_id  TEXT NOT NULL,
  part_no    INTEGER NOT NULL,
  bytes      INTEGER NOT NULL,
  PRIMARY KEY (upload_id, part_no)
);

-- Short-lived reservations for single-shot uploads. Chunk sessions and URL
-- fetches are durable lifecycle rows already, so their reservation ids are
-- `upload_sessions.upload_id` and `uploads.file_id` respectively.
CREATE TABLE IF NOT EXISTS storage_reservations (
  reservation_id TEXT PRIMARY KEY,
  user           TEXT NOT NULL,
  kind           TEXT NOT NULL,
  bytes          INTEGER NOT NULL CHECK (bytes >= 0),
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_storage_reservations_user ON storage_reservations(user);
CREATE INDEX IF NOT EXISTS idx_storage_reservations_updated ON storage_reservations(updated_at);

CREATE TABLE IF NOT EXISTS user_data_purges (
  purge_id                    TEXT PRIMARY KEY,
  user                        TEXT NOT NULL,
  cutoff_at                   TEXT NOT NULL,
  requested_at                TEXT NOT NULL,
  local_delivery_completed_at TEXT NOT NULL DEFAULT '',
  local_delivery_attempts     INTEGER NOT NULL DEFAULT 0,
  local_delivery_last_error   TEXT NOT NULL DEFAULT '',
  sidecar_acknowledged_at     TEXT NOT NULL DEFAULT '',
  sidecar_completed_at        TEXT NOT NULL DEFAULT '',
  sidecar_status              TEXT NOT NULL DEFAULT 'pending',
  sidecar_attempts            INTEGER NOT NULL DEFAULT 0,
  sidecar_last_error          TEXT NOT NULL DEFAULT '',
  completed_at                TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_user_data_purges_owner
  ON user_data_purges(user, requested_at);
CREATE INDEX IF NOT EXISTS idx_user_data_purges_pending
  ON user_data_purges(completed_at, requested_at);

CREATE TABLE IF NOT EXISTS user_data_purge_files (
  purge_id TEXT NOT NULL,
  file_id  TEXT NOT NULL,
  user     TEXT NOT NULL,
  PRIMARY KEY (purge_id, file_id, user)
);
CREATE TABLE IF NOT EXISTS user_data_purge_paths (
  purge_id       TEXT NOT NULL,
  user           TEXT NOT NULL,
  kind           TEXT NOT NULL,
  object_id      TEXT NOT NULL,
  attempts       INTEGER NOT NULL DEFAULT 0,
  last_attempt_at TEXT NOT NULL DEFAULT '',
  last_error     TEXT NOT NULL DEFAULT '',
  completed_at   TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (purge_id, kind, object_id)
);
CREATE INDEX IF NOT EXISTS idx_user_data_purge_paths_pending
  ON user_data_purge_paths(completed_at, purge_id);
"""


#: Columns added to `uploads` after the table first shipped, in the order they
#: were added. Fresh databases get them from `_SCHEMA`; deployed ones get them
#: from `_migrate`. Both lists have to agree — a column here and not in the
#: schema string means a fresh install silently lacks it.
_UPLOADS_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "TEXT NOT NULL DEFAULT 'ready'"),
    ("lease_id", "TEXT NOT NULL DEFAULT ''"),
    ("leased_at", "TEXT NOT NULL DEFAULT ''"),
    ("attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("failure_reason", "TEXT NOT NULL DEFAULT ''"),
    ("duration_s", "REAL NOT NULL DEFAULT 0"),
    ("summary", "TEXT NOT NULL DEFAULT ''"),
    ("completed_at", "TEXT NOT NULL DEFAULT ''"),
    ("source_freed_at", "TEXT NOT NULL DEFAULT ''"),
    ("source_url", "TEXT NOT NULL DEFAULT ''"),
    ("fetched_transcript", "TEXT NOT NULL DEFAULT ''"),
    ("transcript_source", "TEXT NOT NULL DEFAULT ''"),
    ("fetch_downloaded_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("fetch_total_bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("quota_reserved_bytes", "INTEGER NOT NULL DEFAULT 0"),
)


@dataclass(frozen=True)
class QuotaUsage:
    """One user's storage budget, split by lifecycle state."""
    stored_bytes: int
    chunk_declared_bytes: int
    chunk_part_bytes: int
    chunk_part_overage_bytes: int
    url_fetch_bytes: int
    single_shot_bytes: int

    @property
    def total_bytes(self) -> int:
        # Parts are already covered by a session's declared total. Only bytes
        # beyond that declaration are additive; exposing both values keeps an
        # on-disk/declared mismatch visible without billing the normal overlap
        # twice.
        return (
            self.stored_bytes
            + self.chunk_declared_bytes
            + self.chunk_part_overage_bytes
            + self.url_fetch_bytes
            + self.single_shot_bytes
        )


@dataclass(frozen=True)
class QuotaDecision:
    accepted: bool
    usage: QuotaUsage
    requested_bytes: int
    max_user_bytes: int


@dataclass(frozen=True)
class UrlReservationRecovery:
    restored_pending: int
    released_stranded: int


#: What a lease sweep does per stage: which status is *held* under a lease, the
#: status a swept row returns to, and who is named when it gives up.
#:
#: Two stages hold leases — the media fetcher on 'fetching' and the media
#: worker on 'processing' — and they are the same problem with different nouns.
#: One parameterised query rather than two that drift; the phase-41 plan asked
#: for exactly this check ("prefer extending to duplicating if the shapes turn
#: out to be the same"), and they do.
_LEASE_STAGES: dict[str, tuple[str, str]] = {
    "processing": ("pending", "media worker"),
    "fetching": ("fetch_pending", "media fetcher"),
}


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
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        # No journal_mode=WAL: we hold exactly one connection guarded by
        # `self._lock`, so the multi-writer story WAL exists for doesn't
        # apply. Default rollback journal avoids creating `-wal`/`-shm`
        # sidecar files. `synchronous=NORMAL` is still worth keeping —
        # rollback journal at NORMAL still gives us crash safety
        # without paying for fsync on every commit.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Additive column migrations for databases created before a column existed.

        The deployed sqlite predates these columns, and `CREATE TABLE IF NOT
        EXISTS` is a no-op against an existing table — so a fresh schema string
        alone would leave the live file without them and every read would
        raise.

        Every default here has to make an existing row correct without being
        touched: rows stored before `status` existed were ingested
        synchronously, so `ready` is true of them; nothing was ever leased, so
        an empty lease and zero attempts are true too.
        """
        with self._lock:
            cols = {
                r["name"]
                for r in self._conn.execute("PRAGMA table_info(uploads)").fetchall()
            }
            for name, ddl in _UPLOADS_ADDED_COLUMNS:
                if name in cols:
                    continue
                self._conn.execute(f"ALTER TABLE uploads ADD COLUMN {name} {ddl}")
                log.info("uploads_db: migrated uploads table — added `%s`", name)
            # Only now that the column certainly exists. The claim query filters
            # on status and orders by uploaded_at, and a worker polls it.
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status)",
            )
            purge_cols = {
                r["name"]
                for r in self._conn.execute(
                    "PRAGMA table_info(user_data_purges)"
                ).fetchall()
            }
            if "sidecar_acknowledged_at" not in purge_cols:
                self._conn.execute(
                    "ALTER TABLE user_data_purges ADD COLUMN "
                    "sidecar_acknowledged_at TEXT NOT NULL DEFAULT ''"
                )
                log.info(
                    "uploads_db: migrated user_data_purges table — "
                    "added `sidecar_acknowledged_at`"
                )
            self._conn.execute(
                "UPDATE user_data_purges SET sidecar_acknowledged_at = "
                "sidecar_completed_at WHERE length(sidecar_acknowledged_at) = 0 "
                "AND length(sidecar_completed_at) > 0"
            )

    @contextmanager
    def _transaction_locked(self) -> Iterator[None]:
        """One cross-connection SQLite write transaction; caller holds `_lock`."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

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
        status: str = "ready",
    ) -> None:
        await asyncio.to_thread(
            self._record_sync, file_id, user, filename, mime, bytes_,
            kind, collection, chunks, uploaded_at, status,
        )

    def _record_sync(
        self, file_id: str, user: str, filename: str, mime: str,
        bytes_: int, kind: str, collection: str, chunks: int, uploaded_at: str,
        status: str,
    ) -> None:
        # Upsert, not INSERT OR REPLACE. The difference is the whole reason
        # this is not a one-liner.
        #
        # `INSERT OR REPLACE` deletes the conflicting row and inserts a new
        # one, so every column NOT named here silently reverts to its schema
        # default. Every one of `_UPLOADS_ADDED_COLUMNS` is absent from this
        # statement, and `reconcile_with_qdrant` calls this for every user file
        # on every boot — which meant a processed video lost its `summary` and
        # its `duration_s` at the next restart of `audrey-ai`, with no error
        # and nothing in the log. Measured, not theorised: a row completed with
        # a 1,257-character summary came back empty after one reconcile pass.
        #
        # The rule the ON CONFLICT clause encodes: **Qdrant is authoritative
        # about content, never about job lifecycle.** A payload knows the
        # filename, the size and the chunk count. It does not know how many
        # attempts a job took, why it failed, or what its summary said — those
        # exist only here, so reconcile must leave them alone.
        #
        # `status` is deliberately in the second group. It arrives as a default
        # argument rather than as something read from Qdrant, so honouring it
        # on conflict would let a boot flip a 'processing' or 'failed' row to
        # 'ready' on the strength of a value the caller never supplied.
        #
        # `source_url` (Phase 41) is in the same group for the same reason, and
        # the consequence is worth stating: a fetched video that reconciles at
        # boot keeps knowing where it came from. Naming it here would clear it
        # from the payload's silence, which is the `summary` bug again.
        #
        # So are `transcript_source` and `fetched_transcript` (Phase 41 step 4).
        # A Qdrant payload cannot know whether a human wrote the captions or
        # whisper guessed at them, and the failure would be the quiet kind: a
        # provenance label that is correct until the next restart and blank
        # afterwards, on rows nobody thinks to re-check.
        with self._lock:
            self._record_locked(
                file_id, user, filename, mime, bytes_, kind, collection,
                chunks, uploaded_at, status,
            )

    def _record_locked(
        self, file_id: str, user: str, filename: str, mime: str,
        bytes_: int, kind: str, collection: str, chunks: int, uploaded_at: str,
        status: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO uploads "
            "(file_id, user, filename, mime, bytes, kind, collection, "
            " chunks, uploaded_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(file_id) DO UPDATE SET "
            "  user = excluded.user, filename = excluded.filename, "
            "  mime = excluded.mime, bytes = excluded.bytes, "
            "  kind = excluded.kind, collection = excluded.collection, "
            "  chunks = excluded.chunks, uploaded_at = excluded.uploaded_at",
            (file_id, user, filename, mime, bytes_, kind, collection,
             chunks, uploaded_at, status),
        )

    async def commit_reserved_upload(
        self,
        *,
        reservation_id: str,
        reservation_kind: str,
        user: str,
        file_id: str,
        filename: str,
        mime: str,
        bytes_: int,
        kind: str,
        collection: str,
        chunks: int,
        uploaded_at: str,
        status: str,
        max_user_bytes: int,
    ) -> QuotaDecision:
        return await asyncio.to_thread(
            self._commit_reserved_upload_sync,
            reservation_id,
            reservation_kind,
            user,
            file_id,
            filename,
            mime,
            bytes_,
            kind,
            collection,
            chunks,
            uploaded_at,
            status,
            max_user_bytes,
        )

    def _commit_reserved_upload_sync(
        self,
        reservation_id: str,
        reservation_kind: str,
        user: str,
        file_id: str,
        filename: str,
        mime: str,
        bytes_: int,
        kind: str,
        collection: str,
        chunks: int,
        uploaded_at: str,
        status: str,
        max_user_bytes: int,
    ) -> QuotaDecision:
        with self._lock, self._transaction_locked():
            committed = self._conn.execute(
                "SELECT user FROM uploads WHERE file_id = ?",
                (file_id,),
            ).fetchone()
            if committed is not None:
                if committed["user"] != user:
                    raise ValueError("file id belongs to another user")
                usage = self._quota_usage_locked(user)
                return QuotaDecision(True, usage, bytes_, max_user_bytes)

            if reservation_kind == "single_shot":
                row = self._conn.execute(
                    "SELECT user, bytes FROM storage_reservations "
                    "WHERE reservation_id = ? AND kind = 'single_shot'",
                    (reservation_id,),
                ).fetchone()
                if row is None or row["user"] != user:
                    raise ValueError("single-shot reservation is missing or not owned")
                reserved_bytes = int(row["bytes"])
            elif reservation_kind == "chunked":
                row = self._conn.execute(
                    "SELECT s.user, s.total_bytes, COALESCE(SUM(p.bytes), 0) AS parts "
                    "FROM upload_sessions s LEFT JOIN upload_parts p USING (upload_id) "
                    "WHERE s.upload_id = ? GROUP BY s.upload_id",
                    (reservation_id,),
                ).fetchone()
                if row is None or row["user"] != user:
                    raise ValueError("chunk reservation is missing or not owned")
                reserved_bytes = max(int(row["total_bytes"]), int(row["parts"]))
            else:
                raise ValueError(f"unknown reservation kind {reservation_kind!r}")

            usage = self._quota_usage_locked(user)
            if usage.total_bytes - reserved_bytes + bytes_ > max_user_bytes:
                return QuotaDecision(False, usage, bytes_, max_user_bytes)

            self._record_locked(
                file_id,
                user,
                filename,
                mime,
                bytes_,
                kind,
                collection,
                chunks,
                uploaded_at,
                status,
            )
            if reservation_kind == "single_shot":
                self._conn.execute(
                    "DELETE FROM storage_reservations WHERE reservation_id = ?",
                    (reservation_id,),
                )
            else:
                self._conn.execute(
                    "DELETE FROM upload_parts WHERE upload_id = ?",
                    (reservation_id,),
                )
                self._conn.execute(
                    "DELETE FROM upload_sessions WHERE upload_id = ?",
                    (reservation_id,),
                )
            return QuotaDecision(
                True, self._quota_usage_locked(user), bytes_, max_user_bytes,
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

    async def request_file_deletion(
        self,
        file_id: str,
        *,
        user: str,
        requested_at: str,
    ) -> bool:
        """Tombstone a file before any external cleanup starts.

        Returns false only when neither an owned upload row nor an earlier
        tombstone exists. Repeating a delete against a pending tombstone is an
        idempotent retry; a completed tombstone stays completed.
        """
        return await asyncio.to_thread(
            self._request_file_deletion_sync,
            file_id,
            user,
            requested_at,
        )

    def _request_file_deletion_sync(
        self,
        file_id: str,
        user: str,
        requested_at: str,
    ) -> bool:
        with self._lock, self._transaction_locked():
            row = self._conn.execute(
                "SELECT 1 FROM uploads WHERE file_id = ? AND user = ?",
                (file_id, user),
            ).fetchone()
            tombstone = self._conn.execute(
                "SELECT completed_at FROM file_deletions "
                "WHERE file_id = ? AND user = ?",
                (file_id, user),
            ).fetchone()
            if row is None:
                return tombstone is not None

            self._conn.execute(
                """
                INSERT INTO file_deletions (file_id, user, requested_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file_id, user) DO UPDATE SET
                  completed_at = '', last_error = ''
                """,
                (file_id, user, requested_at),
            )
            self._conn.execute(
                "UPDATE uploads SET status = 'deleting', lease_id = '', leased_at = '' "
                "WHERE file_id = ? AND user = ?",
                (file_id, user),
            )
            return True

    async def pending_file_deletions(self, *, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._pending_file_deletions_sync, limit)

    def _pending_file_deletions_sync(self, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id, user, requested_at, attempts, "
                "       last_attempt_at, last_error "
                "FROM file_deletions WHERE completed_at = '' "
                "ORDER BY requested_at, file_id LIMIT ?",
                (max(1, limit),),
            ).fetchall()
            return [dict(row) for row in rows]

    async def pending_file_deletion_count(self) -> int:
        return await asyncio.to_thread(self._pending_file_deletion_count_sync)

    def _pending_file_deletion_count_sync(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM file_deletions WHERE completed_at = ''"
            ).fetchone()
            return int(row["count"])

    async def file_deletion_stats(self) -> dict[str, int]:
        """Global repair counts for the authenticated admin surface."""
        return await asyncio.to_thread(self._file_deletion_stats_sync)

    def _file_deletion_stats_sync(self) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                    THEN 1 ELSE 0 END), 0) AS pending,
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                    THEN attempts ELSE 0 END), 0) AS attempts,
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                      AND length(last_error) > 0
                                    THEN 1 ELSE 0 END), 0) AS with_error,
                  COALESCE(SUM(CASE WHEN length(completed_at) > 0
                                    THEN 1 ELSE 0 END), 0) AS completed
                FROM file_deletions
                """
            ).fetchone()
        return {
            "pending": int(row["pending"]),
            "attempts": int(row["attempts"]),
            "with_error": int(row["with_error"]),
            "exhausted": 0,
            "completed": int(row["completed"]),
        }

    async def user_file_deletion_stats(self, user: str) -> dict[str, int]:
        """Current-user repair counts without exposing another owner or errors."""
        return await asyncio.to_thread(
            self._user_file_deletion_stats_sync,
            user,
        )

    def _user_file_deletion_stats_sync(self, user: str) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                    THEN 1 ELSE 0 END), 0) AS pending,
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                    THEN attempts ELSE 0 END), 0) AS attempts,
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                      AND length(last_error) > 0
                                    THEN 1 ELSE 0 END), 0) AS with_error,
                  COALESCE(SUM(CASE WHEN length(completed_at) > 0
                                    THEN 1 ELSE 0 END), 0) AS completed
                FROM file_deletions
                WHERE user = ?
                """,
                (user,),
            ).fetchone()
        return {
            "pending": int(row["pending"]),
            "attempts": int(row["attempts"]),
            "with_error": int(row["with_error"]),
            "exhausted": 0,
            "completed": int(row["completed"]),
        }

    async def file_deletion_keys(self) -> set[tuple[str, str]]:
        """Every durable tombstone, including completed deletion history."""
        return await asyncio.to_thread(self._file_deletion_keys_sync)

    def _file_deletion_keys_sync(self) -> set[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id, user FROM file_deletions"
            ).fetchall()
            return {(str(row["user"]), str(row["file_id"])) for row in rows}

    async def file_deletion_ids(self, user: str) -> set[str]:
        return await asyncio.to_thread(self._file_deletion_ids_sync, user)

    def _file_deletion_ids_sync(self, user: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id FROM file_deletions WHERE user = ?",
                (user,),
            ).fetchall()
            return {str(row["file_id"]) for row in rows}

    async def begin_file_deletion_attempt(
        self,
        file_id: str,
        *,
        user: str,
        attempted_at: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._begin_file_deletion_attempt_sync,
            file_id,
            user,
            attempted_at,
        )

    def _begin_file_deletion_attempt_sync(
        self,
        file_id: str,
        user: str,
        attempted_at: str,
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE file_deletions "
                "SET attempts = attempts + 1, last_attempt_at = ?, last_error = '' "
                "WHERE file_id = ? AND user = ? AND completed_at = ''",
                (attempted_at, file_id, user),
            )
            return cursor.rowcount > 0

    async def fail_file_deletion(
        self,
        file_id: str,
        *,
        user: str,
        error: str,
    ) -> None:
        await asyncio.to_thread(
            self._fail_file_deletion_sync,
            file_id,
            user,
            error,
        )

    def _fail_file_deletion_sync(self, file_id: str, user: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE file_deletions SET last_error = ? "
                "WHERE file_id = ? AND user = ? AND completed_at = ''",
                (error[:500], file_id, user),
            )

    async def complete_file_deletion(
        self,
        file_id: str,
        *,
        user: str,
        completed_at: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._complete_file_deletion_sync,
            file_id,
            user,
            completed_at,
        )

    def _complete_file_deletion_sync(
        self,
        file_id: str,
        user: str,
        completed_at: str,
    ) -> bool:
        with self._lock, self._transaction_locked():
            cursor = self._conn.execute(
                "UPDATE file_deletions "
                "SET completed_at = ?, last_error = '' "
                "WHERE file_id = ? AND user = ? AND completed_at = ''",
                (completed_at, file_id, user),
            )
            if cursor.rowcount == 0:
                return False
            self._conn.execute(
                "DELETE FROM uploads WHERE file_id = ? AND user = ?",
                (file_id, user),
            )
            return True

    async def user_data_purge_stats(self, user: str) -> dict[str, int]:
        """Current-user account-purge repair counts without raw errors."""
        return await asyncio.to_thread(self._user_data_purge_stats_sync, user)

    async def data_purge_stats(self) -> dict[str, int]:
        """Global account-purge repair counts for authenticated admins."""
        return await asyncio.to_thread(self._data_purge_stats_sync)

    def _data_purge_stats_sync(self) -> dict[str, int]:
        with self._lock:
            purge = self._conn.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                    THEN 1 ELSE 0 END), 0) AS pending,
                  COALESCE(SUM(local_delivery_attempts + sidecar_attempts), 0)
                    AS attempts,
                  COALESCE(SUM(CASE WHEN length(local_delivery_last_error) > 0
                                      OR length(sidecar_last_error) > 0
                                    THEN 1 ELSE 0 END), 0) AS with_error,
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                      AND sidecar_status = ?
                                    THEN 1 ELSE 0 END), 0) AS exhausted,
                  COALESCE(SUM(CASE WHEN length(completed_at) > 0
                                    THEN 1 ELSE 0 END), 0) AS completed
                FROM user_data_purges
                """,
                ("attention_required",),
            ).fetchone()
            paths = self._conn.execute(
                """
                SELECT COALESCE(SUM(p.attempts), 0) AS attempts,
                       COALESCE(SUM(CASE WHEN length(p.last_error) > 0
                                         THEN 1 ELSE 0 END), 0) AS with_error
                FROM user_data_purge_paths AS p
                JOIN user_data_purges AS u ON u.purge_id = p.purge_id
                WHERE length(u.completed_at) = 0
                """
            ).fetchone()
            files = self._conn.execute(
                """
                SELECT COALESCE(SUM(d.attempts), 0) AS attempts,
                       COALESCE(SUM(CASE WHEN length(d.last_error) > 0
                                         THEN 1 ELSE 0 END), 0) AS with_error
                FROM user_data_purge_files AS p
                JOIN user_data_purges AS u ON u.purge_id = p.purge_id
                LEFT JOIN file_deletions AS d
                  ON d.file_id = p.file_id AND d.user = p.user
                WHERE length(u.completed_at) = 0
                """
            ).fetchone()
        return {
            "pending": int(purge["pending"]),
            "attempts": (
                int(purge["attempts"])
                + int(paths["attempts"])
                + int(files["attempts"])
            ),
            "with_error": (
                int(purge["with_error"])
                + int(paths["with_error"])
                + int(files["with_error"])
            ),
            "exhausted": int(purge["exhausted"]),
            "completed": int(purge["completed"]),
        }

    def _user_data_purge_stats_sync(self, user: str) -> dict[str, int]:
        with self._lock:
            purge = self._conn.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                    THEN 1 ELSE 0 END), 0) AS pending,
                  COALESCE(SUM(local_delivery_attempts + sidecar_attempts), 0)
                    AS attempts,
                  COALESCE(SUM(CASE WHEN length(local_delivery_last_error) > 0
                                      OR length(sidecar_last_error) > 0
                                    THEN 1 ELSE 0 END), 0) AS with_error,
                  COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                      AND sidecar_status = ?
                                    THEN 1 ELSE 0 END), 0) AS exhausted,
                  COALESCE(SUM(CASE WHEN length(completed_at) > 0
                                    THEN 1 ELSE 0 END), 0) AS completed
                FROM user_data_purges
                WHERE user = ?
                """,
                ("attention_required", user),
            ).fetchone()
            paths = self._conn.execute(
                """
                SELECT COALESCE(SUM(p.attempts), 0) AS attempts,
                       COALESCE(SUM(CASE WHEN length(p.last_error) > 0
                                         THEN 1 ELSE 0 END), 0) AS with_error
                FROM user_data_purge_paths AS p
                JOIN user_data_purges AS u ON u.purge_id = p.purge_id
                WHERE u.user = ? AND length(u.completed_at) = 0
                """,
                (user,),
            ).fetchone()
            files = self._conn.execute(
                """
                SELECT COALESCE(SUM(d.attempts), 0) AS attempts,
                       COALESCE(SUM(CASE WHEN length(d.last_error) > 0
                                         THEN 1 ELSE 0 END), 0) AS with_error
                FROM user_data_purge_files AS p
                JOIN user_data_purges AS u ON u.purge_id = p.purge_id
                LEFT JOIN file_deletions AS d
                  ON d.file_id = p.file_id AND d.user = p.user
                WHERE u.user = ? AND length(u.completed_at) = 0
                """,
                (user,),
            ).fetchone()
        return {
            "pending": int(purge["pending"]),
            "attempts": (
                int(purge["attempts"])
                + int(paths["attempts"])
                + int(files["attempts"])
            ),
            "with_error": (
                int(purge["with_error"])
                + int(paths["with_error"])
                + int(files["with_error"])
            ),
            "exhausted": int(purge["exhausted"]),
            "completed": int(purge["completed"]),
        }

    async def request_user_data_purge(
        self,
        *,
        purge_id: str,
        user: str,
        cutoff_at: str,
        requested_at: str,
    ) -> dict[str, int]:
        """Atomically create a receipt and tombstone every local pre-cutoff row."""
        return await asyncio.to_thread(
            self._request_user_data_purge_sync,
            purge_id,
            user,
            cutoff_at,
            requested_at,
        )

    def _request_user_data_purge_sync(
        self,
        purge_id: str,
        user: str,
        cutoff_at: str,
        requested_at: str,
    ) -> dict[str, int]:
        with self._lock, self._transaction_locked():
            existing = self._conn.execute(
                "SELECT user, cutoff_at FROM user_data_purges WHERE purge_id = ?",
                (purge_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["user"]) != user or str(existing["cutoff_at"]) != cutoff_at:
                    raise ValueError("purge_id is already bound to another request")
                return {"files": 0, "paths": 0, "created": 0}

            self._conn.execute(
                "INSERT INTO user_data_purges "
                "(purge_id, user, cutoff_at, requested_at) VALUES (?, ?, ?, ?)",
                (purge_id, user, cutoff_at, requested_at),
            )
            files = self._conn.execute(
                "SELECT file_id FROM uploads WHERE user = ? AND uploaded_at <= ?",
                (user, cutoff_at),
            ).fetchall()
            self._conn.executemany(
                "INSERT OR IGNORE INTO user_data_purge_files "
                "(purge_id, file_id, user) VALUES (?, ?, ?)",
                [(purge_id, str(row["file_id"]), user) for row in files],
            )
            for row in files:
                self._conn.execute(
                    """
                    INSERT INTO file_deletions (file_id, user, requested_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(file_id, user) DO UPDATE SET
                      completed_at = ?, last_error = ?
                    """,
                    (str(row["file_id"]), user, requested_at, "", ""),
                )
            self._conn.execute(
                "UPDATE uploads SET status = ?, lease_id = ?, leased_at = ? "
                "WHERE user = ? AND uploaded_at <= ?",
                ("deleting", "", "", user, cutoff_at),
            )

            sessions = self._conn.execute(
                "SELECT upload_id FROM upload_sessions "
                "WHERE user = ? AND created_at <= ?",
                (user, cutoff_at),
            ).fetchall()
            reservations = self._conn.execute(
                "SELECT reservation_id FROM storage_reservations "
                "WHERE user = ? AND created_at <= ?",
                (user, cutoff_at),
            ).fetchall()
            path_rows = [
                (purge_id, user, "session", str(row["upload_id"]))
                for row in sessions
            ] + [
                (purge_id, user, "file_prefix", str(row["reservation_id"]))
                for row in reservations
            ]
            self._conn.executemany(
                "INSERT OR IGNORE INTO user_data_purge_paths "
                "(purge_id, user, kind, object_id) VALUES (?, ?, ?, ?)",
                path_rows,
            )
            self._conn.executemany(
                "DELETE FROM upload_parts WHERE upload_id = ?",
                [(str(row["upload_id"]),) for row in sessions],
            )
            self._conn.execute(
                "DELETE FROM upload_sessions WHERE user = ? AND created_at <= ?",
                (user, cutoff_at),
            )
            self._conn.execute(
                "DELETE FROM storage_reservations WHERE user = ? AND created_at <= ?",
                (user, cutoff_at),
            )
            return {
                "files": len(files),
                "paths": len(path_rows),
                "created": 1,
            }

    async def pending_user_data_purges(self, *, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._pending_user_data_purges_sync, limit)

    def _pending_user_data_purges_sync(self, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM user_data_purges WHERE length(completed_at) = 0 "
                "ORDER BY requested_at, purge_id LIMIT ?",
                (max(1, limit),),
            ).fetchall()
            return [dict(row) for row in rows]

    async def unacknowledged_user_data_purges(self) -> list[dict]:
        """Receipts whose sidecar cutoff is not yet known to be installed."""
        return await asyncio.to_thread(
            self._unacknowledged_user_data_purges_sync
        )

    def _unacknowledged_user_data_purges_sync(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT purge_id, user FROM user_data_purges "
                "WHERE length(completed_at) = 0 "
                "AND length(sidecar_acknowledged_at) = 0 "
                "ORDER BY requested_at, purge_id"
            ).fetchall()
            return [dict(row) for row in rows]

    async def get_user_data_purge(self, purge_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_user_data_purge_sync, purge_id)

    def _get_user_data_purge_sync(self, purge_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM user_data_purges WHERE purge_id = ?",
                (purge_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    async def begin_user_data_purge_component(
        self,
        purge_id: str,
        *,
        component: str,
        attempted_at: str,
    ) -> None:
        await asyncio.to_thread(
            self._begin_user_data_purge_component_sync,
            purge_id,
            component,
            attempted_at,
        )

    def _begin_user_data_purge_component_sync(
        self,
        purge_id: str,
        component: str,
        attempted_at: str,
    ) -> None:
        if component == "local_delivery":
            sql = (
                "UPDATE user_data_purges SET local_delivery_attempts = "
                "local_delivery_attempts + 1, local_delivery_last_error = ? "
                "WHERE purge_id = ?"
            )
        elif component == "sidecar":
            sql = (
                "UPDATE user_data_purges SET sidecar_attempts = "
                "sidecar_attempts + 1, sidecar_last_error = ? WHERE purge_id = ?"
            )
        else:
            raise ValueError("unknown purge component")
        with self._lock:
            self._conn.execute(sql, ("", purge_id))

    async def finish_user_data_purge_component(
        self,
        purge_id: str,
        *,
        component: str,
        completed_at: str,
        status: str = "completed",
    ) -> None:
        await asyncio.to_thread(
            self._finish_user_data_purge_component_sync,
            purge_id,
            component,
            completed_at,
            status,
        )

    def _finish_user_data_purge_component_sync(
        self,
        purge_id: str,
        component: str,
        completed_at: str,
        status: str,
    ) -> None:
        if component == "local_delivery":
            sql = (
                "UPDATE user_data_purges SET local_delivery_completed_at = ?, "
                "local_delivery_last_error = ? WHERE purge_id = ?"
            )
            params = (completed_at, "", purge_id)
        elif component == "sidecar":
            sql = (
                "UPDATE user_data_purges SET sidecar_acknowledged_at = ?, "
                "sidecar_completed_at = ?, "
                "sidecar_status = ?, sidecar_last_error = ? WHERE purge_id = ?"
            )
            params = (completed_at, completed_at, status, "", purge_id)
        else:
            raise ValueError("unknown purge component")
        with self._lock:
            self._conn.execute(sql, params)

    async def acknowledge_user_data_purge_sidecar(
        self,
        purge_id: str,
        *,
        acknowledged_at: str,
        status: str,
    ) -> None:
        await asyncio.to_thread(
            self._acknowledge_user_data_purge_sidecar_sync,
            purge_id,
            acknowledged_at,
            status,
        )

    def _acknowledge_user_data_purge_sidecar_sync(
        self,
        purge_id: str,
        acknowledged_at: str,
        status: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE user_data_purges SET sidecar_acknowledged_at = ?, "
                "sidecar_status = ?, sidecar_last_error = ? WHERE purge_id = ?",
                (acknowledged_at, status, "", purge_id),
            )

    async def fail_user_data_purge_component(
        self,
        purge_id: str,
        *,
        component: str,
        error: str,
        status: str = "pending",
    ) -> None:
        await asyncio.to_thread(
            self._fail_user_data_purge_component_sync,
            purge_id,
            component,
            error,
            status,
        )

    def _fail_user_data_purge_component_sync(
        self,
        purge_id: str,
        component: str,
        error: str,
        status: str,
    ) -> None:
        if component == "local_delivery":
            sql = (
                "UPDATE user_data_purges SET local_delivery_last_error = ? "
                "WHERE purge_id = ?"
            )
            params = (error[:500], purge_id)
        elif component == "sidecar":
            sql = (
                "UPDATE user_data_purges SET sidecar_status = ?, "
                "sidecar_last_error = ? WHERE purge_id = ?"
            )
            params = (status, error[:500], purge_id)
        else:
            raise ValueError("unknown purge component")
        with self._lock:
            self._conn.execute(sql, params)

    async def pending_user_data_purge_paths(
        self,
        purge_id: str,
        *,
        limit: int = 50,
    ) -> list[dict]:
        return await asyncio.to_thread(
            self._pending_user_data_purge_paths_sync,
            purge_id,
            limit,
        )

    def _pending_user_data_purge_paths_sync(
        self,
        purge_id: str,
        limit: int,
    ) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM user_data_purge_paths "
                "WHERE purge_id = ? AND length(completed_at) = 0 "
                "ORDER BY kind, object_id LIMIT ?",
                (purge_id, max(1, limit)),
            ).fetchall()
            return [dict(row) for row in rows]

    async def begin_user_data_purge_path(
        self,
        purge_id: str,
        *,
        kind: str,
        object_id: str,
        attempted_at: str,
    ) -> None:
        await asyncio.to_thread(
            self._begin_user_data_purge_path_sync,
            purge_id,
            kind,
            object_id,
            attempted_at,
        )

    def _begin_user_data_purge_path_sync(
        self,
        purge_id: str,
        kind: str,
        object_id: str,
        attempted_at: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE user_data_purge_paths SET attempts = attempts + 1, "
                "last_attempt_at = ?, last_error = ? "
                "WHERE purge_id = ? AND kind = ? AND object_id = ? "
                "AND length(completed_at) = 0",
                (attempted_at, "", purge_id, kind, object_id),
            )

    async def finish_user_data_purge_path(
        self,
        purge_id: str,
        *,
        kind: str,
        object_id: str,
        completed_at: str,
        error: str = "",
    ) -> None:
        await asyncio.to_thread(
            self._finish_user_data_purge_path_sync,
            purge_id,
            kind,
            object_id,
            completed_at,
            error,
        )

    def _finish_user_data_purge_path_sync(
        self,
        purge_id: str,
        kind: str,
        object_id: str,
        completed_at: str,
        error: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE user_data_purge_paths SET completed_at = ?, last_error = ? "
                "WHERE purge_id = ? AND kind = ? AND object_id = ?",
                (completed_at, error[:500], purge_id, kind, object_id),
            )

    async def fail_user_data_purge_path(
        self,
        purge_id: str,
        *,
        kind: str,
        object_id: str,
        error: str,
    ) -> None:
        await asyncio.to_thread(
            self._fail_user_data_purge_path_sync,
            purge_id,
            kind,
            object_id,
            error,
        )

    def _fail_user_data_purge_path_sync(
        self,
        purge_id: str,
        kind: str,
        object_id: str,
        error: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE user_data_purge_paths SET last_error = ? "
                "WHERE purge_id = ? AND kind = ? AND object_id = ?",
                (error[:500], purge_id, kind, object_id),
            )

    async def user_data_purge_status(
        self,
        *,
        user: str,
        purge_id: str,
    ) -> dict | None:
        return await asyncio.to_thread(
            self._user_data_purge_status_sync,
            user,
            purge_id,
        )

    def _user_data_purge_status_sync(
        self,
        user: str,
        purge_id: str,
    ) -> dict | None:
        with self._lock:
            purge = self._conn.execute(
                "SELECT * FROM user_data_purges WHERE user = ? AND purge_id = ?",
                (user, purge_id),
            ).fetchone()
            if purge is None:
                return None
            files = self._conn.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN d.completed_at IS NULL
                                          OR length(d.completed_at) = 0
                                         THEN 1 ELSE 0 END), 0) AS pending,
                       COALESCE(SUM(d.attempts), 0) AS attempts,
                       COALESCE(SUM(CASE WHEN length(d.last_error) > 0
                                         THEN 1 ELSE 0 END), 0) AS with_error
                FROM user_data_purge_files AS p
                LEFT JOIN file_deletions AS d
                  ON d.file_id = p.file_id AND d.user = p.user
                WHERE p.purge_id = ? AND p.user = ?
                """,
                (purge_id, user),
            ).fetchone()
            paths = self._conn.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN length(completed_at) = 0
                                         THEN 1 ELSE 0 END), 0) AS pending,
                       COALESCE(SUM(attempts), 0) AS attempts,
                       COALESCE(SUM(CASE WHEN length(last_error) > 0
                                         THEN 1 ELSE 0 END), 0) AS with_error
                FROM user_data_purge_paths
                WHERE purge_id = ? AND user = ?
                """,
                (purge_id, user),
            ).fetchone()

        completed_at = str(purge["completed_at"] or "")
        sidecar_status = str(purge["sidecar_status"] or "pending")
        if completed_at:
            status = "completed"
        elif sidecar_status == "attention_required":
            status = "attention_required"
        else:
            status = "pending"
        return {
            "schema_version": 1,
            "purge_id": purge_id,
            "cutoff_at": str(purge["cutoff_at"]),
            "requested_at": str(purge["requested_at"]),
            "status": status,
            "completed_at": completed_at,
            "files": {
                "pending": int(files["pending"]),
                "attempts": int(files["attempts"]),
                "with_error": int(files["with_error"]),
                "completed": int(files["total"]) - int(files["pending"]),
            },
            "paths": {
                "pending": int(paths["pending"]),
                "attempts": int(paths["attempts"]),
                "with_error": int(paths["with_error"]),
                "completed": int(paths["total"]) - int(paths["pending"]),
            },
            "local_delivery": {
                "completed": bool(purge["local_delivery_completed_at"]),
                "attempts": int(purge["local_delivery_attempts"]),
                "with_error": bool(purge["local_delivery_last_error"]),
            },
            "sidecar": {
                "acknowledged": bool(purge["sidecar_acknowledged_at"]),
                "completed": bool(purge["sidecar_completed_at"]),
                "status": sidecar_status,
                "attempts": int(purge["sidecar_attempts"]),
                "with_error": bool(purge["sidecar_last_error"]),
            },
        }

    async def finalize_user_data_purge(
        self,
        purge_id: str,
        *,
        completed_at: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._finalize_user_data_purge_sync,
            purge_id,
            completed_at,
        )

    def _finalize_user_data_purge_sync(
        self,
        purge_id: str,
        completed_at: str,
    ) -> bool:
        with self._lock, self._transaction_locked():
            row = self._conn.execute(
                "SELECT local_delivery_completed_at, sidecar_completed_at "
                "FROM user_data_purges WHERE purge_id = ?",
                (purge_id,),
            ).fetchone()
            if row is None or not row["local_delivery_completed_at"] or not row["sidecar_completed_at"]:
                return False
            file_pending = int(self._conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM user_data_purge_files AS p
                LEFT JOIN file_deletions AS d
                  ON d.file_id = p.file_id AND d.user = p.user
                WHERE p.purge_id = ?
                  AND (d.completed_at IS NULL OR length(d.completed_at) = 0)
                """,
                (purge_id,),
            ).fetchone()["count"])
            path_pending = int(self._conn.execute(
                "SELECT COUNT(*) AS count FROM user_data_purge_paths "
                "WHERE purge_id = ? AND length(completed_at) = 0",
                (purge_id,),
            ).fetchone()["count"])
            if file_pending or path_pending:
                return False
            self._conn.execute(
                "UPDATE user_data_purges SET completed_at = ? WHERE purge_id = ?",
                (completed_at, purge_id),
            )
            return True

    async def list_user(self, user: str) -> list[dict]:
        return await asyncio.to_thread(self._list_user_sync, user)

    def _list_user_sync(self, user: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                # Every column `FileRow` declares has to be in this list. A
                # column added to the schema and the migration but not here
                # exists in the database and still raises on read, which
                # presents as a 500 on the file list rather than as anything
                # resembling a missing column. `TestListReturnsEveryFileRowField`
                # pins the two together.
                "SELECT file_id, filename, mime, bytes, kind, collection, "
                "       chunks, uploaded_at, status, failure_reason, duration_s, "
                "       summary, source_freed_at, leased_at, source_url, "
                "       transcript_source, fetch_downloaded_bytes, "
                "       fetch_total_bytes "
                "FROM uploads WHERE user = ? AND status != 'deleting' "
                "ORDER BY uploaded_at DESC",
                (user,),
            )
            return [dict(r) for r in cur.fetchall()]

    async def user_total_bytes(self, user: str) -> int:
        return await asyncio.to_thread(self._user_total_sync, user)

    def _user_total_sync(self, user: str) -> int:
        with self._lock:
            # Rows whose source has been reclaimed contribute nothing, because
            # the quota exists to bound bytes on disk and those bytes are gone.
            #
            # Note what is NOT done here: setting `bytes = 0` on reclamation
            # would be simpler and would be undone at the next boot, because
            # `reconcile_with_qdrant` refreshes `bytes` from the Qdrant payload
            # — which still carries the original size and always will. Freed
            # space that silently un-frees itself on restart is worse than no
            # reclamation at all, so the flag lives in a column reconcile does
            # not touch and `bytes` stays true to what was uploaded.
            cur = self._conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN source_freed_at = '' THEN bytes END), 0) "
                "AS total FROM uploads WHERE user = ?",
                (user,),
            )
            return int(cur.fetchone()["total"])
    async def quota_usage(self, user: str) -> QuotaUsage:
        return await asyncio.to_thread(self._quota_usage_sync, user)

    def _quota_usage_sync(self, user: str) -> QuotaUsage:
        with self._lock:
            return self._quota_usage_locked(user)

    def _quota_usage_locked(self, user: str) -> QuotaUsage:
        stored = int(self._conn.execute(
            "SELECT COALESCE(SUM(bytes), 0) AS total FROM uploads "
            "WHERE user = ? AND source_freed_at = ''",
            (user,),
        ).fetchone()["total"])
        session = self._conn.execute(
            "WITH part_totals AS ("
            "  SELECT upload_id, COALESCE(SUM(bytes), 0) AS part_bytes "
            "  FROM upload_parts GROUP BY upload_id"
            ") "
            "SELECT COALESCE(SUM(s.total_bytes), 0) AS declared, "
            "       COALESCE(SUM(COALESCE(p.part_bytes, 0)), 0) AS parts, "
            "       COALESCE(SUM(CASE "
            "         WHEN COALESCE(p.part_bytes, 0) > s.total_bytes "
            "         THEN p.part_bytes - s.total_bytes ELSE 0 END), 0) AS overage "
            "FROM upload_sessions s LEFT JOIN part_totals p USING (upload_id) "
            "WHERE s.user = ?",
            (user,),
        ).fetchone()
        url_fetch = int(self._conn.execute(
            "SELECT COALESCE(SUM(quota_reserved_bytes), 0) AS total "
            "FROM uploads WHERE user = ? AND quota_reserved_bytes > 0",
            (user,),
        ).fetchone()["total"])
        single = int(self._conn.execute(
            "SELECT COALESCE(SUM(bytes), 0) AS total "
            "FROM storage_reservations WHERE user = ?",
            (user,),
        ).fetchone()["total"])
        return QuotaUsage(
            stored_bytes=stored,
            chunk_declared_bytes=int(session["declared"]),
            chunk_part_bytes=int(session["parts"]),
            chunk_part_overage_bytes=int(session["overage"]),
            url_fetch_bytes=url_fetch,
            single_shot_bytes=single,
        )

    async def reserve_single_upload(
        self,
        *,
        reservation_id: str,
        user: str,
        bytes_: int,
        max_user_bytes: int,
        now: str,
        expired_before: str,
    ) -> QuotaDecision:
        return await asyncio.to_thread(
            self._reserve_single_sync,
            reservation_id,
            user,
            bytes_,
            max_user_bytes,
            now,
            expired_before,
        )

    def _reserve_single_sync(
        self,
        reservation_id: str,
        user: str,
        bytes_: int,
        max_user_bytes: int,
        now: str,
        expired_before: str,
    ) -> QuotaDecision:
        if bytes_ < 0:
            raise ValueError("reservation bytes must be non-negative")
        with self._lock, self._transaction_locked():
            self._conn.execute(
                "DELETE FROM storage_reservations "
                "WHERE kind = 'single_shot' AND updated_at < ?",
                (expired_before,),
            )
            existing = self._conn.execute(
                "SELECT user, kind, bytes FROM storage_reservations "
                "WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if existing is not None and (
                existing["user"] != user or existing["kind"] != "single_shot"
            ):
                raise ValueError("reservation id belongs to another lifecycle")
            usage = self._quota_usage_locked(user)
            old_bytes = int(existing["bytes"]) if existing is not None else 0
            if usage.total_bytes - old_bytes + bytes_ > max_user_bytes:
                return QuotaDecision(False, usage, bytes_, max_user_bytes)
            self._conn.execute(
                "INSERT INTO storage_reservations "
                "(reservation_id, user, kind, bytes, created_at, updated_at) "
                "VALUES (?, ?, 'single_shot', ?, ?, ?) "
                "ON CONFLICT(reservation_id) DO UPDATE SET "
                "bytes = excluded.bytes, updated_at = excluded.updated_at",
                (reservation_id, user, bytes_, now, now),
            )
            return QuotaDecision(
                True, self._quota_usage_locked(user), bytes_, max_user_bytes,
            )

    async def release_storage_reservation(self, reservation_id: str, *, user: str) -> bool:
        return await asyncio.to_thread(
            self._release_storage_reservation_sync, reservation_id, user,
        )

    def _release_storage_reservation_sync(
        self, reservation_id: str, user: str,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM storage_reservations "
                "WHERE reservation_id = ? AND user = ?",
                (reservation_id, user),
            )
            return cur.rowcount > 0

    async def reserve_upload_session(
        self,
        *,
        upload_id: str,
        user: str,
        filename: str,
        total_bytes: int,
        part_size: int,
        parts_total: int,
        max_user_bytes: int,
        now: str,
        expired_before: str,
    ) -> QuotaDecision:
        return await asyncio.to_thread(
            self._reserve_session_sync,
            upload_id,
            user,
            filename,
            total_bytes,
            part_size,
            parts_total,
            max_user_bytes,
            now,
            expired_before,
        )

    def _reserve_session_sync(
        self,
        upload_id: str,
        user: str,
        filename: str,
        total_bytes: int,
        part_size: int,
        parts_total: int,
        max_user_bytes: int,
        now: str,
        expired_before: str,
    ) -> QuotaDecision:
        with self._lock, self._transaction_locked():
            self._conn.execute(
                "DELETE FROM storage_reservations "
                "WHERE kind = 'single_shot' AND updated_at < ?",
                (expired_before,),
            )
            existing = self._conn.execute(
                "SELECT user, total_bytes FROM upload_sessions WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if existing is not None:
                if existing["user"] != user:
                    raise ValueError("upload session belongs to another user")
                usage = self._quota_usage_locked(user)
                return QuotaDecision(True, usage, total_bytes, max_user_bytes)
            usage = self._quota_usage_locked(user)
            if usage.total_bytes + total_bytes > max_user_bytes:
                return QuotaDecision(False, usage, total_bytes, max_user_bytes)
            self._conn.execute(
                "INSERT INTO upload_sessions "
                "(upload_id, user, filename, total_bytes, part_size, "
                " parts_total, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    upload_id,
                    user,
                    filename,
                    total_bytes,
                    part_size,
                    parts_total,
                    now,
                    now,
                ),
            )
            return QuotaDecision(
                True, self._quota_usage_locked(user), total_bytes, max_user_bytes,
            )

    async def expire_upload_sessions(self, *, older_than: str) -> list[dict]:
        return await asyncio.to_thread(self._expire_sessions_sync, older_than)

    def _expire_sessions_sync(self, older_than: str) -> list[dict]:
        with self._lock, self._transaction_locked():
            rows = self._conn.execute(
                "SELECT * FROM upload_sessions WHERE updated_at < ? "
                "ORDER BY updated_at, upload_id",
                (older_than,),
            ).fetchall()
            for row in rows:
                self._conn.execute(
                    "DELETE FROM upload_parts WHERE upload_id = ?",
                    (row["upload_id"],),
                )
                self._conn.execute(
                    "DELETE FROM upload_sessions WHERE upload_id = ?",
                    (row["upload_id"],),
                )
            return [dict(row) for row in rows]


    async def all_users(self) -> list[str]:
        return await asyncio.to_thread(self._all_users_sync)

    def _all_users_sync(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute("SELECT DISTINCT user FROM uploads")
            return [r["user"] for r in cur.fetchall()]

    async def prunable_file_ids(self, user: str) -> set[str]:
        """File ids that *claim* Qdrant content, and so may be pruned if absent.

        `reconcile_with_qdrant`'s ghost sweep asks "is this row's content
        really in Qdrant?" — a question only worth asking of rows that
        assert there is any. Two kinds of row assert nothing and must be
        withheld from that sweep, or boot deletes them:

          - `status != 'ready'` — a pending or processing video hasn't been
            ingested yet, and a failed one never will be. Having no points
            is the expected state, not a ghost.
          - `chunks = 0` — a video the worker completed with no transcript
            (silent footage). It is 'ready' in the sense that the pipeline
            is done with it, but nothing was ever written to Qdrant.

        The upload path can't produce a 'ready' row with 0 chunks: images
        record 1 and empty text is rejected with `EmptyExtractionError`
        before any row is written. So `chunks = 0` here means a completed
        video specifically, and excluding it costs no real ghost coverage.
        """
        return await asyncio.to_thread(self._prunable_ids_sync, user)

    def _prunable_ids_sync(self, user: str) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT file_id FROM uploads "
                "WHERE user = ? AND status = 'ready' AND chunks > 0",
                (user,),
            )
            return {r["file_id"] for r in cur.fetchall()}

    # ── URL fetch lifecycle (Phase 41) ────────────────────────────────
    #
    # One stage in front of the video job lifecycle below, with the same
    # shape: a *_pending status waits, a claim leases it, a result hands it
    # on. What it hands on to is the existing 'pending' queue, so nothing in
    # `media/worker.py` learns that a URL was ever involved.

    async def reserve_url_fetch(
        self,
        *,
        file_id: str,
        user: str,
        source_url: str,
        filename: str,
        ceiling_bytes: int,
        max_user_bytes: int,
        uploaded_at: str,
        expired_before: str,
    ) -> QuotaDecision:
        return await asyncio.to_thread(
            self._reserve_url_fetch_sync,
            file_id,
            user,
            source_url,
            filename,
            ceiling_bytes,
            max_user_bytes,
            uploaded_at,
            expired_before,
        )

    def _reserve_url_fetch_sync(
        self,
        file_id: str,
        user: str,
        source_url: str,
        filename: str,
        ceiling_bytes: int,
        max_user_bytes: int,
        uploaded_at: str,
        expired_before: str,
    ) -> QuotaDecision:
        with self._lock, self._transaction_locked():
            self._conn.execute(
                "DELETE FROM storage_reservations "
                "WHERE kind = 'single_shot' AND updated_at < ?",
                (expired_before,),
            )
            existing = self._conn.execute(
                "SELECT user, source_url FROM uploads WHERE file_id = ?",
                (file_id,),
            ).fetchone()
            if existing is not None:
                if existing["user"] != user or existing["source_url"] != source_url:
                    raise ValueError("file id belongs to another upload lifecycle")
                usage = self._quota_usage_locked(user)
                return QuotaDecision(True, usage, ceiling_bytes, max_user_bytes)
            usage = self._quota_usage_locked(user)
            if usage.total_bytes + ceiling_bytes > max_user_bytes:
                return QuotaDecision(False, usage, ceiling_bytes, max_user_bytes)
            self._conn.execute(
                "INSERT INTO uploads "
                "(file_id, user, filename, mime, bytes, kind, collection, "
                " chunks, uploaded_at, status, source_url, quota_reserved_bytes) "
                "VALUES (?, ?, ?, '', 0, 'video', '', 0, ?, 'fetch_pending', ?, ?)",
                (
                    file_id,
                    user,
                    filename,
                    uploaded_at,
                    source_url,
                    ceiling_bytes,
                ),
            )
            return QuotaDecision(
                True, self._quota_usage_locked(user), ceiling_bytes, max_user_bytes,
            )

    async def restore_url_fetch_reservations(
        self, *, ceiling_bytes: int,
    ) -> UrlReservationRecovery:
        """Repair URL holds after restart and report both idempotent actions."""
        return await asyncio.to_thread(
            self._restore_url_fetch_reservations_sync, ceiling_bytes,
        )

    def _restore_url_fetch_reservations_sync(
        self, ceiling_bytes: int,
    ) -> UrlReservationRecovery:
        with self._lock, self._transaction_locked():
            released = self._conn.execute(
                "UPDATE uploads SET quota_reserved_bytes = 0 "
                "WHERE status NOT IN ('fetch_pending', 'fetching') "
                "AND quota_reserved_bytes != 0",
            ).rowcount
            restored = self._conn.execute(
                "UPDATE uploads SET quota_reserved_bytes = ? "
                "WHERE status IN ('fetch_pending', 'fetching') "
                "AND quota_reserved_bytes = 0",
                (ceiling_bytes,),
            ).rowcount
            return UrlReservationRecovery(
                restored_pending=restored,
                released_stranded=released,
            )

    async def reserve_url_refetch(
        self,
        *,
        file_id: str,
        user: str,
        ceiling_bytes: int,
        max_user_bytes: int,
        expired_before: str,
    ) -> QuotaDecision:
        return await asyncio.to_thread(
            self._reserve_url_refetch_sync,
            file_id,
            user,
            ceiling_bytes,
            max_user_bytes,
            expired_before,
        )

    def _reserve_url_refetch_sync(
        self,
        file_id: str,
        user: str,
        ceiling_bytes: int,
        max_user_bytes: int,
        expired_before: str,
    ) -> QuotaDecision:
        with self._lock, self._transaction_locked():
            self._conn.execute(
                "DELETE FROM storage_reservations "
                "WHERE kind = 'single_shot' AND updated_at < ?",
                (expired_before,),
            )
            row = self._conn.execute(
                "SELECT user, source_url, source_freed_at, quota_reserved_bytes "
                "FROM uploads WHERE file_id = ?",
                (file_id,),
            ).fetchone()
            if row is None or row["user"] != user:
                raise ValueError("no such user-owned URL fetch")
            if not row["source_url"] or not row["source_freed_at"]:
                raise ValueError("row is not a reclaimable URL fetch")
            usage = self._quota_usage_locked(user)
            old_bytes = int(row["quota_reserved_bytes"])
            if usage.total_bytes - old_bytes + ceiling_bytes > max_user_bytes:
                return QuotaDecision(False, usage, ceiling_bytes, max_user_bytes)
            self._conn.execute(
                "UPDATE uploads SET quota_reserved_bytes = ? WHERE file_id = ?",
                (ceiling_bytes, file_id),
            )
            return QuotaDecision(
                True, self._quota_usage_locked(user), ceiling_bytes, max_user_bytes,
            )

    async def release_url_reservation(self, file_id: str, *, user: str) -> bool:
        return await asyncio.to_thread(
            self._release_url_reservation_sync, file_id, user,
        )

    def _release_url_reservation_sync(self, file_id: str, user: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE uploads SET quota_reserved_bytes = 0 "
                "WHERE file_id = ? AND user = ? AND quota_reserved_bytes != 0",
                (file_id, user),
            )
            return cur.rowcount > 0

    async def record_url_fetch(
        self, *, file_id: str, user: str, source_url: str, filename: str,
        uploaded_at: str,
    ) -> None:
        """Insert a row that has a URL and no bytes yet.

        Deliberately not `record_upload` with a different status. That method
        upserts, and upserting here would let a second paste of the same URL
        silently reset a row a fetcher is already holding. This is a plain
        INSERT: a `file_id` collision is a uuid4 collision, which is a bug
        worth raising on rather than papering over.

        `bytes` is 0 and `mime` is empty because neither is known — the size
        arrives with the download and the mime is sniffed from the bytes, the
        same gate an upload passes. Writing a guess into either would make the
        quota and the mime allowlist read a value nobody measured.
        """
        await asyncio.to_thread(
            self._record_url_fetch_sync, file_id, user, source_url, filename,
            uploaded_at,
        )

    def _record_url_fetch_sync(
        self, file_id: str, user: str, source_url: str, filename: str,
        uploaded_at: str,
    ) -> None:
        with self._lock:
            # `kind` is 'video' from the start rather than being decided when
            # the bytes land, because the host allowlist is a video-site
            # allowlist — there is no other kind this path can produce. The
            # mime sniff on the result is still the gate, and it fails the row
            # rather than quietly reclassifying it.
            self._conn.execute(
                "INSERT INTO uploads "
                "(file_id, user, filename, mime, bytes, kind, collection, "
                " chunks, uploaded_at, status, source_url) "
                "VALUES (?, ?, ?, '', 0, 'video', '', 0, ?, 'fetch_pending', ?)",
                (file_id, user, filename, uploaded_at, source_url),
            )

    async def find_by_source_url(self, *, user: str, source_url: str) -> dict | None:
        """An existing row for this user and this exact URL, or None.

        Exact string match, and that is the whole promise. `youtu.be/X` and
        `youtube.com/watch?v=X` are the same video and will not match each
        other — canonicalising them means knowing the video id, which only
        yt-dlp knows and only after a metadata call. What this does catch is
        the double-clicked submit button and the paste of a link already in
        the list, which is the case that costs a second download of the same
        300 MB.

        'failed' rows are excluded: a fetch that failed is exactly the one a
        user would reasonably retry by pasting the link again.
        """
        return await asyncio.to_thread(self._find_by_url_sync, user, source_url)

    def _find_by_url_sync(self, user: str, source_url: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT file_id, filename, status FROM uploads "
                "WHERE user = ? AND source_url = ? AND status != 'failed' "
                "LIMIT 1",
                (user, source_url),
            ).fetchone()
            return dict(row) if row else None

    async def claim_fetch(self, *, lease_id: str, now: str) -> dict | None:
        """Lease the oldest row awaiting download, or None when none is."""
        return await asyncio.to_thread(self._claim_fetch_sync, lease_id, now)

    def _claim_fetch_sync(self, lease_id: str, now: str) -> dict | None:
        # SELECT and UPDATE under one lock hold, exactly as `_claim_job_sync`
        # does and for the same reason: the status transition is what stops two
        # fetchers taking the same row, so the two statements cannot be split.
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM uploads WHERE status = 'fetch_pending' "
                "ORDER BY uploaded_at ASC LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            attempts = int(row["attempts"]) + 1
            self._conn.execute(
                "UPDATE uploads SET status = 'fetching', lease_id = ?, "
                "       leased_at = ?, attempts = ? WHERE file_id = ?",
                (lease_id, now, attempts, row["file_id"]),
            )
            claimed = dict(row)
            claimed.update(
                status="fetching", lease_id=lease_id, leased_at=now, attempts=attempts,
            )
            return claimed

    async def record_fetch_progress(
        self, *, file_id: str, lease_id: str, title: str = "",
        downloaded_bytes: int = 0, total_bytes: int = 0,
    ) -> bool:
        """Note how far a download has got. False if the lease no longer holds.

        The lease guard is the same one every other handover has, and it is
        doing real work here rather than being belt-and-braces: a fetcher that
        stalled past its lease and woke up must not paint its stale byte count
        over the run that replaced it, which would show a download going
        backwards for no reason anyone could find afterwards.

        **This route cannot change a row's state.** It writes three display
        fields under a `status = 'fetching'` predicate and nothing else — no
        transition, no completion, no failure. That is what keeps a progress
        channel from being one more thing that can strand a row.

        `title` is written into `filename`, replacing the video-id placeholder
        `record_url_fetch` put there. Safe while fetching *because* the bytes
        are in staging: nothing derives a path from this column until
        `complete_fetch` writes the real name with its real extension.
        """
        return await asyncio.to_thread(
            self._record_progress_sync, file_id, lease_id, title,
            downloaded_bytes, total_bytes,
        )

    def _record_progress_sync(
        self, file_id: str, lease_id: str, title: str,
        downloaded_bytes: int, total_bytes: int,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE uploads SET fetch_downloaded_bytes = ?, "
                "       fetch_total_bytes = ?, "
                # An empty title leaves the existing name alone rather than
                # blanking it: a site with no title should not turn a row that
                # already had a usable placeholder into an unnamed one.
                "       filename = CASE WHEN ? <> '' THEN ? ELSE filename END "
                "WHERE file_id = ? AND lease_id = ? AND status = 'fetching'",
                (downloaded_bytes, total_bytes, title, title, file_id, lease_id),
            )
            return cur.rowcount > 0

    async def complete_fetch(
        self, *, file_id: str, lease_id: str, filename: str, mime: str, bytes_: int,
        transcript: str = "",
    ) -> bool:
        """Hand a downloaded row to the video queue. False if the lease lapsed.

        Everything the fetcher learned lands here at once — the real title, the
        sniffed mime, the size on disk and any caption track the site had — and
        the row becomes indistinguishable from an upload that has been waiting
        for the worker, apart from having its transcript already written.

        **`attempts` resets to 0.** The counter is per stage, and carrying a
        fetch's attempts into the ingest stage would hand a video that took two
        tries to download only one try to transcribe before `max_attempts`
        failed it. The two stages fail for unrelated reasons; sharing a budget
        between them makes a slow network look like a bad video.
        """
        return await asyncio.to_thread(
            self._complete_fetch_sync, file_id, lease_id, filename, mime, bytes_,
            transcript,
        )

    def _complete_fetch_sync(
        self, file_id: str, lease_id: str, filename: str, mime: str, bytes_: int,
        transcript: str,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                # The two progress columns zero out here: the download is over,
                # `bytes` is now the real number, and a stale "247 MB of 288 MB"
                # left on a ready row is a fact about a moment that has passed.
                "UPDATE uploads SET status = 'pending', filename = ?, mime = ?, "
                "       bytes = ?, lease_id = '', leased_at = '', attempts = 0, "
                "       failure_reason = '', fetched_transcript = ?, "
                "       fetch_downloaded_bytes = 0, fetch_total_bytes = 0, "
                "       quota_reserved_bytes = 0 "
                "WHERE file_id = ? AND lease_id = ? AND status = 'fetching'",
                (filename, mime, bytes_, transcript, file_id, lease_id),
            )
            return cur.rowcount > 0

    async def fetch_stage_file_ids(self) -> set[str]:
        """Every file_id somewhere in the fetch stage.

        The staging directory is swept against this: a file there whose id is
        not in this set belongs to no live download and is an orphan. Asking
        the database which rows are live is the only way to tell a partial
        download in progress from one abandoned three attempts ago — an age
        heuristic would eventually delete a slow one.
        """
        return await asyncio.to_thread(self._fetch_stage_ids_sync)

    def _fetch_stage_ids_sync(self) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT file_id FROM uploads "
                "WHERE status IN ('fetch_pending', 'fetching')",
            )
            return {str(r["file_id"]) for r in cur.fetchall()}

    async def work_queue_stats(self) -> dict[str, dict[str, int | str]]:
        """Aggregate media/fetch queue pressure without exposing owners."""
        return await asyncio.to_thread(self._work_queue_stats_sync)

    def _work_queue_stats_sync(self) -> dict[str, dict[str, int | str]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0)
                    AS processing_pending,
                  COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0)
                    AS processing_active,
                  COALESCE(MIN(CASE WHEN status = ? THEN uploaded_at END), ?)
                    AS processing_oldest,
                  COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0)
                    AS fetch_pending,
                  COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0)
                    AS fetch_active,
                  COALESCE(MIN(CASE WHEN status = ? THEN uploaded_at END), ?)
                    AS fetch_oldest
                FROM uploads
                """,
                (
                    "pending",
                    "processing",
                    "pending",
                    "",
                    "fetch_pending",
                    "fetching",
                    "fetch_pending",
                    "",
                ),
            ).fetchone()
        return {
            "media_processing": {
                "pending": int(row["processing_pending"]),
                "active": int(row["processing_active"]),
                "oldest_pending_at": str(row["processing_oldest"] or ""),
            },
            "media_fetch": {
                "pending": int(row["fetch_pending"]),
                "active": int(row["fetch_active"]),
                "oldest_pending_at": str(row["fetch_oldest"] or ""),
            },
        }

    # ── Video job lifecycle (Phase 33) ────────────────────────────────

    async def get_upload(self, file_id: str) -> dict | None:
        """Fetch one row by file_id, NOT double-keyed on user.

        Every other read here is keyed on user as well, deliberately. This one
        cannot be: the caller is the media worker, which is authenticated by
        service token and has no user of its own — resolving the owner is the
        whole reason it asks. The routes that use this are service-only for
        exactly that reason.
        """
        return await asyncio.to_thread(self._get_upload_sync, file_id)

    def _get_upload_sync(self, file_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM uploads WHERE file_id = ?", (file_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    async def claim_job(self, *, lease_id: str, now: str) -> dict | None:
        """Lease the oldest pending row, or None when nothing is waiting."""
        return await asyncio.to_thread(self._claim_job_sync, lease_id, now)

    def _claim_job_sync(self, lease_id: str, now: str) -> dict | None:
        # SELECT and UPDATE under one lock hold. With a single connection that
        # is the whole atomicity story: two workers polling at the same moment
        # serialize here, so the second sees 'processing' and takes the next
        # row rather than leasing the same one twice. Splitting this into two
        # public calls would reintroduce exactly that race.
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM uploads WHERE status = 'pending' "
                "ORDER BY uploaded_at ASC LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            attempts = int(row["attempts"]) + 1
            self._conn.execute(
                "UPDATE uploads SET status = 'processing', lease_id = ?, "
                "       leased_at = ?, attempts = ? WHERE file_id = ?",
                (lease_id, now, attempts, row["file_id"]),
            )
            claimed = dict(row)
            claimed.update(
                status="processing", lease_id=lease_id, leased_at=now, attempts=attempts,
            )
            return claimed

    async def complete_job(
        self, *, file_id: str, lease_id: str, collection: str, chunks: int,
        duration_s: float = 0.0, summary: str = "", completed_at: str = "",
        transcript_source: str = "",
    ) -> bool:
        """Flip a leased row to 'ready'. False if the lease no longer holds.

        The `lease_id` and `status` predicates are the guard against a late
        worker: one that stalled past its lease, had the job swept back and
        re-leased, then finally woke up and posted. Without them that stale
        result overwrites the newer one and the row looks perfectly healthy
        afterwards — no error, no log line, just a transcript that does not
        match its video.

        This is where `fetched_transcript` is cleared — the transcript it held
        is now chunked, indexed and on disk as a sidecar, so the column would
        otherwise be a third copy of the same text sitting on a row that
        `SELECT *` reads on every claim.
        """
        return await asyncio.to_thread(
            self._complete_job_sync, file_id, lease_id, collection, chunks,
            duration_s, summary, completed_at, transcript_source,
        )

    def _complete_job_sync(
        self, file_id: str, lease_id: str, collection: str, chunks: int,
        duration_s: float, summary: str, completed_at: str, transcript_source: str,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE uploads SET status = 'ready', collection = ?, chunks = ?, "
                "       lease_id = '', leased_at = '', failure_reason = '', "
                "       duration_s = ?, summary = ?, completed_at = ?, "
                "       transcript_source = ?, fetched_transcript = '' "
                "WHERE file_id = ? AND lease_id = ? AND status = 'processing'",
                (collection, chunks, duration_s, summary, completed_at,
                 transcript_source, file_id, lease_id),
            )
            return cur.rowcount > 0

    async def fail_job(
        self, *, file_id: str, lease_id: str, reason: str,
        stage: str = "processing",
    ) -> bool:
        """Mark a leased row 'failed' with a reason. Same lease guard as completion.

        `stage` names which leased status the caller expects to hold — the
        media worker's 'processing' or the fetcher's 'fetching'. The lease id
        is a uuid4 that only the current holder has, so the status predicate is
        already belt-and-braces; naming it keeps a fetcher from failing a row
        that has moved on to ingest, which is the one way the two stages could
        reach across each other.
        """
        return await asyncio.to_thread(
            self._fail_job_sync, file_id, lease_id, reason, stage,
        )

    def _fail_job_sync(
        self, file_id: str, lease_id: str, reason: str, stage: str,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE uploads SET status = 'failed', failure_reason = ?, "
                "       lease_id = '', leased_at = '', "
                "       quota_reserved_bytes = CASE WHEN status = 'fetching' "
                "         THEN 0 ELSE quota_reserved_bytes END "
                "WHERE file_id = ? AND lease_id = ? AND status = ?",
                (reason[:500], file_id, lease_id, stage),
            )
            return cur.rowcount > 0

    async def requeue_job(
        self, file_id: str, *, bytes_: int | None = None, refetch: bool = False,
    ) -> bool:
        """Put a row back in the queue as if it had just been uploaded.

        False when there is no such row.

        `refetch` sends it back to `fetch_pending` instead of `pending`, for a
        URL-sourced video whose bytes were reclaimed: there is nothing on disk
        for a media worker to open, and the row carries the address of the
        thing that produced them. This is what makes reclaiming a fetched
        source recoverable rather than merely cheap, and `kb.video.
        keep_fetched_source: false` is only defensible because it exists.

        It clears more than the normal path because more is stale. `bytes` goes
        to 0 — the row is about to have no file, and leaving the old size there
        bills the user for something reclamation just deleted. `source_freed_at`
        clears because the statement it makes ("the bytes are gone, and that is
        final") stops being true the moment a re-fetch is queued.
        `transcript_source` clears because the next download decides its own,
        and an attribution left over from the last one labels a transcript that
        does not exist yet.

        The caption blob and the progress pair are cleared **defensively, not
        because a live row carries them**: `complete_fetch` zeroes the progress
        columns and `complete_job` clears `fetched_transcript`, so a row that
        got far enough to be reclaimed has all three empty already. They are in
        the statement so that "requeued to fetch" means the same thing from any
        state, rather than depending on which upstream write happened to run.

        `bytes_` corrects the recorded size when the caller has re-read it
        from disk; omitted, the stored value stands. It is part of this
        statement rather than a separate one because a re-run reads its source
        size from this row — a requeue that left a wrong size in place would
        stamp it onto the new points and outlive the fix.

        `attempts` resets to 0 deliberately. Requeue means "the reason this
        failed is fixed, try it properly" — carrying the old count forward
        would let a row that already burned its attempts fail out on the first
        pass and look like the fix didn't work.

        Any status may be requeued, including 'processing'. Taking a row back
        from a running worker invalidates its lease, so that worker's eventual
        post is refused rather than landing on a row it no longer owns — which
        is what the lease guard in `complete_job` is for.

        `collection` and `chunks` are cleared because the caller is expected to
        have removed the Qdrant points first. A row that still claimed them
        after they were deleted would read as ingested with nothing behind it.
        """
        return await asyncio.to_thread(
            self._requeue_job_sync, file_id, bytes_, refetch,
        )

    #: Cleared by every requeue, whichever queue the row goes back to.
    #:
    #: `completed_at` is in here for a reason worth keeping: left behind, a
    #: requeued video would carry the *old* completion time, so its retention
    #: window would be measured from a run whose results have just been thrown
    #: away — and a re-run of an old video would be eligible for reclamation
    #: the moment it finished.
    _REQUEUE_CLEARS = (
        "lease_id = '', leased_at = '', attempts = 0, failure_reason = '', "
        "collection = '', chunks = 0, duration_s = 0, summary = '', "
        "completed_at = ''"
    )

    def _requeue_job_sync(
        self, file_id: str, bytes_: int | None, refetch: bool,
    ) -> bool:
        with self._lock:
            if refetch:
                cur = self._conn.execute(
                    f"UPDATE uploads SET status = 'fetch_pending', "  # noqa: S608
                    f"       {self._REQUEUE_CLEARS}, "
                    "       source_freed_at = '', bytes = 0, "
                    "       fetched_transcript = '', transcript_source = '', "
                    "       fetch_downloaded_bytes = 0, fetch_total_bytes = 0 "
                    "WHERE file_id = ? AND source_url != ''",
                    (file_id,),
                )
                return cur.rowcount > 0
            cur = self._conn.execute(
                f"UPDATE uploads SET status = 'pending', "  # noqa: S608
                f"       {self._REQUEUE_CLEARS}, bytes = COALESCE(?, bytes) "
                "WHERE file_id = ?",
                (bytes_, file_id),
            )
            return cur.rowcount > 0

    # ── Source reclamation (Phase 38) ─────────────────────────────────

    async def reclaimable_sources(
        self, *, completed_before: str, limit: int = 50,
        uploaded: bool = True, fetched: bool = True,
    ) -> list[dict]:
        """Processed videos whose uploaded bytes are past the retention window.

        `uploaded` and `fetched` select which *kind of source* is eligible,
        because the two are not the same decision. Deleting an uploaded video's
        bytes destroys the only copy. Deleting a fetched one's leaves the row
        holding the URL it came from, and `requeue_job(refetch=True)` can put it
        back — so phase 38's argument against reclamation ("nothing can read the
        bytes back") is simply not an argument about these rows. Both default
        True so a caller that has already decided cannot silently widen; the
        route passes what config says.

        Four predicates, each load-bearing:

          - `status = 'ready'` — a pending, processing or failed row still has
            work that reads the source. Deleting under a running worker turns a
            job into an unrecoverable failure.
          - `kind = 'video'` — the source of a text upload is what the chunks
            were extracted from and there is no worker to re-run; only video
            has artifacts complete enough to stand without it.
          - `completed_at` non-empty and older than the window — a row restored
            by `reconcile_with_qdrant` into a fresh sqlite has no completion
            time and is never eligible. Withholding it is right: nothing knows
            when it finished, and the conservative answer to "may I delete this
            irreversibly?" is no.
          - `source_freed_at = ''` — do it once. The unlink is idempotent, but
            re-listing a reclaimed row would keep it in every sweep forever.

        `limit` caps one pass so a backlog is drained over several polls rather
        than in one blocking burst of unlinks on the claim path.
        """
        if not uploaded and not fetched:
            # Nothing is eligible, so ask nothing. The route short-circuits on
            # this too; belt and braces, because the failure mode of getting it
            # wrong here is an irreversible delete.
            return []
        return await asyncio.to_thread(
            self._reclaimable_sync, completed_before, limit, uploaded, fetched,
        )

    def _reclaimable_sync(
        self, completed_before: str, limit: int, uploaded: bool, fetched: bool,
    ) -> list[dict]:
        # A row is fetched if it kept the URL it came from. Only phase 41's
        # `record_url_upload` writes that column, and nothing clears it, so it
        # survives the ingest that makes the row eligible in the first place.
        source_clause = ""
        if not fetched:
            source_clause = "  AND source_url = '' "
        elif not uploaded:
            source_clause = "  AND source_url != '' "
        with self._lock:
            cur = self._conn.execute(
                "SELECT file_id, user, filename, bytes, source_url FROM uploads "  # noqa: S608
                "WHERE status = 'ready' AND kind = 'video' "
                "  AND source_freed_at = '' "
                "  AND completed_at != '' AND completed_at < ? "
                f"{source_clause}"
                "ORDER BY completed_at LIMIT ?",
                (completed_before, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    async def mark_source_freed(self, file_id: str, *, freed_at: str) -> bool:
        """Record that the uploaded bytes are gone. False if already recorded.

        The `source_freed_at = ''` predicate makes this the transition, not the
        assignment — two sweeps racing the same row cannot both count it as
        freed, and the caller can trust the return value as "I am the one who
        reclaimed this" rather than merely "the row exists".
        """
        return await asyncio.to_thread(self._mark_freed_sync, file_id, freed_at)

    def _mark_freed_sync(self, file_id: str, freed_at: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE uploads SET source_freed_at = ? "
                "WHERE file_id = ? AND source_freed_at = ''",
                (freed_at, file_id),
            )
            return cur.rowcount > 0

    async def sweep_expired_leases(
        self, *, expired_before: str, max_attempts: int, stage: str = "processing",
    ) -> dict[str, int]:
        """Return timed-out jobs to the queue, or fail them once attempts run out.

        A worker crash leaves `status='processing'` forever otherwise. Same
        spirit as the boot-time `reconcile_with_qdrant`: the recovery is a
        sweep, not a human noticing.

        The attempts cap is what makes this terminate. A video that crashes the
        worker every time would otherwise cycle forever, taking the worker down
        with it on every pass.

        `stage` picks which leased status to sweep — see `_LEASE_STAGES`. Each
        stage's claim route sweeps its own, so recovery stays attached to the
        heartbeat of the subsystem that needs it and there is still no
        background task to supervise.
        """
        return await asyncio.to_thread(
            self._sweep_leases_sync, expired_before, max_attempts, stage,
        )

    def _sweep_leases_sync(
        self, expired_before: str, max_attempts: int, stage: str,
    ) -> dict[str, int]:
        try:
            requeue_to, actor = _LEASE_STAGES[stage]
        except KeyError:
            raise ValueError(
                f"unknown lease stage {stage!r}; expected one of "
                f"{sorted(_LEASE_STAGES)}",
            ) from None
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id, attempts FROM uploads "
                "WHERE status = ? AND leased_at < ?",
                (stage, expired_before),
            ).fetchall()
            requeued = failed = 0
            for row in rows:
                attempts = int(row["attempts"])
                if attempts >= max_attempts:
                    self._conn.execute(
                        "UPDATE uploads SET status = 'failed', lease_id = '', "
                        "       leased_at = '', failure_reason = ?, "
                        "       quota_reserved_bytes = CASE WHEN status = 'fetching' "
                        "         THEN 0 ELSE quota_reserved_bytes END "
                        "WHERE file_id = ?",
                        (f"abandoned by the {actor} after {attempts} attempt(s)",
                         row["file_id"]),
                    )
                    failed += 1
                else:
                    self._conn.execute(
                        "UPDATE uploads SET status = ?, lease_id = '', "
                        "       leased_at = '' WHERE file_id = ?",
                        (requeue_to, row["file_id"]),
                    )
                    requeued += 1
            if requeued or failed:
                log.info(
                    "uploads_db: %s lease sweep requeued=%d failed=%d",
                    stage, requeued, failed,
                )
            return {"requeued": requeued, "failed": failed}

    # ── Chunked-upload sessions (Phase 32) ────────────────────────────

    async def open_session(
        self, *, upload_id: str, user: str, filename: str,
        total_bytes: int, part_size: int, parts_total: int, now: str,
    ) -> None:
        await asyncio.to_thread(
            self._open_session_sync, upload_id, user, filename,
            total_bytes, part_size, parts_total, now,
        )

    def _open_session_sync(
        self, upload_id: str, user: str, filename: str,
        total_bytes: int, part_size: int, parts_total: int, now: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO upload_sessions "
                "(upload_id, user, filename, total_bytes, part_size, "
                " parts_total, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (upload_id, user, filename, total_bytes, part_size,
                 parts_total, now, now),
            )

    async def get_session(self, upload_id: str, *, user: str) -> dict | None:
        """Fetch a session, double-keyed on user like every other read here.

        A leaked upload_id from another user must not resolve — the parts
        directory is derived from this row, so a permissive lookup would
        let one user append to another's assembly.
        """
        return await asyncio.to_thread(self._get_session_sync, upload_id, user)

    def _get_session_sync(self, upload_id: str, user: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM upload_sessions WHERE upload_id = ? AND user = ?",
                (upload_id, user),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    async def record_part(
        self, *, upload_id: str, part_no: int, bytes_: int, now: str,
    ) -> None:
        await asyncio.to_thread(self._record_part_sync, upload_id, part_no, bytes_, now)

    def _record_part_sync(
        self, upload_id: str, part_no: int, bytes_: int, now: str,
    ) -> None:
        with self._lock:
            # REPLACE, not INSERT: a client retrying a part it already sent
            # must not inflate the received-bytes total.
            self._conn.execute(
                "INSERT OR REPLACE INTO upload_parts (upload_id, part_no, bytes) "
                "VALUES (?, ?, ?)",
                (upload_id, part_no, bytes_),
            )
            self._conn.execute(
                "UPDATE upload_sessions SET updated_at = ? WHERE upload_id = ?",
                (now, upload_id),
            )

    async def record_reserved_part(
        self,
        *,
        upload_id: str,
        user: str,
        part_no: int,
        bytes_: int,
        now: str,
    ) -> None:
        await asyncio.to_thread(
            self._record_reserved_part_sync,
            upload_id,
            user,
            part_no,
            bytes_,
            now,
        )

    def _record_reserved_part_sync(
        self,
        upload_id: str,
        user: str,
        part_no: int,
        bytes_: int,
        now: str,
    ) -> None:
        with self._lock, self._transaction_locked():
            session = self._conn.execute(
                "SELECT user, total_bytes, parts_total FROM upload_sessions "
                "WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if session is None or session["user"] != user:
                raise ValueError("upload session is missing or not owned")
            if not 0 <= part_no < int(session["parts_total"]):
                raise ValueError("part number is outside the session")
            previous = self._conn.execute(
                "SELECT bytes FROM upload_parts "
                "WHERE upload_id = ? AND part_no = ?",
                (upload_id, part_no),
            ).fetchone()
            received = int(self._conn.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS total FROM upload_parts "
                "WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()["total"])
            previous_bytes = int(previous["bytes"]) if previous is not None else 0
            if received - previous_bytes + bytes_ > int(session["total_bytes"]):
                raise ValueError("chunk parts exceed the declared upload size")
            self._conn.execute(
                "INSERT INTO upload_parts (upload_id, part_no, bytes) "
                "VALUES (?, ?, ?) ON CONFLICT(upload_id, part_no) DO UPDATE SET "
                "bytes = excluded.bytes",
                (upload_id, part_no, bytes_),
            )
            self._conn.execute(
                "UPDATE upload_sessions SET updated_at = ? WHERE upload_id = ?",
                (now, upload_id),
            )

    async def session_progress(self, upload_id: str) -> tuple[int, int]:
        """Return (parts_received, bytes_received) for a session."""
        return await asyncio.to_thread(self._session_progress_sync, upload_id)

    def _session_progress_sync(self, upload_id: str) -> tuple[int, int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(bytes), 0) AS total "
                "FROM upload_parts WHERE upload_id = ?",
                (upload_id,),
            )
            row = cur.fetchone()
            return int(row["n"]), int(row["total"])

    async def received_part_numbers(self, upload_id: str) -> set[int]:
        """Which parts have landed — lets a resumed client skip what it sent."""
        return await asyncio.to_thread(self._received_parts_sync, upload_id)

    def _received_parts_sync(self, upload_id: str) -> set[int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT part_no FROM upload_parts WHERE upload_id = ?", (upload_id,),
            )
            return {int(r["part_no"]) for r in cur.fetchall()}

    async def close_session(self, upload_id: str) -> None:
        await asyncio.to_thread(self._close_session_sync, upload_id)

    def _close_session_sync(self, upload_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM upload_parts WHERE upload_id = ?", (upload_id,),
            )
            self._conn.execute(
                "DELETE FROM upload_sessions WHERE upload_id = ?", (upload_id,),
            )

    async def stale_sessions(self, *, older_than: str) -> list[dict]:
        """Sessions untouched since `older_than` (ISO stamp), for the TTL sweep.

        A closed browser tab leaves parts on disk forever otherwise; this is
        what the sweep enumerates before unlinking them.
        """
        return await asyncio.to_thread(self._stale_sessions_sync, older_than)

    def _stale_sessions_sync(self, older_than: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM upload_sessions WHERE updated_at < ?", (older_than,),
            )
            return [dict(r) for r in cur.fetchall()]


async def reconcile_with_qdrant(db: UploadsDB, qdrant) -> dict[str, int]:
    """Make sqlite agree with Qdrant.

    Strategy:
      1. For every user collection that exists in Qdrant, scroll it,
         build the same `(file_id → row)` map as `qdrant.list_user_files`,
         and INSERT OR REPLACE every row. Brings sqlite up to whatever
         is actually in vector storage right now.
      2. For every user already in sqlite, delete sqlite rows whose
         file_id no longer appears in either of that user's Qdrant
         collections. Catches files that were deleted in Qdrant out of
         band (manual purge, migration, etc.). Only rows that assert
         Qdrant content are eligible — `prunable_file_ids` says which.

    Returns a small stats dict for the boot log.

    PRECONDITION — call before serving traffic.

    Step 2 reads `db.all_users()` and then reads the matching live ids
    from Qdrant in two separate awaits. A fresh upload that lands
    between those two reads would have its sqlite row recorded but its
    Qdrant points not yet visible in the scroll, so step 2 would
    incorrectly prune that user's just-uploaded sqlite row.

    `main.py:lifespan` calls this during startup before uvicorn accepts
    requests, so the race window is empty in production. If you ever
    add a code path that calls this concurrently with the upload route,
    add per-user locking around steps 1 and 2 first.
    """
    from audrey.kb.user_store import (
        USER_IMAGE_PREFIX,
        USER_TEXT_PREFIX,
        user_image_collection,
        user_text_collection,
    )

    # A tombstone outranks Qdrant. Without this set, a crash after the user
    # requested deletion but before Qdrant acknowledged it would let step 1
    # recreate the visible row from the very points still awaiting cleanup.
    deletion_keys = await db.file_deletion_keys()

    # 1. Pull every kb_user_* collection from qdrant and backfill sqlite.
    all_collections = await qdrant.list_collections()
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
                if (raw_user, str(f["file_id"])) in deletion_keys:
                    continue
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
    #    Scoped to rows that claim Qdrant content — see `prunable_file_ids`.
    #    Without that scope every boot deletes the whole video queue, since
    #    a pending video has no points by definition.
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
        sqlite_ids = await db.prunable_file_ids(user)
        for fid in sqlite_ids - live_ids:
            if await db.delete_upload(fid, user=user):
                pruned += 1

    backfilled = sum(len(v) for v in user_to_collections.values())
    log.info(
        "uploads_db: reconcile complete (backfilled=%d collections, pruned=%d ghost rows)",
        backfilled, pruned,
    )
    return {"backfilled_collections": backfilled, "pruned_rows": pruned}


async def _scroll_user_rows(qdrant, collection: str) -> dict[str, list[dict]]:
    """Scroll a single collection and group `list_user_files`-shaped rows by raw user.

    Reuses the same payload-shape contract as QdrantKB._list_user_files_sync,
    but doesn't pre-filter by user — we don't know the user yet.

    ## `kind` is derived, not copied, and that is a bug fix

    A point's `kind` says what that POINT is, not what the FILE is. Every text
    point carries `kind: "text"` (`qdrant.py`'s text-point builder sets it, and
    the value is baked into the deterministic `point_id(source, kind, idx)`, so
    it cannot be changed without invalidating every stored id). A video's
    transcript, frame descriptions and summary are all ingested through that
    same text path.

    So copying it through said `kind = "text"` for a video, and because
    `record_upload`'s ON CONFLICT updates `kind`, **every boot silently
    rewrote a processed video's row to `kind='text'`**. Observed on the
    deployed box 2026-08-05: a `ready` row with a `summary` and a
    `completed_at` — fields only a video ever gets — sitting under
    `kind='text'`.

    Nothing crashed and no data was lost; `summary` and `completed_at` are in
    the second ON CONFLICT group and survived. What broke is everything that
    trusts `kind`: `list_my_files` told the model a video was a text file, and
    `reclaimable_sources` filters on `kind = 'video'`.

    **`artifact` is the reliable signal.** Only the three video stages set it
    (`transcript`, `visual`, `summary`); an ordinary text or image upload never
    does. Deriving from it repairs existing rows on the next boot with no
    migration, because reconcile rewrites `kind` anyway — the same mechanism
    that broke them fixes them once it is computing the right answer.
    """
    by_user_file: dict[tuple[str, str], dict] = {}
    for _point_id, payload in await qdrant.scroll_collection(collection):
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
        # Checked per point rather than only on the first: a file's points are
        # not ordered, so the one that created the row above may be any of
        # them — and only some of a video's points are artifacts if the
        # summary stage ran and the visual pass did not.
        if payload.get("artifact"):
            row["kind"] = "video"
    grouped: dict[str, list[dict]] = {}
    for (user, _fid), row in by_user_file.items():
        grouped.setdefault(user, []).append(row)
    return grouped


__all__ = [
    "QuotaDecision",
    "QuotaUsage",
    "UploadsDB",
    "UrlReservationRecovery",
    "reconcile_with_qdrant",
]
