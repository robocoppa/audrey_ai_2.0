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
  uploaded_at TEXT NOT NULL,
  -- 'ready'      — ingested, searchable now.
  -- 'pending'    — bytes stored, nothing extracted yet, waiting for a worker.
  --                Video lands here on upload.
  -- 'processing' — leased by a media worker (Phase 33). `lease_id` says which
  --                lease, `leased_at` says when it expires from.
  -- 'failed'     — gave up. `failure_reason` says why, and it is shown to the
  --                user: a row that sits at 'pending' forever with no
  --                explanation is the failure mode this state exists to avoid.
  status      TEXT NOT NULL DEFAULT 'ready',
  -- Empty unless status is 'processing'. Presented back on completion and
  -- checked, so a worker that stalled past its lease and woke up after the job
  -- was re-leased cannot overwrite the newer run's result.
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
  source_freed_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads(user);
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
)


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
        # default. All seven of `_UPLOADS_ADDED_COLUMNS` are absent from this
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
        with self._lock:
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
                # Every column `FileRow` declares has to be in this list. A
                # column added to the schema and the migration but not here
                # exists in the database and still raises on read, which
                # presents as a 500 on the file list rather than as anything
                # resembling a missing column. `TestListReturnsEveryFileRowField`
                # pins the two together.
                "SELECT file_id, filename, mime, bytes, kind, collection, "
                "       chunks, uploaded_at, status, failure_reason, duration_s, "
                "       summary, source_freed_at "
                "FROM uploads WHERE user = ? ORDER BY uploaded_at DESC",
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
    ) -> bool:
        """Flip a leased row to 'ready'. False if the lease no longer holds.

        The `lease_id` and `status` predicates are the guard against a late
        worker: one that stalled past its lease, had the job swept back and
        re-leased, then finally woke up and posted. Without them that stale
        result overwrites the newer one and the row looks perfectly healthy
        afterwards — no error, no log line, just a transcript that does not
        match its video.
        """
        return await asyncio.to_thread(
            self._complete_job_sync, file_id, lease_id, collection, chunks,
            duration_s, summary, completed_at,
        )

    def _complete_job_sync(
        self, file_id: str, lease_id: str, collection: str, chunks: int,
        duration_s: float, summary: str, completed_at: str,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE uploads SET status = 'ready', collection = ?, chunks = ?, "
                "       lease_id = '', leased_at = '', failure_reason = '', "
                "       duration_s = ?, summary = ?, completed_at = ? "
                "WHERE file_id = ? AND lease_id = ? AND status = 'processing'",
                (collection, chunks, duration_s, summary, completed_at,
                 file_id, lease_id),
            )
            return cur.rowcount > 0

    async def fail_job(self, *, file_id: str, lease_id: str, reason: str) -> bool:
        """Mark a leased row 'failed' with a reason. Same lease guard as completion."""
        return await asyncio.to_thread(self._fail_job_sync, file_id, lease_id, reason)

    def _fail_job_sync(self, file_id: str, lease_id: str, reason: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE uploads SET status = 'failed', failure_reason = ?, "
                "       lease_id = '', leased_at = '' "
                "WHERE file_id = ? AND lease_id = ? AND status = 'processing'",
                (reason[:500], file_id, lease_id),
            )
            return cur.rowcount > 0

    async def requeue_job(self, file_id: str, *, bytes_: int | None = None) -> bool:
        """Put a row back in the queue as if it had just been uploaded.

        False when there is no such row.

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
        return await asyncio.to_thread(self._requeue_job_sync, file_id, bytes_)

    def _requeue_job_sync(self, file_id: str, bytes_: int | None) -> bool:
        with self._lock:
            # `completed_at` clears with the rest of the previous run's output.
            # Left behind, a requeued video would carry the *old* completion
            # time, so its retention window would be measured from a run whose
            # results have just been thrown away — and a re-run of an old video
            # would be eligible for reclamation the moment it finished.
            cur = self._conn.execute(
                "UPDATE uploads SET status = 'pending', lease_id = '', "
                "       leased_at = '', attempts = 0, failure_reason = '', "
                "       collection = '', chunks = 0, duration_s = 0, summary = '', "
                "       completed_at = '', bytes = COALESCE(?, bytes) "
                "WHERE file_id = ?",
                (bytes_, file_id),
            )
            return cur.rowcount > 0

    # ── Source reclamation (Phase 38) ─────────────────────────────────

    async def reclaimable_sources(
        self, *, completed_before: str, limit: int = 50,
    ) -> list[dict]:
        """Processed videos whose uploaded bytes are past the retention window.

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
        return await asyncio.to_thread(
            self._reclaimable_sync, completed_before, limit,
        )

    def _reclaimable_sync(self, completed_before: str, limit: int) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT file_id, user, filename, bytes FROM uploads "
                "WHERE status = 'ready' AND kind = 'video' "
                "  AND source_freed_at = '' "
                "  AND completed_at != '' AND completed_at < ? "
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
        self, *, expired_before: str, max_attempts: int,
    ) -> dict[str, int]:
        """Return timed-out jobs to the queue, or fail them once attempts run out.

        A worker crash leaves `status='processing'` forever otherwise. Same
        spirit as the boot-time `reconcile_with_qdrant`: the recovery is a
        sweep, not a human noticing.

        The attempts cap is what makes this terminate. A video that crashes the
        worker every time would otherwise cycle forever, taking the worker down
        with it on every pass.
        """
        return await asyncio.to_thread(
            self._sweep_leases_sync, expired_before, max_attempts,
        )

    def _sweep_leases_sync(
        self, expired_before: str, max_attempts: int,
    ) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id, attempts FROM uploads "
                "WHERE status = 'processing' AND leased_at < ?",
                (expired_before,),
            ).fetchall()
            requeued = failed = 0
            for row in rows:
                attempts = int(row["attempts"])
                if attempts >= max_attempts:
                    self._conn.execute(
                        "UPDATE uploads SET status = 'failed', lease_id = '', "
                        "       leased_at = '', failure_reason = ? WHERE file_id = ?",
                        (f"abandoned by the media worker after {attempts} attempt(s)",
                         row["file_id"]),
                    )
                    failed += 1
                else:
                    self._conn.execute(
                        "UPDATE uploads SET status = 'pending', lease_id = '', "
                        "       leased_at = '' WHERE file_id = ?",
                        (row["file_id"],),
                    )
                    requeued += 1
            if requeued or failed:
                log.info(
                    "uploads_db: lease sweep requeued=%d failed=%d", requeued, failed,
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
    grouped: dict[str, list[dict]] = {}
    for (user, _fid), row in by_user_file.items():
        grouped.setdefault(user, []).append(row)
    return grouped


__all__ = ["UploadsDB", "reconcile_with_qdrant"]
