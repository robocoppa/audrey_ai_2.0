"""Tests for the media-worker job lifecycle (Phase 33).

The state machine is the whole deliverable of this phase, so the tests are the
deliverable's proof. The ones that matter most are not "does a happy job
complete" — they are the transitions that go wrong quietly:

  - a stale lease posting over a newer run's result,
  - a crashed worker's row sitting in `processing` forever,
  - a poison video cycling through the queue without end.

Each of those fails silently in production if it regresses. None of them are
observable from the outside afterwards, which is what makes them worth pinning.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.kb.uploads_db import _UPLOADS_ADDED_COLUMNS, UploadsDB
from audrey.routes.files import router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


async def _add(db: UploadsDB, file_id: str, *, user="a@b.c", status="pending",
               uploaded_at="2026-08-01T00:00:00+00:00") -> None:
    await db.record_upload(
        file_id=file_id, user=user, filename=f"{file_id}.mp4", mime="video/mp4",
        bytes_=1024, kind="video", collection="", chunks=0,
        uploaded_at=uploaded_at, status=status,
    )


class TestMigration:
    def test_every_added_column_is_also_in_the_fresh_schema(self, tmp_path: Path):
        """Two lists that must agree. A column in the migration but not the
        schema string means a fresh install silently lacks it."""
        fresh = UploadsDB(tmp_path / "fresh.sqlite")
        cols = {r["name"] for r in fresh._conn.execute("PRAGMA table_info(uploads)")}
        for name, _ddl in _UPLOADS_ADDED_COLUMNS:
            assert name in cols, f"{name} is migrated but missing from _SCHEMA"

    def test_a_preexisting_database_gains_the_job_columns(self, tmp_path: Path):
        """The deployed sqlite predates all of these, and CREATE TABLE IF NOT
        EXISTS is a no-op against a table that already exists."""
        path = tmp_path / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE uploads (file_id TEXT PRIMARY KEY, user TEXT NOT NULL, "
            "filename TEXT NOT NULL, mime TEXT NOT NULL, bytes INTEGER NOT NULL, "
            "kind TEXT NOT NULL, collection TEXT NOT NULL, chunks INTEGER NOT NULL, "
            "uploaded_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO uploads VALUES ('f1','a@b.c','a.txt','text/plain',5,"
            "'text','col',1,'2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        migrated = UploadsDB(path)
        cols = {r["name"] for r in migrated._conn.execute("PRAGMA table_info(uploads)")}
        assert {"status", "lease_id", "leased_at", "attempts", "failure_reason"} <= cols

    @pytest.mark.asyncio
    async def test_preexisting_rows_default_to_ready_and_unleased(self, tmp_path: Path):
        """Everything stored before this phase was ingested synchronously, so
        `ready` with no lease is true of it — the defaults have to make an
        untouched row correct."""
        path = tmp_path / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE uploads (file_id TEXT PRIMARY KEY, user TEXT NOT NULL, "
            "filename TEXT NOT NULL, mime TEXT NOT NULL, bytes INTEGER NOT NULL, "
            "kind TEXT NOT NULL, collection TEXT NOT NULL, chunks INTEGER NOT NULL, "
            "uploaded_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO uploads VALUES ('f1','a@b.c','a.txt','text/plain',5,"
            "'text','col',1,'2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        row = await UploadsDB(path).get_upload("f1")
        assert row["status"] == "ready"
        assert row["lease_id"] == ""
        assert row["attempts"] == 0


class TestClaim:
    @pytest.mark.asyncio
    async def test_an_empty_queue_claims_nothing(self, db: UploadsDB):
        assert await db.claim_job(lease_id="L1", now="2026-08-01T00:00:00+00:00") is None

    @pytest.mark.asyncio
    async def test_only_pending_rows_are_claimable(self, db: UploadsDB):
        await _add(db, "ready1", status="ready")
        await _add(db, "failed1", status="failed")
        assert await db.claim_job(lease_id="L1", now="t") is None

    @pytest.mark.asyncio
    async def test_the_oldest_pending_row_goes_first(self, db: UploadsDB):
        await _add(db, "newer", uploaded_at="2026-08-02T00:00:00+00:00")
        await _add(db, "older", uploaded_at="2026-08-01T00:00:00+00:00")
        claimed = await db.claim_job(lease_id="L1", now="t")
        assert claimed["file_id"] == "older"

    @pytest.mark.asyncio
    async def test_a_claimed_row_is_not_claimed_twice(self, db: UploadsDB):
        """The race this phase exists to not have. Two workers polling at once
        must not both get the same video."""
        await _add(db, "v1")
        first = await db.claim_job(lease_id="L1", now="t")
        second = await db.claim_job(lease_id="L2", now="t")
        assert first["file_id"] == "v1"
        assert second is None

    @pytest.mark.asyncio
    async def test_claiming_records_the_lease_and_counts_the_attempt(self, db: UploadsDB):
        await _add(db, "v1")
        await db.claim_job(lease_id="L1", now="2026-08-01T12:00:00+00:00")
        row = await db.get_upload("v1")
        assert row["status"] == "processing"
        assert row["lease_id"] == "L1"
        assert row["leased_at"] == "2026-08-01T12:00:00+00:00"
        assert row["attempts"] == 1

    @pytest.mark.asyncio
    async def test_the_returned_row_reflects_the_claim_not_the_pre_claim_state(
        self, db: UploadsDB,
    ):
        """The SELECT runs before the UPDATE, so the row handed back has to be
        corrected or the worker is told it holds a lease of ''."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        assert claimed["lease_id"] == "L1"
        assert claimed["status"] == "processing"
        assert claimed["attempts"] == 1


class TestComplete:
    @pytest.mark.asyncio
    async def test_a_held_lease_completes(self, db: UploadsDB):
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        assert await db.complete_job(
            file_id="v1", lease_id=claimed["lease_id"], collection="col", chunks=7,
        )
        row = await db.get_upload("v1")
        assert (row["status"], row["chunks"], row["collection"]) == ("ready", 7, "col")
        assert row["lease_id"] == ""

    @pytest.mark.asyncio
    async def test_a_stale_lease_cannot_complete(self, db: UploadsDB):
        """The failure with no symptom. A worker that stalled, had its job
        swept and re-leased, then woke up and posted would otherwise overwrite
        the newer run — and the row would look perfectly healthy after."""
        await _add(db, "v1")
        first = await db.claim_job(lease_id="L1", now="2026-08-01T00:00:00+00:00")
        await db.sweep_expired_leases(
            expired_before="2026-08-01T01:00:00+00:00", max_attempts=5,
        )
        second = await db.claim_job(lease_id="L2", now="2026-08-01T02:00:00+00:00")
        assert await db.complete_job(
            file_id="v1", lease_id=second["lease_id"], collection="new", chunks=2,
        )

        assert not await db.complete_job(
            file_id="v1", lease_id=first["lease_id"], collection="stale", chunks=99,
        )
        row = await db.get_upload("v1")
        assert (row["collection"], row["chunks"]) == ("new", 2)

    @pytest.mark.asyncio
    async def test_completing_an_unleased_row_is_refused(self, db: UploadsDB):
        await _add(db, "v1", status="ready")
        assert not await db.complete_job(
            file_id="v1", lease_id="", collection="col", chunks=1,
        )

    @pytest.mark.asyncio
    async def test_completing_clears_a_previous_failure_reason(self, db: UploadsDB):
        """A retry that succeeds must not leave the old explanation on a ready row."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.fail_job(file_id="v1", lease_id=claimed["lease_id"], reason="boom")
        await db.record_upload(
            file_id="v1", user="a@b.c", filename="v1.mp4", mime="video/mp4",
            bytes_=1024, kind="video", collection="", chunks=0,
            uploaded_at="2026-08-01T00:00:00+00:00", status="pending",
        )
        again = await db.claim_job(lease_id="L2", now="t")
        await db.complete_job(
            file_id="v1", lease_id=again["lease_id"], collection="col", chunks=3,
        )
        assert (await db.get_upload("v1"))["failure_reason"] == ""


class TestFail:
    @pytest.mark.asyncio
    async def test_a_failure_records_its_reason(self, db: UploadsDB):
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        assert await db.fail_job(
            file_id="v1", lease_id=claimed["lease_id"], reason="unreadable container",
        )
        row = await db.get_upload("v1")
        assert row["status"] == "failed"
        assert row["failure_reason"] == "unreadable container"

    @pytest.mark.asyncio
    async def test_a_stale_lease_cannot_fail_a_job(self, db: UploadsDB):
        await _add(db, "v1")
        first = await db.claim_job(lease_id="L1", now="2026-08-01T00:00:00+00:00")
        await db.sweep_expired_leases(
            expired_before="2026-08-01T01:00:00+00:00", max_attempts=5,
        )
        await db.claim_job(lease_id="L2", now="2026-08-01T02:00:00+00:00")
        assert not await db.fail_job(
            file_id="v1", lease_id=first["lease_id"], reason="late",
        )
        assert (await db.get_upload("v1"))["status"] == "processing"

    @pytest.mark.asyncio
    async def test_a_runaway_reason_is_truncated(self, db: UploadsDB):
        """A worker can put anything here, including a whole stack trace."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.fail_job(file_id="v1", lease_id=claimed["lease_id"], reason="x" * 5000)
        assert len((await db.get_upload("v1"))["failure_reason"]) == 500

    @pytest.mark.asyncio
    async def test_a_failed_row_is_not_reclaimed(self, db: UploadsDB):
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.fail_job(file_id="v1", lease_id=claimed["lease_id"], reason="no")
        assert await db.claim_job(lease_id="L2", now="t") is None


class TestSweep:
    @pytest.mark.asyncio
    async def test_an_expired_lease_returns_to_the_queue(self, db: UploadsDB):
        await _add(db, "v1")
        await db.claim_job(lease_id="L1", now="2026-08-01T00:00:00+00:00")
        stats = await db.sweep_expired_leases(
            expired_before="2026-08-01T01:00:00+00:00", max_attempts=5,
        )
        assert stats == {"requeued": 1, "failed": 0}
        row = await db.get_upload("v1")
        assert row["status"] == "pending"
        assert row["lease_id"] == ""

    @pytest.mark.asyncio
    async def test_a_live_lease_is_left_alone(self, db: UploadsDB):
        """Sweeping a worker that is still working steals its job mid-transcode."""
        await _add(db, "v1")
        await db.claim_job(lease_id="L1", now="2026-08-01T02:00:00+00:00")
        stats = await db.sweep_expired_leases(
            expired_before="2026-08-01T01:00:00+00:00", max_attempts=5,
        )
        assert stats == {"requeued": 0, "failed": 0}
        assert (await db.get_upload("v1"))["status"] == "processing"

    @pytest.mark.asyncio
    async def test_attempts_terminate_rather_than_cycling_forever(self, db: UploadsDB):
        """A video that crashes the worker every time must stop being handed
        back out, or it takes the worker down on every pass."""
        await _add(db, "v1")
        for n in range(3):
            claimed = await db.claim_job(lease_id=f"L{n}", now="2026-08-01T00:00:00+00:00")
            assert claimed is not None, f"should still be claimable on attempt {n + 1}"
            stats = await db.sweep_expired_leases(
                expired_before="2026-08-01T01:00:00+00:00", max_attempts=3,
            )
        assert stats == {"requeued": 0, "failed": 1}
        row = await db.get_upload("v1")
        assert row["status"] == "failed"
        assert "3 attempt" in row["failure_reason"]
        assert await db.claim_job(lease_id="L9", now="t") is None

    @pytest.mark.asyncio
    async def test_the_sweep_reports_both_outcomes_at_once(self, db: UploadsDB):
        await _add(db, "young", uploaded_at="2026-08-01T00:00:00+00:00")
        await _add(db, "old", uploaded_at="2026-08-01T00:00:01+00:00")
        await db.claim_job(lease_id="L1", now="2026-08-01T00:00:00+00:00")
        await db.claim_job(lease_id="L2", now="2026-08-01T00:00:00+00:00")
        # Push one to the cap so the sweep has to do both things in one pass.
        db._conn.execute("UPDATE uploads SET attempts = 9 WHERE file_id = 'old'")
        stats = await db.sweep_expired_leases(
            expired_before="2026-08-01T01:00:00+00:00", max_attempts=3,
        )
        assert stats == {"requeued": 1, "failed": 1}

    @pytest.mark.asyncio
    async def test_a_pending_row_is_not_swept(self, db: UploadsDB):
        """Only leases expire. An unclaimed row has no lease to time out."""
        await _add(db, "v1")
        stats = await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=1,
        )
        assert stats == {"requeued": 0, "failed": 0}
        assert (await db.get_upload("v1"))["status"] == "pending"


def _build_app(db: UploadsDB, tmp_path: Path, *, service_token: str = SECRET) -> FastAPI:
    """Minimal app carrying just what the job routes reach for."""
    app = FastAPI()
    app.include_router(router)
    app.state.uploads_db = db
    app.state.qdrant = object()
    app.state.text_embedder = object()
    app.state.image_embedder = object()
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=service_token, owui_url="http://owui"),
        raw={"kb": {"upload_root": str(tmp_path / "uploads"),
                    "video": {"lease_minutes": 30, "max_attempts": 3}}},
    )
    return app


class TestRouteAuth:
    """These routes hand out filesystem paths and write into an arbitrary
    user's collection. A user JWT must not be enough to reach them."""

    @pytest.mark.asyncio
    async def test_claim_without_a_token_is_401(self, db: UploadsDB, tmp_path: Path):
        await _add(db, "v1")
        r = TestClient(_build_app(db, tmp_path)).post("/v1/files/jobs/claim")
        assert r.status_code == 401
        # And the row was not leased on the way to being refused.
        assert (await db.get_upload("v1"))["status"] == "pending"

    @pytest.mark.asyncio
    async def test_claim_with_a_wrong_token_is_401(self, db: UploadsDB, tmp_path: Path):
        r = TestClient(_build_app(db, tmp_path)).post(
            "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": "nope"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_a_blank_configured_secret_never_authenticates(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """Fail closed — an unset KB_SERVICE_TOKEN must not open the routes."""
        app = _build_app(db, tmp_path, service_token="")
        r = TestClient(app).post(
            "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": ""},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_result_and_failed_are_also_service_only(
        self, db: UploadsDB, tmp_path: Path,
    ):
        client = TestClient(_build_app(db, tmp_path))
        assert client.post(
            "/v1/files/v1/ingest-result", json={"lease_id": "L1"},
        ).status_code == 401
        assert client.post(
            "/v1/files/v1/ingest-failed", json={"lease_id": "L1", "reason": "x"},
        ).status_code == 401


class TestRouteBehaviour:
    @pytest.mark.asyncio
    async def test_an_empty_queue_is_204_not_an_error(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """The steady state. A worker polling an idle Audrey must not see failures."""
        r = TestClient(_build_app(db, tmp_path)).post(
            "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": SECRET},
        )
        assert r.status_code == 204

    @pytest.mark.asyncio
    async def test_a_claim_returns_the_path_the_worker_needs(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _add(db, "v1", user="a@b.c")
        r = TestClient(_build_app(db, tmp_path)).post(
            "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": SECRET},
        )
        assert r.status_code == 200
        job = r.json()
        assert job["file_id"] == "v1"
        assert job["user"] == "a@b.c"
        assert job["lease_id"]
        # The path is derived from the sanitized user and the file_id, never
        # from the client-supplied filename.
        assert job["path"].endswith("v1.mp4")
        assert "a@b.c" not in job["path"]

    @pytest.mark.asyncio
    async def test_a_stale_lease_is_refused_with_409(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _add(db, "v1")
        client = TestClient(_build_app(db, tmp_path))
        first = client.post(
            "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": SECRET},
        ).json()
        await db.sweep_expired_leases(expired_before="2099-01-01", max_attempts=9)
        client.post("/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": SECRET})

        r = client.post(
            "/v1/files/v1/ingest-result",
            headers={"X-Audrey-Service-Token": SECRET},
            json={"lease_id": first["lease_id"], "segments": [
                {"t_start": 0.0, "t_end": 1.0, "text": "stale"},
            ]},
        )
        assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_an_unknown_file_id_is_404(self, db: UploadsDB, tmp_path: Path):
        r = TestClient(_build_app(db, tmp_path)).post(
            "/v1/files/nope/ingest-result",
            headers={"X-Audrey-Service-Token": SECRET},
            json={"lease_id": "L1"},
        )
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_a_failure_reaches_the_row_through_the_route(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _add(db, "v1")
        client = TestClient(_build_app(db, tmp_path))
        job = client.post(
            "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": SECRET},
        ).json()
        r = client.post(
            "/v1/files/v1/ingest-failed",
            headers={"X-Audrey-Service-Token": SECRET},
            json={"lease_id": job["lease_id"], "reason": "unreadable container"},
        )
        assert r.status_code == 200
        row = await db.get_upload("v1")
        assert (row["status"], row["failure_reason"]) == ("failed", "unreadable container")

    @pytest.mark.asyncio
    async def test_an_empty_transcript_completes_without_ingesting(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """A silent video is a normal case, not a failure. No segments means no
        text path is touched at all — which is also why this test can run with
        a stub embedder that would explode if it were."""
        await _add(db, "v1")
        client = TestClient(_build_app(db, tmp_path))
        job = client.post(
            "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": SECRET},
        ).json()
        r = client.post(
            "/v1/files/v1/ingest-result",
            headers={"X-Audrey-Service-Token": SECRET},
            json={"lease_id": job["lease_id"], "duration_s": 12.0, "segments": []},
        )
        assert r.status_code == 200
        assert r.json()["chunks"] == 0
        assert (await db.get_upload("v1"))["status"] == "ready"


class TestListing:
    @pytest.mark.asyncio
    async def test_the_failure_reason_reaches_the_file_list(self, db: UploadsDB):
        """A row that stops moving without saying why is the failure mode the
        whole `failed` state exists to prevent."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.fail_job(file_id="v1", lease_id=claimed["lease_id"], reason="no audio")
        row = (await db.list_user("a@b.c"))[0]
        assert row["status"] == "failed"
        assert row["failure_reason"] == "no audio"

    @pytest.mark.asyncio
    async def test_get_upload_is_not_keyed_on_user(self, db: UploadsDB):
        """Unlike every other read here — the worker has no user of its own and
        resolving the owner is the whole reason it asks."""
        await _add(db, "v1", user="owner@x.y")
        assert (await db.get_upload("v1"))["user"] == "owner@x.y"

    @pytest.mark.asyncio
    async def test_an_unknown_file_id_resolves_to_nothing(self, db: UploadsDB):
        assert await db.get_upload("nope") is None


class _RecordingQdrant:
    """Records delete calls; optionally fails, to test the ordering guarantee."""

    def __init__(self, *, fail: bool = False):
        self.deletes: list[tuple[str, str, str]] = []  # (file_id, user, collection)
        self._fail = fail

    async def delete_by_file_id(self, file_id: str, *, user: str, collection: str) -> None:
        if self._fail:
            raise RuntimeError("qdrant is down")
        self.deletes.append((file_id, user, collection))


def _app_with_qdrant(db: UploadsDB, tmp_path: Path, qdrant) -> FastAPI:
    app = _build_app(db, tmp_path)
    app.state.qdrant = qdrant
    return app


class TestRequeue:
    """Requeue is the only route back into the queue. Without it a video that
    failed for a since-fixed reason can only be deleted and re-uploaded."""

    @pytest.mark.asyncio
    async def test_a_ready_row_goes_back_to_pending_and_forgets_its_ingest(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.complete_job(
            file_id="v1", lease_id=claimed["lease_id"], collection="kb_user_text_a_b_c",
            chunks=7,
        )

        r = TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 200
        row = await db.get_upload("v1")
        assert row["status"] == "pending"
        assert row["chunks"] == 0
        assert row["collection"] == ""
        assert row["lease_id"] == ""

    @pytest.mark.asyncio
    async def test_attempts_reset_to_zero(self, db: UploadsDB, tmp_path: Path):
        """A row that burned its attempts would otherwise fail out on the first
        pass after a fix, and look like the fix didn't work."""
        await _add(db, "v1")
        for _ in range(3):
            claimed = await db.claim_job(lease_id="L", now="t")
            await db.fail_job(file_id="v1", lease_id=claimed["lease_id"], reason="boom")
            await db.requeue_job("v1")
        assert (await db.get_upload("v1"))["attempts"] == 0

        claimed = await db.claim_job(lease_id="L9", now="t")
        assert claimed["attempts"] == 1

    @pytest.mark.asyncio
    async def test_the_failure_reason_is_cleared(self, db: UploadsDB, tmp_path: Path):
        """A queued row showing last run's error would misreport its own state."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.fail_job(file_id="v1", lease_id=claimed["lease_id"], reason="no audio")

        TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        row = await db.get_upload("v1")
        assert row["failure_reason"] == ""
        assert row["status"] == "pending"

    @pytest.mark.asyncio
    async def test_the_old_points_are_deleted_from_the_users_collection(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """A re-run that produces no transcript never calls the ingest path, so
        nothing else would ever clear these — and reconcile exempts `chunks = 0`
        rows, so its ghost sweep won't either."""
        await _add(db, "v1", user="owner@x.y")
        qdrant = _RecordingQdrant()

        TestClient(_app_with_qdrant(db, tmp_path, qdrant)).post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert qdrant.deletes == [("v1", "owner@x.y", "kb_user_text_owner_x_y")]

    @pytest.mark.asyncio
    async def test_a_failed_qdrant_delete_leaves_the_row_untouched(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """The ordering guarantee. Resetting the row first would leave points
        searchable under a row claiming none, with no sweep that collects them."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.complete_job(
            file_id="v1", lease_id=claimed["lease_id"], collection="c", chunks=7,
        )

        client = TestClient(
            _app_with_qdrant(db, tmp_path, _RecordingQdrant(fail=True)),
            raise_server_exceptions=False,
        )
        r = client.post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 500
        row = await db.get_upload("v1")
        assert row["status"] == "ready"
        assert row["chunks"] == 7

    @pytest.mark.asyncio
    async def test_requeueing_a_processing_row_invalidates_the_running_lease(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """Taking a job back from a live worker is allowed — with `force`, see
        TestRequeueForceGuard — and its late post must then be refused rather
        than landing on a row it no longer owns."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")

        TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/v1/requeue?force=true",
            headers={"X-Audrey-Service-Token": SECRET},
        )

        assert (await db.get_upload("v1"))["status"] == "pending"
        assert not await db.complete_job(
            file_id="v1", lease_id=claimed["lease_id"], collection="c", chunks=3,
        )

    @pytest.mark.asyncio
    async def test_a_requeued_row_is_claimable_again(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """The point of the whole route."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")
        await db.complete_job(
            file_id="v1", lease_id=claimed["lease_id"], collection="c", chunks=2,
        )
        assert await db.claim_job(lease_id="L2", now="t") is None

        TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        again = await db.claim_job(lease_id="L3", now="t")
        assert again is not None
        assert again["file_id"] == "v1"

    @pytest.mark.asyncio
    async def test_an_unknown_file_id_is_404(self, db: UploadsDB, tmp_path: Path):
        r = TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/nope/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_requeue_is_service_only(self, db: UploadsDB, tmp_path: Path):
        """It discards a user's ingested chunks — a user JWT must not reach it."""
        await _add(db, "v1")
        client = TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant()))

        assert client.post("/v1/files/v1/requeue").status_code == 401
        assert client.post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": "nope"},
        ).status_code == 401
        assert client.post(
            "/v1/files/v1/requeue", headers={"Authorization": "Bearer user-jwt"},
        ).status_code == 401
        assert (await db.get_upload("v1"))["status"] == "pending"


class TestRequeueForceGuard:
    """Requeueing a live job silently discards its work.

    It is allowed — a genuinely stuck worker needs someone to take the job
    back — but it is almost never what you meant. On the run that prompted
    this guard it threw away 74 seconds of whisper: the worker finished,
    posted, got a 409 because its lease had been cleared underneath it, and
    started over. Nothing broke, nothing was corrupted, and nothing said so.
    """

    @pytest.mark.asyncio
    async def test_a_processing_row_is_refused_by_default(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _add(db, "v1")
        await db.claim_job(lease_id="L1", now="t")
        qdrant = _RecordingQdrant()

        r = TestClient(_app_with_qdrant(db, tmp_path, qdrant)).post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 409
        # Refused before Qdrant is touched, so a refused call changes nothing
        # at all — including for the worker still running.
        assert qdrant.deletes == []
        row = await db.get_upload("v1")
        assert row["status"] == "processing"
        assert row["lease_id"] == "L1"

    @pytest.mark.asyncio
    async def test_the_refusal_names_the_lease_it_would_break(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """'Conflict' alone sends the operator to the logs. The lease id and
        its start time are what tell them whether it is stuck or just slow."""
        await _add(db, "v1")
        await db.claim_job(lease_id="L1", now="2026-08-03T17:57:22+00:00")

        r = TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        detail = r.json()["detail"]
        assert "L1" in detail
        assert "2026-08-03T17:57:22+00:00" in detail
        assert "force=true" in detail

    @pytest.mark.asyncio
    async def test_force_takes_the_job_back(self, db: UploadsDB, tmp_path: Path):
        """The escape hatch has to still work — a worker that died without
        releasing its lease is exactly what it is for."""
        await _add(db, "v1")
        claimed = await db.claim_job(lease_id="L1", now="t")

        r = TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/v1/requeue?force=true",
            headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 200
        assert (await db.get_upload("v1"))["status"] == "pending"
        # And the displaced worker's late result is still refused.
        assert not await db.complete_job(
            file_id="v1", lease_id=claimed["lease_id"], collection="c", chunks=3,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["ready", "failed", "pending"])
    async def test_every_other_status_needs_no_force(
        self, db: UploadsDB, tmp_path: Path, status: str,
    ):
        """The guard is narrow on purpose. Requeueing a finished or failed row
        is the ordinary case and must not grow a flag."""
        await _add(db, "v1", status=status)

        r = TestClient(_app_with_qdrant(db, tmp_path, _RecordingQdrant())).post(
            "/v1/files/v1/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 200
        assert (await db.get_upload("v1"))["status"] == "pending"
