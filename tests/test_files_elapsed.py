"""Elapsed time on an in-flight upload (Phase 40, step 4).

A video ingest takes ~458 seconds and, until this step, said nothing at all
while it ran: the row went pending → processing → ready and the page only
noticed if you acted on it. "Processing…" for eight minutes is
indistinguishable from a job that has hung.

Three things are pinned here, and the first is the one with history:

  - **`leased_at` reaches the response.** Adding a column to `uploads` means
    the schema, the migration, the explicit SELECT in `_list_user_sync`, and
    `FileRow`. `summary` shipped to two of those in phase 37 and 500'd the
    whole file list. `source_freed_at` shipped to three in phase 38 and was
    silently `""` on every response until this step — a quieter version of the
    same bug, which is why the route now projects from `FileRow.model_fields`
    rather than a hand-written list.
  - **Elapsed is measured from the right timestamp**, which differs by status:
    `leased_at` once a worker holds it, `uploaded_at` while it is still
    queued. Those answer different questions and conflating them makes a
    backlog look like a stall.
  - **The server publishes its own clock.** Elapsed computed browser-side
    against a laptop that slept through a timezone renders "processing for 47
    hours", which reads as a broken queue rather than a wrong clock.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.auth import AuthedUser, require_user
from audrey.kb.uploads_db import UploadsDB
from audrey.routes.files import FileRow, _waiting_for_s
from audrey.routes.files import router as files_router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)
ME = "a@b.c"


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


def _build_app(db: UploadsDB, tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(files_router)
    # `require_user` proxies the bearer to OWUI, which is not reachable from a
    # hermetic test. These cases are about the projection and the clock, not
    # about auth — `test_files_service_list.py` owns that.
    app.dependency_overrides[require_user] = lambda: AuthedUser(
        email=ME, role="user", owui_id="u1")
    app.state.uploads_db = db
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=SECRET, owui_url="http://owui"),
        raw={"kb": {"upload_root": str(tmp_path / "uploads"),
                    "max_upload_mb": 50, "max_user_bytes": 10 * 1024**3,
                    "chunked": {"max_upload_mb": 2048, "part_size_mb": 8},
                    "video": {"lease_minutes": 30, "max_attempts": 3}}},
    )
    return app


async def _add(db: UploadsDB, file_id: str, *, user=ME, status="ready",
               uploaded_at="2026-08-01T00:00:00+00:00") -> None:
    await db.record_upload(
        file_id=file_id, user=user, filename=f"{file_id}.mp4", mime="video/mp4",
        bytes_=1024, kind="video", collection="", chunks=0,
        uploaded_at=uploaded_at, status=status,
    )


def _service_list(app: FastAPI, user: str):
    return TestClient(app).post(
        "/v1/files/list", json={"user": user},
        headers={"X-Audrey-Service-Token": SECRET},
    ).json()


class TestTheThreePlacesLesson:
    @pytest.mark.asyncio
    async def test_leased_at_survives_the_explicit_select(self, db: UploadsDB):
        """The half that 500'd when `summary` missed it."""
        await _add(db, "v1", status="pending")
        await db.claim_job(lease_id="L1", now="2026-08-01T10:00:00+00:00")

        row = (await db.list_user(ME))[0]

        assert row["leased_at"] == "2026-08-01T10:00:00+00:00"

    @pytest.mark.asyncio
    async def test_every_filerow_field_is_selected(self, db: UploadsDB):
        """Pins the read path against the model, in both directions from the
        route's point of view: anything `FileRow` declares must be a key the
        query returns, or constructing it raises."""
        await _add(db, "v1")

        row = (await db.list_user(ME))[0]

        assert not set(FileRow.model_fields) - set(row)

    @pytest.mark.asyncio
    async def test_source_freed_at_actually_reaches_the_response(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """The quiet version of the bug. It was in the schema, the migration,
        the SELECT and `FileRow` — and absent from the route's hand-written
        projection, so it defaulted to `""` on every response and the page's
        reclamation marker could never render. Nothing errored."""
        await _add(db, "v1")
        await db.mark_source_freed("v1", freed_at="2026-08-02T00:00:00+00:00")

        r = TestClient(_build_app(db, tmp_path)).get("/v1/files")

        assert r.status_code == 200
        assert r.json()["files"][0]["source_freed_at"] == "2026-08-02T00:00:00+00:00"


class TestWaitingFor:
    NOW = _dt.datetime(2026, 8, 1, 12, 0, 0, tzinfo=_dt.UTC)

    def test_a_finished_row_is_not_waiting(self):
        row = {"status": "ready", "leased_at": "", "uploaded_at": "2026-08-01T11:00:00+00:00"}
        assert _waiting_for_s(row, now=self.NOW) == 0.0

    def test_a_failed_row_is_not_waiting(self):
        row = {"status": "failed", "leased_at": "", "uploaded_at": "2026-08-01T11:00:00+00:00"}
        assert _waiting_for_s(row, now=self.NOW) == 0.0

    def test_processing_measures_from_the_lease(self):
        """Not from upload — a video that sat in a queue for an hour has not
        been *processing* for an hour, and reporting that would make a healthy
        worker look stuck."""
        row = {"status": "processing", "leased_at": "2026-08-01T11:55:00+00:00",
               "uploaded_at": "2026-08-01T11:00:00+00:00"}
        assert _waiting_for_s(row, now=self.NOW) == 300.0

    def test_pending_measures_from_upload(self):
        """It has no lease yet, and "queued 60m" is the honest number."""
        row = {"status": "pending", "leased_at": "", "uploaded_at": "2026-08-01T11:00:00+00:00"}
        assert _waiting_for_s(row, now=self.NOW) == 3600.0

    def test_a_future_timestamp_clamps_to_zero(self):
        """Impossible from one clock, but a clock step or a restored backup
        can produce it. "-3 minutes" reads as a broken queue and sends someone
        to debug the worker; 0 is the honest degradation."""
        row = {"status": "processing", "leased_at": "2026-08-01T13:00:00+00:00",
               "uploaded_at": "2026-08-01T11:00:00+00:00"}
        assert _waiting_for_s(row, now=self.NOW) == 0.0

    def test_an_unparseable_timestamp_is_zero_not_an_exception(self):
        """A bad stamp must not 500 the file list for every other row."""
        row = {"status": "processing", "leased_at": "not a date",
               "uploaded_at": "2026-08-01T11:00:00+00:00"}
        assert _waiting_for_s(row, now=self.NOW) == 0.0

    def test_a_naive_timestamp_is_read_as_utc(self):
        """Rows written before the timestamps carried offsets. Subtracting a
        naive datetime from an aware one raises, which would 500 the list."""
        row = {"status": "processing", "leased_at": "2026-08-01T11:55:00",
               "uploaded_at": "2026-08-01T11:00:00+00:00"}
        assert _waiting_for_s(row, now=self.NOW) == 300.0


class TestServiceListReportsWaiting:
    @pytest.mark.asyncio
    async def test_a_processing_file_reports_seconds_so_far(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """Computed server-side rather than shipping a timestamp: "how long
        has my video been processing?" is a chat question, and date arithmetic
        against an unstated now is what a model should not be doing."""
        await _add(db, "v1", status="pending")
        now = _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=4)
        await db.claim_job(lease_id="L1", now=now.isoformat(timespec="seconds"))

        row = _service_list(_build_app(db, tmp_path), ME)["files"][0]

        assert row["status"] == "processing"
        assert 230 <= row["waiting_for_s"] <= 250

    @pytest.mark.asyncio
    async def test_a_ready_file_reports_zero(self, db: UploadsDB, tmp_path: Path):
        await _add(db, "v1")

        row = _service_list(_build_app(db, tmp_path), ME)["files"][0]

        assert row["waiting_for_s"] == 0.0


class TestServerTime:
    @pytest.mark.asyncio
    async def test_the_list_publishes_the_servers_clock(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """Without it the page computes elapsed against the browser's clock,
        and a laptop minutes out renders a wrong duration that looks like a
        broken job queue rather than a wrong clock."""
        await _add(db, "v1")

        r = TestClient(_build_app(db, tmp_path)).get("/v1/files")

        assert r.status_code == 200
        parsed = _dt.datetime.fromisoformat(r.json()["server_time"])
        assert parsed.tzinfo is not None
        assert abs((_dt.datetime.now(_dt.UTC) - parsed).total_seconds()) < 60
