"""Tests for source reclamation (Phase 38).

`keep_source: false` was planned for phase 35 and deferred four phases,
because until phase 37 landed something downstream still read the bytes: 36
extracts frames from the source, 37 summarises what 36 produced, and 33's
requeue points a re-run at a path that has to exist.

What makes this worth heavy testing is that it is the only irreversible
operation in the video pipeline. Every other mistake in phases 32-39 costs a
re-run; this one costs the file. So the tests below are weighted toward the
cases where a source must NOT be deleted, and toward the two ways the
bookkeeping could lie about it afterwards:

  - deleting bytes without freeing quota (the user pays for nothing), and
  - freeing quota without deleting bytes (the disk fills silently).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.kb.uploads_db import UploadsDB
from audrey.routes.files import router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)
USER = "a@b.c"


def _iso(hours_ago: float) -> str:
    stamp = _dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=hours_ago)
    return stamp.isoformat(timespec="seconds")


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


async def _processed_video(
    db: UploadsDB,
    file_id: str = "vid",
    *,
    user: str = USER,
    completed_hours_ago: float = 48.0,
    kind: str = "video",
    bytes_: int = 301936597,
) -> None:
    """A video that finished ingest, exactly as `ingest_result` leaves it."""
    await db.record_upload(
        file_id=file_id, user=user, filename=f"{file_id}.mp4", mime="video/mp4",
        bytes_=bytes_, kind=kind, collection="", chunks=0,
        uploaded_at=_iso(completed_hours_ago + 1), status="pending",
    )
    await db.claim_job(lease_id=f"L-{file_id}", now=_iso(completed_hours_ago))
    assert await db.complete_job(
        file_id=file_id, lease_id=f"L-{file_id}", collection="kb_user_text_a_b_c",
        chunks=25, duration_s=565.0, summary="A retirement party.",
        completed_at=_iso(completed_hours_ago),
    )


class TestEligibility:
    """Which rows the sweep will even consider. Every exclusion here is a file
    that would otherwise be deleted while something still needed it."""

    async def test_a_video_past_the_window_is_eligible(self, db: UploadsDB):
        await _processed_video(db, completed_hours_ago=48)

        rows = await db.reclaimable_sources(completed_before=_iso(24))

        assert [r["file_id"] for r in rows] == ["vid"]

    async def test_a_video_inside_the_window_is_not(self, db: UploadsDB):
        """The window is the entire escape hatch for "that ingest came out
        wrong, run it again"."""
        await _processed_video(db, completed_hours_ago=2)

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []

    async def test_a_pending_video_is_never_eligible(self, db: UploadsDB):
        """Its bytes are the input to a job that has not run."""
        await db.record_upload(
            file_id="vid", user=USER, filename="vid.mp4", mime="video/mp4",
            bytes_=999, kind="video", collection="", chunks=0,
            uploaded_at=_iso(500), status="pending",
        )

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []

    async def test_a_processing_video_is_never_eligible(self, db: UploadsDB):
        """Deleting under a running worker turns a job into a failure that
        cannot succeed on any retry."""
        await db.record_upload(
            file_id="vid", user=USER, filename="vid.mp4", mime="video/mp4",
            bytes_=999, kind="video", collection="", chunks=0,
            uploaded_at=_iso(500), status="pending",
        )
        await db.claim_job(lease_id="L1", now=_iso(499))

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []

    async def test_a_failed_video_is_never_eligible(self, db: UploadsDB):
        """A failed row is the one most likely to be requeued once whatever
        broke is fixed. Reclaiming it removes the only way to retry."""
        await db.record_upload(
            file_id="vid", user=USER, filename="vid.mp4", mime="video/mp4",
            bytes_=999, kind="video", collection="", chunks=0,
            uploaded_at=_iso(500), status="pending",
        )
        await db.claim_job(lease_id="L1", now=_iso(499))
        await db.fail_job(file_id="vid", lease_id="L1", reason="ffmpeg said no")

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []

    async def test_a_text_upload_is_never_eligible(self, db: UploadsDB):
        """A document's chunks were extracted from its source and there is no
        worker to re-run. Only video has artifacts that stand without it."""
        await _processed_video(db, kind="text", completed_hours_ago=500)

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []

    async def test_a_row_with_no_completion_time_is_never_eligible(
        self, db: UploadsDB,
    ):
        """`reconcile_with_qdrant` can restore a row into a fresh sqlite from
        Qdrant payloads, which carry no completion time. Nothing then knows
        when the video finished, and the conservative answer to "may I delete
        this irreversibly?" is no."""
        await db.record_upload(
            file_id="vid", user=USER, filename="vid.mp4", mime="video/mp4",
            bytes_=999, kind="video", collection="col", chunks=25,
            uploaded_at=_iso(500), status="ready",
        )

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []

    async def test_an_already_freed_row_is_not_listed_again(self, db: UploadsDB):
        await _processed_video(db, completed_hours_ago=48)
        await db.mark_source_freed("vid", freed_at=_iso(1))

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []

    async def test_the_window_is_measured_from_completion_not_upload(
        self, db: UploadsDB,
    ):
        """A video that sat in a stalled queue for a week and finished ten
        minutes ago is inside its window, even though it was uploaded long
        before the cutoff. Measured from `uploaded_at` it would be eligible
        immediately — a retention window of zero for exactly the file whose
        processing most deserved a second look."""
        await db.record_upload(
            file_id="vid", user=USER, filename="vid.mp4", mime="video/mp4",
            bytes_=999, kind="video", collection="", chunks=0,
            uploaded_at=_iso(24 * 7), status="pending",
        )
        await db.claim_job(lease_id="L1", now=_iso(0.5))
        await db.complete_job(
            file_id="vid", lease_id="L1", collection="col", chunks=9,
            completed_at=_iso(0.2),
        )

        assert await db.reclaimable_sources(completed_before=_iso(24)) == []


class TestMarking:
    async def test_marking_is_a_transition_not_an_assignment(self, db: UploadsDB):
        """The second caller must get False. Two sweeps racing one row would
        otherwise both believe they reclaimed it."""
        await _processed_video(db, completed_hours_ago=48)

        assert await db.mark_source_freed("vid", freed_at=_iso(0))
        assert not await db.mark_source_freed("vid", freed_at=_iso(0))

    async def test_marking_an_absent_row_is_false(self, db: UploadsDB):
        assert not await db.mark_source_freed("nope", freed_at=_iso(0))


class TestQuota:
    """The point of the whole exercise. Bytes that no longer exist must stop
    counting, and must stay stopped."""

    async def test_a_reclaimed_video_stops_counting_against_the_quota(
        self, db: UploadsDB,
    ):
        await _processed_video(db, completed_hours_ago=48, bytes_=301936597)
        assert await db.user_total_bytes(USER) == 301936597

        await db.mark_source_freed("vid", freed_at=_iso(0))

        assert await db.user_total_bytes(USER) == 0

    async def test_other_files_still_count(self, db: UploadsDB):
        await _processed_video(db, "kept", completed_hours_ago=2, bytes_=1000)
        await _processed_video(db, "freed", completed_hours_ago=48, bytes_=9000)

        await db.mark_source_freed("freed", freed_at=_iso(0))

        assert await db.user_total_bytes(USER) == 1000

    async def test_the_recorded_size_is_left_alone(self, db: UploadsDB):
        """`bytes` is the honest record of what was uploaded and what the file
        list shows. Zeroing it instead of flagging the row would be simpler and
        would be undone at the next boot, because `reconcile_with_qdrant`
        refreshes `bytes` from a Qdrant payload that still carries the original
        size and always will."""
        await _processed_video(db, completed_hours_ago=48, bytes_=301936597)

        await db.mark_source_freed("vid", freed_at=_iso(0))

        assert (await db.get_upload("vid"))["bytes"] == 301936597

    async def test_freed_quota_survives_a_reconcile_pass(self, db: UploadsDB):
        """The standing gotcha this design exists to dodge: anything wrong in
        a payload becomes wrong in sqlite one restart later, quota included.
        `source_freed_at` lives in the group reconcile does not touch."""
        await _processed_video(db, completed_hours_ago=48, bytes_=301936597)
        await db.mark_source_freed("vid", freed_at=_iso(0))

        # What reconcile's backfill does with a Qdrant payload for this file.
        await db.record_upload(
            file_id="vid", user=USER, filename="vid.mp4", mime="video/mp4",
            bytes_=301936597, kind="video", collection="kb_user_text_a_b_c",
            chunks=25, uploaded_at=_iso(49),
        )

        assert await db.user_total_bytes(USER) == 0


# ─── The route-level sweep ─────────────────────────────────────────────

class _Qdrant:
    """Only what the requeue route reaches for."""

    def __init__(self) -> None:
        self.deletes: list[str] = []

    async def delete_by_file_id(self, file_id: str, *, user: str, collection: str) -> None:
        self.deletes.append(file_id)


def _client(db: UploadsDB, upload_root: Path, video_cfg: dict) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.uploads_db = db
    app.state.qdrant = _Qdrant()
    app.state.text_embedder = object()
    app.state.image_embedder = object()
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=SECRET, owui_url="http://owui"),
        raw={"kb": {"upload_root": str(upload_root), "video": video_cfg}},
    )
    return TestClient(app)


def _claim(client: TestClient):
    return client.post(
        "/v1/files/jobs/claim", headers={"X-Audrey-Service-Token": SECRET},
    )


class TestTheSweepOnClaim:
    """It rides the worker's existing 10s poll rather than a background task,
    the same way phase 33's lease sweep does."""

    async def test_a_claim_reclaims_an_eligible_source(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _processed_video(db, completed_hours_ago=48)
        root = tmp_path / "uploads"
        source = root / "a_b_c" / "vid.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"video bytes")

        client = _client(db, root, {"keep_source": False, "source_retention_hours": 24})
        _claim(client)

        assert not source.exists()
        assert (await db.get_upload("vid"))["source_freed_at"]

    async def test_keep_source_stops_it_entirely(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """The escape hatch. Set before deploying any change you might want to
        re-run old videos through."""
        await _processed_video(db, completed_hours_ago=48)
        root = tmp_path / "uploads"
        source = root / "a_b_c" / "vid.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"video bytes")

        client = _client(db, root, {"keep_source": True})
        _claim(client)

        assert source.exists()
        assert not (await db.get_upload("vid"))["source_freed_at"]

    async def test_the_sidecars_are_not_touched(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """`.frames.txt` and `.summary.txt` sit in the same directory under the
        same file_id, and Qdrant payloads point at them. The delete route globs
        `{file_id}.*` and would take all three; this must take exactly one."""
        await _processed_video(db, completed_hours_ago=48)
        root = tmp_path / "uploads" / "a_b_c"
        root.mkdir(parents=True)
        (root / "vid.mp4").write_bytes(b"video bytes")
        (root / "vid.frames.txt").write_text("[00:00:30] A whiteboard.")
        (root / "vid.summary.txt").write_text("A retirement party.")

        client = _client(
            db, tmp_path / "uploads",
            {"keep_source": False, "source_retention_hours": 24},
        )
        _claim(client)

        assert not (root / "vid.mp4").exists()
        assert (root / "vid.frames.txt").exists()
        assert (root / "vid.summary.txt").exists()

    async def test_a_missing_source_still_frees_the_quota(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """A file deleted out of band leaves a row billing the user for bytes
        that are not there. The sweep is the only thing that would ever notice,
        so an absent file is a success, not an error."""
        await _processed_video(db, completed_hours_ago=48)

        client = _client(
            db, tmp_path / "uploads",
            {"keep_source": False, "source_retention_hours": 24},
        )
        _claim(client)

        assert await db.user_total_bytes(USER) == 0

    async def test_a_claim_still_returns_work(self, db: UploadsDB, tmp_path: Path):
        """The sweep runs on the claim path, so it must not be able to stand
        between a worker and its job."""
        await _processed_video(db, "old", completed_hours_ago=48)
        await db.record_upload(
            file_id="new", user=USER, filename="new.mp4", mime="video/mp4",
            bytes_=10, kind="video", collection="", chunks=0,
            uploaded_at=_iso(1), status="pending",
        )

        client = _client(
            db, tmp_path / "uploads",
            {"keep_source": False, "source_retention_hours": 24},
        )
        r = _claim(client)

        assert r.status_code == 200
        assert r.json()["file_id"] == "new"

    async def test_an_empty_queue_is_still_a_204(self, db: UploadsDB, tmp_path: Path):
        await _processed_video(db, completed_hours_ago=48)

        client = _client(
            db, tmp_path / "uploads",
            {"keep_source": False, "source_retention_hours": 24},
        )

        assert _claim(client).status_code == 204


class TestRequeueAfterReclamation:
    async def test_requeueing_a_reclaimed_video_is_refused(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """Proceeding would delete the video's existing chunks, queue a job
        against a path that no longer exists, and burn all three attempts
        failing on it — ending with a video that WAS fully searchable and now
        is not."""
        await _processed_video(db, completed_hours_ago=48)
        await db.mark_source_freed("vid", freed_at=_iso(1))

        client = _client(db, tmp_path / "uploads", {"keep_source": True})
        r = client.post(
            "/v1/files/vid/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 409
        assert "reclaimed" in r.json()["detail"]
        assert (await db.get_upload("vid"))["status"] == "ready"

    async def test_force_does_not_override_it(self, db: UploadsDB, tmp_path: Path):
        """`force` exists to overrule a guard protecting work in progress.
        There is nothing here to overrule — the bytes are gone."""
        await _processed_video(db, completed_hours_ago=48)
        await db.mark_source_freed("vid", freed_at=_iso(1))

        client = _client(db, tmp_path / "uploads", {"keep_source": True})
        r = client.post(
            "/v1/files/vid/requeue?force=true",
            headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 409

    async def test_a_requeue_clears_the_completion_time(self, db: UploadsDB):
        """Left behind, a re-run would carry the previous run's completion
        time and be eligible for reclamation the moment it finished."""
        await _processed_video(db, completed_hours_ago=48)

        assert await db.requeue_job("vid")

        assert (await db.get_upload("vid"))["completed_at"] == ""
        assert await db.reclaimable_sources(completed_before=_iso(24)) == []
