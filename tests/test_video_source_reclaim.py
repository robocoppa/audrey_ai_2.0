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


async def _fetched_video(
    db: UploadsDB,
    file_id: str = "url1",
    *,
    user: str = USER,
    completed_hours_ago: float = 48.0,
    source_url: str = "https://www.youtube.com/watch?v=abc",
) -> None:
    """A video that arrived by URL and finished ingest (Phase 41).

    The only difference that matters downstream is `source_url`: the row
    remembers where the bytes came from, which is what makes reclaiming them
    recoverable rather than final.
    """
    await db.record_url_fetch(
        file_id=file_id, user=user, source_url=source_url,
        filename=f"{file_id}.mp4", uploaded_at=_iso(completed_hours_ago + 2),
    )
    claimed = await db.claim_fetch(lease_id=f"F-{file_id}", now=_iso(completed_hours_ago + 1))
    assert claimed is not None
    assert await db.complete_fetch(
        file_id=file_id, lease_id=f"F-{file_id}", filename=f"{file_id}.mp4",
        mime="video/mp4", bytes_=42_000_000,
    )
    await db.claim_job(lease_id=f"L-{file_id}", now=_iso(completed_hours_ago))
    assert await db.complete_job(
        file_id=file_id, lease_id=f"L-{file_id}", collection="kb_user_text_a_b_c",
        chunks=25, duration_s=565.0, summary="A chess opening.",
        completed_at=_iso(completed_hours_ago),
        transcript_source="auto_captions",
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

    async def test_an_absent_config_key_keeps_the_source(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """The default, and the single most important assertion here.

        This deletes a file the user uploaded. A deployment whose `config.yaml`
        predates the setting is exactly the one least expecting that, so the
        absent key must mean keep — not "fall through to the phase plan's
        preferred behaviour".
        """
        await _processed_video(db, completed_hours_ago=48 * 30)
        root = tmp_path / "uploads"
        source = root / "a_b_c" / "vid.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"video bytes")

        client = _client(db, root, {})  # no keep_source key at all
        _claim(client)

        assert source.exists()
        assert not (await db.get_upload("vid"))["source_freed_at"]

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


class TestFetchedSourcesAreADifferentDecision:
    """Phase 41. Reclaiming an upload destroys the only copy; reclaiming a
    fetched video leaves the row holding the address it came from.

    That difference is the entire argument, so it is tested as a difference —
    the two policies are exercised against each other, not each in isolation.
    """

    async def test_keeping_uploads_still_reclaims_fetched_videos(self, db: UploadsDB):
        await _processed_video(db, "uploaded", completed_hours_ago=48)
        await _fetched_video(db, "fetched", completed_hours_ago=48)

        rows = await db.reclaimable_sources(
            completed_before=_iso(24), uploaded=False, fetched=True,
        )

        assert [r["file_id"] for r in rows] == ["fetched"]

    async def test_the_reverse_policy_reclaims_only_uploads(self, db: UploadsDB):
        await _processed_video(db, "uploaded", completed_hours_ago=48)
        await _fetched_video(db, "fetched", completed_hours_ago=48)

        rows = await db.reclaimable_sources(
            completed_before=_iso(24), uploaded=True, fetched=False,
        )

        assert [r["file_id"] for r in rows] == ["uploaded"]

    async def test_keeping_both_reclaims_nothing(self, db: UploadsDB):
        await _processed_video(db, "uploaded", completed_hours_ago=48)
        await _fetched_video(db, "fetched", completed_hours_ago=48)

        assert await db.reclaimable_sources(
            completed_before=_iso(24), uploaded=False, fetched=False,
        ) == []

    async def test_the_sweep_reads_both_settings(self, db: UploadsDB, tmp_path: Path):
        """The shipped combination: keep uploads, reclaim fetches."""
        await _processed_video(db, "uploaded", completed_hours_ago=48)
        await _fetched_video(db, "fetched", completed_hours_ago=48)
        root = tmp_path / "uploads"
        (root / "a_b_c").mkdir(parents=True)
        for name in ("uploaded.mp4", "fetched.mp4"):
            (root / "a_b_c" / name).write_bytes(b"x")

        client = _client(
            db, root,
            {"keep_source": True, "keep_fetched_source": False,
             "source_retention_hours": 24},
        )
        _claim(client)

        assert (await db.get_upload("uploaded"))["source_freed_at"] == ""
        assert (await db.get_upload("fetched"))["source_freed_at"] != ""
        assert not (root / "a_b_c" / "fetched.mp4").exists()
        assert (root / "a_b_c" / "uploaded.mp4").exists()

    async def test_an_old_config_reclaims_nothing_at_all(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """A deployment whose config.yaml predates `keep_fetched_source` is
        exactly the one least expecting a delete. Silence means keep, for both
        settings — the guarantee is about what an absent key means, not about
        what this box happens to be configured for."""
        await _fetched_video(db, "fetched", completed_hours_ago=48)
        root = tmp_path / "uploads"
        (root / "a_b_c").mkdir(parents=True)
        (root / "a_b_c" / "fetched.mp4").write_bytes(b"x")

        client = _client(db, root, {"source_retention_hours": 24})
        _claim(client)

        assert (await db.get_upload("fetched"))["source_freed_at"] == ""
        assert (root / "a_b_c" / "fetched.mp4").exists()


class TestRequeueingAReclaimedFetch:
    """The path that makes reclaiming a fetch recoverable. Without it,
    `keep_fetched_source: false` would be an irreversible delete wearing a
    different name."""

    async def test_it_goes_back_to_the_download_queue_not_the_worker(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _fetched_video(db, "fetched", completed_hours_ago=48)
        await db.mark_source_freed("fetched", freed_at=_iso(1))

        client = _client(db, tmp_path / "uploads", {"keep_source": True})
        r = client.post(
            "/v1/files/fetched/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 200
        assert r.json()["status"] == "fetch_pending"
        row = await db.get_upload("fetched")
        # `pending` would hand a media worker a path that does not exist and
        # burn all three attempts discovering it.
        assert row["status"] == "fetch_pending"
        assert row["source_url"] == "https://www.youtube.com/watch?v=abc"

    async def test_the_reclamation_marker_is_lifted(self, db: UploadsDB, tmp_path: Path):
        await _fetched_video(db, "fetched", completed_hours_ago=48)
        await db.mark_source_freed("fetched", freed_at=_iso(1))

        client = _client(db, tmp_path / "uploads", {"keep_source": True})
        client.post(
            "/v1/files/fetched/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        row = await db.get_upload("fetched")
        # "The bytes are gone, and that is final" stops being true the moment a
        # re-fetch is queued. Left set, the row would show a strikethrough size
        # for a video that is downloading again, and the next sweep would skip
        # it forever.
        assert row["source_freed_at"] == ""
        assert row["bytes"] == 0

    async def test_the_old_transcript_attribution_does_not_ride_along(
        self, db: UploadsDB, tmp_path: Path,
    ):
        await _fetched_video(db, "fetched", completed_hours_ago=48)
        assert (await db.get_upload("fetched"))["transcript_source"] == "auto_captions"
        await db.mark_source_freed("fetched", freed_at=_iso(1))

        client = _client(db, tmp_path / "uploads", {"keep_source": True})
        client.post(
            "/v1/files/fetched/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        # Of the four fetch columns this is the only one a `ready` row still
        # holds — `complete_fetch` zeroes the progress pair and `complete_job`
        # clears the caption blob, so those three are cleared defensively
        # rather than because a live row carries them. This one is real: the
        # next download decides its own source, and "auto_captions" left on a
        # row that is downloading again labels a transcript that does not exist.
        assert (await db.get_upload("fetched"))["transcript_source"] == ""

    async def test_an_upload_with_no_url_is_still_refused(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """The 409 is not superseded — it is narrowed to the rows that really
        have no way back."""
        await _processed_video(db, "uploaded", completed_hours_ago=48)
        await db.mark_source_freed("uploaded", freed_at=_iso(1))

        client = _client(db, tmp_path / "uploads", {"keep_source": True})
        r = client.post(
            "/v1/files/uploaded/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.status_code == 409
        assert "reclaimed" in r.json()["detail"]

    async def test_a_fetched_video_that_still_has_its_bytes_reprocesses_normally(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """Having a URL does not mean re-downloading. The file is right there;
        a second download would be bandwidth spent to get what we already have —
        and one more request to a host we would rather not be noticed by."""
        await _fetched_video(db, "fetched", completed_hours_ago=48)
        root = tmp_path / "uploads"
        (root / "a_b_c").mkdir(parents=True)
        (root / "a_b_c" / "fetched.mp4").write_bytes(b"x" * 7)

        client = _client(db, root, {"keep_source": True})
        r = client.post(
            "/v1/files/fetched/requeue", headers={"X-Audrey-Service-Token": SECRET},
        )

        assert r.json()["status"] == "pending"
        row = await db.get_upload("fetched")
        assert row["status"] == "pending"
        # And the size is re-read from disk, as it always was.
        assert row["bytes"] == 7

    async def test_the_refetch_branch_cannot_strand_an_upload(self, db: UploadsDB):
        """Defence in depth against a caller that gets the flag wrong: a row
        with no URL sent to `fetch_pending` would be claimed by a fetcher with
        nothing to fetch. The statement refuses instead."""
        await _processed_video(db, "uploaded", completed_hours_ago=48)

        assert not await db.requeue_job("uploaded", refetch=True)
        assert (await db.get_upload("uploaded"))["status"] == "ready"

    async def test_refetch_reservation_releases_if_requeue_write_fails(
        self,
        db: UploadsDB,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        await _fetched_video(db, "fetched", completed_hours_ago=48)
        await db.mark_source_freed("fetched", freed_at=_iso(1))

        async def fail_requeue(*_args, **_kwargs):
            raise RuntimeError("sqlite write failed")

        monkeypatch.setattr(db, "requeue_job", fail_requeue)
        client = _client(db, tmp_path / "uploads", {"keep_source": True})

        with pytest.raises(RuntimeError, match="sqlite write failed"):
            client.post(
                "/v1/files/fetched/requeue",
                headers={"X-Audrey-Service-Token": SECRET},
            )

        usage = await db.quota_usage(USER)
        assert usage.url_fetch_bytes == 0
        assert usage.total_bytes == 0
