"""Tests for `reconcile_with_qdrant`'s ghost-row sweep.

The sweep enforces "Qdrant has nothing for this file_id → sqlite must not
have a row either". That rule was written when every row in `uploads` was
a file that had already been ingested, so "no points" could only mean the
row was a ghost.

Phase 32 broke that assumption by adding rows that are *supposed* to have
no points: a video sits at `status='pending'` holding bytes on disk until
the media worker gets to it. The sweep read those as ghosts and deleted
them on every boot, which is invisible in the moment — the row is simply
gone from `GET /v1/files` after a restart, with the mp4 still on disk and
nothing left pointing at it.

The tests below pin both halves: the sweep still removes real ghosts, and
it no longer removes rows that never claimed Qdrant content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audrey.kb.uploads_db import UploadsDB, reconcile_with_qdrant

USER = "a@b.c"
TEXT_COL = "kb_user_text_a_b_c"
IMAGE_COL = "kb_user_images_a_b_c"


class _FakeQdrant:
    """Only the four methods reconcile calls.

    Points are held as `(collection, payload)` so the fake can answer both
    `scroll_collection` (step 1, no user filter) and `list_user_files`
    (step 2, filtered) from one source of truth — the way the real client
    does, and the reason a row can't be live in one and dead in the other.
    """

    def __init__(self, points: dict[str, list[dict]] | None = None):
        self._points = points or {}

    async def list_collections(self) -> list[str]:
        return list(self._points)

    async def collection_exists(self, name: str) -> bool:
        return name in self._points

    async def scroll_collection(self, collection: str) -> list[tuple[str, dict]]:
        return [(f"pt-{i}", p) for i, p in enumerate(self._points.get(collection, []))]

    async def list_user_files(self, *, user: str, collection: str) -> list[dict]:
        seen: dict[str, dict] = {}
        for p in self._points.get(collection, []):
            if p.get("user") == user:
                seen.setdefault(p["file_id"], {"file_id": p["file_id"]})
        return list(seen.values())


def _point(file_id: str, *, user: str = USER, kind: str = "text") -> dict:
    return {
        "user": user, "file_id": file_id, "filename": f"{file_id}.txt",
        "mime": "text/plain", "bytes": 10, "kind": kind,
        "uploaded_at": "2026-08-01T00:00:00+00:00",
    }


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


async def _add(db: UploadsDB, file_id: str, *, status: str = "ready",
               chunks: int = 3, kind: str = "text", collection: str = TEXT_COL,
               user: str = USER) -> None:
    await db.record_upload(
        file_id=file_id, user=user, filename=f"{file_id}.bin", mime="text/plain",
        bytes_=10, kind=kind, collection=collection, chunks=chunks,
        uploaded_at="2026-08-01T00:00:00+00:00", status=status,
    )


async def _ids(db: UploadsDB, user: str = USER) -> set[str]:
    return {r["file_id"] for r in await db.list_user(user)}


# ─── The sweep still does its job ──────────────────────────────────────

class TestGhostsStillPruned:
    async def test_a_ready_row_with_no_points_is_pruned(self, db: UploadsDB):
        """The original contract. Points deleted out of band (manual purge,
        migration) must not leave a row the user can see but not open."""
        await _add(db, "ghost")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))

        assert stats["pruned_rows"] == 1
        assert await _ids(db) == set()

    async def test_a_ready_row_with_points_survives(self, db: UploadsDB):
        await _add(db, "live")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("live")]}))

        assert stats["pruned_rows"] == 0
        assert await _ids(db) == {"live"}

    async def test_pruning_is_scoped_to_the_owning_user(self, db: UploadsDB):
        """Another user's live file must not keep this user's ghost alive,
        and vice versa. The sweep reads per-user collections for a reason."""
        await _add(db, "mine-ghost")
        await _add(db, "theirs-live", user="other@b.c",
                   collection="kb_user_text_other_b_c")

        await reconcile_with_qdrant(db, _FakeQdrant({
            TEXT_COL: [],
            "kb_user_text_other_b_c": [_point("theirs-live", user="other@b.c")],
        }))

        assert await _ids(db) == set()
        assert await _ids(db, "other@b.c") == {"theirs-live"}

    async def test_an_image_row_with_its_single_point_survives(self, db: UploadsDB):
        """Images record `chunks=1`, so they stay inside the prunable set —
        the carve-out must not accidentally exempt them."""
        await _add(db, "pic", kind="image", chunks=1, collection=IMAGE_COL)

        await reconcile_with_qdrant(db, _FakeQdrant({
            IMAGE_COL: [_point("pic", kind="image")],
        }))

        assert await _ids(db) == {"pic"}


# ─── Rows that never claimed Qdrant content ────────────────────────────

class TestVideoRowsSurvive:
    @pytest.mark.parametrize("status", ["pending", "processing", "failed"])
    async def test_an_unfinished_video_survives_a_boot(self, db: UploadsDB, status: str):
        """The regression this file exists for. A video that hasn't been
        ingested has no points *by definition*; reading that as a ghost
        deleted the entire queue on every restart."""
        await _add(db, "vid", status=status, chunks=0, kind="video", collection="")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))

        assert stats["pruned_rows"] == 0
        assert await _ids(db) == {"vid"}

    async def test_a_completed_silent_video_survives(self, db: UploadsDB):
        """`ingest-result` with no segments writes `chunks=0, collection=''`
        and calls the row ready — the pipeline is done, there was just no
        speech. Nothing was written to Qdrant, so the sweep must leave it."""
        await _add(db, "silent", status="ready", chunks=0, kind="video", collection="")

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))

        assert await _ids(db) == {"silent"}

    async def test_a_completed_video_with_a_transcript_is_swept_normally(
        self, db: UploadsDB,
    ):
        """Once a video has chunks it makes the same claim a text file does,
        and earns the same scrutiny. Exempting videos wholesale would have
        been the lazy fix and would have leaked ghosts forever."""
        await _add(db, "spoken", status="ready", chunks=7, kind="video")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))

        assert stats["pruned_rows"] == 1
        assert await _ids(db) == set()

    async def test_a_pending_video_is_still_claimable_after_reconcile(
        self, db: UploadsDB,
    ):
        """The consequence in one test. Surviving the sweep is only worth
        anything if the worker can still pick the row up afterwards — this
        is the boot-then-poll sequence the deployed box actually runs."""
        await _add(db, "vid", status="pending", chunks=0, kind="video", collection="")

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))
        job = await db.claim_job(lease_id="L1", now="2026-08-01T01:00:00+00:00")

        assert job is not None
        assert job["file_id"] == "vid"


# ─── prunable_file_ids directly ────────────────────────────────────────

class TestPrunableFileIds:
    async def test_it_selects_only_ready_rows_with_chunks(self, db: UploadsDB):
        await _add(db, "ready-with-chunks", status="ready", chunks=2)
        await _add(db, "ready-no-chunks", status="ready", chunks=0)
        await _add(db, "pending", status="pending", chunks=0)
        await _add(db, "processing", status="processing", chunks=0)
        await _add(db, "failed", status="failed", chunks=0)

        assert await db.prunable_file_ids(USER) == {"ready-with-chunks"}

    async def test_it_is_keyed_on_user(self, db: UploadsDB):
        await _add(db, "mine")
        await _add(db, "theirs", user="other@b.c")

        assert await db.prunable_file_ids(USER) == {"mine"}


# ─── Backfill (step 1) is unaffected ───────────────────────────────────

class TestBackfill:
    async def test_a_row_present_only_in_qdrant_is_restored(self, db: UploadsDB):
        """Step 1 is the other half of the contract and shares no code with
        the prune — this pins that the change didn't disturb it."""
        qdrant = _FakeQdrant({TEXT_COL: [_point("restored"), _point("restored")]})

        stats = await reconcile_with_qdrant(db, qdrant)

        assert stats["backfilled_collections"] == 1
        rows = await db.list_user(USER)
        assert [(r["file_id"], r["chunks"]) for r in rows] == [("restored", 2)]

    async def test_backfill_does_not_resurrect_a_deleted_pending_video(
        self, db: UploadsDB,
    ):
        """A pending video has no points, so nothing in Qdrant can bring it
        back. Its survival depends entirely on the prune, which is why the
        prune is where the fix lives."""
        qdrant = _FakeQdrant({TEXT_COL: []})

        await reconcile_with_qdrant(db, qdrant)

        assert await db.list_user(USER) == []


# ─── Backfill must not clobber what only sqlite knows (Phase 38) ───────

class TestBackfillPreservesJobState:
    """Step 1 refreshes every user file from its Qdrant payload on every boot.
    It used to do that with `INSERT OR REPLACE`, which deletes the conflicting
    row and re-inserts it — so every column the statement did not name reverted
    to its schema default.

    All seven of `_UPLOADS_ADDED_COLUMNS` were unnamed. The visible cost was
    phase 37's summary: a video completed with a 1,257-character summary showed
    it in `GET /v1/files` until `audrey-ai` restarted, and then showed nothing,
    with no error and no log line. `duration_s` went the same way.

    The rule being pinned: **a Qdrant payload is authoritative about content,
    never about job lifecycle.** It knows the filename, the size and the chunk
    count. It does not know how many attempts a job took, why it failed, or
    what its summary said.
    """

    async def _completed_video(self, db: UploadsDB) -> None:
        await db.record_upload(
            file_id="vid", user=USER, filename="jasonRetirement.mp4",
            mime="video/mp4", bytes_=301936597, kind="video", collection="",
            chunks=0, uploaded_at="2026-08-01T00:00:00+00:00", status="pending",
        )
        await db.claim_job(lease_id="L1", now="2026-08-01T00:01:00+00:00")
        assert await db.complete_job(
            file_id="vid", lease_id="L1", collection=TEXT_COL, chunks=25,
            duration_s=565.0, summary="Two colleagues mark Jason's retirement.",
        )

    async def test_a_summary_survives_a_reconcile_pass(self, db: UploadsDB):
        await self._completed_video(db)

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("vid")]}))

        row = await db.get_upload("vid")
        assert row["summary"] == "Two colleagues mark Jason's retirement."

    async def test_the_audio_duration_survives_a_reconcile_pass(self, db: UploadsDB):
        await self._completed_video(db)

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("vid")]}))

        assert (await db.get_upload("vid"))["duration_s"] == 565.0

    async def test_a_failed_row_is_not_flipped_to_ready(self, db: UploadsDB):
        """`status` reaches `record_upload` as a default argument, not as
        something read from Qdrant, so honouring it on conflict would let a
        boot silently mark a failed video ready — chunkless, summaryless, and
        indistinguishable from one that worked."""
        await db.record_upload(
            file_id="vid", user=USER, filename="broken.mp4", mime="video/mp4",
            bytes_=99, kind="video", collection="", chunks=0,
            uploaded_at="2026-08-01T00:00:00+00:00", status="pending",
        )
        await db.claim_job(lease_id="L1", now="2026-08-01T00:01:00+00:00")
        await db.fail_job(file_id="vid", lease_id="L1", reason="ffmpeg said no")

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("vid")]}))

        row = await db.get_upload("vid")
        assert row["status"] == "failed"
        assert row["failure_reason"] == "ffmpeg said no"

    async def test_qdrant_still_wins_on_the_fields_it_owns(self, db: UploadsDB):
        """The other half. Preserving job state must not turn the backfill
        into a no-op — a chunk count that drifted from Qdrant is exactly what
        reconcile exists to correct."""
        await self._completed_video(db)

        qdrant = _FakeQdrant({TEXT_COL: [_point("vid"), _point("vid"), _point("vid")]})
        await reconcile_with_qdrant(db, qdrant)

        row = await db.get_upload("vid")
        assert row["chunks"] == 3
        assert row["filename"] == "vid.txt"


# ─── A video must not be demoted to a text file ────────────────────────
#
# Found on the deployed box 2026-08-05, not by a test: a `ready` row carrying
# a `summary` and a `completed_at` — fields only a video ever gets — sitting
# under `kind='text'`.
#
# The mechanism: a point's `kind` describes that POINT, not the FILE. Every
# text point is stamped `kind: "text"`, and a video's transcript, frames and
# summary all go through the text ingest path. `kind` is in `record_upload`'s
# ON CONFLICT update list, so every boot copied "text" onto the video's row.
#
# `artifact` is the signal that survives, because only the three video stages
# set it.

class TestVideoKindSurvivesReconcile:
    def _artifact_point(self, file_id: str, artifact: str) -> dict:
        # Deliberately `kind="text"`: that IS what the deployed payloads say,
        # and a fixture writing "video" here would test nothing.
        return {**_point(file_id, kind="text"), "artifact": artifact}

    async def _completed_video(self, db: UploadsDB) -> None:
        await db.record_upload(
            file_id="vid", user=USER, filename="jason retirement.mp4",
            mime="video/mp4", bytes_=288_000_000, kind="video", collection="",
            chunks=0, uploaded_at="2026-08-01T00:00:00+00:00", status="pending",
        )
        await db.claim_job(lease_id="L1", now="2026-08-01T00:01:00+00:00")
        await db.complete_job(
            file_id="vid", lease_id="L1", collection=TEXT_COL, chunks=25,
            duration_s=565.0, summary="Colleagues mark Jason's retirement.",
            completed_at="2026-08-01T00:10:00+00:00",
        )

    @pytest.mark.parametrize("artifact", ["transcript", "visual", "summary"])
    async def test_a_video_stays_a_video(self, db: UploadsDB, artifact: str):
        await self._completed_video(db)

        await reconcile_with_qdrant(db, _FakeQdrant(
            {TEXT_COL: [self._artifact_point("vid", artifact)]}))

        assert (await db.get_upload("vid"))["kind"] == "video"

    async def test_a_row_already_demoted_is_repaired(self, db: UploadsDB):
        await self._completed_video(db)
        # The deployed state: an earlier boot already wrote "text".
        await db.record_upload(
            file_id="vid", user=USER, filename="jason retirement.mp4",
            mime="video/mp4", bytes_=288_000_000, kind="text",
            collection=TEXT_COL, chunks=25,
            uploaded_at="2026-08-01T00:00:00+00:00",
        )
        assert (await db.get_upload("vid"))["kind"] == "text"

        await reconcile_with_qdrant(db, _FakeQdrant(
            {TEXT_COL: [self._artifact_point("vid", "transcript")]}))

        # No migration needed: the mechanism that broke it repairs it once it
        # is computing the right answer.
        assert (await db.get_upload("vid"))["kind"] == "video"

    async def test_one_artifact_point_among_many_is_enough(self, db: UploadsDB):
        """Points come back unordered, and a summary is one point among 25."""
        await self._completed_video(db)

        points = [_point("vid", kind="text") for _ in range(24)]
        points.insert(12, self._artifact_point("vid", "summary"))
        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: points}))

        assert (await db.get_upload("vid"))["kind"] == "video"

    async def test_an_ordinary_text_upload_is_untouched(self, db: UploadsDB):
        # The other direction: nothing here may promote a document to a video.
        await _add(db, "doc", kind="text")

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("doc")]}))

        assert (await db.get_upload("doc"))["kind"] == "text"

    async def test_an_image_upload_is_untouched(self, db: UploadsDB):
        await _add(db, "pic", kind="image", collection=IMAGE_COL, chunks=1)

        await reconcile_with_qdrant(db, _FakeQdrant(
            {IMAGE_COL: [_point("pic", kind="image")]}))

        assert (await db.get_upload("pic"))["kind"] == "image"

    async def test_the_repaired_kind_makes_it_reclaimable_again(self, db: UploadsDB):
        """Why this matters beyond a cosmetic label.

        `reclaimable_sources` filters on `kind = 'video'`, so a demoted row was
        silently exempt from source reclamation — the bytes would never be
        freed and nothing would ever say why.
        """
        await self._completed_video(db)
        await reconcile_with_qdrant(db, _FakeQdrant(
            {TEXT_COL: [self._artifact_point("vid", "transcript")]}))

        rows = await db.reclaimable_sources(
            completed_before="2099-01-01T00:00:00+00:00")
        assert [r["file_id"] for r in rows] == ["vid"]
