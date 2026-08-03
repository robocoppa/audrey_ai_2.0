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
