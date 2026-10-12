"""Startup recovery preserves the upload catalog when its search index is lost."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from audrey.identity import Principal
from audrey.kb.storage_lifecycle import StorageLifecycle
from audrey.kb.uploads_db import UploadsDB, reconcile_with_qdrant
from audrey.kb.user_store import sanitize_user
from audrey.routes.app import files as native_files

USER = "a@b.c"
TEXT_COL = "kb_user_text_a_b_c"
IMAGE_COL = "kb_user_images_a_b_c"


class _FakeQdrant:
    """Owner-bearing payloads returned by an otherwise empty search index."""

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


def _point(
    file_id: str,
    *,
    user: str = USER,
    kind: str = "text",
    mime: str = "text/plain",
) -> dict:
    suffix = ".mp3" if mime == "audio/mpeg" else ".txt"
    return {
        "user": user, "file_id": file_id, "filename": f"{file_id}{suffix}",
        "mime": mime, "bytes": 10, "kind": kind,
        "uploaded_at": "2026-08-01T00:00:00+00:00",
    }


@pytest.fixture
def db(tmp_path: Path):
    database = UploadsDB(tmp_path / "uploads.sqlite")
    yield database
    database.close()


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


# ─── Missing index content never deletes owned files ──────────────────

class TestMissingIndexPreservesCatalog:
    @pytest.mark.parametrize("collections", [{}, {TEXT_COL: []}])
    @pytest.mark.parametrize("kind", ["text", "image", "audio", "video"])
    async def test_ingested_files_survive_missing_or_empty_collections(
        self, db: UploadsDB, collections: dict, kind: str, caplog,
    ):
        await _add(db, "retained", kind=kind)
        before = await db.get_upload("retained")
        usage = await db.quota_usage(USER)

        stats = await reconcile_with_qdrant(db, _FakeQdrant(collections))

        assert await db.get_upload("retained") == before
        assert await db.quota_usage(USER) == usage
        assert await _ids(db) == {"retained"}
        assert stats["missing_index_files"] == 1
        assert stats["missing_index_users"] == 1
        assert "catalog records retained" in caplog.text

    async def test_partial_point_loss_preserves_both_files(self, db: UploadsDB):
        await _add(db, "missing")
        await _add(db, "live")
        rows = await db.list_user(USER)

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("live")]}))

        assert await db.list_user(USER) == rows
        assert stats["missing_index_files"] == 1
        assert stats["backfilled_rows"] == 0

    async def test_missing_index_counts_are_scoped_to_the_owner(self, db: UploadsDB):
        await _add(db, "mine-missing")
        await _add(db, "theirs-live", user="other@b.c",
                   collection="kb_user_text_other_b_c")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({
            TEXT_COL: [],
            "kb_user_text_other_b_c": [_point("theirs-live", user="other@b.c")],
        }))

        assert await _ids(db) == {"mine-missing"}
        assert await _ids(db, "other@b.c") == {"theirs-live"}
        assert stats["missing_index_files"] == 1
        assert stats["missing_index_users"] == 1

    async def test_a_ready_file_with_points_needs_no_recovery(self, db: UploadsDB):
        await _add(db, "live")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("live")]}))

        assert stats["missing_index_files"] == 0
        assert stats["missing_index_users"] == 0
        assert await _ids(db) == {"live"}

    async def test_an_image_with_its_point_needs_no_recovery(self, db: UploadsDB):
        await _add(db, "pic", kind="image", chunks=1, collection=IMAGE_COL)

        stats = await reconcile_with_qdrant(db, _FakeQdrant({
            IMAGE_COL: [_point("pic", kind="image")],
        }))

        assert stats["missing_index_files"] == 0
        assert await _ids(db) == {"pic"}

    async def test_unreachable_qdrant_keeps_existing_rows(self, db: UploadsDB):
        class UnreachableQdrant(_FakeQdrant):
            async def list_collections(self):
                raise ConnectionError("Qdrant unavailable")

        await _add(db, "retained")
        before = await db.get_upload("retained")
        with pytest.raises(ConnectionError, match="Qdrant unavailable"):
            await reconcile_with_qdrant(db, UnreachableQdrant())

        assert await db.get_upload("retained") == before

    async def test_index_payload_cannot_transfer_ownership(self, db: UploadsDB):
        await _add(db, "mine")
        before = await db.get_upload("mine")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({
            "kb_user_text_other_b_c": [_point("mine", user="other@b.c")],
        }))

        assert await db.get_upload("mine") == before
        assert await _ids(db, "other@b.c") == set()
        assert stats["backfilled_rows"] == 0
        assert stats["missing_index_files"] == 1


# ─── Rows that never claimed Qdrant content ────────────────────────────

class TestVideoRowsSurvive:
    @pytest.mark.parametrize("status", ["pending", "processing", "failed"])
    async def test_an_unfinished_video_survives_a_boot(self, db: UploadsDB, status: str):
        """The regression this file exists for. A video that hasn't been
        ingested has no points *by definition*; reading that as a ghost
        deleted the entire queue on every restart."""
        await _add(db, "vid", status=status, chunks=0, kind="video", collection="")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))

        assert stats["missing_index_files"] == 0
        assert await _ids(db) == {"vid"}

    async def test_a_completed_silent_video_survives(self, db: UploadsDB):
        """`ingest-result` with no segments writes `chunks=0, collection=''`
        and calls the row ready — the pipeline is done, there was just no
        speech. Nothing was written to Qdrant, so the sweep must leave it."""
        await _add(db, "silent", status="ready", chunks=0, kind="video", collection="")

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))

        assert await _ids(db) == {"silent"}

    async def test_a_completed_video_with_a_lost_transcript_remains_listed(
        self, db: UploadsDB,
    ):
        """A missing transcript index cannot delete a video's catalog record."""
        await _add(db, "spoken", status="ready", chunks=7, kind="video")

        stats = await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: []}))

        assert stats["missing_index_files"] == 1
        assert await _ids(db) == {"spoken"}

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


# ─── indexed_file_ids directly ────────────────────────────────────────

class TestIndexedFileIds:
    async def test_it_selects_only_ready_rows_with_chunks(self, db: UploadsDB):
        await _add(db, "ready-with-chunks", status="ready", chunks=2)
        await _add(db, "ready-no-chunks", status="ready", chunks=0)
        await _add(db, "pending", status="pending", chunks=0)
        await _add(db, "processing", status="processing", chunks=0)
        await _add(db, "failed", status="failed", chunks=0)

        assert await db.indexed_file_ids(USER) == {"ready-with-chunks"}

    async def test_it_is_keyed_on_user(self, db: UploadsDB):
        await _add(db, "mine")
        await _add(db, "theirs", user="other@b.c")

        assert await db.indexed_file_ids(USER) == {"mine"}


# ─── Backfill (step 1) is unaffected ───────────────────────────────────

class TestBackfill:
    async def test_a_row_present_only_in_qdrant_is_restored(self, db: UploadsDB):
        """Legacy recovery restores metadata when the catalog itself is missing."""
        qdrant = _FakeQdrant({TEXT_COL: [_point("restored"), _point("restored")]})

        stats = await reconcile_with_qdrant(db, qdrant)

        assert stats["backfilled_collections"] == 1
        rows = await db.list_user(USER)
        assert [(r["file_id"], r["chunks"]) for r in rows] == [("restored", 2)]

    async def test_backfill_does_not_resurrect_a_deleted_pending_video(
        self, db: UploadsDB,
    ):
        """No index payload means no row can be recovered into an empty catalog."""
        qdrant = _FakeQdrant({TEXT_COL: []})

        await reconcile_with_qdrant(db, qdrant)

        assert await db.list_user(USER) == []


# ─── Backfill must not clobber what only sqlite knows (Phase 38) ───────

class TestBackfillPreservesJobState:
    """Index recovery cannot replace catalog metadata or job lifecycle state."""

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
            transcript_source="auto_captions",
        )

    async def test_the_transcript_provenance_survives_a_reconcile_pass(
        self, db: UploadsDB,
    ):
        """Phase 41 step 4, and the `summary` bug's exact shape.

        A Qdrant payload cannot know whether a human wrote the captions or
        whisper guessed at them. Naming this column in the upsert would clear
        it from the payload's silence — correct until the next restart, blank
        afterwards, on rows nobody thinks to re-check.
        """
        await self._completed_video(db)

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [_point("vid")]}))

        assert (await db.get_upload("vid"))["transcript_source"] == "auto_captions"

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
        """Legacy recovery's ready default cannot replace a failed job's state."""
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

    async def test_stale_payload_cannot_replace_catalog_metadata(self, db: UploadsDB):
        await self._completed_video(db)
        before = await db.get_upload("vid")
        usage = await db.quota_usage(USER)

        qdrant = _FakeQdrant({TEXT_COL: [_point("vid"), _point("vid"), _point("vid")]})
        await reconcile_with_qdrant(db, qdrant)

        assert await db.get_upload("vid") == before
        assert await db.quota_usage(USER) == usage


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
    def _artifact_point(
        self,
        file_id: str,
        artifact: str,
        *,
        mime: str = "video/mp4",
    ) -> dict:
        # Deliberately kind=text: that is what deployed artifact payloads say,
        # while MIME still describes the source file and distinguishes media
        # summaries from document summaries during repair.
        return {
            **_point(file_id, kind="text", mime=mime),
            "artifact": artifact,
        }

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

        # No migration needed: recovery repairs the label from stored MIME.
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

    async def test_a_document_summary_does_not_promote_the_pdf_to_video(
        self,
        db: UploadsDB,
    ):
        await db.record_upload(
            file_id="report",
            user=USER,
            filename="report.pdf",
            mime="application/pdf",
            bytes_=2048,
            kind="text",
            collection=TEXT_COL,
            chunks=3,
            uploaded_at="2026-08-01T00:00:00+00:00",
        )
        point = self._artifact_point(
            "report",
            "summary",
            mime="application/pdf",
        )

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [point]}))

        assert (await db.get_upload("report"))["kind"] == "text"

    async def test_an_image_upload_is_untouched(self, db: UploadsDB):
        await _add(db, "pic", kind="image", collection=IMAGE_COL, chunks=1)

        await reconcile_with_qdrant(db, _FakeQdrant(
            {IMAGE_COL: [_point("pic", kind="image")]}))

        assert (await db.get_upload("pic"))["kind"] == "image"

    @pytest.mark.parametrize("artifact", ["transcript", "summary"])
    async def test_audio_artifacts_stay_audio_after_reconcile(
        self, db: UploadsDB, artifact: str,
    ):
        await db.record_upload(
            file_id="audio", user=USER, filename="interview.mp3",
            mime="audio/mpeg", bytes_=1024, kind="audio", collection=TEXT_COL,
            chunks=2, uploaded_at="2026-08-01T00:00:00+00:00", status="ready",
        )
        point = {
            **_point("audio", kind="text", mime="audio/mpeg"),
            "artifact": artifact,
        }

        await reconcile_with_qdrant(db, _FakeQdrant({TEXT_COL: [point]}))

        assert (await db.get_upload("audio"))["kind"] == "audio"

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


async def test_empty_index_keeps_native_file_access_owner_bound(
    db: UploadsDB, tmp_path: Path,
):
    content = "This original remains readable after search index loss."
    owner_dir = tmp_path / sanitize_user(USER)
    owner_dir.mkdir()
    (owner_dir / "document.txt").write_text(content)
    await db.record_upload(
        file_id="document", user=USER, filename="document.txt", mime="text/plain",
        bytes_=len(content.encode()), kind="text", collection=TEXT_COL,
        chunks=1, uploaded_at="2026-10-10T00:00:00+00:00",
    )
    await _add(db, "foreign", user="other@b.c")
    await reconcile_with_qdrant(db, _FakeQdrant())

    principal = Principal(
        user_id="usr_owner", storage_namespace=USER, provider="cloudflare_access",
        provider_subject="owner", email=USER, display_name="Owner", role="user",
        status="active", auth_method="cloudflare_access",
    )
    app = FastAPI()
    app.include_router(native_files.router, prefix="/api")
    app.state.cfg = SimpleNamespace(raw={"kb": {"upload_root": str(tmp_path)}})
    app.state.uploads_db = db
    app.state.storage_lifecycle = StorageLifecycle(db)
    app.dependency_overrides[native_files._files_access] = lambda: principal
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        listing = await client.get("/api/files")
        assert listing.status_code == 200
        assert [row["id"] for row in listing.json()["items"]] == ["document"]
        download = await client.get("/api/files/document/download")
        assert download.status_code == 200
        assert download.text == content
        text = await client.get("/api/files/document/text")
        assert text.status_code == 200
        assert text.json()["text"] == content
        assert (await client.get("/api/files/foreign/download")).status_code == 404

    request = SimpleNamespace(app=app)
    attachments = await native_files.resolve_owned_attachments(request, principal, ["document"])
    assert [item.file_id for item in attachments] == ["document"]
    with pytest.raises(HTTPException) as denied:
        await native_files.resolve_owned_attachments(request, principal, ["foreign"])
    assert denied.value.status_code == 422
