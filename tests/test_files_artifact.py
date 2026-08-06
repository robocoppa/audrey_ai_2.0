"""Reading a video's transcript back, in pages (2026-08-05).

Ingest writes `.transcript.txt`, `.frames.txt` and `.summary.txt` beside the
source, and until this route nothing ever read any of them. The only way to ask
"what did they say" was `kb_search`, which returns ranked fragments — at the
fast path's cap, one 992-char chunk out of twenty. A model asked for a
transcript said it could give only a partial excerpt, which was true, and then
invented a way to get the rest, which was not.

What is pinned here:

  - **Paging is honest.** Every page states where it is in the document and
    whether more exists, because the failure being fixed is a model reasoning
    about how much it cannot see.
  - **Pages break between lines.** Transcript lines are `[00:04:12] text`; a
    page cut mid-line hands over a fragment with no timestamp that is
    indistinguishable from the end of the document.
  - **A missing artifact says which kind of missing.** Still processing, wrong
    file type, and genuinely-produced-nothing need three different answers, and
    a bare 404 gets reported as "the video has no transcript" for all three.
  - **The user comes from the service caller, never the model.** Same
    arrangement as `POST /v1/files/list`, higher stakes: this returns document
    text rather than a listing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.kb.uploads_db import UploadsDB
from audrey.routes.files import router as files_router
from audrey.tools.dispatch import _USER_SCOPED_TOOLS

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)
ME = "a@b.c"
SVC = {"X-Audrey-Service-Token": SECRET}
FID = "11111111-1111-1111-1111-111111111111"

# Realistic transcript shape: timestamped lines, which is what makes the
# line-boundary snap matter.
LINES = [f"[00:{m:02d}:00] line {m} of the retirement video, spoken aloud."
         for m in range(60)]
TRANSCRIPT = "\n".join(LINES)


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


@pytest.fixture
def app(db: UploadsDB, tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(files_router)
    app.state.uploads_db = db
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=SECRET, owui_url="http://owui"),
        raw={"kb": {"upload_root": str(tmp_path / "uploads"),
                    "max_upload_mb": 50, "max_user_bytes": 10 * 1024**3,
                    "chunked": {"max_upload_mb": 2048, "part_size_mb": 8},
                    "video": {"lease_minutes": 30, "max_attempts": 3}}},
    )
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


async def _ready_video(db: UploadsDB, tmp_path: Path, *, sidecars=("transcript",),
                       filename="jason retirement.mp4", file_id=FID,
                       uploaded_at="2026-08-01T00:00:00+00:00", kind="video",
                       status="ready") -> None:
    await db.record_upload(
        file_id=file_id, user=ME, filename=filename, mime="video/mp4",
        bytes_=1024, kind=kind, collection="c", chunks=3,
        uploaded_at=uploaded_at, status=status,
    )
    root = tmp_path / "uploads" / "a_b_c"
    root.mkdir(parents=True, exist_ok=True)
    bodies = {"transcript": TRANSCRIPT, "frames": "a stage, a microphone",
              "summary": "Colleagues say goodbye to Jason."}
    for name in sidecars:
        (root / f"{file_id}.{name}.txt").write_text(bodies[name], "utf-8")


def _read(client: TestClient, **kw):
    body = {"user": ME, "filename": "jason retirement.mp4", **kw}
    return client.post("/v1/files/artifact", headers=SVC, json=body)


class TestReading:
    async def test_a_short_transcript_comes_back_whole(self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        r = _read(client, limit=100_000)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"] == TRANSCRIPT
        assert body["total_chars"] == len(TRANSCRIPT)
        # The end marker is explicit rather than something the model has to
        # derive from offset + len(text) == total_chars.
        assert body["next_offset"] is None

    async def test_the_visual_artifact_maps_to_the_frames_sidecar(
            self, client, db, tmp_path):
        # The one naming mismatch in the system: the Qdrant payload says
        # 'visual', the file on disk is '.frames.txt'.
        await _ready_video(db, tmp_path, sidecars=("frames",))
        r = _read(client, artifact="visual")
        assert r.status_code == 200
        assert r.json()["text"] == "a stage, a microphone"

    async def test_the_summary_artifact_reads(self, client, db, tmp_path):
        await _ready_video(db, tmp_path, sidecars=("summary",))
        assert _read(client, artifact="summary").json()["text"].startswith("Colleagues")

    async def test_an_unknown_artifact_names_the_real_ones(self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        r = _read(client, artifact="audio")
        assert r.status_code == 422
        for name in ("transcript", "visual", "summary"):
            assert name in r.json()["detail"]


class TestPaging:
    async def test_a_long_transcript_is_paged_and_says_so(self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        body = _read(client, limit=200).json()
        assert body["next_offset"] is not None
        assert body["total_chars"] == len(TRANSCRIPT)
        # The sentence that stops the model implying it has the whole thing.
        assert f"of {len(TRANSCRIPT):,}" in body["note"]
        assert f"offset={body['next_offset']}" in body["note"]

    async def test_paging_through_reassembles_the_document_exactly(
            self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        seen, offset, pages = "", 0, 0
        while True:
            body = _read(client, limit=200, offset=offset).json()
            seen += body["text"]
            pages += 1
            if body["next_offset"] is None:
                break
            offset = body["next_offset"]
            assert pages < 50, "paging did not terminate"
        # No gaps, no overlaps, no lost characters.
        assert seen == TRANSCRIPT
        assert pages > 1

    async def test_pages_break_between_lines_never_inside_one(
            self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        offset = 0
        while True:
            body = _read(client, limit=150, offset=offset).json()
            if body["next_offset"] is None:
                break
            # A page cut mid-line gives the model half a sentence with no
            # timestamp, which it cannot distinguish from the document ending.
            assert body["text"].endswith("\n"), repr(body["text"][-40:])
            offset = body["next_offset"]

    async def test_an_offset_past_the_end_returns_nothing_and_ends(
            self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        body = _read(client, offset=10_000_000).json()
        assert body["text"] == ""
        assert body["next_offset"] is None

    async def test_a_negative_offset_is_clamped_to_the_start(
            self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        assert _read(client, offset=-500).json()["offset"] == 0


class TestMissingArtifacts:
    async def test_a_video_still_processing_says_so(self, client, db, tmp_path):
        await _ready_video(db, tmp_path, sidecars=(), status="processing")
        r = _read(client)
        assert r.status_code == 404
        # Not "no transcript" — it does not exist *yet*, which is a different
        # answer and the difference is the whole point.
        assert "processing" in r.json()["detail"]
        assert "yet" in r.json()["detail"]

    async def test_a_text_file_says_transcripts_are_video_only(
            self, client, db, tmp_path):
        await _ready_video(db, tmp_path, sidecars=(), kind="text",
                           filename="notes.pdf")
        r = client.post("/v1/files/artifact", headers=SVC,
                        json={"user": ME, "filename": "notes.pdf"})
        assert r.status_code == 404
        assert "only for video" in r.json()["detail"]

    async def test_a_silent_video_says_it_produced_none(self, client, db, tmp_path):
        await _ready_video(db, tmp_path, sidecars=())
        r = _read(client)
        assert r.status_code == 404
        # The genuinely-empty case: ready, a video, and no speech in it.
        assert "no speech" in r.json()["detail"]

    async def test_an_unknown_filename_names_the_alternatives(
            self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        r = client.post("/v1/files/artifact", headers=SVC,
                        json={"user": ME, "filename": "wrong.mp4"})
        assert r.status_code == 404
        # A mistyped filename is correctable in one turn if the alternatives
        # are named; "not found" alone gets reported as a missing file.
        assert "jason retirement.mp4" in r.json()["detail"]

    async def test_a_user_with_nothing_is_told_that(self, client, db, tmp_path):
        r = client.post("/v1/files/artifact", headers=SVC,
                        json={"user": "nobody@x.y", "filename": "a.mp4"})
        assert r.status_code == 404
        assert "nothing uploaded" in r.json()["detail"]


class TestIsolation:
    async def test_a_user_token_cannot_reach_it(self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        # No service token. `require_service`, not `resolve_kb_caller`: the
        # route names its user in the body, so a JWT holder could otherwise
        # read anyone's documents by typing a different address.
        assert client.post(
            "/v1/files/artifact",
            json={"user": ME, "filename": "jason retirement.mp4"},
        ).status_code == 401

    async def test_another_users_file_is_not_readable(self, client, db, tmp_path):
        await _ready_video(db, tmp_path)
        r = client.post("/v1/files/artifact", headers=SVC,
                        json={"user": "other@x.y",
                              "filename": "jason retirement.mp4"})
        # Resolution is keyed on user, so the filename simply does not exist
        # for anybody else.
        assert r.status_code == 404

    def test_the_tool_is_user_scoped_in_the_dispatcher(self):
        # The other half of the security property. `audit_user_scoping` only
        # warns about a missing entry, and a warning is not a gate — so this
        # is the gate.
        assert "get_file_text" in _USER_SCOPED_TOOLS

    async def test_duplicate_filenames_resolve_newest_and_say_so(
            self, client, db, tmp_path):
        await _ready_video(db, tmp_path, file_id=FID,
                           uploaded_at="2026-08-01T00:00:00+00:00")
        await _ready_video(db, tmp_path, file_id="22222222-2222-2222-2222-222222222222",
                           uploaded_at="2026-08-04T00:00:00+00:00")
        body = _read(client, limit=100_000).json()
        # Silently picking one of two identically named files is how a
        # confident answer comes from the wrong source.
        assert "2 files are named" in body["note"]
        assert "most recent" in body["note"]
