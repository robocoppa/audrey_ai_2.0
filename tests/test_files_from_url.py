"""Paste a link, get a video (Phase 41, step 1).

The route half of the fetch stage: accepting a URL, leasing the download to a
`media-fetcher`, and verifying what comes back. `tests/test_uploads_db.py` owns
the state machine underneath; this file owns the gates around it.

Three groups of case, and the middle one is where the risk is:

  - **Accepting a link.** Scheme, host allowlist, duplicates, quota. The
    allowlist is the only thing standing between this route and a
    general-purpose file downloader, so it is tested for what it *refuses*.
  - **Trusting the fetcher.** `media-fetcher` is a separate container that can
    be wrong, stale or broken, so `fetch/{id}/result` re-derives everything it
    is told: that the file is at the path the row implies, that the bytes are a
    video, and that they fit the quota. Every one of those failing must fail
    the *row*, with a reason and without leaving bytes behind.
  - **The lease.** Same guards as phase 33, one stage earlier: a fetcher that
    stalled past its lease and woke up cannot land its download on a row that
    has been swept and re-leased.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.auth import AuthedUser, require_user
from audrey.kb.uploads_db import UploadsDB
from audrey.routes.files import FileRow, _url_placeholder_name, _waiting_for_s
from audrey.routes.files import router as files_router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)
ME = "a@b.c"
SVC = {"X-Audrey-Service-Token": SECRET}

# 24 bytes of ftyp box is enough for libmagic to call it video/mp4, which is
# the real gate `fetch/{id}/result` runs. A fixture that faked the sniff would
# not test the thing that stops an HTML error page reaching the video queue.
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

ALLOWED = ["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"]


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


def _build_app(
    db: UploadsDB, tmp_path: Path, *, allowed=None, max_user_bytes=10 * 1024**3,
    max_bytes_mb=2048,
) -> FastAPI:
    app = FastAPI()
    app.include_router(files_router)
    app.dependency_overrides[require_user] = lambda: AuthedUser(
        email=ME, role="user", owui_id="u1")
    app.state.uploads_db = db
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=SECRET, owui_url="http://owui"),
        raw={"kb": {
            "upload_root": str(tmp_path / "uploads"),
            "max_upload_mb": 50, "max_user_bytes": max_user_bytes,
            "chunked": {"max_upload_mb": 2048, "part_size_mb": 8},
            "video": {"lease_minutes": 30, "max_attempts": 3},
            "fetch": {
                "allowed_hosts": ALLOWED if allowed is None else allowed,
                "max_bytes_mb": max_bytes_mb,
                "lease_minutes": 20,
                "max_attempts": 3,
                "max_duration_s": 7200,
            },
        }},
    )
    return app


@pytest.fixture
def app(db: UploadsDB, tmp_path: Path) -> FastAPI:
    return _build_app(db, tmp_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ── Accepting a link ──────────────────────────────────────────────────


class TestAccept:
    def test_a_youtube_link_is_accepted_and_returns_immediately(self, client):
        r = client.post("/v1/files/from-url", json={"url": URL})
        assert r.status_code == 200, r.text
        body = r.json()
        # Not 'pending'. The bytes do not exist yet, and saying 'pending' here
        # would put the row in the media worker's queue pointing at nothing.
        assert body["status"] == "fetch_pending"
        assert body["bytes"] == 0
        assert body["kind"] == "video"
        assert body["mime"] == ""

    async def test_the_row_records_where_it_came_from(self, client, db):
        client.post("/v1/files/from-url", json={"url": URL})
        rows = await db.list_user(ME)
        assert len(rows) == 1
        assert rows[0]["source_url"] == URL
        assert rows[0]["status"] == "fetch_pending"

    def test_source_url_reaches_the_file_list(self, client):
        client.post("/v1/files/from-url", json={"url": URL})
        rows = client.get("/v1/files").json()["files"]
        # The fourth-place bug phase 40 fixed: a column in the schema, the
        # migration and the SELECT still has to be on `FileRow` to be projected.
        assert "source_url" in FileRow.model_fields
        assert rows[0]["source_url"] == URL

    def test_a_host_not_on_the_list_is_refused(self, client, db):
        r = client.post("/v1/files/from-url", json={"url": "https://evil.test/v.mp4"})
        assert r.status_code == 403
        assert "evil.test" in r.json()["detail"]

    def test_a_lookalike_host_is_refused(self, client):
        # The reason matching is exact rather than endswith.
        r = client.post(
            "/v1/files/from-url", json={"url": "https://youtube.com.evil.test/v"},
        )
        assert r.status_code == 403

    def test_userinfo_cannot_smuggle_a_host_past_the_allowlist(self, client):
        # Resolves to evil.test. Matching on `netloc` instead of `hostname`
        # would see the literal string 'www.youtube.com@evil.test' — or, worse,
        # a naive split on '@' — and this is the case that catches it.
        r = client.post(
            "/v1/files/from-url",
            json={"url": "https://www.youtube.com@evil.test/watch?v=x"},
        )
        assert r.status_code == 403
        assert "evil.test" in r.json()["detail"]

    def test_a_port_does_not_defeat_the_allowlist(self, client):
        # `hostname` drops the port, so this is allowed — the host really is
        # youtube.com. Pinned so a future switch to netloc matching fails here.
        r = client.post(
            "/v1/files/from-url", json={"url": "https://www.youtube.com:443/watch?v=z"},
        )
        assert r.status_code == 200

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://youtube.com/v.mp4",
        "gopher://youtube.com/",
    ])
    def test_only_http_and_https_are_fetchable(self, client, url):
        assert client.post("/v1/files/from-url", json={"url": url}).status_code == 422

    def test_an_empty_url_is_refused(self, client):
        assert client.post("/v1/files/from-url", json={"url": "   "}).status_code == 422

    def test_an_empty_allowlist_refuses_everything(self, db, tmp_path):
        # The safe direction for an absent config: a deployment that predates
        # this setting must not silently acquire an open downloader.
        c = TestClient(_build_app(db, tmp_path, allowed=[]))
        r = c.post("/v1/files/from-url", json={"url": URL})
        assert r.status_code == 403
        assert "none configured" in r.json()["detail"]

    async def test_nothing_is_recorded_when_the_host_is_refused(self, client, db):
        client.post("/v1/files/from-url", json={"url": "https://evil.test/v.mp4"})
        assert await db.list_user(ME) == []

    def test_the_same_link_twice_is_refused(self, client):
        assert client.post("/v1/files/from-url", json={"url": URL}).status_code == 200
        r = client.post("/v1/files/from-url", json={"url": URL})
        assert r.status_code == 409
        assert "already in your files" in r.json()["detail"]

    async def test_a_failed_fetch_can_be_pasted_again(self, client, db):
        client.post("/v1/files/from-url", json={"url": URL})
        rows = await db.list_user(ME)
        claimed = await db.claim_fetch(lease_id="L", now="2026-08-05T00:00:00+00:00")
        await db.fail_job(
            file_id=claimed["file_id"], lease_id="L", reason="private video",
            stage="fetching",
        )
        # The one case a user would obviously retry by hand. Refusing it would
        # mean deleting the failed row first, for no reason anyone could guess.
        assert client.post("/v1/files/from-url", json={"url": URL}).status_code == 200
        assert len(await db.list_user(ME)) == len(rows) + 1

    def test_a_full_account_is_refused_before_any_download(self, db, tmp_path):
        # 100 MB quota against a 2048 MB per-URL ceiling: no fetch can fit.
        c = TestClient(_build_app(
            db, tmp_path, max_user_bytes=100 * 1024 * 1024, max_bytes_mb=2048))
        r = c.post("/v1/files/from-url", json={"url": URL})
        assert r.status_code == 413
        assert "2048MB" in r.json()["detail"]

    @pytest.mark.parametrize(("url", "expected"), [
        ("https://www.youtube.com/watch?v=abc123", "abc123"),
        ("https://youtu.be/xyz789", "xyz789"),
        ("https://www.youtube.com/", "video"),
    ])
    def test_the_placeholder_name_is_recognisable(self, url, expected):
        # Shown in the list until the fetcher reports the real title. A blank
        # name for the whole download reads as a broken row.
        assert _url_placeholder_name(url) == expected

    def test_a_fetching_row_reports_elapsed_time(self):
        now = _dt.datetime(2026, 8, 5, 12, 0, tzinfo=_dt.UTC)
        row = {"status": "fetch_pending", "leased_at": "",
               "uploaded_at": "2026-08-05T11:59:00+00:00"}
        # A download is the stage a user is most likely to be watching, because
        # they just started it by hand.
        assert _waiting_for_s(row, now=now) == 60.0
        row = {"status": "fetching", "leased_at": "2026-08-05T11:58:00+00:00",
               "uploaded_at": "2026-08-05T11:00:00+00:00"}
        assert _waiting_for_s(row, now=now) == 120.0


# ── Leasing the download ──────────────────────────────────────────────


class TestClaim:
    def test_an_empty_queue_is_204_not_an_error(self, client):
        # The steady state. A fetcher polling an idle box must not fill the log.
        assert client.post("/v1/files/fetch/claim", headers=SVC).status_code == 204

    def test_a_user_token_cannot_claim(self, client):
        # `require_service`, not `resolve_kb_caller`: this hands out a
        # filesystem path and acts for an arbitrary user.
        assert client.post("/v1/files/fetch/claim").status_code == 401

    def test_a_claim_carries_the_url_the_caps_and_a_writable_directory(self, client):
        client.post("/v1/files/from-url", json={"url": URL})
        job = client.post("/v1/files/fetch/claim", headers=SVC).json()
        assert job["source_url"] == URL
        assert job["user"] == ME
        assert job["lease_seconds"] == 20 * 60
        assert job["max_bytes"] == 2048 * 1024 * 1024
        assert job["max_duration_s"] == 7200
        # A directory, not a path: the extension is not known until yt-dlp has
        # picked a container.
        assert Path(job["stage_dir"]).is_dir()

    def test_the_claim_hands_out_staging_and_not_a_user_directory(self, client, tmp_path):
        client.post("/v1/files/from-url", json={"url": URL})
        job = client.post("/v1/files/fetch/claim", headers=SVC).json()
        stage = Path(job["stage_dir"])
        # The fetcher is the container with internet egress running a
        # downloader against arbitrary URLs. It writes here and Audrey moves
        # the file once it has checked it, so a bug in yt-dlp cannot reach
        # anybody's stored files — and a partial download never exists at the
        # path the row implies.
        assert stage == tmp_path / "uploads" / ".staging"
        assert "dest_dir" not in job
        # Leading dot so it can never collide with a user directory:
        # `sanitize_user` strips the edges of what it produces, so no sanitized
        # user id begins with one.
        assert stage.name.startswith(".")

    async def test_claiming_moves_the_row_out_of_the_queue(self, client, db):
        client.post("/v1/files/from-url", json={"url": URL})
        client.post("/v1/files/fetch/claim", headers=SVC)
        assert (await db.list_user(ME))[0]["status"] == "fetching"

    def test_two_fetchers_do_not_get_the_same_row(self, client):
        client.post("/v1/files/from-url", json={"url": URL})
        first = client.post("/v1/files/fetch/claim", headers=SVC)
        second = client.post("/v1/files/fetch/claim", headers=SVC)
        assert first.status_code == 200
        # The status transition under one lock hold is the mutual exclusion.
        assert second.status_code == 204


# ── Trusting the fetcher ──────────────────────────────────────────────


def _accept_and_claim(client: TestClient) -> dict:
    client.post("/v1/files/from-url", json={"url": URL})
    return client.post("/v1/files/fetch/claim", headers=SVC).json()


def _land(job: dict, payload: bytes = MP4, ext: str = ".mp4") -> Path:
    """Write bytes where the fetcher would have written them: staging."""
    path = Path(job["stage_dir"]) / f"{job['file_id']}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _stored(tmp_path: Path, file_id: str, ext: str = ".mp4") -> Path:
    """Where a verified download ends up — the same path `_source_path` builds."""
    return tmp_path / "uploads" / "a_b_c" / f"{file_id}{ext}"


def _age_out_lease(db: UploadsDB, file_id: str) -> None:
    """Backdate a lease so the next sweep considers it expired.

    Reaching into the row rather than configuring `lease_minutes: 0`, which
    does not work and is worth recording why: both `leased_at` and the expiry
    cutoff are `isoformat(timespec="seconds")`, and the sweep compares with a
    strict `<`. A zero-minute lease claimed and re-polled inside the same
    second gives `leased_at == expiry`, so nothing is swept. Backdating states
    the thing being simulated — an hour passing — instead of depending on
    clock resolution.
    """
    db._conn.execute(
        "UPDATE uploads SET leased_at = '2020-01-01T00:00:00+00:00' "
        "WHERE file_id = ?", (file_id,),
    )


class TestResult:
    async def test_a_landed_download_joins_the_video_queue(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "Real Title.mp4",
                  "duration_s": 565.0},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "pending"
        row = (await db.list_user(ME))[0]
        # Everything the fetcher learned lands at once, and from here the row
        # is indistinguishable from an upload waiting on the worker.
        assert row["status"] == "pending"
        assert row["filename"] == "Real Title.mp4"
        assert row["mime"] == "video/mp4"
        assert row["bytes"] == len(MP4)
        assert row["source_url"] == URL

    async def test_a_verified_download_is_moved_out_of_staging(self, client, tmp_path):
        job = _accept_and_claim(client)
        staged = _land(job)
        client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "Real Title.mp4"},
        )
        # Audrey does the move, not the fetcher, and only after every check has
        # passed — so the path the media worker derives never names a file that
        # has not been sniffed, sized and quota-checked.
        assert not staged.exists()
        assert _stored(tmp_path, job["file_id"]).read_bytes() == MP4

    async def test_the_fetch_attempt_count_does_not_follow_the_row(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "t.mp4"},
        )
        row = await db.get_upload(job["file_id"])
        # Otherwise a video that took two tries to download gets one try to
        # transcribe, and a slow network reads as a bad video.
        assert row["attempts"] == 0

    async def test_a_filename_whose_extension_is_wrong_fails_the_row(self, client, db):
        job = _accept_and_claim(client)
        _land(job, ext=".mp4")
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "Real Title.webm"},
        )
        # Both paths are rebuilt from the reported extension, so this would
        # otherwise hand the worker a path to nothing — three attempts later.
        assert r.status_code == 422
        row = await db.get_upload(job["file_id"])
        assert row["status"] == "failed"
        assert "must match the file written" in row["failure_reason"]

    async def test_an_html_error_page_never_reaches_the_video_queue(self, client, db):
        job = _accept_and_claim(client)
        landed = _land(job, payload=b"<html><body>Video unavailable</body></html>")
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "unavailable.mp4"},
        )
        assert r.status_code == 415
        row = await db.get_upload(job["file_id"])
        assert row["status"] == "failed"
        assert "not a video" in row["failure_reason"]
        # And the bytes are gone — a rejected file left on disk would bill the
        # user's quota for something nobody can see or delete.
        assert not landed.exists()

    async def test_an_empty_download_fails_the_row(self, client, db):
        job = _accept_and_claim(client)
        _land(job, payload=b"")
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "empty.mp4"},
        )
        assert r.status_code == 422
        assert (await db.get_upload(job["file_id"]))["status"] == "failed"

    async def test_the_real_size_is_checked_against_the_quota(self, db, tmp_path):
        c = TestClient(_build_app(
            db, tmp_path, max_user_bytes=len(MP4) - 1, max_bytes_mb=0))
        job = _accept_and_claim(c)
        landed = _land(job)
        r = c.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "big.mp4"},
        )
        # Accept time could only check a configured ceiling. This is the first
        # moment the true size exists, and it is the check that counts.
        assert r.status_code == 413
        assert (await db.get_upload(job["file_id"]))["status"] == "failed"
        assert not landed.exists()

    async def test_a_stale_lease_cannot_land_its_download(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        # The sweep took it back and another fetcher has it now.
        await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=3,
            stage="fetching",
        )
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "late.mp4"},
        )
        assert r.status_code == 409
        assert (await db.get_upload(job["file_id"]))["status"] == "fetch_pending"

    def test_an_unknown_file_is_404(self, client):
        r = client.post(
            "/v1/files/fetch/nope/result", headers=SVC,
            json={"lease_id": "L", "filename": "x.mp4"},
        )
        assert r.status_code == 404

    def test_a_user_token_cannot_post_a_result(self, client):
        r = client.post(
            "/v1/files/fetch/anything/result",
            json={"lease_id": "L", "filename": "x.mp4"},
        )
        assert r.status_code == 401


class TestFailed:
    async def test_the_reason_reaches_the_row_verbatim(self, client, db):
        job = _accept_and_claim(client)
        reason = "Private video. Sign in if you've been granted access to this video"
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/failed", headers=SVC,
            json={"lease_id": job["lease_id"], "reason": reason},
        )
        assert r.status_code == 200
        row = await db.get_upload(job["file_id"])
        assert row["status"] == "failed"
        # Not editorialised into "download failed", which is the message that
        # generates the support question this field exists to prevent.
        assert row["failure_reason"] == reason

    async def test_a_stale_lease_cannot_fail_a_reclaimed_row(self, client, db):
        job = _accept_and_claim(client)
        await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=3,
            stage="fetching",
        )
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/failed", headers=SVC,
            json={"lease_id": job["lease_id"], "reason": "too late"},
        )
        assert r.status_code == 409
        assert (await db.get_upload(job["file_id"]))["status"] == "fetch_pending"


# ── The lease sweep ───────────────────────────────────────────────────


class TestSweep:
    async def test_a_dead_fetcher_returns_its_row_to_the_queue(self, client, db):
        job = _accept_and_claim(client)
        assert (await db.get_upload(job["file_id"]))["status"] == "fetching"
        await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=3,
            stage="fetching",
        )
        # Back to 'fetch_pending', not 'pending' — there are still no bytes.
        assert (await db.get_upload(job["file_id"]))["status"] == "fetch_pending"

    async def test_the_sweep_gives_up_and_says_who_did(self, client, db):
        job = _accept_and_claim(client)
        await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=1,
            stage="fetching",
        )
        row = await db.get_upload(job["file_id"])
        assert row["status"] == "failed"
        # Naming the fetcher rather than the worker is the whole point of
        # parameterising the sweep instead of copying it.
        assert "media fetcher" in row["failure_reason"]

    async def test_the_claim_route_sweeps_its_own_stage(self, client, db):
        job = _accept_and_claim(client)
        _age_out_lease(db, job["file_id"])
        again = client.post("/v1/files/fetch/claim", headers=SVC)
        # Recovery rides the poll, exactly as `/jobs/claim` does — no
        # background task to supervise.
        assert again.status_code == 200
        assert again.json()["file_id"] == job["file_id"]
        assert again.json()["attempts"] == 2

    async def test_the_two_stages_do_not_sweep_each_other(self, client, db):
        job = _accept_and_claim(client)
        # The media worker's sweep must leave a downloading row alone: it is
        # not stuck, it is in a stage that worker knows nothing about.
        await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=1,
            stage="processing",
        )
        assert (await db.get_upload(job["file_id"]))["status"] == "fetching"


# ── The staging sweep ─────────────────────────────────────────────────
#
# A killed fetcher leaves a partial gigabyte behind. The lease sweep above
# returns the row; this returns the disk. Both ride the claim poll, because the
# only thing that needs either is a fetcher asking for work.


class TestStagingSweep:
    def _stage(self, client: TestClient, tmp_path: Path) -> Path:
        client.post("/v1/files/fetch/claim", headers=SVC)
        return tmp_path / "uploads" / ".staging"

    def test_a_file_belonging_to_no_row_is_deleted(self, client, tmp_path):
        stage = self._stage(client, tmp_path)
        orphan = stage / "99999999-0000-0000-0000-000000000000.mp4"
        orphan.write_bytes(b"half a download from a fetcher that died")
        client.post("/v1/files/fetch/claim", headers=SVC)
        assert not orphan.exists()

    def test_a_live_downloads_partial_file_is_left_alone(self, client, tmp_path):
        job = _accept_and_claim(client)
        stage = tmp_path / "uploads" / ".staging"
        partial = stage / f"{job['file_id']}.f137.mp4.part"
        partial.write_bytes(b"still being written to")
        client.post("/v1/files/fetch/claim", headers=SVC)
        # Liveness is asked of the database, not of the file's age. An age
        # heuristic would eventually delete a slow download mid-write — and it
        # would do it to the largest files, which took longest to fetch.
        assert partial.exists()

    def test_every_artifact_of_a_dead_job_goes(self, client, tmp_path):
        stage = self._stage(client, tmp_path)
        dead = "88888888-0000-0000-0000-000000000000"
        for suffix in (".mp4", ".f137.mp4.part", ".en.vtt", ".webm.ytdl"):
            (stage / f"{dead}{suffix}").write_bytes(b"x")
        client.post("/v1/files/fetch/claim", headers=SVC)
        # yt-dlp writes fragments, subtitle files and its own resume metadata.
        # The file_id is a uuid4 and contains no dot, so the first component
        # identifies the job whatever the rest turns out to be.
        assert list(stage.iterdir()) == []

    async def test_a_swept_row_keeps_its_staged_file_until_it_is_dead(
        self, client, db, tmp_path,
    ):
        job = _accept_and_claim(client)
        stage = tmp_path / "uploads" / ".staging"
        partial = stage / f"{job['file_id']}.mp4"
        partial.write_bytes(b"partial")
        _age_out_lease(db, job["file_id"])
        client.post("/v1/files/fetch/claim", headers=SVC)
        # The row went back to `fetch_pending` and was re-leased, so it is
        # still live. Deleting here would take the file out from under the
        # retry that is about to overwrite it anyway.
        assert partial.exists()

    async def test_the_file_goes_once_the_row_gives_up(self, client, db, tmp_path):
        job = _accept_and_claim(client)
        stage = tmp_path / "uploads" / ".staging"
        partial = stage / f"{job['file_id']}.mp4"
        partial.write_bytes(b"partial")
        await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=1,
            stage="fetching",
        )
        assert (await db.get_upload(job["file_id"]))["status"] == "failed"
        client.post("/v1/files/fetch/claim", headers=SVC)
        # A 'failed' row is in neither fetch state, so its bytes are garbage —
        # and they are the largest garbage this system produces.
        assert not partial.exists()

    def test_an_unreadable_staging_directory_does_not_break_the_claim(
        self, client, tmp_path, monkeypatch,
    ):
        client.post("/v1/files/from-url", json={"url": URL})

        def boom(_self):
            raise OSError("the disk is having a day")

        monkeypatch.setattr(Path, "iterdir", boom)
        # Tidying up must never stop a fetcher from getting work.
        assert client.post("/v1/files/fetch/claim", headers=SVC).status_code == 200


# ── Captions arriving with the download (step 4) ───────────────────────


SEGMENTS = [
    {"t_start": 0.0, "t_end": 3.0, "text": "Hello and welcome."},
    {"t_start": 3.0, "t_end": 9.0, "text": "Today we are talking about retirement."},
]


class TestFetchedTranscript:
    async def test_a_caption_track_is_stored_for_the_worker(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "t.mp4",
                  "segments": SEGMENTS, "transcript_source": "subtitles"},
        )
        assert r.status_code == 200, r.text
        row = await db.get_upload(job["file_id"])
        stored = json.loads(row["fetched_transcript"])
        # Stored rather than posted straight through, because the two stages
        # are separated by a queue and a container: there is no moment when
        # both are holding the row.
        assert stored["source"] == "subtitles"
        assert stored["segments"] == SEGMENTS

    async def test_no_captions_stores_nothing(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "t.mp4"},
        )
        row = await db.get_upload(job["file_id"])
        # The pre-step-4 behaviour, and the path an older fetcher takes.
        assert row["fetched_transcript"] == ""

    async def test_a_transcript_with_no_provenance_is_refused(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "t.mp4",
                  "segments": SEGMENTS},
        )
        # This string is shown to the user as an explanation of where their
        # transcript came from. A sidecar that could leave it blank could
        # leave it blank on a transcript that is wrong.
        assert r.status_code == 422
        assert (await db.get_upload(job["file_id"]))["status"] == "failed"

    async def test_an_invented_provenance_is_refused(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "t.mp4",
                  "segments": SEGMENTS, "transcript_source": "a human typed it"},
        )
        assert r.status_code == 422
        assert "not one of" in (await db.get_upload(job["file_id"]))["failure_reason"]

    async def test_whisper_is_not_a_thing_the_fetcher_may_claim(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "t.mp4",
                  "segments": SEGMENTS, "transcript_source": "whisper"},
        )
        # 'whisper' is the media worker's answer, not the fetcher's. Accepting
        # it here would let a row claim a transcript came from a model that
        # was never run.
        assert r.status_code == 422

    async def test_blank_segments_do_not_count_as_a_transcript(self, client, db):
        job = _accept_and_claim(client)
        _land(job)
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "t.mp4",
                  "segments": [{"t_start": 0.0, "t_end": 1.0, "text": "   "}]},
        )
        # A caption file of empty cues is a caption file with nothing in it,
        # and the right answer is whisper — not a 422 about attribution.
        assert r.status_code == 200
        assert (await db.get_upload(job["file_id"]))["fetched_transcript"] == ""


# ── Progress while downloading ────────────────────────────────────────
#
# Phase 40 declined a progress protocol for the *ingest* stage and that
# refusal stands. It does not transfer: the ingest stage has no honest
# denominator ("whisper done, frames next" is three coarse steps), and the
# surface it was avoiding was a route that could half-complete a row. This one
# writes three display fields under a `status='fetching'` predicate and cannot
# transition anything.


class TestProgress:
    async def test_the_real_title_replaces_the_placeholder(self, client, db):
        job = _accept_and_claim(client)
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/progress", headers=SVC,
            json={"lease_id": job["lease_id"], "title": "A Retirement Speech",
                  "downloaded_bytes": 0, "total_bytes": 288 * 1024 * 1024},
        )
        assert r.status_code == 200, r.text
        row = (await db.list_user(ME))[0]
        # Until this lands the row shows the video id, so "is it fetching the
        # one I meant?" is unanswerable for the whole download — which is
        # exactly when someone wants to ask.
        assert row["filename"] == "A Retirement Speech"
        assert row["status"] == "fetching"

    async def test_the_byte_counts_reach_the_file_list(self, client, db):
        job = _accept_and_claim(client)
        client.post(
            f"/v1/files/fetch/{job['file_id']}/progress", headers=SVC,
            json={"lease_id": job["lease_id"], "downloaded_bytes": 1024,
                  "total_bytes": 4096},
        )
        rows = client.get("/v1/files").json()["files"]
        assert (rows[0]["fetch_downloaded_bytes"], rows[0]["fetch_total_bytes"]) \
            == (1024, 4096)

    async def test_an_empty_title_leaves_the_name_alone(self, client, db):
        job = _accept_and_claim(client)
        before = (await db.get_upload(job["file_id"]))["filename"]
        client.post(
            f"/v1/files/fetch/{job['file_id']}/progress", headers=SVC,
            json={"lease_id": job["lease_id"], "downloaded_bytes": 10},
        )
        # A site with no title must not turn a row that had a usable
        # placeholder into an unnamed one.
        assert (await db.get_upload(job["file_id"]))["filename"] == before

    async def test_progress_cannot_change_a_rows_state(self, client, db):
        job = _accept_and_claim(client)
        client.post(
            f"/v1/files/fetch/{job['file_id']}/progress", headers=SVC,
            json={"lease_id": job["lease_id"], "title": "T",
                  "downloaded_bytes": 999, "total_bytes": 1000},
        )
        row = await db.get_upload(job["file_id"])
        # 99% is not "done". Only `fetch/{id}/result` moves a row on, and only
        # after it has verified the bytes — this route has no state to strand
        # anything in, which is the whole reason it is safe to add.
        assert row["status"] == "fetching"
        assert row["bytes"] == 0
        assert row["mime"] == ""

    async def test_a_stale_lease_cannot_paint_over_a_newer_run(self, client, db):
        job = _accept_and_claim(client)
        await db.sweep_expired_leases(
            expired_before="2099-01-01T00:00:00+00:00", max_attempts=3,
            stage="fetching",
        )
        again = client.post("/v1/files/fetch/claim", headers=SVC).json()
        client.post(
            f"/v1/files/fetch/{again['file_id']}/progress", headers=SVC,
            json={"lease_id": again["lease_id"], "downloaded_bytes": 500},
        )
        r = client.post(
            f"/v1/files/fetch/{job['file_id']}/progress", headers=SVC,
            json={"lease_id": job["lease_id"], "downloaded_bytes": 20},
        )
        # Otherwise the download appears to go backwards, for a reason nobody
        # could reconstruct afterwards.
        assert r.status_code == 409
        assert (await db.get_upload(job["file_id"]))["fetch_downloaded_bytes"] == 500

    async def test_the_counters_are_cleared_when_the_download_lands(self, client, db):
        job = _accept_and_claim(client)
        client.post(
            f"/v1/files/fetch/{job['file_id']}/progress", headers=SVC,
            json={"lease_id": job["lease_id"], "title": "Real Title",
                  "downloaded_bytes": 40, "total_bytes": 100},
        )
        _land(job)
        client.post(
            f"/v1/files/fetch/{job['file_id']}/result", headers=SVC,
            json={"lease_id": job["lease_id"], "filename": "Real Title.mp4"},
        )
        row = await db.get_upload(job["file_id"])
        # "40 of 100 bytes" on a finished row is a fact about a moment that has
        # passed, and `bytes` is now the real number.
        assert (row["fetch_downloaded_bytes"], row["fetch_total_bytes"]) == (0, 0)
        assert row["bytes"] == len(MP4)

    def test_a_user_token_cannot_report_progress(self, client):
        assert client.post(
            "/v1/files/fetch/anything/progress",
            json={"lease_id": "L", "downloaded_bytes": 1},
        ).status_code == 401

    async def test_a_row_nobody_holds_is_refused(self, client, db):
        client.post("/v1/files/from-url", json={"url": URL})
        rows = await db.list_user(ME)
        r = client.post(
            f"/v1/files/fetch/{rows[0]['file_id']}/progress", headers=SVC,
            json={"lease_id": "made-up", "downloaded_bytes": 1},
        )
        # 'fetch_pending', not 'fetching'. Nothing is downloading it.
        assert r.status_code == 409


class TestExtractorArgsRideTheClaim:
    """The 403 knob reaches the fetcher without rebuilding its image.

    Same argument as every other cap on the claim, and this one earns it
    hardest: which download client YouTube will serve changes on YouTube's
    schedule, so finding the working value has to cost a config edit and a
    restart rather than an image build per attempt.
    """

    def test_an_unconfigured_deployment_still_gets_one_attempt(self, client):
        client.post("/v1/files/from-url", json={"url": URL})
        job = client.post("/v1/files/fetch/claim", headers=SVC).json()
        # `[""]`, not `[]`. One attempt with no `--extractor-args` is yt-dlp's
        # own behaviour; an empty list would be a downloader configured to try
        # nothing. Unlike `allowed_hosts`, the safe direction here is to try —
        # this is a workaround for someone else's rate limiting, not a
        # security boundary.
        assert job["extractor_args"] == [""]

    def test_the_whole_list_travels_with_the_job(self, db, tmp_path):
        app = _build_app(db, tmp_path)
        app.state.cfg.raw["kb"]["fetch"]["extractor_args"] = [
            "youtube:player_client=android_vr", "youtube:player_client=tv", "",
        ]
        c = TestClient(app)
        c.post("/v1/files/from-url", json={"url": URL})
        job = c.post("/v1/files/fetch/claim", headers=SVC).json()
        # Order is meaningful: the fetcher tries them in this sequence.
        assert job["extractor_args"] == [
            "youtube:player_client=android_vr", "youtube:player_client=tv", "",
        ]

    def test_the_older_single_string_spelling_still_works(self, db, tmp_path):
        app = _build_app(db, tmp_path)
        app.state.cfg.raw["kb"]["fetch"]["extractor_args"] = "youtube:player_client=tv"
        c = TestClient(app)
        c.post("/v1/files/from-url", json={"url": URL})
        job = c.post("/v1/files/fetch/claim", headers=SVC).json()
        # This setting shipped as a bare string before it was a list. A
        # deployment whose config still says so must keep downloading rather
        # than silently trying nothing.
        assert job["extractor_args"] == ["youtube:player_client=tv"]
