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
        assert Path(job["dest_dir"]).is_dir()

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
    """Write bytes where the fetcher would have written them."""
    path = Path(job["dest_dir"]) / f"{job['file_id']}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


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
        # `_source_path` rebuilds from the reported extension, so this would
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
