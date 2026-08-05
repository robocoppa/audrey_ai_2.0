"""`POST /v1/files/list` — the service-token file listing behind `list_my_files`.

Phase 40. The tool exists so a chat turn can answer "what videos have I
uploaded?", which needs the tools-server to read a file list. It holds
`KB_SERVICE_TOKEN` and cannot obtain a user JWT, so `GET /v1/files` — which
depends on `require_user` — is permanently out of reach. This route is the
service-token counterpart.

The security shape is worth stating because it is unusual for this codebase:
**this route names its target user in the request body.** Every other user
endpoint derives identity from a validated token and cannot be pointed
elsewhere. The property that keeps it safe lives in a different file — the
dispatcher's `_USER_SCOPED_TOOLS` overwrite, pinned in `test_dispatch.py`.
What is pinned here is the half this file owns: a user JWT must not reach it,
and the rows it returns must be the named user's and nobody else's.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.kb.uploads_db import UploadsDB
from audrey.routes.files import router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


def _build_app(db: UploadsDB, tmp_path: Path, *, service_token: str = SECRET) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.uploads_db = db
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=service_token, owui_url="http://owui"),
        raw={"kb": {"upload_root": str(tmp_path / "uploads"),
                    "video": {"lease_minutes": 30, "max_attempts": 3}}},
    )
    return app


async def _add(
    db: UploadsDB, file_id: str, *, user: str, filename: str | None = None,
    kind: str = "video", status: str = "ready",
    uploaded_at: str = "2026-08-01T00:00:00+00:00",
) -> None:
    await db.record_upload(
        file_id=file_id, user=user, filename=filename or f"{file_id}.mp4",
        mime="video/mp4", bytes_=1024, kind=kind, collection="", chunks=0,
        uploaded_at=uploaded_at, status=status,
    )


def _post(app: FastAPI, body: dict, *, token: str | None = SECRET):
    headers = {"X-Audrey-Service-Token": token} if token is not None else {}
    return TestClient(app).post("/v1/files/list", json=body, headers=headers)


class TestRouteAuth:
    """A route that takes its user from the body must not be reachable by
    anyone who merely holds a valid user token."""

    def test_without_a_token_is_401(self, db: UploadsDB, tmp_path: Path):
        assert _post(_build_app(db, tmp_path), {"user": "a@b.c"}, token=None).status_code == 401

    def test_with_the_wrong_token_is_401(self, db: UploadsDB, tmp_path: Path):
        r = _post(_build_app(db, tmp_path), {"user": "a@b.c"}, token="nope")  # noqa: S106
        assert r.status_code == 401

    def test_with_an_empty_token_is_401(self, db: UploadsDB, tmp_path: Path):
        """An empty configured secret must not make everything match."""
        app = _build_app(db, tmp_path, service_token="")
        assert _post(app, {"user": "a@b.c"}, token="").status_code == 401


class TestIsolation:
    @pytest.mark.asyncio
    async def test_returns_only_the_named_users_files(self, db: UploadsDB, tmp_path: Path):
        await _add(db, "v1", user="alice@example.com", filename="standup.mp4")
        await _add(db, "v2", user="bob@example.com", filename="secret.mp4")

        r = _post(_build_app(db, tmp_path), {"user": "alice@example.com"})

        assert r.status_code == 200
        names = [f["filename"] for f in r.json()["files"]]
        assert names == ["standup.mp4"]
        assert "secret.mp4" not in r.text

    @pytest.mark.asyncio
    async def test_an_unknown_user_gets_an_empty_list_not_a_404(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """"You have not uploaded anything" is a true answer, and a 404 would
        distinguish a real address from an unused one."""
        await _add(db, "v1", user="alice@example.com")

        r = _post(_build_app(db, tmp_path), {"user": "nobody@example.com"})

        assert r.status_code == 200
        assert r.json() == {"user": "nobody@example.com", "files": []}

    def test_a_blank_user_is_422(self, db: UploadsDB, tmp_path: Path):
        """Whitespace must not fall through to a query matching nothing —
        or, worse, to some future query where it matches everything."""
        assert _post(_build_app(db, tmp_path), {"user": "   "}).status_code == 422

    def test_a_missing_user_is_422(self, db: UploadsDB, tmp_path: Path):
        assert _post(_build_app(db, tmp_path), {}).status_code == 422


class TestModelShape:
    """The response is shaped for a language model, not for the upload page.
    These pin the difference, because the temptation to reuse `FileRow` will
    recur every time someone adds a column."""

    @pytest.mark.asyncio
    async def test_carries_what_a_model_can_speak_from(self, db: UploadsDB, tmp_path: Path):
        await _add(db, "v1", user="alice@example.com", filename="standup.mp4",
                   status="pending")
        await db.claim_job(lease_id="L1", now="2026-08-01T10:00:00+00:00")
        await db.complete_job(
            file_id="v1", lease_id="L1", collection="kb_user_text_alice", chunks=12,
            duration_s=565.0, summary="A nine-minute standup about the handover.",
        )

        row = _post(_build_app(db, tmp_path), {"user": "alice@example.com"}).json()["files"][0]

        assert row["filename"] == "standup.mp4"
        assert row["kind"] == "video"
        assert row["status"] == "ready"
        assert row["duration_s"] == 565.0
        assert row["summary"] == "A nine-minute standup about the handover."
        assert row["uploaded_at"] == "2026-08-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_omits_the_fields_a_model_has_no_use_for(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """`bytes`, `collection` and `source_freed_at` are page furniture, and
        `file_id` is a UUID a model would only mangle — the filter resolves a
        filename instead. Every extra field is context spent and one more thing
        to be wrong about."""
        await _add(db, "v1", user="alice@example.com")

        row = _post(_build_app(db, tmp_path), {"user": "alice@example.com"}).json()["files"][0]

        for absent in ("file_id", "bytes", "collection", "source_freed_at", "mime", "chunks"):
            assert absent not in row

    @pytest.mark.asyncio
    async def test_a_failed_file_carries_its_reason(self, db: UploadsDB, tmp_path: Path):
        """Same principle as the upload page: a file that stopped moving
        without saying why is the failure this field exists to prevent, and
        that is as true in a chat answer as on a page."""
        await _add(db, "v1", user="alice@example.com", status="pending")
        await db.claim_job(lease_id="L1", now="2026-08-01T10:00:00+00:00")
        await db.fail_job(file_id="v1", lease_id="L1", reason="no audio stream")

        row = _post(_build_app(db, tmp_path), {"user": "alice@example.com"}).json()["files"][0]

        assert row["status"] == "failed"
        assert row["failure_reason"] == "no audio stream"

    @pytest.mark.asyncio
    async def test_a_processing_file_is_listed_not_hidden(
        self, db: UploadsDB, tmp_path: Path,
    ):
        """A video mid-ingest must appear. Hiding it would have the model
        report the file as absent while the user is watching it process."""
        await _add(db, "v1", user="alice@example.com", status="pending")
        await db.claim_job(lease_id="L1", now="2026-08-01T10:00:00+00:00")

        rows = _post(_build_app(db, tmp_path), {"user": "alice@example.com"}).json()["files"]

        assert [r["status"] for r in rows] == ["processing"]
