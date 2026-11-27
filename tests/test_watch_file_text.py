"""Project file text Watches use real canonical attachments and local bytes."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.routes.app import router
from audrey.watch_executor import WatchExecutor
from audrey.watch_file_text import ProjectFileTextWatchSource
from audrey.watch_runtime import WatchRuntime

FILENAME = "release-notes.txt"


@pytest.fixture
async def state(tmp_path, monkeypatch):
    async def unused(*_args, **_kwargs):
        pytest.fail("The file text Watch must stay on the local file path.")

    root = tmp_path / "files"
    root.mkdir()
    store = ApplicationStore(tmp_path / "application.sqlite")
    clock = [dt.datetime(2026, 11, 26, tzinfo=dt.UTC)]
    store.watches._clock = lambda: clock[0]
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    owner = await store.resolve_external_identity(
        provider="test", subject="owner", email="owner@example.com", display_name="Owner", role="user",
        auth_method="test",
    )
    project = await store.projects.create(user_id=owner.user_id, name="File Watch test")
    app = FastAPI()
    app.state.application_store = store
    app.state.uploads_db = uploads
    app.state.cfg = SimpleNamespace(raw={"kb": {"upload_root": str(root)}, "passthrough": {"enabled": False}})
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner
    monkeypatch.setattr("audrey.watch_executor.resolve_model", unused)
    executor = WatchExecutor(app, resolver=unused, fetcher=unused)
    runtime = WatchRuntime(store.watches, executor, project_files_executor=executor.complete_project_files, max_attempts=1)
    directory = root / sanitize_user(owner.storage_namespace)
    directory.mkdir()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, uploads=uploads, owner=owner, project=project, directory=directory,
                              runtime=runtime, client=client, clock=clock)
    uploads.close()
    store.close()


async def attach(state, file_id, text, *, filename=FILENAME, status="ready"):
    path = state.directory / f"{file_id}.txt"
    path.write_bytes(text)
    await state.uploads.record_upload(
        file_id=file_id, user=state.owner.storage_namespace, filename=filename, mime="text/plain",
        bytes_=len(text), kind="text", collection="", chunks=0, status=status,
        uploaded_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    await state.store.projects.add_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=file_id)
    return path


async def detach(state, file_id):
    assert await state.store.projects.remove_file(
        user_id=state.owner.user_id, project_id=state.project.project_id, file_id=file_id,
    )


async def create(state, **changes):
    return await state.client.post("/api/watches", json={
        "title": "Release notes text", "source_kind": "project_files", "source_filename": FILENAME,
        "project_id": state.project.project_id, "request_id": "file-watch", "enabled": False,
        "schedule": {"kind": "interval", "interval_minutes": 60, "timezone": "UTC"}, **changes,
    })


async def check(state, watch):
    path = f"/api/watches/{watch['id']}"
    state.clock[0] += dt.timedelta(seconds=1)
    queued = await state.client.post(f"{path}/actions", json={"action": "run", "expected_revision": watch["revision"]})
    assert queued.status_code == 200, queued.text
    assert queued.json()["next_run_at"] is None
    assert await state.runtime.run_once()
    run_path = f"{path}/runs/{queued.json()['active_run_id']}"
    response = await state.client.get(run_path)
    assert response.status_code == 200, response.text
    if response.json()["status"] == "queued":
        # A retryable failure consumes this fixture's one allowed attempt.
        state.clock[0] += dt.timedelta(minutes=2)
        await state.runtime.run_once()
        response = await state.client.get(run_path)
        assert response.status_code == 200, response.text
    return response.json()


async def test_watch_compares_exact_same_name_attachments_and_rebaselines_after_detach(state):
    base_text = b"Release channel is stable.\n"
    next_text = b"Release channel is long-term support.\n"
    await attach(state, "original", base_text)
    response = await create(state)
    assert response.status_code == 201, response.text
    watch = response.json()
    assert watch["source_filename"] == FILENAME
    assert (await check(state, watch))["outcome"] == "baseline"

    await attach(state, "other", next_text, filename="RELEASE-NOTES.txt")
    await attach(state, "same-text", b"\xef\xbb\xbfRelease channel is stable.\r\n")
    assert (await check(state, watch))["outcome"] == "unchanged"
    await attach(state, "revision", next_text)
    changed = await check(state, watch)
    assert changed["outcome"] == "changed" and changed["significant"] is True
    assert changed["definition"]["source_filename"] == FILENAME
    assert changed["report"] == "The saved file text changed."
    notices = (await state.client.get("/api/notifications")).json()["items"]
    assert [notice["run_id"] for notice in notices] == [changed["id"]]

    await detach(state, "revision")
    with state.store._lock:
        assert state.store._conn.execute("SELECT COUNT(*) FROM app_watch_file_text_baselines").fetchone()[0] == 0
        assert all("Release channel" not in row[0] for row in state.store._conn.execute(
            "SELECT record_json FROM app_watch_snapshots",
        ))
        assert state.store._conn.execute(
            "SELECT COUNT(*) FROM app_watch_runs WHERE baseline_json IS NOT NULL",
        ).fetchone()[0] == 0
    assert (await check(state, watch))["outcome"] == "baseline"
    assert len((await state.client.get("/api/notifications")).json()["items"]) == 1


async def test_watch_rejects_unavailable_and_revoked_sources_without_advancing_baseline(state, monkeypatch):
    missing = await create(state)
    assert missing.status_code == 404
    await attach(state, "original", b"Baseline text.")
    response = await create(state)
    assert response.status_code == 201, response.text
    watch = response.json()
    assert (await check(state, watch))["outcome"] == "baseline"
    await attach(state, "large", b"x" * 64_001)
    assert (await check(state, watch))["status"] == "failed"
    await detach(state, "large")

    await attach(state, "newer", b"Changed text.")
    original = ProjectFileTextWatchSource.capture

    async def detach_after_read(reader, *args, **kwargs):
        captured = await original(reader, *args, **kwargs)
        await detach(state, "newer")
        return captured

    monkeypatch.setattr(ProjectFileTextWatchSource, "capture", detach_after_read)
    assert (await check(state, watch))["status"] == "failed"
    notices = (await state.client.get("/api/notifications")).json()["items"]
    assert all(notice["kind"] == "failure" for notice in notices)
    with state.store._lock:
        baseline = state.store._conn.execute("SELECT file_id FROM app_watch_file_text_baselines").fetchone()
        assert baseline["file_id"] == "original"
