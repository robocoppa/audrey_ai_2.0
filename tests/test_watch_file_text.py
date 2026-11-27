"""Project file text Watches use real canonical attachments and local bytes."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.kb.private_documents import private_text_collection
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.processing_policy import ProcessingPolicy, processing_scope
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
    executor = WatchExecutor(app, client_factory=unused, resolver=unused, fetcher=unused)
    runtime = WatchRuntime(store.watches, executor, project_files_executor=executor.complete_project_files, max_attempts=1)
    directory = root / sanitize_user(owner.storage_namespace)
    directory.mkdir()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, uploads=uploads, owner=owner, project=project, directory=directory,
                              runtime=runtime, client=client, clock=clock, app=app)
    uploads.close()
    store.close()


@pytest.fixture
async def private_state(state, monkeypatch):
    state.project = await state.store.projects.create(
        user_id=state.owner.user_id, name="Confidential file Project", private_required=True,
    )
    state.app.state.cfg.raw["private_processing"] = {"enabled": True}
    state.granted = [True]
    state.model_access_reads = []

    async def private_access(_cfg, _store, principal, model_id):
        assert principal.user_id == state.owner.user_id
        assert model_id == "private"
        state.model_access_reads.append(model_id)
        return SimpleNamespace(id="private") if state.granted[0] else None

    monkeypatch.setattr("audrey.watch_executor.resolve_model", private_access)
    return state


async def attach_private(state, label, text, *, filename=FILENAME, status="ready", **changes):
    """Seed a genuinely restricted owned upload and canonical Private attachment."""
    file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"watch-file-test:{label}"))
    path = state.directory / f"{file_id}.txt"
    path.write_bytes(text)
    stamp = state.clock[0].isoformat()
    protected = bool(changes.get("private_required", True))
    origin = changes.get("origin", "upload")
    reservation = await state.uploads.reserve_single_upload(
        reservation_id=file_id, user=state.owner.storage_namespace, bytes_=len(text),
        max_user_bytes=1_000_000, now=stamp,
        expired_before=(state.clock[0] - dt.timedelta(hours=1)).isoformat(), private_required=protected,
    )
    assert reservation.accepted
    committed = await state.uploads.commit_reserved_upload(
        reservation_id=file_id, reservation_kind="single_shot", file_id=file_id,
        user=state.owner.storage_namespace, filename=filename, mime="text/plain", bytes_=len(text),
        kind="text", collection="" if origin == "generated" else private_text_collection(state.owner.storage_namespace),
        chunks=0 if origin == "generated" else 1, origin=origin,
        uploaded_at=stamp, status=status, max_user_bytes=1_000_000,
    )
    assert committed.accepted
    if changes:
        with state.uploads._lock, state.uploads._conn:
            metadata = {
                "collection": "" if origin == "generated" else private_text_collection(state.owner.storage_namespace),
                "chunks": 0 if origin == "generated" else 1, "source_freed_at": "",
            }
            assert changes.keys() <= metadata.keys() | {"origin", "private_required"}
            metadata.update({key: value for key, value in changes.items() if key in metadata})
            state.uploads._conn.execute(
                "UPDATE uploads SET collection=?,chunks=?,source_freed_at=? WHERE file_id=?",
                (*metadata.values(), file_id),
            )
    # The production add-file boundary requires a real protected scope. No
    # model client or metadata transport is used to seed this local test data.
    policy = ProcessingPolicy({"test-local:latest": "a" * 64}, "http://ollama.test")
    with processing_scope(policy):
        await state.store.projects.add_file(
            user_id=state.owner.user_id, project_id=state.project.project_id,
            file_id=file_id, file_is_private=True, file_origin=origin,
        )
    return file_id, path


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


async def test_private_file_watch_compares_real_local_sources_without_copying_content(private_state):
    state = private_state
    before = b"Confidential launch budget is 937.51.\n"
    after = b"Confidential launch budget is 1082.73.\n"
    await attach_private(state, "original", before)
    response = await create(state, private_required=True, title="Sensitive launch review")
    assert response.status_code == 201, response.text
    watch = response.json()
    assert watch["private_required"] is True and watch["project"]["private_required"] is True
    assert watch["active_run_id"] is None
    baseline = await check(state, watch)
    assert baseline["outcome"] == "baseline"
    assert baseline["definition"]["private_required"] is True
    assert baseline["definition"]["project"]["private_required"] is True

    await attach_private(state, "case-mismatch", after, filename=FILENAME.upper())
    await attach_private(state, "line-endings", b"\xef\xbb\xbfConfidential launch budget is 937.51.\r\n")
    assert (await check(state, watch))["outcome"] == "unchanged"
    await attach_private(state, "revision", after)
    changed = await check(state, watch)
    assert changed["status"] == "completed" and changed["outcome"] == "changed"
    assert changed["significant"] is True and changed["diff"] == ""
    notices = (await state.client.get("/api/notifications")).json()["items"]
    assert len(notices) == 1 and notices[0]["run_id"] == changed["id"]
    assert notices[0]["title"] == "Private file watch"
    assert "Sensitive launch review" not in json.dumps(notices)
    assert "Confidential launch" not in json.dumps(notices)
    assert FILENAME not in json.dumps(notices)
    assert (await state.client.get(f"/api/watches/{watch['id']}/runs/{changed['id']}/snapshot")).status_code == 404
    listed = await state.client.get("/api/watches", params={"project_id": state.project.project_id})
    assert listed.status_code == 200 and listed.json()["items"][0]["id"] == watch["id"]
    assert state.model_access_reads and set(state.model_access_reads) == {"private"}
    with state.store._lock:
        retained = [tuple(row) for row in state.store._conn.execute(
            "SELECT definition_json,baseline_json,report,diff,error FROM app_watch_runs",
        )]
        retained += [tuple(row) for row in state.store._conn.execute("SELECT record_json FROM app_watch_snapshots")]
    assert "Confidential launch" not in json.dumps(retained)
    assert "937.51" not in json.dumps(retained) and "1082.73" not in json.dumps(retained)


@pytest.mark.parametrize("revocation", ["disabled", "access"])
async def test_private_watch_refuses_unavailable_capability_before_file_admission(private_state, monkeypatch, revocation):
    state = private_state
    await attach_private(state, "original", b"Protected source text.")
    if revocation == "disabled":
        state.app.state.cfg.raw["private_processing"]["enabled"] = False
    else:
        state.granted[0] = False

    async def unexpected_file_admission(*_args, **_kwargs):
        pytest.fail("An unavailable Private capability reached file admission.")

    monkeypatch.setattr(ProjectFileTextWatchSource, "validate", unexpected_file_admission)
    response = await create(state, private_required=True)
    assert response.status_code == 503, response.text
    with state.store._lock:
        assert state.store._conn.execute("SELECT COUNT(*) FROM app_watches").fetchone()[0] == 0


@pytest.mark.parametrize("scope", ["ordinary-project", "ordinary-file", "wrong-collection", "generated-collection"])
async def test_private_watch_rejects_incompatible_project_or_storage_scope(private_state, scope):
    state = private_state
    if scope == "ordinary-project":
        state.project = await state.store.projects.create(user_id=state.owner.user_id, name="Ordinary Project")
        await attach(state, "ordinary", b"Ordinary source.")
    else:
        changes = {
            "ordinary-file": {"private_required": 0},
            "wrong-collection": {"collection": "ordinary_text"},
            "generated-collection": {"origin": "generated", "collection": "ordinary_text", "chunks": 0},
        }[scope]
        await attach_private(state, "original", b"Protected source text.", **changes)
    response = await create(state, private_required=True)
    assert response.status_code in {404, 422}, response.text
    with state.store._lock:
        assert state.store._conn.execute("SELECT COUNT(*) FROM app_watches").fetchone()[0] == 0


@pytest.mark.parametrize("unavailable", ["pending", "ordinary", "wrong-collection", "reclaimed", "symlink"])
async def test_private_watch_does_not_fall_back_from_latest_unavailable_source(private_state, unavailable):
    state = private_state
    original_id, _ = await attach_private(state, "original", b"Original protected source.")
    created = await create(state, private_required=True)
    assert created.status_code == 201, created.text
    watch = created.json()
    assert (await check(state, watch))["outcome"] == "baseline"
    changes = {
        "ordinary": {"private_required": 0},
        "wrong-collection": {"collection": "ordinary_text"},
        "reclaimed": {"source_freed_at": state.clock[0].isoformat()},
    }.get(unavailable, {})
    _, path = await attach_private(
        state, "newest", b"Newest protected source.", status="pending" if unavailable == "pending" else "ready", **changes,
    )
    if unavailable == "symlink":
        target = path.with_name("outside-source.txt")
        target.write_text("Unapproved target.", encoding="utf-8")
        path.unlink()
        path.symlink_to(target)
    result = await check(state, watch)
    assert result["status"] == "failed" and result["outcome"] is None
    notices = (await state.client.get("/api/notifications")).json()["items"]
    assert all(notice["kind"] == "failure" for notice in notices)
    assert all(notice["title"] == "Private file watch" for notice in notices)
    with state.store._lock:
        baseline = state.store._conn.execute("SELECT file_id FROM app_watch_file_text_baselines").fetchone()
        assert baseline["file_id"] == original_id


@pytest.mark.parametrize("revocation", ["access", "disabled", "detach", "delete"])
async def test_private_watch_rechecks_authority_and_source_after_capture(private_state, monkeypatch, revocation):
    state = private_state
    original_id, _ = await attach_private(state, "original", b"Original protected source.")
    created = await create(state, private_required=True)
    assert created.status_code == 201, created.text
    watch = created.json()
    assert (await check(state, watch))["outcome"] == "baseline"
    newer_id, _ = await attach_private(state, "newer", b"Changed protected source.")
    capture = ProjectFileTextWatchSource.capture
    captured_once = []

    async def revoke_after_capture(reader, *args, **kwargs):
        captured = await capture(reader, *args, **kwargs)
        captured_once.append(captured.file_id)
        if revocation == "access":
            state.granted[0] = False
        elif revocation == "disabled":
            state.app.state.cfg.raw["private_processing"]["enabled"] = False
        elif revocation == "detach":
            await detach(state, newer_id)
        else:
            await state.uploads.request_file_deletion(
                file_id=newer_id, user=state.owner.storage_namespace, requested_at=state.clock[0].isoformat(),
            )
        return captured

    monkeypatch.setattr(ProjectFileTextWatchSource, "capture", revoke_after_capture)
    result = await check(state, watch)
    assert captured_once == [newer_id]
    assert result["status"] == "failed" and result["outcome"] is None
    notices = (await state.client.get("/api/notifications")).json()["items"]
    assert all(notice["kind"] == "failure" and notice["title"] == "Private file watch" for notice in notices)
    with state.store._lock:
        baseline = state.store._conn.execute("SELECT file_id FROM app_watch_file_text_baselines").fetchone()
        assert baseline["file_id"] == original_id
