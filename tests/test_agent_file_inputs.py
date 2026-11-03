"""Selected Project file pins survive restart and revoke every Agent boundary."""

from __future__ import annotations

import dataclasses
import uuid
from types import SimpleNamespace

import pytest

from audrey.agent_file_inputs import get_agent_file_inputs
from audrey.app_state import ApplicationStore
from audrey.app_state.agent_projects import token_count
from audrey.app_state.agents import (
    AgentTaskAccessError,
    AgentTaskConflictError,
    AgentTaskContextError,
    AgentTaskError,
)
from audrey.kb.extract import ocr_text_path
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user

STAMP = "2026-11-03T12:00:00+00:00"


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(provider="test", subject=name, email=f"{name}@example.com",
        display_name=name, role="user", auth_method="test") for name in ("alice", "bob")]
    project = await store.projects.create(user_id=owners[0].user_id, name="Task files")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    root = tmp_path / "files"
    root.mkdir()
    app = SimpleNamespace(state=SimpleNamespace(application_store=store, uploads_db=uploads,
        cfg=SimpleNamespace(raw={"kb": {"upload_root": str(root)}})))
    service = get_agent_file_inputs(app)
    yield SimpleNamespace(store=store, uploads=uploads, app=app, root=root, service=service,
                          owner=owners[0], other=owners[1], project=project)
    uploads.close()
    store.close()


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def file(state, *, kind="text", private=False, owner=None, project=None, origin="upload", text="SELECTED FILE BODY"):
    owner, project = owner or state.owner, project or state.project
    fid = str(uuid.uuid4())
    suffix = {"text": ".txt", "image": ".png", "audio": ".mp3", "video": ".mp4"}[kind]
    directory = state.root / sanitize_user(owner.storage_namespace)
    directory.mkdir(exist_ok=True)
    path = directory / (fid + suffix)
    path.write_bytes(text.encode())
    if kind == "image" or origin == "generated":
        ocr_text_path(path).write_text(text)
    if kind in {"audio", "video"}:
        (directory / (fid + ".transcript.txt")).write_text(text)
    with state.uploads._lock:
        state.uploads._record_locked(fid, owner.storage_namespace, "source" + suffix, "text/plain", len(text.encode()),
            kind, "", 0, STAMP, "ready", private_required=private, origin=origin)
    if private:
        # Existing Private canonical relation; this fixture does not invoke the
        # separate verified Private upload/model pipeline.
        with state.store._lock, state.store._conn:
            state.store._conn.execute("INSERT INTO app_project_files(project_id,file_id,added_at,origin) VALUES (?,?,?,?)",
                (project.project_id, fid, STAMP, origin))
    else:
        await state.store.projects.add_file(user_id=owner.user_id, project_id=project.project_id, file_id=fid,
            file_is_private=private, file_origin=origin)
    return fid, path


async def create(state, ids, *, request_id=None):
    captured = await state.service.capture(state.owner, state.project.project_id, ids)
    return await state.store.agents.create(user_id=state.owner.user_id, objective="Prepare a task deliverable.",
        request_id=request_id or uuid.uuid4().hex, project_id=state.project.project_id, file_inputs=captured)


@pytest.mark.parametrize(("kind", "origin", "artifact"), [("text", "upload", "text"), ("image", "upload", "text"),
    ("audio", "upload", "transcript"), ("video", "upload", "transcript"), ("text", "generated", "text")])
async def test_explicit_saved_artifact_prefix_metadata_and_no_body_manifest(state, kind, origin, artifact):
    fid, _ = await file(state, kind=kind, origin=origin)
    await file(state, text="UNSELECTED SECRET BODY")
    task = await create(state, [fid])
    assert task["file_inputs"] == {"current": True, "files": [{"file_id": fid, "filename": "source" + {
        "text": ".txt", "image": ".png", "audio": ".mp3", "video": ".mp4"}[kind], "artifact": artifact,
        "chars": 18, "total_chars": 18, "truncated": False, "origin": origin}]}
    pin_json = state.store._conn.execute("SELECT pin_json FROM app_agent_file_inputs").fetchone()[0]
    assert "SELECTED FILE BODY" not in pin_json and "UNSELECTED" not in pin_json
    assert str(state.root) not in pin_json
    claimed = await state.store.agents.claim()
    resolved = await state.service.resolve(claimed, lease(claimed))
    assert resolved[0]["text"] == "SELECTED FILE BODY" and len(resolved) == 1
    assert state.store.schema_version == 37


async def test_prefix_is_per_file_token_bounded_and_visible_as_partial(state):
    ids = [(await file(state, text="Long source text. " * 2000))[0] for _ in range(3)]
    task = await create(state, ids)
    claimed = await state.store.agents.claim()
    resolved = await state.service.resolve(claimed, lease(claimed))
    assert len(resolved) == 3 and all(token_count(row["text"]) <= 1500 for row in resolved)
    assert all(row["truncated"] and row["chars"] < row["total_chars"] for row in task["file_inputs"]["files"])


async def test_restart_rereads_saved_text_and_durable_metadata_proofs(state):
    fid, _ = await file(state)
    task = await create(state, [fid])
    second = ApplicationStore(state.store.path)
    try:
        # Missing runtime binding fails closed; binding restores unchanged pins.
        assert not (await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["file_inputs"]["current"]
        app = SimpleNamespace(state=SimpleNamespace(application_store=second, uploads_db=state.uploads, cfg=state.app.state.cfg))
        service = get_agent_file_inputs(app)
        assert await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == task
        claimed = await second.agents.claim()
        assert (await service.resolve(claimed, lease(claimed)))[0]["text"] == "SELECTED FILE BODY"
    finally:
        second.close()


@pytest.mark.parametrize("change", ["text", "delete", "detach", "catalog", "project", "namespace", "owner"])
async def test_changes_fence_lease_operations_checkpoints_outputs_and_response(state, change):
    fid, path = await file(state)
    task = await create(state, [fid])
    claimed = await state.store.agents.claim()
    scope = lease(claimed)
    if change == "text":
        path.write_text("Changed source body")
    elif change == "delete":
        await state.uploads.request_file_deletion(fid, user=state.owner.storage_namespace, requested_at=STAMP)
    elif change == "detach":
        await state.store.projects.remove_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=fid)
    elif change == "project":
        await state.store.projects.delete(user_id=state.owner.user_id, project_id=state.project.project_id)
    elif change == "catalog":
        with state.uploads._lock:
            state.uploads._conn.execute("UPDATE uploads SET filename='changed.txt' WHERE file_id=?", (fid,))
    else:
        with state.store._lock, state.store._conn:
            state.store._conn.execute("UPDATE app_users SET storage_namespace='changed' WHERE user_id=?" if change == "namespace"
                else "UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
    expected = AgentTaskAccessError if change == "owner" else AgentTaskContextError
    for call in (state.store.agents.validate_lease(**scope), state.store.agents.reserve_operation(**scope, step_id="plan"),
                 state.store.agents.checkpoint(**scope, step_id="plan", result="late"),
                 state.store.agents.complete(**scope, result="late"),
                 state.store.agents.reserve_output(**scope, file_id=str(uuid.uuid4()), filename="late.txt", identity_sha256="a" * 64)):
        with pytest.raises(expected):
            await call
    with pytest.raises(expected):
        state.store.agents.fence_execution(**scope)
    with pytest.raises(AgentTaskAccessError if change == "owner" else AgentTaskConflictError):
        state.store.agents.fence_response(user_id=state.owner.user_id, tasks=[task])
    if change != "owner":
        assert not (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["file_inputs"]["current"]


async def test_unrelated_project_memberships_renames_and_files_preserve_pins(state):
    fid, _ = await file(state)
    task = await create(state, [fid])
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Other")
    await state.store.projects.add_file(user_id=state.owner.user_id, project_id=project.project_id, file_id=fid)
    await state.store.projects.update(user_id=state.owner.user_id, project_id=state.project.project_id, name="Renamed")
    await file(state, text="Another unselected file")
    assert (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["file_inputs"]["current"]
    claimed = await state.store.agents.claim()
    assert (await state.service.resolve(claimed, lease(claimed)))[0]["text"] == "SELECTED FILE BODY"


async def test_retained_ocr_or_transcript_available_without_original(state):
    for kind in ("image", "video"):
        fid, path = await file(state, kind=kind)
        path.unlink()
        task = await create(state, [fid])
        assert task["file_inputs"]["current"]


@pytest.mark.parametrize("invalid", ["foreign", "private", "other_project", "missing", "unreadable", "oversize", "symlink"])
async def test_unavailable_inputs_refused_without_task_or_body_writes(state, invalid):
    owner, project, private = state.owner, state.project, False
    if invalid == "foreign":
        owner = state.other
        project = await state.store.projects.create(user_id=owner.user_id, name="Foreign")
    elif invalid in {"private", "other_project"}:
        private = invalid == "private"
        project = await state.store.projects.create(user_id=owner.user_id, name="Other", private_required=private)
    fid, path = await file(state, owner=owner, project=project, private=private)
    if invalid == "missing":
        fid = "missing"
    elif invalid == "unreadable":
        path.write_text("")
    elif invalid == "oversize":
        path.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    elif invalid == "symlink":
        target = state.root / "secret.txt"
        target.write_text("EXTERNAL SECRET")
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(AgentTaskError):
        await create(state, [fid])
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_tasks").fetchone()[0] == 0


async def test_capture_and_transaction_races_refuse_changed_pins_and_roll_back_task(state):
    fid, path = await file(state)
    captured = await state.service.capture(state.owner, state.project.project_id, [fid])
    path.write_text("changed after preview")
    with pytest.raises(AgentTaskContextError):
        await state.store.agents.create(user_id=state.owner.user_id, objective="task", request_id="one",
            project_id=state.project.project_id, file_inputs=captured)
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_tasks").fetchone()[0] == 0
    assert not state.store._conn.in_transaction


async def test_canonical_file_set_idempotency_summaries_cleanup_and_internal_types(state):
    ids = [(await file(state))[0] for _ in range(2)]
    first = await create(state, ids, request_id="same")
    assert await create(state, list(reversed(ids)), request_id="same") == first
    with pytest.raises(AgentTaskConflictError):
        await create(state, ids[:1], request_id="same")
    summary = (await state.store.agents.list(user_id=state.owner.user_id, summary=True))[0]
    assert summary["file_inputs"] == first["file_inputs"]
    state.store.agents.fence_response(user_id=state.owner.user_id, tasks=[summary])
    captured = await state.service.capture(state.owner, state.project.project_id, ids)
    for invalid in ({"pins": []}, dataclasses.replace(captured, owner_id=state.other.user_id)):
        with pytest.raises(AgentTaskError):
            await state.store.agents.create(user_id=state.owner.user_id, objective="task", request_id=uuid.uuid4().hex,
                project_id=state.project.project_id, file_inputs=invalid)
    task = await state.store.agents.control(user_id=state.owner.user_id, task_id=first["id"], expected_revision=first["revision"], action="cancel")
    await state.store.agents.delete(user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"])
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_file_inputs").fetchone()[0] == 0
    assert await state.uploads.get_upload(ids[0]) is not None


async def test_restarted_text_digest_detects_same_size_timestamp_preserving_replacement(state):
    import os

    fid, path = await file(state)
    await create(state, [fid])
    before = path.stat()
    path.write_text("ALTERED FILE BODY ")
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    claimed = await state.store.agents.claim()
    # Stat-only hot fences intentionally stay bounded; the actual step read
    # always verifies the full canonical text digest before any provider call.
    with pytest.raises(AgentTaskContextError):
        await state.service.resolve(claimed, lease(claimed))


def test_missing_storage_binding_is_compatible_for_legacy_tasks():
    assert get_agent_file_inputs(SimpleNamespace(state=SimpleNamespace())) is None


@pytest.mark.parametrize("ids", [["same", "same"], ["a", "b", "c", "d"], ["../escape"], [None], "file-id"])
async def test_explicit_selection_shape_and_three_file_limit(state, ids):
    with pytest.raises(AgentTaskError):
        await state.service.capture(state.owner, state.project.project_id, ids)


async def test_saved_parser_failures_and_changes_during_read_return_safe_refusals(state, monkeypatch):
    fid, path = await file(state)

    def failed(*_args):
        raise RuntimeError("SECRET PARSER DETAIL /private/source/path")

    original = state.service.reader._read_file
    monkeypatch.setattr(state.service.reader, "_read_file", failed)
    with pytest.raises(AgentTaskError) as error:
        await create(state, [fid])
    assert "SECRET" not in str(error.value) and "/private" not in str(error.value)

    def changed(*args):
        result = original(*args)
        path.write_text("A concurrently replaced source")
        return result

    monkeypatch.setattr(state.service.reader, "_read_file", changed)
    with pytest.raises(AgentTaskContextError):
        await create(state, [fid])
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_tasks").fetchone()[0] == 0


async def test_hot_fences_do_not_parse_text_and_nested_publication_reads_use_owned_upload_lock(state, monkeypatch):
    fid, _ = await file(state)
    await create(state, [fid])
    claimed = await state.store.agents.claim()

    def forbidden(*_args):
        raise AssertionError("A metadata fence must not parse source text")

    monkeypatch.setattr(state.service.reader, "_read_file", forbidden)
    state.store.agents.fence_execution(**lease(claimed))
    with state.store._lock, state.uploads._lock, state.service.catalog_lock_already_held():
        state.store.agents.fence_execution(**lease(claimed))
    # The explicit nested-read permission ends with its thread-local scope.
    assert not getattr(state.service._catalog_lock_state, "held", False)


async def test_explicit_detach_and_reattach_does_not_restore_frozen_membership(state):
    fid, _ = await file(state)
    task = await create(state, [fid])
    await state.store.projects.remove_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=fid)
    await state.store.projects.add_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=fid)
    assert not (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["file_inputs"]["current"]
    fresh = await create(state, [fid])
    assert fresh["file_inputs"]["current"]


async def test_cached_service_rebinds_replaced_store_and_catalog(state):
    previous = state.service
    replacement = ApplicationStore(state.store.path)
    replacement_uploads = UploadsDB(state.uploads._path)
    try:
        state.app.state.application_store = replacement
        state.app.state.uploads_db = replacement_uploads
        current = get_agent_file_inputs(state.app)
        assert current is not previous
        assert current.store is replacement and current.uploads is replacement_uploads
        assert replacement.agents._file_input_service is current
    finally:
        replacement.close()
        replacement_uploads.close()
