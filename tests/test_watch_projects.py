"""Project links organize public watches without widening their source authority."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.app_state.watches import WatchAccessError, WatchConflictError
from audrey.auth import require_principal
from audrey.routes.app import router
from audrey.watch_executor import WatchExecutor
from audrey.watch_runtime import WatchRuntime


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    project = await store.projects.create(user_id=owners[0].user_id, name="Public releases",
                                          instructions="Private instruction body must never be copied.")
    other = await store.projects.create(user_id=owners[1].user_id, name="Other owner")
    private = await store.projects.create(user_id=owners[0].user_id, name="Protected", private_required=True)
    current = [owners[0]]
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}})
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: current[0]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, owner=owners[0], other=owners[1], project=project,
                              foreign=other, private=private, current=current, client=client, app=app)
    store.close()


def inputs(state, **changes):
    return {"user_id": state.owner.user_id, "title": "Release notes", "url": "https://example.com/releases",
            "criterion": "New stable releases.", "request_id": "watch-1", "project_id": state.project.project_id,
            "schedule": {"kind": "interval", "timezone": "UTC", "interval_minutes": 60}, **changes}


def key(state, watch, **changes):
    return {"user_id": state.owner.user_id, "watch_id": watch["id"], "expected_revision": watch["revision"], **changes}


def lease(run):
    return {name: run[name] for name in ("run_id", "lease_token", "fence")}


async def claim(state, watch):
    await state.store.watches.run_now(**key(state, watch))
    return await state.store.watches.claim()


def result(run):
    text = "Version one."
    return {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(), "outcome": "baseline",
            "significant": None, "report": "Initial public page baseline.", "diff": "",
            "final_url": run["url"], "extraction_version": "test-v1"}


async def delete_project(state):
    await state.store.projects.delete(user_id=state.owner.user_id, project_id=state.project.project_id)


def attachment_watch_body(state, **changes):
    return {
        "title": "Project file changes", "source_kind": "project_files", "enabled": False,
        "project_id": state.project.project_id, "request_id": "attachments",
        "schedule": {"kind": "interval", "timezone": "UTC", "interval_minutes": 60}, **changes,
    }


async def test_project_attachment_watch_reports_additions_and_removals_via_saved_checks(state, monkeypatch):
    async def unexpected_model(*_args, **_kwargs):
        pytest.fail("Project attachment Watch must run independently of model selection.")

    monkeypatch.setattr("audrey.watch_executor.resolve_model", unexpected_model)
    created = await state.client.post("/api/watches", json=attachment_watch_body(state))
    assert created.status_code == 201, created.text
    watch = created.json()
    assert watch["source_kind"] == "project_files" and watch["url"] is None and watch["active_run_id"] is None
    replayed = await state.client.post("/api/watches", json=attachment_watch_body(state))
    assert replayed.json()["id"] == watch["id"]

    executor = WatchExecutor(state.app, resolver=unexpected_model, fetcher=unexpected_model)
    runtime = WatchRuntime(state.store.watches, executor, project_files_executor=executor.complete_project_files)

    async def run_check(action="run"):
        queued = await state.client.post(f"/api/watches/{watch['id']}/actions", json={
            "action": action, "expected_revision": watch["revision"],
        })
        assert queued.status_code == 200, queued.text
        assert not queued.json()["enabled"] and queued.json()["next_run_at"] is None
        assert await runtime.run_once()
        path = f"/api/watches/{watch['id']}/runs/{queued.json()['active_run_id']}"
        response = await state.client.get(path)
        assert response.status_code == 200 and response.json()["status"] == "completed", response.text
        assert (await state.client.get(f"{path}/snapshot")).status_code == 404
        return response.json()

    baseline = await run_check()
    assert baseline["outcome"] == "baseline" and baseline["definition"]["source_kind"] == "project_files"
    assert (await run_check())["outcome"] == "unchanged"

    attachment = {"user_id": state.owner.user_id, "project_id": state.project.project_id, "file_id": "attachment-test-id"}
    await state.store.projects.add_file(**attachment)
    added = await run_check()
    assert added["outcome"] == "changed" and added["significant"] is True and "Added: 1. Removed: 0." in added["report"]
    notices = (await state.client.get("/api/notifications")).json()["items"]
    assert len(notices) == 1 and notices[0]["run_id"] == added["id"]
    assert (await run_check())["outcome"] == "unchanged"
    assert len((await state.client.get("/api/notifications")).json()["items"]) == 1

    assert await state.store.projects.remove_file(**attachment)
    removed = await run_check()
    assert removed["outcome"] == "changed" and removed["significant"] is True and "Added: 0. Removed: 1." in removed["report"]
    reset = await run_check("rebaseline")
    assert reset["outcome"] == "baseline" and reset["rebaseline"]
    assert len((await state.client.get("/api/notifications")).json()["items"]) == 2
    history = await state.client.get(f"/api/watches/{watch['id']}/runs")
    assert attachment["file_id"] not in history.text


async def test_project_attachment_watch_rejects_foreign_private_and_changed_sources(state):
    for project_id in (state.foreign.project_id, state.private.project_id):
        response = await state.client.post("/api/watches", json=attachment_watch_body(state, project_id=project_id))
        assert response.status_code == 404
    for changes in ({"project_id": None}, {"url": "https://example.com"}, {"criterion": "unrelated scope"}):
        response = await state.client.post("/api/watches", json=attachment_watch_body(state, **changes))
        assert response.status_code == 422

    created = await state.client.post("/api/watches", json=attachment_watch_body(state))
    assert created.status_code == 201
    watch = created.json()
    path = f"/api/watches/{watch['id']}"
    for changes in ({"project_id": None}, {"url": "https://example.com"}, {"criterion": "unrelated scope"}):
        response = await state.client.patch(path, json={"expected_revision": watch["revision"], **changes})
        assert response.status_code == 422
    state.current[0] = state.other
    assert (await state.client.get(path)).status_code == 404
    state.current[0] = state.owner

    queued = await state.client.post(f"{path}/actions", json={"action": "run", "expected_revision": watch["revision"]})
    assert queued.status_code == 200
    run = await state.store.watches.claim()
    await delete_project(state)
    with pytest.raises((WatchAccessError, WatchConflictError)):
        await state.store.watches.complete_project_files(**lease(run))
    assert (await state.client.get(path)).json()["project"]["available"] is False
    blocked = await state.client.post(f"{path}/actions", json={"action": "run", "expected_revision": watch["revision"]})
    assert blocked.status_code in {404, 409}


async def test_owned_project_create_filter_and_idempotency_do_not_dispatch_or_copy_context(state):
    watch = await state.store.watches.create(**inputs(state))
    assert watch["project"] == {"id": state.project.project_id, "name": state.project.name, "available": True}
    assert watch["active_run_id"] is None
    assert await state.store.watches.claim() is None
    assert await state.store.watches.create(**inputs(state)) == watch
    with pytest.raises(WatchConflictError):
        await state.store.watches.create(**inputs(state, project_id=None))
    unscoped = await state.store.watches.create(**inputs(state, request_id="unscoped", project_id=None))
    assert unscoped["project"] is None
    assert await state.store.watches.list(user_id=state.owner.user_id, project_id=state.project.project_id) == [watch]
    assert {item["id"] for item in await state.store.watches.list(user_id=state.owner.user_id)} == {watch["id"], unscoped["id"]}
    run = await claim(state, watch)
    definition = json.loads(state.store._conn.execute(
        "SELECT definition_json FROM app_watch_runs WHERE run_id=?", (run["id"],),
    ).fetchone()[0])
    assert definition == {"title": watch["title"], "url": watch["url"], "criterion": watch["criterion"],
                          "schedule": watch["schedule"], "project_id": state.project.project_id}
    assert state.project.name not in json.dumps(definition) and state.project.instructions not in json.dumps(run)
    assert "project_id" not in run["definition"]
    with state.store._lock, pytest.raises(sqlite3.IntegrityError, match="immutable"):
        state.store._conn.execute("UPDATE app_watch_runs SET definition_json='{}'")
    state.store._conn.rollback()


@pytest.mark.parametrize("target", ["foreign", "private", "missing"])
async def test_unavailable_project_create_update_filter_and_api_are_owner_bound(state, target):
    project_id = getattr(state, target).project_id if target != "missing" else "missing"
    watch = await state.store.watches.create(**inputs(state, project_id=None))
    with pytest.raises(WatchAccessError):
        await state.store.watches.create(**inputs(state, request_id="new", project_id=project_id))
    with pytest.raises(WatchAccessError):
        await state.store.watches.update(**key(state, watch), project_id=project_id)
    with pytest.raises(WatchAccessError):
        await state.store.watches.list(user_id=state.owner.user_id, project_id=project_id)
    body = inputs(state, request_id="api", project_id=project_id)
    body.pop("user_id")
    assert (await state.client.post("/api/watches", json=body)).status_code == 404
    assert (await state.client.get("/api/watches", params={"project_id": project_id})).status_code == 404
    assert (await state.client.patch(f"/api/watches/{watch['id']}", json={
        "expected_revision": watch["revision"], "project_id": project_id,
    })).status_code == 404


async def test_project_reassignment_preserves_url_baseline_and_immutable_run_scope(state):
    watch = await state.store.watches.create(**inputs(state))
    initial = await claim(state, watch)
    await state.store.watches.reserve_operation(**lease(initial), kind="fetch")
    await state.store.watches.complete(**lease(initial), **result(initial))
    live = await claim(state, watch)
    next_project = await state.store.projects.create(user_id=state.owner.user_id, name="Another public Project")
    edited = await state.store.watches.update(**key(state, watch), project_id=next_project.project_id)
    assert edited["project"]["id"] == next_project.project_id
    assert edited["latest_run"]["status"] == "cancelled"
    with pytest.raises(WatchConflictError):
        await state.store.watches.validate_lease(**lease(live))
    new_run = await claim(state, edited)
    assert new_run["baseline"]["run_id"] == initial["id"]
    assert new_run["definition"]["project"]["id"] == next_project.project_id
    saved = await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=initial["id"])
    assert saved["definition"]["project"]["id"] == state.project.project_id
    await delete_project(state)
    saved = await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=initial["id"])
    assert saved["definition"]["project"] == {"id": state.project.project_id, "name": None, "available": False}
    assert await state.store.watches.validate_lease(**lease(new_run))


@pytest.mark.parametrize("running", [False, True])
async def test_claim_sweep_pauses_missing_project_and_cancels_queued_or_running_work(state, running):
    watch = await state.store.watches.create(**inputs(state))
    queued = await state.store.watches.run_now(**key(state, watch))
    run = await state.store.watches.claim() if running else None
    await delete_project(state)
    assert await state.store.watches.claim() is None
    saved = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    assert not saved["enabled"] and saved["next_run_at"] is None and saved["active_run_id"] is None
    assert saved["latest_run"]["id"] == queued["active_run_id"] and saved["latest_run"]["status"] == "cancelled"
    assert saved["revision"] == watch["revision"] + 1
    if run:
        with pytest.raises(WatchConflictError):
            await state.store.watches.validate_lease(**lease(run))
    for call in (
        state.store.watches.run_now(**key(state, saved)),
        state.store.watches.run_now(**key(state, saved), rebaseline=True),
        state.store.watches.update(**key(state, saved), enabled=True),
    ):
        with pytest.raises(WatchAccessError):
            await call
    # A saved stale link may be explicitly removed and then resumed.
    unlinked = await state.store.watches.update(**key(state, saved), project_id=None, enabled=True)
    assert unlinked["project"] is None and unlinked["enabled"]
    assert (await claim(state, unlinked))["definition"]["project"] is None


@pytest.mark.parametrize("change", ["delete", "private"])
async def test_project_unavailability_immediately_fences_dispatch_heartbeat_and_commit_with_cleanup(state, change):
    watch = await state.store.watches.create(**inputs(state))
    run = await claim(state, watch)
    await state.store.watches.reserve_operation(**lease(run), kind="fetch")
    if change == "delete":
        await delete_project(state)
    else:
        with state.store._lock, state.store._conn:
            state.store._conn.execute("UPDATE app_projects SET private_required=1 WHERE project_id=?", (state.project.project_id,))
    for call in (
        state.store.watches.validate_lease(**lease(run)),
        state.store.watches.heartbeat(**lease(run)),
        state.store.watches.reserve_operation(**lease(run), kind="model"),
        state.store.watches.complete(**lease(run), **result(run)),
    ):
        with pytest.raises(WatchAccessError):
            await call
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_execution(**lease(run))
    failed = await state.store.watches.fail(**lease(run), error="Project unavailable.", retry=True)
    assert failed["status"] == "failed" and failed["definition"]["project"]["name"] is None
    assert await state.store.watches.delete(**key(state, watch))
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_watch_runs").fetchone()[0] == 0


async def test_rename_and_instruction_edits_rehydrate_metadata_without_changing_execution_scope(state):
    watch = await state.store.watches.create(**inputs(state))
    run = await claim(state, watch)
    await state.store.projects.update(user_id=state.owner.user_id, project_id=state.project.project_id,
                                      name="Renamed", instructions="Changed local instructions.")
    assert await state.store.watches.validate_lease(**lease(run))
    saved = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    assert saved["project"]["name"] == saved["latest_run"]["definition"]["project"]["name"] == "Renamed"
    for payloads in ({"watches": [watch]}, {"runs": [run]}):
        with pytest.raises(WatchConflictError):
            state.store.watches.fence_response(user_id=state.owner.user_id, **payloads)


async def test_final_response_fences_saved_run_project_even_after_watch_reassignment(state):
    watch = await state.store.watches.create(**inputs(state))
    run = await claim(state, watch)
    await state.store.watches.fail(**lease(run), error="Failed source.")
    unlinked = await state.store.watches.update(**key(state, watch), project_id=None)
    old_run = await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=run["id"])
    await delete_project(state)
    with pytest.raises(WatchConflictError):
        state.store.watches.fence_response(user_id=state.owner.user_id, runs=[old_run])
    with pytest.raises(WatchConflictError):
        state.store.watches.fence_response(user_id=state.owner.user_id, watches=[unlinked])
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_response(user_id=state.owner.user_id, project_id=state.project.project_id)


async def test_api_exposes_owned_project_filter_and_explicit_null_unlink(state):
    body = inputs(state)
    body.pop("user_id")
    created = await state.client.post("/api/watches", json=body)
    assert created.status_code == 201, created.text
    watch = created.json()
    listing = await state.client.get("/api/watches", params={"project_id": state.project.project_id})
    assert listing.json() == {"items": [watch]}
    assert listing.headers["cache-control"] == "private, no-store"
    state.current[0] = state.other
    assert (await state.client.get("/api/watches", params={"project_id": state.project.project_id})).status_code == 404
    state.current[0] = state.owner
    queued = await state.client.post(f"/api/watches/{watch['id']}/actions", json={
        "action": "run", "expected_revision": watch["revision"],
    })
    run_id = queued.json()["active_run_id"]
    unlinked = await state.client.patch(f"/api/watches/{watch['id']}", json={
        "expected_revision": watch["revision"], "project_id": None,
    })
    assert unlinked.status_code == 200 and unlinked.json()["project"] is None
    saved = await state.client.get(f"/api/watches/{watch['id']}/runs/{run_id}")
    assert saved.status_code == 200 and saved.json()["definition"]["project"]["id"] == state.project.project_id
    assert (await state.client.get("/api/watches", params={"project_id": state.project.project_id})).json() == {"items": []}


async def test_account_purge_erases_linked_watch_derivatives(state):
    watch = await state.store.watches.create(**inputs(state))
    run = await claim(state, watch)
    await state.store.watches.reserve_operation(**lease(run), kind="fetch")
    await state.store.watches.complete(**lease(run), **result(run))
    await state.store.purge_local_user_data(user_id=state.owner.user_id)
    for table in ("app_watches", "app_watch_runs", "app_watch_snapshots", "app_watch_baselines", "app_watch_operations"):
        assert state.store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # noqa: S608


async def test_due_schedule_never_enqueues_new_work_after_project_deletion(state):
    watch = await state.store.watches.create(**inputs(state))
    with state.store._lock, state.store._conn:
        state.store._conn.execute("UPDATE app_watches SET next_run_at='2000-01-01T00:00:00+00:00' WHERE watch_id=?", (watch["id"],))
    await delete_project(state)
    assert await state.store.watches.claim() is None
    assert await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"]) == []
    saved = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    assert not saved["enabled"] and saved["next_run_at"] is None


async def test_schema33_upgrade_preserves_legacy_unscoped_request_digest_and_run_definition(state):
    watch = await state.store.watches.create(**inputs(state, project_id=None))
    queued = await state.store.watches.run_now(**key(state, watch))
    with state.store._lock, state.store._conn:
        trigger = state.store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='trg_app_watch_run_definition_immutable'",
        ).fetchone()[0]
        state.store._conn.execute("DROP TRIGGER trg_app_watch_run_definition_immutable")
        definition = {name: watch[name] for name in ("title", "url", "criterion", "schedule")}
        state.store._conn.execute("UPDATE app_watch_runs SET definition_json=?", (json.dumps(definition),))
        state.store._conn.execute(trigger)
        state.store._conn.execute("DROP INDEX idx_app_watches_project")
        state.store._conn.execute("ALTER TABLE app_watches DROP COLUMN project_id")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version=33")
    upgraded = ApplicationStore(state.store.path)
    try:
        assert upgraded.schema_version == 47
        saved = await upgraded.watches.create(**inputs(state, project_id=None))
        assert saved["id"] == watch["id"] and saved["project"] is None
        assert saved["active_run_id"] == queued["active_run_id"] and saved["latest_run"]["definition"]["project"] is None
        run = await upgraded.watches.claim()
        assert run["definition"]["project"] is None and run["rebaseline"] is False
    finally:
        upgraded.close()


async def test_project_deletion_after_await_rejects_watch_and_run_response_bodies(state, monkeypatch):
    watch = await state.store.watches.create(**inputs(state))
    run = await claim(state, watch)
    await state.store.watches.fail(**lease(run), error="Unavailable public page.")
    get = state.store.watches.get_run

    async def erase_after_read(**kwargs):
        payload = await get(**kwargs)
        await delete_project(state)
        return payload

    monkeypatch.setattr(state.store.watches, "get_run", erase_after_read)
    response = await state.client.get(f"/api/watches/{watch['id']}/runs/{run['id']}")
    assert response.status_code == 409 and "definition" not in response.text


async def test_empty_project_filter_is_revalidated_after_await(state, monkeypatch):
    listing = state.store.watches.list

    async def erase_after_read(**kwargs):
        payload = await listing(**kwargs)
        await delete_project(state)
        return payload

    monkeypatch.setattr(state.store.watches, "list", erase_after_read)
    response = await state.client.get("/api/watches", params={"project_id": state.project.project_id})
    assert response.status_code == 404
