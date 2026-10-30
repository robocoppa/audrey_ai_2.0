"""Native watch APIs bind explicit schedules, history and notices to their owner."""

from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.routes.app import router
from audrey.watch_executor import evidence_diff
from audrey.watch_runtime import WatchRuntime


@pytest.fixture
async def watches(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    current = [owners[0]]
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}})
    wakes = []
    app.state.watch_runtime = SimpleNamespace(wake=lambda: wakes.append(True))
    app.include_router(router)

    def authenticate():
        if current[0] is None:
            raise HTTPException(401, "Sign in required.")
        return current[0]

    app.dependency_overrides[require_principal] = authenticate
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, owners=owners, current=current, app=app, client=client, wakes=wakes)
    store.close()


async def _create(state, **changes):
    return await state.client.post("/api/watches", json={
        "title": "Release notes", "url": "https://example.com/releases",
        "criterion": "Changes affecting local model support.",
        "schedule": {"kind": "daily", "time": "09:00", "timezone": "America/Denver"},
        "request_id": "create-1", **changes,
    })


async def _action(state, watch, action, **changes):
    return await state.client.post(f"/api/watches/{watch['id']}/actions", json={
        "action": action, "expected_revision": watch["revision"], **changes,
    })


async def test_create_schedules_future_without_running_and_retries_idempotently(watches):
    created = await _create(watches)
    assert created.status_code == 201, created.text
    watch = created.json()
    assert watch["enabled"] and watch["active_run_id"] is None and watch["latest_run"] is None
    assert dt.datetime.fromisoformat(watch["next_run_at"]) > dt.datetime.now(dt.UTC)
    assert not {"user_id", "limits", "request_id", "baseline", "fence", "lease_token"} & watch.keys()
    assert (await _create(watches)).json() == watch
    assert (await _create(watches, criterion="Different scope")).status_code == 409
    listing = await watches.client.get("/api/watches")
    assert listing.status_code == 200 and listing.json() == {"items": [watch]}
    assert listing.headers["cache-control"] == "private, no-store"
    assert (await watches.client.get(f"/api/watches/{watch['id']}/runs")).json() == {"items": []}


async def test_run_now_is_explicit_deduplicated_and_does_not_enable_paused_schedule(watches):
    watch = (await _create(watches, enabled=False)).json()
    queued = await _action(watches, watch, "run")
    assert queued.status_code == 200, queued.text
    queued = queued.json()
    assert queued["enabled"] is False and queued["next_run_at"] is None
    assert queued["active_run_id"] and queued["revision"] == watch["revision"]
    assert (await _action(watches, watch, "run")).json()["active_run_id"] == queued["active_run_id"]
    runs = (await watches.client.get(f"/api/watches/{watch['id']}/runs")).json()["items"]
    assert len(runs) == 1 and runs[0]["kind"] == "manual" and runs[0]["status"] == "queued"
    detail = await watches.client.get(f"/api/watches/{watch['id']}/runs/{runs[0]['id']}")
    assert detail.json() == runs[0]
    assert not {"snapshot", "text", "baseline", "lease_token", "fence", "operations"} & detail.json().keys()


async def test_pause_edit_conflict_and_delete_fence_queued_work(watches):
    watch = (await _create(watches)).json()
    await _action(watches, watch, "run")
    paused = (await _action(watches, watch, "pause")).json()
    assert paused["enabled"] is False and paused["active_run_id"] is None
    assert paused["revision"] > watch["revision"]
    assert (await _action(watches, watch, "resume")).status_code == 409
    path = f"/api/watches/{watch['id']}"
    stale = await watches.client.patch(path, json={"expected_revision": watch["revision"], "title": "New title"})
    assert stale.status_code == 409
    updated = await watches.client.patch(path, json={"expected_revision": paused["revision"], "title": "New title"})
    assert updated.status_code == 200 and updated.json()["title"] == "New title"
    assert (await watches.client.delete(path, params={"expected_revision": paused["revision"]})).status_code == 409
    assert (await watches.client.delete(path, params={"expected_revision": updated.json()["revision"]})).json() == {"deleted": True}
    assert (await watches.client.get(path)).status_code == 404


async def test_foreign_and_missing_watch_run_and_notice_are_indistinguishable(watches):
    watch = (await _create(watches)).json()
    queued = (await _action(watches, watch, "run")).json()
    watches.current[0] = watches.owners[1]
    assert (await watches.client.get("/api/watches")).json() == {"items": []}
    assert (await watches.client.get("/api/notifications")).json() == {"items": [], "unread_count": 0}
    for target in (watch["id"], "missing"):
        path = f"/api/watches/{target}"
        assert (await watches.client.get(path)).status_code == 404
        assert (await watches.client.get(f"{path}/runs")).status_code == 404
        assert (await watches.client.get(f"{path}/runs/{queued['active_run_id']}")).status_code == 404
        assert (await watches.client.patch(path, json={"expected_revision": watch["revision"], "title": "Foreign"})).status_code == 404
        assert (await watches.client.post(f"{path}/actions", json={"expected_revision": watch["revision"], "action": "run"})).status_code == 404
        assert (await watches.client.delete(path, params={"expected_revision": watch["revision"]})).status_code == 404
    assert (await watches.client.post("/api/notifications/missing/read")).status_code == 404


async def test_disabled_execution_preserves_saved_reads_pause_and_delete(watches):
    watch = (await _create(watches)).json()
    watches.app.state.cfg.raw["watches"] = {"enabled": False}
    assert (await _create(watches, request_id="disabled")).status_code == 503
    assert (await _action(watches, watch, "run")).status_code == 503
    assert (await watches.client.get(f"/api/watches/{watch['id']}")).status_code == 200
    paused = await _action(watches, watch, "pause")
    assert paused.status_code == 200
    assert (await _action(watches, paused.json(), "resume")).status_code == 503
    assert (await watches.client.delete(f"/api/watches/{watch['id']}", params={"expected_revision": paused.json()["revision"]})).status_code == 200


async def test_bots_group_blocks_execution_but_preserves_owned_reads_pause_and_delete(watches):
    watch = (await _create(watches)).json()
    queued = (await _action(watches, watch, "run")).json()
    administrator = await watches.store.resolve_external_identity(
        provider="test", subject="operator", email="operator@example.com", display_name="Operator",
        role="admin", auth_method="test",
    )
    await watches.store.admin_update_user(
        actor_user_id=administrator.user_id,
        target_user_id=watches.owners[0].user_id,
        groups=["users", "bots"],
    )
    principal = await watches.store.get_principal(user_id=watches.owners[0].user_id)
    assert principal.role == "user" and principal.groups == frozenset({"users", "bots"})
    watches.current[0] = principal
    wakes = len(watches.wakes)
    path = f"/api/watches/{watch['id']}"
    assert (await _create(watches, request_id="bot-work")).status_code == 503
    assert (await _action(watches, watch, "run")).status_code == 503
    assert (await watches.client.get(path)).json() == queued
    assert (await watches.client.get("/api/watches")).json()["items"][0]["id"] == watch["id"]
    assert (await watches.client.get(f"{path}/runs/{queued['active_run_id']}")).status_code == 200
    assert len(watches.wakes) == wakes
    paused = await _action(watches, watch, "pause")
    assert paused.status_code == 200 and paused.json()["enabled"] is False
    assert paused.json()["active_run_id"] is None
    assert (await _action(watches, paused.json(), "resume")).status_code == 503
    assert (await watches.client.patch(path, json={
        "expected_revision": paused.json()["revision"], "enabled": True,
    })).status_code == 503
    assert (await watches.client.get(path)).json() == paused.json()
    assert len(watches.wakes) == wakes + 1
    assert (await watches.client.delete(path, params={"expected_revision": paused.json()["revision"]})).json() == {"deleted": True}


@pytest.mark.parametrize("extra", [
    {"private_required": True}, {"project_id": "foreign"}, {"file_ids": ["foreign"]},
    {"user_id": "bob"}, {"model": "cloud"}, {"limits": {"max_operations": 999}},
    {"enabled": "true"}, {"title": "x" * 101}, {"criterion": "x" * 2001},
])
async def test_create_rejects_unsupported_scope_and_untrusted_limits(watches, extra):
    assert (await _create(watches, **extra)).status_code == 422
    assert (await watches.client.get("/api/watches")).json() == {"items": []}


@pytest.mark.parametrize("url", [
    "file:///tmp/private", "http://127.0.0.1/", "http://192.168.1.1/", "https://alice:secret@example.com/",
    "https://example.com/?token=secret", "https://example.com/#secret", "https://example.com:8443/",
])
async def test_saved_watch_url_has_no_private_or_credential_scope(watches, url):
    assert (await _create(watches, url=url)).status_code == 422


@pytest.mark.parametrize("schedule", [
    {"kind": "interval", "interval_minutes": 1, "timezone": "UTC"},
    {"kind": "interval", "interval_minutes": True, "timezone": "UTC"},
    {"kind": "daily", "time": "25:00", "timezone": "UTC"},
    {"kind": "daily", "time": "09:00", "timezone": "Not/AZone"},
    {"kind": "weekly", "time": "09:00", "weekday": 7, "timezone": "UTC"},
    {"kind": "daily", "time": "09:00", "timezone": "UTC", "owner": "bob"},
    {"kind": [], "timezone": "UTC"}, {"kind": {}, "timezone": "UTC"},
])
async def test_preview_and_create_use_same_strict_schedule_rules(watches, schedule):
    assert (await watches.client.post("/api/watches/preview", json={"schedule": schedule})).status_code == 422
    assert (await _create(watches, schedule=schedule)).status_code == 422


async def test_preview_is_ordered_future_utc_without_creating_work(watches):
    preview = await watches.client.post("/api/watches/preview", json={
        "schedule": {"kind": "weekly", "weekday": 0, "time": "09:00", "timezone": "America/Denver"},
    })
    assert preview.status_code == 200, preview.text
    values = [dt.datetime.fromisoformat(value) for value in preview.json()["occurrences"]]
    assert len(values) == 3 and values == sorted(set(values))
    assert all(value.utcoffset() == dt.timedelta() and value > dt.datetime.now(dt.UTC) for value in values)
    assert (await watches.client.get("/api/watches")).json() == {"items": []}


@pytest.mark.parametrize("changes", [
    {"expected_revision": True, "title": "Changed"}, {"expected_revision": 1, "enabled": "false"},
    {"expected_revision": 1, "title": None}, {"expected_revision": 1},
])
async def test_patch_rejects_null_empty_and_coerced_fields(watches, changes):
    watch = (await _create(watches)).json()
    assert (await watches.client.patch(f"/api/watches/{watch['id']}", json=changes)).status_code == 422


async def test_storage_failures_are_safe_and_do_not_expose_provider_text(watches, monkeypatch):
    async def fail(**_):
        raise sqlite3.OperationalError("sensitive private internal path")

    monkeypatch.setattr(watches.store.watches, "list", fail)
    response = await watches.client.get("/api/watches")
    assert response.status_code == 503 and "sensitive" not in response.text


async def test_all_watch_routes_require_sign_in(watches):
    watches.current[0] = None
    for method, path, body in (
        ("GET", "/api/watches", None), ("GET", "/api/notifications", None),
        ("GET", "/api/notifications/unread", None),
        ("POST", "/api/watches/preview", {"schedule": {"kind": "interval", "interval_minutes": 60, "timezone": "UTC"}}),
        ("GET", "/api/watches/missing/runs", None), ("POST", "/api/notifications/missing/read", None),
    ):
        assert (await watches.client.request(method, path, json=body)).status_code == 401


async def test_saved_watch_results_and_notices_link_to_exact_owned_run(watches):
    watch = (await _create(watches, enabled=False)).json()
    source = ["Local model support: unchanged."]

    async def execute(run):
        lease = {key: run[key] for key in ("run_id", "lease_token", "fence")}
        await watches.store.watches.reserve_operation(**lease, kind="fetch")
        baseline = run["baseline"]
        previous = baseline["text"] if baseline else None
        outcome = "baseline" if previous is None else "unchanged" if previous == source[0] else "changed"
        if outcome == "changed":
            await watches.store.watches.reserve_operation(**lease, kind="model")
        return {
            "outcome": outcome, "significant": True if outcome == "changed" else None,
            "report": "The local model judged the support change significant." if outcome == "changed" else "Source check complete.",
            "diff": evidence_diff(previous, source[0]) if outcome == "changed" else "",
            "text": source[0], "sha256": hashlib.sha256(source[0].encode()).hexdigest(),
            "extraction_version": "test-v1", "final_url": watch["url"],
        }

    runtime = WatchRuntime(watches.store.watches, execute)
    expected = ("baseline", "unchanged", "changed")
    run_ids = []
    for outcome in expected:
        if outcome == "changed":
            source[0] = "Local model support: expanded."
        queued = (await _action(watches, watch, "run")).json()
        run_ids.append(queued["active_run_id"])
        assert await runtime.run_once()
        detail = (await watches.client.get(f"/api/watches/{watch['id']}/runs/{run_ids[-1]}")).json()
        assert detail["status"] == "completed" and detail["outcome"] == outcome
        assert detail["significant"] is (True if outcome == "changed" else None)
    history = (await watches.client.get(f"/api/watches/{watch['id']}/runs")).json()["items"]
    assert [run["outcome"] for run in history] == list(reversed(expected))
    notices = (await watches.client.get("/api/notifications")).json()
    assert notices["unread_count"] == 1 and len(notices["items"]) == 1
    assert (await watches.client.get("/api/notifications/unread")).json() == {"unread_count": 1}
    notice = notices["items"][0]
    assert notice["watch_id"] == watch["id"] and notice["run_id"] == run_ids[-1]
    assert notice["kind"] == "change" and notice["read_at"] is None
    watches.current[0] = watches.owners[1]
    assert (await watches.client.post(f"/api/notifications/{notice['id']}/read")).status_code == 404
    watches.current[0] = watches.owners[0]
    read = await watches.client.post(f"/api/notifications/{notice['id']}/read")
    assert read.status_code == 200 and read.json()["read_at"] is not None
    assert (await watches.client.post(f"/api/notifications/{notice['id']}/read")).json() == read.json()
    assert (await watches.client.get("/api/notifications")).json()["unread_count"] == 0
    assert (await watches.client.get("/api/notifications/unread")).json() == {"unread_count": 0}
    await watches.client.delete(f"/api/watches/{watch['id']}", params={"expected_revision": watch["revision"]})
    assert (await watches.client.get("/api/notifications")).json() == {"items": [], "unread_count": 0}
    assert (await watches.client.get(f"/api/watches/{watch['id']}/runs/{run_ids[-1]}")).status_code == 404


async def test_activity_uses_one_canonical_feed_without_awaiting_a_later_count(watches, monkeypatch):
    async def count_would_delete_between_reads(**_):
        raise AssertionError("Activity must not retain items across a second count await.")

    monkeypatch.setattr(watches.store.watches, "unread_count", count_would_delete_between_reads)
    response = await watches.client.get("/api/notifications")
    assert response.status_code == 200 and response.json() == {"items": [], "unread_count": 0}


async def test_historical_check_keeps_its_saved_url_and_criterion_after_watch_edit(watches):
    watch = (await _create(watches, enabled=False)).json()
    queued = (await _action(watches, watch, "run")).json()
    updated = await watches.client.patch(f"/api/watches/{watch['id']}", json={
        "expected_revision": watch["revision"], "url": "https://example.com/new-source",
        "criterion": "Only changes affecting the new source.",
    })
    assert updated.status_code == 200
    historical = await watches.client.get(f"/api/watches/{watch['id']}/runs/{queued['active_run_id']}")
    assert historical.status_code == 200
    assert historical.json()["definition"] == {key: watch[key] for key in ("title", "url", "criterion", "schedule")}
    assert historical.json()["status"] == "cancelled"


@pytest.mark.parametrize("method_name,path_template,http_method", [
    ("get", "/api/watches/{watch_id}", "GET"),
    ("list", "/api/watches", "GET"),
    ("list_runs", "/api/watches/{watch_id}/runs", "GET"),
    ("get_run", "/api/watches/{watch_id}/runs/{run_id}", "GET"),
    ("notification_feed", "/api/notifications", "GET"),
    ("mark_notice_read", "/api/notifications/{notice_id}/read", "POST"),
])
@pytest.mark.parametrize("remove", ["watch", "user_data"])
async def test_deletion_after_repository_await_cannot_return_removed_watch_bodies(
    watches, monkeypatch, method_name, path_template, http_method, remove,
):
    watch = (await _create(watches, enabled=False)).json()
    run = (await _action(watches, watch, "run")).json()
    claimed = await watches.store.watches.claim()
    await watches.store.watches.fail(
        **{key: claimed[key] for key in ("run_id", "lease_token", "fence")},
        error="Saved result body that must disappear.", retry=False,
    )
    notice = (await watches.store.watches.list_notices(user_id=watches.owners[0].user_id))[0]
    repository = watches.store.watches
    original = getattr(repository, method_name)

    async def revoke_after_read(**kwargs):
        payload = await original(**kwargs)
        if remove == "watch":
            await repository.delete(user_id=watches.owners[0].user_id, watch_id=watch["id"],
                                    expected_revision=watch["revision"])
        else:
            await watches.store.purge_local_user_data(user_id=watches.owners[0].user_id)
        return payload

    monkeypatch.setattr(repository, method_name, revoke_after_read)
    path = path_template.format(watch_id=watch["id"], run_id=run["active_run_id"], notice_id=notice["id"])
    response = await watches.client.request(http_method, path)
    assert response.status_code == 404
    assert "Saved result body" not in response.text and "Release notes" not in response.text
