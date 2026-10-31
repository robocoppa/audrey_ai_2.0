"""Unified native Activity only opens the saved result named by its notice."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from audrey.agent_runtime import AgentRuntime
from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.routes.app import router


@pytest.fixture
async def activity(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    current = [owners[0]]
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)

    def authenticate():
        if current[0] is None:
            raise HTTPException(401, "Sign in required.")
        return current[0]

    app.dependency_overrides[require_principal] = authenticate
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, owners=owners, current=current, app=app, client=client)
    store.close()


async def _complete(state):
    created = await state.store.agents.create(
        user_id=state.owners[0].user_id, objective="Compare public release notes.", request_id="agent-1",
    )

    async def execute(_task, step_id):
        return f"Full saved {step_id} with source-derived content."

    runtime = AgentRuntime(state.store.agents, execute)
    assert await runtime.run_once()
    task = await state.store.agents.get(user_id=state.owners[0].user_id, task_id=created["id"])
    return task, runtime


async def _watch_failure(state):
    watch = await state.store.watches.create(
        user_id=state.owners[0].user_id, title="Public release page", url="https://example.com/releases",
        criterion="New stable releases.", request_id="watch-1", enabled=False,
        schedule={"kind": "interval", "interval_minutes": 60, "timezone": "UTC"},
    )
    await state.store.watches.run_now(user_id=state.owners[0].user_id, watch_id=watch["id"],
                                     expected_revision=watch["revision"])
    run = await state.store.watches.claim()
    await state.store.watches.fail(
        **{key: run[key] for key in ("run_id", "lease_token", "fence")},
        error="The public page could not be fetched safely.", retry=False,
    )
    return watch, run


async def test_completion_is_in_unified_activity_with_a_revision_bound_target(activity):
    task, _ = await _complete(activity)
    response = await activity.client.get("/api/activity")
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    feed = response.json()
    assert feed["unread_count"] == 1 and len(feed["items"]) == 1
    notice = feed["items"][0]
    assert notice["origin"] == "agent" and notice["kind"] == "completed"
    assert notice["target"] == {"kind": "agent", "task_id": task["id"], "revision": task["revision"]}
    assert not {"result", "steps", "objective", "error", "user_id"} & notice.keys()
    assert "source-derived content" not in response.text
    assert (await activity.client.get("/api/activity/unread")).json() == {"unread_count": 1}
    resolved = await activity.client.get(f"/api/activity/agent/{notice['id']}/task")
    assert resolved.status_code == 200 and resolved.json() == task
    assert (await activity.client.get("/api/activity/unread")).json() == {"unread_count": 1}


async def test_watch_and_agent_activity_share_counts_but_keep_typed_owned_targets(activity):
    task, _ = await _complete(activity)
    watch, run = await _watch_failure(activity)
    feed = (await activity.client.get("/api/activity")).json()
    assert feed["unread_count"] == 2
    notices = {notice["origin"]: notice for notice in feed["items"]}
    assert notices["watch"]["target"] == {"kind": "watch", "watch_id": watch["id"], "run_id": run["run_id"]}
    assert notices["agent"]["target"]["task_id"] == task["id"]
    legacy = (await activity.client.get("/api/notifications")).json()
    assert legacy["unread_count"] == 1 and len(legacy["items"]) == 1
    for origin, notice in notices.items():
        first = await activity.client.post(f"/api/activity/{origin}/{notice['id']}/read")
        assert first.status_code == 200 and first.json()["read_at"]
        assert (await activity.client.post(f"/api/activity/{origin}/{notice['id']}/read")).json() == first.json()
    assert (await activity.client.get("/api/activity/unread")).json() == {"unread_count": 0}


async def test_rerun_removes_old_notice_and_cannot_open_the_replacement_result(activity):
    task, runtime = await _complete(activity)
    original = (await activity.client.get("/api/activity")).json()["items"][0]
    queued = await activity.store.agents.control(
        user_id=activity.owners[0].user_id, task_id=task["id"], expected_revision=task["revision"], action="rerun",
    )
    assert queued["id"] == task["id"]
    assert (await activity.client.get("/api/activity")).json() == {"items": [], "unread_count": 0}
    assert (await activity.client.get(f"/api/activity/agent/{original['id']}/task")).status_code == 404
    assert (await activity.client.post(f"/api/activity/agent/{original['id']}/read")).status_code == 404
    assert await runtime.run_once()
    fresh = (await activity.client.get("/api/activity")).json()["items"][0]
    assert fresh["id"] != original["id"] and fresh["target"]["task_id"] == task["id"]
    assert fresh["target"]["revision"] > original["target"]["revision"]
    assert (await activity.client.get(f"/api/activity/agent/{original['id']}/task")).status_code == 404


async def test_foreign_activity_notice_and_agent_target_are_unavailable(activity):
    await _complete(activity)
    notices = (await activity.client.get("/api/activity")).json()["items"]
    activity.current[0] = activity.owners[1]
    assert (await activity.client.get("/api/activity")).json() == {"items": [], "unread_count": 0}
    assert (await activity.client.get("/api/activity/unread")).json() == {"unread_count": 0}
    for notice_id in (notices[0]["id"], "missing"):
        assert (await activity.client.get(f"/api/activity/agent/{notice_id}/task")).status_code == 404
        assert (await activity.client.post(f"/api/activity/agent/{notice_id}/read")).status_code == 404
        assert (await activity.client.post(f"/api/activity/watch/{notice_id}/read")).status_code == 404


async def test_retry_and_explicit_cancel_are_silent_but_exhausted_work_has_one_failure_notice(activity):
    created = await activity.store.agents.create(
        user_id=activity.owners[0].user_id, objective="Read public documentation.", request_id="failure-1",
    )
    claimed = await activity.store.agents.claim()
    await activity.store.agents.fail(
        task_id=claimed["id"], lease_token=claimed["lease_token"], fence=claimed["fence"],
        error="Private source-derived failure body must not enter Activity.", retry=True,
    )
    assert (await activity.client.get("/api/activity")).json() == {"items": [], "unread_count": 0}
    assert await activity.store.agents.claim(max_attempts=1) is None
    feed = (await activity.client.get("/api/activity")).json()
    assert len(feed["items"]) == feed["unread_count"] == 1
    assert feed["items"][0]["kind"] == "failure"
    assert "source-derived" not in str(feed)
    task = await activity.store.agents.get(user_id=activity.owners[0].user_id, task_id=created["id"])
    assert (await activity.client.get(f"/api/activity/agent/{feed['items'][0]['id']}/task")).json() == task
    cancelled = await activity.store.agents.create(
        user_id=activity.owners[0].user_id, objective="Another public task.", request_id="cancel-1",
    )
    await activity.store.agents.control(user_id=activity.owners[0].user_id, task_id=cancelled["id"],
                                       expected_revision=cancelled["revision"], action="cancel")
    assert (await activity.client.get("/api/activity/unread")).json() == {"unread_count": 1}


@pytest.mark.parametrize("read", [False, True])
@pytest.mark.parametrize("remove", ["delete", "purge", "rerun"])
async def test_post_await_deletion_or_replacement_fences_activity_bodies(activity, monkeypatch, read, remove):
    task, _ = await _complete(activity)
    notice = (await activity.client.get("/api/activity")).json()["items"][0]
    repository = activity.store.notifications
    name = "mark_read" if read else "feed"
    original = getattr(repository, name)

    async def revoke_after_read(**kwargs):
        payload = await original(**kwargs)
        if remove == "delete":
            await activity.store.agents.delete(user_id=activity.owners[0].user_id, task_id=task["id"],
                                               expected_revision=task["revision"])
        elif remove == "purge":
            await activity.store.purge_local_user_data(user_id=activity.owners[0].user_id)
        else:
            await activity.store.agents.control(user_id=activity.owners[0].user_id, task_id=task["id"],
                                                expected_revision=task["revision"], action="rerun")
        return payload

    monkeypatch.setattr(repository, name, revoke_after_read)
    response = await activity.client.post(f"/api/activity/agent/{notice['id']}/read") if read else await activity.client.get("/api/activity")
    assert response.status_code == 404
    assert "Compare public release notes" not in response.text


async def test_notice_target_checks_the_current_task_after_the_last_await(activity, monkeypatch):
    task, _ = await _complete(activity)
    notice = (await activity.client.get("/api/activity")).json()["items"][0]
    original = activity.store.agents.get

    async def rerun_after_get(**kwargs):
        saved = await original(**kwargs)
        await activity.store.agents.control(user_id=activity.owners[0].user_id, task_id=task["id"],
                                            expected_revision=task["revision"], action="rerun")
        return saved

    monkeypatch.setattr(activity.store.agents, "get", rerun_after_get)
    response = await activity.client.get(f"/api/activity/agent/{notice['id']}/task")
    assert response.status_code == 404 and "source-derived content" not in response.text


async def test_activity_routes_require_native_scope_and_sign_in(activity):
    activity.current[0] = replace(activity.owners[0], auth_method="personal_token", scopes=frozenset({"models:read"}))
    for method, path in (
        ("GET", "/api/activity"), ("GET", "/api/activity/unread"),
        ("GET", "/api/activity/agent/missing/task"), ("POST", "/api/activity/agent/missing/read"),
    ):
        assert (await activity.client.request(method, path)).status_code == 403
    activity.current[0] = None
    for method, path in (
        ("GET", "/api/activity"), ("GET", "/api/activity/unread"),
        ("GET", "/api/activity/agent/missing/task"), ("POST", "/api/activity/agent/missing/read"),
    ):
        assert (await activity.client.request(method, path)).status_code == 401


async def test_activity_storage_failure_does_not_expose_raw_provider_error(activity, monkeypatch):
    async def fail(**_):
        raise sqlite3.OperationalError("private storage location")

    monkeypatch.setattr(activity.store.notifications, "feed", fail)
    response = await activity.client.get("/api/activity")
    assert response.status_code == 503 and "private storage location" not in response.text
