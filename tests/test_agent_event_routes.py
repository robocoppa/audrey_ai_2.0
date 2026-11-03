"""Per-task execution history uses owned, fresh, bounded metadata pages."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.routes.app import router


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "events.sqlite")
    owners = [await store.resolve_external_identity(provider="test", subject=name,
        email=f"{name}@example.com", display_name=name, role="user", auth_method="test")
        for name in ("alice", "bob")]
    current = [owners[0]]
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={})
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: current[0]
    task = await store.agents.create(user_id=owners[0].user_id, objective="Draft a checklist.", request_id="events")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(store=store, owners=owners, current=current, client=client, task=task)
    store.close()


async def started(s):
    task = await s.store.agents.claim()
    lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
    await s.store.agents.reserve_operation(**lease, step_id="plan")
    seq = await s.store.agents.start_event(**lease, stage="plan", kind="model")
    return lease, seq


async def test_events_are_separate_owned_bounded_metadata_with_no_revision_side_effect(state):
    s = state
    url = f"/api/agents/{s.task['id']}/events"
    empty = await s.client.get(url)
    assert empty.status_code == 200 and empty.json() == {"items": [], "has_more": False, "truncated": False}
    lease, seq = await started(s)
    revision = (await s.store.agents.get(user_id=s.owners[0].user_id, task_id=s.task["id"]))["revision"]
    await s.store.agents.finish_event(**lease, seq=seq, state="completed", duration_ms=12, input_tokens=21, output_tokens=9)
    page = await s.client.get(url, params={"limit": 1})
    assert page.status_code == 200 and page.headers["cache-control"] == "private, no-store"
    event = page.json()["items"][0]
    assert event["seq"] == seq and event["kind"] == "model" and event["state"] == "completed"
    assert event["duration_ms"] == 12 and event["input_tokens"] == 21 and event["output_tokens"] == 9
    assert set(event) == {"seq", "attempt", "stage", "kind", "state", "tool", "created_at", "finished_at", "duration_ms", "input_tokens", "output_tokens"}
    assert page.json()["has_more"]
    older = await s.client.get(url, params={"before": seq, "limit": 1})
    assert older.json()["items"][0]["kind"] == "stage" and not older.json()["has_more"]
    current = (await s.client.get(f"/api/agents/{s.task['id']}")).json()
    assert current["revision"] == revision and "events" not in current
    assert "events" not in (await s.client.get("/api/agents")).json()["items"][0]
    s.current[0] = s.owners[1]
    assert (await s.client.get(url)).status_code == 404
    assert (await s.client.get("/api/agents/missing/events")).status_code == 404


@pytest.mark.parametrize("query", [{"limit": 0}, {"limit": 51}, {"before": 0}, {"before": "true"}])
async def test_invalid_event_pagination_is_rejected(state, query):
    assert (await state.client.get(f"/api/agents/{state.task['id']}/events", params=query)).status_code == 422


async def test_late_event_completion_invalidates_a_captured_response(state, monkeypatch):
    lease, seq = await started(state)
    original = state.store.agents.events

    async def changed(**kwargs):
        page = await original(**kwargs)
        await state.store.agents.finish_event(**lease, seq=seq, state="completed")
        return page

    monkeypatch.setattr(state.store.agents, "events", changed)
    response = await state.client.get(f"/api/agents/{state.task['id']}/events")
    assert response.status_code == 409 and "items" not in response.json()


async def test_task_deleted_after_event_read_does_not_deliver_stale_history(state, monkeypatch):
    await started(state)
    current = await state.store.agents.get(user_id=state.owners[0].user_id, task_id=state.task["id"])
    cancelled = await state.store.agents.control(user_id=state.owners[0].user_id, task_id=state.task["id"],
        expected_revision=current["revision"], action="cancel")
    original = state.store.agents.events

    async def erased(**kwargs):
        page = await original(**kwargs)
        await state.store.agents.delete(user_id=state.owners[0].user_id, task_id=state.task["id"],
            expected_revision=cancelled["revision"])
        return page

    monkeypatch.setattr(state.store.agents, "events", erased)
    response = await state.client.get(f"/api/agents/{state.task['id']}/events")
    assert response.status_code == 404 and "items" not in response.json()
