"""Explicit report suggestions stay local, transient, bounded and revocable."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest

from audrey.agent_knowledge_drafts import DraftError, ReportKnowledgeDrafts
from audrey.app_state import ApplicationStore
from audrey.models.ollama import OllamaClient
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.routes.inflight import UserInflightRegistry
from audrey.scheduling import current_local_workload

MODEL = "qwen3.8:latest"


def answer(suggestions=None, **changes):
    suggestions = [{"kind": "question", "text": "Should the orchard gate close at dusk?"}] if suggestions is None else suggestions
    return {"done": True, "done_reason": "stop", "eval_count": 25,
            "message": {"role": "assistant", "content": json.dumps({"suggestions": suggestions})}, **changes}


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, other, admin = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="admin" if name == "admin" else "user", auth_method="test",
    ) for name in ("alice", "bob", "admin")]
    app = SimpleNamespace(state=SimpleNamespace(
        application_store=store,
        cfg=SimpleNamespace(raw={"agent_tasks": {}}, env=SimpleNamespace(ollama_host="http://ollama.test")),
        gate=FairLocalGate(), inflight=UserInflightRegistry(max_inflight_per_user=1),
    ))
    value = SimpleNamespace(app=app, store=store, owner=owner, other=other, admin=admin,
                            requests=[], clients=[], options=[], response=answer(), on_tags=None, on_show=None,
                            on_chat=None, admissions=[],
                            installed=[{"name": MODEL, "model": MODEL, "digest": "a" * 64}],
                            metadata={"details": {"format": "gguf"}, "model_info": {
                                "general.architecture": "qwen3", "general.parameter_count": 27000000000,
                            }})

    async def provider(request):
        value.requests.append(request)
        path = request.url.path
        if path == "/api/tags":
            if value.on_tags:
                await value.on_tags()
            return httpx.Response(200, json={"models": value.installed})
        if path == "/api/show":
            if value.on_show:
                await value.on_show()
            return httpx.Response(200, json=value.metadata)
        assert path == "/api/chat"
        value.admissions.append((current_local_workload(), app.state.inflight.snapshot(),
                                 app.state.gate.pressure_snapshot()))
        if value.on_chat:
            await value.on_chat()
        return httpx.Response(200, json=value.response)

    def factory(endpoint, **kwargs):
        value.options.append((endpoint, kwargs))
        client = OllamaClient(endpoint, transport=httpx.MockTransport(provider), **kwargs)
        value.clients.append(client)
        return client

    value.service = ReportKnowledgeDrafts(app, client_factory=factory)
    yield value
    for client in value.clients:
        await client.aclose()
    store.close()


async def report(state, *, owner=None, project=None, result="The report recommends reviewing whether the gate should close at dusk."):
    owner = owner or state.owner
    await state.store.agents.create(user_id=owner.user_id, objective="Inspect public orchard guidance.",
                                           project_id=project.project_id if project else None,
                                           request_id=uuid.uuid4().hex)
    claimed = await state.store.agents.claim()
    lease = {"task_id": claimed["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
    for step in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**lease, step_id=step)
        await state.store.agents.checkpoint(**lease, step_id=step, result="Saved checkpoint.")
    return await state.store.agents.complete(**lease, result=result)


def key(state, task):
    return {"user_id": state.owner.user_id, "task_id": task["id"], "expected_revision": task["revision"]}


def chats(state):
    return [request for request in state.requests if request.url.path == "/api/chat"]


def snapshot(state):
    with state.store._lock:
        return {table: [tuple(row) for row in state.store._conn.execute(f"SELECT * FROM {table}").fetchall()]  # noqa: S608 - fixed table allowlist below
                for table in ("app_agent_tasks", "app_agent_operations", "app_project_knowledge",
                              "app_project_knowledge_revisions", "app_agent_notices")}


def released(state):
    assert not any(state.app.state.inflight.snapshot().values())
    assert state.app.state.gate.pressure_snapshot()["in_use"] == 0
    assert state.app.state.gate.pressure_snapshot()["waiting"] == 0


async def wait_for(predicate):
    async with asyncio.timeout(2):
        while not predicate():  # noqa: ASYNC110 - observe the scheduler's shared queue without changing its behavior
            await asyncio.sleep(0.005)


async def test_one_verified_local_call_returns_transient_suggestions_without_tools_or_task_mutation(state):
    task = await report(state)
    before = snapshot(state)
    result = await state.service.generate(**key(state, task))
    assert result == {"task_id": task["id"], "task_revision": task["revision"],
                      "suggestions": [{"kind": "question", "text": "Should the orchard gate close at dusk?"}]}
    assert snapshot(state) == before
    assert [request.url.path for request in state.requests] == ["/api/tags", "/api/show", "/api/tags", "/api/show", "/api/chat"]
    payload = json.loads(chats(state)[0].content)
    assert payload["model"] == MODEL and payload["stream"] is False and payload["think"] is False
    assert payload["options"] == {"temperature": 0, "num_predict": 1024, "num_ctx": 8192}
    assert "tools" not in payload and payload["format"]["additionalProperties"] is False
    assert json.loads(payload["messages"][1]["content"]) == {"saved_report": task["result"]}
    assert state.options == [("http://ollama.test", {"trust_env": False})]
    assert state.admissions[0][0] == "background"
    assert state.admissions[0][1][state.owner.storage_namespace] == 1
    assert state.admissions[0][2]["in_use"] == 1
    assert state.clients[0]._client.is_closed
    released(state)


async def test_empty_suggestions_are_a_successful_result_and_respect_configured_token_cap(state):
    task = await report(state)
    state.response = answer([])
    state.app.state.cfg.raw["agent_tasks"]["max_tokens_per_call"] = 128
    assert (await state.service.generate(**key(state, task)))["suggestions"] == []
    assert json.loads(chats(state)[0].content)["options"]["num_predict"] == 128


async def test_project_rename_or_loss_does_not_revoke_a_saved_report_or_supply_project_data(state):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Secret Project title",
                                                instructions="Hidden Project instruction body.")
    task = await report(state, project=project)
    await state.store.projects.delete(user_id=state.owner.user_id, project_id=project.project_id)
    await state.service.generate(**key(state, task))
    payload = chats(state)[0].content.decode()
    assert "Secret Project title" not in payload and "Hidden Project instruction" not in payload
    state.service.fence_response(**key(state, task))


@pytest.mark.parametrize("change,code", [("foreign", 404), ("deleted", 404), ("revision", 409), ("rerun", 409),
                                         ("disabled", 404), ("bot", 404), ("agent_disabled", 503), ("local_disabled", 503)])
async def test_unavailable_authority_or_report_stops_before_provider_requests(state, change, code):
    task = await report(state)
    if change == "foreign":
        task = await report(state, owner=state.other)
    elif change == "deleted":
        await state.store.agents.delete(**key(state, task))
    elif change == "revision":
        task = {**task, "revision": task["revision"] + 1}
    elif change == "rerun":
        await state.store.agents.control(**key(state, task), action="rerun")
    elif change == "disabled":
        with state.store._lock, state.store._conn:
            state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
    elif change == "bot":
        await state.store.admin_update_user(actor_user_id=state.admin.user_id, target_user_id=state.owner.user_id,
                                             groups=[*state.owner.groups, "bots"])
    elif change == "agent_disabled":
        state.app.state.cfg.raw["agent_tasks"]["enabled"] = False
    else:
        await state.store.set_model_access_policy(actor_user_id=state.admin.user_id, model_id="local",
                                                  enabled=False, audience="users")
    with pytest.raises(DraftError) as error:
        await state.service.generate(**key(state, task))
    assert error.value.status_code == code and state.requests == []
    released(state)


async def test_full_report_over_prompt_budget_is_refused_without_truncation_or_provider_request(state):
    task = await report(state, result="word " * 9000)
    with pytest.raises(DraftError) as error:
        await state.service.generate(**key(state, task))
    assert error.value.status_code == 422 and "input limit" in str(error.value) and state.requests == []
    assert (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["result"] == task["result"]


@pytest.mark.parametrize("response", [
    answer(done=False), answer(done_reason="length"), answer(done_reason="unknown"), answer(error="private detail"),
    answer(message={"content": "{}"}), answer(message={"content": '{"suggestions":[] ,"extra":true}'}),
    answer(message={"content": '{"suggestions":[],"suggestions":[]}'}),
    answer(message={"content": "```json\n{\"suggestions\":[]}\n```"}),
    answer(message={"content": '{"suggestions":[{"kind":"fact","text":"x","extra":1}]}'}),
    answer(message={"content": '{"suggestions":[{"kind":"unknown","text":"x"}]}'}),
    answer(message={"content": '{"suggestions":[{"kind":"fact","text":"   "}]}'}),
    answer(message={"content": '{"suggestions":[{"kind":"fact","text":17}]}'}),
    answer([{"kind": "fact", "text": "x" * 1201}]),
    answer([{"kind": "fact", "text": "\ud800"}]),
    answer([{"kind": "fact", "text": str(index)} for index in range(4)]),
    answer([{"kind": "fact", "text": "Repeated statement"}, {"kind": "question", "text": " repeated  STATEMENT "}]),
    answer(message={"content": '{"suggestions":[]}', "tool_calls": [{"function": {"name": "web_fetch"}}]}),
    answer(eval_count=1025), answer(eval_count=None),
])
async def test_incomplete_invalid_duplicate_or_tool_outputs_fail_atomically_without_saving(state, response):
    task = await report(state)
    before = snapshot(state)
    state.response = copy.deepcopy(response)
    with pytest.raises(DraftError) as error:
        await state.service.generate(**key(state, task))
    assert error.value.status_code == 503 and len(chats(state)) == 1
    assert "private detail" not in str(error.value) and snapshot(state) == before
    released(state)


@pytest.mark.parametrize("change", ["not_installed", "remote", "digest", "metadata"])
async def test_model_identity_is_verified_and_pinned_without_cloud_fallback(state, change):
    task = await report(state)
    if change == "not_installed":
        state.installed = []
    elif change == "remote":
        state.metadata["remote_host"] = "sensitive-cloud-host"
    elif change == "metadata":
        state.metadata["details"]["format"] = "remote"
    else:
        reads = 0

        async def change_digest():
            nonlocal reads
            reads += 1
            if reads == 2:
                state.installed[0]["digest"] = "b" * 64

        state.on_tags = change_digest
    with pytest.raises(DraftError) as error:
        await state.service.generate(**key(state, task))
    assert error.value.status_code == 503 and chats(state) == []
    assert "sensitive" not in str(error.value)
    released(state)


async def test_account_model_or_report_changes_during_metadata_reads_block_generation(state):
    task = await report(state)

    async def revoke():
        await state.store.set_model_access_policy(actor_user_id=state.admin.user_id, model_id="local",
                                                  enabled=False, audience="users")

    state.on_show = revoke
    with pytest.raises(DraftError) as error:
        await state.service.generate(**key(state, task))
    assert error.value.status_code == 503 and chats(state) == []


async def test_sync_fence_rechecks_canonical_grant_after_an_async_catalog_result(state, monkeypatch):
    from audrey.agent_knowledge_drafts import resolve_model

    task = await report(state)

    async def stale_catalog(*args, **kwargs):
        allowed = await resolve_model(*args, **kwargs)
        await state.store.set_model_access_policy(actor_user_id=state.admin.user_id, model_id="local",
                                                  enabled=False, audience="users")
        return allowed

    monkeypatch.setattr("audrey.agent_knowledge_drafts.resolve_model", stale_catalog)
    with pytest.raises(DraftError) as error:
        await state.service.generate(**key(state, task))
    assert error.value.status_code == 503 and state.requests == []


async def test_final_sync_fence_honors_changed_publication_roles_without_cached_grants(state):
    task = await report(state)
    await state.service.generate(**key(state, task))
    await state.store.set_model_publication_profile(
        actor_user_id=state.admin.user_id, model_id="local", visibility="restricted",
        roles=["admins"], display_name="Local",
    )
    with pytest.raises(DraftError) as error:
        state.service.fence_response(**key(state, task))
    assert error.value.status_code == 503


@pytest.mark.parametrize("change,code", [("delete", 404), ("rerun", 409), ("disable", 404), ("grant", 503)])
async def test_revocation_after_generation_never_returns_stale_suggestions(state, change, code):
    task = await report(state)

    async def revoke():
        if change == "delete":
            await state.store.agents.delete(**key(state, task))
        elif change == "rerun":
            await state.store.agents.control(**key(state, task), action="rerun")
        elif change == "disable":
            with state.store._lock, state.store._conn:
                state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        else:
            await state.store.set_model_access_policy(actor_user_id=state.admin.user_id, model_id="local",
                                                      enabled=False, audience="users")

    state.on_chat = revoke
    with pytest.raises(DraftError) as error:
        await state.service.generate(**key(state, task))
    assert error.value.status_code == code and len(chats(state)) == 1
    released(state)


async def test_same_owner_duplicate_is_rejected_while_queued_and_cancellation_releases_all_resources(state):
    task = await report(state)
    async with state.app.state.gate.acquire(MODEL, location="local", user_id="occupied"):
        running = asyncio.create_task(state.service.generate(**key(state, task)))
        await wait_for(lambda: state.app.state.gate.pressure_snapshot()["waiting"] == 1)
        with pytest.raises(DraftError) as error:
            await state.service.generate(**key(state, task))
        assert error.value.status_code == 429 and chats(state) == []
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not any(state.app.state.inflight.snapshot().values())
    released(state)
    assert (await state.service.generate(**key(state, task)))["suggestions"]


async def test_queued_request_revalidates_report_after_admission(state):
    task = await report(state)
    async with state.app.state.gate.acquire(MODEL, location="local", user_id="occupied"):
        running = asyncio.create_task(state.service.generate(**key(state, task)))
        await wait_for(lambda: state.app.state.gate.pressure_snapshot()["waiting"] == 1)
        await state.store.agents.control(**key(state, task), action="rerun")
    with pytest.raises(DraftError) as error:
        await running
    assert error.value.status_code == 409 and chats(state) == []
    released(state)


async def test_total_timeout_includes_admission_and_clears_duplicate_guard(state, monkeypatch):
    task = await report(state)
    monkeypatch.setattr("audrey.agent_knowledge_drafts._TOTAL_TIMEOUT_SECONDS", 0.05)
    async with state.app.state.gate.acquire(MODEL, location="local", user_id="occupied"):
        with pytest.raises(DraftError) as error:
            await state.service.generate(**key(state, task))
        assert error.value.status_code == 504 and chats(state) == []
    released(state)
    monkeypatch.setattr("audrey.agent_knowledge_drafts._TOTAL_TIMEOUT_SECONDS", 5)
    await state.service.generate(**key(state, task))


async def test_cancelling_a_generation_closes_provider_and_releases_resources(state):
    task = await report(state)
    entered = asyncio.Event()

    async def wait():
        entered.set()
        await asyncio.Event().wait()

    state.on_chat = wait
    running = asyncio.create_task(state.service.generate(**key(state, task)))
    await asyncio.wait_for(entered.wait(), 2)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert state.clients[0]._client.is_closed
    released(state)
    state.on_chat = None
    await state.service.generate(**key(state, task))


async def test_route_can_apply_a_final_sync_fence_after_service_result_is_ready(state):
    task = await report(state)
    await state.service.generate(**key(state, task))
    await state.store.agents.delete(**key(state, task))
    with pytest.raises(DraftError) as error:
        state.service.fence_response(**key(state, task))
    assert error.value.status_code == 404
