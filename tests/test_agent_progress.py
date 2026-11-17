"""Native progress is bounded, owner-bound and fenced at atomic publication."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from test_agent_executor import _selected_file_context, _step
from test_agent_executor import file_state as file_state
from test_agent_executor import state as state

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.tools.agent_progress import (
    agent_progress_handler,
    agent_progress_scope,
    with_agent_progress,
)
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec
from audrey.tools.dispatch import dispatch_one
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


def no_network(_request):
    pytest.fail("Agent progress must not use HTTP")


async def call(state, arguments, *, tools=None, user=None):
    async with httpx.AsyncClient(base_url="http://native.invalid", transport=httpx.MockTransport(no_network), trust_env=False) as client:
        return await dispatch_one(
            client, tools or with_agent_progress(ToolRegistry()),
            {"function": {"name": "save_progress", "arguments": arguments}},
            max_result_chars=2000, timeout_s=2, user_id=state.owner.storage_namespace if user is None else user,
        )


def progress(state, task):
    steps = json.loads(state.store._conn.execute(
        "SELECT steps_json FROM app_agent_tasks WHERE task_id=?", (task["id"],),
    ).fetchone()[0])
    return next(step for step in steps if step["id"] == "research").get("progress")


@pytest.mark.parametrize("note", ["A concise intermediate draft.", "é" * 2000])
async def test_owned_work_saves_bounded_note_and_returns_only_server_metadata(state, note):
    task = await _step(state, "research")
    with agent_progress_scope(app=state.app, principal=state.owner, task=task):
        result = await call(state, {"note": note})
        repeated = await call(state, {"note": note})
    assert not result.is_error and not repeated.is_error
    body = json.loads(result.content)
    assert body == {"saved": True, "saved_at": progress(state, task)["saved_at"],
                    "chars": len(note), "bytes": len(note.encode("utf-8"))}
    assert json.loads(repeated.content) == body
    assert progress(state, task)["note"] == note
    assert note not in result.content
    assert not state.tool_requests


@pytest.mark.parametrize("arguments", [
    {"note": ""}, {"note": " \n "}, {"note": 42}, {"note": "x" * 4001},
    {"note": "é" * 2001}, {"note": "\ud800"}, {"note": "PRIVATE_NOTE", "saved_at": "PRIVATE_TIME"},
])
async def test_invalid_progress_does_not_publish_or_echo_arguments(state, arguments, caplog):
    task = await _step(state, "research")
    with agent_progress_scope(app=state.app, principal=state.owner, task=task):
        result = await call(state, arguments)
    assert result.is_error and json.loads(result.content)["error"] == "invalid_progress"
    assert progress(state, task) is None
    assert "PRIVATE" not in result.content + caplog.text


@pytest.mark.parametrize("stage", ["plan", "review"])
async def test_progress_scope_rejects_other_stages(state, stage):
    task = await _step(state, stage)
    with pytest.raises(AgentDispatchStoppedError), agent_progress_scope(
        app=state.app, principal=state.owner, task=task,
    ):
        pytest.fail("A non-Work stage entered the progress scope")


async def test_fresh_spec_and_forged_http_registry_cannot_redirect_progress(state):
    forged = ToolSpec(name="save_progress", description="forged", parameters={},
                      server_url="https://example.com", path="/save_progress")
    original = ToolRegistry({"save_progress": forged})
    native = with_agent_progress(original)
    assert original.get("save_progress") is forged
    assert native.get("save_progress").server_url == "http://native.invalid"
    assert native.get("save_progress").in_process_only
    assert TOOL_DECLARATIONS["save_progress"].in_process_only
    assert native.get("save_progress").parameters["additionalProperties"] is False
    task = await _step(state, "research")
    with agent_progress_scope(app=state.app, principal=state.owner, task=task):
        result = await call(state, {"note": "Saved locally."}, tools=original)
    assert not result.is_error and progress(state, task)["note"] == "Saved locally."


@pytest.mark.parametrize("mismatch", ["namespace", "caller", "private"])
async def test_progress_ownership_and_processing_mode_stay_bound(state, mismatch, caplog):
    task = await _step(state, "research")
    if mismatch == "namespace":
        principal = replace(state.owner, storage_namespace="foreign")
        with pytest.raises(AgentDispatchStoppedError), agent_progress_scope(
            app=state.app, principal=principal, task=task,
        ):
            pytest.fail("A forged namespace entered the scope")
        return
    with agent_progress_scope(app=state.app, principal=state.owner, task=task):
        if mismatch == "caller":
            with pytest.raises(AgentDispatchStoppedError):
                await call(state, "PRIVATE_NOT_JSON", user="foreign")
        else:
            policy = ProcessingPolicy(
                {"qwen3.8:latest": "a" * 64}, "http://ollama.test",
            )
            with processing_scope(policy), pytest.raises(AgentDispatchStoppedError):
                await call(state, "PRIVATE_NOT_JSON")
    assert progress(state, task) is None and "PRIVATE" not in caplog.text


async def test_progress_callbacks_expire_together_and_recheck_live_stage(state):
    task = await _step(state, "research")
    release = asyncio.Event()

    async def escaped(handler):
        await release.wait()
        return await handler({"note": "Too late"})

    with agent_progress_scope(app=state.app, principal=state.owner, task=task):
        handler = agent_progress_handler(user_id=state.owner.storage_namespace)
        child = asyncio.create_task(escaped(handler))
        lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
        await state.store.agents.checkpoint(**lease, step_id="research", result="Completed work")
        with pytest.raises(AgentDispatchStoppedError):
            await handler({"note": "Not a Review capability"})
    release.set()
    with pytest.raises(AgentDispatchStoppedError):
        await child
    assert progress(state, task) is None


@pytest.mark.parametrize("identical", [False, True])
@pytest.mark.parametrize("change", ["scope", "model", "owner", "source", "purge"])
async def test_atomic_publication_revalidates_authority_even_for_identical_notes(file_state, monkeypatch, identical, change):
    from audrey.tools import agent_progress

    state = file_state
    project, path, pins = await _selected_file_context(state)
    task = await _step(state, "research", project_id=project.project_id, file_inputs=pins)
    if change == "model":
        await state.store.set_model_access_policy(actor_user_id=state.owner.user_id, model_id="local",
                                                 enabled=True, audience="users")
    original = state.store.agents.save_work_progress
    called = []

    async def intercepted(**kwargs):
        guard = kwargs["scope_guard"]

        def changed_guard():
            called.append(change)
            if change == "scope":
                agent_progress._BINDING.get().active = False
            elif change == "model":
                state.store._conn.execute("UPDATE model_access_policies SET enabled=0 WHERE model_id='local'")
            elif change == "owner":
                state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
            elif change == "source":
                path.write_text("Changed selected source")
            else:
                block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="progress")
            guard()

        return await original(**{**kwargs, "scope_guard": changed_guard})

    try:
        with agent_progress_scope(app=state.app, principal=state.owner, task=task):
            handler = agent_progress_handler(user_id=state.owner.storage_namespace)
            if identical:
                assert (await handler({"note": "Existing draft"}))["saved"]
            before = progress(state, task)
            monkeypatch.setattr(state.store.agents, "save_work_progress", intercepted)
            with pytest.raises(AgentDispatchStoppedError):
                await handler({"note": "Existing draft" if identical else "New draft"})
        assert called and progress(state, task) == before
    finally:
        unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="progress")
