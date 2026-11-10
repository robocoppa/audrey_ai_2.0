"""Selected-file pages use frozen source proofs and revocable Work/Review authority."""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from test_agent_file_inputs import create, file, lease
from test_agent_file_inputs import state as state

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state import ApplicationStore
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.tools.agent_file_reads import (
    agent_file_read_handler,
    agent_file_read_scope,
    with_selected_file_read,
)
from audrey.tools.discovery import ToolRegistry
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


async def _work(state, file_ids, *, stage="research"):
    await create(state, file_ids)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**lease(task), step_id="plan")
    task = await state.store.agents.checkpoint(**lease(task), step_id="plan", result="Use the selected files.")
    if stage == "review":
        await state.store.agents.reserve_operation(**lease(task), step_id="research")
        task = await state.store.agents.checkpoint(**lease(task), step_id="research", result="Prepared the work.")
    return task


async def _read(state, task, **arguments):
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        return await agent_file_read_handler(user_id=state.owner.storage_namespace)(arguments)


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_reads_past_prompt_prefix_and_parses_only_requested_selection(state, monkeypatch, stage):
    text = "A useful line of local text. " * 1000 + "ONLY AT THE END"
    first, _ = await file(state, text=text)
    second, _ = await file(state, text="OTHER SELECTED TEXT")
    await file(state, text="UNSELECTED PRIVATE CONTENT")
    task = await _work(state, [first, second], stage=stage)
    prefix = next(item for item in task["file_inputs"]["files"] if item["file_id"] == first)
    assert prefix["chars"] < text.index("ONLY AT THE END")
    original = state.service.reader._read_file
    reads = []

    def tracked(namespace, row, artifact):
        reads.append(row["file_id"])
        return original(namespace, row, artifact)

    monkeypatch.setattr(state.service.reader, "_read_file", tracked)
    result = await _read(state, task, file_id=first, offset=len(text) - 15)
    assert result == {"file_id": first, "filename": "source.txt", "artifact": "text", "origin": "upload",
        "text": text[-15:], "start": len(text) - 15, "end": len(text), "total_chars": len(text),
        "next_offset": None, "has_more": False}
    assert reads == [first]
    assert "ONLY AT THE END" not in state.store._conn.execute("SELECT pin_json FROM app_agent_file_inputs").fetchone()[0]
    assert str(state.root) not in json.dumps(result)


@pytest.mark.parametrize("unit", ["ordinary text ", "😀漢字e\u0301", '"\\\n\t\x01'])
async def test_encoded_json_budget_preserves_every_unicode_character_and_truthful_cursor(state, unit):
    text = unit * 1000
    fid, _ = await file(state, text=text)
    task = await _work(state, [fid])
    offset, pieces = 0, []
    while True:
        result = await _read(state, task, file_id=fid, offset=offset, limit=6000)
        encoded = json.dumps(result, ensure_ascii=False)
        assert len(encoded) <= 5000 and json.loads(encoded) == result
        assert result["start"] == offset and result["total_chars"] == len(text)
        assert result["end"] == offset + len(result["text"])
        assert result["text"] == text[offset:result["end"]]
        pieces.append(result["text"])
        if not result["has_more"]:
            assert result["next_offset"] is None and result["end"] == len(text)
            break
        assert result["next_offset"] == result["end"] > offset
        offset = result["next_offset"]
    assert "".join(pieces) == text
    eof = await _read(state, task, file_id=fid, offset=len(text), limit=1)
    assert eof["text"] == "" and eof["end"] == len(text) and eof["next_offset"] is None
    past = await _read(state, task, file_id=fid, offset=len(text) + 1)
    assert past["error"] == "offset_out_of_range" and "text" not in past


@pytest.mark.parametrize("arguments", [{}, {"file_id": "../secret"}, {"file_id": "x", "path": "/forbidden/secret"},
    {"file_id": "x", "offset": True}, {"file_id": "x", "offset": -1}, {"file_id": "x", "offset": 500001},
    {"file_id": "x", "limit": False}, {"file_id": "x", "limit": 0}, {"file_id": "x", "limit": 6001},
    {"file_id": "x", "limit": "3000"}, {"file_id": "x", "offset": 1.0}])
async def test_invalid_arguments_fail_without_parsing_any_source(state, monkeypatch, arguments):
    fid, _ = await file(state)
    task = await _work(state, [fid])
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Invalid arguments caused a read"))
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        result = await agent_file_read_handler(user_id=state.owner.storage_namespace)(arguments)
    assert result["error"] == "invalid_arguments" and "text" not in result


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_unselected_foreign_and_private_files_are_identically_refused_without_read(state, monkeypatch, stage):
    fid, _ = await file(state)
    unselected, _ = await file(state, text="DO NOT READ")
    foreign_project = await state.store.projects.create(user_id=state.other.user_id, name="Foreign")
    foreign, _ = await file(state, owner=state.other, project=foreign_project, text="FOREIGN SECRET")
    private_project = await state.store.projects.create(user_id=state.owner.user_id, name="Private", private_required=True)
    private, _ = await file(state, private=True, project=private_project, text="PRIVATE SECRET")
    task = await _work(state, [fid], stage=stage)
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Unselected file was parsed"))
    responses = [await _read(state, task, file_id=value) for value in (unselected, foreign, private, "unknown")]
    assert all(result == responses[0] for result in responses)
    assert responses[0] == {"error": "file_not_selected", "detail": "Read only a file explicitly selected for this agent."}


@pytest.mark.parametrize(("kind", "origin", "artifact"), [("image", "upload", "text"), ("video", "upload", "transcript"),
    ("text", "generated", "text")])
async def test_retained_artifact_and_generated_origin_remain_exact(state, kind, origin, artifact):
    text = "Retained selected saved text."
    fid, path = await file(state, kind=kind, origin=origin, text=text)
    if origin == "upload":
        path.unlink()
    task = await _work(state, [fid])
    page = await _read(state, task, file_id=fid, offset=5, limit=8)
    assert page["text"] == text[5:13] and page["artifact"] == artifact and page["origin"] == origin


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_same_stat_changed_body_fails_digest_proof(state, stage):
    fid, path = await file(state, text="Frozen original body")
    task = await _work(state, [fid], stage=stage)
    previous = path.stat()
    path.write_text("Changed revised body")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    assert state.service.is_current(state.owner.user_id, state.project.project_id,
        state.store.agents._file_input_pins(task["id"], state.owner.user_id)[0][1])
    with pytest.raises(AgentDispatchStoppedError):
        await _read(state, task, file_id=fid)


@pytest.mark.parametrize("change", ["cancel", "detach", "body", "delete", "disabled", "expired", "model", "purge", "owner"])
@pytest.mark.parametrize("stage", ["research", "review"])
async def test_final_authority_fence_after_disk_read_prevents_late_text(state, monkeypatch, change, stage):
    fid, path = await file(state, text="TEXT MUST NOT ESCAPE")
    task = await _work(state, [fid], stage=stage)
    original = state.service.reader._disk

    async def intercepted(*args, **kwargs):
        value = await original(*args, **kwargs)
        if change == "cancel":
            current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
            await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=current["revision"], action="cancel")
        elif change == "detach":
            await state.store.projects.remove_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=fid)
        elif change == "body":
            path.write_text("CHANGED TEXT")
        elif change == "delete":
            await state.uploads.request_file_deletion(fid, user=state.owner.storage_namespace, requested_at="2026-11-03T00:00:00+00:00")
        elif change == "disabled":
            state.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
        elif change == "expired":
            with state.store._lock, state.store._conn:
                state.store._conn.execute("UPDATE app_agent_tasks SET lease_expires_at='2000-01-01T00:00:00+00:00'")
        elif change == "model":
            admin = await state.store.resolve_external_identity(provider="test", subject="admin", email="admin@example.com",
                display_name="Admin", role="admin", auth_method="test")
            await state.store.set_model_access_policy(actor_user_id=admin.user_id, model_id="local", enabled=False, audience="users")
        elif change == "purge":
            block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="test-read")
        else:
            with state.store._lock, state.store._conn:
                state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        return value

    monkeypatch.setattr(state.service.reader, "_disk", intercepted)
    try:
        with pytest.raises(AgentDispatchStoppedError):
            await _read(state, task, file_id=fid)
    finally:
        unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="test-read")


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_scope_expiration_invalidates_captured_and_inherited_handlers(state, monkeypatch, stage):
    fid, _ = await file(state)
    task = await _work(state, [fid], stage=stage)
    proceed = asyncio.Event()

    async def inherited():
        await proceed.wait()
        return await agent_file_read_handler(user_id=state.owner.storage_namespace)({"file_id": fid})

    assert agent_file_read_handler(user_id=state.owner.storage_namespace) is None
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        captured = agent_file_read_handler(user_id=state.owner.storage_namespace)
        child = asyncio.create_task(inherited())
        with pytest.raises(AgentDispatchStoppedError):
            agent_file_read_handler(user_id=state.other.storage_namespace)
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Expired scope reached source"))
    with pytest.raises(AgentDispatchStoppedError):
        await captured({"file_id": fid})
    proceed.set()
    with pytest.raises(AgentDispatchStoppedError):
        await child


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_read_started_in_scope_cannot_finish_after_scope_exit(state, monkeypatch, stage):
    fid, _ = await file(state)
    task = await _work(state, [fid], stage=stage)
    started, release = asyncio.Event(), asyncio.Event()
    original = state.service.reader._disk

    async def paused(*args, **kwargs):
        result = await original(*args, **kwargs)
        started.set()
        await release.wait()
        return result

    monkeypatch.setattr(state.service.reader, "_disk", paused)
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        child = asyncio.create_task(agent_file_read_handler(user_id=state.owner.storage_namespace)({"file_id": fid}))
        await started.wait()
    release.set()
    with pytest.raises(AgentDispatchStoppedError):
        await child


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_plan_private_and_changed_policy_cannot_supply_read_scope(state, stage):
    fid, _ = await file(state)
    task = await _work(state, [fid], stage=stage)
    with pytest.raises(AgentDispatchStoppedError), agent_file_read_scope(app=state.app, principal=state.owner, task={**task, "current_step": "plan"}):
        pytest.fail("Plan received read scope")
    policy = ProcessingPolicy(local_models={"qwen3.8:latest": "a" * 64}, ollama_base_url="http://local.test")
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        handler = agent_file_read_handler(user_id=state.owner.storage_namespace)
        with processing_scope(policy), pytest.raises(AgentDispatchStoppedError):
            await handler({"file_id": fid})
    with processing_scope(policy), pytest.raises(AgentDispatchStoppedError), agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        pytest.fail("Private policy received read scope")


async def test_work_read_callback_cannot_cross_into_live_review(state, monkeypatch):
    fid, _ = await file(state, text="LATE WORK MUST NOT READ THIS")
    task = await _work(state, [fid])
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        captured = agent_file_read_handler(user_id=state.owner.storage_namespace)
        await state.store.agents.reserve_operation(**lease(task), step_id="research")
        await state.store.agents.checkpoint(**lease(task), step_id="research", result="Finished Work.")
        monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Stale Work parsed a source"))
        with pytest.raises(AgentDispatchStoppedError):
            await captured({"file_id": fid})


async def test_review_read_binding_cannot_run_before_work_finishes(state, monkeypatch):
    fid, _ = await file(state)
    task = await _work(state, [fid])
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Early Review parsed a source"))
    with pytest.raises(AgentDispatchStoppedError):
        await _read(state, {**task, "current_step": "review"}, file_id=fid)


async def test_restart_reuses_frozen_source_without_saved_page_text(state):
    fid, _ = await file(state, text="Long selected file. " * 500)
    task = await _work(state, [fid])
    page = await _read(state, task, file_id=fid, offset=8000, limit=100)
    second = ApplicationStore(state.store.path)
    from audrey.agent_file_inputs import get_agent_file_inputs

    state.app.state.application_store = second
    try:
        get_agent_file_inputs(state.app)
        repeated = await _read(state, task, file_id=fid, offset=8000, limit=100)
        assert repeated == page
        assert page["text"] not in second._conn.execute("SELECT pin_json FROM app_agent_file_inputs").fetchone()[0]
    finally:
        second.close()
        state.app.state.application_store = state.store


def test_native_spec_is_fresh_and_contains_only_selected_id_and_character_cursor():
    shared = ToolRegistry()
    first, second = with_selected_file_read(shared), with_selected_file_read(shared)
    assert shared.names() == [] and first.get("read_selected_file") is not second.get("read_selected_file")
    spec = first.get("read_selected_file")
    assert spec.in_process_only and spec.path == "/read_selected_file"
    assert set(spec.parameters["properties"]) == {"file_id", "offset", "limit"}
    assert spec.parameters["additionalProperties"] is False
