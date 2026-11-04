"""Approved file-backed notes freeze provenance without copying source bodies."""

from __future__ import annotations

import dataclasses
import json
import uuid

import pytest
from test_agent_file_inputs import file, lease
from test_agent_file_inputs import state as state
from test_agent_project_sources import message, ref, report

from audrey.agent_file_inputs import get_agent_file_inputs
from audrey.agent_knowledge_files import CapturedKnowledgeFileSources, get_agent_knowledge_files
from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskAccessError, AgentTaskContextError, AgentTaskError
from audrey.kb.extract import ocr_text_path


async def _source(state, *, artifact="text", text="CANONICAL FILE BODY, NEVER MODEL CONTEXT", **changes):
    fid, path = await file(state, text=text, **changes)
    if changes.get("private"):
        ocr_text_path(path).write_text(text)
    if artifact != "text":
        suffix = {"transcript": "transcript.txt", "summary": "summary.txt", "visual": "frames.txt"}[artifact]
        path.with_name(fid + "." + suffix).write_text(text)
    owner = changes.get("owner", state.owner)
    source = await state.service.reader.resolve(owner.user_id, owner.storage_namespace, f"file:{fid}:{artifact}")
    assert source is not None
    return source, path


async def _note(state, sources, **changes):
    return await state.store.knowledge.create(user_id=state.owner.user_id, project_id=state.project.project_id,
        kind=changes.pop("kind", "decision"), text=changes.pop("text", "APPROVED NOTE: keep the task concise."),
        sources=sources, **changes)


async def _prepare(state, **changes):
    return await get_agent_knowledge_files(state.app).prepare(user_id=state.owner.user_id,
        project_id=state.project.project_id, objective="Complete the assigned task.", **changes)


async def _create(state, *, prepared=True, **changes):
    sources = await _prepare(state, **{key: changes[key] for key in ("knowledge_entry_id", "knowledge_entry_revision") if key in changes}) if prepared else None
    return await state.store.agents.create(user_id=state.owner.user_id, project_id=state.project.project_id,
        objective="Complete the assigned task.", request_id=uuid.uuid4().hex, knowledge_file_sources=sources, **changes)


@pytest.mark.parametrize(("kind", "artifact", "origin"), [("text", "text", "upload"), ("image", "text", "upload"),
    ("audio", "transcript", "upload"), ("video", "transcript", "upload"), ("video", "visual", "upload"),
    ("text", "summary", "upload"), ("text", "text", "generated")])
async def test_current_artifacts_include_only_approved_note_and_hash_pin(state, kind, artifact, origin):
    source, _ = await _source(state, kind=kind, artifact=artifact, origin=origin)
    entry = await _note(state, [ref(source)])
    captured = await _prepare(state)
    assert isinstance(captured, CapturedKnowledgeFileSources) and len(captured.proofs) == 1
    assert source.text not in repr(captured) and str(state.root) not in repr(captured)
    task = await state.store.agents.create(user_id=state.owner.user_id, project_id=state.project.project_id,
        objective="Complete the assigned task.", request_id="one", knowledge_file_sources=captured)
    assert task["project_context"]["knowledge"] == [{"id": entry["id"], "revision": entry["revision"]}]
    assert "file_inputs" not in task
    manifest = state.store._conn.execute("SELECT knowledge_json FROM app_agent_project_contexts").fetchone()[0]
    pin = json.loads(manifest)[0]
    assert set(pin) == {"id", "revision", "chars", "sources_sha256"} and len(pin["sources_sha256"]) == 64
    assert len(manifest) <= 4000 and source.text not in manifest and source.source_id not in manifest
    assert entry["text"] not in manifest and str(state.root) not in manifest
    claimed = await state.store.agents.claim()
    context = await state.store.agents.resolve_project_context(**lease(claimed))
    assert context["knowledge"][0]["text"] == entry["text"]
    assert source.text not in json.dumps(context)


async def test_mixed_database_and_file_sources_share_one_aggregate_proof(state):
    source, _ = await _source(state)
    state.reader = state.service.reader
    message_source, _ = await message(state)
    report_source, _ = await report(state)
    entry = await _note(state, [ref(source), ref(message_source), ref(report_source)])
    task = await _create(state)
    assert task["project_context"]["knowledge"] == [{"id": entry["id"], "revision": entry["revision"]}]
    claimed = await state.store.agents.claim()
    context = await state.store.agents.resolve_project_context(**lease(claimed))
    assert context["knowledge"][0]["text"] == entry["text"]
    assert all(source.text not in json.dumps(context) for source in (source, message_source, report_source))


async def test_unprepared_or_forged_batches_cannot_adopt_file_refs(state):
    source, _ = await _source(state)
    await _note(state, [ref(source)])
    fallback = await _note(state, [], text="Sourceless fallback.")
    task = await _create(state, prepared=False)
    assert task["project_context"]["knowledge"] == [{"id": fallback["id"], "revision": 1}]
    valid = await _prepare(state)
    for invalid in ({"proofs": valid.proofs}, dataclasses.replace(valid, owner_id=state.other.user_id),
                    dataclasses.replace(valid, project_id="another-project"), dataclasses.replace(valid, proofs=valid.proofs * 97)):
        with pytest.raises(AgentTaskError):
            await state.store.agents.create(user_id=state.owner.user_id, project_id=state.project.project_id,
                objective="Complete the assigned task.", request_id=uuid.uuid4().hex, knowledge_file_sources=invalid)
    forged = dataclasses.replace(valid, proofs=((source.source_id, str(source.revision), "a" * 64),))
    task = await state.store.agents.create(user_id=state.owner.user_id, project_id=state.project.project_id,
        objective="Complete the assigned task.", request_id="forged-hash", knowledge_file_sources=forged)
    assert task["project_context"]["knowledge"] == [{"id": fallback["id"], "revision": 1}]


@pytest.mark.parametrize("invalid", ["stale", "foreign", "private", "other_project", "missing", "wrong_artifact", "malformed", "unapproved"])
async def test_unavailable_or_unsupported_refs_do_not_hide_sourceless_fallback(state, invalid, monkeypatch):
    owner, project, private = state.owner, state.project, False
    if invalid == "foreign":
        owner = state.other
        project = await state.store.projects.create(user_id=owner.user_id, name="Foreign")
    elif invalid in {"private", "other_project"}:
        private = invalid == "private"
        project = await state.store.projects.create(user_id=owner.user_id, name="Other", private_required=private)
    source, _ = await _source(state, owner=owner, project=project, private=private)
    sources = [ref(source)]
    if invalid == "stale":
        sources[0]["revision"] = "old"
    elif invalid == "missing":
        sources[0]["source_id"] = "file:missing:text"
    elif invalid == "wrong_artifact":
        sources[0]["source_id"] = source.source_id.replace(":text", ":transcript")
    entry = await _note(state, sources, pinned=True)
    if invalid in {"malformed", "unapproved"}:
        with state.store._lock, state.store._conn:
            if invalid == "malformed":
                state.store._conn.execute("UPDATE app_project_knowledge SET sources_json=? WHERE entry_id=?",
                    (json.dumps([{"source_id": "file:" + chr(0xD800) + ":text", "revision": "x"}]), entry["id"]))
            else:
                state.store._conn.execute("UPDATE app_project_knowledge SET approval_state='proposed' WHERE entry_id=?", (entry["id"],))
    fallback = await _note(state, [], text="Sourceless fallback.")
    if invalid in {"foreign", "private", "other_project", "missing", "wrong_artifact", "malformed", "unapproved"}:
        monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Unavailable source body was opened"))
    task = await _create(state)
    assert task["project_context"]["knowledge"] == [{"id": fallback["id"], "revision": 1}]


async def test_restart_requires_binding_then_restores_metadata_fences_without_parsing(state, monkeypatch):
    source, _ = await _source(state)
    entry = await _note(state, [ref(source)])
    task = await _create(state)
    second = ApplicationStore(state.store.path)
    state.app.state.application_store = second
    try:
        assert not (await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["project_context"]["current"]
        inputs = get_agent_file_inputs(state.app)
        monkeypatch.setattr(inputs.reader, "_read_file", lambda *_: pytest.fail("Runtime metadata fence parsed a source"))
        assert await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == task
        claimed = await second.agents.claim()
        context = await second.agents.resolve_project_context(**lease(claimed))
        assert context["knowledge"][0]["text"] == entry["text"]
        second.agents.fence_execution(**lease(claimed))
    finally:
        second.close()
        state.app.state.application_store = state.store


@pytest.mark.parametrize("change", ["body", "delete", "detach", "reattach", "ocr_appears", "artifact", "namespace", "project", "catalog"])
async def test_file_provenance_changes_fence_all_task_boundaries(state, change):
    source, path = await _source(state, kind="video" if change == "artifact" else "text",
        artifact="transcript" if change == "artifact" else "text")
    await _note(state, [ref(source)])
    task = await _create(state)
    claimed = await state.store.agents.claim()
    if change in {"body", "ocr_appears", "artifact"}:
        target = ocr_text_path(path) if change == "ocr_appears" else path.with_name(source.target["file_id"] + ".transcript.txt") if change == "artifact" else path
        target.write_text("Changed source text.")
    elif change == "delete":
        await state.uploads.request_file_deletion(source.target["file_id"], user=state.owner.storage_namespace, requested_at="2026-11-03T12:00:00+00:00")
    elif change in {"detach", "reattach"}:
        await state.store.projects.remove_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=source.target["file_id"])
        if change == "reattach":
            await state.store.projects.add_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=source.target["file_id"])
    elif change == "project":
        await state.store.projects.delete(user_id=state.owner.user_id, project_id=state.project.project_id)
    elif change == "catalog":
        with state.uploads._lock, state.uploads._conn:
            state.uploads._conn.execute("UPDATE uploads SET filename='renamed.txt' WHERE file_id=?", (source.target["file_id"],))
    else:
        with state.store._lock, state.store._conn:
            state.store._conn.execute("UPDATE app_users SET storage_namespace='changed' WHERE user_id=?", (state.owner.user_id,))
    for operation in (state.store.agents.validate_lease(**lease(claimed)),
        state.store.agents.reserve_operation(**lease(claimed), step_id="plan"),
        state.store.agents.checkpoint(**lease(claimed), step_id="plan", result="Late Plan"),
        state.store.agents.complete(**lease(claimed), result="Late result")):
        with pytest.raises(AgentTaskContextError):
            await operation
    with pytest.raises(AgentTaskContextError):
        state.store.agents.fence_execution(**lease(claimed))
    assert not (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["project_context"]["current"]


async def test_duplicate_sources_decode_once_and_later_fences_never_parse(state, monkeypatch):
    source, _ = await _source(state)
    for index in range(12):
        await _note(state, [ref(source)], text=f"Approved task note {index}.")
    original, reads = state.service.reader._read_file, []

    def tracked(*args):
        reads.append(args[1]["file_id"])
        return original(*args)

    monkeypatch.setattr(state.service.reader, "_read_file", tracked)
    task = await _create(state)
    assert reads == [source.target["file_id"]] and len(task["project_context"]["knowledge"]) == 12
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Fence parsed source body"))
    claimed = await state.store.agents.claim()
    state.store.agents.fence_execution(**lease(claimed))
    await state.store.agents.resolve_project_context(**lease(claimed))
    manifest = state.store._conn.execute("SELECT knowledge_json FROM app_agent_project_contexts").fetchone()[0]
    assert len(manifest) <= 4000


async def test_preparation_is_bounded_to_twelve_sourced_candidates_and_fallback_survives(state, monkeypatch):
    for index in range(20):
        await _note(state, [{"source_id": f"file:missing{index}:text", "revision": "old"}], pinned=True)
    fallback = await _note(state, [], text="Sourceless fallback.")
    service = get_agent_knowledge_files(state.app)
    original, reads = service._metadata, []

    def tracked(user_id, project_id, source_id):
        reads.append(source_id)
        return original(user_id, project_id, source_id)

    monkeypatch.setattr(service, "_metadata", tracked)
    task = await _create(state)
    assert len(reads) == 24  # twelve preflight metadata reads plus twelve transactional proofs
    assert task["project_context"]["knowledge"] == [{"id": fallback["id"], "revision": 1}]


async def test_changed_file_between_prepare_and_commit_is_excluded_and_origin_rolls_back(state):
    source, path = await _source(state)
    entry = await _note(state, [ref(source)], kind="action")
    captured = await _prepare(state, knowledge_entry_id=entry["id"], knowledge_entry_revision=entry["revision"])
    path.write_text("Changed after source capture.")
    with pytest.raises(AgentTaskAccessError):
        await state.store.agents.create(user_id=state.owner.user_id, project_id=state.project.project_id,
            objective="Complete the assigned task.", request_id="origin", knowledge_file_sources=captured,
            knowledge_entry_id=entry["id"], knowledge_entry_revision=entry["revision"])
    assert await state.store.agents.list(user_id=state.owner.user_id) == []
    task = await state.store.agents.create(user_id=state.owner.user_id, project_id=state.project.project_id,
        objective="Complete the assigned task.", request_id="ordinary", knowledge_file_sources=captured)
    assert task["project_context"]["knowledge"] == []


async def test_file_changed_during_preparation_is_not_trusted(state, monkeypatch):
    source, path = await _source(state)
    await _note(state, [ref(source)])
    original = state.service.reader.resolve

    async def changed(*args):
        result = await original(*args)
        path.write_text("Changed after parser await.")
        return result

    monkeypatch.setattr(state.service.reader, "resolve", changed)
    assert (await _prepare(state)).proofs == ()


async def test_missing_original_with_retained_ocr_is_available_at_capture(state):
    source, path = await _source(state, kind="image")
    path.unlink()
    # The saved reference must identify the retained-text revision.
    source = await state.service.reader.resolve(state.owner.user_id, state.owner.storage_namespace, source.source_id)
    entry = await _note(state, [ref(source)])
    task = await _create(state)
    assert task["project_context"]["knowledge"] == [{"id": entry["id"], "revision": 1}]
