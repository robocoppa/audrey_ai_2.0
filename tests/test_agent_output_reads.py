"""Actual saved Agent outputs are bounded, owner-fenced local read authority."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
import uuid
import zipfile

import pytest
from fastapi import Request
from test_agent_files import create, finish_file_work, lease, work
from test_agent_files import state as state

from audrey import agent_output_reads as reads
from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_file_inputs import get_agent_file_inputs
from audrey.agent_files import agent_file_handler, agent_file_scope, create_agent_file
from audrey.app_state import ApplicationStore
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes import files as uploads
from audrey.tools.agent_output_reads import agent_output_read_handler, with_output_file_read
from audrey.tools.discovery import ToolRegistry
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


async def read(state, task, **arguments):
    with agent_file_scope(app=state.app, principal=state.owner, task=task, files=[]):
        return await agent_output_read_handler(user_id=state.owner.storage_namespace)(arguments)


async def source(state, output):
    row = await state.db.get_upload(output["file_id"])
    return uploads._source_path(Request({"type": "http", "app": state.app}), row)


@pytest.mark.parametrize("stage", ["research", "review"])
@pytest.mark.parametrize("file_format,body,expected", [
    ("txt", {"content": "Original 😀 text"}, "Original 😀 text"),
    ("md", {"content": "# Original 😀\n\nSaved text"}, "# Original 😀\n\nSaved text"),
    ("csv", {"rows": [["Item", "Amount"], ["Saved", 260.30]]}, '"Saved","260.3"'),
    ("json", {"data": {"saved": "Original 😀", "total": 260.3}}, '"total": 260.3'),
    ("docx", {"content": "# Original 😀\n\n| Item | Amount |\n|---|---|\n| Saved | 260.30 |", "content_format": "markdown"}, "Saved\t260.30"),
    ("xlsx", {"sheets": [{"name": "Budget", "rows": [["Item", "Amount"], ["Saved", 260.3]]}]}, "Saved\t260.3"),
])
async def test_six_actual_download_formats_in_work_and_review(state, stage, file_format, body, expected):
    task, _ = await work(state, stage=stage)
    output = await create_agent_file(state.app, state.owner, task,
        {"filename": "saved." + file_format, "format": file_format, **body})
    path = await source(state, output)
    path.with_name(output["file_id"] + ".ocr.txt").write_text("WRONG SIDECAR CONTENT", encoding="utf-8")
    download = await state.client.get(output["download_url"])
    assert download.status_code == 200
    result = await read(state, task, file_id=output["file_id"])
    assert expected in result["text"] and "WRONG SIDECAR" not in result["text"]
    assert result["revision"] == hashlib.sha256(download.content).hexdigest()
    assert result["origin"] == "generated" and result["artifact"] == "text"
    assert result["representation"] == ("extracted_text" if file_format in {"docx", "xlsx"} else "plain_text")
    if file_format not in {"docx", "xlsx"}:
        assert result["text"] == download.content.decode("utf-8")
    assert result["start"] == 0 and result["end"] == result["total_chars"] == len(result["text"])
    assert result["next_offset"] is None and result["has_more"] is False
    assert str(path.parent) not in json.dumps(result)


async def test_read_new_receipt_in_same_native_scope(state):
    task, _ = await work(state)
    with agent_file_scope(app=state.app, principal=state.owner, task=task, files=[]):
        output = await agent_file_handler(user_id=state.owner.storage_namespace)(
            {"filename": "same-step.md", "format": "md", "content": "Just saved"})
        result = await agent_output_read_handler(user_id=state.owner.storage_namespace)({"file_id": output["file_id"]})
    assert result["text"] == "Just saved"


async def test_unicode_escape_budget_and_truthful_continuation(state):
    task, _ = await work(state)
    text = '😀"\\\n\t' * 9000
    output = await create_agent_file(state.app, state.owner, task,
        {"filename": "escaped.txt", "format": "txt", "content": text})
    offset, parts, revision = 0, [], None
    while True:
        arguments = {"file_id": output["file_id"], "offset": offset, "limit": 6000}
        if revision:
            arguments["expected_revision"] = revision
        page = await read(state, task, **arguments)
        assert len(json.dumps(page, ensure_ascii=False)) <= 5000
        assert page["start"] == offset and page["text"] == text[offset:page["end"]]
        assert page["end"] > offset
        assert page["total_chars"] == len(text)
        assert revision is None or page["revision"] == revision
        revision = page["revision"]
        parts.append(page["text"])
        if not page["has_more"]:
            assert page["next_offset"] is None and page["end"] == len(text)
            break
        offset = page["next_offset"]
        assert offset == page["end"]
    assert "".join(parts) == text
    end = await read(state, task, file_id=output["file_id"], offset=len(text), expected_revision=revision)
    assert end["text"] == "" and end["has_more"] is False
    assert (await read(state, task, file_id=output["file_id"], offset=len(text) + 1))["error"] == "offset_invalid"
    assert (await read(state, task, file_id=output["file_id"], expected_revision="0" * 64))["error"] == "output_changed"


@pytest.mark.parametrize("arguments", [
    None, [], {}, {"file_id": ""}, {"file_id": "../secret"}, {"file_id": "x/y"},
    {"file_id": "x", "path": "/unselected/secret"}, {"file_id": "x", "user_id": "other"},
    {"file_id": "x", "task_id": "other"}, {"file_id": "x", "offset": True},
    {"file_id": "x", "offset": -1}, {"file_id": "x", "offset": 500001},
    {"file_id": "x", "offset": 1.5}, {"file_id": "x", "limit": False},
    {"file_id": "x", "limit": 0}, {"file_id": "x", "limit": 6001},
    {"file_id": "x", "expected_revision": None}, {"file_id": "x", "expected_revision": "A" * 64},
    {"file_id": "x", "expected_revision": "0" * 63},
])
async def test_strict_arguments_do_not_reach_source(state, monkeypatch, arguments):
    task, _ = await work(state)
    monkeypatch.setattr(reads, "_read_source", lambda *_: pytest.fail("Invalid arguments reached disk"))
    result = await reads.read_agent_output(state.app, state.owner, task, arguments)
    assert result["error"] == "arguments_invalid"


@pytest.mark.parametrize("change", ["unknown", "other_task", "other_owner", "reserved", "private", "upload_origin", "filename", "not_ready", "deleted"])
async def test_unauthorized_outputs_uniformly_refused_before_parse(state, monkeypatch, change):
    task, _ = await work(state)
    output = await create(state, task)
    file_id = output["file_id"]
    if change == "unknown":
        file_id = "unknown-output"
    elif change == "deleted":
        assert await state.db.request_file_deletion(file_id, user=state.owner.storage_namespace,
                                                   requested_at="2026-11-20T00:00:00+00:00")
    elif change in {"other_task", "other_owner"}:
        owner_id = state.other.user_id if change == "other_owner" else state.owner.user_id
        other = await state.store.agents.create(user_id=owner_id, objective="Other task", request_id="other")
        with state.store._lock:
            state.store._conn.execute("UPDATE app_agent_outputs SET task_id=?,user_id=? WHERE file_id=?",
                                       (other["id"], owner_id, file_id))
            state.store._conn.commit()
    elif change == "reserved":
        with state.store._lock:
            state.store._conn.execute("UPDATE app_agent_outputs SET status='reserved',committed_at=NULL WHERE file_id=?", (file_id,))
            state.store._conn.commit()
    elif change in {"private", "upload_origin"}:
        # A distinct supported canonical upload record, rather than violating
        # the immutable origin/processing triggers on an existing row.
        with state.db._lock:
            state.db._conn.execute("DELETE FROM uploads WHERE file_id=?", (file_id,))
            state.db._record_locked(file_id, state.owner.storage_namespace, "budget.csv", "text/csv", 33,
                "text", "", 0, "2026-11-20T00:00:00+00:00", "ready", private_required=change == "private",
                origin="generated" if change == "private" else "upload")
            state.db._conn.commit()
    else:
        with state.db._lock:
            if change == "filename":
                state.db._conn.execute("UPDATE uploads SET filename='different.csv' WHERE file_id=?", (file_id,))
            else:
                state.db._conn.execute("UPDATE uploads SET status='pending' WHERE file_id=?", (file_id,))
            state.db._conn.commit()
    monkeypatch.setattr(reads, "_read_source", lambda *_: pytest.fail("Unauthorized output reached parser"))
    assert await read(state, task, file_id=file_id) == reads._UNAVAILABLE


@pytest.mark.parametrize("change", ["missing", "empty", "invalid_utf8", "nul", "oversized", "file_symlink", "root_symlink", "directory_symlink", "fifo"])
async def test_actual_source_required_and_safe_without_sidecar_fallback(state, change):
    task, _ = await work(state)
    output = await create_agent_file(state.app, state.owner, task,
        {"filename": "source.txt", "format": "txt", "content": "Original saved body"})
    path = await source(state, output)
    assert path.with_name(output["file_id"] + ".ocr.txt").read_text() == "Original saved body"
    if change == "missing":
        path.unlink()
    elif change == "empty":
        path.write_bytes(b"")
    elif change == "invalid_utf8":
        path.write_bytes(b"\xff")
    elif change == "nul":
        path.write_bytes(b"text\x00secret")
    elif change == "oversized":
        with path.open("wb") as stream:
            stream.truncate(reads.MAX_OUTPUT_BYTES + 1)
    elif change == "file_symlink":
        target = state.tmp_path / "secret.txt"
        target.write_text("OTHER FILE SECRET")
        path.unlink()
        path.symlink_to(target)
    elif change in {"root_symlink", "directory_symlink"}:
        directory = path.parent.parent if change == "root_symlink" else path.parent
        renamed = directory.with_name(directory.name + "-moved")
        directory.rename(renamed)
        directory.symlink_to(renamed, target_is_directory=True)
    else:
        path.unlink()
        os.mkfifo(path)
    assert await read(state, task, file_id=output["file_id"]) == reads._UNAVAILABLE


@pytest.mark.parametrize("file_format,change", [("docx", "broken"), ("xlsx", "broken"), ("docx", "expanded"), ("xlsx", "parser_error")])
async def test_untrusted_office_parser_failures_are_content_free(state, monkeypatch, file_format, change):
    task, _ = await work(state)
    body = {"content": "Saved original"} if file_format == "docx" else {"sheets": [{"name": "Data", "rows": [["Saved"]]}]}
    output = await create_agent_file(state.app, state.owner, task,
        {"filename": "office." + file_format, "format": file_format, **body})
    path = await source(state, output)
    if change == "broken":
        path.write_bytes(b"PK OTHER FILE SECRET")
    elif change == "expanded":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("untrusted", b"x" * (8 * 1024 * 1024 + 1))
        path.write_bytes(buffer.getvalue())
    else:
        def fail(*_):
            raise RuntimeError("OTHER FILE SECRET /private/path")
        monkeypatch.setattr(reads, "_office_text", fail)
    assert await read(state, task, file_id=output["file_id"]) == reads._UNAVAILABLE


@pytest.mark.parametrize("stage", ["research", "review"])
@pytest.mark.parametrize("change", ["cancel", "project_delete", "model", "disabled", "expiry", "owner", "purge", "stage", "private_reads"])
async def test_attempt_authority_rechecked_after_await(state, monkeypatch, stage, change):
    task, project = await work(state, project=True, stage=stage)
    output = await create(state, task)
    original = reads._disk

    async def changed(*args):
        result = await original(*args)
        if change == "cancel":
            current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
            await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=current["revision"], action="cancel")
        elif change == "project_delete":
            await state.store.projects.delete(user_id=state.owner.user_id, project_id=project.project_id)
        elif change == "model":
            await state.store.set_model_access_policy(actor_user_id=state.admin.user_id, model_id="local", enabled=False, audience="users")
        elif change == "disabled":
            state.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
        elif change == "private_reads":
            block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="output-read-test")
        elif change == "stage":
            await state.store.agents.reserve_operation(**lease(task), step_id=stage)
            await state.store.agents.checkpoint(**lease(task), step_id=stage, result="Stage completed concurrently.")
        else:
            with state.store._lock:
                if change == "expiry":
                    state.store._conn.execute("UPDATE app_agent_tasks SET lease_expires_at='2000-01-01T00:00:00+00:00'")
                elif change == "owner":
                    state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
                elif change == "purge":
                    state.store._conn.execute("DELETE FROM app_agent_tasks WHERE task_id=?", (task["id"],))
                state.store._conn.commit()
        return result

    monkeypatch.setattr(reads, "_disk", changed)
    try:
        with pytest.raises(AgentDispatchStoppedError):
            await read(state, task, file_id=output["file_id"])
    finally:
        unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="output-read-test")


@pytest.mark.parametrize("change", ["receipt", "catalog", "source", "delete"])
async def test_output_changes_after_io_discard_text(state, monkeypatch, change):
    task, _ = await work(state)
    output = await create(state, task)
    path = await source(state, output)
    original = reads._disk

    async def changed(*args):
        result = await original(*args)
        if change == "source":
            path.write_text("REPLACED AFTER READ")
        elif change == "delete":
            await state.db.request_file_deletion(output["file_id"], user=state.owner.storage_namespace,
                                                requested_at="2026-11-20T00:00:00+00:00")
        elif change == "receipt":
            with state.store._lock:
                state.store._conn.execute("UPDATE app_agent_outputs SET identity_sha256=?", ("f" * 64,))
                state.store._conn.commit()
        else:
            with state.db._lock:
                state.db._conn.execute("UPDATE uploads SET filename='changed.csv'")
                state.db._conn.commit()
        return result

    monkeypatch.setattr(reads, "_disk", changed)
    assert await read(state, task, file_id=output["file_id"]) == reads._UNAVAILABLE


async def test_selected_source_change_fences_generated_output_read(state, monkeypatch):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Selected sources")
    file_id = str(uuid.uuid4())
    directory = state.tmp_path / "files" / state.owner.storage_namespace
    directory.mkdir(parents=True)
    path = directory / (file_id + ".txt")
    path.write_text("Selected original source text", encoding="utf-8")
    await state.db.record_upload(file_id=file_id, user=state.owner.storage_namespace, filename="selected.txt",
        mime="text/plain", bytes_=path.stat().st_size, kind="text", collection="", chunks=0,
        uploaded_at="2026-11-20T00:00:00+00:00")
    await state.store.projects.add_file(user_id=state.owner.user_id, project_id=project.project_id,
                                       file_id=file_id, file_is_private=False)
    service = get_agent_file_inputs(state.app)
    captured = await service.capture(state.owner, project.project_id, [file_id])
    await state.store.agents.create(user_id=state.owner.user_id, objective="Use the selected source.",
        request_id="selected-output", project_id=project.project_id, file_inputs=captured)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**lease(task), step_id="plan")
    task = await state.store.agents.checkpoint(**lease(task), step_id="plan", result="Prepare the output.")
    output = await create(state, task)
    original = reads._disk

    async def changed(*args):
        result = await original(*args)
        path.write_text("Selected source changed during the output read", encoding="utf-8")
        return result

    monkeypatch.setattr(reads, "_disk", changed)
    with pytest.raises(AgentDispatchStoppedError):
        await read(state, task, file_id=output["file_id"])


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_scope_callbacks_expire_and_private_or_foreign_calls_stop(state, monkeypatch, stage):
    task, _ = await work(state, stage=stage)
    output = await create(state, task)
    assert agent_output_read_handler(user_id=state.owner.storage_namespace) is None
    ready, proceed = asyncio.Event(), asyncio.Event()

    async def inherited():
        handler = agent_output_read_handler(user_id=state.owner.storage_namespace)
        ready.set()
        await proceed.wait()
        return await handler({"file_id": output["file_id"]})

    with agent_file_scope(app=state.app, principal=state.owner, task=task, files=[]):
        handler = agent_output_read_handler(user_id=state.owner.storage_namespace)
        child = asyncio.create_task(inherited())
        await ready.wait()
        with pytest.raises(AgentDispatchStoppedError):
            agent_output_read_handler(user_id=state.other.storage_namespace)
    monkeypatch.setattr(reads, "_read_source", lambda *_: pytest.fail("Expired scope parsed source"))
    with pytest.raises(AgentDispatchStoppedError):
        await handler({"file_id": output["file_id"]})
    proceed.set()
    with pytest.raises(AgentDispatchStoppedError):
        await child
    with pytest.raises(AgentDispatchStoppedError):
        await reads.read_agent_output(state.app, state.other, task, {"file_id": output["file_id"]})
    policy = ProcessingPolicy(local_models={"qwen3.8:latest": "a" * 64}, ollama_base_url="http://ollama",
        allowed_tools=frozenset({"create_file"}), tool_servers=frozenset({"http://ollama"}))
    with processing_scope(policy), pytest.raises(AgentDispatchStoppedError):
        await reads.read_agent_output(state.app, state.owner, task, {"file_id": output["file_id"]})
    with pytest.raises(AgentDispatchStoppedError):
        await read(state, {**task, "current_step": "plan"}, file_id=output["file_id"])


async def test_scope_exit_during_disk_read_cannot_return_body(state, monkeypatch):
    task, _ = await work(state)
    output = await create(state, task)
    started, release = asyncio.Event(), asyncio.Event()
    original = reads._disk

    async def paused(*args):
        result = await original(*args)
        started.set()
        await release.wait()
        return result

    monkeypatch.setattr(reads, "_disk", paused)
    with agent_file_scope(app=state.app, principal=state.owner, task=task, files=[]):
        handler = agent_output_read_handler(user_id=state.owner.storage_namespace)
        pending = asyncio.create_task(handler({"file_id": output["file_id"]}))
        await started.wait()
    release.set()
    with pytest.raises(AgentDispatchStoppedError):
        await pending


@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancel_settles_started_disk_worker_without_returning_content(state, monkeypatch, worker_fails):
    task, _ = await work(state)
    output = await create(state, task)
    started, release, settled = threading.Event(), threading.Event(), threading.Event()
    original = reads._read_source

    def paused(*args):
        started.set()
        assert release.wait(5)
        try:
            if worker_fails:
                raise OSError("Source disappeared while the cancelled worker settled")
            return original(*args)
        finally:
            settled.set()

    monkeypatch.setattr(reads, "_read_source", paused)
    pending = asyncio.create_task(read(state, task, file_id=output["file_id"]))
    assert await asyncio.to_thread(started.wait, 5)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert settled.is_set()


async def test_restart_rerun_and_immutable_versions_do_not_need_saved_body_cache(state):
    task, _ = await work(state)
    first = await create(state, task)
    second = await create(state, task, rows=[["Corrected", 260.3]])
    original = await read(state, task, file_id=first["file_id"])
    corrected = await read(state, task, file_id=second["file_id"])
    assert original["revision"] != corrected["revision"]
    completed = await finish_file_work(state, task)
    await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"],
                                     expected_revision=completed["revision"], action="rerun")
    second_store = ApplicationStore(state.store.path)
    state.app.state.application_store = second_store
    try:
        rerun = await second_store.agents.claim()
        assert rerun["id"] == task["id"] and rerun["steps"][0] == task["steps"][0]
        assert await read(state, rerun, file_id=first["file_id"], expected_revision=original["revision"]) == original
        assert await read(state, rerun, file_id=second["file_id"], expected_revision=corrected["revision"]) == corrected
        assert (await state.client.get(first["download_url"])).content.decode() == original["text"]
        assert (await state.client.get(second["download_url"])).content.decode() == corrected["text"]
        receipts = second_store._conn.execute("SELECT * FROM app_agent_outputs").fetchall()
        assert all(original["text"] not in json.dumps(dict(receipt)) for receipt in receipts)
    finally:
        second_store.close()
        state.app.state.application_store = state.store


def test_native_spec_has_only_output_id_character_cursor_and_revision():
    shared = ToolRegistry()
    first, second = with_output_file_read(shared), with_output_file_read(shared)
    assert shared.names() == [] and first.get("read_output_file") is not second.get("read_output_file")
    spec = first.get("read_output_file")
    assert spec.in_process_only and spec.path == "/read_output_file"
    assert set(spec.parameters["properties"]) == {"file_id", "offset", "limit", "expected_revision"}
    assert spec.parameters["additionalProperties"] is False
