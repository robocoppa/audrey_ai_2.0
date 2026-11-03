"""Agent working outputs use real owned storage with fenced durable receipts."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_files import (
    agent_file_handler,
    agent_file_scope,
    create_agent_file,
    recover_agent_output_staging_for_startup,
)
from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.file_rendering import CreateFileRequest
from audrey.generated_files import _output_ids
from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
from audrey.kb.storage_lifecycle import StorageLifecycle
from audrey.kb.uploads_db import UploadsDB
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes import files as uploads
from audrey.routes.app import router


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner, other = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    admin = await store.resolve_external_identity(provider="test", subject="admin", email="admin@example.com",
        display_name="Admin", role="admin", auth_method="test")
    db = UploadsDB(tmp_path / "uploads.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.state.uploads_db = db
    app.state.storage_lifecycle = StorageLifecycle(db)
    app.state.file_operation_locks = FileOperationLocks()
    app.state.cfg = SimpleNamespace(raw={"kb": {"upload_root": str(tmp_path / "files"), "max_user_bytes": 1_000_000}})
    app.state.file_deletions = FileDeletionWorker(db=db, qdrant=AsyncMock(),
        upload_root=tmp_path / "files", locks=app.state.file_operation_locks)
    app.include_router(router)
    current = [owner]
    app.dependency_overrides[require_principal] = lambda: current[0]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(app=app, store=store, db=db, owner=owner, other=other, admin=admin, current=current,
                              client=client, tmp_path=tmp_path)
    db.close()
    app.state.application_store.close()


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def work(state, *, project=False):
    p = await state.store.projects.create(user_id=state.owner.user_id, name="Working files") if project else None
    await state.store.agents.create(user_id=state.owner.user_id, objective="Create a budget file.", request_id="first",
                                   project_id=p.project_id if p else None)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**lease(task), step_id="plan")
    return await state.store.agents.checkpoint(**lease(task), step_id="plan", result="Make a file."), p


async def create(state, task, **changes):
    return await create_agent_file(state.app, state.owner, task, {
        "filename": "budget.csv", "format": "csv", "rows": [["Item", "Amount"], ["Rent", 1200]], **changes,
    })


def receipts(state):
    return [dict(row) for row in state.store._conn.execute("SELECT * FROM app_agent_outputs").fetchall()]


async def test_download_owner_project_immutable_and_identical_concurrent_retry(state):
    task, project = await work(state, project=True)
    one, two = await asyncio.gather(create(state, task), create(state, task))
    assert one["file_id"] == two["file_id"]
    assert {one["created"], two["created"]} == {True, False}
    assert one["project_id"] == project.project_id
    download = await state.client.get(one["download_url"])
    assert download.status_code == 200 and '"Rent","1200"' in download.text
    assert download.headers["cache-control"] == "private, no-store"
    assert download.headers["x-content-type-options"] == "nosniff"
    row = await state.db.get_upload(one["file_id"])
    assert row["origin"] == "generated" and row["collection"] == "" and row["chunks"] == 0
    source = uploads._source_path(Request({"type": "http", "app": state.app}), row)
    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    assert receipts(state)[0]["status"] == "ready"
    assert set(receipts(state)[0]) == {"file_id", "task_id", "user_id", "filename", "identity_sha256", "status", "created_at", "committed_at"}
    assert [f.file_id for f in await state.store.projects.list_files(user_id=state.owner.user_id, project_id=project.project_id)] == [one["file_id"]]
    changed = await create(state, task, rows=[["Different"]])
    assert changed["file_id"] != one["file_id"]
    assert (await state.client.get(one["download_url"])).text == download.text
    state.current[0] = state.other
    assert (await state.client.get(one["download_url"])).status_code == 404
    assert (await state.client.get("/api/files")).json()["items"] == []


@pytest.mark.parametrize("file_format,body,expected", [
    ("txt", {"content": "A draft."}, b"A draft."),
    ("md", {"content": "# A draft"}, b"# A draft"),
    ("json", {"data": {"items": [1, 2]}}, b'"items"'),
    ("docx", {"content": "# Heading", "content_format": "markdown"}, b"PK"),
    ("xlsx", {"sheets": [{"name": "Budget", "rows": [["Amount"], [12]]}]}, b"PK"),
])
async def test_existing_safe_renderer_formats(state, file_format, body, expected):
    task, _ = await work(state)
    result = await create_agent_file(state.app, state.owner, task,
                                    {"filename": "draft." + file_format, "format": file_format, **body})
    download = await state.client.get(result["download_url"])
    assert download.status_code == 200 and expected in download.content


async def test_saved_links_survive_review_omission_and_task_history_deletion(state):
    task, _ = await work(state)
    output = await create(state, task)
    await state.store.agents.reserve_operation(**lease(task), step_id="research")
    task = await state.store.agents.checkpoint(**lease(task), step_id="research", result="Completed the budget.")
    assert output["download_url"] in task["steps"][1]["result"]
    assert task["outputs"] == [{key: output[key] for key in ("file_id", "filename", "download_url", "created_at")}]
    await state.store.agents.reserve_operation(**lease(task), step_id="review")
    task = await state.store.agents.checkpoint(**lease(task), step_id="review", result="Checked.")
    final = await state.store.agents.complete(**lease(task), result="Checked.")
    assert output["download_url"] in final["result"]
    await state.store.agents.delete(user_id=state.owner.user_id, task_id=task["id"], expected_revision=final["revision"])
    assert receipts(state) == []
    assert (await state.client.get(output["download_url"])).status_code == 200


async def test_utf8_saved_links_stay_within_byte_limit(state):
    task, _ = await work(state)
    output = await create(state, task)
    await state.store.agents.reserve_operation(**lease(task), step_id="research")
    task = await state.store.agents.checkpoint(**lease(task), step_id="research", result="😀" * 16000)
    saved = task["steps"][1]["result"]
    assert len(saved.encode()) <= 64000 and output["download_url"] in saved


async def test_deleted_file_tombstone_blocks_retry_without_new_bytes(state):
    task, _ = await work(state)
    output = await create(state, task)
    response = await state.client.delete("/api/files/" + output["file_id"])
    assert response.status_code == 200
    repeated = await create(state, task)
    assert repeated["error"] == "generated_file_deleted"
    assert await state.db.get_upload(output["file_id"]) is None
    assert len(receipts(state)) == 1


async def test_deletion_requested_after_catalog_read_blocks_replay_publication(state, monkeypatch):
    task, _ = await work(state)
    output = await create(state, task)
    original = asyncio.to_thread

    async def intercepted(function, /, *args, **kwargs):
        if function.__name__ == "publish":
            assert await state.db.request_file_deletion(output["file_id"], user=state.owner.storage_namespace,
                                                       requested_at="2026-11-02T00:00:00+00:00")
        return await original(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", intercepted)
    result = await create(state, task)
    assert result["error"] == "generated_file_deleted"
    assert (await state.db.get_upload(output["file_id"]))["status"] == "deleting"


async def test_restart_reconciles_ready_and_catalog_published_pending_without_rewrite(state, monkeypatch):
    task, _ = await work(state)
    import audrey.agent_files as module
    transaction = module.write_transaction

    @contextmanager
    def interrupted_application_commit(connection):
        with transaction(connection):
            yield
            # The nested upload transaction has committed by this point. An
            # application commit failure leaves a durable catalog file with
            # a pending receipt, the real cross-database recovery boundary.
            raise sqlite3.IntegrityError("fixture")

    monkeypatch.setattr(module, "write_transaction", interrupted_application_commit)
    with pytest.raises(sqlite3.IntegrityError):
        await create(state, task)
    assert len(await state.db.list_user(state.owner.storage_namespace)) == 1
    assert receipts(state)[0]["status"] == "reserved"
    monkeypatch.setattr(module, "write_transaction", transaction)
    with state.store._lock:
        state.store._conn.execute("UPDATE app_agent_tasks SET lease_expires_at='2000-01-01T00:00:00+00:00'")
        state.store._conn.commit()
    state.store.close()
    state.store = ApplicationStore(state.tmp_path / "app.sqlite")
    state.app.state.application_store = state.store
    previous_fence = task["fence"]
    task = await state.store.agents.claim()
    assert task["fence"] > previous_fence and task["current_step"] == "research"
    monkeypatch.setattr(module, "_write_new", lambda *_: pytest.fail("Existing output bytes were rewritten"))
    recovered = await create(state, task)
    assert recovered["created"] is False and receipts(state)[0]["status"] == "ready"
    assert (await create(state, task))["file_id"] == recovered["file_id"]
    assert len(await state.db.list_user(state.owner.storage_namespace)) == 1


async def test_receipt_failure_before_catalog_commit_rolls_back_upload_and_cleans_staging(state):
    task, _ = await work(state)
    with state.store._lock:
        state.store._conn.execute("CREATE TRIGGER reject_ready BEFORE UPDATE OF status ON app_agent_outputs "
                                  "WHEN NEW.status='ready' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        state.store._conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        await create(state, task)
    assert await state.db.list_user(state.owner.storage_namespace) == []
    assert receipts(state)[0]["status"] == "reserved"
    assert (await state.db.quota_usage(state.owner.storage_namespace)).total_bytes == 0
    assert not list((state.tmp_path / "files").rglob("*.csv"))
    assert not list((state.tmp_path / "files").rglob("*.ocr.txt"))


@pytest.mark.parametrize("change", ["cancel", "project_delete", "model_grant", "disabled", "expired"])
async def test_final_publication_rechecks_authority_after_last_io(state, monkeypatch, change):
    task, project = await work(state, project=True)
    original = asyncio.to_thread
    changed = False

    async def intercepted(function, /, *args, **kwargs):
        nonlocal changed
        if function.__name__ == "publish" and not changed:
            changed = True
            if change == "cancel":
                current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
                await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=current["revision"], action="cancel")
            elif change == "project_delete":
                await state.store.projects.delete(user_id=state.owner.user_id, project_id=project.project_id)
            elif change == "model_grant":
                await state.store.set_model_access_policy(actor_user_id=state.admin.user_id, model_id="local",
                                                          enabled=False, audience="users")
            elif change == "disabled":
                state.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
            else:
                with state.store._lock:
                    state.store._conn.execute("UPDATE app_agent_tasks SET lease_expires_at='2000-01-01T00:00:00+00:00'")
                    state.store._conn.commit()
        return await original(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", intercepted)
    with pytest.raises(AgentDispatchStoppedError):
        await create(state, task)
    assert changed and await state.db.list_user(state.owner.storage_namespace) == []
    assert receipts(state)[0]["status"] == "reserved"
    assert not list((state.tmp_path / "files").rglob("*.csv"))


async def test_limits_persist_pending_and_deleted_identities_across_rerun(state):
    task, _ = await work(state)
    for index in range(8):
        result = await create(state, task, filename=f"budget-{index}.csv")
        assert "error" not in result
    assert (await create(state, task, filename="ninth.csv"))["error"] == "generated_file_limit"
    assert len(receipts(state)) == 8
    assert "error" not in await create(state, task, filename="budget-0.csv")
    first = next(item for item in receipts(state) if item["filename"] == "budget-0.csv")
    assert (await state.client.delete("/api/files/" + first["file_id"])).status_code == 200
    current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    cancelled = await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=current["revision"], action="cancel")
    await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"], action="rerun")
    resumed = await state.store.agents.claim()
    assert (await create(state, resumed, filename="ninth.csv"))["error"] == "generated_file_limit"


async def test_quota_rejection_keeps_pending_identity_and_no_orphans(state):
    task, _ = await work(state)
    state.app.state.cfg.raw["kb"]["max_user_bytes"] = 1
    result = await create(state, task)
    assert result["error"] == "file_quota_exceeded"
    assert receipts(state)[0]["status"] == "reserved"
    assert await state.db.list_user(state.owner.storage_namespace) == []
    assert not list((state.tmp_path / "files").rglob("*.csv"))


async def test_pending_quota_failures_consume_lifetime_identities_but_identical_retry_can_recover(state):
    task, _ = await work(state)
    state.app.state.cfg.raw["kb"]["max_user_bytes"] = 1
    for index in range(8):
        assert (await create(state, task, filename=f"pending-{index}.csv"))["error"] == "file_quota_exceeded"
    assert len(receipts(state)) == 8
    state.app.state.cfg.raw["kb"]["max_user_bytes"] = 1_000_000
    assert (await create(state, task, filename="ninth.csv"))["error"] == "generated_file_limit"
    assert "error" not in await create(state, task, filename="pending-0.csv")


async def test_revoked_scope_foreign_identity_and_private_policy_refuse(state):
    task, _ = await work(state)
    assert agent_file_handler(user_id=state.owner.storage_namespace) is None
    files = []
    with agent_file_scope(app=state.app, principal=state.owner, task=task, files=files):
        handler = agent_file_handler(user_id=state.owner.storage_namespace)
        with pytest.raises(AgentDispatchStoppedError):
            agent_file_handler(user_id=state.other.storage_namespace)
        output = await handler({"filename": "notes.txt", "format": "txt", "content": "Draft"})
    assert files[0]["file_id"] == output["file_id"]
    with pytest.raises(AgentDispatchStoppedError):
        await handler({"filename": "late.txt", "format": "txt", "content": "Late"})
    with pytest.raises(AgentDispatchStoppedError):
        await create_agent_file(state.app, state.other, task, {"filename": "foreign.txt", "format": "txt", "content": "No"})
    policy = ProcessingPolicy(local_models={"qwen3.8:latest": "a" * 64}, ollama_base_url="http://ollama",
                              allowed_tools=frozenset({"create_file"}), tool_servers=frozenset({"http://ollama"}))
    with processing_scope(policy), pytest.raises(AgentDispatchStoppedError):
        await create(state, task)


async def test_invalid_requests_never_reserve_or_write(state):
    task, _ = await work(state)
    bad = await create(state, task, filename="../budget.csv")
    assert bad["error"] == "arguments_invalid" and receipts(state) == []
    assert await state.db.list_user(state.owner.storage_namespace) == []


async def test_schema33_upgrade_preserves_task_and_adds_owner_fenced_metadata_only(state):
    task, _ = await work(state)
    with state.store._lock:
        state.store._conn.execute("DROP TABLE app_agent_outputs")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version=34")
        state.store._conn.commit()
    state.store.close()
    state.store = ApplicationStore(state.tmp_path / "app.sqlite")
    state.app.state.application_store = state.store
    assert state.store.schema_version == 35
    restored = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert restored["steps"] == task["steps"]
    with state.store._lock, pytest.raises(sqlite3.IntegrityError):
        state.store._conn.execute(
            "INSERT INTO app_agent_outputs(file_id,task_id,user_id,filename,identity_sha256,status,created_at) "
            "VALUES (?,?,?,?,?,'reserved','2026-11-02')",
            (str(uuid.uuid4()), task["id"], state.other.user_id, "notes.txt", "a" * 64),
        )
    state.store._conn.rollback()
    assert state.store._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert "error" not in await create(state, task)


async def test_startup_cleans_exact_unregistered_staging_and_quota_keeps_intent_then_retry_works(state):
    task, _ = await work(state)
    owned = await create(state, task)
    body = CreateFileRequest(filename="staged.txt", format="txt", content="A staged draft.")
    file_id = _output_ids(state.owner, "agent:" + task["id"], False, body)[-1]
    identity = hashlib.sha256(json.dumps([file_id, body.model_dump(mode="json", exclude_none=True)],
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    await state.store.agents.reserve_output(**lease(task), file_id=file_id, filename=body.filename,
                                           identity_sha256=identity)
    request = Request({"type": "http", "app": state.app})
    source = uploads._source_path(request, {"user": state.owner.storage_namespace, "file_id": file_id, "filename": body.filename})
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(body.content)
    sidecar = source.with_name(file_id + ".ocr.txt")
    sidecar.write_text(body.content)
    now, expired = uploads._reservation_window(request)
    await state.app.state.storage_lifecycle.reserve_single_upload(reservation_id=file_id,
        user=state.owner.storage_namespace, bytes_=100, max_user_bytes=1_000_000, now=now, expired_before=expired)
    state.store.close()
    state.store = ApplicationStore(state.tmp_path / "app.sqlite")
    state.app.state.application_store = state.store
    assert await recover_agent_output_staging_for_startup(state.app, batch_size=1) == 1
    assert not source.exists() and not sidecar.exists()
    assert (await state.app.state.storage_lifecycle.usage(state.owner.storage_namespace)).single_shot_bytes == 0
    assert next(item for item in receipts(state) if item["file_id"] == file_id)["status"] == "reserved"
    assert (await state.client.get(owned["download_url"])).status_code == 200
    recreated = await create_agent_file(state.app, state.owner, task, body.model_dump(exclude_unset=True))
    assert recreated["file_id"] == file_id and recreated["created"]
    assert await recover_agent_output_staging_for_startup(state.app) == 0


async def test_startup_retains_cataloged_pending_output_for_authorized_adoption(state):
    task, _ = await work(state)
    owned = await create(state, task)
    with state.store._lock:
        state.store._conn.execute("UPDATE app_agent_outputs SET status='reserved',committed_at=NULL")
        state.store._conn.commit()
    assert await recover_agent_output_staging_for_startup(state.app) == 0
    assert receipts(state)[0]["status"] == "reserved"
    assert (await state.client.get(owned["download_url"])).status_code == 200
    assert (await create(state, task))["created"] is False
    assert receipts(state)[0]["status"] == "ready"


async def test_startup_recovery_refuses_running_worker_and_hides_storage_error(state, monkeypatch):
    state.app.state.agent_runtime = SimpleNamespace(_worker=object())
    with pytest.raises(RuntimeError, match="requires startup"):
        await recover_agent_output_staging_for_startup(state.app)
    state.app.state.agent_runtime = SimpleNamespace(_worker=None)
    task, _ = await work(state)
    state.app.state.cfg.raw["kb"]["max_user_bytes"] = 1
    assert (await create(state, task))["error"] == "file_quota_exceeded"
    monkeypatch.setattr(state.db, "get_upload", AsyncMock(side_effect=OSError("sensitive/path/contents")))
    with pytest.raises(RuntimeError) as error:
        await recover_agent_output_staging_for_startup(state.app)
    assert str(error.value) == "Agent output staging recovery failed."
