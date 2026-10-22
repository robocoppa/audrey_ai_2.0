"""Admission rechecks real canonical permissions and exact Private source identity."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native
from test_private_project_runs import EVIDENCE, _ready_project
from test_tax_review_runtime import _RELEASE, _enable

from audrey.app_state.private_tasks import (
    PrivateTaskAccessError,
    PrivateTaskConflictError,
    PrivateTaskError,
)
from audrey.kb.user_store import sanitize_user
from audrey.private_task_admission import PrivateTaskAdmission
from audrey.private_task_runner import PrivateTaskStepRunner, StepResult
from audrey.processing_policy import ProcessingPolicyError
from audrey.tools.private_dispatch import private_tool_handler
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


@pytest.fixture
async def admission(native, monkeypatch):
    state = native
    # Upload and select a real local text source using the native Project API.
    project_id, cid, file_id = await _ready_project(state)
    _enable(state)

    async def prepare(_app, *, tax_review=False):
        assert _app is state.app and tax_review
        return state.env.runtime

    monkeypatch.setattr("audrey.private_task_admission.prepare_native_private_runtime", prepare)
    case = await state.store.tax_cases.create(
        user_id=state.owner.user_id, conversation_id=cid,
        tax_year=2025, goal="Review only the selected synthetic invoice", reference_release=_RELEASE,
    )
    path = (Path(state.app.state.cfg.raw["kb"]["upload_root"])
            / sanitize_user(state.owner.storage_namespace) / (file_id + ".txt"))
    admin = await state.store.resolve_external_identity(
        provider="test", subject="admin", email="admin@example.com",
        display_name="Admin", role="admin", auth_method="test",
    )
    yield SimpleNamespace(state=state, service=PrivateTaskAdmission(state.app), cid=cid,
                          project_id=project_id, file_id=file_id, path=path, case=case, admin=admin)


async def _create(context, **changes):
    return await context.service.create(**dict(
        user_id=context.state.owner.user_id, conversation_id=context.cid,
        expected_case_revision=context.case["revision"], **changes,
    ))


async def _pause(context, task):
    return await context.state.store.private_tasks.control(
        user_id=context.state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], action="pause",
    )


async def _resume(context, task):
    return await context.service.resume(
        user_id=context.state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], expected_case_revision=context.case["revision"],
    )


async def test_server_admission_pins_exact_selected_inputs_without_content_or_generation(admission):
    context, state = admission, admission.state
    state.env.seen.clear()
    state.env.qdrant.reset_mock()
    task = await _create(context)
    assert task["status"] == "queued" and task["attempts"] == 0
    assert len(task["pins"]["input_digest"]) == 64
    assert task["pins"]["model_digests"] == dict(state.env.runtime.policy.local_models)
    assert EVIDENCE not in json.dumps(task)
    assert context.path.name not in json.dumps(task)
    assert all(r.url.path in {"/api/tags", "/api/show"} for r in state.env.seen)
    state.env.qdrant.search_hybrid.assert_not_called()
    assert (await state.client.get(f"/api/conversations/{context.cid}/messages")).json()["items"] == []
    before = len(state.env.seen)
    async with context.service.prepare(state.owner.user_id, context.cid) as prepared:
        assert prepared.pins == task["pins"]
        assert prepared.app.state.gate is state.app.state.gate
        before = len(state.env.seen)
        await prepared.guard()
        await prepared.guard()
        # Polling does not generate, embed or repeatedly fetch provider metadata.
        assert len(state.env.seen) == before


@pytest.mark.parametrize("change", ["disabled", "bots", "model_grant", "purge", "release", "archived"])
async def test_live_guard_refuses_revoked_or_changed_authority(admission, change):
    context, state = admission, admission.state
    task = await _create(context)
    async with context.service.prepare(state.owner.user_id, context.cid) as prepared:
        if change in {"disabled", "bots"}:
            await state.store.admin_update_user(
                actor_user_id=context.admin.user_id, target_user_id=state.owner.user_id,
                **({"status": "disabled"} if change == "disabled" else {"groups": ["users", "bots"]}),
            )
        elif change == "model_grant":
            await state.store.set_model_access_policy(
                actor_user_id=context.admin.user_id, model_id="private", enabled=False, audience="users",
            )
        elif change == "purge":
            block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="test-purge")
        elif change == "release":
            state.app.state.reference_releases.published = lambda *_: {"id": "rel_" + "b" * 32}
        elif change == "archived":
            await state.store.conversations.update(
                user_id=state.owner.user_id, conversation_id=context.cid, archived=True,
            )
        before = len(state.env.seen)
        try:
            with pytest.raises(PrivateTaskError):
                await prepared.guard()
        finally:
            unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="test-purge")
        assert len(state.env.seen) == before
    # Admission cannot claim execution solely from its persisted hashes.
    called = False
    async def work(_app, _step):
        nonlocal called
        called = True
        return StepResult({"stage": "inspect", "summary": "Must not run"})
    if change != "purge":
        with pytest.raises((PrivateTaskError, ProcessingPolicyError)):
            await PrivateTaskStepRunner(state.store.private_tasks, context.service.prepare).execute(
                user_id=state.owner.user_id, task_id=task["id"], work=work,
            )
        assert not called


@pytest.mark.parametrize("part", ["original", "text", "project_instructions", "membership"])
async def test_guard_rejects_input_changes_and_same_size_file_tampering(admission, part):
    context, state = admission, admission.state
    async with context.service.prepare(state.owner.user_id, context.cid) as prepared:
        if part in {"original", "text"}:
            path = context.path if part == "original" else context.path.with_suffix(".ocr.txt")
            original = path.read_bytes()
            path.write_bytes(b"X" + original[1:])
            assert path.stat().st_size == len(original)
        elif part == "project_instructions":
            await state.store.projects.update(user_id=state.owner.user_id,
                project_id=context.project_id, instructions="A different review goal")
        else:
            await state.store.projects.remove_file(user_id=state.owner.user_id,
                project_id=context.project_id, file_id=context.file_id)
        with pytest.raises(PrivateTaskConflictError):
            await prepared.guard()


async def test_project_rename_keeps_valid_task_inputs(admission):
    context, state = admission, admission.state
    task = await _create(context)
    async with context.service.prepare(state.owner.user_id, context.cid) as prepared:
        await state.store.projects.update(user_id=state.owner.user_id,
            project_id=context.project_id, name="A clearer project name")
        await prepared.guard()
    async with context.service.prepare(state.owner.user_id, context.cid) as prepared:
        assert prepared.pins == task["pins"]


async def test_changed_source_cannot_claim_old_checkpoint_and_explicit_resume_rebinds(admission):
    context, state = admission, admission.state
    task = await _create(context)
    # Save a useful checkpoint first, without completing the overall task.
    async def inspect(_app, _step):
        return StepResult({"stage": "inspect", "summary": "Source inventory inspected."})
    runner = PrivateTaskStepRunner(state.store.private_tasks, context.service.prepare)
    task = await runner.execute(user_id=state.owner.user_id, task_id=task["id"], work=inspect)
    original_digest = task["pins"]["input_digest"]
    original = context.path.with_suffix(".ocr.txt").read_bytes()
    context.path.with_suffix(".ocr.txt").write_bytes(b"Y" + original[1:])
    async def forbidden(_app, _step):
        pytest.fail("Old source identity must not execute")
    with pytest.raises(PrivateTaskConflictError):
        await runner.execute(user_id=state.owner.user_id, task_id=task["id"], work=forbidden)
    task = await _pause(context, task)
    resumed = await _resume(context, task)
    assert resumed["status"] == "queued"
    assert resumed["pins"]["input_digest"] != original_digest
    assert resumed["checkpoint"] is None
    assert resumed["usage"] == task["usage"]
    assert resumed["limits"] == task["limits"]


async def test_new_extraction_continues_under_full_live_admission_guard(admission):
    context, state = admission, admission.state
    task = await _create(context)
    from audrey.tax_documents import private_source_identity

    async def work(app, step):
        principal = await state.store.get_principal(user_id=state.owner.user_id)
        identity = await private_source_identity(app, principal, context.file_id)
        handler = private_tool_handler("tax_review", user_id=principal.storage_namespace)
        result = await handler({
            "operation": "propose_fact", "expected_revision": context.case["revision"],
            "data": {"label": "Invoice total", "value": "425", "kind": "extracted",
                     "provenance": {**identity, "page": 1, "original_text": EVIDENCE,
                                    "normalization": "decimal USD"}},
        })
        assert "error" not in result
        await step.check()
        return StepResult({"stage": "facts", "summary": "Invoice total needs confirmation.",
                           "fact_ids": [result["updated_entry"]["id"]]}, "waiting_for_user")
    saved = await PrivateTaskStepRunner(state.store.private_tasks, context.service.prepare).execute(
        user_id=state.owner.user_id, task_id=task["id"], work=work,
    )
    assert saved["status"] == "waiting_for_user" and saved["usage"]["tool_calls"] == 1
    case = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=context.cid)
    assert case["facts"][0]["status"] == "proposed"
    assert saved["sources"]["files"] == case["files"]
    assert saved["pins"]["input_digest"] == task["pins"]["input_digest"]


async def test_foreign_owner_and_stale_case_do_not_create_tasks(admission):
    context, state = admission, admission.state
    with pytest.raises(PrivateTaskAccessError):
        await context.service.create(user_id=context.admin.user_id, conversation_id=context.cid,
                                     expected_case_revision=context.case["revision"])
    with pytest.raises(PrivateTaskConflictError):
        await context.service.create(user_id=state.owner.user_id, conversation_id=context.cid,
                                     expected_case_revision=context.case["revision"] + 1)
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_private_tasks").fetchone()[0] == 0


@pytest.mark.parametrize("part", ["original", "text"])
async def test_missing_source_fails_without_saving_task_or_content_error(admission, part):
    context, state = admission, admission.state
    path = context.path if part == "original" else context.path.with_suffix(".ocr.txt")
    path.unlink()
    with pytest.raises((PrivateTaskError, ProcessingPolicyError)) as error:
        await _create(context)
    assert str(path) not in str(error.value)
    assert EVIDENCE not in str(error.value)
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_private_tasks").fetchone()[0] == 0


async def test_conversation_attachment_change_invalidates_existing_inputs(admission):
    from audrey.app_state.records import AttachmentSnapshot

    context, state = admission, admission.state
    async with context.service.prepare(state.owner.user_id, context.cid) as prepared:
        upload = await state.client.post(f"/api/conversations/{context.cid}/files", files={
            "file": ("another.txt", "ANOTHER_SELECTED_SOURCE", "text/plain"),
        })
        assert upload.status_code == 200, upload.text
        item = upload.json()
        started = await state.store.conversations.begin_run(
            user_id=state.owner.user_id, conversation_id=context.cid,
            user_content="Use this second document too", expected_private_required=True,
            attachments=(AttachmentSnapshot(file_id=item["id"], filename="another.txt",
                         mime="text/plain", kind="text", bytes=len("ANOTHER_SELECTED_SOURCE")),),
        )
        await state.store.conversations.finish_run(
            user_id=state.owner.user_id, run_id=started.run.run_id,
            outcome="succeeded", assistant_content="Selected.",
        )
        with pytest.raises(PrivateTaskConflictError):
            await prepared.guard()


@pytest.mark.parametrize("change", ["bots", "model_grant", "roles"])
async def test_revocation_during_source_verification_prevents_late_case_write(admission, monkeypatch, change):
    from audrey import tax_documents
    from audrey.tax_documents import private_source_identity

    context, state = admission, admission.state
    task = await _create(context)
    original = tax_documents.validate_fact_source
    verified_source = False

    async def revoke_after_verification(*args):
        nonlocal verified_source
        result = await original(*args)
        verified_source = True
        # Simulate an independent administrator request arriving while the tool
        # awaits local provenance validation, after its initial tool guard.
        if change == "bots":
            await state.store.admin_update_user(actor_user_id=context.admin.user_id,
                target_user_id=state.owner.user_id, groups=["users", "bots"])
        elif change == "model_grant":
            await state.store.set_model_access_policy(actor_user_id=context.admin.user_id,
                model_id="private", enabled=False, audience="users")
        else:
            await state.store.set_model_publication_profile(actor_user_id=context.admin.user_id,
                model_id="private", visibility="public", roles=["testers"], display_name="Private")
        return result

    monkeypatch.setattr(tax_documents, "validate_fact_source", revoke_after_verification)

    async def work(app, _step):
        principal = await state.store.get_principal(user_id=state.owner.user_id)
        identity = await private_source_identity(app, principal, context.file_id)
        result = await private_tool_handler("tax_review", user_id=principal.storage_namespace)({
            "operation": "propose_fact", "expected_revision": context.case["revision"],
            "data": {"label": "Invoice total", "value": "425", "kind": "extracted",
                     "provenance": {**identity, "page": 1, "original_text": EVIDENCE,
                                    "normalization": "decimal USD"}},
        })
        pytest.fail(f"Revoked task unexpectedly returned a result: {result}")

    with pytest.raises(PrivateTaskError):
        await PrivateTaskStepRunner(state.store.private_tasks, context.service.prepare).execute(
            user_id=state.owner.user_id, task_id=task["id"], work=work,
        )
    assert verified_source
    # Read canonical storage directly; a revoked owner cannot reopen task data.
    case = json.loads(state.store._conn.execute(
        "SELECT record_json FROM app_tax_cases WHERE conversation_id=?", (context.cid,),
    ).fetchone()[0])
    assert case["revision"] == context.case["revision"]
    assert case["facts"] == [] and case["files"] == []
    task = state.store._conn.execute(
        "SELECT status,record_json FROM app_private_tasks WHERE task_id=?", (task["id"],),
    ).fetchone()
    assert task["status"] == "paused"
    record = json.loads(task["record_json"])
    assert record["stop_reason"] == "access_changed"
    assert record["usage"]["tool_calls"] == 1
