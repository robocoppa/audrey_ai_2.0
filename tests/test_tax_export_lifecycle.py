"""Local Tax review files follow canonical conversation and account deletion."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from test_user_data_controls import _purge_status
from test_user_data_purge import _ArchiveQueue, _FileWorker, _Transport

from audrey.app_state import ApplicationStore
from audrey.auth import AuthedUser
from audrey.kb.uploads_db import UploadsDB
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes.app.conversations import delete_conversation, discard_empty_conversation
from audrey.routes.user_data import AccountPurgeRequest, request_account_purge
from audrey.tax_exports import TaxExportError, TaxReviewExports
from audrey.user_data_purge import UserDataPurgeCoordinator


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    exports = TaxReviewExports(root=tmp_path / "tax-review-exports")
    try:
        yield store, exports
    finally:
        store.close()


async def _owner(store, name):
    return await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com",
        display_name=name, role="user", auth_method="test",
    )


async def _case(store, owner):
    project = await store.projects.create(
        user_id=owner.user_id, name="Tax review", private_required=True,
    )
    conversation = await store.conversations.create(
        user_id=owner.user_id, project_id=project.project_id,
    )
    case = await store.tax_cases.create(
        user_id=owner.user_id, conversation_id=conversation.conversation_id,
        tax_year=2025, goal="Review synthetic worksheet", reference_release="rel_current",
    )
    case = await store.tax_cases.update(
        user_id=owner.user_id, conversation_id=conversation.conversation_id,
        expected_revision=case["revision"], operation="save_review",
        data={"summary": "Synthetic selected-source review; no eligibility claim.", "findings": []},
        actor="model",
    )
    return conversation, case


async def _workbook(exports, *, user_id, case):
    reference = await exports.prepare(user_id=user_id, case=case)
    artifact = await exports.download(user_id=user_id, case=case, export_id=reference["id"])
    artifact.file.close()


def _request(store, exports, **services):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        application_store=store, tax_review_exports=exports, **services,
    )))


@pytest.mark.parametrize("archived", [False, True])
async def test_delete_conversation_erases_exact_owner_exports_and_blocks_recreation(
    state, tmp_path, archived,
):
    store, exports = state
    alice, bob = await _owner(store, "alice"), await _owner(store, "bob")
    conversation, case = await _case(store, alice)
    await _workbook(exports, user_id=alice.user_id, case=case)
    assert len(list(tmp_path.rglob("*.xlsx"))) == 1
    if archived:
        await store.conversations.update(
            user_id=alice.user_id, conversation_id=conversation.conversation_id,
            archived=True,
        )
    request = _request(store, exports)

    with pytest.raises(HTTPException) as denied:
        await delete_conversation(conversation.conversation_id, request, bob)
    assert denied.value.status_code == 404
    assert len(list(tmp_path.rglob("*.xlsx"))) == 1

    deleted = await delete_conversation(conversation.conversation_id, request, alice)
    assert deleted.status_code == 204
    assert await store.conversations.get(
        user_id=alice.user_id, conversation_id=conversation.conversation_id,
    ) is None
    assert not list(tmp_path.rglob("*.xlsx"))
    with pytest.raises(TaxExportError):
        await exports.prepare(user_id=alice.user_id, case=case)
    assert not list(tmp_path.rglob("*.xlsx"))


async def test_refused_active_delete_keeps_case_exportable(state, tmp_path):
    store, exports = state
    owner = await _owner(store, "active")
    conversation, case = await _case(store, owner)
    await _workbook(exports, user_id=owner.user_id, case=case)
    with processing_scope(ProcessingPolicy({"local:latest": "a" * 64}, "http://ollama:11434")):
        await store.conversations.begin_run(
            user_id=owner.user_id, conversation_id=conversation.conversation_id,
            user_content="Check this case", expected_private_required=True,
        )

    with pytest.raises(HTTPException) as denied:
        await delete_conversation(conversation.conversation_id, _request(store, exports), owner)
    assert denied.value.status_code == 409
    assert await store.conversations.get(
        user_id=owner.user_id, conversation_id=conversation.conversation_id,
    ) is not None
    await _workbook(exports, user_id=owner.user_id, case=case)
    assert len(list(tmp_path.rglob("*.xlsx"))) == 1


async def test_empty_draft_cleanup_keeps_saved_case_and_export(state, tmp_path):
    store, exports = state
    owner = await _owner(store, "draft")
    conversation, case = await _case(store, owner)
    await _workbook(exports, user_id=owner.user_id, case=case)
    with pytest.raises(HTTPException) as denied:
        await discard_empty_conversation(
            conversation.conversation_id, _request(store, exports), owner,
        )
    assert denied.value.status_code == 409
    assert len(list(tmp_path.rglob("*.xlsx"))) == 1


async def test_account_purge_removes_only_owner_exports_and_allows_new_cases(state, tmp_path):
    store, exports = state
    alice, bob = await _owner(store, "alice"), await _owner(store, "bob")
    _, alice_case = await _case(store, alice)
    _, bob_case = await _case(store, bob)
    await _workbook(exports, user_id=alice.user_id, case=alice_case)
    await _workbook(exports, user_id=bob.user_id, case=bob_case)
    assert len(list(tmp_path.rglob("*.xlsx"))) == 2
    coordinator = SimpleNamespace(request=AsyncMock(return_value=_purge_status()))

    result = await request_account_purge(
        _request(store, exports, user_data_purges=coordinator),
        AccountPurgeRequest(confirmation="DELETE ALL MY AUDREY DATA"),
        idempotency_key=None,
        me=AuthedUser(email=alice.storage_namespace, role="user", principal=alice),
        principal=alice,
    )
    assert result.status == "pending"
    assert len(list(tmp_path.rglob("*.xlsx"))) == 1
    await _workbook(exports, user_id=bob.user_id, case=bob_case)
    _, new_case = await _case(store, alice)
    await _workbook(exports, user_id=alice.user_id, case=new_case)
    assert len(list(tmp_path.rglob("*.xlsx"))) == 2


async def test_admin_account_deletion_retries_export_cleanup_before_removing_identity(
    state, tmp_path, monkeypatch,
):
    store, exports = state
    admin, target = await _owner(store, "admin"), await _owner(store, "target")
    await store.bootstrap_admin(user_id=admin.user_id)
    _, case = await _case(store, target)
    await _workbook(exports, user_id=target.user_id, case=case)
    await store.begin_admin_account_deletion(
        actor_user_id=admin.user_id, target_user_id=target.user_id,
    )
    cleanup = exports.delete_owner
    failures = 1

    async def fail_once(**kwargs):
        nonlocal failures
        if failures:
            failures -= 1
            raise OSError("injected private snapshot cleanup failure")
        await cleanup(**kwargs)

    monkeypatch.setattr(exports, "delete_owner", fail_once)
    db = UploadsDB(tmp_path / "uploads.sqlite")
    coordinator = UserDataPurgeCoordinator(
        db=db, file_deletions=_FileWorker(), archive_queue=_ArchiveQueue(),
        archive_transport=_Transport([{"status": "completed"}]), registry=object(),
        upload_root=tmp_path / "uploads", application_store=store,
        tax_review_exports=exports,
    )
    try:
        await coordinator.drain_account_deletions_once()
        assert (await store.get_admin_user(user_id=target.user_id)).deletion_pending
        assert len(list(tmp_path.rglob("*.xlsx"))) == 1
        await coordinator.drain_account_deletions_once()
        assert await store.get_admin_user(user_id=target.user_id) is None
        assert not list(tmp_path.rglob("*.xlsx"))
    finally:
        db.close()
