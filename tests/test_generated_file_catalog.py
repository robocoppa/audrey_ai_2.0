"""Generated working files share owner storage without becoming source evidence."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request
from test_native_private_workflow import _conversation, _settle
from test_native_private_workflow import native as native
from test_private_document_ingestion import OLD, STAMP, USER, _land
from test_private_document_ingestion import setup as document_setup  # noqa: F401

from audrey.app_state import ApplicationStore
from audrey.kb.file_deletion import FileDeletionWorker, FileOperationLocks
from audrey.kb.private_documents import read_private_document
from audrey.kb.storage_lifecycle import StorageLifecycle
from audrey.kb.uploads_db import UploadsDB, reconcile_with_qdrant
from audrey.processing_policy import ProcessingPolicyError, processing_scope
from audrey.project_context import project_context_system_message, resolve_project_context
from audrey.routes.app.files import _file_record, _read_document_text
from audrey.routes.app.projects import _validate_private_project_file
from audrey.routes.files import FileRow


async def _commit(db, *, file_id="working-file", private=False, **overrides):
    storage = StorageLifecycle(db)
    reservation = await storage.reserve_single_upload(
        reservation_id=file_id, user=USER, bytes_=20,
        max_user_bytes=100_000, now=STAMP, expired_before=OLD, private_required=private,
    )
    data = dict(file_id=file_id, filename="working.txt", mime="text/plain", bytes_=20,
                kind="text", collection="", chunks=0, uploaded_at=STAMP,
                status="ready", max_user_bytes=100_000, origin="generated")
    data.update(overrides)
    await storage.commit_upload(reservation, **data)
    return reservation


@pytest.mark.parametrize("private", [False, True])
async def test_generated_registration_preserves_quota_provenance_and_retry(tmp_path, private):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    try:
        reservation = await _commit(db, private=private)
        row = await db.get_upload("working-file")
        assert row["origin"] == "generated" and bool(row["private_required"]) == private
        assert row["collection"] == "" and row["chunks"] == 0
        assert (await db.list_user(USER))[0]["origin"] == "generated"
        assert _file_record(FileRow(**{key: row[key] for key in FileRow.model_fields})).origin == "generated"
        assert await db.indexed_file_ids(USER) == set()
        assert (await db.quota_usage(USER)).stored_bytes == 20
        assert (await db.quota_usage(USER)).single_shot_bytes == 0
        await StorageLifecycle(db).commit_upload(
            reservation, file_id="working-file", filename="working.txt", mime="text/plain",
            bytes_=20, kind="text", collection="", chunks=0, uploaded_at=STAMP,
            status="ready", max_user_bytes=100_000, origin="generated",
        )
        with pytest.raises(sqlite3.IntegrityError, match="origin is immutable"):
            db._conn.execute("UPDATE uploads SET origin='upload' WHERE file_id='working-file'")
        with pytest.raises(ValueError, match="cannot overwrite"):
            await db.record_upload(file_id="working-file", user=USER, filename="source.txt",
                                   mime="text/plain", bytes_=20, kind="text", collection="ordinary",
                                   chunks=1, uploaded_at=STAMP)
        await db.backfill_upload(dict(file_id="working-file", filename="source.mp4", mime="video/mp4",
                                     bytes=20, kind="video", chunks=1, uploaded_at=STAMP),
                                 user=USER, collection="ordinary")
        assert (await db.get_upload("working-file"))["kind"] == "text"
    finally:
        db.close()


@pytest.mark.parametrize("invalid", [{"chunks": 1}, {"collection": "kb_text"}, {"status": "pending"}, {"kind": "image"}])
async def test_generated_registration_cannot_claim_indexed_or_pending_work(tmp_path, invalid):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    try:
        with pytest.raises(ValueError, match="ready unindexed"):
            await _commit(db, **invalid)
        assert await db.list_user(USER) == []
    finally:
        db.close()


@pytest.mark.parametrize("purge", [False, True])
async def test_generated_deletion_does_not_need_qdrant(tmp_path, purge):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    qdrant = AsyncMock()
    qdrant.delete_by_file_id.side_effect = RuntimeError("Qdrant unavailable")
    root = tmp_path / "uploads"
    from audrey.kb.user_store import sanitize_user

    directory = root / sanitize_user(USER)
    directory.mkdir(parents=True)
    (directory / "working-file.txt").write_text("output")
    (directory / "working-file.ocr.txt").write_text("output")
    try:
        await _commit(db, private=True)
        worker = FileDeletionWorker(db=db, qdrant=qdrant, upload_root=root, locks=FileOperationLocks())
        if purge:
            receipt = await db.request_user_data_purge(
                purge_id="purge-1", user=USER, cutoff_at=STAMP, requested_at=STAMP,
            )
            assert receipt["files"] == 1
            await worker.drain_once()
        else:
            assert (await worker.request("working-file", user=USER)).completed
        assert await db.list_user(USER) == []
        assert await db.get_upload("working-file") is None
        assert list(directory.iterdir()) == []
        qdrant.delete_by_file_id.assert_not_called()
        assert (await db.quota_usage(USER)).stored_bytes == 0
    finally:
        db.close()


async def test_generated_catalog_is_not_missing_index_drift(tmp_path):
    db = UploadsDB(tmp_path / "uploads.sqlite")
    qdrant = AsyncMock()
    qdrant.list_collections.return_value = []
    try:
        await _commit(db)
        assert (await reconcile_with_qdrant(db, qdrant))["missing_index_files"] == 0
        assert (await db.get_upload("working-file"))["origin"] == "generated"
    finally:
        db.close()


async def test_generated_private_output_reads_local_text_and_is_not_project_evidence(document_setup, tmp_path):  # noqa: F811
    env = document_setup
    file_id, path, reservation = await _land(env, content="Generated working text")
    path.with_suffix(".ocr.txt").write_text("Generated working text", encoding="utf-8")
    await env.storage.commit_upload(
        reservation, file_id=file_id, filename="working.txt", mime="text/plain",
        bytes_=path.stat().st_size, kind="text", collection="", chunks=0, uploaded_at=STAMP,
        status="ready", max_user_bytes=100_000, origin="generated",
    )
    store = ApplicationStore(tmp_path / "project.sqlite")
    owner = await store.resolve_external_identity(
        provider="cloudflare_access", subject="generated-owner", email="alice@example.com",
        display_name="Alice", role="user", auth_method="cloudflare_access",
        legacy_storage_namespace=USER,
    )
    project = await store.projects.create(user_id=owner.user_id, name="Working outputs", private_required=True)
    conversation = await store.conversations.create(user_id=owner.user_id, project_id=project.project_id)
    row = await env.db.get_upload(file_id)
    file_row = FileRow(**{key: row[key] for key in FileRow.model_fields})
    try:
        with processing_scope(env.runtime.policy):
            await store.projects.add_file(user_id=owner.user_id, project_id=project.project_id,
                                          file_id=file_id, file_is_private=True, file_origin="generated")
            await _validate_private_project_file(SimpleNamespace(app=env.app), owner, file_row)
            document = await read_private_document(env.app, owner, file_id)
            assert document.text == "Generated working text"
            with pytest.raises(ProcessingPolicyError):
                await read_private_document(env.app, SimpleNamespace(storage_namespace="another-owner"), file_id)
            snapshot = await resolve_project_context(
                SimpleNamespace(app=env.app), store=store, principal=owner,
                conversation=conversation, query="What sources support the case?",
            )
        assert snapshot.files[0].origin == "generated"
        assert (await store.projects.list_files(user_id=owner.user_id, project_id=project.project_id))[0].origin == "generated"
        assert snapshot.passages == snapshot.sources == ()
        assert "working outputs, not independent source evidence" in project_context_system_message(snapshot)["content"]
        env.qdrant.search_hybrid.assert_not_called()
        assert env.seen == []
    finally:
        store.close()


async def test_generated_json_viewer_uses_local_rendered_text(tmp_path):
    from audrey.kb.user_store import sanitize_user

    directory = tmp_path / sanitize_user(USER)
    directory.mkdir()
    (directory / "output.json").write_text('{"ok":true}', encoding="utf-8")
    (directory / "output.ocr.txt").write_text('{"ok":true}', encoding="utf-8")
    app = SimpleNamespace(state=SimpleNamespace(cfg=SimpleNamespace(raw={"kb": {"upload_root": str(tmp_path)}})))
    request = Request({"type": "http", "app": app})
    row = FileRow(file_id="output", filename="output.json", mime="application/json", bytes=11,
                  uploaded_at=STAMP, chunks=0, origin="generated")
    result = await _read_document_text(request, SimpleNamespace(storage_namespace=USER), row)
    assert result.text == '{"ok":true}'


async def test_generated_ordinary_project_is_manifest_only(tmp_path, monkeypatch):
    from test_project_context import _file, _listing, _principal

    import audrey.project_context as context

    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _principal(store)
    project = await store.projects.create(user_id=owner.user_id, name="Working files")
    conversation = await store.conversations.create(user_id=owner.user_id, project_id=project.project_id)
    await store.projects.add_file(user_id=owner.user_id, project_id=project.project_id, file_id="working", file_origin="generated")
    row = _file("working", "working.txt").model_copy(update={"origin": "generated", "chunks": 0})
    monkeypatch.setattr(context, "_list_for_owner", AsyncMock(return_value=_listing([row])))
    search = AsyncMock(side_effect=AssertionError("Generated output must not enter automatic retrieval"))
    monkeypatch.setattr(context, "search_private_file_text", search)
    try:
        snapshot = await resolve_project_context(SimpleNamespace(), store=store, principal=owner,
                                                 conversation=conversation, query="Sources?")
        assert snapshot.files[0].origin == "generated"
        assert snapshot.passages == snapshot.sources == ()
        search.assert_not_called()
    finally:
        store.close()


async def test_generated_service_reader_is_bounded_and_noted(tmp_path):
    from audrey.kb.user_store import sanitize_user
    from audrey.routes.files import ArtifactRequest, read_artifact

    db = UploadsDB(tmp_path / "uploads.sqlite")
    directory = tmp_path / sanitize_user(USER)
    directory.mkdir()
    (directory / "working-file.txt").write_text("actual output")
    (directory / "working-file.ocr.txt").write_text("x" * 20_000)
    app = SimpleNamespace(state=SimpleNamespace(uploads_db=db, cfg=SimpleNamespace(
        raw={"kb": {"upload_root": str(tmp_path)}},
    )))
    try:
        await _commit(db)
        result = await read_artifact(ArtifactRequest(user=USER, filename="working.txt", limit=50),
                                     Request({"type": "http", "app": app}), None)
        assert result.text == "x" * 50
        assert result.next_offset == 50 and result.total_chars == 20_000
        assert "not independent source evidence" in result.note
        assert not await db.requeue_job("working-file")
        assert (await db.get_upload("working-file"))["status"] == "ready"
    finally:
        db.close()


async def test_generated_private_output_cannot_verify_tax_source_facts(document_setup, tmp_path):  # noqa: F811
    from audrey.app_state.tax_cases import TaxCaseError
    from audrey.tax_documents import (
        private_source_identity,
        tax_source_file_ids,
        validate_fact_source,
    )

    env = document_setup
    file_id, path, reservation = await _land(env, content="Generated tax working text")
    path.with_suffix(".ocr.txt").write_text("Generated tax working text", encoding="utf-8")
    await env.storage.commit_upload(
        reservation, file_id=file_id, filename="working.txt", mime="text/plain",
        bytes_=path.stat().st_size, kind="text", collection="", chunks=0, uploaded_at=STAMP,
        status="ready", max_user_bytes=100_000, origin="generated",
    )
    store = ApplicationStore(tmp_path / "project.sqlite")
    env.app.state.application_store = store
    owner = await store.resolve_external_identity(
        provider="test", subject="tax-output-owner", email="alice@example.com", display_name="Alice",
        role="user", auth_method="test", legacy_storage_namespace=USER,
    )
    project = await store.projects.create(user_id=owner.user_id, name="Tax outputs", private_required=True)
    conversation = await store.conversations.create(user_id=owner.user_id, project_id=project.project_id)
    try:
        with processing_scope(env.runtime.policy):
            await store.projects.add_file(user_id=owner.user_id, project_id=project.project_id,
                                          file_id=file_id, file_is_private=True, file_origin="generated")
            assert await tax_source_file_ids(env.app, owner, {file_id}) == set()
            with pytest.raises(TaxCaseError, match="original uploaded source"):
                await private_source_identity(env.app, owner, file_id)
            with pytest.raises(TaxCaseError, match="original uploaded source"):
                await validate_fact_source(env.app, owner, conversation.conversation_id,
                                           {"kind": "extracted", "provenance": {"file_id": file_id}})
    finally:
        store.close()


async def test_explicit_generated_private_attachment_is_working_context(native):
    import json

    state = native
    conversation = await _conversation(state)
    file_id, path, reservation = await _land(state.env, content="Generated working draft")
    path.with_suffix(".ocr.txt").write_text("Generated working draft", encoding="utf-8")
    await state.env.storage.commit_upload(
        reservation, file_id=file_id, filename="working.txt", mime="text/plain",
        bytes_=path.stat().st_size, kind="text", collection="", chunks=0,
        uploaded_at=STAMP, status="ready", max_user_bytes=100_000, origin="generated",
    )
    response = await state.client.post(f"/api/conversations/{conversation['id']}/runs", json={
        "content": "Improve this working draft", "attachment_ids": [file_id],
    })
    assert response.status_code == 202, response.text
    await _settle(state, response.json()["id"])
    messages = state.captured[-1]
    assert "Generated working draft" in json.dumps(messages)
    assert any("generated working output, not independent source evidence" in message.get("content", "")
               for message in messages if isinstance(message.get("content"), str))
