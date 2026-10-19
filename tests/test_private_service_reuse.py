"""Protected content stays out of service reads before policy propagation exists."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from audrey.app_state import ApplicationStore
from audrey.app_state.history_import import ArchiveExport
from audrey.auth import KBCaller
from audrey.kb.qdrant import QdrantKB, SearchScope
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import user_image_collection, user_text_collection
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes import evidence_tools, kb, tool_history

_OWNER = "alice@example.com"
_SECRET = "private-service-test-token"  # noqa: S105 — hermetic fixture


def _request(**state):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def _uploads(*rows, protected=(), deleted=()):
    return SimpleNamespace(
        list_user=AsyncMock(return_value=list(rows)),
        private_file_ids=AsyncMock(return_value=set(protected)),
        file_deletion_ids=AsyncMock(return_value=set(deleted)),
    )


def _policy():
    return ProcessingPolicy(
        local_models={"local-test:latest": "a" * 64},
        ollama_base_url="http://ollama:11434",
    )


async def test_kb_restrictions_union_existing_scope_without_mutation():
    db = _uploads(protected={"protected"}, deleted={"deleted"})
    original = SearchScope(file_ids=["ordinary"], artifact="transcript", excluded_file_ids=["prior"])
    scope = await kb._exclude_deleted_private_files(_request(uploads_db=db), user=_OWNER, scope=original)
    assert scope == SearchScope(
        file_ids=["ordinary"], artifact="transcript",
        excluded_file_ids=["deleted", "prior", "protected"],
    )
    assert original.excluded_file_ids == ["prior"]
    db.private_file_ids.assert_awaited_once_with(_OWNER)
    db.file_deletion_ids.assert_awaited_once_with(_OWNER)


@pytest.mark.parametrize("kind", ["dense", "hybrid", "image"])
async def test_kb_private_exclusion_reaches_storage_filters_without_changing_global_reads(kind):
    request = _request(uploads_db=_uploads(protected={"protected"}, deleted={"deleted"}))
    scope = await kb._exclude_deleted_private_files(request, user=_OWNER, scope=None)
    client = SimpleNamespace(query_points=Mock(return_value=SimpleNamespace(points=[])))
    qdrant = object.__new__(QdrantKB)
    qdrant._client = client
    qdrant.text_collection = "kb_text"
    qdrant.image_collection = "kb_images"
    qdrant.collection_exists = AsyncMock(return_value=True)
    qdrant.validate_text_embedding = AsyncMock()
    qdrant.has_sparse = AsyncMock(return_value=True)
    if kind == "hybrid":
        await kb._search_text_hybrid(
            qdrant, [0.1], query="invoice amount", top_k=5, user=_OWNER,
            min_score=0, cfg={"min_term_overlap": 0}, scope=scope,
        )
        target = user_text_collection(_OWNER)
    elif kind == "dense":
        await kb._search_text_merged(qdrant, [0.1], top_k=5, user=_OWNER, scope=scope)
        target = user_text_collection(_OWNER)
    else:
        await kb._search_images_merged(qdrant, [0.1], top_k=5, user=_OWNER, scope=scope)
        target = user_image_collection(_OWNER)
    calls = [call.kwargs for call in client.query_points.call_args_list]
    local = [call for call in calls if call["collection_name"] == target]
    global_calls = [call for call in calls if call["collection_name"] != target]
    assert len(local) == (2 if kind == "hybrid" else 1)
    assert len(global_calls) == len(local)
    assert all(call["query_filter"] is None for call in global_calls)
    for call in local:
        applied = call["query_filter"]
        assert applied.must[0].key == "user"
        assert applied.must[0].match.value == _OWNER
        assert applied.must_not[0].key == "file_id"
        assert applied.must_not[0].match.any == ["deleted", "protected"]


async def test_selected_protected_file_cannot_be_retrieved_or_silently_removed():
    request = _request(uploads_db=_uploads(protected={"protected"}))
    with processing_scope(_policy()), pytest.raises(HTTPException) as raised:
        await kb._exclude_deleted_private_files(
            request, user=_OWNER, scope=SearchScope(file_ids=["ordinary", "protected"]),
        )
    assert raised.value.status_code == 409
    assert raised.value.detail == "private_processing_unavailable"


@pytest.mark.parametrize("failure", ["missing_db", "lookup_error", "old_db"])
async def test_user_kb_reads_fail_closed_when_restriction_authority_is_unavailable(failure):
    db = _uploads()
    if failure == "lookup_error":
        db.private_file_ids.side_effect = RuntimeError("PRIVATE_CONTENT_IN_BACKEND_ERROR")
    elif failure == "old_db":
        del db.private_file_ids
    request = _request(uploads_db=None if failure == "missing_db" else db)
    with pytest.raises(HTTPException) as raised:
        await kb._exclude_deleted_private_files(request, user=_OWNER, scope=None)
    assert raised.value.status_code == 503
    assert raised.value.detail == "private_processing_unavailable"
    assert await kb._exclude_deleted_private_files(request, user=None, scope=None) is None


@pytest.mark.parametrize("image", [False, True])
async def test_kb_refuses_before_embedding_or_search_when_authority_is_missing(image):
    embedder = SimpleNamespace(embed_one=AsyncMock(), embed_text=AsyncMock())
    request = _request(qdrant=object(), text_embedder=embedder, image_embedder=embedder)
    caller = KBCaller(email=_OWNER, is_service=True)
    with pytest.raises(HTTPException) as raised:
        if image:
            await kb.kb_query_image(kb.ImageQuery(query="invoice", user=_OWNER), request, caller)
        else:
            await kb.kb_query(kb.TextQuery(query="invoice", user=_OWNER), request, caller)
    assert raised.value.status_code == 503
    embedder.embed_one.assert_not_awaited()
    embedder.embed_text.assert_not_awaited()


async def test_filename_resolution_hides_protected_names_from_alternatives():
    request = _request(uploads_db=_uploads(
        {"file_id": "protected", "filename": "SECRET-TAX-RETURN.pdf", "private_required": 1},
        {"file_id": "ordinary", "filename": "public-notes.pdf", "private_required": 0},
    ))
    assert await kb._resolve_filename(request, user=_OWNER, filename="missing.pdf") == (
        [], ["public-notes.pdf"],
    )
    with pytest.raises(HTTPException) as raised:
        await kb._resolve_filename(request, user=_OWNER, filename=" secret-tax-return.PDF ")
    assert raised.value.status_code == 409
    assert raised.value.detail == "private_processing_unavailable"


@pytest.mark.parametrize("newest_private", [False, True])
async def test_filename_duplicate_never_falls_back_around_protected_newest(newest_private):
    request = _request(uploads_db=_uploads(
        {"file_id": "newest", "filename": "invoice.pdf", "private_required": int(newest_private)},
        {"file_id": "older", "filename": "invoice.pdf", "private_required": int(not newest_private)},
    ))
    if newest_private:
        with pytest.raises(HTTPException) as raised:
            await kb._resolve_filename(request, user=_OWNER, filename="invoice.pdf")
        assert raised.value.status_code == 409
    else:
        assert await kb._resolve_filename(request, user=_OWNER, filename="invoice.pdf") == (
            ["newest"], ["invoice.pdf"],
        )


@pytest.mark.parametrize(("suffix", "fields"), [
    ("table", {}), ("inspect-pdf-page", {"page": 1}),
    ("inspect-video-frame", {"timestamp_s": 1}),
])
async def test_protected_evidence_refused_before_disk_render_or_vision(suffix, fields, monkeypatch):
    source = Mock(side_effect=AssertionError("protected bytes must not be read"))
    render = AsyncMock(side_effect=AssertionError("protected bytes must not be rendered"))
    vision = AsyncMock(side_effect=AssertionError("protected bytes must not reach vision"))
    monkeypatch.setattr(evidence_tools, "_source_path", source)
    monkeypatch.setattr(evidence_tools, "render_pdf_page", render)
    monkeypatch.setattr(evidence_tools, "render_video_frame", render)
    monkeypatch.setattr(evidence_tools, "describe_one_image", vision)
    app = FastAPI()
    app.include_router(evidence_tools.router)
    app.state.cfg = SimpleNamespace(env=SimpleNamespace(kb_service_token=_SECRET))
    app.state.uploads_db = _uploads(
        {"file_id": "protected", "filename": "SECRET.pdf", "private_required": 1},
        {"file_id": "ordinary", "filename": "SECRET.pdf", "private_required": 0},
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with processing_scope(_policy()):
            response = await client.post(
                f"/v1/files/{suffix}", headers={"X-Audrey-Service-Token": _SECRET},
                json={"user": _OWNER, "filename": "SECRET.pdf", **fields},
            )
    assert response.status_code == 409
    assert response.json() == {"detail": "private_processing_unavailable"}
    source.assert_not_called()
    render.assert_not_awaited()
    vision.assert_not_awaited()


@pytest.mark.parametrize("imported", [False, True])
async def test_protected_conversations_cannot_be_read_by_native_or_imported_id(tmp_path, imported):
    store = ApplicationStore(tmp_path / "app.sqlite")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    try:
        owner = await store.resolve_external_identity(
            provider="owui", subject="private-reuse", email=_OWNER, display_name="Alice",
            role="user", auth_method="owui_bearer", legacy_storage_namespace=_OWNER,
        )
        if imported:
            export = ArchiveExport.model_validate({"schema_version": 1, "items": [{
                "message_id": "source-message", "conversation_id": "legacy-source",
                "conversation_title": "SECRET-TITLE", "role": "user", "content": "SECRET-CONTENT",
                "conversation_created_at": "2026-10-18T00:00:00+00:00",
                "conversation_updated_at": "2026-10-18T00:00:00+00:00",
                "created_at": "2026-10-18T00:00:00+00:00", "archived_at": "2026-10-18T00:00:01+00:00",
                "partial": False, "virtual_model": "", "concrete_model": "",
                "prompt_tokens": 0, "completion_tokens": 0,
            }]})
            store.history_imports.apply(user_id=owner.user_id, email=owner.email, export=export)
            read_id = "legacy-source"
            with sqlite3.connect(store.path) as connection:
                conversation_id = connection.execute(
                    "SELECT conversation_id FROM app_history_import_conversations WHERE user_id=?",
                    (owner.user_id,),
                ).fetchone()[0]
        else:
            conversation = await store.conversations.create(user_id=owner.user_id, title="SECRET-TITLE")
            run = await store.conversations.begin_run(
                user_id=owner.user_id, conversation_id=conversation.conversation_id,
                user_content="SECRET-CONTENT",
            )
            await store.conversations.finish_run(
                user_id=owner.user_id, run_id=run.run.run_id,
                outcome="succeeded", assistant_content="SECRET-ANSWER",
            )
            conversation_id = read_id = conversation.conversation_id
        # Seed the one-way durable restriction directly, without claiming that
        # an ordinary historic turn was originally processed privately.
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                "UPDATE app_conversations SET private_required=1 WHERE conversation_id=?",
                (conversation_id,),
            )
        app = FastAPI()
        app.include_router(tool_history.router)
        app.state.application_store = store
        app.state.uploads_db = uploads
        app.state.cfg = SimpleNamespace(env=SimpleNamespace(kb_service_token=_SECRET))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with processing_scope(_policy()):
                response = await client.post(
                    "/v1/chat/history/conversation", headers={"X-Audrey-Service-Token": _SECRET},
                    json={"user": owner.storage_namespace, "conversation_id": read_id},
                )
        assert response.status_code == 404
        assert response.json() == {"detail": "conversation_not_found"}
    finally:
        store.close()
        uploads.close()
