"""Canonical search sources stay local, bounded, owner-bound and deletion-aware."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.content_sources import MAX_FILE_BYTES, ContentSourceReader, SearchCoverageIssue
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user

STAMP = "2026-10-29T12:00:00+00:00"


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owners = [
        await store.resolve_external_identity(
            provider="test",
            subject=name,
            email=f"{name}@example.com",
            display_name=name,
            role="user",
            auth_method="test",
        )
        for name in ("alice", "bob")
    ]
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    root = tmp_path / "uploads"
    root.mkdir()
    reader = ContentSourceReader(store, uploads, root)
    yield SimpleNamespace(
        store=store, uploads=uploads, root=root, reader=reader, owner=owners[0], other=owners[1]
    )
    uploads.close()
    store.close()


def args(state, owner=None):
    owner = owner or state.owner
    return owner.user_id, owner.storage_namespace


async def message(
    state, *, owner=None, role="user", status="completed", text="Crusades timeline.", mid=None
):
    owner = owner or state.owner
    conversation = await state.store.conversations.create(user_id=owner.user_id, title="History")
    mid = mid or f"msg_{uuid.uuid4().hex}"
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO app_messages (message_id,conversation_id,user_id,sequence_no,"
            "role,status,content,created_at,updated_at) VALUES (?,?,?,1,?,?,?,?,?)",
            (mid, conversation.conversation_id, owner.user_id, role, status, text, STAMP, STAMP),
        )
    return mid, conversation.conversation_id


def file(
    state,
    *,
    owner=None,
    kind="text",
    name="history.txt",
    text="Crusades source text.",
    private=False,
    origin="upload",
    status="ready",
    fid=None,
):
    owner = owner or state.owner
    fid = fid or str(uuid.uuid4())
    directory = state.root / sanitize_user(owner.storage_namespace)
    directory.mkdir(exist_ok=True)
    path = directory / f"{fid}{Path(name).suffix}"
    path.write_text(text, encoding="utf-8")
    mime = {
        "text": "text/plain",
        "video": "video/mp4",
        "audio": "audio/mpeg",
        "image": "image/png",
    }[kind]
    with state.uploads._lock:
        state.uploads._record_locked(
            fid,
            owner.storage_namespace,
            name,
            mime,
            len(text.encode()),
            kind,
            "",
            0,
            STAMP,
            status,
            private_required=private,
            origin=origin,
        )
    return fid, path


async def test_messages_include_native_bodies_and_exclude_tools_and_drafts(state):
    mid, cid = await message(state, status="incomplete")
    await message(state, role="tool", text="Secret raw tool output")
    await message(state, role="assistant", status="in_progress", text="Streaming draft")
    await message(state, owner=state.other)
    docs, cursor, warnings = await state.reader.list_page(*args(state), "message")
    assert cursor is None and warnings == []
    assert [doc.source_id for doc in docs] == [f"message:{mid}"]
    assert docs[0].text == "Crusades timeline."
    assert docs[0].target == {
        "kind": "message",
        "conversation_id": cid,
        "message_id": mid,
        "sequence": 1,
    }
    assert state.reader.is_current(docs[0])
    assert await state.reader.resolve(*args(state, state.other), f"message:{mid}") is None


async def test_message_page_is_keyset_bounded_and_metadata_changes_invalidate(state):
    mid, cid = await message(state, mid="msg_a")
    await message(state, mid="msg_b")
    page, cursor, _ = await state.reader.list_page(*args(state), "message", limit=1)
    assert [doc.source_id for doc in page] == [f"message:{mid}"] and cursor == mid
    second, _, _ = await state.reader.list_page(*args(state), "message", after=cursor, limit=1)
    assert second[0].source_id == "message:msg_b"
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "UPDATE app_conversations SET archived_at=? WHERE conversation_id=?", (STAMP, cid)
        )
    assert not state.reader.is_current(page[0])
    current = await state.reader.resolve(*args(state), f"message:{mid}")
    assert current.archived and current.revision != page[0].revision
    await state.store.conversations.delete(user_id=state.owner.user_id, conversation_id=cid)
    assert await state.reader.resolve(*args(state), f"message:{mid}") is None
    assert not state.reader.is_current(current)


async def test_oversized_messages_warn_instead_of_truncating(state):
    await message(state, text="x" * 500_001)
    docs, _, warnings = await state.reader.list_page(*args(state), "message")
    assert docs == [] and warnings


async def test_namespace_cannot_be_selected_independently_of_owner(state):
    await message(state)
    fid, _ = file(state)
    for kind in ("message", "file", "agent"):
        assert await state.reader.list_page(
            state.other.user_id, state.owner.storage_namespace, kind
        ) == ([], None, [])
    assert (
        await state.reader.resolve(
            state.other.user_id, state.owner.storage_namespace, f"file:{fid}:text"
        )
        is None
    )


async def test_file_text_uses_canonical_identity_and_new_pages_are_bounded(state):
    file(state, fid="file_a")
    fid, _ = file(state, fid="file_b")
    docs, cursor, warnings = await state.reader.list_page(*args(state), "file", limit=1)
    assert (
        [doc.source_id for doc in docs] == ["file:file_a:text"]
        and cursor == "file_a"
        and warnings == []
    )
    second, _, _ = await state.reader.list_page(*args(state), "file", after=cursor, limit=1)
    assert second[0].source_id == f"file:{fid}:text"
    assert state.reader.is_current(second[0])
    assert await state.reader.resolve(*args(state, state.other), second[0].source_id) is None


@pytest.mark.parametrize("status", ["pending", "processing", "failed", "fetch_pending"])
async def test_unfinished_and_failed_uploads_do_not_report_missing_completed_text(state, status):
    file(state, kind="video", name="lecture.mp4", status=status)
    assert await state.reader.list_page(*args(state), "file") == ([], None, [])


async def test_unsupported_file_kind_has_no_missing_text_warning(state):
    fid, _ = file(state)
    with state.uploads._lock:
        state.uploads._conn.execute("UPDATE uploads SET kind='binary' WHERE file_id=?", (fid,))
    assert await state.reader.list_page(*args(state), "file") == ([], None, [])


async def test_ready_image_without_saved_text_reports_one_scoped_coverage_issue(state):
    fid, _ = file(state, kind="image", name="picture.png")
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Pictures")
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO app_project_files(project_id,file_id,added_at) VALUES (?,?,?)",
            (project.project_id, fid, STAMP),
        )
    docs, _, warnings = await state.reader.list_page(*args(state), "file")
    assert docs == []
    assert warnings == [SearchCoverageIssue(
        source_id=f"file:{fid}", kind="file", reason="text_unavailable", updated_at=STAMP,
        project_ids=(project.project_id,),
    )]


async def test_missing_optional_artifact_does_not_hide_valid_mixed_results(state):
    text_id, _ = file(state)
    media_id, path = file(state, kind="audio", name="lecture.mp3")
    path.with_suffix(".transcript.txt").write_text("Saved speech evidence.", encoding="utf-8")
    file(state, kind="image", name="still-pending.png", status="pending")
    docs, _, warnings = await state.reader.list_page(*args(state), "file")
    assert {doc.source_id for doc in docs} == {
        f"file:{text_id}:text", f"file:{media_id}:transcript",
    }
    assert warnings == []


async def test_readable_summary_does_not_hide_oversized_full_text_coverage_gap(state):
    fid, path = file(state, text="x" * 500_001)
    path.with_suffix(".summary.txt").write_text("A short saved summary.", encoding="utf-8")
    docs, _, warnings = await state.reader.list_page(*args(state), "file")
    assert [doc.source_id for doc in docs] == [f"file:{fid}:summary"]
    assert len(warnings) == 1 and warnings[0].reason == "size_limit"


async def test_protected_file_reads_cached_text_without_extracting_or_calling_models(
    state, monkeypatch
):
    fid, path = file(state, private=True, text="Unparsed protected source.")
    path.with_suffix(".ocr.txt").write_text("Protected saved evidence.", encoding="utf-8")

    def forbidden(*_):
        pytest.fail("Protected text must not be extracted or sent to a model")

    monkeypatch.setattr("audrey.content_sources.extract_text", forbidden)
    doc = await state.reader.resolve(*args(state), f"file:{fid}:text")
    assert doc.text == "Protected saved evidence." and doc.private_required
    assert state.reader.is_current(doc)
    path.with_suffix(".ocr.txt").unlink()
    assert await state.reader.resolve(*args(state), doc.source_id) is None
    assert not state.reader.is_current(doc)


async def test_generated_output_uses_saved_renderer_text(state):
    fid, path = file(state, origin="generated", name="report.docx", text="This is not a real docx")
    path.with_suffix(".ocr.txt").write_text("Rendered working output.", encoding="utf-8")
    doc = await state.reader.resolve(*args(state), f"file:{fid}:text")
    assert doc.text == "Rendered working output." and doc.target["origin"] == "generated"
    assert state.reader.is_current(doc)
    path.unlink()
    assert await state.reader.resolve(*args(state), doc.source_id) is None


async def test_image_search_uses_existing_ocr_without_any_image_model(state, monkeypatch):
    fid, path = file(state, kind="image", name="scan.png")

    def forbidden(*_):
        pytest.fail("An image search must only read its existing OCR text")

    monkeypatch.setattr("audrey.content_sources.extract_text", forbidden)
    assert await state.reader.resolve(*args(state), f"file:{fid}:text") is None
    path.with_suffix(".ocr.txt").write_text("Saved image OCR evidence.", encoding="utf-8")
    doc = await state.reader.resolve(*args(state), f"file:{fid}:text")
    assert doc.text == "Saved image OCR evidence." and state.reader.is_current(doc)


async def test_missing_original_is_labeled_and_presence_changes_invalidate(state):
    fid, path = file(state)
    path.with_suffix(".ocr.txt").write_text("Retained extracted evidence.", encoding="utf-8")
    initial = await state.reader.resolve(*args(state), f"file:{fid}:text")
    assert initial.availability == "available"
    path.unlink()
    assert not state.reader.is_current(initial)
    retained = await state.reader.resolve(*args(state), initial.source_id)
    assert retained.availability == "text_only" and retained.text == "Retained extracted evidence."
    path.write_text("Restored original.", encoding="utf-8")
    assert not state.reader.is_current(retained)


async def test_office_archive_expansion_is_bounded_before_parser_runs(state, monkeypatch):
    fid, path = file(state, name="large.docx")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"x" * (MAX_FILE_BYTES + 1))

    def forbidden(*_):
        pytest.fail("Oversized expanded packages must be refused before parsing")

    monkeypatch.setattr("audrey.content_sources.extract_text", forbidden)
    assert path.stat().st_size < MAX_FILE_BYTES
    assert await state.reader.resolve(*args(state), f"file:{fid}:text") is None


@pytest.mark.parametrize("failure", ["oversized_stream", "page_failure", "too_many_pages", None])
async def test_pdf_fallback_bounds_pages_and_streams_and_refuses_partial_extraction(
    state, monkeypatch, failure
):
    fid, _ = file(state, name="history.pdf")

    def extract():
        if failure == "page_failure":
            raise ValueError("Do not expose parser content")
        return "Canonical PDF page."

    page = SimpleNamespace(
        extract_text=extract,
        get_contents=lambda: SimpleNamespace(
            get_data=lambda: b"x" * (MAX_FILE_BYTES + 1 if failure == "oversized_stream" else 10)
        ),
    )
    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda _: SimpleNamespace(pages=[page] * (2001 if failure == "too_many_pages" else 2)),
    )
    doc = await state.reader.resolve(*args(state), f"file:{fid}:text")
    if failure:
        assert doc is None
    else:
        assert doc.text == "Canonical PDF page.\n\nCanonical PDF page."


async def test_reclaimed_media_retains_transcript_visual_and_summary(state):
    fid, path = file(state, kind="video", name="lecture.mp4")
    for suffix, text in (
        ("transcript", "[00:01:20] Crusades speech."),
        ("frames", "[00:01:20] Visual description."),
        ("summary", "Generated summary."),
    ):
        path.with_suffix(f".{suffix}.txt").write_text(text, encoding="utf-8")
    path.unlink()
    with state.uploads._lock:
        state.uploads._conn.execute(
            "UPDATE uploads SET source_freed_at=? WHERE file_id=?", (STAMP, fid)
        )
    docs, _, warnings = await state.reader.list_page(*args(state), "file")
    assert warnings == [] and len(docs) == 3
    assert {doc.target["artifact"] for doc in docs} == {"transcript", "visual", "summary"}
    assert all(doc.availability == "text_only" and state.reader.is_current(doc) for doc in docs)
    assert (
        next(doc for doc in docs if doc.target["artifact"] == "visual").target["origin"]
        == "generated"
    )


async def test_tombstones_fence_even_if_catalog_status_is_restored(state):
    fid, _ = file(state)
    doc = await state.reader.resolve(*args(state), f"file:{fid}:text")
    await state.uploads.request_file_deletion(
        file_id=fid, user=state.owner.storage_namespace, requested_at=STAMP
    )
    with state.uploads._lock:
        state.uploads._conn.execute("UPDATE uploads SET status='ready' WHERE file_id=?", (fid,))
    assert not state.reader.is_current(doc)
    assert await state.reader.resolve(*args(state), doc.source_id) is None
    assert await state.reader.list_page(*args(state), "file") == ([], None, [])


async def test_deletion_during_awaited_file_read_does_not_escape(state, monkeypatch):
    fid, _ = file(state)
    original = state.reader._read_file

    def read_then_delete(*values):
        result = original(*values)
        state.uploads._request_file_deletion_sync(fid, state.owner.storage_namespace, STAMP)
        return result

    monkeypatch.setattr(state.reader, "_read_file", read_then_delete)
    assert await state.reader.resolve(*args(state), f"file:{fid}:text") is None


async def test_changed_file_and_changed_project_membership_invalidate_saved_pins(state):
    fid, path = file(state)
    doc = await state.reader.resolve(*args(state), f"file:{fid}:text")
    path.write_text("Different saved content.", encoding="utf-8")
    assert not state.reader.is_current(doc)
    current = await state.reader.resolve(*args(state), doc.source_id)
    assert current.text == "Different saved content." and current.revision != doc.revision
    project = await state.store.projects.create(user_id=state.owner.user_id, name="History")
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO app_project_files(project_id,file_id,added_at) VALUES (?,?,?)",
            (project.project_id, fid, STAMP),
        )
    assert not state.reader.is_current(current)
    fresh = await state.reader.resolve(*args(state), doc.source_id)
    assert fresh.project_id is None and fresh.target["project_ids"] == [project.project_id]


@pytest.mark.parametrize("mode", ["bytes", "characters", "symlink"])
async def test_oversized_or_linked_file_is_unavailable_without_partial_text(state, mode):
    fid, path = file(state)
    if mode == "bytes":
        with path.open("wb") as stream:
            stream.truncate(MAX_FILE_BYTES + 1)
    elif mode == "characters":
        path.write_text("x" * 500_001, encoding="utf-8")
    else:
        original = path.with_suffix(".actual")
        path.rename(original)
        path.symlink_to(original)
    assert await state.reader.resolve(*args(state), f"file:{fid}:text") is None
    docs, _, warnings = await state.reader.list_page(*args(state), "file")
    assert docs == [] and warnings
    assert warnings[0].reason == ("text_unavailable" if mode == "symlink" else "size_limit")


async def test_saved_agent_final_report_is_generated_and_revision_fenced(state):
    task = await state.store.agents.create(
        user_id=state.owner.user_id, objective="Research crusades", request_id="test"
    )
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "UPDATE app_agent_tasks SET result=?,revision=revision+1 WHERE task_id=?",
            ("Final generated report.", task["id"]),
        )
    docs, _, _ = await state.reader.list_page(*args(state), "agent")
    assert len(docs) == 1 and docs[0].target["origin"] == "generated"
    assert docs[0].text == "Final generated report." and state.reader.is_current(docs[0])
    assert await state.reader.resolve(*args(state, state.other), docs[0].source_id) is None
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "UPDATE app_agent_tasks SET revision=revision+1 WHERE task_id=?", (task["id"],)
        )
    assert not state.reader.is_current(docs[0])


async def test_disabled_owner_is_fenced_on_resolve_and_final_serialization(state):
    mid, _ = await message(state)
    fid, _ = file(state)
    documents = [
        await state.reader.resolve(*args(state), source)
        for source in (f"message:{mid}", f"file:{fid}:text")
    ]
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,)
        )
    for doc in documents:
        assert not state.reader.is_current(doc)
        assert await state.reader.resolve(*args(state), doc.source_id) is None
    assert not state.reader.is_current(dataclasses.replace(doc, revision="forged"))


async def test_pending_account_deletion_fences_even_active_owner(state):
    mid, _ = await message(state)
    fid, _ = file(state)
    docs = [
        await state.reader.resolve(*args(state), source)
        for source in (f"message:{mid}", f"file:{fid}:text")
    ]
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "INSERT INTO account_deletion_requests(user_id,purge_id,storage_namespace,requested_at) VALUES (?,?,?,?)",
            (state.owner.user_id, "purge_test", state.owner.storage_namespace, STAMP),
        )
    for doc in docs:
        assert not state.reader.is_current(doc)
        assert await state.reader.resolve(*args(state), doc.source_id) is None


async def test_cancellation_settles_started_disk_worker_before_caller_exits(state):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def worker():
        entered.set()
        release.wait(2)
        finished.set()

    task = asyncio.create_task(state.reader._disk(worker))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait), 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done() and not finished.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.parametrize(
    "source", ["file:../secret:text", "file:nope:password", "message:a:b", "agent:nope:plan", None]
)
async def test_malformed_sources_do_not_select_paths_or_alternate_artifacts(state, source):
    assert await state.reader.resolve(*args(state), source) is None
