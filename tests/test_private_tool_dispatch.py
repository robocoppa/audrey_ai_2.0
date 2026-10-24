"""Protected tool callbacks stay local, owner-bound, bounded and revocable."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from audrey.identity import Principal
from audrey.kb.file_deletion import FileOperationLocks
from audrey.kb.private_documents import private_text_collection
from audrey.kb.qdrant import KBHit
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.processing_policy import ProcessingPolicy, ProcessingPolicyError, processing_scope
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec
from audrey.tools.dispatch import dispatch_one
from audrey.tools.private_dispatch import private_tool_scope
from audrey.tools.private_handlers import make_private_tool_handlers

_SERVER = "http://custom-tools:8001"
_PRIVATE_TEXT = "PRIVATE_TOOL_CONTENT_91863"


def policy(names=("calculate", "get_file_text")):
    return ProcessingPolicy(
        {"local:latest": "a" * 64}, "http://ollama:11434",
        allowed_tools=frozenset(names), tool_servers=frozenset({_SERVER}),
    )


def registry(name):
    declaration = TOOL_DECLARATIONS[name]
    return ToolRegistry({name: ToolSpec(
        name=name, description=name, parameters={}, server_url=_SERVER, path=f"/{name}",
        user_scope=declaration.user_scope,
    )})


async def call(client, name="calculate", args=None, user="owner", timeout_s=2):
    return await dispatch_one(client, registry(name), {
        "id": "call_local", "function": {"name": name, "arguments": args or {}},
    }, max_result_chars=2_000, timeout_s=timeout_s, user_id=user)


def no_network(request):
    pytest.fail("Protected tool arguments must never reach an HTTP transport")


async def test_private_tool_without_binding_never_falls_back_to_http():
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with processing_scope(policy()):
            result = await call(client, args={"expression": _PRIVATE_TEXT})
    assert result.is_error
    assert _PRIVATE_TEXT not in result.content


async def test_private_callback_gets_only_bound_owner_without_http():
    handler = AsyncMock(return_value={"text": "owned"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with processing_scope(policy()), private_tool_scope(user_id="owner", handlers={"get_file_text": handler}):
            result = await call(client, "get_file_text", {"user": "foreign", "filename": "document.txt"})
    assert not result.is_error
    handler.assert_awaited_once_with({"user": "owner", "filename": "document.txt"})


@pytest.mark.parametrize("owner", [None, "foreign", ""])
async def test_mismatched_owner_refuses_before_parsing_arguments(owner, caplog):
    handler = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with caplog.at_level(logging.DEBUG), processing_scope(policy()), private_tool_scope(user_id="owner", handlers={"calculate": handler}):
            result = await call(client, args=_PRIVATE_TEXT, user=owner)
    assert result.is_error
    handler.assert_not_awaited()
    assert _PRIVATE_TEXT not in result.content + caplog.text


def test_binding_cannot_be_created_without_a_policy_or_widen_an_existing_binding():
    handler = AsyncMock()
    with pytest.raises(ProcessingPolicyError), private_tool_scope(user_id="owner", handlers={}):
        pass
    with processing_scope(policy()), private_tool_scope(user_id="owner", handlers={"calculate": handler}):
        with pytest.raises(ProcessingPolicyError), private_tool_scope(user_id="foreign", handlers={"calculate": handler}):
            pass
        with pytest.raises(ProcessingPolicyError), private_tool_scope(user_id="owner", handlers={"get_file_text": handler}):
            pass


async def test_child_task_inherited_binding_expires_after_scope_exit():
    ready = asyncio.Event()
    handler = AsyncMock(return_value={"value": 1})
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        async def later():
            await ready.wait()
            return await call(client)
        with processing_scope(policy()):
            with private_tool_scope(user_id="owner", handlers={"calculate": handler}):
                task = asyncio.create_task(later())
            ready.set()
            result = await task
    assert result.is_error
    handler.assert_not_awaited()


@pytest.mark.parametrize("failure", ["timeout", "error", "serialization"])
async def test_private_callback_failure_is_content_free(failure, caplog):
    async def handler(args):
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "error":
            raise RuntimeError(_PRIVATE_TEXT)
        return {"unserializable": object()}
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with caplog.at_level(logging.DEBUG), processing_scope(policy()), private_tool_scope(user_id="owner", handlers={"calculate": handler}):
            result = await call(client, args={"expression": _PRIVATE_TEXT}, timeout_s=0.01)
    assert result.is_error
    assert _PRIVATE_TEXT not in result.content + caplog.text
    assert json.loads(result.content)["error"] == ("timeout" if failure == "timeout" else "private_tool_unavailable")


async def test_cancelling_private_callback_does_not_turn_cancellation_into_tool_success():
    async def handler(args):
        raise asyncio.CancelledError
    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        with processing_scope(policy()), private_tool_scope(user_id="owner", handlers={"calculate": handler}):
            with pytest.raises(asyncio.CancelledError):
                await call(client)


async def test_ordinary_dispatch_keeps_existing_http_path():
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={"result": "1350"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await call(client, args={"expression": "1250 * 1.08"})
    assert not result.is_error
    assert len(seen) == 1


@pytest.fixture
async def documents(tmp_path):
    uploads = UploadsDB(tmp_path / "uploads.sqlite")
    owner = Principal("usr_private", "owner", "test", "subject", "owner@example.com", "Owner", "user", "active", "test")
    root = tmp_path / "uploads"
    directory = root / sanitize_user(owner.storage_namespace)
    directory.mkdir(parents=True)
    content = "category,amount\nwork,12.50\nwork,7.50\n"
    stamp = "2026-10-19T12:00:00+00:00"
    file_id = str(uuid.uuid4())
    path = directory / f"{file_id}.csv"
    path.write_text(content, encoding="utf-8")
    path.with_suffix(".ocr.txt").write_text(content, encoding="utf-8")
    reserved = await uploads.reserve_single_upload(
        reservation_id=file_id, user="owner", bytes_=len(content), max_user_bytes=100_000,
        now=stamp, expired_before="2026-10-18T00:00:00+00:00", private_required=True,
    )
    assert reserved.accepted
    committed = await uploads.commit_reserved_upload(
        reservation_id=file_id, reservation_kind="single_shot", file_id=file_id,
        user="owner", filename="costs.csv", mime="text/csv", bytes_=len(content), kind="text",
        collection=private_text_collection("owner"), chunks=1, uploaded_at=stamp,
        status="ready", summary="Work expenses total twenty dollars.", max_user_bytes=100_000,
    )
    assert committed.accepted
    app = SimpleNamespace(state=SimpleNamespace(
        uploads_db=uploads, file_operation_locks=FileOperationLocks(),
        cfg=SimpleNamespace(raw={"kb": {"upload_root": str(root)}}),
    ))
    try:
        yield SimpleNamespace(app=app, owner=owner, file_id=file_id, path=path, content=content, uploads=uploads)
    finally:
        uploads.close()


async def invoke_factory(resources, name, args):
    with processing_scope(policy([name])):
        handlers = make_private_tool_handlers(resources.app, resources.owner)
        async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
            with private_tool_scope(user_id="owner", handlers=handlers):
                return await call(client, name, args)


async def test_local_calculator_uses_shared_decimal_core(documents):
    result = await invoke_factory(documents, "calculate", {"expression": "1250 * 1.08"})
    assert not result.is_error
    assert json.loads(result.content)["result"] == "1350"


async def test_private_list_is_metadata_only_and_file_read_is_real_owned_text(documents):
    result = await invoke_factory(documents, "list_my_files", {"user": "foreign"})
    payload = json.loads(result.content)
    assert not result.is_error
    assert payload["files"][0]["filename"] == "costs.csv"
    assert payload["files"][0]["unread_artifacts"] == ["document", "summary"]
    assert "twenty dollars" not in result.content
    result = await invoke_factory(documents, "get_file_text", {"filename": "costs.csv", "artifact": "document"})
    assert not result.is_error
    assert json.loads(result.content)["text"] == documents.content
    result = await invoke_factory(documents, "get_file_text", {"filename": "costs.csv", "artifact": "summary"})
    assert json.loads(result.content)["text"] == "Work expenses total twenty dollars."


async def test_private_table_aggregates_full_owned_original_without_http(documents):
    result = await invoke_factory(documents, "analyze_table", {
        "filename": "costs.csv", "action": "aggregate", "aggregations": [{"op": "sum", "column": "amount"}],
    })
    assert not result.is_error
    payload = json.loads(result.content)
    assert payload["filename"] == "costs.csv"
    assert payload["artifact"] == "table"
    assert "20" in result.content


@pytest.mark.parametrize("name", ["get_file_text", "analyze_table"])
async def test_private_file_tools_do_not_read_foreign_names_or_symlink_targets(documents, name):
    missing = await invoke_factory(documents, name, {"filename": "foreign.csv"})
    assert missing.is_error
    target = documents.path.parent.parent / "foreign.csv"
    target.write_text(_PRIVATE_TEXT, encoding="utf-8")
    documents.path.unlink()
    documents.path.symlink_to(target)
    result = await invoke_factory(documents, name, {"filename": "costs.csv"})
    assert result.is_error
    assert _PRIVATE_TEXT not in result.content


@pytest.mark.parametrize(("name", "route_name", "args"), [
    ("reference_lookup", "catalog", {"product_number": "Publication 17"}),
    ("read_reference", "read", {"document_id": "a" * 64, "page": 2}),
])
async def test_reference_callbacks_reuse_bounded_owned_library_routes(documents, monkeypatch, name, route_name, args):
    target = AsyncMock(return_value={"source": "owned local reference"})
    monkeypatch.setattr(f"audrey.tools.private_handlers.references.{route_name}", target)
    result = await invoke_factory(documents, name, args)
    assert not result.is_error
    target.assert_awaited_once()
    assert target.await_args.args[1].app is documents.app


async def test_private_kb_search_uses_verified_embedder_and_exact_owned_hybrid_scope(documents):
    embedder = SimpleNamespace(model="local:latest", profile=SimpleNamespace(digest="a" * 64),
        embed_one=AsyncMock(return_value=[1.0, 0.0]))
    hit = KBHit(0.9, str(documents.path), "text", 0, documents.content, {
        "user": "owner", "file_id": documents.file_id, "private_required": True,
        "filename": "costs.csv", "artifact": "text",
    })
    qdrant = SimpleNamespace(collection_exists=AsyncMock(return_value=True),
        search_hybrid=AsyncMock(return_value=([hit], [hit])))
    documents.app.state.text_embedder, documents.app.state.qdrant = embedder, qdrant
    result = await invoke_factory(documents, "kb_search", {"query": "work expenses", "filename": "costs.csv"})
    assert not result.is_error
    payload = json.loads(result.content)
    assert payload["results"][0]["text"] == documents.content
    assert payload["results"][0]["filename"] == "costs.csv"
    embedder.embed_one.assert_awaited_once_with("work expenses")
    options = qdrant.search_hybrid.await_args.kwargs
    assert options["collection"] == private_text_collection("owner")
    assert options["scope"].user == "owner"
    assert options["scope"].file_ids == [documents.file_id]
    assert options["scope"].artifact is None


@pytest.mark.parametrize("case", ["missing_file", "digest", "foreign_hit"])
async def test_private_kb_search_refuses_broadening_or_unverified_embeddings(documents, case):
    hit = KBHit(0.9, "foreign", "text", 0, _PRIVATE_TEXT, {
        "user": "foreign", "file_id": "foreign", "private_required": True,
    })
    embedder = SimpleNamespace(model="local:latest",
        profile=SimpleNamespace(digest="b" * 64 if case == "digest" else "a" * 64),
        embed_one=AsyncMock(return_value=[1.0, 0.0]))
    qdrant = SimpleNamespace(collection_exists=AsyncMock(return_value=True),
        search_hybrid=AsyncMock(return_value=([hit], [])))
    documents.app.state.text_embedder, documents.app.state.qdrant = embedder, qdrant
    args = {"query": "private expenses", **({"filename": "not-owned.csv"} if case == "missing_file" else {})}
    result = await invoke_factory(documents, "kb_search", args)
    assert _PRIVATE_TEXT not in result.content
    if case == "missing_file":
        assert not result.is_error
        assert json.loads(result.content)["results"] == []
    else:
        assert result.is_error
    if case != "foreign_hit":
        embedder.embed_one.assert_not_awaited()
        qdrant.search_hybrid.assert_not_awaited()


@pytest.mark.parametrize("tool", ["get_file_text", "analyze_table"])
async def test_ordinary_duplicate_name_never_shadows_private_document(documents, tool):
    await documents.uploads.record_upload(
        file_id=str(uuid.uuid4()), user="owner", filename="costs.csv", mime="text/csv",
        bytes_=20, kind="text", collection="ordinary", chunks=1,
        uploaded_at="2026-10-20T12:00:00+00:00",
    )
    args = {"filename": "costs.csv"}
    if tool == "analyze_table":
        args.update(action="aggregate", aggregations=[{"op": "sum", "column": "amount"}])
    result = await invoke_factory(documents, tool, args)
    assert not result.is_error
    if tool == "get_file_text":
        assert json.loads(result.content)["text"] == documents.content
    else:
        assert "20" in result.content


async def test_duplicate_private_names_use_newest_and_disclose_selection(documents):
    original = await documents.uploads.get_upload(documents.file_id)
    duplicate_id = str(uuid.uuid4())
    duplicate_path = documents.path.with_name(duplicate_id + ".csv")
    text = "category,amount\nnew,99.00\n"
    duplicate_path.write_text(text)
    duplicate_path.with_suffix(".ocr.txt").write_text(text)
    # A second real private row; no model or generated summary is needed.
    reserved = await documents.uploads.reserve_single_upload(
        reservation_id=duplicate_id, user="owner", bytes_=len(text), max_user_bytes=100_000,
        now="2026-10-20T12:00:00+00:00", expired_before="2026-10-18T00:00:00+00:00",
        private_required=True,
    )
    assert reserved.accepted
    saved = await documents.uploads.commit_reserved_upload(
        reservation_id=duplicate_id, reservation_kind="single_shot", file_id=duplicate_id,
        user="owner", filename=original["filename"], mime="text/csv", bytes_=len(text), kind="text",
        collection=private_text_collection("owner"), chunks=1, uploaded_at="2026-10-20T12:00:00+00:00",
        status="ready", summary="", max_user_bytes=100_000,
    )
    assert saved.accepted
    result = await invoke_factory(documents, "get_file_text", {"filename": "costs.csv"})
    payload = json.loads(result.content)
    assert not result.is_error
    assert payload["text"] == text
    assert "using the newest upload" in payload["note"]


async def test_invalid_private_record_is_skipped_without_disabling_other_files(documents, caplog, monkeypatch):
    good = await documents.uploads.list_user("owner")
    # Simulate a legacy/half-migrated read projection; normal admission rejects it.
    bad = {**good[0], "file_id": str(uuid.uuid4()), "filename": "SECRET_BAD_FILENAME.csv",
           "collection": "wrong-private-collection"}
    monkeypatch.setattr(documents.uploads, "list_user", AsyncMock(return_value=[bad, *good]))
    with caplog.at_level(logging.WARNING):
        result = await invoke_factory(documents, "list_my_files", {})
    assert not result.is_error
    assert [row["filename"] for row in json.loads(result.content)["files"]] == ["costs.csv"]
    assert "SECRET_BAD_FILENAME" not in result.content + caplog.text
    assert "wrong-private-collection" not in result.content + caplog.text
    assert caplog.records
    # Search also skips the bad row, then uses only the good file's identity.
    documents.app.state.text_embedder = SimpleNamespace(
        model="local:latest", profile=SimpleNamespace(digest="a" * 64),
        embed_one=AsyncMock(return_value=[1.0, 0.0]),
    )
    documents.app.state.qdrant = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True), search_hybrid=AsyncMock(return_value=([], [])),
    )
    result = await invoke_factory(documents, "kb_search", {"query": "work expenses"})
    assert not result.is_error
    search = documents.app.state.qdrant.search_hybrid.await_args
    assert search.kwargs["scope"].file_ids == [documents.file_id]
