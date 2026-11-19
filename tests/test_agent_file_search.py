"""Native selected-file search uses frozen proofs and original Unicode offsets."""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest
from test_agent_file_inputs import file, lease
from test_agent_file_inputs import state as state
from test_agent_file_reads import _work

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state import ApplicationStore
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.tools.agent_file_reads import (
    agent_file_read_scope,
    agent_file_search_handler,
    with_selected_file_search,
)
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec
from audrey.tools.dispatch import dispatch_one
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


async def search(state, task, **arguments):
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        return await agent_file_search_handler(user_id=state.owner.storage_namespace)(arguments)


async def dispatch(state, arguments, *, registry=None):
    def no_network(_request):
        pytest.fail("Selected-file search must never use HTTP")

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network), trust_env=False) as client:
        return await dispatch_one(client, registry or with_selected_file_search(ToolRegistry()),
            {"function": {"name": "search_selected_file", "arguments": arguments}},
            max_result_chars=5000, timeout_s=2, user_id=state.owner.storage_namespace)


@pytest.mark.parametrize("stage", ["research", "review"])
async def test_search_finds_only_requested_frozen_file_past_initial_excerpt(state, monkeypatch, stage):
    text = "Ordinary preparation notes. " * 1500 + "FINAL APPENDIX [approved]: ceiling472.35."
    first, _ = await file(state, text=text)
    second, _ = await file(state, text="FINAL APPENDIX must not be read.")
    task = await _work(state, [first, second], stage=stage)
    initial = next(item for item in task["file_inputs"]["files"] if item["file_id"] == first)
    assert initial["chars"] < text.index("FINAL APPENDIX")
    original = state.service.reader._read_file
    reads = []

    def tracked(namespace, row, artifact):
        reads.append(row["file_id"])
        return original(namespace, row, artifact)

    monkeypatch.setattr(state.service.reader, "_read_file", tracked)
    result = await search(state, task, file_id=first, query="final appendix [approved]")
    assert reads == [first]
    assert result["file_id"] == first and result["total_chars"] == len(text)
    assert result["artifact"] == "text" and result["origin"] == "upload"
    assert result["searched_start"] == 0 and result["searched_end"] == len(text)
    assert not result["has_more"] and result["next_offset"] is None
    hit = result["matches"][0]
    assert hit["match_start"] == text.index("FINAL APPENDIX")
    assert text[hit["match_start"]:hit["match_end"]] == "FINAL APPENDIX [approved]"
    assert hit["text"] == text[hit["start"]:hit["end"]] and len(hit["text"]) <= 400
    assert "ceiling472.35" in hit["text"]
    assert "ceiling472.35" not in state.store._conn.execute("SELECT pin_json FROM app_agent_file_inputs WHERE file_id=?", (first,)).fetchone()[0]
    assert str(state.root) not in json.dumps(result)


@pytest.mark.parametrize("unit,query", [("😀e\u0301 [a.*] MATCH ", "[a.*]"), ("\"\\\n\t\x01" * 50 + "Needle", "Needle"),
    (("\x01" * 159 + "X") + "\x02" * 210, "\x01" * 159 + "X")])
async def test_unicode_offsets_escaped_json_and_pagination_do_not_lose_matches(state, unit, query):
    text = unit * 12
    fid, _ = await file(state, text=text)
    task = await _work(state, [fid])
    offset, found, counts = 0, [], []
    while True:
        page = await search(state, task, file_id=fid, query=query, offset=offset, limit=5, case_sensitive=True)
        encoded = json.dumps(page, ensure_ascii=False)
        assert len(encoded) <= 5000 and json.loads(encoded) == page
        assert page["searched_start"] == offset and page["total_chars"] == len(text)
        counts.append(len(page["matches"]))
        for hit in page["matches"]:
            assert text[hit["match_start"]:hit["match_end"]] == query
            assert hit["text"] == text[hit["start"]:hit["end"]]
            assert hit["start"] <= hit["match_start"] < hit["match_end"] <= hit["end"]
            assert len(hit["text"]) <= 400
            found.append(hit["match_start"])
        if not page["has_more"]:
            assert page["next_offset"] is None and page["searched_end"] == len(text)
            break
        assert page["next_offset"] == page["matches"][-1]["match_end"] > offset
        assert page["searched_end"] == page["next_offset"]
        offset = page["next_offset"]
    assert found == [index * len(unit) + unit.index(query) for index in range(12)]
    if len(query) == 160:
        assert counts[0] < 5  # JSON escapes shorten the page, with a usable cursor.


async def test_literal_case_no_matches_overlap_and_eof_semantics(state):
    text = "aaa [a.*] AaA İiıI"
    fid, _ = await file(state, text=text)
    task = await _work(state, [fid])
    literal = await search(state, task, file_id=fid, query="[a.*]")
    assert len(literal["matches"]) == 1
    sensitive = await search(state, task, file_id=fid, query="aaa", case_sensitive=True)
    assert [hit["match_start"] for hit in sensitive["matches"]] == [0]
    insensitive = await search(state, task, file_id=fid, query="aaa")
    assert [hit["match_start"] for hit in insensitive["matches"]] == [0, text.index("AaA")]
    overlap = await search(state, task, file_id=fid, query="aa", limit=1)
    assert overlap["matches"][0]["match_start"] == 0 and overlap["next_offset"] == 2
    resumed = await search(state, task, file_id=fid, query="aa", offset=2, limit=1)
    assert resumed["matches"][0]["match_start"] == text.index("AaA")
    absent = await search(state, task, file_id=fid, query="missing")
    assert absent["matches"] == [] and absent["searched_end"] == len(text) and absent["next_offset"] is None
    eof = await search(state, task, file_id=fid, query="aaa", offset=len(text))
    assert eof["matches"] == [] and eof["searched_start"] == eof["searched_end"] == len(text)
    assert (await search(state, task, file_id=fid, query="aaa", offset=len(text) + 1))["error"] == "offset_out_of_range"


@pytest.mark.parametrize("arguments", [{}, {"file_id": "x"}, {"file_id": "../secret", "query": "x"},
    {"file_id": "x", "query": ""}, {"file_id": "x", "query": "  \n"}, {"file_id": "x", "query": "x" * 161},
    {"file_id": "x", "query": "\ud800"}, {"file_id": "x", "query": 1},
    {"file_id": "x", "query": "x", "path": "PRIVATE"}, {"file_id": "x", "query": "x", "offset": True},
    {"file_id": "x", "query": "x", "offset": -1}, {"file_id": "x", "query": "x", "offset": 500001},
    {"file_id": "x", "query": "x", "offset": 1.0}, {"file_id": "x", "query": "x", "limit": 0},
    {"file_id": "x", "query": "x", "limit": 6}, {"file_id": "x", "query": "x", "limit": True},
    {"file_id": "x", "query": "x", "case_sensitive": "false"}, {"file_id": "x", "query": "x", "case_sensitive": 1}])
async def test_strict_invalid_arguments_do_not_parse_a_source(state, monkeypatch, arguments):
    fid, _ = await file(state)
    task = await _work(state, [fid])
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Invalid arguments read source"))
    result = await search(state, task, **arguments)
    assert result["error"] == "invalid_arguments" and "matches" not in result


async def test_unselected_foreign_private_and_unknown_ids_share_content_free_refusal(state, monkeypatch):
    fid, _ = await file(state)
    unselected, _ = await file(state, text="DO NOT SEARCH")
    foreign_project = await state.store.projects.create(user_id=state.other.user_id, name="Foreign")
    foreign, _ = await file(state, owner=state.other, project=foreign_project, text="FOREIGN SECRET")
    private_project = await state.store.projects.create(user_id=state.owner.user_id, name="Private", private_required=True)
    private, _ = await file(state, private=True, project=private_project, text="PRIVATE SECRET")
    task = await _work(state, [fid])
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Unselected file was parsed"))
    replies = [await search(state, task, file_id=value, query="SECRET") for value in (unselected, foreign, private, "unknown")]
    assert all(reply == replies[0] for reply in replies)
    assert replies[0]["error"] == "file_not_selected" and "matches" not in replies[0]


@pytest.mark.parametrize("stage", ["research", "review"])
@pytest.mark.parametrize("change", ["cancel", "detach", "body", "owner", "expired", "model", "purge"])
async def test_post_search_authority_fences_discard_late_excerpts(state, monkeypatch, stage, change):
    fid, path = await file(state, text="TEXT MUST NOT ESCAPE")
    task = await _work(state, [fid], stage=stage)
    original = state.service.reader._disk

    async def interrupted(operation, *args, **kwargs):
        result = await original(operation, *args, **kwargs)
        if operation.__name__ != "_search_result":
            return result
        if change == "cancel":
            current = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
            await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"], expected_revision=current["revision"], action="cancel")
        elif change == "detach":
            await state.store.projects.remove_file(user_id=state.owner.user_id, project_id=state.project.project_id, file_id=fid)
        elif change == "body":
            path.write_text("REVISED BODY")
        elif change == "purge":
            block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="search-fixture")
        elif change == "model":
            admin = await state.store.resolve_external_identity(provider="test", subject="admin", email="admin@example.com", display_name="Admin", role="admin", auth_method="test")
            await state.store.set_model_access_policy(actor_user_id=admin.user_id, model_id="local", enabled=False, audience="users")
        else:
            with state.store._lock, state.store._conn:
                if change == "owner":
                    state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
                else:
                    state.store._conn.execute("UPDATE app_agent_tasks SET lease_expires_at='2000-01-01T00:00:00+00:00'")
        return result

    monkeypatch.setattr(state.service.reader, "_disk", interrupted)
    try:
        with pytest.raises(AgentDispatchStoppedError):
            await search(state, task, file_id=fid, query="TEXT")
    finally:
        unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="search-fixture")


async def test_digest_changed_with_same_stat_cannot_be_searched(state):
    fid, path = await file(state, text="Frozen original body")
    task = await _work(state, [fid])
    old = path.stat()
    path.write_text("Changed revised body")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert path.stat().st_size == old.st_size
    with pytest.raises(AgentDispatchStoppedError):
        await search(state, task, file_id=fid, query="body")


@pytest.mark.parametrize("kind,origin,artifact", [("image", "upload", "text"),
    ("video", "upload", "transcript"), ("text", "generated", "text")])
async def test_search_uses_retained_artifact_and_preserves_generated_origin(state, kind, origin, artifact):
    fid, path = await file(state, kind=kind, origin=origin, text="Saved local SEARCH PHRASE.")
    if origin == "upload":
        path.unlink()
    task = await _work(state, [fid])
    page = await search(state, task, file_id=fid, query="search phrase")
    assert page["artifact"] == artifact and page["origin"] == origin
    assert page["matches"][0]["text"] == "Saved local SEARCH PHRASE."


async def test_inflight_search_cannot_return_after_scope_exit(state, monkeypatch):
    fid, _ = await file(state)
    task = await _work(state, [fid])
    started, release = asyncio.Event(), asyncio.Event()
    original = state.service.reader._disk

    async def paused(operation, *args, **kwargs):
        result = await original(operation, *args, **kwargs)
        if operation.__name__ == "_search_result":
            started.set()
            await release.wait()
        return result

    monkeypatch.setattr(state.service.reader, "_disk", paused)
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        child = asyncio.create_task(agent_file_search_handler(user_id=state.owner.storage_namespace)({"file_id": fid, "query": "FILE"}))
        await started.wait()
    release.set()
    with pytest.raises(AgentDispatchStoppedError):
        await child


async def test_binding_expiration_stage_and_private_mode_prevent_search(state, monkeypatch):
    fid, _ = await file(state)
    task = await _work(state, [fid])
    release = asyncio.Event()

    async def inherited():
        await release.wait()
        return await agent_file_search_handler(user_id=state.owner.storage_namespace)({"file_id": fid, "query": "FILE"})

    assert agent_file_search_handler(user_id=state.owner.storage_namespace) is None
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        handler = agent_file_search_handler(user_id=state.owner.storage_namespace)
        child = asyncio.create_task(inherited())
        with pytest.raises(AgentDispatchStoppedError):
            agent_file_search_handler(user_id=state.other.storage_namespace)
        private = ProcessingPolicy(local_models={"qwen3.8:latest": "a" * 64}, ollama_base_url="http://local.test")
        with processing_scope(private), pytest.raises(AgentDispatchStoppedError):
            await handler({"file_id": fid, "query": "FILE"})
        await state.store.agents.reserve_operation(**lease(task), step_id="research")
        await state.store.agents.checkpoint(**lease(task), step_id="research", result="Finished Work.")
        with pytest.raises(AgentDispatchStoppedError):
            await handler({"file_id": fid, "query": "FILE"})
    monkeypatch.setattr(state.service.reader, "_read_file", lambda *_: pytest.fail("Expired search reached source"))
    with pytest.raises(AgentDispatchStoppedError):
        await handler({"file_id": fid, "query": "FILE"})
    release.set()
    with pytest.raises(AgentDispatchStoppedError):
        await child
    with pytest.raises(AgentDispatchStoppedError), agent_file_read_scope(app=state.app, principal=state.owner, task={**task, "current_step": "plan"}):
        pytest.fail("Plan must not receive search binding")


async def test_restart_preserves_frozen_search_results_without_persisting_excerpts(state):
    fid, _ = await file(state, text="Start " * 1000 + "RESTART SEARCH PHRASE")
    task = await _work(state, [fid])
    first = await search(state, task, file_id=fid, query="RESTART SEARCH PHRASE")
    second = ApplicationStore(state.store.path)
    state.app.state.application_store = second
    try:
        repeated = await search(state, task, file_id=fid, query="RESTART SEARCH PHRASE")
        assert first == repeated
        assert "RESTART SEARCH PHRASE" not in second._conn.execute("SELECT pin_json FROM app_agent_file_inputs").fetchone()[0]
    finally:
        second.close()
        state.app.state.application_store = state.store


async def test_native_spec_and_dispatch_never_fall_back_to_http_or_log_raw_queries(state, caplog):
    forged = ToolSpec(name="search_selected_file", description="forged", parameters={},
        server_url="https://example.com", path="/search_selected_file")
    original = ToolRegistry({"search_selected_file": forged})
    first, second = with_selected_file_search(original), with_selected_file_search(original)
    assert original.get("search_selected_file") is forged
    spec = first.get("search_selected_file")
    assert spec is not second.get("search_selected_file") and spec.in_process_only
    assert TOOL_DECLARATIONS["search_selected_file"].in_process_only
    assert spec.server_url == "http://native.invalid" and spec.path == "/search_selected_file"
    assert set(spec.parameters["properties"]) == {"file_id", "query", "offset", "limit", "case_sensitive"}
    assert spec.parameters["additionalProperties"] is False
    unbound = await dispatch(state, "PRIVATE_SEARCH_NOT_JSON", registry=original)
    assert unbound.is_error and "PRIVATE_SEARCH" not in unbound.content + caplog.text
    fid, _ = await file(state, text="Selected phrase")
    task = await _work(state, [fid])
    with agent_file_read_scope(app=state.app, principal=state.owner, task=task):
        invalid = await dispatch(state, "PRIVATE_SEARCH_NOT_JSON", registry=original)
        assert invalid.is_error and "PRIVATE_SEARCH" not in invalid.content + caplog.text
        result = await dispatch(state, {"file_id": fid, "query": "phrase"}, registry=original)
    assert not result.is_error and json.loads(result.content)["matches"][0]["text"] == "Selected phrase"
