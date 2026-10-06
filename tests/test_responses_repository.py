"""Durability, isolation, and atomic bounds for retained Responses replay."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.migrations import MIGRATIONS
from audrey.app_state.responses import (
    RESPONSE_MAX_RECORD_BYTES,
    RESPONSE_TTL_SECONDS,
    InvalidResponseStateError,
    ResponseParentNotFoundError,
    ResponseStorageLimitError,
)


async def _owner(store, name="alice"):
    return (await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    )).user_id


def _response(identifier="resp_root", text="A retained answer"):
    return {
        "id": identifier, "object": "response", "status": "completed", "store": True,
        "output": [{
            "id": "msg_example", "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
    }


def _replay(response):
    return [{"role": "user", "content": "Keep this question"}, *response["output"]]


@pytest.fixture
def store(tmp_path):
    value = ApplicationStore(tmp_path / "app.sqlite")
    yield value
    value.close()


async def test_retained_response_roundtrip_is_owner_bound_and_detached(store):
    alice, bob = await _owner(store), await _owner(store, "bob")
    response = _response(text="Café — 你好")
    replay = _replay(response)
    await store.responses.save(alice, response, replay, now=100)
    response["output"][0]["content"][0]["text"] = "Changed after saving"
    replay.clear()
    saved = await store.responses.get(alice, "resp_root", now=101)
    assert saved["response"]["output"][0]["content"][0]["text"] == "Café — 你好"
    assert saved["depth"] == 1
    assert saved["created_at"] == 100
    assert saved["expires_at"] == 100 + RESPONSE_TTL_SECONDS
    assert saved["previous_response_id"] is None
    assert len(saved["replay"]) == 2
    saved["response"]["output"].clear()
    assert (await store.responses.get(alice, "resp_root", now=101))["response"]["output"]
    assert await store.responses.get(bob, "resp_root", now=101) is None
    assert await store.responses.get(alice, "resp_missing", now=101) is None
    assert await store.responses.delete(bob, "resp_root", now=101) is False
    assert await store.responses.get(alice, "resp_root", now=101) is not None


async def test_reopen_retains_response_output_and_canonical_function_replay(tmp_path):
    path = tmp_path / "app.sqlite"
    original = ApplicationStore(path)
    alice = await _owner(original)
    call = {
        "type": "function_call", "id": "fc_one", "call_id": "call_one", "name": "get_marker",
        "arguments": '{"nonce":"abc"}', "status": "completed",
    }
    response = _response()
    response["output"] = [call]
    replay = [{"role": "user", "content": "Call get_marker"}, call]
    await original.responses.save(alice, response, replay)
    original.close()
    reopened = ApplicationStore(path)
    try:
        retained = await reopened.responses.get(alice, "resp_root")
        assert retained["response"] == response
        assert retained["replay"] == replay
        assert reopened.schema_version == 21
        assert reopened._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close()


async def test_additive_schema20_upgrade_keeps_existing_owner_and_conversation(tmp_path):
    path = tmp_path / "app.sqlite"
    stamp = "2026-10-05T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE app_schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        for version, sql in MIGRATIONS:
            if version >= 21:
                break
            connection.executescript(sql)
            connection.execute(
                "INSERT INTO app_schema_migrations VALUES (?, ?)", (version, stamp),
            )
        connection.execute(
            "INSERT INTO app_users (user_id, storage_namespace, current_email, display_name, "
            "role, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("usr_existing", "existing", "alice@example.com", "Alice", "user", "active", stamp, stamp),
        )
        connection.execute(
            "INSERT INTO app_conversations (conversation_id, user_id, title, default_mode, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("con_existing", "usr_existing", "Existing conversation", "auto", stamp, stamp),
        )
    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 21
        assert upgraded._conn.execute("SELECT title FROM app_conversations").fetchone()[0] == "Existing conversation"
        await upgraded.responses.save("usr_existing", _response(), [], now=100)
        assert await upgraded.responses.get("usr_existing", "resp_root", now=100) is not None
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        upgraded.close()


async def test_delete_parent_removes_all_descendants_but_preserves_other_roots(store):
    alice, bob = await _owner(store), await _owner(store, "bob")
    for owner, identifier, parent in [
        (alice, "resp_root", None), (alice, "resp_child", "resp_root"),
        (alice, "resp_grandchild", "resp_child"), (alice, "resp_branch", "resp_root"),
        (alice, "resp_other", None), (bob, "resp_bob", None),
    ]:
        await store.responses.save(owner, _response(identifier), [], parent, now=100)
    assert (await store.responses.get(alice, "resp_grandchild", now=101))["depth"] == 3
    assert await store.responses.delete(alice, "resp_root", now=101) is True
    assert await store.responses.delete(alice, "resp_root", now=101) is False
    for identifier in ("resp_root", "resp_child", "resp_grandchild", "resp_branch"):
        assert await store.responses.get(alice, identifier, now=101) is None
    assert await store.responses.get(alice, "resp_other", now=101) is not None
    assert await store.responses.get(bob, "resp_bob", now=101) is not None


async def test_expired_parent_prunes_newer_descendants_and_returns_cascade_count(store):
    alice = await _owner(store)
    await store.responses.save(alice, _response(), [], now=100)
    await store.responses.save(alice, _response("resp_child"), [], "resp_root", now=200)
    await store.responses.save(alice, _response("resp_other"), [], now=200)
    assert await store.responses.get(alice, "resp_root", now=100 + RESPONSE_TTL_SECONDS - 1) is not None
    assert await store.responses.prune_expired(now=100 + RESPONSE_TTL_SECONDS) == 2
    assert await store.responses.get(alice, "resp_child", now=100 + RESPONSE_TTL_SECONDS) is None
    assert await store.responses.get(alice, "resp_other", now=100 + RESPONSE_TTL_SECONDS) is not None
    assert await store.responses.prune_expired(now=100 + RESPONSE_TTL_SECONDS) == 0


@pytest.mark.parametrize("operation", ["get", "delete", "save"])
async def test_each_repository_operation_prunes_expired_records(store, operation):
    alice = await _owner(store)
    await store.responses.save(alice, _response(), [], now=100)
    now = 100 + RESPONSE_TTL_SECONDS
    if operation == "save":
        await store.responses.save(alice, _response("resp_new"), [], now=now)
    else:
        await getattr(store.responses, operation)(alice, "resp_missing", now=now)
    assert store._conn.execute("SELECT COUNT(*) FROM app_responses WHERE response_id = 'resp_root'").fetchone()[0] == 0


async def test_store_startup_prunes_expired_parent_and_children(tmp_path, monkeypatch):
    path = tmp_path / "app.sqlite"
    current = int(time.time())
    original = ApplicationStore(path)
    alice = await _owner(original)
    await original.responses.save(alice, _response(), [], now=current)
    await original.responses.save(alice, _response("resp_child"), [], "resp_root", now=current + 100)
    original.close()
    monkeypatch.setattr("audrey.app_state.responses.time.time", lambda: current + RESPONSE_TTL_SECONDS)
    reopened = ApplicationStore(path)
    try:
        assert reopened._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == 0
    finally:
        reopened.close()


@pytest.mark.parametrize("parent_kind", ["missing", "foreign", "deleted", "expired"])
async def test_unavailable_parents_have_same_error_and_do_not_create_child(store, parent_kind):
    alice, bob = await _owner(store), await _owner(store, "bob")
    now = 100
    if parent_kind != "missing":
        await store.responses.save(bob if parent_kind == "foreign" else alice, _response(), [], now=100)
    if parent_kind == "deleted":
        await store.responses.delete(alice, "resp_root", now=100)
    if parent_kind == "expired":
        now += RESPONSE_TTL_SECONDS
    with pytest.raises(ResponseParentNotFoundError, match="previous response not found"):
        await store.responses.save(alice, _response("resp_child"), [], "resp_root", now=now)
    if parent_kind == "expired":
        assert store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == 0
    assert await store.responses.get(alice, "resp_child", now=now) is None
    assert not store._conn.in_transaction


async def test_chain_depth32_is_allowed_and_next_save_is_atomic(store):
    alice = await _owner(store)
    parent = None
    for depth in range(1, 33):
        response = _response(f"resp_depth_{depth}")
        await store.responses.save(alice, response, [], parent, now=100)
        parent = response["id"]
    assert (await store.responses.get(alice, parent, now=100))["depth"] == 32
    with pytest.raises(ResponseStorageLimitError, match="32 responses"):
        await store.responses.save(alice, _response("resp_too_deep"), [], parent, now=100)
    assert await store.responses.get(alice, "resp_too_deep", now=100) is None
    assert store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == 32
    assert not store._conn.in_transaction


async def test_default_record_count_quota_is_per_owner_and_delete_frees_capacity(store):
    alice, bob = await _owner(store), await _owner(store, "bob")
    for index in range(100):
        await store.responses.save(alice, _response(f"resp_{index}"), [], now=100)
    with pytest.raises(ResponseStorageLimitError, match="100 stored-response"):
        await store.responses.save(alice, _response("resp_overflow"), [], now=100)
    await store.responses.save(bob, _response("resp_bob"), [], now=100)
    assert await store.responses.delete(alice, "resp_0", now=100)
    await store.responses.save(alice, _response("resp_replacement"), [], now=100)
    assert await store.responses.get(alice, "resp_overflow", now=100) is None
    assert await store.responses.get(bob, "resp_bob", now=100) is not None


async def test_actual_30mib_owner_quota_counts_utf8_bytes_atomically(store):
    alice, bob = await _owner(store), await _owner(store, "bob")
    response = _response("resp_large_00", "")
    overhead = len(json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode()) + 2
    body = "x" * (RESPONSE_MAX_RECORD_BYTES - overhead)
    for index in range(15):
        await store.responses.save(alice, _response(f"resp_large_{index:02}", body), [], now=100)
    before = store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0]
    with pytest.raises(ResponseStorageLimitError, match="30 MiB"):
        await store.responses.save(alice, _response("resp_large_15", body), [], now=100)
    assert store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == before
    await store.responses.save(bob, _response("resp_large_99", body), [], now=100)
    assert not store._conn.in_transaction


async def test_record_size_limit_counts_both_json_blobs_and_multibyte_text(store):
    alice = await _owner(store)
    response = _response(text="é" * (RESPONSE_MAX_RECORD_BYTES // 2))
    with pytest.raises(ResponseStorageLimitError, match="2 MiB"):
        await store.responses.save(alice, response, [], now=100)
    response = _response(text="a" * (RESPONSE_MAX_RECORD_BYTES // 2))
    replay = [{"role": "user", "content": "b" * (RESPONSE_MAX_RECORD_BYTES // 2)}]
    with pytest.raises(ResponseStorageLimitError, match="2 MiB"):
        await store.responses.save(alice, response, replay, now=100)
    assert store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), {1: "bad"}, ("tuple",), b"bytes", {"set"}, "\ud800"])
async def test_non_json_values_and_invalid_unicode_are_rejected_atomically(store, value):
    alice = await _owner(store)
    response = _response()
    response["extra"] = value
    with pytest.raises(InvalidResponseStateError):
        await store.responses.save(alice, response, [], now=100)
    assert store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == 0
    assert not store._conn.in_transaction


@pytest.mark.parametrize("status", ["in_progress", "incomplete", "failed", "cancelled", None])
async def test_only_completed_responses_are_stored(store, status):
    alice = await _owner(store)
    response = _response()
    response["status"] = status
    with pytest.raises(InvalidResponseStateError, match="completed"):
        await store.responses.save(alice, response, [], now=100)
    assert await store.responses.get(alice, "resp_root", now=100) is None


@pytest.mark.parametrize("replay", [None, {}, "bad", ["bad"], [None]])
async def test_replay_must_be_json_object_array(store, replay):
    alice = await _owner(store)
    with pytest.raises(InvalidResponseStateError, match="array of objects"):
        await store.responses.save(alice, _response(), replay, now=100)


@pytest.mark.parametrize("now", [-1, True, 1.1, "100", 1 << 63])
async def test_invalid_timestamps_are_rejected(store, now):
    alice = await _owner(store)
    with pytest.raises(InvalidResponseStateError, match="timestamp"):
        await store.responses.save(alice, _response(), [], now=now)


async def test_duplicate_save_does_not_replace_response_or_break_child(store):
    alice = await _owner(store)
    await store.responses.save(alice, _response(), [], now=100)
    await store.responses.save(alice, _response("resp_child"), [], "resp_root", now=100)
    with pytest.raises(InvalidResponseStateError, match="already retained"):
        await store.responses.save(alice, _response(text="Overwrite"), [], now=100)
    assert (await store.responses.get(alice, "resp_root", now=100))["response"]["output"][0]["content"][0]["text"] == "A retained answer"
    assert (await store.responses.get(alice, "resp_child", now=100))["depth"] == 2
    assert not store._conn.in_transaction


async def test_parallel_saves_atomically_enforce_final_quota_slot(store, monkeypatch):
    alice = await _owner(store)
    monkeypatch.setattr("audrey.app_state.responses.RESPONSE_MAX_OWNER_RECORDS", 1)
    results = await asyncio.gather(
        store.responses.save(alice, _response("resp_a"), [], now=100),
        store.responses.save(alice, _response("resp_b"), [], now=100),
        return_exceptions=True,
    )
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, ResponseStorageLimitError) for result in results) == 1
    assert store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == 1
    assert not store._conn.in_transaction


async def test_local_user_data_purge_removes_retention_but_keeps_identity_and_other_owner(store):
    alice, bob = await _owner(store), await _owner(store, "bob")
    await store.responses.save(alice, _response(), [], now=100)
    await store.responses.save(alice, _response("resp_child"), [], "resp_root", now=100)
    await store.responses.save(bob, _response("resp_bob"), [], now=100)
    await store.purge_local_user_data(user_id=alice)
    assert await store.responses.get(alice, "resp_root", now=100) is None
    assert await store.responses.get(alice, "resp_child", now=100) is None
    assert await store.responses.get(bob, "resp_bob", now=100) is not None
    assert store._conn.execute("SELECT 1 FROM app_users WHERE user_id = ?", (alice,)).fetchone() is not None


async def test_account_foreign_key_removes_retained_chain(store):
    alice, bob = await _owner(store), await _owner(store, "bob")
    await store.responses.save(alice, _response(), [], now=100)
    await store.responses.save(alice, _response("resp_child"), [], "resp_root", now=100)
    await store.responses.save(bob, _response("resp_bob"), [], now=100)
    with store._lock:
        store._conn.execute("DELETE FROM app_users WHERE user_id = ?", (alice,))
        store._conn.commit()
    assert store._conn.execute("SELECT response_id FROM app_responses").fetchone()[0] == "resp_bob"
    assert store._conn.execute("PRAGMA foreign_key_check").fetchall() == []


async def test_parent_owner_foreign_key_blocks_cross_owner_direct_insert(store):
    alice, bob = await _owner(store), await _owner(store, "bob")
    await store.responses.save(alice, _response(), [], now=100)
    with store._lock:
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO app_responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("resp_foreign", bob, "resp_root", 100, 200, json.dumps(_response("resp_foreign")), "[]"),
            )
        store._conn.rollback()
    assert await store.responses.get(bob, "resp_foreign", now=100) is None


@pytest.mark.parametrize("corrupt_json", ['{"id":"resp_root","status":"completed","status":"failed"}', '{"id":"resp_root","status":"completed","bad":NaN}', '{"id":"resp_root","status":"completed","bad":"\\ud800"}', '{"id":"resp_other","status":"completed"}', '[]', 'not json'])
async def test_corrupt_json_is_rejected_on_read_without_partial_state_change(store, corrupt_json):
    alice = await _owner(store)
    await store.responses.save(alice, _response(), [], now=100)
    with store._lock:
        store._conn.execute("UPDATE app_responses SET response_json = ?", (corrupt_json,))
        store._conn.commit()
    with pytest.raises(InvalidResponseStateError, match="malformed"):
        await store.responses.get(alice, "resp_root", now=100)
    assert not store._conn.in_transaction


async def test_missing_owner_rejects_save_without_orphan_rows(store):
    with pytest.raises(InvalidResponseStateError, match="owner does not exist"):
        await store.responses.save("usr_missing", _response(), [], now=100)
    assert store._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0] == 0
