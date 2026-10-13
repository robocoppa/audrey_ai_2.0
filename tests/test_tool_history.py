"""Privacy and bounded replay for the model's saved-conversation reader."""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore
from audrey.app_state.history_import import ArchiveExport
from audrey.app_state.records import ToolCallSnapshot
from audrey.kb.uploads_db import UploadsDB
from audrey.routes.tool_history import router
from audrey.user_data_visibility import (
    block_remote_personal_reads,
    unblock_remote_personal_reads,
)

SECRET = "history-test-service-token"  # noqa: S105 — hermetic test fixture


@pytest.fixture
def history(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    uploads = UploadsDB(tmp_path / "uploads.sqlite")

    async def owner(subject):
        return await store.resolve_external_identity(
            provider="owui", subject=subject, email=f"{subject}@example.com",
            display_name=subject, role="user", auth_method="owui_bearer",
            legacy_storage_namespace=f"{subject}@example.com",
        )

    alice = asyncio.run(owner("alice"))
    bob = asyncio.run(owner("bob"))
    app = FastAPI()
    app.include_router(router)
    app.state.application_store = store
    app.state.uploads_db = uploads
    app.state.cfg = SimpleNamespace(env=SimpleNamespace(kb_service_token=SECRET))
    with TestClient(app) as client:
        yield client, store, uploads, alice, bob
    store.close()
    uploads.close()


def _post(history, conversation_id, *, user=None, token=SECRET, **fields):
    client, _, _, alice, _ = history
    headers = {"X-Audrey-Service-Token": token} if token is not None else {
        "Authorization": "Bearer ordinary-user-token",
    }
    return client.post(
        "/v1/chat/history/conversation",
        json={
            "user": user if user is not None else alice.storage_namespace,
            "conversation_id": conversation_id, **fields,
        },
        headers=headers,
    )


def _conversation(history, *, owner=None, question="Saved question", answer="Saved answer",
                  outcome="succeeded", tool_calls=()):
    _, store, _, alice, _ = history
    owner = owner or alice

    async def create():
        conversation = await store.conversations.create(user_id=owner.user_id, title="A decision")
        started = await store.conversations.begin_run(
            user_id=owner.user_id, conversation_id=conversation.conversation_id,
            user_content=question,
        )
        assert started is not None
        await store.conversations.finish_run(
            user_id=owner.user_id, run_id=started.run.run_id,
            outcome=outcome, assistant_content=answer, tool_calls=tool_calls,
        )
        return conversation

    return asyncio.run(create())


@pytest.mark.parametrize("token", [None, "wrong-token", ""])
def test_service_token_required_even_when_a_user_bearer_exists(history, token):
    conversation = _conversation(history)
    assert _post(history, conversation.conversation_id, token=token).status_code == 401


def test_reader_returns_saved_exchange_without_tool_payloads(history):
    tool = ToolCallSnapshot(
        tool_call_id="private-tool-call", name="memory_search", status="succeeded",
        arguments={"query": "PRIVATE_TOOL_ARGUMENT"}, result={"private": "PRIVATE_TOOL_RESULT"},
        error_code="",
    )
    conversation = _conversation(history, tool_calls=[tool])
    response = _post(history, conversation.conversation_id)
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == conversation.conversation_id
    assert body["title"] == "A decision"
    assert body["text"] == "\n[user · completed]\nSaved question\n[assistant · completed]\nSaved answer"
    assert body["total_chars"] == len(body["text"])
    assert body["next_offset"] is None
    assert "PRIVATE_TOOL" not in response.text
    assert set(body) == {"conversation_id", "title", "text", "offset", "next_offset", "total_chars"}


def test_foreign_missing_deleted_empty_and_unimported_legacy_share_not_found(history):
    _, store, _, alice, bob = history
    owned = _conversation(history)
    foreign = _conversation(history, owner=bob, answer="FOREIGN_SECRET")
    draft = asyncio.run(store.conversations.create(user_id=alice.user_id))
    asyncio.run(store.conversations.delete(
        user_id=alice.user_id, conversation_id=owned.conversation_id,
    ))
    for conversation_id in (
        owned.conversation_id, foreign.conversation_id, draft.conversation_id,
        "con_missing", "unimported-compatibility-conversation",
    ):
        response = _post(history, conversation_id)
        assert response.status_code == 404
        assert response.json() == {"detail": "conversation_not_found"}
        assert "FOREIGN_SECRET" not in response.text


def test_character_pages_reconstruct_long_unicode_exchange_without_loss(history):
    conversation = _conversation(history, question="Keep Δ units.", answer="x" * 6200 + " ✓ done")
    expected = _post(history, conversation.conversation_id, limit=3000).json()
    pieces = []
    offset = 0
    while True:
        body = _post(history, conversation.conversation_id, offset=offset, limit=997).json()
        assert body["offset"] == offset
        assert body["total_chars"] == expected["total_chars"]
        assert len(body["text"]) <= 997
        pieces.append(body["text"])
        if body["next_offset"] is None:
            break
        assert body["next_offset"] == offset + len(body["text"])
        offset = body["next_offset"]
    assert "".join(pieces) == (
        "\n[user · completed]\nKeep Δ units.\n[assistant · completed]\n" + "x" * 6200 + " ✓ done"
    )
    tail = _post(history, conversation.conversation_id, offset=100000).json()
    assert tail["text"] == "" and tail["next_offset"] is None


@pytest.mark.parametrize("fields", [
    {"offset": -1}, {"limit": 0}, {"limit": 3001}, {"user": "  "},
    {"conversation_id": "  "}, {"unknown": "ignored"},
])
def test_invalid_or_unbounded_requests_are_rejected(history, fields):
    conversation = _conversation(history)
    body = {"conversation_id": conversation.conversation_id}
    body.update(fields)
    assert _post(history, **body).status_code == 422


def test_only_exact_storage_namespace_is_used_and_disabled_accounts_are_hidden(history):
    _, store, _, alice, _ = history
    conversation = _conversation(history)
    assert _post(
        history, conversation.conversation_id, user=alice.user_id,
    ).status_code == 404
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE app_users SET status = 'disabled' WHERE user_id = ?",
                           (alice.user_id,))
    assert _post(history, conversation.conversation_id).status_code == 404


def test_running_answer_is_not_replayed_as_a_completed_exchange(history):
    _, store, _, alice, _ = history
    conversation = asyncio.run(store.conversations.create(user_id=alice.user_id))
    asyncio.run(store.conversations.begin_run(
        user_id=alice.user_id, conversation_id=conversation.conversation_id,
        user_content="Pending question",
    ))
    body = _post(history, conversation.conversation_id).json()
    assert body["text"] == "\n[user · completed]\nPending question"
    assert "assistant" not in body["text"]


def test_cancelled_partial_answer_is_explicitly_labelled(history):
    conversation = _conversation(history, answer="Partial answer", outcome="cancelled")
    body = _post(history, conversation.conversation_id).json()
    assert "\n[assistant · incomplete]\nPartial answer" in body["text"]


def test_pending_purge_gate_blocks_history_reads(history):
    _, _, _, alice, _ = history
    conversation = _conversation(history)
    block_remote_personal_reads(user=alice.storage_namespace, purge_id="test-purge")
    try:
        response = _post(history, conversation.conversation_id)
        assert response.status_code == 409
        assert response.json() == {"detail": "personal_data_purge_in_progress"}
    finally:
        unblock_remote_personal_reads(user=alice.storage_namespace, purge_id="test-purge")


def test_canonical_purge_removes_old_history_but_allows_new_conversations(history):
    _, store, _, alice, _ = history
    old = _conversation(history)
    asyncio.run(store.purge_local_user_data(user_id=alice.user_id))
    assert _post(history, old.conversation_id).status_code == 404
    new = _conversation(history, answer="After purge")
    assert "After purge" in _post(history, new.conversation_id).json()["text"]


def _import_legacy(history):
    _, store, _, alice, _ = history
    export = ArchiveExport.model_validate({
        "schema_version": 1, "items": [{
            "message_id": "source-message",
            "conversation_id": "legacy-search-id",
            "conversation_title": "Imported decision",
            "conversation_created_at": "2026-01-01T00:00:00+00:00",
            "conversation_updated_at": "2026-01-01T00:00:00+00:00",
            "role": "user", "content": "Imported secret",
            "created_at": "2026-01-01T00:00:00+00:00",
            "archived_at": "2026-01-01T00:00:01+00:00",
            "partial": False, "virtual_model": "", "concrete_model": "",
            "prompt_tokens": 0, "completion_tokens": 0,
        }],
    })
    store.history_imports.apply(user_id=alice.user_id, email=alice.email, export=export)


def test_search_ids_resolve_imported_provenance_only_for_the_owner(history):
    _, store, _, alice, bob = history
    _import_legacy(history)
    response = _post(history, "legacy-search-id")
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"].startswith("con_")
    assert body["title"] == "Imported decision"
    assert "Imported secret" in body["text"]
    assert _post(history, "legacy-search-id", user=bob.storage_namespace).status_code == 404
    asyncio.run(store.conversations.delete(
        user_id=alice.user_id, conversation_id=body["conversation_id"],
    ))
    assert _post(history, "legacy-search-id").status_code == 404


def test_durable_purge_cutoff_hides_reimported_old_history_after_acknowledgement(history):
    _, _, uploads, alice, _ = history
    asyncio.run(uploads.request_user_data_purge(
        user=alice.storage_namespace, purge_id="completed-cutoff",
        cutoff_at="2026-06-01T00:00:00+00:00",
        requested_at="2026-06-01T00:00:00+00:00",
    ))
    _import_legacy(history)
    assert _post(history, "legacy-search-id").status_code == 404
    fresh = _conversation(history, answer="New retained answer")
    assert _post(history, fresh.conversation_id).status_code == 200


def test_history_store_is_required(history):
    client, _, _, _, _ = history
    del client.app.state.application_store
    assert _post(history, "con_missing").status_code == 503

def test_durable_purge_store_is_required_even_with_readable_canonical_history(history):
    client, _, _, _, _ = history
    conversation = _conversation(history)
    del client.app.state.uploads_db
    response = _post(history, conversation.conversation_id)
    assert response.status_code == 503
    assert response.json() == {"detail": "history_unavailable"}


def test_account_deletion_request_hides_history_even_before_status_changes(history):
    _, store, _, alice, _ = history
    conversation = _conversation(history)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO account_deletion_requests "
            "(user_id, purge_id, storage_namespace, requested_at) VALUES (?, ?, ?, ?)",
            (alice.user_id, "pending-deletion", alice.storage_namespace,
             "2026-10-13T00:00:00+00:00"),
        )
    assert _post(history, conversation.conversation_id).status_code == 404


def test_new_accounts_bind_to_private_storage_namespace_instead_of_email(history):
    _, store, _, _, _ = history
    owner = asyncio.run(store.resolve_external_identity(
        provider="cloudflare_access", subject="new-owner", email="new@example.com",
        display_name="New", role="user", auth_method="cloudflare_access",
    ))
    conversation = _conversation(history, owner=owner)
    assert owner.storage_namespace != owner.email
    assert _post(history, conversation.conversation_id, user=owner.email).status_code == 404
    response = _post(history, conversation.conversation_id, user=owner.storage_namespace)
    assert response.status_code == 200


@pytest.mark.parametrize("visible_after", [
    "2026-10-13T00:00:00", "not-a-timestamp", "",
])
def test_archive_cutoff_requires_an_aware_timestamp(history, visible_after):
    conversation = _conversation(history)
    assert _post(
        history, conversation.conversation_id, visible_after=visible_after,
    ).status_code == 422


def test_archive_delete_cutoff_hides_old_content_and_title_but_retains_new_turns(history):
    _, store, _, alice, _ = history
    conversation = _conversation(
        history, question="OLD_PRIVATE_QUESTION", answer="OLD_PRIVATE_ANSWER",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE app_messages SET created_at = ? WHERE conversation_id = ?",
            ("2026-01-01T00:00:00+00:00", conversation.conversation_id),
        )

    async def add_new_turn():
        started = await store.conversations.begin_run(
            user_id=alice.user_id, conversation_id=conversation.conversation_id,
            user_content="New question",
        )
        assert started is not None
        await store.conversations.finish_run(
            user_id=alice.user_id, run_id=started.run.run_id,
            outcome="succeeded", assistant_content="New retained answer",
        )

    asyncio.run(add_new_turn())
    response = _post(
        history, conversation.conversation_id,
        visible_after="2026-06-01T00:00:00+00:00",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == (
        "\n[user · completed]\nNew question"
        "\n[assistant · completed]\nNew retained answer"
    )
    assert body["title"] == ""
    assert "OLD_PRIVATE" not in response.text
    assert body["total_chars"] == len(body["text"])
    assert body["next_offset"] is None


def test_archive_delete_cutoff_hides_conversation_when_no_new_turns_exist(history):
    conversation = _conversation(history)
    response = _post(
        history, conversation.conversation_id,
        visible_after="2099-01-01T00:00:00+00:00",
    )
    assert response.status_code == 404
