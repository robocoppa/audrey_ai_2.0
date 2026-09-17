"""Preview-first import of Audrey archive exports into owner-bound native history."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from audrey import admin_cli
from audrey.app_state import ApplicationStore
from audrey.app_state.history_import import (
    ArchiveExport,
    HistoryImportError,
    load_archive_export,
)


def _message(
    message_id: str,
    role: str,
    content: str,
    *,
    conversation_id: str = "legacy-chat",
    created_at: str = "2026-08-01T12:00:00+00:00",
    partial: bool = False,
) -> dict:
    return {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "conversation_title": "Old conversation",
        "conversation_created_at": "2026-08-01T11:59:00+00:00",
        "conversation_updated_at": "2026-08-01T12:05:00+00:00",
        "role": role,
        "content": content,
        "created_at": created_at,
        "archived_at": "2026-08-01T12:06:00+00:00",
        "partial": partial,
        "virtual_model": "audrey_fast",
        "concrete_model": "qwen-test",
        "prompt_tokens": 3,
        "completion_tokens": 4,
    }


def _export(user_id: str, email: str, items: list[dict]) -> ArchiveExport:
    return ArchiveExport.model_validate({
        "schema_version": 1,
        "audrey_user_id": user_id,
        "account_email": email,
        "exported_at": "2026-09-16T12:00:00Z",
        "items": items,
    })


async def _owner(store: ApplicationStore, subject: str, email: str):
    return await store.resolve_external_identity(
        provider="owui",
        subject=subject,
        email=email,
        display_name=email.split("@", maxsplit=1)[0],
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=email,
    )


@pytest.mark.asyncio
async def test_import_is_archived_owner_bound_provenanced_and_idempotent(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _owner(store, "alice", "alice@example.com")
    bob = await _owner(store, "bob", "bob@example.com")
    export = _export(alice.user_id, alice.email, [
        _message("source-assistant", "assistant", "Old answer", partial=True),
        _message("source-user", "user", "Old question"),
    ])
    try:
        plan = store.history_imports.preview(
            user_id=alice.user_id, email=alice.email, export=export
        )
        assert plan.new_conversations == 1
        assert plan.new_messages == 2
        assert await store.conversations.list_for_user(user_id=alice.user_id) == ()

        applied = store.history_imports.apply(
            user_id=alice.user_id, email=alice.email, export=export
        )
        assert applied == plan
        (conversation,) = await store.conversations.list_for_user(user_id=alice.user_id)
        assert conversation.conversation_id.startswith("con_")
        assert conversation.title == "Old conversation"
        assert conversation.archived_at is not None
        assert conversation.default_model_id == "auto"
        messages = await store.conversations.list_messages(
            user_id=alice.user_id, conversation_id=conversation.conversation_id
        )
        assert messages is not None
        assert [(item.role, item.content, item.status) for item in messages] == [
            ("user", "Old question", "completed"),
            ("assistant", "Old answer", "incomplete"),
        ]
        assert all(item.run_id is None for item in messages)
        assert await store.conversations.get(
            user_id=bob.user_id, conversation_id=conversation.conversation_id
        ) is None
        assert await store.conversations.list_messages(
            user_id=bob.user_id, conversation_id=conversation.conversation_id
        ) is None
        with sqlite3.connect(store.path) as connection:
            source = connection.execute(
                "SELECT source_message_id, virtual_model, concrete_model, partial "
                "FROM app_history_import_messages ORDER BY source_message_id"
            ).fetchall()
        assert source == [
            ("source-assistant", "audrey_fast", "qwen-test", 1),
            ("source-user", "audrey_fast", "qwen-test", 0),
        ]

        repeat = store.history_imports.apply(
            user_id=alice.user_id, email=alice.email, export=export
        )
        assert repeat.new_conversations == 0
        assert repeat.new_messages == 0
        assert repeat.existing_messages == 2
        assert len(await store.conversations.list_messages(
            user_id=alice.user_id, conversation_id=conversation.conversation_id
        ) or ()) == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_import_appends_newer_messages_and_rejects_changed_source(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _owner(store, "alice", "alice@example.com")
    first = _export(alice.user_id, alice.email, [_message("first", "user", "First")])
    try:
        store.history_imports.apply(user_id=alice.user_id, email=alice.email, export=first)
        later = _export(alice.user_id, alice.email, [
            _message("first", "user", "First"),
            _message(
                "second", "assistant", "Second",
                created_at="2026-08-01T12:01:00+00:00",
            ),
        ])
        result = store.history_imports.apply(
            user_id=alice.user_id, email=alice.email, export=later
        )
        assert result.new_conversations == 0
        assert result.new_messages == 1
        assert result.existing_messages == 1

        changed = _export(alice.user_id, alice.email, [
            _message("first", "user", "Changed"),
        ])
        with pytest.raises(HistoryImportError, match="changed after import"):
            store.history_imports.preview(
                user_id=alice.user_id, email=alice.email, export=changed
            )
        too_old = _export(alice.user_id, alice.email, [
            _message(
                "late-old", "user", "Old",
                created_at="2026-08-01T11:00:00+00:00",
            ),
        ])
        with pytest.raises(HistoryImportError, match="older late-arriving"):
            store.history_imports.preview(
                user_id=alice.user_id, email=alice.email, export=too_old
            )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_import_never_duplicates_native_or_resurrects_deleted_history(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _owner(store, "alice", "alice@example.com")
    native = await store.conversations.create(user_id=alice.user_id)
    try:
        collision = _export(alice.user_id, alice.email, [
            _message("native-copy", "user", "Copy", conversation_id=native.conversation_id),
        ])
        plan = store.history_imports.apply(
            user_id=alice.user_id, email=alice.email, export=collision
        )
        assert plan.skipped_native_conversations == 1
        assert plan.new_messages == 0

        export = _export(alice.user_id, alice.email, [_message("old", "user", "Old")])
        store.history_imports.apply(user_id=alice.user_id, email=alice.email, export=export)
        imported = next(
            item for item in await store.conversations.list_for_user(user_id=alice.user_id)
            if item.conversation_id != native.conversation_id
        )
        assert await store.conversations.delete(
            user_id=alice.user_id, conversation_id=imported.conversation_id
        )
        with sqlite3.connect(store.path) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM app_history_import_messages"
            ).fetchone()[0] == 0
            assert connection.execute(
                "SELECT COUNT(*) FROM app_history_import_conversations"
            ).fetchone()[0] == 1
        repeat = store.history_imports.apply(
            user_id=alice.user_id, email=alice.email, export=export
        )
        assert repeat.skipped_deleted_conversations == 1
        assert repeat.new_messages == 0
        assert await store.conversations.get(
            user_id=alice.user_id, conversation_id=imported.conversation_id
        ) is None
        await store.purge_local_user_data(user_id=alice.user_id)
        with sqlite3.connect(store.path) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM app_history_import_conversations"
            ).fetchone()[0] == 0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_import_rejects_wrong_exact_owner_and_duplicate_source_ids(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _owner(store, "alice", "alice@example.com")
    bob = await _owner(store, "bob", "bob@example.com")
    export = _export(alice.user_id, alice.email, [_message("one", "user", "One")])
    try:
        with pytest.raises(HistoryImportError, match="different Audrey user"):
            store.history_imports.preview(
                user_id=bob.user_id, email=bob.email, export=export
            )
        with pytest.raises(HistoryImportError, match="email does not match"):
            store.history_imports.preview(
                user_id=alice.user_id, email=bob.email, export=export
            )
        file = tmp_path / "duplicate.json"
        file.write_text(json.dumps({
            "schema_version": 1,
            "items": [
                _message("same", "user", "One"),
                _message("same", "assistant", "Two"),
            ],
        }))
        with pytest.raises(HistoryImportError, match="duplicate message ids"):
            load_archive_export(file)
    finally:
        store.close()


def test_operator_preview_is_read_only_and_unbound_apply_requires_opt_in(
    tmp_path, monkeypatch, capsys,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = asyncio.run(_owner(store, "alice", "alice@example.com"))
    db_path = store.path
    store.close()
    monkeypatch.setattr(
        admin_cli, "get_config",
        lambda: SimpleNamespace(raw={"application": {"sqlite_path": str(db_path)}}),
    )
    file = tmp_path / "export.json"
    file.write_text(json.dumps({
        "schema_version": 1,
        "items": [_message("first", "user", "First")],
    }))
    assert admin_cli._import_chat_export(
        file=file, user_id=alice.user_id, email=alice.email
    ) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "preview"
    assert preview["owner_bound"] is False
    assert preview["new_messages"] == 1
    assert admin_cli._import_chat_export(
        file=file, user_id=alice.user_id, email=alice.email, apply=True
    ) == 1
    assert "allow-unbound" in capsys.readouterr().out
    assert admin_cli._import_chat_export(
        file=file, user_id=alice.user_id, email=alice.email,
        apply=True, allow_unbound=True,
    ) == 1
    assert "--backup-to" in capsys.readouterr().out
    backup = tmp_path / "before-import.sqlite"
    assert admin_cli._import_chat_export(
        file=file, user_id=alice.user_id, email=alice.email,
        apply=True, allow_unbound=True, backup_to=backup,
    ) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["status"] == "applied"
    assert applied["backup_path"] == str(backup)
    assert backup.is_file()
    assert backup.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(backup) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute(
            "SELECT COUNT(*) FROM app_history_import_conversations"
        ).fetchone()[0] == 0


def test_preview_does_not_migrate_v11_and_apply_migrates_after_backup(
    tmp_path, monkeypatch, capsys,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = asyncio.run(_owner(store, "alice", "alice@example.com"))
    db_path = store.path
    store.close()
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TRIGGER trg_app_history_import_conversation_deleted")
        connection.execute("DROP TABLE app_history_import_messages")
        connection.execute("DROP TABLE app_history_import_conversations")
        connection.execute("DELETE FROM app_schema_migrations WHERE version = 12")
    monkeypatch.setattr(
        admin_cli, "get_config",
        lambda: SimpleNamespace(raw={"application": {"sqlite_path": str(db_path)}}),
    )
    file = tmp_path / "bound-export.json"
    file.write_text(json.dumps({
        "schema_version": 1,
        "audrey_user_id": alice.user_id,
        "account_email": alice.email,
        "items": [_message("first", "user", "First")],
    }))
    assert admin_cli._import_chat_export(
        file=file, user_id=alice.user_id, email=alice.email
    ) == 0
    assert json.loads(capsys.readouterr().out)["new_messages"] == 1
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM app_schema_migrations"
        ).fetchone()[0] == 11

    backup = tmp_path / "pre-import.sqlite"
    assert admin_cli._import_chat_export(
        file=file, user_id=alice.user_id, email=alice.email,
        apply=True, backup_to=backup,
    ) == 0
    assert json.loads(capsys.readouterr().out)["owner_bound"] is True
    with sqlite3.connect(backup) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM app_schema_migrations"
        ).fetchone()[0] == 11
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM app_schema_migrations"
        ).fetchone()[0] == 12
        assert connection.execute(
            "SELECT COUNT(*) FROM app_history_import_conversations"
        ).fetchone()[0] == 1

    assert admin_cli._import_chat_export(
        file=file, user_id=alice.user_id, email=alice.email,
        apply=True, backup_to=backup,
    ) == 1
    assert "already exists" in capsys.readouterr().out
    with sqlite3.connect(backup) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
