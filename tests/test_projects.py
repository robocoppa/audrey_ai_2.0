"""Schema 19 Projects storage and owner-scoped native API contracts."""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.app_state import (
    ApplicationStore,
    InvalidProjectStateError,
    ProjectFileConflictError,
    ProjectFileLimitError,
    ProjectNotFoundError,
)
from audrey.app_state.migrations import MIGRATIONS
from audrey.auth import require_principal
from audrey.identity import Principal
from audrey.routes import files as upload_routes
from audrey.routes.app import files as native_files
from audrey.routes.app import projects as project_routes
from audrey.routes.app import router


async def _resolve(
    store: ApplicationStore,
    *,
    subject: str,
    email: str,
) -> Principal:
    return await store.resolve_external_identity(
        provider="owui",
        subject=subject,
        email=email,
        display_name=email.split("@", maxsplit=1)[0].title(),
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=email,
    )


def _create_schema_18_database(path) -> None:
    stamp = "2026-10-01T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "CREATE TABLE app_schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        for version, sql in MIGRATIONS:
            if version > 18:
                break
            connection.executescript(sql)
            connection.execute(
                "INSERT INTO app_schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, stamp),
            )
        connection.execute(
            "INSERT INTO app_users "
            "(user_id, storage_namespace, current_email, display_name, role, status, "
            "created_at, updated_at) VALUES "
            "('usr_existing', 'existing@example.com', 'existing@example.com', "
            "'Existing', 'user', 'active', ?, ?)",
            (stamp, stamp),
        )
        connection.execute(
            "INSERT INTO app_conversations "
            "(conversation_id, user_id, title, default_mode, default_model_id, "
            "created_at, updated_at, last_message_at, archived_at) "
            "VALUES ('con_existing', 'usr_existing', 'Existing conversation', "
            "'auto', 'auto', ?, ?, NULL, NULL)",
            (stamp, stamp),
        )
        connection.commit()


async def test_schema_19_preserves_existing_conversations_with_no_project(tmp_path):
    path = tmp_path / "app.sqlite"
    _create_schema_18_database(path)

    store = ApplicationStore(path)
    try:
        assert store.schema_version == 39
        conversation = await store.conversations.get(
            user_id="usr_existing",
            conversation_id="con_existing",
        )
        assert conversation is not None
        assert conversation.project_id is None
        assert await store.projects.list_page(user_id="usr_existing", limit=10) == ()
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(app_conversations)")}
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "project_id" in columns
        assert {"app_projects", "app_project_files"} <= tables
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


async def test_projects_repository_enforces_ownership_limits_moves_and_delete(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(
        store,
        subject="alice-projects",
        email="alice@example.com",
    )
    bob = await _resolve(
        store,
        subject="bob-projects",
        email="bob@example.com",
    )
    try:
        first = await store.projects.create(
            user_id=alice.user_id,
            name="  Client launch  ",
            instructions="  Prefer concise status updates.  ",
        )
        second = await store.projects.create(
            user_id=alice.user_id,
            name="Operations",
        )
        foreign = await store.projects.create(user_id=bob.user_id, name="Private")

        assert first.name == "Client launch"
        assert first.instructions == "Prefer concise status updates."
        assert (
            await store.projects.get(
                user_id=bob.user_id,
                project_id=first.project_id,
            )
            is None
        )

        page = await store.projects.list_page(user_id=alice.user_id, limit=1)
        assert len(page) == 1
        following = await store.projects.list_page(
            user_id=alice.user_id,
            limit=2,
            before_updated_at=page[0].updated_at,
            before_project_id=page[0].project_id,
        )
        assert {record.project_id for record in (*page, *following)} == {
            first.project_id,
            second.project_id,
        }

        conversation = await store.conversations.create(
            user_id=alice.user_id,
            title="Launch plan",
            project_id=first.project_id,
        )
        assert conversation.project_id == first.project_id
        moved = await store.conversations.update(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
            project_id=second.project_id,
            update_project=True,
        )
        assert moved is not None and moved.project_id == second.project_id
        ungrouped = await store.conversations.update(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
            project_id=None,
            update_project=True,
        )
        assert ungrouped is not None and ungrouped.project_id is None
        with pytest.raises(ProjectNotFoundError):
            await store.conversations.update(
                user_id=alice.user_id,
                conversation_id=conversation.conversation_id,
                project_id=foreign.project_id,
                update_project=True,
            )

        await store.projects.add_file(
            user_id=alice.user_id,
            project_id=first.project_id,
            file_id="file_00",
        )
        with pytest.raises(ProjectFileConflictError):
            await store.projects.add_file(
                user_id=alice.user_id,
                project_id=first.project_id,
                file_id="file_00",
            )
        for index in range(1, 20):
            await store.projects.add_file(
                user_id=alice.user_id,
                project_id=first.project_id,
                file_id=f"file_{index:02d}",
            )
        with pytest.raises(ProjectFileLimitError):
            await store.projects.add_file(
                user_id=alice.user_id,
                project_id=first.project_id,
                file_id="file_20",
            )
        assert (
            len(
                await store.projects.list_files(
                    user_id=alice.user_id,
                    project_id=first.project_id,
                )
                or ()
            )
            == 20
        )

        linked = await store.conversations.update(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
            project_id=first.project_id,
            update_project=True,
        )
        assert linked is not None and linked.project_id == first.project_id
        assert await store.projects.delete(
            user_id=alice.user_id,
            project_id=first.project_id,
        )
        retained = await store.conversations.get(
            user_id=alice.user_id,
            conversation_id=conversation.conversation_id,
        )
        assert retained is not None and retained.project_id is None
        assert (
            await store.projects.get(
                user_id=alice.user_id,
                project_id=first.project_id,
            )
            is None
        )

        with pytest.raises(InvalidProjectStateError):
            await store.projects.create(user_id=alice.user_id, name=" ")
        with pytest.raises(InvalidProjectStateError):
            await store.projects.create(user_id=alice.user_id, name="x" * 101)
    finally:
        store.close()


async def test_project_directory_counts_current_owned_chats_and_batches_activity(tmp_path, monkeypatch):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(store, subject="alice-summary", email="alice@example.com")
    bob = await _resolve(store, subject="bob-summary", email="bob@example.com")
    try:
        first = await store.projects.create(user_id=alice.user_id, name="Garden")
        second = await store.projects.create(user_id=alice.user_id, name="Other garden")
        foreign = await store.projects.create(user_id=bob.user_id, name="Foreign")
        chats = [await store.conversations.create(
            user_id=alice.user_id, title=title, project_id=first.project_id,
        ) for title in ("Active", "Archived", "Deleted", "Moved")]
        await store.conversations.update(
            user_id=alice.user_id, conversation_id=chats[1].conversation_id, archived=True,
        )
        await store.conversations.delete(
            user_id=alice.user_id, conversation_id=chats[2].conversation_id,
        )
        await store.conversations.update(
            user_id=alice.user_id, conversation_id=chats[3].conversation_id,
            project_id=second.project_id, update_project=True,
        )
        foreign_chat = await store.conversations.create(
            user_id=bob.user_id, title="Foreign", project_id=foreign.project_id,
        )
        base = "2026-01-01T00:00:00+00:00"
        with store._lock:
            store._conn.execute("UPDATE app_projects SET updated_at=?", (base,))
            store._conn.execute("UPDATE app_conversations SET created_at=?,updated_at=?", (base, base))
            store._conn.execute(
                "UPDATE app_conversations SET last_message_at=? WHERE conversation_id=?",
                ("2026-01-03T00:00:00+00:00", chats[0].conversation_id),
            )
            store._conn.execute(
                "UPDATE app_conversations SET updated_at=? WHERE conversation_id=?",
                ("2026-01-04T00:00:00+00:00", chats[1].conversation_id),
            )
            store._conn.execute(
                "UPDATE app_conversations SET updated_at=? WHERE conversation_id=?",
                ("2026-01-30T00:00:00+00:00", chats[3].conversation_id),
            )
            # A malformed legacy link must never contribute another owner's data.
            store._conn.execute(
                "UPDATE app_conversations SET project_id=?,updated_at=? WHERE conversation_id=?",
                (first.project_id, "2050-01-01T00:00:00+00:00", foreign_chat.conversation_id),
            )
            store._conn.commit()
            statements = []
            store._conn.set_trace_callback(statements.append)
        try:
            page = await store.projects.list_page(user_id=alice.user_id, limit=10)
        finally:
            store._conn.set_trace_callback(None)
        assert len([sql for sql in statements if sql.lstrip().startswith(("SELECT", "WITH"))]) == 2
        by_id = {record.project_id: record for record in page}
        assert set(by_id) == {first.project_id, second.project_id}
        assert by_id[first.project_id].conversation_count == 2
        assert by_id[first.project_id].last_activity_at == "2026-01-04T00:00:00+00:00"
        assert by_id[second.project_id].conversation_count == 1
        assert by_id[second.project_id].last_activity_at == "2026-01-30T00:00:00+00:00"
        assert await store.projects.get(user_id=bob.user_id, project_id=first.project_id) is None

        monkeypatch.setattr("audrey.app_state.projects._utc_now", lambda: "2026-01-05T00:00:00+00:00")
        edited = await store.projects.update(user_id=alice.user_id, project_id=first.project_id, name="Renamed")
        assert edited.conversation_count == 2
        assert edited.last_activity_at == "2026-01-05T00:00:00+00:00"
        await store.conversations.delete(user_id=alice.user_id, conversation_id=chats[1].conversation_id)
        await store.conversations.update(
            user_id=alice.user_id, conversation_id=chats[0].conversation_id,
            project_id=None, update_project=True,
        )
        empty = await store.projects.get(user_id=alice.user_id, project_id=first.project_id)
        assert empty.conversation_count == 0
        assert empty.last_activity_at == "2026-01-05T00:00:00+00:00"
        monkeypatch.setattr("audrey.app_state.repositories._utc_now", lambda: "2026-01-31T00:00:00+00:00")
        started = await store.conversations.begin_run(
            user_id=alice.user_id, conversation_id=chats[3].conversation_id, user_content="A new message",
        )
        await store.conversations.finish_run(
            user_id=alice.user_id, run_id=started.run.run_id, outcome="succeeded", assistant_content="Saved reply",
        )
        active = await store.projects.get(user_id=alice.user_id, project_id=second.project_id)
        assert active.conversation_count == 1
        assert active.last_activity_at == "2026-01-31T00:00:00+00:00"
    finally:
        store.close()


@pytest.mark.parametrize("view", ["get", "list"])
async def test_project_summary_keeps_one_snapshot_during_concurrent_deletion(tmp_path, monkeypatch, view):
    store = ApplicationStore(tmp_path / "app.sqlite")
    other = ApplicationStore(store.path)
    owner = await _resolve(store, subject="summary-delete", email="alice@example.com")
    try:
        project = await store.projects.create(user_id=owner.user_id, name="Garden")
        await store.conversations.create(user_id=owner.user_id, project_id=project.project_id)
        summarize = store.projects._with_summaries_locked

        def delete_before_summary(user_id, rows):
            assert other.projects._delete_sync(user_id, project.project_id)
            return summarize(user_id, rows)

        monkeypatch.setattr(store.projects, "_with_summaries_locked", delete_before_summary)
        if view == "get":
            saved = await store.projects.get(user_id=owner.user_id, project_id=project.project_id)
        else:
            saved = (await store.projects.list_page(user_id=owner.user_id, limit=10))[0]
        assert saved.project_id == project.project_id
        assert saved.conversation_count == 1
        assert await store.projects.get(user_id=owner.user_id, project_id=project.project_id) is None
    finally:
        other.close()
        store.close()


async def test_project_activity_includes_retained_knowledge_agents_and_watch_checks(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store, subject="activity-summary", email="alice@example.com")
    try:
        project = await store.projects.create(user_id=owner.user_id, name="Garden")
        note = await store.knowledge.create(
            user_id=owner.user_id, project_id=project.project_id, kind="fact", text="Garden plan",
        )
        task = await store.agents.create(
            user_id=owner.user_id, project_id=project.project_id,
            objective="Prepare a garden checklist", request_id="summary-agent",
        )
        watch = await store.watches.create(
            user_id=owner.user_id, project_id=project.project_id, request_id="summary-watch",
            title="Garden supplies", url="https://example.com/supplies", criterion="Discounts",
            schedule={"kind": "interval", "timezone": "UTC", "interval_minutes": 60}, enabled=False,
        )
        run = await store.watches.run_now(
            user_id=owner.user_id, watch_id=watch["id"], expected_revision=watch["revision"],
        )
        activity_updates = (
            ("UPDATE app_project_knowledge SET updated_at=? WHERE entry_id=?", note["id"]),
            ("UPDATE app_agent_tasks SET updated_at=? WHERE task_id=?", task["id"]),
            ("UPDATE app_watches SET updated_at=? WHERE watch_id=?", watch["id"]),
            ("UPDATE app_watch_runs SET updated_at=? WHERE run_id=?", run["active_run_id"]),
        )
        with store._lock:
            store._conn.execute("UPDATE app_projects SET updated_at='2026-01-01T00:00:00+00:00'")
            for query, identity in activity_updates:
                store._conn.execute(query, ("2026-01-02T00:00:00+00:00", identity))
            store._conn.commit()
        for query, identity in activity_updates:
            with store._lock:
                store._conn.execute(query, ("2026-01-03T00:00:00+00:00", identity))
                store._conn.commit()
            current = await store.projects.get(user_id=owner.user_id, project_id=project.project_id)
            assert current.conversation_count == 0
            assert current.last_activity_at == "2026-01-03T00:00:00+00:00"
            with store._lock:
                store._conn.execute(query, ("2026-01-02T00:00:00+00:00", identity))
                store._conn.commit()
    finally:
        store.close()


def _file(
    file_id: str,
    *,
    status: str = "ready",
    filename: str | None = None,
) -> upload_routes.FileRow:
    return upload_routes.FileRow(
        file_id=file_id,
        filename=filename or f"{file_id}.txt",
        mime="text/plain",
        bytes=42,
        uploaded_at="2026-10-02T12:00:00+00:00",
        chunks=1 if status == "ready" else 0,
        status=status,
    )


def _listing(principal: Principal) -> upload_routes.ListResponse:
    files = (
        [_file("alice-ready"), _file("alice-pending", status="pending")]
        if principal.email == "alice@example.com"
        else [_file("bob-ready")]
    )
    return upload_routes.ListResponse(
        user=principal.storage_namespace,
        files=files,
        total_bytes=sum(row.bytes for row in files),
        server_time="2026-10-02T12:01:00+00:00",
        limits=upload_routes.Limits(
            max_upload_bytes=50_000_000,
            max_user_bytes=1_000_000_000,
            allowed_extensions=[".txt"],
            chunked_max_bytes=2_000_000_000,
            part_size=8_000_000,
            fetch_hosts=[],
        ),
    )


def test_projects_api_is_owner_scoped_and_revalidates_ready_files(tmp_path, monkeypatch):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = asyncio.run(_resolve(store, subject="alice-project-api", email="alice@example.com"))
    bob = asyncio.run(_resolve(store, subject="bob-project-api", email="bob@example.com"))
    current = {"principal": alice}

    async def fake_list(_request, principal):
        return _listing(principal)

    monkeypatch.setattr(native_files, "_list_for_owner", fake_list)
    monkeypatch.setattr(project_routes.native_files, "_list_for_owner", fake_list)

    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}})
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: current["principal"]

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/projects",
                json={
                    "name": "Research",
                    "instructions": "Use the selected documents.",
                },
            )
            assert created.status_code == 201
            project_id = created.json()["id"]
            assert created.json()["conversation_count"] == 0
            assert created.json()["last_activity_at"] == created.json()["updated_at"]

            second = client.post("/api/projects", json={"name": "Second"})
            assert second.status_code == 201
            second_id = second.json()["id"]

            first_page = client.get("/api/projects?limit=1")
            assert first_page.status_code == 200
            assert first_page.json()["limits"] == {
                "max_name_chars": 100,
                "max_instructions_chars": 4000,
                "max_files": 20,
            }
            cursor = first_page.json()["next_cursor"]
            assert cursor
            assert (
                client.get(
                    "/api/projects",
                    params={"limit": 1, "cursor": cursor},
                ).status_code
                == 200
            )

            renamed = client.patch(
                f"/api/projects/{project_id}",
                json={"name": "Grounded research", "instructions": ""},
            )
            assert renamed.status_code == 200
            assert renamed.json()["name"] == "Grounded research"
            assert renamed.json()["instructions"] == ""

            added = client.post(
                f"/api/projects/{project_id}/files",
                json={"file_id": "alice-ready"},
            )
            assert added.status_code == 201
            assert added.json()["filename"] == "alice-ready.txt"
            assert (
                client.post(
                    f"/api/projects/{project_id}/files",
                    json={"file_id": "alice-ready"},
                ).status_code
                == 409
            )
            pending = client.post(
                f"/api/projects/{project_id}/files",
                json={"file_id": "alice-pending"},
            )
            assert pending.status_code == 201
            assert pending.json()["status"] == "pending"
            assert (
                client.post(
                    f"/api/projects/{project_id}/files",
                    json={"file_id": "bob-ready"},
                ).status_code
                == 404
            )

            listed_files = client.get(f"/api/projects/{project_id}/files")
            assert listed_files.status_code == 200
            assert [item["id"] for item in listed_files.json()["items"]] == [
                "alice-pending",
                "alice-ready",
            ]
            assert [item["status"] for item in listed_files.json()["items"]] == [
                "pending",
                "ready",
            ]

            conversation = client.post(
                f"/api/projects/{project_id}/conversations",
                json={"title": "Project conversation"},
            )
            assert conversation.status_code == 201
            conversation_id = conversation.json()["id"]
            assert conversation.json()["project_id"] == project_id

            another = client.post(
                f"/api/projects/{project_id}/conversations",
                json={"title": "Another project conversation"},
            )
            assert another.status_code == 201
            summary = client.get(f"/api/projects/{project_id}").json()
            assert summary["conversation_count"] == 2
            assert summary["last_activity_at"] == another.json()["updated_at"]
            listed_summary = next(item for item in client.get("/api/projects").json()["items"]
                                  if item["id"] == project_id)
            assert listed_summary["conversation_count"] == summary["conversation_count"]
            assert listed_summary["last_activity_at"] == summary["last_activity_at"]
            conversation_page = client.get(f"/api/projects/{project_id}/conversations?limit=1")
            assert conversation_page.status_code == 200
            conversation_cursor = conversation_page.json()["next_cursor"]
            assert conversation_cursor
            assert (
                client.get(
                    f"/api/projects/{project_id}/conversations",
                    params={"limit": 1, "cursor": conversation_cursor},
                ).status_code
                == 200
            )

            moved = client.patch(
                f"/api/conversations/{conversation_id}",
                json={"project_id": second_id},
            )
            assert moved.status_code == 200
            assert moved.json()["project_id"] == second_id
            ungrouped = client.patch(
                f"/api/conversations/{conversation_id}",
                json={"project_id": None},
            )
            assert ungrouped.status_code == 200
            assert ungrouped.json()["project_id"] is None

            current["principal"] = bob
            assert client.get(f"/api/projects/{project_id}").status_code == 404
            assert (
                client.patch(
                    f"/api/conversations/{conversation_id}",
                    json={"project_id": project_id},
                ).status_code
                == 404
            )

            current["principal"] = alice
            assert client.delete(f"/api/projects/{project_id}/files/alice-ready").status_code == 204
            assert client.delete(f"/api/projects/{project_id}").status_code == 204
            retained = client.get(f"/api/conversations/{another.json()['id']}")
            assert retained.status_code == 200
            assert retained.json()["project_id"] is None
    finally:
        store.close()



def test_ordinary_project_conversation_refuses_private_before_saving(tmp_path, monkeypatch):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = asyncio.run(_resolve(
        store, subject="private-project-owner", email="owner@example.com",
    ))
    project = asyncio.run(store.projects.create(
        user_id=owner.user_id, name="Protected admission",
    ))
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={
        "passthrough": {"enabled": False}, "private_processing": {"enabled": True},
    })
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner

    async def forbidden(*args, **kwargs):
        pytest.fail("Ordinary Projects must refuse a Private conversation before saving")

    monkeypatch.setattr(store.conversations, "create", forbidden)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/projects/{project.project_id}/conversations",
                json={"model_id": "private", "title": "A private conversation"},
            )
            assert response.status_code == 409
            assert response.json()["detail"] == "Start a Private Project for Private conversations."
        assert asyncio.run(store.conversations.list_page(
            user_id=owner.user_id, project_id=project.project_id, archived=False, limit=10,
        )) == ()
    finally:
        store.close()


def test_native_file_delete_prunes_project_memberships(tmp_path, monkeypatch):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = asyncio.run(_resolve(store, subject="alice-file-prune", email="alice@example.com"))
    project = asyncio.run(store.projects.create(user_id=alice.user_id, name="File cleanup"))
    asyncio.run(
        store.projects.add_file(
            user_id=alice.user_id,
            project_id=project.project_id,
            file_id="alice-ready",
        )
    )

    async def fake_delete(*, file_id, request, principal):
        assert file_id == "alice-ready"
        assert principal.storage_namespace == alice.storage_namespace
        return upload_routes.DeleteResponse(
            file_id=file_id,
            deleted=True,
            pending_cleanup=False,
        )

    monkeypatch.setattr(native_files.file_service, "delete_file", fake_delete)

    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: alice
    try:
        response = TestClient(app).delete("/api/files/alice-ready")
        assert response.status_code == 200
        relations = asyncio.run(
            store.projects.list_files(
                user_id=alice.user_id,
                project_id=project.project_id,
            )
        )
        assert relations == ()
    finally:
        store.close()
