"""Schema 38 preserves populated knowledge, history, run pins and Agent fences."""

import sqlite3

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state import store as store_module
from audrey.app_state.agents import AgentTaskContextError
from audrey.app_state.migrations import MIGRATIONS

TABLES = (
    "app_project_knowledge", "app_project_knowledge_revisions", "app_run_project_knowledge",
    "app_run_project_knowledge_snapshots", "app_agent_project_contexts",
)


def rows(connection):
    return {table: sorted(tuple(row) for row in connection.execute(f"SELECT * FROM {table}")) for table in TABLES}  # noqa: S608 - fixed table names


async def completed_report(store, user_id):
    await store.agents.create(user_id=user_id, objective="Prepare a garden report.", request_id="legacy-report")
    task = await store.agents.claim()
    lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
    for step_id in ("plan", "research", "review"):
        await store.agents.reserve_operation(**lease, step_id=step_id)
        await store.agents.checkpoint(**lease, step_id=step_id, result="Saved garden report.")
    return await store.agents.complete(**lease, result="A completed garden report.")


async def test_schema38_preserves_populated_schema37_and_rolls_back_failed_rebuild(tmp_path, monkeypatch):
    path = tmp_path / "application.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "MIGRATIONS", tuple(item for item in MIGRATIONS if item[0] <= 37))
        legacy = ApplicationStore(path)
        try:
            owner = await legacy.resolve_external_identity(
                provider="test", subject="alice", email="alice@example.com", display_name="Alice",
                role="user", auth_method="test",
            )
            project = await legacy.projects.create(user_id=owner.user_id, name="Garden", instructions="Use the saved label.")
            args = {"user_id": owner.user_id, "project_id": project.project_id}
            entry = await legacy.knowledge.create(**args, kind="decision", text="The garden label is Fern Circle.", pinned=True)
            for revision in range(2, 101):
                entry = await legacy.knowledge.update(
                    **args, entry_id=entry["id"], expected_revision=entry["revision"],
                    text=f"The garden label is Fern Circle. Legacy revision {revision}.",
                )
            report = await completed_report(legacy, owner.user_id)
            proposal = await legacy.knowledge.create(
                **args, kind="action", text="Review the buying list.", origin="agent_report",
                source_task_id=report["id"], replaces_id=entry["id"], base_revision=100,
                sources=[{"source_id": f"agent:{report['id']}:report", "revision": str(report["revision"])}],
            )
            rejected = await legacy.knowledge.create(
                **args, kind="question", text="An unused suggestion.", origin="agent_report", source_task_id=report["id"],
            )
            rejected = await legacy.knowledge.decide(**args, entry_id=rejected["id"], expected_revision=1, action="reject")
            conversation = await legacy.conversations.create(**args)
            run = await legacy.conversations.begin_run(
                user_id=owner.user_id, conversation_id=conversation.conversation_id, user_content="Which garden label?",
            )
            await legacy.knowledge.record_run_snapshot(**args, run_id=run.run.run_id, entries=[{"id": entry["id"], "revision": 100}])
            await legacy.conversations.finish_run(user_id=owner.user_id, run_id=run.run.run_id,
                outcome="succeeded", assistant_content="Fern Circle.")
            frozen = await legacy.agents.create(**args, objective="Use the garden label.", request_id="frozen-legacy")
            before = rows(legacy._conn)
            assert legacy.schema_version == 37 and len(before["app_project_knowledge_revisions"]) == 103
        finally:
            legacy.close()

    with monkeypatch.context() as patch:
        broken = tuple((version, sql.replace(
            "ALTER TABLE app_project_knowledge_v38", "SELECT missing_migration_fixture; ALTER TABLE app_project_knowledge_v38",
        )) if version == 38 else (version, sql) for version, sql in MIGRATIONS)
        patch.setattr(store_module, "MIGRATIONS", broken)
        with pytest.raises(sqlite3.OperationalError):
            ApplicationStore(path)
    with sqlite3.connect(path) as connection:
        assert rows(connection) == before
        assert connection.execute("SELECT MAX(version) FROM app_schema_migrations").fetchone()[0] == 37
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name LIKE '%_v38'").fetchall()
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 39 and rows(upgraded._conn) == before
        assert await upgraded.agents.get(user_id=owner.user_id, task_id=frozen["id"]) == frozen
        assert await upgraded.knowledge.get(**args, entry_id=proposal["id"]) == proposal
        assert await upgraded.knowledge.get(**args, entry_id=rejected["id"]) == rejected
        assert upgraded._conn.execute("SELECT name FROM sqlite_master WHERE name='idx_app_project_knowledge_owner_updated'").fetchone()
        for sql in (
            "UPDATE app_project_knowledge_revisions SET record_json=record_json",
            "UPDATE app_run_project_knowledge SET revision=revision",
            "UPDATE app_run_project_knowledge_snapshots SET project_id=project_id",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"), upgraded._conn:
                upgraded._conn.execute(sql)
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_project_knowledge SET revision=? WHERE entry_id=?", (2**53, entry["id"]))
        claimed = await upgraded.agents.claim()
        lease = {"task_id": claimed["id"], "lease_token": claimed["lease_token"], "fence": claimed["fence"]}
        context = await upgraded.agents.resolve_project_context(**lease)
        assert context["knowledge"][0]["revision"] == 100
        snapshot = await upgraded.knowledge.get_text(**args)
        result, _changed = await upgraded.knowledge.save_text(
            **args, text="The reviewed garden label is Moss Circle.", expected_revision=snapshot["revision"],
        )
        assert result["text"] == "The reviewed garden label is Moss Circle."
        old = await upgraded.knowledge.get(**args, entry_id=entry["id"])
        assert old["revision"] == 101 and old["approval_state"] == "superseded"
        assert upgraded._conn.execute("SELECT COUNT(*) FROM app_project_knowledge_revisions WHERE entry_id=?", (entry["id"],)).fetchone()[0] == 101
        assert await upgraded.knowledge.get_run_snapshot(**args, run_id=run.run.run_id) == [{"id": entry["id"], "revision": 100}]
        with pytest.raises(AgentTaskContextError):
            await upgraded.agents.resolve_project_context(**lease)

        document = (await upgraded.knowledge.list_approved(**args))[0]
        for revision in range(2, 102):
            document = await upgraded.knowledge.update(
                **args, entry_id=document["id"], expected_revision=document["revision"], text=f"Document revision {revision}.",
            )
        next_run = await upgraded.conversations.begin_run(
            user_id=owner.user_id, conversation_id=conversation.conversation_id, user_content="Use the current note.",
        )
        await upgraded.knowledge.record_run_snapshot(**args, run_id=next_run.run.run_id,
            entries=[{"id": document["id"], "revision": 101}])
        assert await upgraded.knowledge.get_run_snapshot(**args, run_id=next_run.run.run_id) == [{"id": document["id"], "revision": 101}]
        assert not upgraded._conn.execute("PRAGMA foreign_key_check").fetchall()
        await upgraded.projects.delete(**args)
        for table in TABLES[:-1]:
            assert upgraded._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # noqa: S608 - fixed table names
        assert not (await upgraded.agents.get(user_id=owner.user_id, task_id=frozen["id"]))["project_context"]["available"]
        assert not upgraded._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        upgraded.close()
