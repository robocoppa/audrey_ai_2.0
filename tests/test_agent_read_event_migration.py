"""The new native read tool preserves existing event history and constraints."""

import sqlite3

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state import store as store_module
from audrey.app_state.migrations import MIGRATIONS


async def test_schema37_preserves_schema36_events_and_allows_selected_file_reads(tmp_path, monkeypatch):
    path = tmp_path / "application.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "MIGRATIONS", tuple(item for item in MIGRATIONS if item[0] <= 36))
        legacy = ApplicationStore(path)
        # Current repository methods need this private field while generating
        # historical rows. Remove it below before exercising any migration.
        legacy._conn.execute(
            "ALTER TABLE app_agent_tasks ADD COLUMN attempt_base INTEGER NOT NULL DEFAULT 0 "
            "CHECK (typeof(attempt_base)='integer' AND attempt_base>=0 AND attempt_base<=attempts)",
        )
        try:
            owner = await legacy.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
                display_name="Alice", role="user", auth_method="test")
            await legacy.agents.create(user_id=owner.user_id, objective="Prepare the requested work.", request_id="legacy")
            task = await legacy.agents.claim()
            lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
            await legacy.agents.reserve_operation(**lease, step_id="plan")
            seq = await legacy.agents.start_event(**lease, stage="plan", kind="model")
            await legacy.agents.finish_event(**lease, seq=seq, input_tokens=31, output_tokens=12, duration_ms=23)
            await legacy.agents.checkpoint(**lease, step_id="plan", result="A saved plan.")
            await legacy.agents.reserve_operation(**lease, step_id="research")
            seq = await legacy.agents.start_event(**lease, stage="research", kind="tool", tool="create_file")
            await legacy.agents.finish_event(**lease, seq=seq, state="failed", duration_ms=7)
            before = await legacy.agents.events(user_id=owner.user_id, task_id=task["id"])
            before_task = await legacy.agents.get(user_id=owner.user_id, task_id=task["id"])
            assert legacy.schema_version == 36
        finally:
            legacy.close()

    with sqlite3.connect(path) as connection:
        assert not connection.execute("SELECT 1 FROM app_agent_tasks WHERE attempt_base!=0").fetchone()
        connection.execute("ALTER TABLE app_agent_tasks DROP COLUMN attempt_base")
        assert "attempt_base" not in {row[1] for row in connection.execute("PRAGMA table_info(app_agent_tasks)")}
    # A failure after the old table is dropped must still restore that table
    # and its rows through the migration runner's transaction rollback.
    with monkeypatch.context() as patch:
        broken = tuple((version, sql.replace("ALTER TABLE app_agent_events_v37", "SELECT missing_migration_fixture; ALTER TABLE app_agent_events_v37"))
                       if version == 37 else (version, sql) for version, sql in MIGRATIONS)
        patch.setattr(store_module, "MIGRATIONS", broken)
        with pytest.raises(sqlite3.OperationalError):
            ApplicationStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == len(before["items"])
        assert connection.execute("SELECT MAX(version) FROM app_schema_migrations").fetchone()[0] == 36
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='app_agent_events_v37'").fetchone()
    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 40
        assert await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"]) == before
        assert await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"]) == before_task
        seq = await upgraded.agents.start_event(**lease, stage="research", kind="tool", tool="read_selected_file")
        await upgraded.agents.finish_event(**lease, seq=seq, duration_ms=8)
        event = (await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"]))["items"][0]
        assert event["tool"] == "read_selected_file" and event["seq"] == before["items"][0]["seq"] + 1
        assert event["state"] == "completed"
        assert not upgraded._conn.execute("PRAGMA foreign_key_check").fetchall()
        assert upgraded._conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_app_agent_events_owner'").fetchone()
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_agent_events SET tool='arbitrary' WHERE task_id=? AND seq=?", (task["id"], seq))
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_agent_events SET stage='plan' WHERE task_id=? AND seq=?", (task["id"], seq))
        current = await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"])
        cancelled = await upgraded.agents.control(user_id=owner.user_id, task_id=task["id"], expected_revision=current["revision"], action="cancel")
        await upgraded.agents.delete(user_id=owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"])
        assert upgraded._conn.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == 0
    finally:
        upgraded.close()
