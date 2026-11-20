"""Schema 42 preserves owned activity while permitting native calculator metadata."""

import sqlite3

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state import store as store_module
from audrey.app_state.agents import AgentTaskError
from audrey.app_state.migrations import MIGRATIONS


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def test_schema42_preserves_schema41_events_and_bounds_calculator_stages(tmp_path, monkeypatch):
    path = tmp_path / "application.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "MIGRATIONS", tuple(item for item in MIGRATIONS if item[0] <= 41))
        legacy = ApplicationStore(path)
        owner = await legacy.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
            display_name="Alice", role="user", auth_method="test")
        other = await legacy.resolve_external_identity(provider="test", subject="bob", email="bob@example.com",
            display_name="Bob", role="user", auth_method="test")
        await legacy.agents.create(user_id=owner.user_id, objective="Prepare a budget.", request_id="legacy")
        task = await legacy.agents.claim()
        await legacy.agents.reserve_operation(**lease(task), step_id="plan")
        seq = await legacy.agents.start_event(**lease(task), stage="plan", kind="model")
        await legacy.agents.finish_event(**lease(task), seq=seq, duration_ms=23, input_tokens=31, output_tokens=12)
        await legacy.agents.checkpoint(**lease(task), step_id="plan", result="A saved plan.")
        await legacy.agents.reserve_operation(**lease(task), step_id="research")
        for tool, outcome in (("read_selected_file", "completed"), ("create_file", "failed")):
            seq = await legacy.agents.start_event(**lease(task), stage="research", kind="tool", tool=tool)
            await legacy.agents.finish_event(**lease(task), seq=seq, duration_ms=7, state=outcome)
        await legacy.agents.checkpoint(**lease(task), step_id="research", result="Saved work.")
        pending = await legacy.agents.start_event(**lease(task), stage="review", kind="model")
        before_events = await legacy.agents.events(user_id=owner.user_id, task_id=task["id"])
        before_task = await legacy.agents.get(user_id=owner.user_id, task_id=task["id"])
        assert legacy.schema_version == 41
        legacy.close()

    # Roll back a failure after replacing the populated old table.
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TRIGGER reject_schema42 BEFORE INSERT ON app_schema_migrations "
            "WHEN NEW.version=42 BEGIN SELECT RAISE(ABORT,'fixture'); END")
    with pytest.raises(sqlite3.IntegrityError):
        ApplicationStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT MAX(version) FROM app_schema_migrations").fetchone()[0] == 41
        assert connection.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == len(before_events["items"])
        assert "calculate" not in connection.execute("SELECT sql FROM sqlite_master WHERE name='app_agent_events'").fetchone()[0]
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='app_agent_events_v42'").fetchone()
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='idx_app_agent_events_owner'").fetchone()
        connection.execute("DROP TRIGGER reject_schema42")

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 45
        assert await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"]) == before_events
        assert await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"]) == before_task
        await upgraded.agents.finish_event(**lease(task), seq=pending, duration_ms=11, input_tokens=49, output_tokens=20)
        seq = await upgraded.agents.start_event(**lease(task), stage="review", kind="tool", tool="calculate")
        await upgraded.agents.finish_event(**lease(task), seq=seq, duration_ms=2)
        for invalid in ("plan", "title"):
            with pytest.raises(AgentTaskError):
                await upgraded.agents.start_event(**lease(task), stage=invalid, kind="tool", tool="calculate")
            with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
                upgraded._conn.execute("UPDATE app_agent_events SET stage=? WHERE task_id=? AND seq=?", (invalid, task["id"], seq))
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_agent_events SET input_tokens=3 WHERE task_id=? AND seq=?", (task["id"], seq))
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_agent_events SET user_id=? WHERE task_id=? AND seq=?", (other.user_id, task["id"], seq))
        await upgraded.agents.create(user_id=other.user_id, objective="Check arithmetic.", request_id="other")
        second = await upgraded.agents.claim()
        await upgraded.agents.reserve_operation(**lease(second), step_id="plan")
        await upgraded.agents.checkpoint(**lease(second), step_id="plan", result="Calculate the total.")
        seq = await upgraded.agents.start_event(**lease(second), stage="research", kind="tool", tool="calculate")
        await upgraded.agents.finish_event(**lease(second), seq=seq, state="failed", duration_ms=2)
        current_events = await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"])
        current_task = await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"])
        assert not upgraded._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        upgraded.close()
    reopened = ApplicationStore(path)
    try:
        assert await reopened.agents.events(user_id=owner.user_id, task_id=task["id"]) == current_events
        assert await reopened.agents.get(user_id=owner.user_id, task_id=task["id"]) == current_task
        cancelled = await reopened.agents.control(user_id=owner.user_id, task_id=task["id"], expected_revision=current_task["revision"], action="cancel")
        await reopened.agents.delete(user_id=owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"])
        assert not reopened._conn.execute("SELECT 1 FROM app_agent_events WHERE task_id=?", (task["id"],)).fetchone()
        assert reopened._conn.execute("SELECT 1 FROM app_agent_events WHERE task_id=?", (second["id"],)).fetchone()
        assert not reopened._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        reopened.close()
