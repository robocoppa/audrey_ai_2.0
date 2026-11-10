"""Review native activity widens the event schema without losing owned history."""

import sqlite3

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state import store as store_module
from audrey.app_state.agents import AgentTaskError
from audrey.app_state.migrations import MIGRATIONS


async def test_schema39_preserves_schema38_history_and_allows_only_native_review_tools(tmp_path, monkeypatch):
    path = tmp_path / "application.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "MIGRATIONS", tuple(item for item in MIGRATIONS if item[0] <= 38))
        legacy = ApplicationStore(path)
        try:
            owner = await legacy.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
                display_name="Alice", role="user", auth_method="test")
            other = await legacy.resolve_external_identity(provider="test", subject="bob", email="bob@example.com",
                display_name="Bob", role="user", auth_method="test")
            await legacy.agents.create(user_id=owner.user_id, objective="Prepare a budget file.", request_id="legacy")
            task = await legacy.agents.claim()
            lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
            await legacy.agents.reserve_operation(**lease, step_id="plan")
            seq = await legacy.agents.start_event(**lease, stage="plan", kind="model")
            await legacy.agents.finish_event(**lease, seq=seq, input_tokens=31, output_tokens=12, duration_ms=23)
            await legacy.agents.checkpoint(**lease, step_id="plan", result="A saved plan.")
            await legacy.agents.reserve_operation(**lease, step_id="research")
            for tool, state in (("web_fetch", "completed"), ("read_selected_file", "failed")):
                seq = await legacy.agents.start_event(**lease, stage="research", kind="tool", tool=tool)
                await legacy.agents.finish_event(**lease, seq=seq, state=state, duration_ms=7)
            await legacy.agents.checkpoint(**lease, step_id="research", result="A saved budget.")
            seq = await legacy.agents.start_event(**lease, stage="review", kind="model")
            before = await legacy.agents.events(user_id=owner.user_id, task_id=task["id"])
            before_task = await legacy.agents.get(user_id=owner.user_id, task_id=task["id"])
            assert legacy.schema_version == 38
        finally:
            legacy.close()

    # Failure after dropping the original table must roll back rows, schema
    # version and owner index together, leaving startup retry safe.
    with monkeypatch.context() as patch:
        broken = tuple((version, sql.replace(
            "ALTER TABLE app_agent_events_v39", "SELECT missing_migration_fixture; ALTER TABLE app_agent_events_v39",
        )) if version == 39 else (version, sql) for version, sql in MIGRATIONS)
        patch.setattr(store_module, "MIGRATIONS", broken)
        with pytest.raises(sqlite3.OperationalError):
            ApplicationStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == len(before["items"])
        assert connection.execute("SELECT MAX(version) FROM app_schema_migrations").fetchone()[0] == 38
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='app_agent_events_v39'").fetchone()
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='idx_app_agent_events_owner'").fetchone()

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 39
        assert await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"]) == before
        assert await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"]) == before_task
        await upgraded.agents.finish_event(**lease, seq=seq, input_tokens=49, output_tokens=20, duration_ms=11)
        for tool in ("read_selected_file", "create_file"):
            seq = await upgraded.agents.start_event(**lease, stage="review", kind="tool", tool=tool)
            await upgraded.agents.finish_event(**lease, seq=seq, duration_ms=8)
        events = (await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"]))["items"]
        assert [(item["stage"], item["tool"], item["state"]) for item in events[:2]] == [
            ("review", "create_file", "completed"), ("review", "read_selected_file", "completed"),
        ]
        assert not upgraded._conn.execute("PRAGMA foreign_key_check").fetchall()
        for tool in ("web_search", "web_fetch"):
            with pytest.raises(AgentTaskError):
                await upgraded.agents.start_event(**lease, stage="review", kind="tool", tool=tool)
            with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
                upgraded._conn.execute("UPDATE app_agent_events SET tool=? WHERE task_id=? AND seq=?", (tool, task["id"], seq))
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_agent_events SET stage='plan' WHERE task_id=? AND seq=?", (task["id"], seq))
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_agent_events SET user_id=? WHERE task_id=? AND seq=?", (other.user_id, task["id"], seq))
        current = await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"])
        cancelled = await upgraded.agents.control(user_id=owner.user_id, task_id=task["id"], expected_revision=current["revision"], action="cancel")
        await upgraded.agents.delete(user_id=owner.user_id, task_id=task["id"], expected_revision=cancelled["revision"])
        assert upgraded._conn.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == 0
    finally:
        upgraded.close()
