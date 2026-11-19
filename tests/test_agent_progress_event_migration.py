"""Schema 43 preserves saved questions and events while bounding progress metadata."""

import sqlite3

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state import store as store_module
from audrey.app_state.agents import AgentTaskError
from audrey.app_state.migrations import MIGRATIONS


def lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def test_schema43_preserves_schema42_questions_and_activity_with_work_only_progress(tmp_path, monkeypatch):
    path = tmp_path / "application.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "MIGRATIONS", tuple(item for item in MIGRATIONS if item[0] <= 42))
        legacy = ApplicationStore(path)
        owner = await legacy.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
            display_name="Alice", role="user", auth_method="test")
        await legacy.agents.create(user_id=owner.user_id, objective="Draft a short checklist.", request_id="legacy")
        task = await legacy.agents.claim()
        await legacy.agents.reserve_operation(**lease(task), step_id="plan")
        waiting = await legacy.agents.request_input(**lease(task), question="Which garden name?")
        await legacy.notifications.mark_read(user_id=owner.user_id, origin="agent", notice_id=waiting["input_requests"][0]["id"])
        await legacy.agents.answer_input(user_id=owner.user_id, task_id=task["id"],
            request_id=waiting["input_requests"][0]["id"], expected_revision=waiting["revision"], answer="Fern Circle")
        task = await legacy.agents.claim()
        await legacy.agents.reserve_operation(**lease(task), step_id="plan")
        seq = await legacy.agents.start_event(**lease(task), stage="plan", kind="model")
        await legacy.agents.finish_event(**lease(task), seq=seq, duration_ms=12, input_tokens=31, output_tokens=16)
        await legacy.agents.checkpoint(**lease(task), step_id="plan", result="The saved Plan.")
        await legacy.agents.reserve_operation(**lease(task), step_id="research")
        seq = await legacy.agents.start_event(**lease(task), stage="research", kind="tool", tool="calculate")
        await legacy.agents.finish_event(**lease(task), seq=seq, state="failed", duration_ms=2)
        pending = await legacy.agents.start_event(**lease(task), stage="research", kind="model")
        before_events = await legacy.agents.events(user_id=owner.user_id, task_id=task["id"])
        before_task = await legacy.agents.get(user_id=owner.user_id, task_id=task["id"])
        before_questions = [dict(row) for row in legacy._conn.execute("SELECT * FROM app_agent_input_requests")]
        assert legacy.schema_version == 42
        legacy.close()

    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TRIGGER reject_schema43 BEFORE INSERT ON app_schema_migrations "
            "WHEN NEW.version=43 BEGIN SELECT RAISE(ABORT,'fixture'); END")
    with pytest.raises(sqlite3.IntegrityError):
        ApplicationStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT MAX(version) FROM app_schema_migrations").fetchone()[0] == 42
        assert connection.execute("SELECT COUNT(*) FROM app_agent_events").fetchone()[0] == len(before_events["items"])
        assert "save_progress" not in connection.execute("SELECT sql FROM sqlite_master WHERE name='app_agent_events'").fetchone()[0]
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='app_agent_events_v43'").fetchone()
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='idx_app_agent_events_owner'").fetchone()
        connection.execute("DROP TRIGGER reject_schema43")

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 44
        assert await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"]) == before_events
        assert await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"]) == before_task
        assert [dict(row) for row in upgraded._conn.execute("SELECT * FROM app_agent_input_requests")] == before_questions
        await upgraded.agents.finish_event(**lease(task), seq=pending, duration_ms=15, input_tokens=42, output_tokens=19)
        progress = await upgraded.agents.save_work_progress(**lease(task), note="The first checklist section is drafted.")
        seq = await upgraded.agents.start_event(**lease(task), stage="research", kind="tool", tool="save_progress")
        await upgraded.agents.finish_event(**lease(task), seq=seq, duration_ms=2)
        for invalid in ("plan", "review", "title"):
            with pytest.raises(AgentTaskError):
                await upgraded.agents.start_event(**lease(task), stage=invalid, kind="tool", tool="save_progress")
            with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
                upgraded._conn.execute("UPDATE app_agent_events SET stage=? WHERE task_id=? AND seq=?", (invalid, task["id"], seq))
        with pytest.raises(sqlite3.IntegrityError), upgraded._conn:
            upgraded._conn.execute("UPDATE app_agent_events SET output_tokens=2 WHERE task_id=? AND seq=?", (task["id"], seq))
        await upgraded.agents.checkpoint(**lease(task), step_id="research", result="Saved checklist draft.")
        seq = await upgraded.agents.start_event(**lease(task), stage="review", kind="tool", tool="calculate")
        await upgraded.agents.finish_event(**lease(task), seq=seq, duration_ms=2)
        current_events = await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"])
        current_task = await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"])
        assert current_task["steps"][1]["progress"] == progress
        columns = {row["name"] for row in upgraded._conn.execute("PRAGMA table_info(app_agent_events)")}
        assert not {"note", "progress", "arguments", "result", "text"} & columns
        assert not upgraded._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        upgraded.close()
    reopened = ApplicationStore(path)
    try:
        assert await reopened.agents.events(user_id=owner.user_id, task_id=task["id"]) == current_events
        assert await reopened.agents.get(user_id=owner.user_id, task_id=task["id"]) == current_task
        assert [dict(row) for row in reopened._conn.execute("SELECT * FROM app_agent_input_requests")] == before_questions
        assert not reopened._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        reopened.close()
