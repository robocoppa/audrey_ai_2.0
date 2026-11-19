"""Schema 41 retains saved questions and atomically permits Work questions."""

import sqlite3

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state import store as store_module
from audrey.app_state.migrations import MIGRATIONS


def _lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def _question(store, owner, request_id):
    await store.agents.create(user_id=owner.user_id, objective="Prepare a garden checklist.", request_id=request_id)
    task = await store.agents.claim()
    await store.agents.reserve_operation(**_lease(task), step_id="plan")
    return await store.agents.request_input(**_lease(task), question=f"Question for {request_id}?")


async def test_schema41_preserves_pending_answered_and_dismissed_plan_questions(tmp_path, monkeypatch):
    path = tmp_path / "application.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "MIGRATIONS", tuple(item for item in MIGRATIONS if item[0] <= 40))
        legacy = ApplicationStore(path)
        owner = await legacy.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
            display_name="Alice", role="user", auth_method="test")
        first = await _question(legacy, owner, "answered")
        await legacy.notifications.mark_read(user_id=owner.user_id, origin="agent", notice_id=first["input_requests"][0]["id"])
        answered = await legacy.agents.answer_input(user_id=owner.user_id, task_id=first["id"],
            request_id=first["input_requests"][0]["id"], expected_revision=first["revision"], answer="Fern Circle")
        await legacy.agents.control(user_id=owner.user_id, task_id=first["id"], expected_revision=answered["revision"], action="cancel")
        pending = await _question(legacy, owner, "pending")
        dismissed = await _question(legacy, owner, "dismissed")
        cancelled = await legacy.agents.control(user_id=owner.user_id, task_id=dismissed["id"],
            expected_revision=dismissed["revision"], action="cancel")
        queued = await legacy.agents.control(user_id=owner.user_id, task_id=dismissed["id"],
            expected_revision=cancelled["revision"], action="rerun")
        await legacy.agents.control(user_id=owner.user_id, task_id=dismissed["id"], expected_revision=queued["revision"], action="pause")
        before_rows = [dict(row) for row in legacy._conn.execute("SELECT * FROM app_agent_input_requests ORDER BY request_id")]
        before_tasks = await legacy.agents.list(user_id=owner.user_id)
        before_feed = await legacy.notifications.feed(user_id=owner.user_id)
        assert len(before_rows) == 3 and any(row["read_at"] for row in before_rows)
        assert any(row["answer"] for row in before_rows) and any(row["dismissed_at"] for row in before_rows)
        assert legacy.schema_version == 40
        legacy.close()

    # Fail after rebuilding the table: the old CHECK and all rows must roll back.
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TRIGGER reject_schema41 BEFORE INSERT ON app_schema_migrations "
            "WHEN NEW.version=41 BEGIN SELECT RAISE(ABORT,'fixture'); END")
    with pytest.raises(sqlite3.IntegrityError):
        ApplicationStore(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("SELECT MAX(version) FROM app_schema_migrations").fetchone()[0] == 40
        assert [dict(row) for row in connection.execute("SELECT * FROM app_agent_input_requests ORDER BY request_id")] == before_rows
        assert "step_id='plan'" in connection.execute("SELECT sql FROM sqlite_master WHERE name='app_agent_input_requests'").fetchone()[0]
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='app_agent_input_requests_v41'").fetchone()
        connection.execute("DROP TRIGGER reject_schema41")

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 44
        assert [dict(row) for row in upgraded._conn.execute("SELECT * FROM app_agent_input_requests ORDER BY request_id")] == before_rows
        assert await upgraded.agents.list(user_id=owner.user_id) == before_tasks
        assert await upgraded.notifications.feed(user_id=owner.user_id) == before_feed
        assert (await upgraded.agents.get(user_id=owner.user_id, task_id=pending["id"]))["input_requests"] == pending["input_requests"]
        current = await upgraded.agents.get(user_id=owner.user_id, task_id=first["id"])
        await upgraded.agents.control(user_id=owner.user_id, task_id=first["id"], expected_revision=current["revision"], action="rerun")
        task = await upgraded.agents.claim()
        await upgraded.agents.reserve_operation(**_lease(task), step_id="plan")
        task = await upgraded.agents.checkpoint(**_lease(task), step_id="plan", result="The original saved Plan.")
        await upgraded.agents.reserve_operation(**_lease(task), step_id="research")
        work_wait = await upgraded.agents.request_input(**_lease(task), step_id="research", question="Which checklist format?")
        assert work_wait["steps"][0]["result"] == "The original saved Plan."
        assert [question["step_id"] for question in work_wait["input_requests"]] == ["plan", "research"]
        for invalid in ("review", "title", "other"):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"), upgraded._conn:
                upgraded._conn.execute("UPDATE app_agent_input_requests SET step_id=? WHERE request_id=?",
                    (invalid, work_wait["input_requests"][-1]["id"]))
        assert not upgraded._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        upgraded.close()
    reopened = ApplicationStore(path)
    try:
        assert await reopened.agents.get(user_id=owner.user_id, task_id=first["id"]) == work_wait
        assert reopened._conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name IN "
            "('idx_app_agent_input_pending','idx_app_agent_input_owner_task')").fetchone()[0] == 2
        assert not reopened._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        reopened.close()
