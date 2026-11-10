"""Recovery windows preserve old charges and survive additive schema replay."""

import sqlite3

import pytest

from audrey.app_state import ApplicationStore


async def test_schema40_preserves_exhausted_legacy_attempts_until_explicit_same_agent_rerun(tmp_path):
    path = tmp_path / "application.sqlite"
    store = ApplicationStore(path)
    owner = await store.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
        display_name="Alice", role="user", auth_method="test")
    await store.agents.create(user_id=owner.user_id, objective="Prepare a garden checklist.", request_id="legacy")
    task = await store.agents.claim()
    lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
    await store.agents.reserve_operation(**lease, step_id="plan")
    await store.agents.checkpoint(**lease, step_id="plan", result="A saved plan.")
    with store._conn:
        store._conn.execute("UPDATE app_agent_tasks SET attempts=6")
    seq = await store.agents.start_event(**lease, stage="research", kind="model")
    await store.agents.finish_event(**lease, seq=seq, input_tokens=31, output_tokens=12, duration_ms=23)
    await store.agents.fail(**lease, error="A temporary failure.", retry=True)
    before = await store.agents.get(user_id=owner.user_id, task_id=task["id"])
    events = await store.agents.events(user_id=owner.user_id, task_id=task["id"])
    store.close()
    # Restore the actual schema39 table, not just an older version receipt.
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE app_agent_tasks DROP COLUMN attempt_base")
        connection.execute("DELETE FROM app_schema_migrations WHERE version=40")
        connection.execute("CREATE TRIGGER reject_schema40 BEFORE INSERT ON app_schema_migrations "
                           "WHEN NEW.version=40 BEGIN SELECT RAISE(ABORT,'fixture'); END")
    with pytest.raises(sqlite3.IntegrityError):
        ApplicationStore(path)
    with sqlite3.connect(path) as connection:
        assert "attempt_base" not in {row[1] for row in connection.execute("PRAGMA table_info(app_agent_tasks)")}
        assert connection.execute("SELECT MAX(version) FROM app_schema_migrations").fetchone()[0] == 39
        assert connection.execute("SELECT attempts FROM app_agent_tasks").fetchone()[0] == 6
        connection.execute("DROP TRIGGER reject_schema40")
    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 40
        assert await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"]) == before
        assert await upgraded.agents.events(user_id=owner.user_id, task_id=task["id"]) == events
        assert upgraded._conn.execute("SELECT attempt_base FROM app_agent_tasks").fetchone()[0] == 0
        assert await upgraded.agents.claim(max_attempts=6) is None
        failed = await upgraded.agents.get(user_id=owner.user_id, task_id=task["id"])
        assert failed["status"] == "failed" and failed["attempts"] == 6
        assert "Use Run again" in failed["error"]
        queued = await upgraded.agents.control(user_id=owner.user_id, task_id=task["id"],
            expected_revision=failed["revision"], action="rerun", max_attempts=6)
        assert queued["id"] == before["id"] and queued["steps"][0] == before["steps"][0]
    finally:
        upgraded.close()
    reopened = ApplicationStore(path)
    try:
        claimed = await reopened.agents.claim(max_attempts=6)
        assert claimed["attempts"] == 7 and claimed["attempt_base"] == 6
        assert claimed["id"] == before["id"] and claimed["current_step"] == "research"
        assert claimed["steps"][0] == before["steps"][0] and claimed["operations"] == 1
        assert "attempt_base" not in await reopened.agents.get(user_id=owner.user_id, task_id=task["id"])
        assert not reopened._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        reopened.close()


async def test_schema40_partial_column_replay_keeps_base_and_validates_integer_bounds(tmp_path):
    path = tmp_path / "application.sqlite"
    store = ApplicationStore(path)
    owner = await store.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
        display_name="Alice", role="user", auth_method="test")
    await store.agents.create(user_id=owner.user_id, objective="Prepare a checklist.", request_id="partial")
    with store._conn:
        store._conn.execute("UPDATE app_agent_tasks SET attempts=8,attempt_base=7")
        store._conn.execute("DELETE FROM app_schema_migrations WHERE version=40")
    store.close()
    replayed = ApplicationStore(path)
    try:
        assert replayed.schema_version == 40
        claimed = await replayed.agents.claim(max_attempts=2)
        assert claimed["attempts"] == 9 and claimed["attempt_base"] == 7
        column = next(row for row in replayed._conn.execute("PRAGMA table_info(app_agent_tasks)") if row["name"] == "attempt_base")
        assert column["type"] == "INTEGER" and column["notnull"] == 1
        for invalid in (-1, 1.5, "invalid", 10, None):
            with pytest.raises(sqlite3.IntegrityError), replayed._conn:
                replayed._conn.execute("UPDATE app_agent_tasks SET attempt_base=?", (invalid,))
        assert replayed._conn.execute("SELECT attempt_base FROM app_agent_tasks").fetchone()[0] == 7
        assert not replayed._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        replayed.close()
