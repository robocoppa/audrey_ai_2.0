"""Planning clarification releases compute and resumes the same fenced task."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskAccessError, AgentTaskConflictError, AgentTaskError
from audrey.app_state.notifications import NotificationAccessError


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 10, 30, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds=1):
        self.now += dt.timedelta(seconds=seconds)


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    users = [await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com", display_name=name,
        role="user", auth_method="test",
    ) for name in ("alice", "bob")]
    clock = Clock()
    store.agents._clock = store.notifications._clock = clock
    yield SimpleNamespace(store=store, owner=users[0], other=users[1], clock=clock)
    store.close()


def _lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def _create(state, **changes):
    values = dict(user_id=state.owner.user_id, objective="Compare public research.", request_id="task-1")
    values.update(changes)
    return await state.store.agents.create(**values)


async def _waiting(state, question="Which date range should I cover?", **changes):
    await _create(state, **changes)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    return await state.store.agents.request_input(**_lease(task), question=question), task


async def _answer(state, task, **changes):
    values = dict(user_id=state.owner.user_id, task_id=task["id"],
                  request_id=task["input_requests"][-1]["id"], expected_revision=task["revision"],
                  answer="Cover the last three years.")
    values.update(changes)
    return await state.store.agents.answer_input(**values)


async def test_request_releases_lease_and_wait_time_does_not_consume_active_allowance(state):
    await _create(state)
    task = await state.store.agents.claim()
    await state.store.agents.reserve_operation(**_lease(task), step_id="plan")
    state.clock.advance(3)
    waiting = await state.store.agents.request_input(**_lease(task), question="Which period?")
    assert waiting["id"] == task["id"] and waiting["status"] == "waiting_for_input"
    assert waiting["steps"][0]["status"] == "pending"
    row = state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone()
    assert row["status"] == "paused" and row["fence"] == task["fence"] + 1
    assert row["lease_token"] == "" and row["lease_expires_at"] is None and row["accounted_at"] is None
    assert row["active_seconds"] == 3 and row["operations"] == 1
    assert state.store._conn.execute("SELECT status FROM app_agent_operations").fetchone()[0] == "committed"
    state.clock.advance(86400)
    assert await state.store.agents.claim() is None
    assert state.store._conn.execute("SELECT active_seconds FROM app_agent_tasks").fetchone()[0] == 3
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.checkpoint(**_lease(task), step_id="plan", result="Stale worker result.")
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.fail(**_lease(task), error="Stale worker failure.")


async def test_question_and_waiting_state_survive_restart_and_answer_resumes_same_task(state):
    waiting, first = await _waiting(state)
    second = ApplicationStore(state.store.path)
    second.agents._clock = state.clock
    try:
        assert await second.agents.get(user_id=state.owner.user_id, task_id=waiting["id"]) == waiting
        assert await second.agents.claim() is None
        queued = await second.agents.answer_input(
            user_id=state.owner.user_id, task_id=waiting["id"], expected_revision=waiting["revision"],
            request_id=waiting["input_requests"][0]["id"], answer="Use 2023 through 2026.",
        )
        assert queued["id"] == waiting["id"] and queued["status"] == "queued"
        assert queued["input_requests"][0]["answer"] == "Use 2023 through 2026."
        resumed = await second.agents.claim()
        assert resumed["id"] == first["id"] and resumed["fence"] > first["fence"]
        assert resumed["attempts"] == 2 and resumed["operations"] == 1
        assert resumed["input_requests"] == queued["input_requests"]
    finally:
        second.close()


async def test_question_activity_has_fixed_body_current_revision_and_idempotent_read(state):
    waiting, _ = await _waiting(state, question="Private question body never belongs in the notice.")
    feed = await state.store.notifications.feed(user_id=state.owner.user_id)
    assert feed["unread_count"] == 1 and len(feed["items"]) == 1
    notice = feed["items"][0]
    assert notice["id"] == waiting["input_requests"][0]["id"] and notice["kind"] == "input_required"
    assert notice["body"] == "The agent needs your answer before it can continue."
    assert "Private question" not in repr(feed)
    assert notice["target"] == {"kind": "agent", "task_id": waiting["id"], "revision": waiting["revision"]}
    first = await state.store.notifications.mark_read(user_id=state.owner.user_id, origin="agent", notice_id=notice["id"])
    state.clock.advance()
    assert await state.store.notifications.mark_read(user_id=state.owner.user_id, origin="agent", notice_id=notice["id"]) == first
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 0
    queued = await _answer(state, waiting)
    assert queued["status"] == "queued"
    assert await state.store.notifications.feed(user_id=state.owner.user_id) == {"items": [], "unread_count": 0}
    with pytest.raises(NotificationAccessError):
        state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])


async def test_summary_reports_waiting_without_copying_question_or_answer_bodies(state):
    waiting, _ = await _waiting(state, question="Unique clarification body.")
    summary = (await state.store.agents.list(user_id=state.owner.user_id, summary=True))[0]
    assert summary["status"] == "waiting_for_input" and summary["input_requests"] == []
    assert "Unique clarification" not in repr(summary)
    state.store.agents.fence_response(user_id=state.owner.user_id, tasks=[waiting, summary])


async def test_answer_retry_is_idempotent_until_new_work_and_never_queues_twice(state):
    waiting, _ = await _waiting(state)
    first, second = await asyncio.gather(_answer(state, waiting), _answer(state, waiting))
    assert first == second and first["revision"] == waiting["revision"] + 1
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_input_requests").fetchone()[0] == 1
    assert await _answer(state, waiting, expected_revision=first["revision"]) == first
    with pytest.raises(AgentTaskConflictError):
        await _answer(state, waiting, answer="A different answer.")
    await state.store.agents.claim()
    with pytest.raises(AgentTaskConflictError):
        await _answer(state, waiting)


@pytest.mark.parametrize("change", ["revision", "request", "owner", "cancel"])
async def test_stale_foreign_and_cancelled_answers_cannot_resume_work(state, change):
    waiting, _ = await _waiting(state)
    if change == "cancel":
        await state.store.agents.control(user_id=state.owner.user_id, task_id=waiting["id"],
                                         expected_revision=waiting["revision"], action="cancel")
    values = {"revision": {"expected_revision": waiting["revision"] - 1},
              "request": {"request_id": "missing"}, "owner": {"user_id": state.other.user_id}, "cancel": {}}[change]
    with pytest.raises(AgentTaskAccessError if change == "owner" else AgentTaskConflictError):
        await _answer(state, waiting, **values)
    assert state.store._conn.execute("SELECT answer FROM app_agent_input_requests").fetchone()[0] is None


async def test_resume_cannot_skip_question_cancel_is_allowed_and_rerun_dismisses_only_unanswered(state):
    waiting, _ = await _waiting(state)
    with pytest.raises(AgentTaskConflictError, match="Answer"):
        await state.store.agents.control(user_id=state.owner.user_id, task_id=waiting["id"],
                                         expected_revision=waiting["revision"], action="resume")
    queued = await _answer(state, waiting)
    task = await state.store.agents.claim()
    waiting = await state.store.agents.request_input(**_lease(task), question="Which region?")
    cancelled = await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"],
                                                expected_revision=waiting["revision"], action="cancel")
    assert cancelled["status"] == "cancelled" and len(cancelled["input_requests"]) == 2
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 0
    rerun = await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"],
                                            expected_revision=cancelled["revision"], action="rerun")
    assert rerun["input_requests"] == queued["input_requests"]
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_input_requests").fetchone()[0] == 2


async def test_answered_clarifications_survive_saved_plan_and_completed_task_rerun(state):
    waiting, _ = await _waiting(state)
    answered = await _answer(state, waiting)
    task = await state.store.agents.claim()
    for step in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**_lease(task), step_id=step)
        task = await state.store.agents.checkpoint(**_lease(task), step_id=step, result=f"Saved {step}.")
    completed = await state.store.agents.complete(**_lease(task), result="Saved report.")
    rerun = await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"],
                                            expected_revision=completed["revision"], action="rerun")
    assert rerun["steps"][0] == completed["steps"][0]
    assert rerun["input_requests"] == answered["input_requests"]
    task = await state.store.agents.claim()
    assert task["current_step"] == "research" and task["input_requests"] == answered["input_requests"]
    with pytest.raises(AgentTaskConflictError, match="Plan"):
        await state.store.agents.request_input(**_lease(task), question="Too late to ask?")


async def test_three_questions_lifetime_limit_includes_answered_questions(state):
    waiting, _ = await _waiting(state)
    for number in range(1, 3):
        await _answer(state, waiting)
        task = await state.store.agents.claim()
        waiting = await state.store.agents.request_input(**_lease(task), question=f"Clarify detail {number}?")
    await _answer(state, waiting)
    task = await state.store.agents.claim()
    with pytest.raises(AgentTaskError, match="clarification allowance"):
        await state.store.agents.request_input(**_lease(task), question="Fourth question?")
    assert len(task["input_requests"]) == 3


async def test_dismissing_unanswered_questions_does_not_reset_lifetime_limit(state):
    waiting, _ = await _waiting(state)
    for number in range(3):
        cancelled = await state.store.agents.control(
            user_id=state.owner.user_id, task_id=waiting["id"],
            expected_revision=waiting["revision"], action="cancel",
        )
        rerun = await state.store.agents.control(
            user_id=state.owner.user_id, task_id=waiting["id"],
            expected_revision=cancelled["revision"], action="rerun",
        )
        assert rerun["input_requests"] == []
        task = await state.store.agents.claim()
        if number < 2:
            waiting = await state.store.agents.request_input(**_lease(task), question=f"Question {number + 2}?")
        else:
            with pytest.raises(AgentTaskError, match="clarification allowance"):
                await state.store.agents.request_input(**_lease(task), question="Fourth question?")
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_input_requests").fetchone()[0] == 3


@pytest.mark.parametrize("value", ["", "  ", "é" * 1001, "\ud800"])
async def test_question_is_bounded_by_utf8_bytes(state, value):
    await _create(state)
    task = await state.store.agents.claim()
    with pytest.raises(AgentTaskError):
        await state.store.agents.request_input(**_lease(task), question=value)
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_input_requests").fetchone()[0] == 0


@pytest.mark.parametrize("value", ["", "  ", "é" * 4001, "\ud800"])
async def test_answer_is_bounded_by_utf8_bytes(state, value):
    waiting, _ = await _waiting(state)
    with pytest.raises(AgentTaskError):
        await _answer(state, waiting, answer=value)
    assert (await state.store.agents.get(user_id=state.owner.user_id, task_id=waiting["id"]))["status"] == "waiting_for_input"


async def test_answer_capacity_excludes_its_already_counted_waiting_task(state):
    waiting, _ = await _waiting(state, limits={"max_queued_per_user": 1, "max_queued_global": 1})
    queued = await _answer(state, waiting)
    assert queued["status"] == "queued"


async def test_current_reduced_capacity_rejects_answer_without_losing_pending_question(state):
    waiting, _ = await _waiting(state)
    await _create(state, request_id="second-task")
    with pytest.raises(AgentTaskError, match="capacity"):
        await _answer(state, waiting, admission_limits={"max_queued_per_user": 1})
    assert (await state.store.agents.get(user_id=state.owner.user_id, task_id=waiting["id"])) == waiting


@pytest.mark.parametrize("limit", ["attempts", "operations", "active_seconds"])
async def test_answer_cannot_reset_or_escape_cumulative_execution_limits(state, limit):
    waiting, _ = await _waiting(state)
    with state.store._lock, state.store._conn:
        sql = {
            "attempts": "UPDATE app_agent_tasks SET attempts=6",
            "operations": "UPDATE app_agent_tasks SET operations=96",
            "active_seconds": "UPDATE app_agent_tasks SET active_seconds=1800",
        }[limit]
        state.store._conn.execute(sql)
    with pytest.raises(AgentTaskError, match="allowance"):
        await _answer(state, waiting)
    assert state.store._conn.execute("SELECT answer FROM app_agent_input_requests").fetchone()[0] is None


@pytest.mark.parametrize("operation", ["question", "answer"])
async def test_question_or_answer_and_task_transition_roll_back_together(state, operation):
    if operation == "question":
        await _create(state)
        task = await state.store.agents.claim()
    else:
        task, _ = await _waiting(state)
    before = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "CREATE TRIGGER reject_input_insert BEFORE INSERT ON app_agent_input_requests "
            "BEGIN SELECT RAISE(ABORT,'simulated question failure'); END" if operation == "question" else
            "CREATE TRIGGER reject_input_update BEFORE UPDATE ON app_agent_tasks "
            "BEGIN SELECT RAISE(ABORT,'simulated answer transition failure'); END",
        )
    with pytest.raises(sqlite3.IntegrityError):
        if operation == "question":
            await state.store.agents.request_input(**_lease(task), question="Which period?")
        else:
            await _answer(state, task)
    assert await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == before
    assert not state.store._conn.in_transaction


async def test_task_response_fence_rejects_answered_or_deleted_question_bodies(state):
    waiting, _ = await _waiting(state)
    await _answer(state, waiting)
    with pytest.raises(AgentTaskConflictError):
        state.store.agents.fence_response(user_id=state.owner.user_id, tasks=[waiting])
    await state.store.purge_local_user_data(user_id=state.owner.user_id)
    with pytest.raises(AgentTaskAccessError):
        state.store.agents.fence_response(user_id=state.owner.user_id, tasks=[waiting])
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_input_requests").fetchone()[0] == 0
    assert state.store._conn.execute("PRAGMA foreign_key_check").fetchall() == []


async def test_additive_migration_preserves_existing_tasks_notices_and_no_input_backfill(state):
    await _create(state)
    task = await state.store.agents.claim()
    failed = await state.store.agents.fail(**_lease(task), error="A saved failure.")
    with state.store._lock, state.store._conn:
        state.store._conn.execute("DROP TABLE app_agent_input_requests")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version=31")
    second = ApplicationStore(state.store.path)
    try:
        assert second.schema_version == 38
        assert await second.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == failed
        assert (await second.notifications.feed(user_id=state.owner.user_id))["unread_count"] == 1
        assert second._conn.execute("SELECT COUNT(*) FROM app_agent_input_requests").fetchone()[0] == 0
    finally:
        second.close()
