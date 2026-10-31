"""Current-result activity is atomic, bounded and fenced by canonical owners."""

from __future__ import annotations

import datetime as dt
import sqlite3
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.agents import AgentTaskConflictError
from audrey.app_state.notifications import (
    NotificationAccessError,
    NotificationConflictError,
    NotificationError,
    insert_agent_notice,
)


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
    for repository in (store.agents, store.watches, store.notifications):
        repository._clock = clock
    yield SimpleNamespace(store=store, owner=users[0], other=users[1], clock=clock)
    store.close()


def _lease(task):
    return {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}


async def _create(state, request_id="agent-1", **changes):
    values = dict(user_id=state.owner.user_id, objective="Research public orchard releases.",
                  request_id=request_id)
    values.update(changes)
    return await state.store.agents.create(**values)


async def _ready(state, request_id="agent-1", **changes):
    await _create(state, request_id=request_id, **changes)
    task = await state.store.agents.claim()
    for step_id in ("plan", "research", "review"):
        await state.store.agents.reserve_operation(**_lease(task), step_id=step_id)
        task = await state.store.agents.checkpoint(
            **_lease(task), step_id=step_id, result="Sensitive step source body.",
        )
    return task


async def _complete(state, request_id="agent-1", **changes):
    task = await _ready(state, request_id=request_id, **changes)
    return await state.store.agents.complete(**_lease(task), result="Sensitive final report body.")


async def _watch_failure(state, request_id="watch-1"):
    watch = await state.store.watches.create(
        user_id=state.owner.user_id, title="Public releases", url="https://example.com/releases",
        criterion="Notify about stable releases.", request_id=request_id,
        schedule={"kind": "interval", "timezone": "UTC", "interval_minutes": 60},
    )
    await state.store.watches.run_now(user_id=state.owner.user_id, watch_id=watch["id"],
                                      expected_revision=watch["revision"])
    run = await state.store.watches.claim()
    await state.store.watches.fail(
        **{key: run[key] for key in ("run_id", "lease_token", "fence")},
        error="Public fetch did not finish.",
    )
    return watch, run


async def test_empty_feed_and_full_unread_count(state):
    assert await state.store.notifications.feed(user_id=state.owner.user_id) == {
        "items": [], "unread_count": 0,
    }
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 0
    state.store.notifications.fence_response(user_id=state.owner.user_id)


@pytest.mark.parametrize("limit", [0, 51, True, "5", None])
async def test_feed_limit_is_strict_and_bounded(state, limit):
    with pytest.raises(NotificationError):
        await state.store.notifications.feed(user_id=state.owner.user_id, limit=limit)


async def test_completion_notice_targets_exact_current_revision_without_report_copies(state):
    task = await _complete(state)
    feed = await state.store.notifications.feed(user_id=state.owner.user_id)
    assert feed["unread_count"] == 1 and len(feed["items"]) == 1
    notice = feed["items"][0]
    assert notice == {
        "id": notice["id"], "origin": "agent", "kind": "completed",
        "title": "Research public orchard releases.",
        "body": "The agent completed its task. Open it to view the saved result.",
        "created_at": state.clock().isoformat(), "read_at": None,
        "target": {"kind": "agent", "task_id": task["id"], "revision": task["revision"]},
    }
    assert "Sensitive" not in repr(dict(state.store._conn.execute(
        "SELECT * FROM app_agent_notices",
    ).fetchone()))
    state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])
    assert await state.store.notifications.get(
        user_id=state.owner.user_id, origin="agent", notice_id=notice["id"],
    ) == notice


async def test_failure_notice_uses_fixed_body_and_sanitized_bounded_title(state):
    await _create(state, objective="<script>\x00 personal prompt " + "long " * 100)
    task = await state.store.agents.claim()
    task = await state.store.agents.fail(**_lease(task), error="Private source error detail.")
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    assert notice["kind"] == "failure" and notice["target"]["revision"] == task["revision"]
    assert notice["body"] == "The agent could not finish its task. Open it to review the saved progress."
    assert "Private source" not in repr(notice) and len(notice["title"]) <= 100
    assert "\x00" not in notice["title"] and "<" not in notice["title"]


async def test_retry_pause_resume_and_cancel_do_not_create_activity(state):
    task = await _create(state)
    task = await state.store.agents.control(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"], action="pause",
    )
    task = await state.store.agents.control(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"], action="resume",
    )
    task = await state.store.agents.claim()
    await state.store.agents.fail(**_lease(task), error="Temporary failure.", retry=True)
    assert (await state.store.notifications.feed(user_id=state.owner.user_id))["items"] == []
    task = await state.store.agents.claim()
    task = await state.store.agents.control(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"], action="cancel",
    )
    assert task["status"] == "cancelled"
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 0


async def test_crash_recovery_and_terminal_delivery_deduplicate(state):
    await _create(state)
    first = await state.store.agents.claim(lease_seconds=1)
    state.clock.advance(2)
    second = await state.store.agents.claim()
    assert second["id"] == first["id"] and second["fence"] > first["fence"]
    with pytest.raises(AgentTaskConflictError):
        await state.store.agents.fail(**_lease(first), error="Stale worker failure.")
    await state.store.agents.fail(**_lease(second), error="Current worker failure.")
    assert await state.store.agents.claim() is None
    with state.store._lock, state.store._conn:
        record = dict(state.store._conn.execute("SELECT * FROM app_agent_tasks").fetchone())
        insert_agent_notice(state.store._conn, record, created_at=state.clock().isoformat())
        insert_agent_notice(state.store._conn, record, created_at=state.clock().isoformat())
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 1


async def test_claim_exhaustion_creates_one_terminal_failure_notice(state):
    task = await _create(state)
    await state.store.agents.claim(lease_seconds=1, max_attempts=1)
    state.clock.advance(2)
    assert await state.store.agents.claim(max_attempts=1) is None
    saved = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    assert saved["status"] == "failed"
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    assert notice["kind"] == "failure" and notice["target"]["revision"] == saved["revision"]
    assert await state.store.agents.claim(max_attempts=1) is None
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 1


@pytest.mark.parametrize("operation", ["complete", "fail", "exhaustion"])
async def test_notice_and_terminal_state_commit_or_rollback_together(state, operation):
    if operation == "complete":
        task = await _ready(state)
    else:
        await _create(state)
        task = await state.store.agents.claim(lease_seconds=1 if operation == "exhaustion" else 60)
    before = await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"])
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "CREATE TRIGGER reject_agent_notice BEFORE INSERT ON app_agent_notices "
            "BEGIN SELECT RAISE(ABORT,'simulated notice write failure'); END",
        )
    with pytest.raises(sqlite3.IntegrityError):
        if operation == "complete":
            await state.store.agents.complete(**_lease(task), result="Report.")
        elif operation == "fail":
            await state.store.agents.fail(**_lease(task), error="Failure.")
        else:
            state.clock.advance(2)
            await state.store.agents.claim(max_attempts=1)
    assert await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]) == before
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 0
    assert not state.store._conn.in_transaction


async def test_rerun_atomically_removes_previous_notice_and_fences_saved_response(state):
    task = await _complete(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"],
                                     expected_revision=task["revision"], action="rerun")
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_agent_notices").fetchone()[0] == 0
    assert await state.store.notifications.feed(user_id=state.owner.user_id) == {"items": [], "unread_count": 0}
    for operation in (state.store.notifications.get, state.store.notifications.mark_read):
        with pytest.raises(NotificationAccessError):
            await operation(user_id=state.owner.user_id, origin="agent", notice_id=notice["id"])
    with pytest.raises(NotificationAccessError):
        state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])


async def test_failed_rerun_preserves_notice_and_report(state):
    task = await _complete(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "CREATE TRIGGER reject_agent_update BEFORE UPDATE ON app_agent_tasks "
            "BEGIN SELECT RAISE(ABORT,'simulated task write failure'); END",
        )
    with pytest.raises(sqlite3.IntegrityError):
        await state.store.agents.control(user_id=state.owner.user_id, task_id=task["id"],
                                         expected_revision=task["revision"], action="rerun")
    assert (await state.store.notifications.feed(user_id=state.owner.user_id))["items"] == [notice]
    assert (await state.store.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["result"] == task["result"]


@pytest.mark.parametrize("change", ["revision", "status"])
async def test_feed_and_count_exclude_stale_canonical_agent_result(state, change):
    await _complete(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "UPDATE app_agent_tasks SET revision=revision+1" if change == "revision" else
            "UPDATE app_agent_tasks SET status='cancelled'",
        )
    assert await state.store.notifications.feed(user_id=state.owner.user_id) == {"items": [], "unread_count": 0}
    with pytest.raises(NotificationAccessError):
        state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])


async def test_unified_feed_order_and_full_count_include_both_origins(state):
    task = await _complete(state)
    state.clock.advance()
    watch, run = await _watch_failure(state)
    feed = await state.store.notifications.feed(user_id=state.owner.user_id, limit=1)
    assert len(feed["items"]) == 1 and feed["unread_count"] == 2
    watch_notice = feed["items"][0]
    assert watch_notice["origin"] == "watch" and watch_notice["kind"] == "failure"
    assert watch_notice["target"] == {"kind": "watch", "watch_id": watch["id"], "run_id": run["run_id"]}
    full = await state.store.notifications.feed(user_id=state.owner.user_id)
    assert full["items"][1]["target"]["task_id"] == task["id"]
    state.store.notifications.fence_response(user_id=state.owner.user_id, notices=full["items"])
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 2


async def test_unread_first_fifty_are_bounded_and_make_all_older_unread_accessible(state):
    tasks = []
    for number in range(52):
        await _create(state, request_id=f"failed-{number}")
        task = await state.store.agents.claim()
        await state.store.agents.fail(**_lease(task), error="Failed without a public report.")
        tasks.append(task["id"])
        state.clock.advance()
    feed = await state.store.notifications.feed(user_id=state.owner.user_id)
    assert len(feed["items"]) == 50 and feed["unread_count"] == 52
    assert [item["target"]["task_id"] for item in feed["items"]] == list(reversed(tasks[2:]))
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 52
    state.store.notifications.fence_response(user_id=state.owner.user_id, notices=feed["items"])
    for notice in feed["items"]:
        await state.store.notifications.mark_read(
            user_id=state.owner.user_id, origin=notice["origin"], notice_id=notice["id"],
        )
    next_feed = await state.store.notifications.feed(user_id=state.owner.user_id)
    assert len(next_feed["items"]) == 50 and next_feed["unread_count"] == 2
    assert [item["target"]["task_id"] for item in next_feed["items"][:2]] == list(reversed(tasks[:2]))
    assert all(item["read_at"] is not None for item in next_feed["items"][2:])
    for notice in next_feed["items"][:2]:
        await state.store.notifications.mark_read(
            user_id=state.owner.user_id, origin=notice["origin"], notice_id=notice["id"],
        )
    final_feed = await state.store.notifications.feed(user_id=state.owner.user_id)
    assert final_feed["unread_count"] == 0
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 0
    assert [item["target"]["task_id"] for item in final_feed["items"]] == list(reversed(tasks[2:]))


async def test_rerun_keeps_one_current_notice_for_same_agent_and_retained_plan(state):
    first = await _complete(state)
    old = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    await state.store.agents.control(user_id=state.owner.user_id, task_id=first["id"],
                                     expected_revision=first["revision"], action="rerun")
    task = await state.store.agents.claim()
    assert task["id"] == first["id"] and task["steps"][0] == first["steps"][0]
    for step_id in ("research", "review"):
        await state.store.agents.reserve_operation(**_lease(task), step_id=step_id)
        task = await state.store.agents.checkpoint(**_lease(task), step_id=step_id, result="New saved step.")
    current = await state.store.agents.complete(**_lease(task), result="Replaced report.")
    feed = await state.store.notifications.feed(user_id=state.owner.user_id)
    assert feed["unread_count"] == 1 and len(feed["items"]) == 1
    assert feed["items"][0]["id"] != old["id"]
    assert feed["items"][0]["target"] == {
        "kind": "agent", "task_id": first["id"], "revision": current["revision"],
    }
    with pytest.raises(NotificationAccessError):
        state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[old])


@pytest.mark.parametrize("origin", ["agent", "watch"])
async def test_read_is_idempotent_and_legacy_watch_receipt_is_shared(state, origin):
    if origin == "agent":
        await _complete(state)
    else:
        await _watch_failure(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    values = dict(user_id=state.owner.user_id, origin=origin, notice_id=notice["id"])
    first = await state.store.notifications.mark_read(**values)
    state.clock.advance()
    assert await state.store.notifications.mark_read(**values) == first
    assert first["read_at"] is not None
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 0
    # A concurrent acknowledgement cannot invalidate unchanged display content.
    state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])
    if origin == "watch":
        assert await state.store.watches.unread_count(user_id=state.owner.user_id) == 0
        assert (await state.store.watches.list_notices(user_id=state.owner.user_id))[0]["read_at"] == first["read_at"]


async def test_foreign_origin_and_missing_ids_cannot_read_or_acknowledge(state):
    await _complete(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    assert await state.store.notifications.feed(user_id=state.other.user_id) == {"items": [], "unread_count": 0}
    errors = []
    for operation in (state.store.notifications.get, state.store.notifications.mark_read):
        for owner, origin, notice_id in (
            (state.other.user_id, "agent", notice["id"]),
            (state.owner.user_id, "watch", notice["id"]),
            (state.owner.user_id, "agent", "missing"),
        ):
            with pytest.raises(NotificationAccessError) as error:
                await operation(user_id=owner, origin=origin, notice_id=notice_id)
            errors.append(str(error.value))
    assert len(set(errors)) == 1
    assert await state.store.notifications.unread_count(user_id=state.owner.user_id) == 1


@pytest.mark.parametrize("origin", ["file", "app_agent_notices", "", None])
async def test_unknown_origin_is_rejected(state, origin):
    with pytest.raises(NotificationError):
        await state.store.notifications.mark_read(user_id=state.owner.user_id, origin=origin, notice_id="n")


@pytest.mark.parametrize("fence", ["disabled", "deletion_requested"])
async def test_active_owner_fence_covers_all_reads_and_acknowledgements(state, fence):
    await _complete(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    with state.store._lock, state.store._conn:
        if fence == "disabled":
            state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        else:
            state.store._conn.execute(
                "INSERT INTO account_deletion_requests(user_id,purge_id,storage_namespace,requested_at) VALUES (?,?,?,?)",
                (state.owner.user_id, "purge", state.owner.storage_namespace, state.clock().isoformat()),
            )
    with pytest.raises(NotificationAccessError):
        await state.store.notifications.feed(user_id=state.owner.user_id)
    with pytest.raises(NotificationAccessError):
        await state.store.notifications.unread_count(user_id=state.owner.user_id)
    with pytest.raises(NotificationAccessError):
        await state.store.notifications.mark_read(user_id=state.owner.user_id, origin="agent", notice_id=notice["id"])
    with pytest.raises(NotificationAccessError):
        state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])


@pytest.mark.parametrize("operation", ["agent_delete", "watch_delete", "purge", "second_connection"])
async def test_deletion_erases_notice_and_fences_already_read_body(state, operation):
    if operation == "watch_delete":
        watch, _ = await _watch_failure(state)
    else:
        task = await _complete(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    if operation == "agent_delete":
        await state.store.agents.delete(user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"])
    elif operation == "watch_delete":
        await state.store.watches.delete(user_id=state.owner.user_id, watch_id=watch["id"], expected_revision=watch["revision"])
    elif operation == "second_connection":
        second = ApplicationStore(state.store.path)
        try:
            await second.agents.delete(user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"])
        finally:
            second.close()
    else:
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    assert await state.store.notifications.feed(user_id=state.owner.user_id) == {"items": [], "unread_count": 0}
    with pytest.raises(NotificationAccessError):
        state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])
    assert state.store._conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("field", ["body", "title"])
async def test_final_response_fence_rejects_changed_notice_body(state, field):
    await _complete(state)
    notice = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    with state.store._lock, state.store._conn:
        state.store._conn.execute(
            "UPDATE app_agent_notices SET body='Changed' WHERE notice_id=?" if field == "body" else
            "UPDATE app_agent_notices SET title='Changed' WHERE notice_id=?", (notice["id"],),
        )
    with pytest.raises(NotificationConflictError):
        state.store.notifications.fence_response(user_id=state.owner.user_id, notices=[notice])


async def test_migration_is_additive_preserves_terminal_task_and_watch_without_backfill(state):
    task = await _complete(state)
    watch, run = await _watch_failure(state)
    with state.store._lock, state.store._conn:
        state.store._conn.execute("DROP TABLE app_agent_notices")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version=30")
    upgraded = ApplicationStore(state.store.path)
    try:
        assert upgraded.schema_version == 30
        assert (await upgraded.agents.get(user_id=state.owner.user_id, task_id=task["id"]))["result"] == task["result"]
        assert (await upgraded.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=run["run_id"]))["status"] == "failed"
        feed = await upgraded.notifications.feed(user_id=state.owner.user_id)
        assert feed["unread_count"] == 1 and [item["origin"] for item in feed["items"]] == ["watch"]
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        upgraded.close()
