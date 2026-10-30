"""Durable watch ownership, scheduling, atomic baselines and notices."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.watches import (
    WatchAccessError,
    WatchConflictError,
    WatchError,
    WatchLimitError,
)


class Clock:
    now = dt.datetime(2026, 10, 30, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += dt.timedelta(seconds=seconds)


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    users = []
    for name in ("alice", "bob"):
        users.append(await store.resolve_external_identity(
            provider="test", subject=name, email=f"{name}@example.com", display_name=name,
            role="user", auth_method="test",
        ))
    clock = Clock()
    store.watches._clock = clock
    store.agents._clock = clock
    yield SimpleNamespace(store=store, owner=users[0], other=users[1], clock=clock)
    store.close()


async def _create(state, **changes):
    values = dict(user_id=state.owner.user_id, title="Release notes", url="https://example.com/releases",
                  criterion="Tell me about new stable releases.", request_id="watch-1",
                  schedule={"kind": "interval", "timezone": "UTC", "interval_minutes": 60})
    values.update(changes)
    return await state.store.watches.create(**values)


def _key(state, watch, **changes):
    values = dict(user_id=state.owner.user_id, watch_id=watch["id"], expected_revision=watch["revision"])
    values.update(changes)
    return values


def _lease(run):
    return {key: run[key] for key in ("run_id", "lease_token", "fence")}


async def _claim(state, watch):
    await state.store.watches.run_now(**_key(state, watch))
    return await state.store.watches.claim()


async def _complete(state, run, text="Version one.", outcome="baseline", significant=None, **changes):
    await state.store.watches.reserve_operation(**_lease(run), kind="fetch")
    if outcome == "changed":
        await state.store.watches.reserve_operation(**_lease(run), kind="model")
    values = dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest(), outcome=outcome,
                  significant=significant, report="Saved comparison.", diff='[{"old":"one","new":"two"}]' if outcome == "changed" else "", final_url=run["url"],
                  extraction_version="public-text-v1")
    values.update(changes)
    return await state.store.watches.complete(**_lease(run), **values)


async def test_create_does_not_dispatch_idempotency_survives_edit_and_reopen(state):
    watch = await _create(state)
    assert watch["next_run_at"] == "2026-10-30T13:00:00+00:00"
    assert watch["latest_run"] is None and watch["active_run_id"] is None
    assert await state.store.watches.claim() is None
    edited = await state.store.watches.update(**_key(state, watch), title="Stable releases")
    assert (await _create(state))["title"] == edited["title"]
    with pytest.raises(WatchConflictError):
        await _create(state, criterion="Other criterion")
    second = ApplicationStore(state.store.path)
    assert (await second.watches.get(user_id=state.owner.user_id, watch_id=watch["id"]))["title"] == "Stable releases"
    second.close()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "https://user:pass@example.com", "https://example.com:8443/a",
    "https://example.com/a?key=value", "https://example.com/a?", "https://example.com/#part",
    "https://127.0.0.1", "http://10.0.0.1", "http://[::1]", "http://169.254.169.254/",
    "http://[::ffff:127.0.0.1]", "http://localhost/", "http://example.com\\@localhost/",
])
async def test_unsafe_urls_rejected_without_network(state, url):
    with pytest.raises(WatchError):
        await _create(state, url=url)


@pytest.mark.parametrize("fence", ["disabled", "deletion_requested"])
async def test_ownership_and_active_account_deletion_fence_every_path(state, fence):
    watch = await _create(state)
    for user_id in (state.other.user_id, "missing"):
        with pytest.raises(WatchAccessError):
            await state.store.watches.get(user_id=user_id, watch_id=watch["id"])
        with pytest.raises(WatchAccessError):
            await state.store.watches.run_now(**_key(state, watch, user_id=user_id))
    run = await _claim(state, watch)
    with state.store._lock:
        if fence == "disabled":
            state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        else:
            state.store._conn.execute(
                "INSERT INTO account_deletion_requests(user_id,purge_id,storage_namespace,requested_at) "
                "VALUES (?,?,?,?)", (state.owner.user_id, "purge", state.owner.storage_namespace,
                                      state.clock().isoformat()),
            )
        state.store._conn.commit()
    with pytest.raises(WatchAccessError):
        await state.store.watches.validate_lease(**_lease(run))
    with pytest.raises(WatchAccessError):
        await state.store.watches.list(user_id=state.owner.user_id)
    assert await state.store.watches.claim() is None


async def test_definition_edit_fences_live_work_and_preserves_only_compatible_baseline(state):
    watch = await _create(state)
    baseline = await _claim(state, watch)
    await _complete(state, baseline)
    running = await _claim(state, watch)
    edited = await state.store.watches.update(**_key(state, watch), criterion="Security releases only")
    with pytest.raises(WatchConflictError):
        await _complete(state, running)
    assert (await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=running["id"]))["status"] == "cancelled"
    changed_criterion = await _claim(state, edited)
    assert changed_criterion["baseline"]["run_id"] == baseline["id"]
    edited = await state.store.watches.update(**_key(state, edited), url="https://example.com/other")
    assert (await _claim(state, edited))["baseline"] is None


async def test_downtime_coalesces_one_occurrence_and_duplicate_workers_do_not_overlap(state):
    watch = await _create(state)
    state.clock.advance(5 * 3600 + 120)
    claims = await asyncio.gather(state.store.watches.claim(), state.store.watches.claim())
    run = next(item for item in claims if item)
    assert sum(item is not None for item in claims) == 1
    assert run["kind"] == "catchup" and run["scheduled_at"] == "2026-10-30T13:00:00+00:00"
    assert (await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"]))["next_run_at"] == "2026-10-30T18:00:00+00:00"
    assert len(await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])) == 1
    await state.store.watches.run_now(**_key(state, watch))
    assert len(await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])) == 1


async def test_manual_paused_run_does_not_resume_schedule_and_concurrent_edit_conflicts(state):
    watch = await _create(state, enabled=False)
    assert watch["next_run_at"] is None
    run = await _claim(state, watch)
    assert run["kind"] == "manual"
    await _complete(state, run)
    second = ApplicationStore(state.store.path)
    results = await asyncio.gather(
        state.store.watches.update(**_key(state, watch), title="One"),
        second.watches.update(**_key(state, watch), title="Two"), return_exceptions=True,
    )
    assert sum(isinstance(item, WatchConflictError) for item in results) == 1
    current = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    assert current["enabled"] is False and current["next_run_at"] is None
    second.close()


async def test_research_and_watch_claims_serialize_per_owner_but_allow_other_owner(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    task = await state.store.agents.create(user_id=state.owner.user_id, objective="Research", request_id="agent-1")
    assert await state.store.agents.claim() is None
    other = await state.store.agents.create(user_id=state.other.user_id, objective="Research", request_id="agent-1")
    assert (await state.store.agents.claim())["id"] == other["id"]
    await _complete(state, run)
    claimed = await state.store.agents.claim()
    assert claimed["id"] == task["id"]
    await state.store.watches.run_now(**_key(state, watch))
    assert await state.store.watches.claim() is None


async def test_expired_lease_and_bounded_retries_charge_uncertain_work(state):
    watch = await _create(state)
    first = await _claim(state, watch)
    await state.store.watches.reserve_operation(**_lease(first), kind="fetch")
    state.clock.advance(61)
    recovered = await state.store.watches.claim()
    assert recovered["id"] == first["id"] and recovered["operations"] == 1
    assert recovered["fence"] > first["fence"] and recovered["active_seconds"] == 60
    with pytest.raises(WatchConflictError):
        await state.store.watches.heartbeat(**_lease(first))
    await state.store.watches.fail(**_lease(recovered), error="Temporary network failure", retry=True)
    assert await state.store.watches.claim() is None
    state.clock.advance(120)
    final = await state.store.watches.claim()
    await state.store.watches.fail(**_lease(final), error="Temporary network failure", retry=True)
    assert await state.store.watches.claim() is None
    notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    assert len(notices) == 1 and notices[0]["kind"] == "failure"


async def test_baselines_changed_results_notices_and_failure_recovery_are_atomic(state):
    watch = await _create(state)
    initial = await _claim(state, watch)
    assert (await _complete(state, initial))["outcome"] == "baseline"
    unchanged = await _claim(state, watch)
    await _complete(state, unchanged, outcome="unchanged")
    assert not await state.store.watches.list_notices(user_id=state.owner.user_id)
    changed = await _claim(state, watch)
    await _complete(state, changed, text="Version two.", outcome="changed", significant=True)
    failure = await _claim(state, watch)
    await state.store.watches.fail(**_lease(failure), error="Page unavailable")
    failure_again = await _claim(state, watch)
    assert failure_again["baseline"]["text"] == "Version two."
    await state.store.watches.fail(**_lease(failure_again), error="Still unavailable")
    notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    assert sorted(item["kind"] for item in notices) == ["change", "failure"]
    assert await state.store.watches.unread_count(user_id=state.owner.user_id) == 2
    assert await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=notices[0]["id"])
    assert await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=notices[0]["id"])
    assert await state.store.watches.unread_count(user_id=state.owner.user_id) == 1
    with pytest.raises(WatchAccessError):
        await state.store.watches.mark_notice_read(user_id=state.other.user_id, notice_id=notices[0]["id"])
    recovery = await _claim(state, watch)
    await _complete(state, recovery, text="Version two.", outcome="unchanged")
    third_failure = await _claim(state, watch)
    await state.store.watches.fail(**_lease(third_failure), error="Unavailable again")
    assert len(await state.store.watches.list_notices(user_id=state.owner.user_id)) == 3


async def test_commit_failure_rolls_back_snapshot_baseline_and_notice(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    await _complete(state, run)
    changed = await _claim(state, watch)
    with state.store._lock:
        state.store._conn.execute(
            "CREATE TRIGGER break_watch_notice BEFORE INSERT ON app_watch_notices "
            "BEGIN SELECT RAISE(ABORT,'receipt failed'); END",
        )
    with pytest.raises(sqlite3.IntegrityError, match="receipt failed"):
        await _complete(state, changed, text="Version two.", outcome="changed", significant=True)
    assert not state.store._conn.in_transaction
    assert state.store.watches._baseline(watch["id"])["text"] == "Version one."
    assert (await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=changed["id"]))["status"] == "running"
    assert not await state.store.watches.list_notices(user_id=state.owner.user_id)


async def test_baseline_identity_version_changes_rebaseline_without_notice(state):
    watch = await _create(state)
    await _complete(state, await _claim(state, watch))
    new_version = await _claim(state, watch)
    await _complete(state, new_version, text="New normalized text", extraction_version="public-text-v2")
    new_redirect = await _claim(state, watch)
    await _complete(state, new_redirect, text="Another text", final_url="https://example.com/new", extraction_version="public-text-v2")
    assert not await state.store.watches.list_notices(user_id=state.owner.user_id)


@pytest.mark.parametrize("erase", ["watch", "purge"])
async def test_deletion_cascades_all_new_derivatives_and_fences_dispatch(state, erase):
    watch = await _create(state)
    await _complete(state, await _claim(state, watch))
    failed = await _claim(state, watch)
    await state.store.watches.fail(**_lease(failed), error="Failure")
    live = await _claim(state, watch)
    if erase == "watch":
        await state.store.watches.delete(**_key(state, watch))
    else:
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    with pytest.raises(WatchAccessError):
        await state.store.watches.validate_lease(**_lease(live))
    for table in ("app_watches", "app_watch_runs", "app_watch_snapshots", "app_watch_baselines",
                  "app_watch_operations", "app_watch_notices"):
        assert state.store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # noqa: S608


async def test_snapshots_are_immutable_and_history_retention_preserves_current_baseline(state):
    watch = await _create(state)
    initial = await _claim(state, watch)
    await _complete(state, initial)
    with state.store._lock:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            state.store._conn.execute("UPDATE app_watch_snapshots SET record_json='{}'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            state.store._conn.execute("UPDATE app_watch_runs SET definition_json='{}'")
        state.store._conn.rollback()
    for _ in range(52):
        run = await _claim(state, watch)
        await state.store.watches.fail(**_lease(run), error="Unavailable")
    assert len(await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])) == 50
    assert state.store.watches._baseline(watch["id"])["run_id"] == initial["id"]
    notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    assert len(notices) == 1
    assert (await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"],
                                            run_id=notices[0]["run_id"]))["status"] == "failed"
    next_run = await _claim(state, watch)
    assert next_run["baseline"]["text"] == "Version one."


async def test_current_server_limits_clamp_recovered_work_and_create_caps(state):
    watch = await _create(state, max_per_user=1)
    with pytest.raises(WatchLimitError):
        await _create(state, request_id="second", max_per_user=1)
    run = await _claim(state, watch)
    await state.store.watches.reserve_operation(**_lease(run), kind="fetch")
    state.clock.advance(61)
    assert await state.store.watches.claim(limits={"max_operations": 1, "max_active_seconds": 300}) is None
    assert (await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"]))["latest_run"]["status"] == "failed"


async def test_failed_or_unreserved_results_cannot_advance_baseline(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    kwargs = dict(text="Version one.", sha256=hashlib.sha256(b"Version one.").hexdigest(),
                  outcome="baseline", significant=None, report="", diff="", final_url=run["url"],
                  extraction_version="v1")
    with pytest.raises(WatchConflictError):
        await state.store.watches.complete(**_lease(run), **kwargs)
    await state.store.watches.reserve_operation(**_lease(run), kind="fetch")
    with pytest.raises(WatchError):
        await state.store.watches.complete(**_lease(run), **{**kwargs, "sha256": "0" * 64})
    with pytest.raises(WatchError):
        await state.store.watches.complete(**_lease(run), **{**kwargs, "text": "é" * 40000})
    assert state.store.watches._baseline(watch["id"]) is None


async def test_notification_feed_reads_owned_bodies_and_total_in_one_snapshot(state):
    assert await state.store.watches.notification_feed(user_id=state.owner.user_id) == {
        "items": [], "unread_count": 0,
    }
    watch = await _create(state)
    await _complete(state, await _claim(state, watch))
    await _complete(state, await _claim(state, watch), text="Version two.", outcome="changed", significant=True)
    await state.store.watches.fail(**_lease(await _claim(state, watch)), error="Page unavailable")
    feed = await state.store.watches.notification_feed(user_id=state.owner.user_id, limit=1)
    assert len(feed["items"]) == 1 and feed["unread_count"] == 2
    await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=feed["items"][0]["id"])
    assert (await state.store.watches.notification_feed(user_id=state.owner.user_id))["unread_count"] == 1
    assert await state.store.watches.notification_feed(user_id=state.other.user_id) == {
        "items": [], "unread_count": 0,
    }
    await state.store.watches.delete(**_key(state, watch))
    assert await state.store.watches.notification_feed(user_id=state.owner.user_id) == {
        "items": [], "unread_count": 0,
    }


async def test_response_fence_rejects_deleted_or_revised_payloads_after_async_read(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    await _complete(state, run)
    failed = await _claim(state, watch)
    await state.store.watches.fail(**_lease(failed), error="Unavailable")
    saved_watch = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    saved_runs = await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])
    saved_notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    state.store.watches.fence_response(user_id=state.owner.user_id, watches=[saved_watch],
                                      runs=saved_runs, notices=saved_notices)
    # Read-state changes cannot revoke the immutable notice body.
    await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=saved_notices[0]["id"])
    state.store.watches.fence_response(user_id=state.owner.user_id, notices=saved_notices)
    updated = await state.store.watches.update(**_key(state, watch), title="Edited")
    with pytest.raises(WatchConflictError):
        state.store.watches.fence_response(user_id=state.owner.user_id, watches=[saved_watch])
    # Old run definitions remain legitimate history after a definition edit.
    state.store.watches.fence_response(user_id=state.owner.user_id, runs=saved_runs, notices=saved_notices)
    await state.store.watches.delete(**_key(state, updated))
    for kind, payload in (("watches", [saved_watch]), ("runs", saved_runs), ("notices", saved_notices)):
        with pytest.raises(WatchAccessError):
            state.store.watches.fence_response(user_id=state.owner.user_id, **{kind: payload})


async def test_response_fence_checks_purge_owner_and_live_run_change(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    saved = await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=run["id"])
    state.clock.advance(1)
    await state.store.watches.heartbeat(**_lease(run))
    with pytest.raises(WatchConflictError):
        state.store.watches.fence_response(user_id=state.owner.user_id, runs=[saved])
    await state.store.purge_local_user_data(user_id=state.owner.user_id)
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_response(user_id=state.owner.user_id, watches=[watch])
    with state.store._lock:
        state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        state.store._conn.commit()
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_response(user_id=state.owner.user_id)


async def test_run_history_retains_the_definition_that_produced_its_report(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    await _complete(state, run)
    await state.store.watches.update(**_key(state, watch), title="Other page", url="https://example.com/other",
                                     criterion="Other criterion")
    saved = await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=run["id"])
    assert saved["definition"] == {key: watch[key] for key in ("title", "url", "criterion", "schedule")}
    assert "text" not in saved and "lease_token" not in saved
