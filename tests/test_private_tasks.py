"""Durable Private work checkpoints preserve owner authority and lease fencing."""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import sqlite3
from types import SimpleNamespace

import pytest

from audrey.admin_cli import _online_application_backup
from audrey.app_state import ApplicationStore
from audrey.app_state.private_tasks import (
    PrivateTaskAccessError,
    PrivateTaskConflictError,
    PrivateTaskError,
)
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.user_data_visibility import (
    block_remote_personal_reads,
    unblock_remote_personal_reads,
)

PINS = {
    "model_digests": {"qwen3.8:latest": "a" * 64},
    "policy_digest": "b" * 64,
}


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 10, 21, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += dt.timedelta(seconds=seconds)


async def _owner(store, name="alice"):
    return await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com",
        display_name=name.title(), role="user", auth_method="test",
    )


async def _case(store, owner, *, project_id=None, files=None):
    conversation = await store.conversations.create(
        user_id=owner.user_id, private_required=True, project_id=project_id,
    )
    case = await store.tax_cases.create(
        user_id=owner.user_id, conversation_id=conversation.conversation_id,
        tax_year=2025, goal="Review only the selected synthetic worksheet",
        reference_release="rel_current", files=files,
    )
    return conversation, case


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _owner(store)
    project = await store.projects.create(
        user_id=owner.user_id, name="Tax review", private_required=True,
    )
    conversation, case = await _case(store, owner, project_id=project.project_id)
    clock = Clock()
    store.private_tasks._clock = clock
    yield SimpleNamespace(
        store=store, owner=owner, project=project, conversation=conversation,
        case=case, clock=clock,
    )
    store.close()


async def _create(state, **changes):
    kwargs = dict(
        user_id=state.owner.user_id,
        conversation_id=state.conversation.conversation_id,
        expected_case_revision=state.case["revision"], pins=copy.deepcopy(PINS),
    )
    kwargs.update(changes)
    return await state.store.private_tasks.create(**kwargs)


async def _claim(state, task, **changes):
    kwargs = dict(
        user_id=state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], pins=copy.deepcopy(PINS),
    )
    kwargs.update(changes)
    return await state.store.private_tasks.claim(**kwargs)


async def _checkpoint(state, task, **changes):
    kwargs = dict(
        user_id=state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], lease_token=task["lease_token"],
        fence=task["fence"], expected_case_revision=state.case["revision"],
        pins=copy.deepcopy(PINS), checkpoint_key="inspect-1",
        checkpoint={"stage": "inspect", "summary": "Selected case inspected."},
    )
    kwargs.update(changes)
    return await state.store.private_tasks.checkpoint(**kwargs)


async def _control(state, task, action, **changes):
    kwargs = dict(
        user_id=state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], action=action,
    )
    if action == "resume":
        kwargs.update(expected_case_revision=state.case["revision"], pins=copy.deepcopy(PINS))
    kwargs.update(changes)
    return await state.store.private_tasks.control(**kwargs)


async def _update_case(state, operation, data, *, actor="model"):
    state.case = await state.store.tax_cases.update(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        expected_revision=state.case["revision"], operation=operation, data=data, actor=actor,
    )
    return state.case


async def test_queued_task_reopens_without_automatic_execution(state):
    task = await _create(state)
    assert task["status"] == "queued"
    assert task["revision"] == 1 and task["fence"] == 0
    assert task["case_revision"] == state.case["revision"]
    assert task["checkpoint"] is None and task["checkpoint_sequence"] == 0
    assert not task["lease_token"] and task["lease_expires_at"] is None
    assert task["usage"] == {
        "steps": 0, "model_tokens": 0, "tool_calls": 0, "wall_seconds": 0.0,
    }
    reopened = ApplicationStore(state.store.path)
    try:
        assert await reopened.private_tasks.get(
            user_id=state.owner.user_id, task_id=task["id"],
        ) == task
        assert reopened._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close()


async def test_checkpoint_releases_lease_and_resumes_saved_work(state):
    task = await _claim(state, await _create(state))
    assert task["status"] == "running" and task["lease_token"]
    state.clock.advance(3)
    saved = await _checkpoint(state, task, usage={"model_tokens": 120, "tool_calls": 2})
    assert saved["status"] == "queued" and not saved["lease_token"]
    assert saved["lease_expires_at"] is None
    assert saved["checkpoint"]["stage"] == "inspect"
    assert saved["checkpoint_sequence"] == 1
    assert saved["usage"]["steps"] == 1
    assert saved["usage"]["model_tokens"] == 120
    assert saved["usage"]["tool_calls"] == 2
    assert saved["usage"]["wall_seconds"] == 3.0
    running = await _claim(state, saved)
    assert running["fence"] > task["fence"]
    assert running["lease_token"] != task["lease_token"]
    assert running["checkpoint"] == saved["checkpoint"]
    finished = await _checkpoint(
        state, running, checkpoint_key="review-1",
        checkpoint={"stage": "review", "summary": "The partial review is saved."},
        status="completed",
    )
    assert finished["status"] == "completed"
    assert finished["checkpoint_sequence"] == 2


@pytest.mark.parametrize("status", ["queued", "running", "waiting_for_user", "paused"])
async def test_only_one_nonterminal_task_per_conversation(state, status):
    task = await _create(state)
    if status == "running":
        await _claim(state, task)
    elif status == "paused":
        await _control(state, task, "pause")
    elif status == "waiting_for_user":
        await _checkpoint(state, await _claim(state, task), status=status)
    with pytest.raises(PrivateTaskConflictError):
        await _create(state)


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
async def test_terminal_task_allows_new_explicit_task(state, status):
    task = await _create(state)
    if status == "cancelled":
        terminal = await _control(state, task, "cancel")
    else:
        terminal = await _checkpoint(state, await _claim(state, task), status=status)
    replacement = await _create(state)
    assert replacement["id"] != terminal["id"]
    assert replacement["status"] == "queued"
    assert await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=terminal["id"],
    ) == terminal


async def test_separate_store_claim_race_grants_exactly_one_lease(state):
    task = await _create(state)
    other = ApplicationStore(state.store.path)
    other.private_tasks._clock = state.clock
    kwargs = dict(
        user_id=state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], pins=PINS,
    )
    try:
        results = await asyncio.gather(
            state.store.private_tasks.claim(**kwargs), other.private_tasks.claim(**kwargs),
            return_exceptions=True,
        )
    finally:
        other.close()
    winners = [result for result in results if isinstance(result, dict)]
    losers = [result for result in results if isinstance(result, Exception)]
    assert len(winners) == len(losers) == 1
    assert isinstance(losers[0], PrivateTaskConflictError)
    assert winners[0]["status"] == "running"


async def test_expired_lease_cannot_commit_and_explicit_recovery_only_pauses(state):
    running = await _claim(state, await _create(state), lease_seconds=10)
    state.clock.advance(11)
    with pytest.raises(PrivateTaskConflictError):
        await _checkpoint(state, running)
    reopened = ApplicationStore(state.store.path)
    reopened.private_tasks._clock = state.clock
    try:
        unchanged = await reopened.private_tasks.get(
            user_id=state.owner.user_id, task_id=running["id"],
        )
        assert unchanged["status"] == "running"
        assert await reopened.private_tasks.recover_expired() == 1
        assert await reopened.private_tasks.recover_expired() == 0
        paused = await reopened.private_tasks.get(
            user_id=state.owner.user_id, task_id=running["id"],
        )
        assert paused["status"] == "paused" and not paused["lease_token"]
        assert paused["stop_reason"] == "lease_expired"
        assert paused["fence"] > running["fence"]
    finally:
        reopened.close()
    resumed = await _control(state, paused, "resume")
    replacement = await _claim(state, resumed)
    assert replacement["fence"] > running["fence"]
    with pytest.raises(PrivateTaskConflictError):
        await _checkpoint(
            state, replacement, lease_token=running["lease_token"], fence=running["fence"],
        )


async def test_renew_requires_new_progress_and_keeps_fence(state):
    running = await _claim(state, await _create(state), lease_seconds=10)
    state.clock.advance(4)
    kwargs = dict(
        user_id=state.owner.user_id, task_id=running["id"],
        expected_revision=running["revision"], lease_token=running["lease_token"],
        fence=running["fence"], pins=PINS, progress=1, lease_seconds=20,
    )
    renewed = await state.store.private_tasks.renew(**kwargs)
    assert renewed["lease_expires_at"] > running["lease_expires_at"]
    assert renewed["lease_token"] == running["lease_token"]
    assert renewed["fence"] == running["fence"]
    kwargs["expected_revision"] = renewed["revision"]
    with pytest.raises(PrivateTaskError):
        await state.store.private_tasks.renew(**kwargs)


@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_owner_control_fences_out_inflight_writer(state, action):
    running = await _claim(state, await _create(state))
    stopped = await _control(state, running, action)
    assert stopped["status"] == {"pause": "paused", "cancel": "cancelled"}[action]
    assert not stopped["lease_token"] and stopped["lease_expires_at"] is None
    assert stopped["fence"] > running["fence"]
    with pytest.raises(PrivateTaskConflictError):
        await _checkpoint(state, running)
    with pytest.raises(PrivateTaskConflictError):
        await _claim(state, stopped)


async def test_lost_checkpoint_reply_replays_once_but_altered_payload_conflicts(state):
    running = await _claim(state, await _create(state))
    saved = await _checkpoint(state, running, usage={"model_tokens": 10, "tool_calls": 1})
    replayed = await _checkpoint(state, running, usage={"model_tokens": 10, "tool_calls": 1})
    assert replayed == saved
    assert replayed["usage"]["steps"] == 1
    with pytest.raises(PrivateTaskConflictError):
        await _checkpoint(
            state, running, checkpoint={"stage": "inspect", "summary": "Different content."},
            usage={"model_tokens": 10, "tool_calls": 1},
        )
    assert await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=running["id"],
    ) == saved


async def test_checkpoint_can_include_writers_new_case_fact(state):
    running = await _claim(state, await _create(state))
    case = await _update_case(state, "propose_fact", {
        "label": "Filing status", "value": "single", "kind": "user_statement",
    })
    saved = await _checkpoint(state, running, checkpoint={
        "stage": "facts", "summary": "A user statement remains proposed.",
        "fact_ids": [case["facts"][0]["id"]],
    })
    assert saved["case_revision"] == case["revision"]
    assert saved["checkpoint"]["fact_ids"] == [case["facts"][0]["id"]]
    assert case["facts"][0]["status"] == "proposed"


async def test_case_revision_changed_before_claim_requires_explicit_resume(state):
    task = await _create(state)
    await _update_case(state, "propose_fact", {
        "label": "Filing status", "value": "single", "kind": "user_statement",
    })
    with pytest.raises(PrivateTaskConflictError):
        await _claim(state, task)
    paused = await _control(state, task, "pause")
    refreshed = await _control(state, paused, "resume")
    assert refreshed["case_revision"] == state.case["revision"]
    assert (await _claim(state, refreshed))["status"] == "running"


@pytest.mark.parametrize("operation,data", [
    ("change_year", {"tax_year": 2024}),
    ("update_reference_release", {"reference_release": "rel_new"}),
])
async def test_changed_source_scope_blocks_inflight_checkpoint(state, operation, data):
    running = await _claim(state, await _create(state))
    await _update_case(state, operation, data, actor="user")
    with pytest.raises(PrivateTaskConflictError):
        await _checkpoint(state, running)


async def test_changed_model_or_policy_requires_explicit_resume_and_recheck(state):
    saved = await _checkpoint(state, await _claim(state, await _create(state)))
    changed = copy.deepcopy(PINS)
    changed["model_digests"]["qwen3.8:latest"] = "c" * 64
    changed["policy_digest"] = "d" * 64
    with pytest.raises(PrivateTaskConflictError):
        await _claim(state, saved, pins=changed)
    paused = await _control(state, saved, "pause")
    resumed = await _control(state, paused, "resume", pins=changed)
    assert resumed["checkpoint"] is None
    assert (await _claim(state, resumed, pins=changed))["status"] == "running"


async def test_project_detachment_blocks_old_binding_until_explicit_resume(state):
    running = await _claim(state, await _create(state))
    assert await state.store.projects.delete(
        user_id=state.owner.user_id, project_id=state.project.project_id,
    )
    with pytest.raises(PrivateTaskConflictError):
        await _checkpoint(state, running)
    paused = await _control(state, running, "pause")
    resumed = await _control(state, paused, "resume")
    assert (await _claim(state, resumed))["status"] == "running"


async def test_foreign_owner_cannot_read_claim_control_or_checkpoint(state):
    running = await _claim(state, await _create(state))
    other = await _owner(state.store, "bob")
    assert await state.store.private_tasks.get(user_id=other.user_id, task_id=running["id"]) is None
    with pytest.raises(PrivateTaskAccessError):
        await _create(state, user_id=other.user_id)
    with pytest.raises(PrivateTaskAccessError):
        await _claim(state, running, user_id=other.user_id)
    with pytest.raises(PrivateTaskAccessError):
        await _checkpoint(state, running, user_id=other.user_id)
    with pytest.raises(PrivateTaskAccessError):
        await _control(state, running, "cancel", user_id=other.user_id)


@pytest.mark.parametrize("restriction", ["disabled", "deleting", "archived", "nonprivate"])
async def test_context_restriction_blocks_further_work(state, restriction):
    running = await _claim(state, await _create(state))
    with state.store._lock:
        if restriction == "disabled":
            state.store._conn.execute(
                "UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,),
            )
        elif restriction == "deleting":
            state.store._conn.execute(
                "INSERT INTO account_deletion_requests "
                "(user_id,purge_id,storage_namespace,requested_at) VALUES (?,?,?,?)",
                (state.owner.user_id, "purge_test", state.owner.storage_namespace,
                 state.clock().isoformat()),
            )
        elif restriction == "archived":
            state.store._conn.execute(
                "UPDATE app_conversations SET archived_at=? WHERE conversation_id=?",
                (state.clock().isoformat(), state.conversation.conversation_id),
            )
        else:
            state.store._conn.execute("DROP TRIGGER trg_app_conversations_private_no_downgrade")
            state.store._conn.execute(
                "UPDATE app_conversations SET private_required=0 WHERE conversation_id=?",
                (state.conversation.conversation_id,),
            )
        state.store._conn.commit()
    with pytest.raises(PrivateTaskAccessError):
        await _checkpoint(state, running)
    with pytest.raises(PrivateTaskAccessError):
        await _claim(state, running)


async def test_ordinary_conversation_cannot_start_private_task(state):
    ordinary = await state.store.conversations.create(user_id=state.owner.user_id)
    with pytest.raises(PrivateTaskAccessError):
        await _create(state, conversation_id=ordinary.conversation_id)


@pytest.mark.parametrize("deletion", ["conversation", "purge", "account"])
async def test_owner_deletion_cascades_tasks_without_affecting_other_owner(state, deletion):
    task = await _claim(state, await _create(state))
    other = await _owner(state.store, "bob")
    other_conversation, other_case = await _case(state.store, other)
    other_task = await state.store.private_tasks.create(
        user_id=other.user_id, conversation_id=other_conversation.conversation_id,
        expected_case_revision=other_case["revision"], pins=PINS,
    )
    if deletion == "conversation":
        assert await state.store.conversations.delete(
            user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        )
    elif deletion == "purge":
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    else:
        with state.store._lock:
            state.store._conn.execute("DELETE FROM app_users WHERE user_id=?", (state.owner.user_id,))
            state.store._conn.commit()
    rows = state.store._conn.execute("SELECT task_id,user_id FROM app_private_tasks").fetchall()
    assert [tuple(row) for row in rows] == [(other_task["id"], other.user_id)]
    with pytest.raises(PrivateTaskAccessError):
        await _checkpoint(state, task)
    assert state.store._conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("changes", [
    {"limits": {"max_steps": 0}},
    {"limits": {"max_wall_seconds": -1}},
    {"limits": {"max_model_tokens": True}},
    {"limits": {"max_tool_calls": 0}},
    {"pins": {"model_digests": {}, "policy_digest": "b" * 64}},
    {"pins": {"model_digests": {"qwen3.8:latest": "not-a-digest"}, "policy_digest": "b" * 64}},
])
async def test_invalid_limits_and_pins_are_rejected_without_saved_task(state, changes):
    with pytest.raises(PrivateTaskError):
        await _create(state, **changes)
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_private_tasks").fetchone()[0] == 0


@pytest.mark.parametrize("checkpoint", [
    {"stage": "arbitrary-code", "summary": "No arbitrary stage."},
    {"stage": "facts", "summary": "x" * 2001},
    {"stage": "facts", "summary": "Unknown fact.", "fact_ids": ["tax_missing"]},
    {"stage": "facts", "summary": "No secret traces.", "chain_of_thought": "private trace"},
])
async def test_invalid_checkpoint_cannot_advance_or_spend_usage(state, checkpoint):
    running = await _claim(state, await _create(state))
    with pytest.raises(PrivateTaskError):
        await _checkpoint(state, running, checkpoint=checkpoint)
    assert await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=running["id"],
    ) == running


async def test_budget_exhaustion_preserves_last_checkpoint_and_rejects_more_work(state):
    running = await _claim(state, await _create(state, limits={"max_steps": 1}))
    saved = await _checkpoint(state, running)
    with pytest.raises(PrivateTaskError):
        await _claim(state, saved)
    retained = await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=running["id"],
    )
    assert retained["checkpoint"] == saved["checkpoint"]
    assert retained["usage"]["steps"] == 1


async def test_schema23_upgrade_preserves_case_and_creates_empty_task_repository(state):
    path = state.store.path
    case = state.case
    state.store.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE app_private_tasks")
        db.execute("DELETE FROM app_schema_migrations WHERE version=24")
    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 26
        assert await upgraded.tax_cases.get(
            user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        ) == case
        assert upgraded._conn.execute("SELECT COUNT(*) FROM app_private_tasks").fetchone()[0] == 0
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        upgraded.close()


async def test_pending_owner_purge_blocks_reads_and_mutations(state):
    running = await _claim(state, await _create(state))
    block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="purge_task_test")
    try:
        with pytest.raises(PrivateTaskAccessError):
            await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=running["id"])
        with pytest.raises(PrivateTaskAccessError):
            await _claim(state, running)
        with pytest.raises(PrivateTaskAccessError):
            await _checkpoint(state, running)
        with pytest.raises(PrivateTaskAccessError):
            await _control(state, running, "resume")
        with pytest.raises(PrivateTaskAccessError):
            await _create(state)
    finally:
        unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="purge_task_test")
    assert await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=running["id"],
    ) == running


async def test_claim_caps_lease_at_remaining_wall_budget(state):
    running = await _claim(
        state, await _create(state, limits={"max_wall_seconds": 8}), lease_seconds=120,
    )
    expiry = dt.datetime.fromisoformat(running["lease_expires_at"])
    assert (expiry - state.clock()).total_seconds() == 8
    state.clock.advance(5)
    renewed = await state.store.private_tasks.renew(
        user_id=state.owner.user_id, task_id=running["id"],
        expected_revision=running["revision"], lease_token=running["lease_token"],
        fence=running["fence"], pins=PINS, progress=1, lease_seconds=120,
    )
    assert dt.datetime.fromisoformat(renewed["lease_expires_at"]) == expiry
    state.clock.advance(3)
    with pytest.raises(PrivateTaskConflictError):
        await _checkpoint(state, renewed)
    assert await state.store.private_tasks.recover_expired() == 1
    paused = await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=running["id"],
    )
    assert paused["usage"]["wall_seconds"] == 8
    with pytest.raises(PrivateTaskConflictError):
        await _control(state, paused, "resume")


@pytest.mark.parametrize("usage", [{"model_tokens": 11}, {"tool_calls": 3}])
async def test_over_budget_results_remain_partial_even_if_worker_claims_completion(state, usage):
    running = await _claim(state, await _create(
        state, limits={"max_model_tokens": 10, "max_tool_calls": 2},
    ))
    saved = await _checkpoint(state, running, usage=usage, status="completed")
    assert saved["status"] == "paused"
    assert saved["stop_reason"] == "budget_exhausted"
    assert saved["checkpoint"]["summary"] == "Selected case inspected."
    with pytest.raises(PrivateTaskConflictError):
        await _control(state, saved, "resume")


async def test_checkpoint_cannot_reference_another_owners_case_fact(state):
    other = await _owner(state.store, "bob")
    conversation, case = await _case(state.store, other)
    case = await state.store.tax_cases.update(
        user_id=other.user_id, conversation_id=conversation.conversation_id,
        expected_revision=case["revision"], operation="propose_fact",
        data={"label": "Filing status", "value": "single", "kind": "user_statement"},
    )
    running = await _claim(state, await _create(state))
    with pytest.raises(PrivateTaskError):
        await _checkpoint(state, running, checkpoint={
            "stage": "facts", "summary": "This must not borrow another case's evidence.",
            "fact_ids": [case["facts"][0]["id"]],
        })
    assert await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=running["id"],
    ) == running


async def test_online_backup_restores_exact_task_and_checkpoint_without_source_changes(state, tmp_path):
    running = await _claim(state, await _create(state))
    state.clock.advance(2)
    saved = await _checkpoint(state, running, usage={"model_tokens": 125, "tool_calls": 1})
    source = state.store.path
    source_files = [source, source.with_name(source.name + "-wal")]
    before = {path: path.read_bytes() for path in source_files if path.exists()}
    restored_path = tmp_path / "private-task-restored.sqlite"
    _online_application_backup(source, restored_path)
    assert restored_path.stat().st_mode & 0o777 == 0o600
    restored = ApplicationStore(restored_path)
    try:
        assert restored.schema_version == 26
        assert await restored.private_tasks.get(
            user_id=state.owner.user_id, task_id=saved["id"],
        ) == saved
        assert await restored.tax_cases.get(
            user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        ) == state.case
        assert restored._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        restored.close()
    assert {path: path.read_bytes() for path in before} == before
    assert await state.store.private_tasks.get(
        user_id=state.owner.user_id, task_id=saved["id"],
    ) == saved


@pytest.mark.parametrize("transition", ["claim", "renew", "checkpoint"])
async def test_project_content_changes_require_recheck_but_rename_does_not(state, transition):
    saved = await _checkpoint(state, await _claim(state, await _create(state)))
    assert await state.store.projects.update(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        name="Renamed tax worksheet",
    )
    # Display-only changes must not invalidate the semantic task bindings.
    task = saved if transition == "claim" else await _claim(state, saved)
    assert await state.store.projects.update(
        user_id=state.owner.user_id, project_id=state.project.project_id,
        instructions="Use only this project's selected synthetic wage statement.",
    )
    retained_case = await state.store.tax_cases.get(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
    )
    assert retained_case["revision"] == state.case["revision"]
    with pytest.raises(PrivateTaskConflictError):
        if transition == "claim":
            await _claim(state, task)
        elif transition == "renew":
            await state.store.private_tasks.renew(
                user_id=state.owner.user_id, task_id=task["id"],
                expected_revision=task["revision"], lease_token=task["lease_token"],
                fence=task["fence"], pins=PINS, progress=1,
            )
        else:
            await _checkpoint(state, task, checkpoint_key="inspect-project-change")
    paused = await _control(state, task, "pause")
    resumed = await _control(state, paused, "resume")
    assert resumed["checkpoint"] is None
    assert resumed["usage"] == paused["usage"]
    assert (await _claim(state, resumed))["status"] == "running"


async def test_cancel_lost_reply_replays_original_request_without_revision_bump(state):
    running = await _claim(state, await _create(state))
    cancelled = await _control(state, running, "cancel")
    replayed = await _control(state, running, "cancel")
    assert replayed == cancelled
    assert replayed["status"] == "cancelled"
    assert replayed["revision"] == running["revision"] + 1


async def test_clock_backstep_cannot_double_charge_wall_time_or_contract_lease(state):
    running = await _claim(state, await _create(state))

    async def renew(task, progress):
        return await state.store.private_tasks.renew(
            user_id=state.owner.user_id, task_id=task["id"],
            expected_revision=task["revision"], lease_token=task["lease_token"],
            fence=task["fence"], pins=PINS, progress=progress,
        )

    state.clock.advance(4)
    first = await renew(running, 1)
    assert first["usage"]["wall_seconds"] == 4
    state.clock.advance(-2)
    backward = await renew(first, 2)
    assert backward["usage"]["wall_seconds"] == 4
    assert backward["lease_accounted_at"] == first["lease_accounted_at"]
    assert backward["lease_expires_at"] >= first["lease_expires_at"]
    state.clock.advance(3)
    forward = await renew(backward, 3)
    assert forward["usage"]["wall_seconds"] == 5


async def test_project_file_membership_change_invalidates_saved_checkpoint(state):
    saved = await _checkpoint(state, await _claim(state, await _create(state)))
    policy = ProcessingPolicy(local_models=PINS["model_digests"], ollama_base_url="http://ollama:11434")
    with processing_scope(policy):
        await state.store.projects.add_file(
            user_id=state.owner.user_id,
            project_id=state.project.project_id,
            file_id="new-project-document",
            file_is_private=True,
        )
    with pytest.raises(PrivateTaskConflictError):
        await _claim(state, saved)
    paused = await _control(state, saved, "pause")
    resumed = await _control(state, paused, "resume")
    assert resumed["checkpoint"] is None
    assert resumed["usage"] == saved["usage"]
    assert (await _claim(state, resumed))["status"] == "running"
