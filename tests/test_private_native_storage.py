"""Native Private result delivery is durable, owned and replay safe."""

from __future__ import annotations

import copy
import datetime as dt
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.private_tasks import (
    PrivateTaskAccessError,
    PrivateTaskConflictError,
    PrivateTaskError,
)
from audrey.app_state.records import ModelUsageSnapshot, SourceSnapshot, ToolCallSnapshot
from audrey.processing_policy import ProcessingPolicy, processing_scope

PINS = {"model_digests": {"qwen3.8:latest": "a" * 64}, "policy_digest": "b" * 64}
POLICY = ProcessingPolicy(PINS["model_digests"], "http://ollama:11434")
DELIVERY = {
    "assistant_content": "The selected worksheet needs your confirmation.",
    "outcome": "succeeded", "finish_reason": "stop", "virtual_model": "audrey_private",
    "concrete_model": "qwen3.8:latest", "prompt_tokens": 100, "completion_tokens": 20,
    "sources": [asdict(SourceSnapshot("source-1", "Official page", "https://www.irs.gov/test"))],
    "tool_calls": [asdict(ToolCallSnapshot(
        "tool-1", "tax_review", "succeeded", {"operation": "inspect"}, None, "",
    ))],
    "models": [asdict(ModelUsageSnapshot("qwen3.8:latest", 1))],
}


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 10, 22, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += dt.timedelta(seconds=seconds)


async def owner(store, name):
    return await store.resolve_external_identity(
        provider="test", subject=name, email=f"{name}@example.com",
        display_name=name.title(), role="user", auth_method="test",
    )


async def case(store, user):
    conversation = await store.conversations.create(user_id=user.user_id, private_required=True)
    saved = await store.tax_cases.create(
        user_id=user.user_id, conversation_id=conversation.conversation_id,
        tax_year=2025, goal="Review a synthetic worksheet", reference_release="release-current",
    )
    return conversation, saved


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    user = await owner(store, "alice")
    conversation, saved = await case(store, user)
    clock = Clock()
    store.private_tasks._clock = clock
    yield SimpleNamespace(store=store, owner=user, conversation=conversation, case=saved, clock=clock)
    store.close()


async def native_run(state, **changes):
    kwargs = dict(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
        user_content="Review my selected worksheet", model_id="private", mode="local",
        skill_id="tax-review", skill_version=1, skill_reason="explicit",
        expected_private_required=True,
    )
    kwargs.update(changes)
    with processing_scope(POLICY):
        return await state.store.conversations.begin_run(**kwargs)


async def create(state, **changes):
    kwargs = dict(user_id=state.owner.user_id,
                  conversation_id=state.conversation.conversation_id,
                  expected_case_revision=state.case["revision"], pins=PINS)
    kwargs.update(changes)
    return await state.store.private_tasks.create(**kwargs)


async def claim(state, task):
    return await state.store.private_tasks.claim(
        user_id=state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], pins=PINS, lease_seconds=20,
    )


async def checkpoint(state, task, **changes):
    kwargs = dict(
        user_id=state.owner.user_id, task_id=task["id"], expected_revision=task["revision"],
        lease_token=task["lease_token"], fence=task["fence"], pins=PINS,
        expected_case_revision=state.case["revision"], checkpoint_key="result-1",
        checkpoint={"stage": "review", "summary": "Worksheet review needs confirmation."},
        status="waiting_for_user", usage={"model_tokens": 120, "tool_calls": 1},
        delivery=copy.deepcopy(DELIVERY),
    )
    kwargs.update(changes)
    return await state.store.private_tasks.checkpoint(**kwargs)


async def control(state, task, action):
    kwargs = dict(user_id=state.owner.user_id, task_id=task["id"],
                  expected_revision=task["revision"], action=action)
    if action == "resume":
        kwargs.update(expected_case_revision=state.case["revision"], pins=PINS)
    return await state.store.private_tasks.control(**kwargs)


async def test_native_binding_is_owned_and_readable_only_by_its_owner(state):
    started = await native_run(state)
    task = await create(state, native_run_id=started.run.run_id)
    assert task["native_run_id"] == started.run.run_id
    assert await state.store.private_tasks.get_for_run(
        user_id=state.owner.user_id, run_id=started.run.run_id,
    ) == task
    assert await state.store.private_tasks.latest_for_conversation(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
    ) == task
    other = await owner(state.store, "bob")
    assert await state.store.private_tasks.get_for_run(
        user_id=other.user_id, run_id=started.run.run_id,
    ) is None
    with pytest.raises(PrivateTaskAccessError):
        await state.store.private_tasks.latest_for_conversation(
            user_id=other.user_id, conversation_id=state.conversation.conversation_id,
        )


@pytest.mark.parametrize("invalid", ["missing", "foreign", "other_conversation", "skill", "model", "terminal"])
async def test_create_refuses_an_unrelated_or_finished_native_run(state, invalid):
    changes = {}
    if invalid == "foreign":
        other = await owner(state.store, "bob")
        conversation, _ = await case(state.store, other)
        changes.update(user_id=other.user_id, conversation_id=conversation.conversation_id)
    if invalid == "other_conversation":
        conversation, _ = await case(state.store, state.owner)
        changes["conversation_id"] = conversation.conversation_id
    if invalid == "skill":
        changes["skill_id"] = ""
    if invalid == "model":
        changes["model_id"] = "auto"
    started = await native_run(state, **changes)
    if invalid == "terminal":
        await state.store.conversations.finish_run(
            user_id=state.owner.user_id, run_id=started.run.run_id,
            assistant_content="Stopped", outcome="cancelled",
        )
    run_id = "run_missing" if invalid == "missing" else started.run.run_id
    with pytest.raises(PrivateTaskError):
        await create(state, native_run_id=run_id)
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_private_tasks").fetchone()[0] == 0


async def test_bind_requires_current_queued_revision_and_one_attempt(state):
    started = await native_run(state)
    task = await create(state)
    with pytest.raises(PrivateTaskConflictError):
        await state.store.private_tasks.bind_native_run(
            user_id=state.owner.user_id, task_id=task["id"],
            expected_revision=task["revision"] + 1, run_id=started.run.run_id,
        )
    bound = await state.store.private_tasks.bind_native_run(
        user_id=state.owner.user_id, task_id=task["id"],
        expected_revision=task["revision"], run_id=started.run.run_id,
    )
    assert bound["revision"] == task["revision"] + 1
    assert await state.store.private_tasks.bind_native_run(
        user_id=state.owner.user_id, task_id=bound["id"],
        expected_revision=bound["revision"], run_id=started.run.run_id,
    ) == bound
    running = await claim(state, bound)
    with pytest.raises(PrivateTaskConflictError):
        await state.store.private_tasks.bind_native_run(
            user_id=state.owner.user_id, task_id=running["id"],
            expected_revision=running["revision"], run_id=started.run.run_id,
        )
    cancelled = await control(state, running, "cancel")
    with pytest.raises(PrivateTaskConflictError):
        await create(state, native_run_id=started.run.run_id)
    assert cancelled["native_run_id"] == started.run.run_id


async def test_checkpoint_and_native_outbox_commit_once_and_survive_reopen(state):
    started = await native_run(state)
    running = await claim(state, await create(state, native_run_id=started.run.run_id))
    committed = await checkpoint(state, running)
    outbox = committed["delivery"]
    assert outbox["pending"] and outbox["run_id"] == started.run.run_id
    assert outbox["checkpoint_sequence"] == committed["checkpoint_sequence"] == 1
    assert outbox["payload"]["tool_calls"][0]["result"] is None
    assert committed["usage"]["model_tokens"] == 120
    assert await checkpoint(state, running) == committed
    reopened = ApplicationStore(state.store.path)
    try:
        assert await reopened.private_tasks.pending_deliveries() == [committed]
    finally:
        reopened.close()
    with pytest.raises(PrivateTaskConflictError):
        await checkpoint(state, running, delivery={**DELIVERY, "assistant_content": "Changed"})
    with pytest.raises(PrivateTaskConflictError):
        await control(state, committed, "resume")


@pytest.mark.parametrize("change", [
    {"unknown": "discard this?"}, {"prompt_tokens": True}, {"outcome": "running"},
    {"assistant_content": "x" * 64001},
    {"sources": [DELIVERY["sources"][0]] * 51},
    {"tool_calls": [{**DELIVERY["tool_calls"][0], "secret": "not permitted"}]},
])
async def test_invalid_delivery_does_not_commit_checkpoint_or_spend(state, change):
    started = await native_run(state)
    running = await claim(state, await create(state, native_run_id=started.run.run_id))
    with pytest.raises(PrivateTaskError):
        await checkpoint(state, running, delivery={**DELIVERY, **change})
    assert await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=running["id"]) == running
    assert await state.store.private_tasks.pending_deliveries() == []


@pytest.mark.parametrize("invalid", ["unbound", "queued_result", "native_cancelled"])
async def test_delivery_requires_an_active_bound_native_run_and_stopped_bounded_step(state, invalid):
    started = await native_run(state)
    running = await claim(state, await create(
        state, native_run_id=None if invalid == "unbound" else started.run.run_id,
    ))
    if invalid == "native_cancelled":
        await state.store.conversations.finish_run(
            user_id=state.owner.user_id, run_id=started.run.run_id,
            assistant_content="Stopped", outcome="cancelled",
        )
    with pytest.raises(PrivateTaskError):
        await checkpoint(state, running, status="queued" if invalid == "queued_result" else "completed")
    saved = await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=running["id"])
    assert saved == running and saved["delivery"] is None


async def test_delivery_ack_is_exact_idempotent_and_does_not_change_lease_revision(state):
    started = await native_run(state)
    running = await claim(state, await create(state, native_run_id=started.run.run_id))
    committed = await checkpoint(state, running)
    kwargs = dict(user_id=state.owner.user_id, task_id=committed["id"],
                  run_id=started.run.run_id, checkpoint_sequence=1)
    other = await owner(state.store, "bob")
    assert not await state.store.private_tasks.ack_delivery(**{**kwargs, "user_id": other.user_id})
    for wrong in ({"run_id": "run_other"}, {"checkpoint_sequence": 2}):
        with pytest.raises(PrivateTaskConflictError):
            await state.store.private_tasks.ack_delivery(**{**kwargs, **wrong})
    assert await state.store.private_tasks.ack_delivery(**kwargs)
    assert await state.store.private_tasks.ack_delivery(**kwargs)
    committed["delivery"]["pending"] = False
    assert await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=committed["id"]) == committed
    assert await state.store.private_tasks.pending_deliveries() == []
    # A retry of the lost checkpoint acknowledgement cannot re-arm delivery.
    assert await checkpoint(state, running) == committed


async def test_trusted_outbox_and_restart_cleanup_can_finalize_a_revoked_owner_without_reads(state):
    started = await native_run(state)
    running = await claim(state, await create(state, native_run_id=started.run.run_id))
    committed = await checkpoint(state, running)
    state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
    state.store._conn.commit()
    with pytest.raises(PrivateTaskAccessError):
        await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=committed["id"])
    assert await state.store.private_tasks.pending_deliveries() == [committed]
    assert await state.store.private_tasks.recover_for_restart() == 0
    assert await state.store.private_tasks.ack_delivery(
        user_id=state.owner.user_id, task_id=committed["id"],
        run_id=started.run.run_id, checkpoint_sequence=1,
    )


async def test_restart_pauses_queued_and_running_once_and_caps_time_at_expiry(state):
    queued = await create(state)
    other_conversation, other_case = await case(state.store, state.owner)
    other = SimpleNamespace(**{**vars(state), "conversation": other_conversation, "case": other_case})
    running = await claim(other, await create(other))
    state.clock.advance(5)
    running = await state.store.private_tasks.renew(
        user_id=state.owner.user_id, task_id=running["id"], expected_revision=running["revision"],
        lease_token=running["lease_token"], fence=running["fence"], pins=PINS, progress=1,
        lease_seconds=10, usage={"model_tokens": 80, "tool_calls": 2},
    )
    state.clock.advance(1000)
    state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
    state.store._conn.commit()
    assert await state.store.private_tasks.recover_for_restart() == 2
    rows = {row["task_id"]: row for row in state.store._conn.execute("SELECT * FROM app_private_tasks")}
    saved = {key: json.loads(row["record_json"]) for key, row in rows.items()}
    for task in (queued, running):
        recovered = saved[task["id"]]
        assert recovered["status"] == "paused" and recovered["stop_reason"] == "server_restart"
        assert recovered["fence"] == task["fence"] + 1
        assert recovered["revision"] == task["revision"] + 1
        assert not recovered["lease_token"] and recovered["lease_expires_at"] is None
        assert recovered["checkpoint"] == task["checkpoint"]
    assert saved[queued["id"]]["usage"]["wall_seconds"] == 0
    assert saved[running["id"]]["usage"] == {
        "steps": 0, "wall_seconds": 15, "model_tokens": 80, "tool_calls": 2,
    }
    assert await state.store.private_tasks.recover_for_restart() == 0


async def test_rebinding_after_delivered_resume_preserves_spend_and_finds_new_attempt(state):
    started = await native_run(state)
    running = await claim(state, await create(state, native_run_id=started.run.run_id))
    saved = await checkpoint(state, running)
    payload = saved["delivery"]["payload"]
    await state.store.conversations.finish_run(
        user_id=state.owner.user_id, run_id=started.run.run_id,
        **{**payload, "sources": [SourceSnapshot(**s) for s in payload["sources"]],
           "tool_calls": [ToolCallSnapshot(**t) for t in payload["tool_calls"]],
           "models": [ModelUsageSnapshot(**m) for m in payload["models"]]},
    )
    assert await state.store.private_tasks.ack_delivery(
        user_id=state.owner.user_id, task_id=saved["id"],
        run_id=started.run.run_id, checkpoint_sequence=1,
    )
    queued = await control(state, saved, "resume")
    next_run = await native_run(state)
    rebound = await state.store.private_tasks.bind_native_run(
        user_id=state.owner.user_id, task_id=queued["id"],
        expected_revision=queued["revision"], run_id=next_run.run.run_id,
    )
    assert rebound["native_run_id"] == next_run.run.run_id
    assert rebound["usage"] == saved["usage"] and rebound["delivery"] is None
    assert await state.store.private_tasks.get_for_run(
        user_id=state.owner.user_id, run_id=started.run.run_id,
    ) is None
    messages = await state.store.conversations.list_messages(
        user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
    )
    assert len(messages) == 4 and messages[1].content == DELIVERY["assistant_content"]


async def test_delivery_preserves_partial_work_when_accounting_reaches_budget(state):
    started = await native_run(state)
    running = await claim(state, await create(
        state, native_run_id=started.run.run_id, limits={"max_model_tokens": 100},
    ))
    saved = await checkpoint(state, running, status="completed")
    assert saved["status"] == "paused" and saved["stop_reason"] == "budget_exhausted"
    assert saved["checkpoint"] is not None and saved["delivery"]["pending"]
    assert saved["usage"]["model_tokens"] == 120
    assert await state.store.private_tasks.pending_deliveries() == [saved]
