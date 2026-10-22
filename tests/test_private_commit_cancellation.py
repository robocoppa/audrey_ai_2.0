"""Committed Private replies survive shutdown and failed worker admission retries."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from test_native_private_reviews import (
    ANSWER,
    proposal_stream,
    saved_case,
    start_review,
    task_status,
)
from test_native_private_workflow import _settle
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native

from audrey.app_state.private_tasks import PrivateTaskConflictError
from audrey.private_reviews import replay_private_deliveries
from audrey.private_task_admission import PrivateTaskAdmission


@pytest.mark.parametrize("failure", ["shutdown", "lost_checkpoint_ack"])
async def test_committed_review_reply_survives_interruption_before_delivery(native, monkeypatch, failure):
    state = native
    conversation, case = await saved_case(state)
    calls = []
    state.manager._stream_factory = proposal_stream(state, case, calls)
    committed = asyncio.Event()
    original = state.store.private_tasks.checkpoint

    async def interrupted_after_commit(**kwargs):
        saved = await original(**kwargs)
        committed.set()
        if failure == "lost_checkpoint_ack":
            raise OSError("Simulated lost checkpoint acknowledgement")
        await asyncio.Event().wait()
        return saved

    monkeypatch.setattr(state.store.private_tasks, "checkpoint", interrupted_after_commit)
    run = await start_review(state, conversation["id"])
    await asyncio.wait_for(committed.wait(), 3)
    if failure == "shutdown":
        assert len(await state.store.private_tasks.pending_deliveries()) == 1
        await asyncio.wait_for(state.manager.stop(), 3)
    else:
        await _settle(state, run["id"])
    await replay_private_deliveries(state.store)
    record = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=run["id"])
    messages = await state.store.conversations.list_messages(
        user_id=state.owner.user_id, conversation_id=conversation["id"],
    )
    assert record.status == "succeeded"
    assert messages[-1].status == "completed" and messages[-1].content == ANSWER
    assert messages[-1].message_id == run["assistant_message_id"] and len(messages) == 2
    assert state.manager._runs[run["id"]].terminal_event.status == "succeeded"
    assert calls == [run["id"]]
    assert await state.store.private_tasks.pending_deliveries() == []


async def test_post_launch_preclaim_refusal_pauses_exact_task_and_allows_retry(native, monkeypatch):
    state = native
    conversation, case = await saved_case(state)
    calls = []
    state.manager._stream_factory = proposal_stream(state, case, calls)
    original = PrivateTaskAdmission.prepare
    preparations = 0

    @asynccontextmanager
    async def refuse_worker(self, user_id, conversation_id):
        nonlocal preparations
        preparations += 1
        if preparations == 2:
            raise PrivateTaskConflictError("Source changed between launch and claim")
        async with original(self, user_id, conversation_id) as prepared:
            yield prepared

    monkeypatch.setattr(PrivateTaskAdmission, "prepare", refuse_worker)
    first = await start_review(state, conversation["id"])
    await _settle(state, first["id"])
    task = await task_status(state, conversation["id"])
    assert preparations == 2 and calls == []
    assert task["status"] == "paused" and task["stop_reason"] == "admission_failed"
    first_run = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=first["id"])
    assert first_run.status != "running"
    monkeypatch.setattr(PrivateTaskAdmission, "prepare", original)
    second = await start_review(state, conversation["id"], "Retry the saved review")
    await _settle(state, second["id"])
    resumed = await task_status(state, conversation["id"])
    assert resumed["id"] == task["id"] and resumed["status"] == "waiting_for_user"
    assert resumed["native_run_id"] == second["id"] and calls == [second["id"]]
