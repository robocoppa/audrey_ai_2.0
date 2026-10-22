"""Native saved Private reviews retain normal chat ownership and controls."""

from __future__ import annotations

import asyncio
import json

from test_native_private_workflow import _conversation, _settle
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native
from test_tax_review_runtime import _RELEASE, _enable

from audrey.app_state import ApplicationStore
from audrey.app_state.private_tasks import PrivateTaskConflictError
from audrey.auth import require_principal
from audrey.private_reviews import replay_private_deliveries
from audrey.tools.dispatch import dispatch_one

ANSWER = "Your filing status is proposed as single. Please confirm it before I use it."


async def saved_case(state):
    _enable(state)
    conversation = await _conversation(state)
    case = await state.store.tax_cases.create(
        user_id=state.owner.user_id, conversation_id=conversation["id"],
        tax_year=2025, goal="Review only the synthetic filing status", reference_release=_RELEASE,
    )
    return conversation, case


async def start_review(state, conversation_id, content="Propose my filing status as single."):
    response = await state.client.post(f"/api/conversations/{conversation_id}/runs", json={
        "content": content, "skill": "tax-review", "model_id": "private",
    })
    assert response.status_code == 202, response.text
    return response.json()


async def task_status(state, conversation_id):
    response = await state.client.get(f"/api/conversations/{conversation_id}/private-task")
    assert response.status_code == 200, response.text
    return response.json()["task"]


async def case_tool(state, execution_app, model_tools, operation, **arguments):
    result = await dispatch_one(execution_app.state.archive_http, model_tools, {
        "function": {"name": "tax_review", "arguments": {
            "operation": operation, **arguments,
        }},
    }, max_result_chars=8000, timeout_s=2, user_id=state.owner.storage_namespace)
    assert not result.is_error, result.content
    return json.loads(result.content)


def proposal_stream(state, case, calls):
    async def stream(execution_app, _payload, _messages, _options, *, event_context,
                     model_tools, **_kwargs):
        calls.append(event_context.run_id)
        saved = await case_tool(state, execution_app, model_tools, "propose_fact",
            expected_revision=case["revision"], data={
                "label": "Filing status", "value": "single", "kind": "user_statement",
            })
        assert saved
        emitter = event_context.emitter
        emitter.run_started()
        emitter.message_started()
        emitter.text_delta(ANSWER)
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop")
        yield "terminal"
    return stream


async def test_tax_review_native_turn_saves_one_owned_task_and_one_final_answer(native):
    state = native
    conversation, case = await saved_case(state)
    calls = []
    state.manager._stream_factory = proposal_stream(state, case, calls)
    run = await start_review(state, conversation["id"])
    await _settle(state, run["id"])
    task = await task_status(state, conversation["id"])
    assert task["status"] == "waiting_for_user" and task["native_run_id"] == run["id"]
    assert task["usage"]["tool_calls"] == 1 and task["usage"]["steps"] == 1
    assert not ({"pins", "lease_token", "lease_expires_at", "checkpoint", "delivery", "user_id",
                 "case_id", "sources"} & task.keys())
    messages = await state.store.conversations.list_messages(
        user_id=state.owner.user_id, conversation_id=conversation["id"],
    )
    assert [(message.role, message.status) for message in messages] == [
        ("user", "completed"), ("assistant", "completed"),
    ]
    assert messages[-1].content == ANSWER and messages[-1].message_id == run["assistant_message_id"]
    record = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=run["id"])
    assert record.status == "succeeded" and record.skill_id == "tax-review"
    saved = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=conversation["id"])
    assert saved["facts"][0]["status"] == "proposed" and saved["facts"][0]["value"] == "single"
    assert calls == [run["id"]]
    assert await state.store.private_tasks.pending_deliveries() == []
    assert await replay_private_deliveries(state.store) == 0
    assert await state.store.chat_projections.due() == ()


async def test_pause_stops_callback_then_normal_review_turn_resumes_same_task_without_resetting_spend(native):
    state = native
    conversation, case = await saved_case(state)
    entered, closed = asyncio.Event(), asyncio.Event()

    async def blocked(execution_app, _payload, _messages, _options, *, model_tools, **_kwargs):
        await case_tool(state, execution_app, model_tools, "inspect")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
        yield "unreachable"

    state.manager._stream_factory = blocked
    first = await start_review(state, conversation["id"])
    await asyncio.wait_for(entered.wait(), 3)
    before = await task_status(state, conversation["id"])
    assert before["status"] == "running" and before["usage"]["tool_calls"] == 1
    response = await state.client.post(f"/api/private-tasks/{before['id']}/actions", json={
        "action": "pause", "expected_revision": before["revision"],
    })
    assert response.status_code == 200, response.text
    await asyncio.wait_for(closed.wait(), 3)
    await _settle(state, first["id"])
    paused = await task_status(state, conversation["id"])
    assert paused["status"] == "paused" and paused["stop_reason"] == "owner_paused"
    first_record = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=first["id"])
    assert first_record.status == "cancelled" and first_record.error_code == "private_task_paused"
    calls = []
    state.manager._stream_factory = proposal_stream(state, case, calls)
    second = await start_review(state, conversation["id"], "Continue the saved review.")
    await _settle(state, second["id"])
    resumed = await task_status(state, conversation["id"])
    assert resumed["id"] == before["id"] and resumed["native_run_id"] == second["id"]
    assert resumed["status"] == "waiting_for_user"
    assert resumed["usage"]["tool_calls"] == paused["usage"]["tool_calls"] + 1
    assert resumed["usage"]["wall_seconds"] >= paused["usage"]["wall_seconds"]
    assert calls == [second["id"]]
    rows = state.store._conn.execute("SELECT COUNT(*) FROM app_private_tasks").fetchone()[0]
    assert rows == 1


async def test_normal_native_stop_cancels_saved_task_and_no_late_case_write_occurs(native):
    state = native
    conversation, _ = await saved_case(state)
    entered, closed = asyncio.Event(), asyncio.Event()

    async def blocked(execution_app, _payload, _messages, _options, *, model_tools, **_kwargs):
        await case_tool(state, execution_app, model_tools, "inspect")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
        yield "unreachable"

    state.manager._stream_factory = blocked
    run = await start_review(state, conversation["id"])
    await asyncio.wait_for(entered.wait(), 3)
    response = await state.client.post(f"/api/runs/{run['id']}/cancel")
    assert response.status_code == 200, response.text
    await asyncio.wait_for(closed.wait(), 3)
    await _settle(state, run["id"])
    task = await task_status(state, conversation["id"])
    assert task["status"] == "cancelled" and task["stop_reason"] == "owner_cancelled"
    record = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=run["id"])
    assert record.status == "cancelled" and record.error_code == "cancelled_by_user"
    case = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=conversation["id"])
    assert case["facts"] == [] and task["usage"]["tool_calls"] == 1


async def test_foreign_private_review_status_and_controls_return_not_found(native):
    state = native
    conversation, case = await saved_case(state)
    state.manager._stream_factory = proposal_stream(state, case, [])
    run = await start_review(state, conversation["id"])
    await _settle(state, run["id"])
    owned = await task_status(state, conversation["id"])
    other = await state.store.resolve_external_identity(
        provider="test", subject="foreign-review-user", email="other@example.com",
        display_name="Other", role="user", auth_method="test",
    )
    state.app.dependency_overrides[require_principal] = lambda: other
    response = await state.client.get(f"/api/conversations/{conversation['id']}/private-task")
    assert response.status_code == 404
    for task_id in (owned["id"], "ptask_missing"):
        response = await state.client.post(f"/api/private-tasks/{task_id}/actions", json={
            "action": "cancel", "expected_revision": owned["revision"],
        })
        assert response.status_code == 404, response.text
    saved = await state.store.private_tasks.get(user_id=state.owner.user_id, task_id=owned["id"])
    assert saved["status"] == "waiting_for_user"


async def test_restart_replays_lost_delivery_ack_without_regeneration_or_duplicate_native_reply(native, monkeypatch):
    state = native
    conversation, case = await saved_case(state)
    calls = []
    state.manager._stream_factory = proposal_stream(state, case, calls)

    async def lost_ack(**_kwargs):
        raise OSError("Simulated process loss after native delivery")

    monkeypatch.setattr(state.store.private_tasks, "ack_delivery", lost_ack)
    run = await start_review(state, conversation["id"])
    await _settle(state, run["id"])
    before = await state.store.conversations.list_messages(
        user_id=state.owner.user_id, conversation_id=conversation["id"],
    )
    assert before[-1].content == ANSWER
    assert state.manager._runs[run["id"]].terminal_event.status == "succeeded"
    assert len(await state.store.private_tasks.pending_deliveries()) == 1
    reopened = ApplicationStore(state.store.path)
    try:
        assert await replay_private_deliveries(reopened) == 1
        assert await reopened.private_tasks.recover_for_restart() == 0
        assert await reopened.conversations.recover_interrupted_runs() == 0
        assert await replay_private_deliveries(reopened) == 0
        after = await reopened.conversations.list_messages(
            user_id=state.owner.user_id, conversation_id=conversation["id"],
        )
        assert after == before and len(after) == 2
        record = await reopened.conversations.get_run(user_id=state.owner.user_id, run_id=run["id"])
        assert record.status == "succeeded"
        assert calls == [run["id"]]
        assert await reopened.private_tasks.pending_deliveries() == []
    finally:
        reopened.close()


async def test_admission_failure_finalizes_saved_run_and_allows_retry(native, monkeypatch):
    from audrey.private_reviews import NativePrivateReviews

    state = native
    conversation, case = await saved_case(state)
    original = NativePrivateReviews.admit

    async def fail_admission(_self, **_kwargs):
        raise PrivateTaskConflictError("Changed before task admission")

    monkeypatch.setattr(NativePrivateReviews, "admit", fail_admission)
    response = await state.client.post(f"/api/conversations/{conversation['id']}/runs", json={
        "content": "Review my synthetic case", "skill": "tax-review", "model_id": "private",
    })
    assert response.status_code == 409, response.text
    rows = state.store._conn.execute("SELECT status,error_code FROM app_runs").fetchall()
    assert [(r["status"], r["error_code"]) for r in rows] == [("failed", "private_task_admission_unavailable")]
    assert state.manager._runs == {}
    monkeypatch.setattr(NativePrivateReviews, "admit", original)
    calls = []
    state.manager._stream_factory = proposal_stream(state, case, calls)
    run = await start_review(state, conversation["id"], "Retry that review.")
    await _settle(state, run["id"])
    assert (await task_status(state, conversation["id"]))["status"] == "waiting_for_user"
    assert calls == [run["id"]]


async def test_restart_pauses_interrupted_native_review_without_scheduling_another_answer(native):
    state = native
    conversation, _ = await saved_case(state)
    entered, closed = asyncio.Event(), asyncio.Event()
    calls = []

    async def interrupted(execution_app, _payload, _messages, _options, *, model_tools,
                          event_context, **_kwargs):
        calls.append(event_context.run_id)
        await case_tool(state, execution_app, model_tools, "inspect")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
        yield "unreachable"

    state.manager._stream_factory = interrupted
    run = await start_review(state, conversation["id"])
    await asyncio.wait_for(entered.wait(), 3)
    before = await task_status(state, conversation["id"])
    reopened = ApplicationStore(state.store.path)
    try:
        assert await replay_private_deliveries(reopened) == 0
        assert await reopened.private_tasks.recover_for_restart() == 1
        assert await reopened.conversations.recover_interrupted_runs() == 1
        task = await reopened.private_tasks.get(user_id=state.owner.user_id, task_id=before["id"])
        assert task["status"] == "paused" and task["stop_reason"] == "server_restart"
        assert task["usage"]["tool_calls"] == before["usage"]["tool_calls"] == 1
        assert task["usage"]["wall_seconds"] >= before["usage"]["wall_seconds"]
        assert task["fence"] > 0 and not task["lease_token"]
        record = await reopened.conversations.get_run(user_id=state.owner.user_id, run_id=run["id"])
        assert record.status == "failed" and record.error_code == "server_restart"
        # The old worker's next guard observes the recovery fence and stops.
        await asyncio.wait_for(closed.wait(), 3)
        await _settle(state, run["id"])
        assert calls == [run["id"]]
        messages = await reopened.conversations.list_messages(
            user_id=state.owner.user_id, conversation_id=conversation["id"],
        )
        assert len(messages) == 2 and messages[-1].status == "incomplete"
        assert await reopened.private_tasks.recover_for_restart() == 0
        assert await reopened.private_tasks.get(user_id=state.owner.user_id, task_id=before["id"]) == task
    finally:
        reopened.close()
