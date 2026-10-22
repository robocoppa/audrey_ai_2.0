"""The leased step helper uses the actual isolated Private runtime and tools."""

from __future__ import annotations

import pytest
from test_native_private_workflow import _conversation
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native
from test_tax_review_runtime import _RELEASE, _enable

from audrey.app_state.private_tasks import PrivateTaskError
from audrey.private_task_runner import (
    PrivateTaskStepRunner,
    StepResult,
    prepared_private_step,
    runtime_task_pins,
)
from audrey.tax_review import require_tax_case
from audrey.tools.private_dispatch import private_tool_handler


async def test_real_runtime_binds_local_tool_writes_to_the_task(native):
    state = native
    _enable(state)
    conversation = await _conversation(state)
    cid = conversation["id"]
    case = await state.store.tax_cases.create(
        user_id=state.owner.user_id,
        conversation_id=cid,
        tax_year=2025,
        goal="Record a proposed fact",
        reference_release=_RELEASE,
    )
    runtime = state.env.runtime
    pins = runtime_task_pins(runtime)
    task = await state.store.private_tasks.create(
        user_id=state.owner.user_id,
        conversation_id=cid,
        expected_case_revision=case["revision"],
        pins=pins,
    )

    async def guard():
        await require_tax_case(state.app, state.owner, cid)

    def prepare(user_id, conversation_id):
        assert user_id == state.owner.user_id and conversation_id == cid
        return prepared_private_step(
            app=state.app,
            runtime=runtime,
            principal=state.owner,
            conversation_id=cid,
            pins=pins,
            guard=guard,
        )

    async def work(execution_app, step):
        assert execution_app is not state.app
        assert execution_app.state.gate is state.app.state.gate
        result = await private_tool_handler("tax_review", user_id=state.owner.storage_namespace)(
            {
                "operation": "propose_fact",
                "expected_revision": case["revision"],
                "data": {"label": "Filing status", "value": "single", "kind": "user_statement"},
            }
        )
        assert "error" not in result
        return StepResult(
            {
                "stage": "facts",
                "summary": "Owner confirmation is needed.",
                "fact_ids": [result["updated_entry"]["id"]],
            },
            "waiting_for_user",
        )

    saved = await PrivateTaskStepRunner(state.store.private_tasks, prepare).execute(
        user_id=state.owner.user_id, task_id=task["id"], work=work
    )
    assert saved["status"] == "waiting_for_user" and saved["usage"]["tool_calls"] == 1
    current = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=cid)
    assert saved["case_revision"] == current["revision"]
    assert current["facts"][0]["status"] == "proposed"


async def test_runtime_pin_mismatch_refuses_before_preparing_tools(native):
    state = native
    _enable(state)
    pins = runtime_task_pins(state.env.runtime)
    pins["policy_digest"] = "0" * 64

    async def guard():
        pytest.fail("Mismatched runtime pins must be refused first")

    with pytest.raises(PrivateTaskError):
        async with prepared_private_step(
            app=state.app,
            runtime=state.env.runtime,
            principal=state.owner,
            conversation_id="unused",
            pins=pins,
            guard=guard,
        ):
            pytest.fail("Mismatched runtime cannot start a Private step")
