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


async def test_verified_extraction_refines_task_file_inventory_without_invalidating_lease(native):
    from test_private_project_runs import EVIDENCE, _ready_project

    from audrey.processing_policy import processing_scope
    from audrey.tax_documents import private_source_identity

    state = native
    _, cid, file_id = await _ready_project(state)
    _enable(state)
    case = await state.store.tax_cases.create(
        user_id=state.owner.user_id, conversation_id=cid, tax_year=2025,
        goal="Extract the selected invoice total", reference_release=_RELEASE,
    )
    runtime = state.env.runtime
    pins = runtime_task_pins(runtime)
    task = await state.store.private_tasks.create(
        user_id=state.owner.user_id, conversation_id=cid,
        expected_case_revision=case["revision"], pins=pins,
    )
    assert task["sources"]["files"] == []
    with processing_scope(runtime.policy):
        identity = await private_source_identity(state.app, state.owner, file_id)

    async def guard():
        await require_tax_case(state.app, state.owner, cid)

    def prepare(user_id, conversation_id):
        return prepared_private_step(
            app=state.app, runtime=runtime, principal=state.owner,
            conversation_id=conversation_id, pins=pins, guard=guard,
        )

    async def work(_app, step):
        handler = private_tool_handler("tax_review", user_id=state.owner.storage_namespace)
        saved = await handler({
            "operation": "propose_fact", "expected_revision": case["revision"],
            "data": {"label": "Invoice total", "value": "425", "kind": "extracted",
                     "provenance": {**identity, "page": 1, "original_text": EVIDENCE,
                                    "normalization": "decimal USD"}},
        })
        assert "error" not in saved
        # The first extraction adds the original checksum to the canonical case.
        # The following tool and final checkpoint must keep the same lease.
        inspected = await handler({"operation": "inspect", "section": "facts"})
        assert "error" not in inspected
        await step.check()
        return StepResult({"stage": "facts", "summary": "Invoice total awaits owner confirmation.",
                           "fact_ids": [saved["updated_entry"]["id"]]}, "waiting_for_user")

    saved = await PrivateTaskStepRunner(state.store.private_tasks, prepare).execute(
        user_id=state.owner.user_id, task_id=task["id"], work=work,
    )
    assert saved["status"] == "waiting_for_user"
    assert len(saved["sources"]["files"]) == 1
    bound = saved["sources"]["files"][0]
    assert {key: bound[key] for key in identity} == identity
    assert bound["revision"] == identity["checksum"]
    assert saved["usage"]["tool_calls"] == 2


@pytest.mark.parametrize("input_digest", ["a" * 64, "not-a-digest"])
async def test_prepared_runtime_validates_optional_input_digest(native, input_digest):
    state = native
    _enable(state)
    pins = dict(runtime_task_pins(state.env.runtime), input_digest=input_digest)

    async def guard():
        pass

    if len(input_digest) != 64:
        with pytest.raises(PrivateTaskError):
            async with prepared_private_step(
                app=state.app, runtime=state.env.runtime, principal=state.owner,
                conversation_id="unused", pins=pins, guard=guard,
            ):
                pytest.fail("Invalid document input digest must refuse preparation")
    else:
        async with prepared_private_step(
            app=state.app, runtime=state.env.runtime, principal=state.owner,
            conversation_id="unused", pins=pins, guard=guard,
        ) as prepared:
            assert prepared.pins == pins
