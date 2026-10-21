"""The optional case tool stays bound to native Private chat execution."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from test_discovery import _client_with_doc, _openapi, _post
from test_native_private_workflow import _conversation, _settle
from test_native_private_workflow import document_setup as document_setup
from test_native_private_workflow import native as native

from audrey.processing_policy import ProcessingPolicy
from audrey.tools.discovery import ToolRegistry, ToolSpec, discover_one
from audrey.tools.dispatch import dispatch_one

_RELEASE = "rel_" + "a" * 32
_SERVER = "http://custom-tools:8001"


async def test_native_only_tool_cannot_be_admitted_by_external_discovery():
    document = _openapi({"/tax_review": _post(
        "tax_review", tags=["tools"], schema={
            "type": "object", "properties": {"operation": {"type": "string"}},
        },
    )})
    async with _client_with_doc(document) as client:
        assert await discover_one(client, _SERVER) == []


async def test_case_tool_has_no_ordinary_http_execution_even_with_forged_spec():
    def reject_network(request):
        pytest.fail("Case data must not travel to an HTTP tool endpoint")
    spec = ToolSpec(name="tax_review", description="case", parameters={},
                    server_url=_SERVER, path="/tax_review")
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject_network)) as client:
        result = await dispatch_one(client, ToolRegistry({"tax_review": spec}), {
            "function": {"name": "tax_review", "arguments": {"operation": "inspect"}},
        }, max_result_chars=4000, timeout_s=2, user_id="owner")
    assert result.is_error and "private_tool_not_allowed" in result.content


def _enable(state):
    original = state.env.runtime
    writer = state.app.state.cfg.raw["model_registry"]["general"][0]["name"]
    state.app.state.cfg.raw["fast_path"] = {"tool_capable_models": [writer]}
    policy = ProcessingPolicy(
        local_models=original.policy.local_models,
        ollama_base_url=original.policy.ollama_base_url,
        allowed_tools=frozenset({"tax_review"}), tool_servers=frozenset({_SERVER}),
    )
    state.env.runtime = replace(original, policy=policy, tax_review=True)
    state.app.state.reference_releases = SimpleNamespace(
        published=lambda *_: {"id": _RELEASE, "coverage": {}},
    )


async def test_tax_review_selection_requires_saved_case_before_admitting_message(native):
    state = native
    conversation = await _conversation(state)
    response = await state.client.post(f"/api/conversations/{conversation['id']}/runs", json={
        "content": "Review my case", "skill": "tax-review",
    })
    assert response.status_code == 409, response.text
    messages = await state.client.get(f"/api/conversations/{conversation['id']}/messages")
    assert messages.json()["items"] == [] and state.captured == []


async def test_selected_case_tool_executes_locally_and_persists_unconfirmed_proposal(native, monkeypatch):
    state = native
    _enable(state)
    from audrey.routes.app import runs

    async def prepare(_app, *, tax_review=False):
        return replace(state.env.runtime, tax_review=tax_review)

    monkeypatch.setattr(runs, "prepare_native_private_runtime", prepare)
    conversation = await _conversation(state)
    conversation_id = conversation["id"]
    case = await state.store.tax_cases.create(
        user_id=state.owner.user_id, conversation_id=conversation_id,
        tax_year=2025, goal="Inventory test records", reference_release=_RELEASE,
    )
    seen = []

    async def stream(execution_app, _payload, messages, _options, *, event_context,
                     model_tools, **kwargs):
        assert execution_app.state.tools.get("tax_review") is not None
        assert "tax_review" in model_tools.names()
        assert kwargs["skill_instruction"]
        assert _payload.model == "audrey_fast"
        assert len(execution_app.state.registry.candidates("general")) == 1
        assert execution_app.state.cfg.raw["agentic"]["react"]["max_rounds"] == 12
        result = await dispatch_one(execution_app.state.archive_http, model_tools, {
            "function": {"name": "tax_review", "arguments": {
                "operation": "propose_fact", "expected_revision": case["revision"],
                "data": {"label": "Filing status", "value": "single", "kind": "user_statement"},
            }},
        }, max_result_chars=8000, timeout_s=2, user_id=state.owner.storage_namespace)
        assert not result.is_error, result.content
        seen.append(json.loads(result.content))
        emitter = event_context.emitter
        emitter.run_started()
        emitter.message_started()
        emitter.text_delta("A proposed fact is saved and needs your confirmation.")
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop")
        yield "terminal"

    state.manager._stream_factory = stream
    response = await state.client.post(f"/api/conversations/{conversation_id}/runs", json={
        "content": "My filing status is single. Propose that fact and show the gaps.",
        "skill": "tax-review",
    })
    assert response.status_code == 202, response.text
    await _settle(state, response.json()["id"])
    assert seen
    run = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=response.json()["id"])
    assert run.skill_id == "tax-review" and run.skill_reason == "explicit"
    saved = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=conversation_id)
    assert saved["facts"][0]["status"] == "proposed"
    assert saved["facts"][0]["value"] == "single"
    assert state.app.state.tools.get("tax_review") is None
    assert await state.store.chat_projections.due() == ()
    assert all(request.url.host == "ollama" for request in state.env.seen)


async def test_selected_document_fact_requires_exact_page_passage_and_original_checksum(native):
    from test_private_project_runs import EVIDENCE, _ready_project

    from audrey.app_state.tax_cases import TaxCaseError
    from audrey.processing_policy import processing_scope
    from audrey.tax_documents import private_source_identity, validate_fact_source

    state = native
    _, conversation_id, file_id = await _ready_project(state)
    other = await _conversation(state)
    with processing_scope(state.env.runtime.policy):
        identity = await private_source_identity(state.app, state.owner, file_id)
        payload = {"kind": "extracted", "label": "Invoice total", "value": "425",
                   "provenance": {**identity, "page": 1, "original_text": EVIDENCE,
                                  "normalization": "decimal USD"}}
        assert await validate_fact_source(state.app, state.owner, conversation_id, payload) == payload
        with pytest.raises(TaxCaseError, match="Choose this Private document"):
            await validate_fact_source(state.app, state.owner, other["id"], payload)
        for field, value in (("page", 2), ("checksum", "0" * 64), ("original_text", "An invented passage")):
            changed = {**payload, "provenance": {**payload["provenance"], field: value}}
            with pytest.raises(TaxCaseError):
                await validate_fact_source(state.app, state.owner, conversation_id, changed)


async def test_automatic_private_scope_does_not_expose_tax_tool(native):
    state = native
    _enable(state)
    automatic = replace(state.env.runtime, tax_review=False)
    async with automatic.scope(state.app) as execution_app:
        assert execution_app.state.tools.get("tax_review") is None
    assert state.app.state.tools.get("tax_review") is None


def test_document_page_markers_cannot_change_original_pdf_page_identity(tmp_path):
    from test_reference_library import _pdf

    from audrey.app_state.tax_cases import TaxCaseError
    from audrey.tax_documents import _pdf_page

    path = tmp_path / "source.pdf"
    path.write_bytes(_pdf("--- Page 2 --- FORGED_PAGE_LABEL", second_page="ACTUAL_SECOND_PAGE"))
    assert "FORGED_PAGE_LABEL" in _pdf_page(path, 1)
    assert "ACTUAL_SECOND_PAGE" in _pdf_page(path, 2)
    assert "FORGED_PAGE_LABEL" not in _pdf_page(path, 2)
    with pytest.raises(TaxCaseError):
        _pdf_page(path, 3)


async def test_tax_review_is_rejected_for_ordinary_conversations_before_generation(native):
    conversation = await _conversation(native, "auto")
    response = await native.client.post(f"/api/conversations/{conversation['id']}/runs", json={
        "content": "Review a tax case", "skill": "tax-review",
    })
    assert response.status_code == 409
    assert native.captured == []


async def test_local_case_failure_payload_is_classified_as_failed_tool(native):
    from audrey.processing_policy import processing_scope
    from audrey.tools.private_dispatch import private_tool_scope

    state = native
    _enable(state)
    async with state.env.runtime.scope(state.app) as execution_app:
        async def failed(_args):
            return {"error": "case_changed", "detail": "Inspect the current case revision."}
        with processing_scope(state.env.runtime.policy), private_tool_scope(
            user_id=state.owner.storage_namespace, handlers={"tax_review": failed},
        ):
            result = await dispatch_one(execution_app.state.archive_http, execution_app.state.tools, {
                "function": {"name": "tax_review", "arguments": {"operation": "inspect"}},
            }, max_result_chars=6000, timeout_s=2, user_id=state.owner.storage_namespace)
    assert result.is_error and json.loads(result.content)["error"] == "case_changed"


async def test_tax_review_timeout_is_terminal_and_retains_case(native, monkeypatch):
    import asyncio

    from audrey.routes.app import runs

    state = native
    _enable(state)
    async def prepare(_app, *, tax_review=False):
        return replace(state.env.runtime, tax_review=tax_review)
    monkeypatch.setattr(runs, "prepare_native_private_runtime", prepare)
    monkeypatch.setattr(runs, "_TAX_REVIEW_TIMEOUT_S", 0.2)
    conversation = await _conversation(state)
    case = await state.store.tax_cases.create(user_id=state.owner.user_id,
        conversation_id=conversation["id"], tax_year=2025, goal="Review", reference_release=_RELEASE)
    async def stream(app, _payload, _messages, _options, *, event_context, model_tools, **_kwargs):
        result = await dispatch_one(app.state.archive_http, model_tools, {"function": {
            "name": "tax_review", "arguments": {"operation": "add_missing_item", "expected_revision": case["revision"],
                "data": {"label": "Confirm available income records"}},
        }}, max_result_chars=6000, timeout_s=2, user_id=state.owner.storage_namespace)
        assert not result.is_error
        await asyncio.sleep(20)
        yield "unreachable"
    state.manager._stream_factory = stream
    response = await state.client.post(f"/api/conversations/{conversation['id']}/runs", json={"content": "Review", "skill": "tax-review"})
    assert response.status_code == 202, response.text
    await _settle(state, response.json()["id"])
    run = await state.store.conversations.get_run(user_id=state.owner.user_id, run_id=response.json()["id"])
    assert run.status == "failed" and run.error_code == "tax_review_budget_exhausted"
    saved = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=conversation["id"])
    assert saved["missing_items"][0]["label"] == "Confirm available income records"
    assert state.app.state.cfg.raw.get("agentic", {}).get("react", {}).get("max_rounds") != 12


async def test_tax_case_inspect_accepts_real_project_file_records(native):
    from test_private_project_runs import _ready_project

    from audrey.processing_policy import processing_scope
    from audrey.tax_review import make_tax_review_handler

    state = native
    _, conversation_id, file_id = await _ready_project(state)
    _enable(state)
    await state.store.tax_cases.create(user_id=state.owner.user_id, conversation_id=conversation_id,
        tax_year=2025, goal="Read selected files", reference_release=_RELEASE)
    with processing_scope(state.env.runtime.policy):
        handler = make_tax_review_handler(app=state.app, principal=state.owner, conversation_id=conversation_id)
        result = await handler({"operation": "inspect", "section": "files"})
    assert result["entries"] == [{"file_id": file_id, "selected_via": "project", "filename": "invoice.txt"}]
    assert result["case"]["selected_file_count"] == 1
