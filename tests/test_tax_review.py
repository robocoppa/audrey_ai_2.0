"""Tax review binds owner context, official source bytes and arithmetic results."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from test_reference_library import _entry, _pdf, _record, _registry
from test_reference_releases import begin, checkpoint

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal
from audrey.processing_policy import ProcessingPolicy, ProcessingPolicyError, processing_scope
from audrey.references.irs_archive import ArchiveStore
from audrey.references.library import document_id
from audrey.references.releases import ReleaseStore
from audrey.routes.app import router
from audrey.tax_review import make_tax_review_handler, tax_review_spec
from audrey.tools.private_dispatch import PrivateToolArgumentsError
from audrey.user_data_visibility import block_remote_personal_reads, unblock_remote_personal_reads


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(
        provider="test", subject="owner", email="alice@example.com", display_name="Alice",
        role="user", auth_method="test",
    )
    conversation = await store.conversations.create(user_id=owner.user_id, private_required=True)
    root = tmp_path / "irs"
    source = _entry()
    with ArchiveStore(root) as archive:
        _record(archive, [source], content=_pdf("Dated official supporting passage."))
    registry = _registry(root)
    metadata = registry.lookup("irs-federal", document_id(source.source_url))
    releases = ReleaseStore(tmp_path / "reference-state")
    release = begin(releases, docs=[metadata])
    checkpoint(releases, release, metadata)
    releases.publish(release["id"], verified_chunks=1)
    app = FastAPI()
    app.state.application_store = store
    app.state.reference_libraries = registry
    app.state.reference_releases = releases
    app.include_router(router)
    current = [owner]
    app.dependency_overrides[require_principal] = lambda: current[0]
    policy = ProcessingPolicy(
        {"local:latest": "a" * 64}, "http://ollama:11434",
        allowed_tools=frozenset({"tax_review"}), tool_servers=frozenset({"http://custom-tools:8001"}),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(
            app=app, store=store, owner=owner, current=current, client=client, policy=policy,
            conversation=conversation, metadata=metadata, release=release, releases=releases,
            path=f"/api/conversations/{conversation.conversation_id}/tax-review",
        )
    store.close()


async def _create(state):
    response = await state.client.post(state.path, json={"tax_year": 2025, "goal": "Review a planning calculation"})
    assert response.status_code == 201, response.text
    return response.json()


async def _call(state, operation="inspect", **fields):
    with processing_scope(state.policy):
        handler = make_tax_review_handler(app=state.app, principal=state.owner, conversation_id=state.conversation.conversation_id)
        return await handler({"operation": operation, **fields})


def _evidence(state, **changes):
    return {
        "claim": "A dated claim requires official support", "document_id": state.metadata["document_id"],
        "reference_release": state.release["id"], "revision": state.metadata["revision_sha256"],
        "page": 1, "passage": "Dated official supporting passage.", "tax_year": 2025, **changes,
    }


async def test_case_creation_derives_pin_and_project_and_owner_routes(state):
    assert (await state.client.get(state.path)).json() == {"case": None}
    case = await _create(state)
    assert case["reference_release"] == state.release["id"]
    assert case["project_id"] is None and case["checklist"]["status"] == "needs_information"
    assert (await state.client.post(state.path, json={"tax_year": 2025, "goal": "Again"})).status_code == 409
    state.current[0] = await state.store.resolve_external_identity(
        provider="test", subject="other", email="other@example.com", display_name="Other",
        role="user", auth_method="test",
    )
    assert (await state.client.get(state.path)).status_code == 404
    assert (await state.client.post(state.path, json={"tax_year": 2025, "goal": "Foreign"})).status_code == 404


@pytest.mark.parametrize("extra", [{"reference_release": "caller-release"}, {"project_id": "foreign"}, {"files": []}, {"owner": "foreign"}])
async def test_browser_cannot_choose_case_source_scope(state, extra):
    response = await state.client.post(state.path, json={"tax_year": 2025, "goal": "Review", **extra})
    assert response.status_code == 422


@pytest.mark.parametrize("tax_year", [True, "2025", 1899, 2101])
async def test_case_year_is_strict_explicit_and_bounded(state, tax_year):
    assert (await state.client.post(state.path, json={"tax_year": tax_year, "goal": "Review"})).status_code == 422


async def test_nonprivate_context_and_absent_published_release_refused(state):
    ordinary = await state.store.conversations.create(user_id=state.owner.user_id)
    response = await state.client.get(f"/api/conversations/{ordinary.conversation_id}/tax-review")
    assert response.status_code == 409
    state.app.state.reference_releases = Mock(published=Mock(return_value=None))
    assert (await state.client.post(state.path, json={"tax_year": 2025, "goal": "Review"})).status_code == 503
    assert await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id) is None


async def test_personal_token_requires_full_scope(state):
    state.current[0] = dataclasses.replace(state.owner, auth_method="personal_token", scopes=frozenset({"account:read"}))
    assert (await state.client.get(state.path)).status_code == 403


async def test_only_owner_action_confirms_proposal_and_model_cannot_confirm(state):
    case = await _create(state)
    proposal = await _call(state, "propose_fact", expected_revision=case["revision"], data={"label": "Filing status", "value": "single", "kind": "user_statement"})
    fact_id = proposal["updated_entry"]["id"]
    inspected = await _call(state)
    assert inspected["entries"][0]["status"] == "proposed"
    with pytest.raises(PrivateToolArgumentsError):
        await _call(state, "confirm_fact", expected_revision=proposal["revision"], data={"fact_id": fact_id})
    confirmed = await state.client.post(state.path + "/actions", json={"operation": "confirm_fact", "expected_revision": proposal["revision"], "data": {"fact_id": fact_id}})
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["facts"][0]["status"] == "confirmed"
    stale = await _call(state, "add_missing_item", expected_revision=1, data={"label": "Check duplicates"})
    assert stale["error"] == "case_changed" and stale["status"] == 409


async def test_grouped_fact_confirmation_is_explicit_owner_action_and_hidden_from_model(state):
    case = await _create(state)
    identifiers = []
    for number in range(2):
        proposal = await _call(state, "propose_fact", expected_revision=case["revision"], data={
            "label": f"Fact {number}", "value": "single", "kind": "user_statement",
        })
        identifiers.append(proposal["updated_entry"]["id"])
        case["revision"] = proposal["revision"]
    with pytest.raises(PrivateToolArgumentsError):
        await _call(state, "confirm_facts", expected_revision=case["revision"], data={"fact_ids": identifiers})
    assert "confirm_facts" not in tax_review_spec("http://custom-tools:8001").parameters["properties"]["operation"]["enum"]
    body = {"operation": "confirm_facts", "expected_revision": case["revision"], "data": {"fact_ids": identifiers}}
    response = await state.client.post(state.path + "/actions", json=body)
    assert response.status_code == 200, response.text
    confirmed = response.json()
    assert confirmed["revision"] == case["revision"] + 1
    assert all(fact["status"] == "confirmed" for fact in confirmed["facts"])
    assert (await state.client.post(state.path + "/actions", json=body)).status_code == 409
    body["expected_revision"] = confirmed["revision"]
    assert (await state.client.post(state.path + "/actions", json=body)).json()["revision"] == confirmed["revision"]
    state.current[0] = await state.store.resolve_external_identity(
        provider="test", subject="group-other", email="group-other@example.com", display_name="Other",
        role="user", auth_method="test",
    )
    assert (await state.client.post(state.path + "/actions", json=body)).status_code == 404


async def test_browser_actions_cannot_append_model_data(state):
    await _create(state)
    response = await state.client.post(state.path + "/actions", json={"operation": "propose_fact", "expected_revision": 1, "data": {"label": "Unreviewed", "value": "1", "kind": "user_statement"}})
    assert response.status_code == 422


async def test_evidence_is_reread_from_actual_pdf_and_year_remains_proposed(state):
    await _create(state)
    written = await _call(state, "record_evidence", expected_revision=1, data=_evidence(state))
    assert "error" not in written, written
    case = (await state.client.get(state.path)).json()["case"]
    entry = case["evidence"][0]
    assert entry["passage"] == "Dated official supporting passage."
    assert entry["tax_year"] == 2025 and entry["tax_year_status"] == "proposed"
    rule = next(item for item in case["checklist"]["checks"] if item["id"] == "rule_evidence")
    assert rule["status"] == "needs_information"
    # Model-facing coverage keeps unresolved work without repeating empty checks.
    model_checks = written["checklist"]["checks"]
    assert any(item["id"] == "rule_evidence" for item in model_checks)
    assert all(item["issue_count"] > 0 for item in model_checks)
    assert {item["id"] for item in model_checks} == {
        item["id"] for item in case["checklist"]["checks"] if item["issues"]
    }
    # Compaction must not turn a bookkeeping warning into verified applicability.
    inspected = await _call(state, section="evidence")
    assert inspected["entries"][0]["tax_year_status"] == "proposed"


@pytest.mark.parametrize("changes", [
    {"passage": "Fabricated rule passage."}, {"revision": "b" * 64},
    {"document_id": "c" * 64}, {"catalog": "prior"}, {"reference_release": "rel_other"},
    {"page": 2}, {"offset": 3},
])
async def test_fabricated_or_unpinned_reference_evidence_cannot_be_saved(state, changes):
    await _create(state)
    result = await _call(state, "record_evidence", expected_revision=1, data=_evidence(state, **changes))
    assert result["status"] == 422, result
    assert (await state.client.get(state.path)).json()["case"]["evidence"] == []


async def test_evidence_model_cannot_set_verified_applicability(state):
    await _create(state)
    with pytest.raises(PrivateToolArgumentsError):
        await _call(state, "record_evidence", expected_revision=1, data=_evidence(state, tax_year_status="verified"))


async def test_arithmetic_is_computed_from_named_decimal_inputs(state):
    await _create(state)
    result = await _call(state, "record_calculation", expected_revision=1, data={"label": "Difference", "inputs": {"wages": "100.025", "deduction": "20.00"}, "expression": "wages - deduction", "rounding": "half_up_cent"})
    assert result["updated_entry"]["result"] == "80.03"
    stored = (await state.client.get(state.path)).json()["case"]["calculations"][0]
    assert stored["result"] == "80.03" and stored["expression"] == "wages - deduction"


@pytest.mark.parametrize("changes", [
    {"inputs": {"wages": 100.0}}, {"inputs": {"wages": "NaN"}},
    {"inputs": {"wages": "1e101"}}, {"expression": "wages + 15"},
    {"expression": "__import__('os')"}, {"expression": "100"}, {"result": "999"},
    {"rounding": "tax-law-approved"},
])
async def test_calculation_forgery_unrecorded_constants_and_invalid_decimals_refused(state, changes):
    await _create(state)
    data = {"label": "Calculation", "inputs": {"wages": "100"}, "expression": "wages", "rounding": "none", **changes}
    try:
        result = await _call(state, "record_calculation", expected_revision=1, data=data)
    except PrivateToolArgumentsError:
        pass
    else:
        assert result["status"] == 422, result
    assert (await state.client.get(state.path)).json()["case"]["calculations"] == []


async def test_release_change_preserves_read_but_blocks_model_work_without_repinning(state):
    await _create(state)
    replacement = begin(state.releases, docs=[state.metadata], fingerprint="e" * 64)
    checkpoint(state.releases, replacement, state.metadata)
    state.releases.publish(replacement["id"], verified_chunks=1)
    response = await state.client.get(state.path)
    assert response.status_code == 200
    assert response.json()["case"]["reference_update"] == {
        "status": "update_available", "release_id": replacement["id"],
    }
    result = await _call(state, "add_missing_item", expected_revision=1, data={"label": "Gap"})
    assert result["status"] == 409
    saved = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id)
    assert saved["reference_release"] == state.release["id"] and saved["revision"] == 1


async def test_callback_rechecks_active_account_purge_and_exact_policy(state):
    await _create(state)
    with processing_scope(state.policy):
        handler = make_tax_review_handler(app=state.app, principal=state.owner, conversation_id=state.conversation.conversation_id)
    with pytest.raises(ProcessingPolicyError):
        await handler({"operation": "inspect"})
    block_remote_personal_reads(user=state.owner.storage_namespace, purge_id="test-purge")
    try:
        assert (await _call(state))["status"] == 403
    finally:
        unblock_remote_personal_reads(user=state.owner.storage_namespace, purge_id="test-purge")
    with state.store._lock:
        state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        state.store._conn.commit()
    assert (await _call(state))["status"] == 403


async def test_inspect_pages_every_entry_and_has_bounded_valid_json(state):
    case = await _create(state)
    for number in range(12):
        case = await state.store.tax_cases.update(user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id, expected_revision=case["revision"], operation="propose_fact", data={"label": f"Fact {number}", "value": "value " * 650, "kind": "user_statement"})
    offset = 0
    seen = []
    while offset is not None:
        result = await _call(state, offset=offset, limit=5)
        assert len(json.dumps(result, ensure_ascii=False)) <= 6000
        assert result["total"] == 12
        seen.extend(entry["id"] for entry in result["entries"])
        offset = result["next_offset"]
    assert len(set(seen)) == 12


def test_tool_spec_is_in_process_owner_free_and_contains_typed_operation_fields():
    spec = tax_review_spec("http://custom-tools:8001/")
    serialized = json.dumps(spec.parameters)
    assert spec.in_process_only and not spec.user_scoped and spec.timeout_s == 150
    assert "$ref" not in serialized
    assert not {"user", "owner", "case_id", "conversation_id"}.intersection(spec.parameters["properties"])
    data = spec.parameters["properties"]["data"]["properties"]
    assert "provenance" in data and "inputs" in data and "passage" in data
    assert "result" not in data and "tax_year_status" not in data


async def test_single_large_dependency_record_never_breaks_json_or_paging(state):
    case = await _create(state)
    facts = [{"id": "fact_" + str(index).zfill(32), "status": "confirmed", "kind": "user_statement", "label": "Fact", "value": "1"} for index in range(100)]
    evidence = [{"id": "evidence_" + str(index).zfill(32), "tax_year": None, "tax_year_status": "unknown", "jurisdiction": "US-federal"} for index in range(100)]
    case.update(facts=facts, evidence=evidence, return_items=[{"id": "return_" + "1" * 32, "label": "Large dependencies", "status": "proposed", "fact_ids": [item["id"] for item in facts], "evidence_ids": [item["id"] for item in evidence]}])
    from audrey.tax_review import _tool_result

    result = _tool_result(case, operation="inspect", section="return_items")
    encoded = json.dumps(result, ensure_ascii=False)
    assert len(encoded) <= 6000
    assert json.loads(encoded)["entries"][0]["id"] == case["return_items"][0]["id"]
    assert result["returned"] == 1 and result["next_offset"] is None
    assert "fact_ids" in result["entries"][0]["truncated_fields"]


@pytest.mark.parametrize("input_count,key_length", [(20, 40), (100, 200)])
async def test_maximum_input_maps_and_long_dependencies_have_valid_bounded_pages(state, input_count, key_length):
    from audrey.app_state.tax_cases import RecordCalculationPayload
    from audrey.tax_review import _tool_result

    case = await _create(state)
    dependencies = [str(index).zfill(200) for index in range(100)]
    payload = RecordCalculationPayload.model_validate({
        "label": "Calculation with bounded inspection",
        "inputs": {f"input_{index}_".ljust(key_length, "x"): "0." + "1" * 158 for index in range(input_count)},
        "expression": "1", "rounding": "none", "result": "1",
        "fact_ids": dependencies, "evidence_ids": dependencies,
    }).model_dump(mode="json")
    case["calculations"] = [{"id": f"tax_calculation_{index}", **payload} for index in range(2)]
    seen = []
    offset = 0
    while offset is not None:
        result = _tool_result(case, operation="inspect", section="calculations", offset=offset, limit=5)
        encoded = json.dumps(result, ensure_ascii=False)
        assert len(encoded) <= 6000
        restored = json.loads(encoded)
        assert restored["returned"] == len(restored["entries"])
        assert len(restored["entries"]) == 1
        record = restored["entries"][0]
        assert {"inputs", "fact_ids", "evidence_ids"}.issubset(record["truncated_fields"])
        assert "notice" in restored
        seen.append(record["id"])
        offset = restored["next_offset"]
    assert seen == ["tax_calculation_0", "tax_calculation_1"]


async def test_owner_year_action_preserves_release_and_model_cannot_change_year(state):
    case = await _create(state)
    changed = await state.client.post(state.path + "/actions", json={
        "operation": "change_year", "expected_revision": case["revision"], "data": {"tax_year": 2024},
    })
    assert changed.status_code == 200, changed.text
    updated = changed.json()
    assert updated["tax_year"] == 2024 and updated["reference_release"] == case["reference_release"]
    same = await state.client.post(state.path + "/actions", json={
        "operation": "change_year", "expected_revision": updated["revision"], "data": {"tax_year": 2024},
    })
    assert same.status_code == 200 and same.json()["revision"] == updated["revision"]
    stale = await state.client.post(state.path + "/actions", json={
        "operation": "change_year", "expected_revision": case["revision"], "data": {"tax_year": 2023},
    })
    assert stale.status_code == 409
    with pytest.raises(PrivateToolArgumentsError):
        await _call(state, "change_year", expected_revision=updated["revision"], data={"tax_year": 2023})
    assert "change_year" not in tax_review_spec("http://custom-tools:8001").parameters["properties"]["operation"]["enum"]


async def test_user_can_reject_saved_fact_but_model_cannot(state):
    case = await _create(state)
    proposed = await _call(state, "propose_fact", expected_revision=case["revision"], data={"label": "Filing status", "value": "single", "kind": "user_statement"})
    fact_id = proposed["updated_entry"]["id"]
    rejected = await state.client.post(state.path + "/actions", json={
        "operation": "reject_fact", "expected_revision": proposed["revision"], "data": {"fact_id": fact_id},
    })
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["facts"][0]["status"] == "rejected"
    assert (await _call(state))["entries"][0]["status"] == "rejected"
    with pytest.raises(PrivateToolArgumentsError):
        await _call(state, "reject_fact", expected_revision=rejected.json()["revision"], data={"fact_id": fact_id})


def _publish_replacement(state, fingerprint="e" * 64):
    replacement = begin(state.releases, docs=[state.metadata], fingerprint=fingerprint)
    checkpoint(state.releases, replacement, state.metadata)
    state.releases.publish(replacement["id"], verified_chunks=1)
    return replacement


async def _adopt(state, release_id, revision):
    return await state.client.post(state.path + "/actions", json={
        "operation": "update_reference_release", "expected_revision": revision,
        "data": {"reference_release": release_id},
    })


async def test_explicit_release_update_keeps_confirmed_facts_and_rechecks_rule_work(state):
    case = await _create(state)
    proposed = await _call(state, "propose_fact", expected_revision=case["revision"], data={
        "label": "Filing status", "value": "single", "kind": "user_statement",
    })
    response = await state.client.post(state.path + "/actions", json={
        "operation": "confirm_fact", "expected_revision": proposed["revision"],
        "data": {"fact_id": proposed["updated_entry"]["id"]},
    })
    case = response.json()
    evidence = await _call(state, "record_evidence", expected_revision=case["revision"], data=_evidence(state))
    calculated = await _call(state, "record_calculation", expected_revision=evidence["revision"], data={
        "label": "Synthetic arithmetic", "inputs": {"wages": "10", "deduction": "2"},
        "expression": "wages - deduction", "rounding": "none",
        "fact_ids": [case["facts"][0]["id"]], "evidence_ids": [evidence["updated_entry"]["id"]],
    })
    replacement = _publish_replacement(state)
    updated = await _adopt(state, replacement["id"], calculated["revision"])
    assert updated.status_code == 200, updated.text
    saved = updated.json()
    assert saved["reference_release"] == replacement["id"]
    assert saved["reference_update"] == {"status": "current"}
    assert saved["facts"][0] == case["facts"][0]
    assert saved["evidence"][0]["reference_release"] == state.release["id"]
    assert saved["evidence"][0]["passage"] == _evidence(state)["passage"]
    assert saved["evidence"][0]["status"] == "needs_recheck"
    assert saved["calculations"][0]["result"] == "8"
    assert saved["calculations"][0]["status"] == "needs_recheck"
    assert "error" not in await _call(state)
    refreshed = await _call(state, "record_evidence", expected_revision=saved["revision"], data=_evidence(
        state, reference_release=replacement["id"], supersedes=[saved["evidence"][0]["id"]],
    ))
    assert "error" not in refreshed, refreshed
    assert refreshed["updated_entry"]["id"] != saved["evidence"][0]["id"]


async def test_release_update_requires_current_target_revision_and_owner(state):
    case = await _create(state)
    replacement = _publish_replacement(state)
    for target in (state.release["id"], "rel_forged"):
        response = await _adopt(state, target, case["revision"])
        assert response.status_code == 409, response.text
    assert (await _adopt(state, replacement["id"], case["revision"] + 1)).status_code == 409
    saved = await state.store.tax_cases.get(user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id)
    assert saved["reference_release"] == state.release["id"] and saved["revision"] == case["revision"]
    response = await _adopt(state, replacement["id"], case["revision"])
    assert response.status_code == 200
    same = await _adopt(state, replacement["id"], response.json()["revision"])
    assert same.status_code == 200 and same.json()["revision"] == response.json()["revision"]
    state.current[0] = await state.store.resolve_external_identity(
        provider="test", subject="foreign-release", email="foreign@example.com", display_name="Other",
        role="user", auth_method="test",
    )
    assert (await state.client.get(state.path)).status_code == 404
    assert (await _adopt(state, replacement["id"], same.json()["revision"])).status_code == 404


async def test_reference_outage_preserves_saved_view_and_blocks_generation_and_updates(state):
    from audrey.tax_review import require_tax_case

    case = await _create(state)
    state.app.state.reference_releases = Mock(published=Mock(return_value=None))
    response = await state.client.get(state.path)
    assert response.status_code == 200
    assert response.json()["case"]["reference_update"] == {"status": "unavailable"}
    assert response.json()["case"]["reference_release"] == case["reference_release"]
    with pytest.raises(HTTPException) as failure:
        await require_tax_case(state.app, state.owner, state.conversation.conversation_id)
    assert failure.value.status_code == 503
    assert (await _adopt(state, case["reference_release"], case["revision"])).status_code == 503
    assert (await _call(state))["status"] == 503


async def test_publication_race_keeps_only_approved_pin_and_exposes_new_choice(state, monkeypatch):
    from audrey.tax_review import require_tax_case

    case = await _create(state)
    approved = _publish_replacement(state)
    original = state.store.tax_cases.update
    latest = {}

    async def update_then_publish(**fields):
        result = await original(**fields)
        latest.update(_publish_replacement(state, fingerprint="f" * 64))
        return result

    monkeypatch.setattr(state.store.tax_cases, "update", update_then_publish)
    response = await _adopt(state, approved["id"], case["revision"])
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["reference_release"] == approved["id"]
    assert saved["reference_update"] == {"status": "update_available", "release_id": latest["id"]}
    with pytest.raises(HTTPException) as failure:
        await require_tax_case(state.app, state.owner, state.conversation.conversation_id)
    assert failure.value.status_code == 409


async def test_model_cannot_update_reference_release(state):
    case = await _create(state)
    with pytest.raises(PrivateToolArgumentsError):
        await _call(state, "update_reference_release", expected_revision=case["revision"], data={
            "reference_release": case["reference_release"],
        })
    assert "update_reference_release" not in tax_review_spec("http://custom-tools:8001").parameters["properties"]["operation"]["enum"]


async def test_saved_review_is_owned_and_invalidated_after_fact_confirmation(state):
    case = await _create(state)
    fact = await _call(state, "propose_fact", expected_revision=case["revision"], data={
        "label": "Wages", "value": "50000", "kind": "user_statement",
    })
    result = await _call(state, "save_review", expected_revision=fact["revision"], data={
        "summary": "Wages need confirmation before calculation.", "findings": [{
            "label": "Proposed wages", "detail": "The stated wages remain a proposal.",
            "fact_ids": [fact["updated_entry"]["id"]],
        }], "next_question": "Are the wages correct?",
    })
    assert "error" not in result, result
    assert result["review"]["status"] == "needs_input" and not result["review"]["stale"]
    review = (await state.client.get(state.path)).json()["case"]["review"]
    assert review["summary"] == "Wages need confirmation before calculation."
    assert review["limitations"] and "Selected passages only" in review["document_scope"]
    confirmed = await state.client.post(state.path + "/actions", json={
        "operation": "confirm_fact", "expected_revision": result["revision"],
        "data": {"fact_id": fact["updated_entry"]["id"]},
    })
    assert confirmed.status_code == 200 and confirmed.json()["review"]["stale"]
    invalid = await _call(state, "save_review", expected_revision=confirmed.json()["revision"], data={
        "summary": "Forged support", "findings": [{"label": "Foreign", "detail": "Unsupported", "fact_ids": ["foreign_id"]}],
    })
    assert invalid["status"] == 422
    assert (await state.client.get(state.path)).json()["case"]["review"]["summary"] == review["summary"]


async def test_inspect_lists_canonical_selected_documents_before_extraction(state):
    from unittest.mock import AsyncMock

    from audrey.app_state.records import ProjectFileRecord

    original = state.store.projects.snapshot_for_conversation
    state.store.projects.snapshot_for_conversation = AsyncMock(return_value=(None, [ProjectFileRecord(project_id="project", file_id="selected_id", added_at="2026-10-21")]))
    try:
        await _create(state)
        result = await _call(state, section="files")
    finally:
        state.store.projects.snapshot_for_conversation = original
    assert result["entries"] == [{"file_id": "selected_id", "selected_via": "project"}]
    assert result["case"]["selected_file_count"] == 1
    assert "listing a file does not establish" in result["case"]["document_scope"]


def test_review_schema_exposes_saved_worksheet_without_certification():
    spec = tax_review_spec("http://custom-tools:8001")
    assert "save_review" in spec.parameters["properties"]["operation"]["enum"]
    properties = spec.parameters["properties"]["data"]["properties"]
    assert {"summary", "findings", "next_question"}.issubset(properties)
    assert not {"verified", "complete", "correct"}.intersection(properties)


async def _native_statement(state, content):
    with processing_scope(state.policy):
        started = await state.store.conversations.begin_run(
            user_id=state.owner.user_id, conversation_id=state.conversation.conversation_id,
            user_content=content, expected_private_required=True,
        )
    assert started is not None
    return {"user_id": state.owner.user_id,
            "conversation_id": state.conversation.conversation_id,
            "run_id": started.run.run_id,
            "message_id": started.user_message.message_id,
            "content": started.user_message.content}


async def test_current_chat_fact_is_saved_without_repeat_confirmation(state):
    from audrey.tax_review import user_statement_scope

    case = await _create(state)
    statement = await _native_statement(state, "For 2025, my W-2 wages are $50,000.")
    with processing_scope(state.policy), user_statement_scope(**statement):
        handler = make_tax_review_handler(app=state.app, principal=state.owner,
                                         conversation_id=state.conversation.conversation_id)
        result = await handler({"operation": "record_user_fact", "expected_revision": case["revision"],
                                "data": {"label": "W-2 wages", "value": "my W-2 wages are $50,000"}})
    assert result["updated_entry"]["status"] == "confirmed"
    saved = (await state.client.get(state.path)).json()["case"]
    assert saved["facts"][0]["value"] == "my W-2 wages are $50,000"
    assert saved["facts"][0]["confirmed_by"] == "user_message"
    assert saved["facts"][0]["source_message_id"] == statement["message_id"]
    assert saved["facts"][0]["confirmed_tax_year"] == 2025
    assert saved["evidence"] == []  # User statements never establish official law.
    duplicate = None
    with processing_scope(state.policy), user_statement_scope(**statement):
        duplicate = await handler({"operation": "record_user_fact", "expected_revision": saved["revision"],
                                   "data": {"label": "W-2 wages", "value": "my W-2 wages are $50,000"}})
    assert duplicate["revision"] == saved["revision"]


@pytest.mark.parametrize("change", [
    {"user_id": "other-owner"}, {"conversation_id": "other-conversation"},
    {"message_id": "forged-message"}, {"run_id": "forged-run"},
    {"content": "I earned $75,000"},
])
async def test_direct_chat_fact_rejects_wrong_bound_source(state, change):
    from audrey.tax_review import user_statement_scope

    case = await _create(state)
    statement = await _native_statement(state, "My wages are $50,000")
    statement.update(change)
    with processing_scope(state.policy), user_statement_scope(**statement):
        result = await _call(state, "record_user_fact", expected_revision=case["revision"],
                             data={"label": "Wages", "value": "$50,000"})
    assert result["status"] == 422
    assert (await state.client.get(state.path)).json()["case"]["facts"] == []


async def test_missing_exited_or_nonverbatim_message_cannot_confirm_fact(state):
    import contextvars

    from audrey.tax_review import user_statement_scope

    case = await _create(state)
    statement = await _native_statement(state, "My W-2 says $50,000")
    with processing_scope(state.policy):
        handler = make_tax_review_handler(app=state.app, principal=state.owner,
                                         conversation_id=state.conversation.conversation_id)
        payload = {"operation": "record_user_fact", "expected_revision": case["revision"],
                   "data": {"label": "Wages", "value": "$50,000"}}
        assert (await handler(payload))["status"] == 422
        with user_statement_scope(**statement):
            copied = contextvars.copy_context()
            normalized = {**payload, "data": {"label": "Wages", "value": "50000.00"}}
            assert (await handler(normalized))["status"] == 422
        assert (await handler(payload))["status"] == 422
        # Even a task carrying a copied ContextVar cannot outlive the source scope.
        import asyncio
        assert (await asyncio.create_task(handler(payload), context=copied))["status"] == 422
    assert (await state.client.get(state.path)).json()["case"]["facts"] == []


async def test_model_cannot_forge_message_authority_in_tool_arguments(state):
    await _create(state)
    with pytest.raises(PrivateToolArgumentsError):
        await _call(state, "record_user_fact", expected_revision=1,
                    data={"label": "Wages", "value": "$50,000", "source_message_id": "forged"})
    assert "edit_fact" not in tax_review_spec("http://custom-tools:8001").parameters["properties"]["operation"]["enum"]


async def test_owner_edits_current_fact_in_place_and_browser_hides_audit(state):
    case = await _create(state)
    proposed = await _call(state, "propose_fact", expected_revision=case["revision"],
                           data={"label": "Wages", "value": "$50,000", "kind": "user_statement"})
    fact_id = proposed["updated_entry"]["id"]
    response = await state.client.post(state.path + "/actions", json={
        "operation": "edit_fact", "expected_revision": proposed["revision"],
        "data": {"fact_id": fact_id, "label": "2025 W-2 wages", "value": "$52,000"},
    })
    assert response.status_code == 200, response.text
    case = response.json()
    assert len(case["facts"]) == 1 and case["facts"][0]["id"] == fact_id
    assert case["facts"][0]["value"] == "$52,000" and case["facts"][0]["status"] == "confirmed"
    assert "history" not in case["facts"][0]
    stored = await state.store.tax_cases.get(user_id=state.owner.user_id,
                                            conversation_id=state.conversation.conversation_id)
    assert stored["facts"][0]["history"][0]["value"] == "$50,000"
    replay = await state.client.post(state.path + "/actions", json={
        "operation": "edit_fact", "expected_revision": proposed["revision"],
        "data": {"fact_id": fact_id, "label": "Wages", "value": "$1"},
    })
    assert replay.status_code == 409


async def test_answered_missing_question_resolves_only_with_current_quoted_fact(state):
    from audrey.tax_review import user_statement_scope

    case = await _create(state)
    gap = await _call(state, "add_missing_item", expected_revision=case["revision"], data={
        "label": "Filer age", "question": "How old were you at the end of 2025?"})
    saved = (await state.client.get(state.path)).json()["case"]
    assert saved["missing_items"][0]["question"] == "How old were you at the end of 2025?"
    statement = await _native_statement(state, "I was 40 at the end of 2025")
    with processing_scope(state.policy), user_statement_scope(**statement):
        result = await _call(state, "record_user_fact", expected_revision=gap["revision"], data={
            "label": "Age at year-end", "value": "40 at the end of 2025",
            "addresses": [gap["updated_entry"]["id"]]})
    assert "error" not in result
    saved = (await state.client.get(state.path)).json()["case"]
    assert saved["missing_items"][0]["status"] == "resolved"
    assert saved["missing_items"][0]["resolved_with_fact_id"] == saved["facts"][0]["id"]


async def test_chat_correction_updates_same_row_and_returns_id_with_answered_gap(state):
    from audrey.tax_review import user_statement_scope

    case = await _create(state)
    proposed = await _call(state, "propose_fact", expected_revision=case["revision"],
                           data={"label": "Wages", "value": "$50,000", "kind": "user_statement"})
    fact_id = proposed["updated_entry"]["id"]
    gap = await _call(state, "add_missing_item", expected_revision=proposed["revision"],
                     data={"label": "Corrected wages", "question": "What are your corrected 2025 wages?"})
    statement = await _native_statement(state, "Correction: my 2025 wages were $52,000.")
    with processing_scope(state.policy), user_statement_scope(**statement):
        result = await _call(state, "record_user_fact", expected_revision=gap["revision"], data={
            "label": "Wages", "value": "my 2025 wages were $52,000", "fact_id": fact_id,
            "addresses": [gap["updated_entry"]["id"]]})
    assert result["updated_entry"] == {"id": fact_id, "status": "confirmed"}
    saved = (await state.client.get(state.path)).json()["case"]
    assert len(saved["facts"]) == 1 and saved["facts"][0]["id"] == fact_id
    assert saved["missing_items"][0]["status"] == "resolved"
    assert "history" not in saved["facts"][0]
    # Changed values require an explicit existing ID, rather than another active row.
    with processing_scope(state.policy), user_statement_scope(**statement):
        refused = await _call(state, "record_user_fact", expected_revision=saved["revision"],
                              data={"label": "Wages", "value": "$52,000"})
    assert refused["status"] == 422
    assert len((await state.client.get(state.path)).json()["case"]["facts"]) == 1


async def test_previous_native_message_cannot_supply_current_fact(state):
    from audrey.tax_review import user_statement_scope

    case = await _create(state)
    old = await _native_statement(state, "My old wages were $50,000")
    await state.store.conversations.finish_run(user_id=state.owner.user_id,
                                              run_id=old["run_id"], outcome="succeeded", assistant_content="Saved")
    await _native_statement(state, "My current wages are $52,000")
    with processing_scope(state.policy), user_statement_scope(**old):
        refused = await _call(state, "record_user_fact", expected_revision=case["revision"],
                              data={"label": "Wages", "value": "$50,000"})
    assert refused["status"] == 422
    assert (await state.client.get(state.path)).json()["case"]["facts"] == []


async def test_quote_cannot_resolve_missing_item_from_other_tax_year(state):
    from audrey.tax_review import user_statement_scope

    case = await _create(state)
    gap = await _call(state, "add_missing_item", expected_revision=case["revision"],
                     data={"label": "Age", "question": "How old were you at the end of 2025?"})
    changed = await state.client.post(state.path + "/actions", json={
        "operation": "change_year", "expected_revision": gap["revision"], "data": {"tax_year": 2024}})
    assert changed.status_code == 200
    statement = await _native_statement(state, "I was 39 at the end of 2024")
    with processing_scope(state.policy), user_statement_scope(**statement):
        refused = await _call(state, "record_user_fact", expected_revision=changed.json()["revision"], data={
            "label": "Age", "value": "39 at the end of 2024", "addresses": [gap["updated_entry"]["id"]]})
    assert refused["status"] == 422
    saved = (await state.client.get(state.path)).json()["case"]
    assert saved["facts"] == [] and saved["missing_items"][0]["status"] == "unresolved"
