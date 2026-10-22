"""Saved worksheet feedback keeps internal coverage separate from user questions."""
from __future__ import annotations

from test_tax_review import _call, _create, _evidence
from test_tax_review import state as state

from audrey.tax_review import tax_review_spec


async def _confirmed_fact(state):
    case = await _create(state)
    proposed = await _call(state, "propose_fact", expected_revision=case["revision"], data={
        "label": "Wages", "value": "50000", "kind": "user_statement",
    })
    fact_id = proposed["updated_entry"]["id"]
    response = await state.client.post(state.path + "/actions", json={
        "operation": "confirm_fact", "expected_revision": proposed["revision"],
        "data": {"fact_id": fact_id},
    })
    assert response.status_code == 200, response.text
    return response.json(), fact_id


async def test_saved_result_separates_applicability_limit_from_a_user_question(state):
    case, fact_id = await _confirmed_fact(state)
    written = await _call(state, "record_evidence", expected_revision=case["revision"], data=_evidence(state))
    result = await _call(state, "save_review", expected_revision=written["revision"], data={
        "summary": "A scoped result with retained support.",
        "findings": [{"label": "Scoped finding", "detail": "The selected support is retained.",
                      "fact_ids": [fact_id], "evidence_ids": [written["updated_entry"]["id"]]}],
        "next_question": "",
    })
    assert result["review"]["status"] == "limited_evidence"
    guidance = result["response_guidance"]
    assert "derived needs_input or limited_evidence status alone does not create a question" in guidance
    assert "do not say that catalog metadata makes the passage undated" in guidance
    assert "Keep revisions, record IDs" in guidance and "click Continue review" in guidance
    saved = (await state.client.get(state.path)).json()["case"]
    assert saved["review"]["next_question"] == ""
    assert saved["evidence"][0]["tax_year_status"] == "proposed"
    assert any("has not been verified" in item for item in saved["review"]["limitations"])


async def test_saved_user_fact_question_is_retained_with_response_guidance(state):
    case = await _create(state)
    proposed = await _call(state, "propose_fact", expected_revision=case["revision"], data={
        "label": "Wages", "value": "50000", "kind": "user_statement",
    })
    result = await _call(state, "save_review", expected_revision=proposed["revision"], data={
        "summary": "Wages need confirmation before use.",
        "findings": [{"label": "Proposed wages", "detail": "The user's amount needs confirmation.",
                      "fact_ids": [proposed["updated_entry"]["id"]]}],
        "next_question": "Are these wages correct?",
    })
    assert result["review"]["status"] == "needs_input"
    assert "Group proposed facts in the review controls" in result["response_guidance"]
    saved = (await state.client.get(state.path)).json()["case"]
    assert saved["review"]["next_question"] == "Are these wages correct?"
    assert saved["facts"][0]["status"] == "proposed"


async def test_guidance_does_not_resolve_or_discard_existing_missing_items(state):
    case, fact_id = await _confirmed_fact(state)
    missing = await _call(state, "add_missing_item", expected_revision=case["revision"], data={
        "label": "Check conflicting supplied records",
    })
    before = (await state.client.get(state.path)).json()["case"]["missing_items"]
    result = await _call(state, "save_review", expected_revision=missing["revision"], data={
        "summary": "The useful partial result is retained; a source conflict remains open.",
        "findings": [{"label": "Saved wages", "detail": "The supplied wages are confirmed.",
                      "fact_ids": [fact_id]}],
        "next_question": "",
    })
    assert result["review"]["status"] == "needs_input"
    assert "Retain substantive missing information and any material source conflict" in result["response_guidance"]
    saved = (await state.client.get(state.path)).json()["case"]
    assert saved["missing_items"] == before
    assert saved["review"]["next_question"] == ""


def test_next_question_schema_limits_questions_to_needed_user_facts_or_choices():
    properties = tax_review_spec("http://custom-tools:8001").parameters["properties"]["data"]["properties"]
    assert "otherwise leave empty" in properties["next_question"]["description"]
    assert "not user decisions" in properties["next_question"]["description"]
