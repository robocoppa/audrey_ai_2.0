"""Trusted worksheet assessments stay bound to one saved candidate and revision."""

from __future__ import annotations

import copy

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.tax_cases import (
    SaveReviewPayload,
    TaxCaseAccessError,
    TaxCaseConflictError,
    TaxCaseError,
    _apply_operation,
    tax_case_review,
    tax_review_scope,
    validate_operation,
)
from audrey.tax_verification import candidate_digest


@pytest.fixture
async def context(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(
        provider="test",
        subject="review-owner",
        email="owner@example.com",
        display_name="Owner",
        role="user",
        auth_method="test",
    )
    conversation = await store.conversations.create(
        user_id=owner.user_id,
        private_required=True,
    )
    case = await store.tax_cases.create(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        tax_year=2025,
        goal="Check the linked filing-status finding",
        reference_release="release-current",
    )
    kwargs = {"user_id": owner.user_id, "conversation_id": conversation.conversation_id}
    case = await store.tax_cases.update(
        **kwargs,
        expected_revision=case["revision"],
        operation="propose_fact",
        data={"label": "Filing status", "value": "single", "kind": "user_statement"},
    )
    case = await store.tax_cases.update(
        **kwargs,
        expected_revision=case["revision"],
        operation="confirm_fact",
        actor="user",
        data={"fact_id": case["facts"][0]["id"]},
    )
    case = await store.tax_cases.update(
        **kwargs,
        expected_revision=case["revision"],
        operation="record_evidence",
        actor="user",
        data={
            "claim": "Selected rule passage",
            "document_id": "pub-17",
            "revision": "2025",
            "reference_release": "release-current",
            "tax_year": 2025,
            "tax_year_status": "verified",
            "page": 1,
            "passage": "An official retained passage for this synthetic worksheet.",
        },
    )
    payload = validate_operation(
        "save_review",
        {
            "summary": "The linked facts and selected rule passage support this limited finding.",
            "findings": [
                {
                    "label": "Filing-status record",
                    "detail": "The saved status is single.",
                    "fact_ids": [case["facts"][0]["id"]],
                    "evidence_ids": [case["evidence"][0]["id"]],
                }
            ],
        },
    )
    yield store, kwargs, case, payload
    store.close()


def assessment(case, payload, **changes):
    return {
        "candidate_sha256": candidate_digest(payload),
        "case_revision": case["revision"],
        "attempt": 1,
        "status": "checked",
        "stop_reason": "checks_complete",
        "findings": [
            {
                "finding_index": 0,
                "verdict": "supported",
                "reason": "The retained passage supports the scoped finding.",
            }
        ],
        "model": "local-checker",
        **changes,
    }


async def save(context, *, case=None, payload=None, report=None, actor="model"):
    store, kwargs, original, original_payload = context
    case = original if case is None else case
    payload = original_payload if payload is None else payload
    return await store.tax_cases.update(
        **kwargs,
        expected_revision=case["revision"],
        operation="save_review",
        data=payload,
        actor=actor,
        verification=assessment(case, payload) if report is None else report,
    )


async def test_assessment_and_worksheet_commit_together_and_survive_reopen(context):
    store, kwargs, original, payload = context
    report = assessment(original, payload)
    current = await save(context, report=report)
    assert current["revision"] == original["revision"] + 1
    assert current["review"]["verification"] == report
    assert current["review"]["summary"] == payload["summary"]
    assert tax_case_review(current)["status"] == "review_saved"
    assert tax_case_review(current)["verification"]["stale"] is False
    view = tax_case_review(current)
    assert {key: view[key] for key in tax_review_scope()} == tax_review_scope()
    assert view["eligibility_status"] == "not_determined"
    # Completed consistency checks do not expand the capability or ask for
    # another confirmation of the already supplied facts.
    assert view["status"] == "review_saved" and view["limitations"] == []
    assert view["next_question"] == ""
    assert current["facts"] == original["facts"]
    assert current["evidence"] == original["evidence"]
    assert "scope" not in current["review"]  # Projection, not a data migration.
    reopened = ApplicationStore(store.path)
    try:
        restored = await reopened.tax_cases.get(**kwargs)
        assert restored == current
        assert tax_case_review(restored) == view
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_sha256": "0" * 64},
        {"case_revision": 999},
    ],
)
async def test_forged_candidate_or_revision_is_rejected_without_partial_save(context, change):
    store, kwargs, case, payload = context
    with pytest.raises(TaxCaseError, match="does not match"):
        await save(context, report=assessment(case, payload, **change))
    assert await store.tax_cases.get(**kwargs) == case


@pytest.mark.parametrize(
    "findings",
    [
        [],
        [{"finding_index": 1, "verdict": "insufficient", "reason": "Wrong candidate index."}],
        [
            {"finding_index": 0, "verdict": "supported", "reason": "Supported."},
            {"finding_index": 1, "verdict": "insufficient", "reason": "Extra candidate index."},
        ],
    ],
)
async def test_assessment_must_cover_every_candidate_finding_exactly(context, findings):
    store, kwargs, case, payload = context
    with pytest.raises(TaxCaseError):
        await save(
            context,
            report=assessment(
                case,
                payload,
                status="limited",
                stop_reason="evidence_gap",
                findings=findings,
            ),
        )
    assert await store.tax_cases.get(**kwargs) == case


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_sha256": "bad"},
        {"case_revision": True},
        {"attempt": 0},
        {"attempt": 3},
        {"status": "approved"},
        {"stop_reason": "tax_correct"},
        {"model": "x" * 201},
        {"findings": [{"finding_index": 12, "verdict": "supported", "reason": "Checked."}]},
        {"findings": [{"finding_index": 0, "verdict": "correct", "reason": "Checked."}]},
        {"findings": [{"finding_index": 0, "verdict": "supported", "reason": "x" * 501}]},
        {
            "findings": [
                {"finding_index": 0, "verdict": "supported", "reason": "Supported."},
                {"finding_index": 0, "verdict": "supported", "reason": "Duplicate."},
            ]
        },
        {"untrusted": True},
    ],
)
async def test_malformed_assessment_is_rejected_atomically(context, change):
    store, kwargs, case, payload = context
    with pytest.raises(TaxCaseError, match="verification report"):
        await save(context, report=assessment(case, payload, **change))
    assert await store.tax_cases.get(**kwargs) == case


async def test_assessment_cannot_be_attached_to_other_operation_or_user_action(context):
    store, kwargs, case, payload = context
    report = assessment(case, payload)
    for operation, data, actor in (
        ("add_missing_item", {"label": "Additional document"}, "model"),
        ("save_review", payload, "user"),
    ):
        with pytest.raises(TaxCaseError, match="Only a model worksheet"):
            await store.tax_cases.update(
                **kwargs,
                expected_revision=case["revision"],
                operation=operation,
                data=data,
                actor=actor,
                verification=report,
            )
    assert await store.tax_cases.get(**kwargs) == case


@pytest.mark.parametrize("field,value", [
    ("verification", {"status": "checked"}),
    ("scope", "eligibility"),
    ("eligibility_status", "determined"),
    ("notice", "Eligibility approved."),
])
def test_model_payload_cannot_inject_assessment_or_scope(field, value):
    assert field not in SaveReviewPayload.model_json_schema()["properties"]
    with pytest.raises(TaxCaseError, match="Invalid save_review"):
        validate_operation(
            "save_review",
            {
                "summary": "Untrusted model worksheet",
                "findings": [],
                field: value,
            },
        )


async def test_historical_and_stale_review_cannot_claim_expanded_scope(context):
    current = await save(context)
    historical = copy.deepcopy(current)
    historical["review"].pop("verification")
    historical["review"].update(
        scope="eligibility", eligibility_status="determined", notice="Eligibility approved.",
    )
    unchanged = copy.deepcopy(historical)
    view = tax_case_review(historical)
    assert {key: view[key] for key in tax_review_scope()} == tax_review_scope()
    assert historical == unchanged
    historical["revision"] += 1
    stale = tax_case_review(historical)
    assert stale["stale"] is True
    assert stale["eligibility_status"] == "not_determined"
    assert stale["summary"] == current["review"]["summary"]


async def test_foreign_owner_cannot_save_trusted_assessment(context):
    store, kwargs, case, payload = context
    other = await store.resolve_external_identity(
        provider="test",
        subject="other",
        email="other@example.com",
        display_name="Other",
        role="user",
        auth_method="test",
    )
    with pytest.raises(TaxCaseAccessError):
        await store.tax_cases.update(
            **{**kwargs, "user_id": other.user_id},
            expected_revision=case["revision"],
            operation="save_review",
            data=payload,
            verification=assessment(case, payload),
        )
    assert await store.tax_cases.get(**kwargs) == case


async def test_report_participates_in_noop_equality_and_new_assessment_advances_revision(context):
    _, _, case, payload = context
    report = assessment(case, payload)
    saved = await save(context, report=report)
    # The operation's no-op check compares the stored trusted report as well as
    # candidate text. The public repository separately enforces report/revision CAS.
    unchanged = copy.deepcopy(saved)
    assert _apply_operation(unchanged, "save_review", payload, "model", report) is False
    assert unchanged == saved
    changed_report = {**report, "model": "another-local-checker"}
    assert _apply_operation(unchanged, "save_review", payload, "model", changed_report) is True
    refreshed = await save(context, case=saved)
    assert refreshed["revision"] == saved["revision"] + 1
    assert refreshed["review"]["verification"]["case_revision"] == saved["revision"]
    with pytest.raises(TaxCaseConflictError):
        await save(context, report=report)


@pytest.mark.parametrize(
    "operation,data,actor",
    [
        ("add_missing_item", {"label": "Additional taxpayer statement"}, "model"),
        ("change_year", {"tax_year": 2024}, "user"),
        ("update_reference_release", {"reference_release": "release-new"}, "user"),
    ],
)
async def test_edits_preserve_assessment_but_project_it_as_stale(context, operation, data, actor):
    store, kwargs, _, _ = context
    saved = await save(context)
    retained = copy.deepcopy(saved["review"])
    changed = await store.tax_cases.update(
        **kwargs,
        expected_revision=saved["revision"],
        operation=operation,
        data=data,
        actor=actor,
    )
    assert changed["review"] == retained
    view = tax_case_review(changed)
    assert view["stale"] is True and view["verification"]["stale"] is True
    assert "stale" not in changed["review"]["verification"]
    assert view["status"] in {"needs_input", "limited_evidence"}


async def test_limited_check_is_visible_and_needs_input_still_dominates(context):
    store, kwargs, case, payload = context
    report = assessment(
        case,
        payload,
        status="limited",
        stop_reason="evidence_gap",
        findings=[
            {
                "finding_index": 0,
                "verdict": "insufficient",
                "reason": "The passage lacks this amount.",
            }
        ],
    )
    current = await save(context, report=report)
    view = tax_case_review(current)
    assert view["status"] == "limited_evidence"
    assert any("lack sufficient saved evidence" in item for item in view["limitations"])
    assert "Finding 1: The passage lacks this amount." in view["limitations"]
    changed = await store.tax_cases.update(
        **kwargs,
        expected_revision=current["revision"],
        operation="add_missing_item",
        data={"label": "Confirm the amount"},
    )
    assert tax_case_review(changed)["status"] == "needs_input"


async def test_check_does_not_promote_year_applicability_or_fact_confirmation(context):
    store, kwargs, case, payload = context
    # Retain a real applicability gap and proposed fact; a model assessment may
    # assess its candidate, never turn these records into verified/confirmed data.
    case = await store.tax_cases.update(
        **kwargs,
        expected_revision=case["revision"],
        operation="record_evidence",
        data={
            "claim": "An uncertain rule",
            "document_id": "pub-unknown",
            "revision": "undated",
            "reference_release": "release-current",
            "page": 1,
            "passage": "Unknown year.",
        },
    )
    case = await store.tax_cases.update(
        **kwargs,
        expected_revision=case["revision"],
        operation="propose_fact",
        data={"label": "Unconfirmed amount", "value": "10", "kind": "user_statement"},
    )
    payload = {
        **payload,
        "findings": [
            {**payload["findings"][0], "evidence_ids": [item["id"] for item in case["evidence"]]}
        ],
    }
    current = await save(context, case=case, payload=payload)
    assert current["evidence"][-1]["tax_year_status"] == "unknown"
    assert current["evidence"][-1]["tax_year"] is None
    assert current["facts"][-1]["status"] == "proposed"
    assert tax_case_review(current)["status"] == "needs_input"
    assert any("Tax-year applicability" in item for item in tax_case_review(current)["limitations"])
