"""Source-grounded local worksheet checks fail explicitly and keep task authority."""
from __future__ import annotations

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from audrey.app_state.private_tasks import PrivateTaskConflictError
from audrey.models.ollama import GenerationUsageError, OllamaClient, generation_budget
from audrey.processing_policy import ProcessingPolicy, ProcessingPolicyError, processing_scope
from audrey.tax_verification import ReviewVerification, candidate_digest, verify_review

_MODEL = "local-chat:latest"
_POLICY = ProcessingPolicy({_MODEL: "a" * 64}, "http://ollama:11434",
                           allowed_tools=frozenset(), tool_servers=frozenset())


def _data():
    fact = {"id": "fact1", "label": "Income", "value": "50000", "status": "confirmed",
            "kind": "user_statement", "case_tax_year": 2025}
    evidence = {"id": "rule1", "claim": "Standard deduction", "document_id": "b" * 64,
                "revision": "c" * 64, "reference_release": "release1", "catalog": "current",
                "tax_year": 2025, "tax_year_status": "proposed", "jurisdiction": "US-federal",
                "page": 1, "offset": 0, "passage": "The deduction is 15750 for 2025.",
                "status": "recorded"}
    calculation = {"id": "calc1", "label": "Remainder", "status": "recorded",
                   "inputs": {"income": "50000", "deduction": "15750"},
                   "expression": "income - deduction", "rounding": "none", "result": "34250",
                   "fact_ids": ["fact1"], "evidence_ids": ["rule1"], "case_tax_year": 2025}
    case = {"id": "case1", "revision": 8, "goal": "Subtract deduction from income",
            "tax_year": 2025, "jurisdiction": "US-federal", "reference_release": "release1",
            "facts": [fact], "evidence": [evidence], "calculations": [calculation],
            "return_items": [], "missing_items": []}
    candidate = {"summary": "Income less deduction is 34250.", "next_question": "",
                 "findings": [{"label": "Remainder", "detail": "50000 minus 15750 is 34250.",
                               "fact_ids": [], "evidence_ids": [], "calculation_ids": ["calc1"],
                               "return_item_ids": []}]}
    return case, candidate


def _assessment(verdict="supported", **changes):
    return {"finding_index": 0, "verdict": verdict,
            "reason": "Pure arithmetic uses the supplied amounts and year; no conditional rule is applied.",
            "conditions": [], **changes}


def _app(reply=None, *, chat_error=None):
    acquired = []

    @asynccontextmanager
    async def acquire(model, **kwargs):
        acquired.append((model, kwargs))
        yield

    response = reply if reply is not None else {"findings": [_assessment()]}
    chat = AsyncMock(side_effect=chat_error, return_value={"done": True, "eval_count": 5,
                     "message": {"content": response if isinstance(response, str) else json.dumps(response)}})
    return SimpleNamespace(state=SimpleNamespace(
        cfg=SimpleNamespace(raw={"model_registry": {"general": [{"name": _MODEL, "priority": 10}]}}),
        gate=SimpleNamespace(acquire=acquire), ollama=SimpleNamespace(chat=chat),
    )), acquired


async def _verify(case=None, candidate=None, *, reply=None, reread=None, error=None, attempt=1):
    default_case, default_candidate = _data()
    case = default_case if case is None else case
    candidate = default_candidate if candidate is None else candidate
    app, acquired = _app(reply, chat_error=error)
    reader = reread if reread is not None else AsyncMock(side_effect=lambda item: dict(item))
    with processing_scope(_POLICY):
        result = await verify_review(app, case, candidate, reread=reader, attempt=attempt, user_id="owner")
    return result, app.state.ollama.chat, reader, acquired


async def test_reopens_transitive_support_and_reviews_exact_summary_and_finding():
    case, candidate = _data()
    original = copy.deepcopy((case, candidate))
    result, chat, reader, acquired = await _verify(case, candidate)
    assert result["status"] == "checked" and result["stop_reason"] == "checks_complete"
    assert result["candidate_sha256"] == candidate_digest(candidate)
    assert result["case_revision"] == 8 and result["attempt"] == 1
    reader.assert_awaited_once()
    assert reader.call_args.args[0]["id"] == "rule1"
    request = chat.call_args.kwargs
    assert request["tools"] is None and request["think"] is False
    assert request["timeout_s"] <= 90 and request["options"]["num_predict"] <= 2048
    assert request["format"].get("$defs") is None
    supplied = json.loads(request["messages"][1]["content"])
    assert supplied["candidate"] == candidate and supplied["support"][0]["evidence"][0]["passage"] == case["evidence"][0]["passage"]
    assert "SUMMARY AND EVERY FINDING" in request["messages"][0]["content"]
    assert acquired == [(_MODEL, {"location": "local", "user_id": "owner"})]
    assert (case, candidate) == original and case["evidence"][0]["tax_year_status"] == "proposed"


@pytest.mark.parametrize("condition", ["no_findings", "proposed", "question", "missing"])
async def test_user_decisions_skip_reviewer_and_source_reads(condition):
    case, candidate = _data()
    if condition == "no_findings":
        candidate["findings"] = []
    elif condition == "proposed":
        case["facts"][0]["status"] = "proposed"
    elif condition == "question":
        candidate["next_question"] = "What is your filing status?"
    else:
        case["missing_items"] = [{"status": "unresolved"}]
    result, chat, reader, _ = await _verify(case, candidate)
    assert result["status"] == "limited"
    assert result["stop_reason"] == ("evidence_gap" if condition == "no_findings" else "needs_user_input")
    chat.assert_not_awaited()
    reader.assert_not_awaited()


@pytest.mark.parametrize("condition,reason", [
    ("wrong_year", "different return year"), ("conflict", "Conflicting"),
    ("stale", "stale or superseded"), ("rejected", "rejected"),
    ("wrong_calculation", "decimal inputs"), ("missing_support", "missing"),
    ("foreign_jurisdiction", "jurisdiction"), ("stale_fact", "another return year"),
])
async def test_deterministic_support_failures_cannot_be_overruled_by_reviewer(condition, reason):
    case, candidate = _data()
    if condition == "wrong_year":
        case["evidence"][0]["tax_year"] = 2024
    elif condition == "conflict":
        case["evidence"][0]["conflicts_with"] = ["other_rule"]
    elif condition == "stale":
        case["calculations"][0]["status"] = "needs_recheck"
    elif condition == "rejected":
        case["facts"][0]["status"] = "rejected"
    elif condition == "wrong_calculation":
        case["calculations"][0]["result"] = "35000"
    elif condition == "missing_support":
        case["evidence"] = []
    elif condition == "foreign_jurisdiction":
        case["evidence"][0]["jurisdiction"] = "foreign"
    else:
        case["facts"][0]["case_tax_year"] = 2024
    result, chat, reader, _ = await _verify(case, candidate)
    assert result["status"] == "needs_repair" and result["findings"][0]["verdict"] == "unsupported"
    assert reason in result["findings"][0]["reason"]
    chat.assert_not_awaited()
    reader.assert_not_awaited()


@pytest.mark.parametrize("reply", [
    {"findings": []}, {"findings": [_assessment(), _assessment()]},
    {"findings": [_assessment(finding_index=1)]},
    {"findings": [_assessment(finding_index="0")]},
    {"findings": [_assessment(reason="r" * 501)]},
    {"findings": [_assessment()], "verified": True},
    "not JSON", "{" * 1000,
])
async def test_invalid_reviewer_results_are_limited_not_supported(reply):
    result, *_ = await _verify(reply=reply)
    assert result["status"] == "limited" and result["stop_reason"] == "reviewer_unavailable"
    assert all(finding["verdict"] == "insufficient" for finding in result["findings"])


@pytest.mark.parametrize("wrap", [lambda s: s, lambda s: "```json\n" + s + "\n```", lambda _s: json.dumps([_assessment()])])
async def test_strict_findings_accept_observed_json_wrappers(wrap):
    result, *_ = await _verify(reply=wrap(json.dumps({"findings": [_assessment()]})))
    assert result["status"] == "checked"


async def test_false_claim_and_insufficient_evidence_trigger_targeted_feedback():
    for verdict in ("unsupported", "insufficient"):
        result, *_ = await _verify(reply={"findings": [_assessment(verdict, reason="The claim adds a bonus the passage does not state.")]})
        assert result["status"] == "needs_repair"
        assert result["findings"][0]["verdict"] == verdict
        assert "bonus" in result["findings"][0]["reason"]


async def test_source_read_failure_or_substitution_is_explicitly_incomplete():
    for reader in (AsyncMock(side_effect=HTTPException(422, "missing page")),
                   AsyncMock(return_value={"passage": "substituted"})):
        result, chat, *_ = await _verify(reread=reader)
        assert result["status"] == "limited" and result["stop_reason"] == "evidence_gap"
        chat.assert_not_awaited()


@pytest.mark.parametrize("error", [PrivateTaskConflictError("changed"), GenerationUsageError("usage"), asyncio.CancelledError()])
async def test_live_task_usage_and_cancellation_refusals_propagate(error):
    with pytest.raises(type(error)):
        await _verify(error=error)


async def test_source_reader_live_authority_refusal_propagates():
    with pytest.raises(PrivateTaskConflictError):
        await _verify(reread=AsyncMock(side_effect=PrivateTaskConflictError("owner changed")))


async def test_full_input_bound_refuses_before_reading_or_generating():
    case, candidate = _data()
    # No field is truncated into a weaker claim or evidence set.
    case["facts"][0]["value"] = "x" * 25000
    result, chat, reader, _ = await _verify(case, candidate)
    assert result["status"] == "limited" and result["stop_reason"] == "verification_limit"
    chat.assert_not_awaited()
    reader.assert_not_awaited()


async def test_processing_scope_required_even_for_empty_review():
    case, candidate = _data()
    candidate["findings"] = []
    app, _ = _app()
    with pytest.raises(ProcessingPolicyError):
        await verify_review(app, case, candidate, reread=AsyncMock(), attempt=1, user_id="owner")


def test_report_rejects_false_checked_or_duplicate_findings_and_digest_tracks_semantics():
    case, candidate = _data()
    original = candidate_digest(candidate)
    candidate["findings"][0]["detail"] += " unsupported bonus"
    assert candidate_digest(candidate) != original
    report = {"candidate_sha256": original, "case_revision": case["revision"], "attempt": 1,
              "status": "checked", "stop_reason": "checks_complete", "findings": [_assessment()]}
    assert ReviewVerification.model_validate(report).model == ""
    for findings in ([], [_assessment("insufficient")], [_assessment(), _assessment()]):
        with pytest.raises(ValidationError):
            ReviewVerification.model_validate({**report, "findings": findings})


async def test_reviewer_uses_real_client_scoped_generation_accounting():
    requests, charged = [], []

    def handle(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": _MODEL, "digest": "a" * 64}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {"format": "gguf"}, "model_info": {
                "general.architecture": "qwen", "general.parameter_count": 100}})
        return httpx.Response(200, json={"done": True, "eval_count": 7,
               "message": {"content": json.dumps({"findings": [_assessment()]})}})

    app, _ = _app()
    client = OllamaClient("http://ollama:11434", transport=httpx.MockTransport(handle), trust_env=False)
    app.state.ollama = client
    case, candidate = _data()
    try:
        with processing_scope(_POLICY), generation_budget(AsyncMock(return_value=20), AsyncMock(side_effect=charged.append)):
            result = await verify_review(app, case, candidate, reread=AsyncMock(side_effect=lambda item: item),
                                         attempt=2, user_id="owner")
    finally:
        await client.aclose()
    assert result["status"] == "checked" and result["attempt"] == 2 and charged == [7]
    request = next(item for item in requests if item.url.path == "/api/chat")
    assert json.loads(request.content)["options"]["num_predict"] == 20


@pytest.mark.parametrize("terminal", [{"done": False}, {"done": True, "done_reason": "length"}])
async def test_valid_json_from_incomplete_generation_is_never_accepted(terminal):
    app, _ = _app()
    app.state.ollama.chat.return_value = {**terminal, "message": {
        "content": json.dumps({"findings": [_assessment()]})}}
    case, candidate = _data()
    with processing_scope(_POLICY):
        result = await verify_review(app, case, candidate, reread=AsyncMock(side_effect=lambda item: item),
                                     attempt=1, user_id="owner")
    assert result["status"] == "limited" and result["stop_reason"] == "reviewer_unavailable"


async def test_distinct_source_bound_is_checked_before_any_reread():
    case, candidate = _data()
    template = case["evidence"][0]
    case["evidence"] = [{**template, "id": f"rule{index}"} for index in range(13)]
    case["calculations"][0]["evidence_ids"] = [entry["id"] for entry in case["evidence"]]
    result, chat, reader, _ = await _verify(case, candidate)
    assert result["status"] == "limited" and result["stop_reason"] == "verification_limit"
    reader.assert_not_awaited()
    chat.assert_not_awaited()


async def test_shared_source_is_reopened_once_and_every_finding_is_assessed():
    case, candidate = _data()
    candidate["findings"].append({**candidate["findings"][0], "label": "Same worksheet support"})
    result, _, reader, _ = await _verify(case, candidate,
        reply={"findings": [_assessment(), _assessment(finding_index=1)]})
    assert result["status"] == "checked"
    assert [finding["finding_index"] for finding in result["findings"]] == [0, 1]
    reader.assert_awaited_once()



def _condition(status="met", **changes):
    value = {"condition": "The supplied profile confirms the source's required condition.",
             "status": status, "fact_ids": ["fact1"], "evidence_ids": ["rule1"], **changes}
    value.setdefault("fact_paths", ["" for _ in value["fact_ids"]])
    return value


async def test_new_reviewer_reply_must_explicitly_inventory_conditions():
    assessment = _assessment()
    del assessment["conditions"]
    result, *_ = await _verify(reply={"findings": [assessment]})
    assert result["status"] == "limited"
    assert result["stop_reason"] == "reviewer_unavailable"


def test_saved_assessments_without_condition_inventory_remain_readable():
    case, candidate = _data()
    assessment = _assessment()
    del assessment["conditions"]
    report = ReviewVerification.model_validate({"candidate_sha256": candidate_digest(candidate),
        "case_revision": case["revision"], "attempt": 1, "status": "checked",
        "stop_reason": "checks_complete", "findings": [assessment]})
    assert report.findings[0].conditions == []


@pytest.mark.parametrize("value", [False, 0, "present"])
async def test_explicit_false_zero_and_present_values_are_not_missing(value):
    case, candidate = _data()
    case["facts"][0]["value"] = value
    result, *_ = await _verify(case, candidate,
        reply={"findings": [_assessment(conditions=[_condition()])]})
    assert result["status"] == "checked"
    assert result["findings"][0]["conditions"][0]["status"] == "met"
    assert case["evidence"][0]["tax_year_status"] == "proposed"


@pytest.mark.parametrize("condition,verdict", [
    (_condition("unknown", fact_ids=[]), "insufficient"),
    (_condition("not_met"), "unsupported"),
    (_condition(fact_ids=[]), "insufficient"),
    (_condition(fact_ids=["another-owner-fact"]), "insufficient"),
    (_condition(evidence_ids=[]), "insufficient"),
    (_condition(evidence_ids=["unlinked-source"]), "insufficient"),
    (_condition(fact_ids=["fact1", "fact1"]), "insufficient"),
    (_condition(evidence_ids=["rule1", "rule1"]), "insufficient"),
])
async def test_declared_condition_gaps_cannot_be_overruled_by_supported_verdict(condition, verdict):
    result, chat, *_ = await _verify(reply={"findings": [_assessment(conditions=[condition])]})
    assert result["status"] == "needs_repair"
    assert result["findings"][0]["verdict"] == verdict
    assert result["findings"][0]["conditions"] == [condition]
    chat.assert_awaited_once()


@pytest.mark.parametrize("change", [
    {"value": None}, {"value": ""}, {"value": {}}, {"status": "recorded"},
    {"confirmed_tax_year": 2024},
])
async def test_condition_support_requires_current_confirmed_non_null_fact(change):
    case, candidate = _data()
    case["facts"][0].update(change)
    result, *_ = await _verify(case, candidate,
        reply={"findings": [_assessment(conditions=[_condition()])]})
    assert result["status"] == "needs_repair"
    assert result["findings"][0]["verdict"] == "insufficient"
    assert "current confirmed fact" in result["findings"][0]["reason"]


async def test_condition_cannot_borrow_a_fact_from_outside_this_finding_support():
    case, candidate = _data()
    case["facts"].append({**case["facts"][0], "id": "unlinked-fact"})
    result, *_ = await _verify(case, candidate, reply={"findings": [
        _assessment(conditions=[_condition(fact_ids=["unlinked-fact"])])]})
    assert result["status"] == "needs_repair"
    assert result["findings"][0]["verdict"] == "insufficient"


async def test_condition_check_uses_source_rules_and_actual_fields_without_tax_field_list():
    case, candidate = _data()
    case["facts"][0]["value"] = {"completed_training": False}
    case["evidence"][0]["passage"] = "Only applicants who have not completed training qualify."
    condition = _condition(condition="completed_training is explicitly false, satisfying the source condition",
                           fact_paths=["/completed_training"])
    result, chat, *_ = await _verify(case, candidate,
        reply={"findings": [_assessment(conditions=[condition])]})
    assert result["status"] == "checked"
    instructions = chat.call_args.kwargs["messages"][0]["content"]
    assert "Include source footnotes" in instructions
    assert "Missing/null does not mean false or zero" in instructions
    assert "source metadata" in instructions


def test_checked_report_rejects_unresolved_condition_even_if_verdict_claims_supported():
    case, candidate = _data()
    for status in ("unknown", "not_met"):
        with pytest.raises(ValidationError):
            ReviewVerification.model_validate({"candidate_sha256": candidate_digest(candidate),
                "case_revision": case["revision"], "attempt": 1, "status": "checked",
                "stop_reason": "checks_complete", "findings": [
                    _assessment(conditions=[_condition(status)])]})


@pytest.mark.parametrize("value,path,expected", [
    ({"age": 40}, "/age", "checked"),
    ({"blind": False}, "/blind", "checked"),
    ({"profile": {"score": 0}}, "/profile/score", "checked"),
    ({"members": [{"eligible": False}]}, "/members/0/eligible", "checked"),
    ({"a/b": {"c~d": False}}, "/a~1b/c~0d", "checked"),
    ({"filing_status": "single"}, "/age", "needs_repair"),
    ({"age": None}, "/age", "needs_repair"),
    ({"age": ""}, "/age", "needs_repair"),
    ({"age": 40}, "", "needs_repair"),
    ({"profile": {"age": 40}}, "/profile", "needs_repair"),
    ({"members": [40]}, "/members/1", "needs_repair"),
    ({"members": [40]}, "/members/00", "needs_repair"),
    ({"members": [40]}, "/members/-1", "needs_repair"),
    ({"age": 40}, "age", "needs_repair"),
    ({"a~b": 40}, "/a~b", "needs_repair"),
])
async def test_conditions_resolve_actual_scalar_fields_in_composite_confirmed_profile(value, path, expected):
    case, candidate = _data()
    case["facts"][0]["value"] = value
    result, *_ = await _verify(case, candidate, reply={"findings": [
        _assessment(conditions=[_condition(fact_paths=[path])])]})
    assert result["status"] == expected
    if expected == "needs_repair":
        assert result["findings"][0]["verdict"] == "insufficient"


async def test_parallel_fact_paths_cannot_omit_the_field_reference():
    result, *_ = await _verify(reply={"findings": [
        _assessment(conditions=[_condition(fact_paths=[])])]})
    assert result["status"] == "needs_repair"
    assert result["findings"][0]["verdict"] == "insufficient"



@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("mode", ["python", "json"])
def test_condition_presence_survives_exact_report_roundtrips(explicit, mode):
    case, candidate = _data()
    assessment = _assessment()
    if not explicit:
        del assessment["conditions"]
    wire = {"candidate_sha256": candidate_digest(candidate), "case_revision": case["revision"],
            "attempt": 1, "status": "checked", "stop_reason": "checks_complete",
            "findings": [assessment], "model": "local-checker"}
    report = ReviewVerification.model_validate(wire)
    assert report.findings[0].conditions == []
    # This is the representation retained and compared by the atomic save.
    stored = report.model_dump(mode=mode)
    assert stored == wire
    assert ReviewVerification.model_validate(stored).model_dump(mode=mode) == stored
    assert ReviewVerification.model_validate_json(report.model_dump_json()).model_dump(mode=mode) == stored
    assert ("conditions" in stored["findings"][0]) is explicit


async def test_new_reviewer_explicit_empty_conditions_remain_on_saved_report_wire():
    result, *_ = await _verify(reply={"findings": [_assessment(conditions=[])]})
    assert result["findings"][0]["conditions"] == []
    assert ReviewVerification.model_validate(result).model_dump(mode="json") == result
