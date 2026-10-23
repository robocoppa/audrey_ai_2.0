"""Private tax cases persist bounded proposals with owner and revision authority."""

from __future__ import annotations

import asyncio
import copy
import sqlite3

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state.tax_cases import (
    TaxCaseAccessError,
    TaxCaseConflictError,
    TaxCaseError,
    tax_case_checklist,
    tax_case_review,
    validate_operation,
    validate_tax_case,
)


async def _owner(store, subject="alice"):
    return await store.resolve_external_identity(
        provider="test",
        subject=subject,
        email=f"{subject}@example.com",
        display_name=subject.title(),
        role="user",
        auth_method="test",
    )


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _owner(store)
    project = await store.projects.create(
        user_id=owner.user_id, name="Tax review", private_required=True
    )
    conversation = await store.conversations.create(
        user_id=owner.user_id, project_id=project.project_id
    )
    yield store, owner, project, conversation
    store.close()


async def _create(state, **changes):
    store, owner, _, conversation = state
    payload = {
        "tax_year": 2025,
        "jurisdiction": "US-federal",
        "goal": "Review planning records",
        "reference_release": "rel_current",
    }
    payload.update(changes)
    return await store.tax_cases.create(
        user_id=owner.user_id, conversation_id=conversation.conversation_id, **payload
    )


async def _update(state, case, operation, data, actor="model"):
    store, owner, _, conversation = state
    return await store.tax_cases.update(
        user_id=owner.user_id,
        conversation_id=conversation.conversation_id,
        expected_revision=case["revision"],
        operation=operation,
        data=data,
        actor=actor,
    )


def _fact(**changes):
    return {"label": "Filing status", "value": "single", "kind": "user_statement", **changes}


def _evidence(**changes):
    return {
        "claim": "A dated rule must be checked",
        "document_id": "pub-17",
        "revision": "2025",
        "reference_release": "rel_current",
        "tax_year": 2025,
        "page": 3,
        "passage": "This is a test passage retained from an official publication.",
        **changes,
    }


async def test_schema23_case_survives_reopen_and_conversation_is_single_case(state):
    store, owner, project, conversation = state
    assert (
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
        is None
    )
    created = await _create(state)
    assert store.schema_version == 24
    assert created["project_id"] == project.project_id
    assert created["conversation_id"] == conversation.conversation_id
    assert created["revision"] == 1 and created["checklist_version"] == 1
    assert created["facts"] == created["evidence"] == created["calculations"] == []
    with pytest.raises(TaxCaseConflictError):
        await _create(state)
    reopened = ApplicationStore(store.path)
    try:
        assert (
            await reopened.tax_cases.get(
                user_id=owner.user_id, conversation_id=conversation.conversation_id
            )
            == created
        )
        assert reopened._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close()


async def test_owner_and_private_context_are_rechecked_for_every_read_and_write(state):
    store, owner, _, conversation = state
    case = await _create(state)
    other = await _owner(store, "other")
    ordinary = await store.conversations.create(user_id=owner.user_id)
    for kwargs in (
        {"user_id": other.user_id, "conversation_id": conversation.conversation_id},
        {"user_id": owner.user_id, "conversation_id": ordinary.conversation_id},
        {"user_id": owner.user_id, "conversation_id": "missing"},
    ):
        with pytest.raises(TaxCaseAccessError):
            await store.tax_cases.get(**kwargs)
        with pytest.raises(TaxCaseAccessError):
            await store.tax_cases.create(
                **kwargs, tax_year=2025, goal="Review", reference_release="rel_current"
            )
        with pytest.raises(TaxCaseAccessError):
            await store.tax_cases.update(
                **kwargs, expected_revision=case["revision"], operation="propose_fact", data=_fact()
            )
    with store._lock:
        store._conn.execute(
            "UPDATE app_users SET status='disabled' WHERE user_id=?", (owner.user_id,)
        )
        store._conn.commit()
    with pytest.raises(TaxCaseAccessError):
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
    with pytest.raises(TaxCaseAccessError):
        await _update(state, case, "propose_fact", _fact())


@pytest.mark.parametrize("blocked_state", ["archived", "deletion_pending"])
async def test_canonical_archive_and_pending_deletion_block_reads_and_writes(state, blocked_state):
    store, owner, _, conversation = state
    case = await _create(state)
    with store._lock:
        if blocked_state == "archived":
            store._conn.execute(
                "UPDATE app_conversations SET archived_at=? WHERE conversation_id=?",
                ("2026-10-20T00:00:00+00:00", conversation.conversation_id),
            )
        else:
            # Even a partially recovered account still marked active must not
            # accept new private work while its durable deletion is pending.
            store._conn.execute(
                "INSERT INTO account_deletion_requests (user_id, purge_id, storage_namespace, requested_at) VALUES (?, ?, ?, ?)",
                (
                    owner.user_id,
                    "purge-pending",
                    owner.storage_namespace,
                    "2026-10-20T00:00:00+00:00",
                ),
            )
        store._conn.commit()
    with pytest.raises(TaxCaseAccessError):
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
    with pytest.raises(TaxCaseAccessError):
        await _update(state, case, "propose_fact", _fact())
    with pytest.raises(TaxCaseAccessError):
        await _create(state)


async def test_canonical_project_is_derived_and_stays_private(state):
    store, owner, project, conversation = state
    case = await _create(state)
    ordinary = await store.projects.create(user_id=owner.user_id, name="Ordinary")
    with store._lock:
        store._conn.execute(
            "UPDATE app_conversations SET project_id=? WHERE conversation_id=?",
            (ordinary.project_id, conversation.conversation_id),
        )
        store._conn.commit()
    with pytest.raises(TaxCaseAccessError):
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
    with pytest.raises(TaxCaseAccessError):
        await _update(state, case, "propose_fact", _fact())
    with store._lock:
        store._conn.execute(
            "UPDATE app_conversations SET project_id=? WHERE conversation_id=?",
            (project.project_id, conversation.conversation_id),
        )
        store._conn.commit()
    assert await store.projects.delete(user_id=owner.user_id, project_id=project.project_id)
    retained = await store.tax_cases.get(
        user_id=owner.user_id, conversation_id=conversation.conversation_id
    )
    assert retained["project_id"] is None
    retained = await _update(state, retained, "propose_fact", _fact())
    assert retained["project_id"] is None and retained["facts"]


async def test_revision_prevents_lost_updates_across_independent_connections(state):
    store, owner, _, conversation = state
    case = await _create(state)
    second = ApplicationStore(store.path)
    try:
        results = await asyncio.gather(
            *(
                repository.update(
                    user_id=owner.user_id,
                    conversation_id=conversation.conversation_id,
                    expected_revision=case["revision"],
                    operation="add_missing_item",
                    data={"label": label},
                )
                for repository, label in (
                    (store.tax_cases, "First document"),
                    (second.tax_cases, "Second document"),
                )
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(result, TaxCaseConflictError) for result in results) == 1
        current = await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
        assert current["revision"] == 2 and len(current["missing_items"]) == 1
    finally:
        second.close()


async def test_only_user_actions_confirm_facts_and_resolve_missing_records(state):
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact(), actor="user")
    fact = case["facts"][0]
    assert fact["status"] == "proposed"
    with pytest.raises(TaxCaseError, match="user action"):
        await _update(state, case, "confirm_fact", {"fact_id": fact["id"]})
    case = await _update(state, case, "confirm_fact", {"fact_id": fact["id"]}, actor="user")
    assert case["facts"][0]["status"] == "confirmed"
    case = await _update(state, case, "add_missing_item", {"label": "Read the missing statement"})
    item = case["missing_items"][0]
    with pytest.raises(TaxCaseError, match="user action"):
        await _update(state, case, "resolve_missing_item", {"id": item["id"]})
    case = await _update(state, case, "resolve_missing_item", {"id": item["id"]}, actor="user")
    assert case["missing_items"][0]["status"] == "resolved"


async def test_grouped_confirmation_is_user_only_one_revision_and_idempotent(state):
    case = await _create(state)
    for number in range(3):
        case = await _update(state, case, "propose_fact", _fact(label=f"Fact {number}"))
    identifiers = [fact["id"] for fact in case["facts"]]
    case = await _update(state, case, "confirm_fact", {"fact_id": identifiers[0]}, actor="user")
    already_confirmed = copy.deepcopy(case["facts"][0])
    data = {"fact_ids": identifiers}
    with pytest.raises(TaxCaseError, match="user action"):
        await _update(state, case, "confirm_facts", data)
    confirmed = await _update(state, case, "confirm_facts", data, actor="user")
    assert confirmed["revision"] == case["revision"] + 1
    assert confirmed["facts"][0] == already_confirmed
    assert all(fact["status"] == "confirmed" for fact in confirmed["facts"])
    assert all(fact["confirmed_by"] == "user" for fact in confirmed["facts"])
    assert all(fact["confirmed_tax_year"] == 2025 for fact in confirmed["facts"])
    assert confirmed["facts"][1]["confirmed_at"] == confirmed["facts"][2]["confirmed_at"]
    assert await _update(state, confirmed, "confirm_facts", data, actor="user") == confirmed
    with pytest.raises(TaxCaseConflictError):
        await _update(state, case, "confirm_facts", data, actor="user")


@pytest.mark.parametrize("invalid", ["missing", "foreign", "rejected"])
async def test_grouped_confirmation_validates_whole_selection_before_changes(state, invalid):
    store, owner, _, conversation = state
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact())
    identifier = "not_a_case_fact"
    if invalid == "foreign":
        other = await _owner(store, "other")
        other_conversation = await store.conversations.create(user_id=other.user_id, private_required=True)
        foreign = await store.tax_cases.create(
            user_id=other.user_id, conversation_id=other_conversation.conversation_id,
            tax_year=2025, goal="Other case", reference_release="rel_current",
        )
        foreign = await store.tax_cases.update(
            user_id=other.user_id, conversation_id=other_conversation.conversation_id,
            expected_revision=foreign["revision"], operation="propose_fact", data=_fact(),
        )
        identifier = foreign["facts"][0]["id"]
    elif invalid == "rejected":
        case = await _update(state, case, "propose_fact", _fact(label="Rejected fact"))
        identifier = case["facts"][1]["id"]
        case = await _update(state, case, "reject_fact", {"fact_id": identifier}, actor="user")
    with pytest.raises(TaxCaseError):
        await _update(state, case, "confirm_facts", {
            "fact_ids": [case["facts"][0]["id"], identifier],
        }, actor="user")
    assert await store.tax_cases.get(
        user_id=owner.user_id, conversation_id=conversation.conversation_id,
    ) == case


@pytest.mark.parametrize("identifiers", [[], ["same", "same"], ["same", " same "], [str(index) for index in range(101)], ["x" * 201], [1]])
def test_grouped_confirmation_ids_are_nonempty_unique_and_bounded(identifiers):
    with pytest.raises(TaxCaseError):
        validate_operation("confirm_facts", {"fact_ids": identifiers})


async def test_append_operations_are_idempotent_and_operation_keys_cannot_change_data(state):
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact(operation_key="first"))
    duplicate = await _update(state, case, "propose_fact", _fact(operation_key="first"))
    assert duplicate == case
    assert await _update(state, case, "propose_fact", _fact()) == case
    with pytest.raises(TaxCaseConflictError, match="operation key"):
        await _update(state, case, "propose_fact", _fact(operation_key="first", value="married"))
    assert len(case["facts"]) == 1


async def test_pinned_source_provenance_and_case_local_dependencies_are_required(state):
    case = await _create(
        state, files=[{"file_id": "file_1", "checksum": "a" * 64, "revision": "v1"}]
    )
    provenance = {
        "file_id": "file_1",
        "checksum": "a" * 64,
        "page": 1,
        "box": "1",
        "original_text": "100.00",
        "normalization": "100.00",
    }
    case = await _update(
        state, case, "propose_fact", _fact(kind="extracted", provenance=provenance)
    )
    assert case["facts"][0]["provenance"]["original_text"] == "100.00"
    with pytest.raises(TaxCaseError, match="pinned case file"):
        await _update(
            state,
            case,
            "propose_fact",
            _fact(kind="extracted", provenance={**provenance, "checksum": "b" * 64}),
        )
    case = await _update(
        state,
        case,
        "propose_fact",
        _fact(kind="extracted", provenance={**provenance, "file_id": "file_2"}),
    )
    assert case["files"][1]["file_id"] == "file_2"
    assert case["files"][1]["revision"] == "a" * 64
    assert case["files"][1]["review_scope"] == "selected_passages"
    assert any("identical content" in issue for issue in tax_case_checklist(case)["next_steps"])
    with pytest.raises(TaxCaseError):
        await _update(state, case, "propose_fact", _fact(kind="extracted"))
    with pytest.raises(TaxCaseError, match="dependency"):
        await _update(
            state,
            case,
            "propose_return_item",
            {"label": "Proposed item", "fact_ids": ["foreign-fact"]},
        )
    with pytest.raises(TaxCaseError, match="pinned case release"):
        await _update(
            state, case, "record_evidence", _evidence(reference_release="different_release")
        )


async def test_checklist_retains_unknown_year_and_requires_confirmed_dated_supported_evidence(
    state,
):
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact())
    with pytest.raises(TaxCaseError, match="cannot verify"):
        await _update(state, case, "record_evidence", _evidence(tax_year_status="verified"))
    case = await _update(state, case, "record_evidence", _evidence(tax_year=None))
    check = tax_case_checklist(case)
    assert check["status"] == "needs_information"
    assert any("unknown" in issue for issue in check["next_steps"])
    assert any("confirmation" in issue for issue in check["next_steps"])
    assert case["evidence"][0]["tax_year"] is None
    wrong = copy.deepcopy(case)
    wrong["evidence"][0]["tax_year"] = 2024
    assert any("different tax year" in issue for issue in tax_case_checklist(wrong)["next_steps"])
    unverified = copy.deepcopy(case)
    unverified["evidence"][0]["tax_year"] = 2025
    assert any(
        "requires verification" in issue for issue in tax_case_checklist(unverified)["next_steps"]
    )
    confirmed = await _update(
        state, case, "confirm_fact", {"fact_id": case["facts"][0]["id"]}, actor="user"
    )
    confirmed["evidence"][0].update(tax_year=2025, tax_year_status="verified")
    result = validate_tax_case(confirmed)
    assert result["status"] == "ready_for_review"
    assert "complete or legally correct" in result["notice"]
    assert (
        confirmed["return_items"] == []
    )  # Planning coverage does not require inventing return lines.
    supported = copy.deepcopy(confirmed)
    supported["evidence"][0]["passage"] = ""
    assert validate_tax_case(supported)["status"] == "needs_information"


async def test_calculations_and_proposed_return_items_keep_dependency_coverage(state):
    case = await _create(state)
    case = await _update(
        state,
        case,
        "record_calculation",
        {
            "label": "Sum",
            "inputs": {"a": "1.20", "b": "2.30"},
            "expression": "1.20 + 2.30",
            "rounding": "none",
            "result": "3.50",
        },
    )
    assert case["calculations"][0]["result"] == "3.50"
    assert any(
        "Link confirmed facts and rule evidence" in issue
        for issue in validate_tax_case(case)["next_steps"]
    )
    case = await _update(state, case, "propose_return_item", {"label": "Planning draft"})
    assert case["return_items"][0]["status"] == "proposed"
    with pytest.raises(TaxCaseError):
        await _update(
            state,
            case,
            "record_calculation",
            {
                "label": "Invalid",
                "inputs": {"x": "NaN"},
                "expression": "x",
                "rounding": "none",
                "result": "NaN",
            },
        )


async def test_case_limits_roll_back_without_advancing_revision(state):
    case = await _create(state)
    for index in range(100):
        case = await _update(
            state, case, "add_missing_item", {"label": f"Required document {index}"}
        )
    with pytest.raises(TaxCaseError, match="entry limit"):
        await _update(state, case, "add_missing_item", {"label": "One too many"})
    store, owner, _, conversation = state
    assert (
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
    )["revision"] == case["revision"]
    for index in range(100):
        try:
            case = await _update(
                state, case, "propose_fact", _fact(label=f"Fact {index}", value="x" * 4000)
            )
        except TaxCaseError as exc:
            assert "storage limit" in str(exc)
            break
    else:
        pytest.fail("The case byte bound was not enforced")
    persisted = await store.tax_cases.get(
        user_id=owner.user_id, conversation_id=conversation.conversation_id
    )
    assert persisted == case


async def test_case_cascades_on_conversation_delete_and_owner_data_purge(state):
    store, owner, _, conversation = state
    await _create(state)
    assert await store.conversations.delete(
        user_id=owner.user_id, conversation_id=conversation.conversation_id
    )
    assert store._conn.execute("SELECT count(*) FROM app_tax_cases").fetchone()[0] == 0
    new_conversation = await store.conversations.create(
        user_id=owner.user_id, private_required=True
    )
    await store.tax_cases.create(
        user_id=owner.user_id,
        conversation_id=new_conversation.conversation_id,
        tax_year=2025,
        goal="Review",
        reference_release="rel_current",
    )
    other = await _owner(store, "other")
    other_conversation = await store.conversations.create(
        user_id=other.user_id, private_required=True
    )
    await store.tax_cases.create(
        user_id=other.user_id,
        conversation_id=other_conversation.conversation_id,
        tax_year=2025,
        goal="Review",
        reference_release="rel_current",
    )
    await store.purge_local_user_data(user_id=owner.user_id)
    rows = store._conn.execute("SELECT user_id FROM app_tax_cases").fetchall()
    assert [row[0] for row in rows] == [other.user_id]
    with store._lock:
        store._conn.execute("DELETE FROM app_users WHERE user_id=?", (other.user_id,))
        store._conn.commit()
    assert store._conn.execute("SELECT count(*) FROM app_tax_cases").fetchone()[0] == 0


def test_operation_payloads_reject_unknown_fields_and_unbounded_identifiers():
    for operation, data in (
        (
            "propose_fact",
            _fact(
                provenance={
                    "file_id": "file_1",
                    "checksum": "a" * 64,
                    "page": 1,
                    "original_text": "A statement",
                }
            ),
        ),
        ("propose_fact", _fact(status="confirmed")),
        ("confirm_fact", {"fact_id": "x", "user_id": "other"}),
        ("propose_return_item", {"label": "x", "fact_ids": ["y" * 201]}),
        ("replace_case", {}),
    ):
        with pytest.raises(TaxCaseError):
            validate_operation(operation, data)


async def test_schema22_upgrades_additively_preserving_private_conversation(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _owner(store)
    conversation = await store.conversations.create(user_id=owner.user_id, private_required=True)
    store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE app_private_tasks")
        connection.execute("DROP TABLE app_tax_cases")
        connection.execute("DELETE FROM app_schema_migrations WHERE version IN (23, 24)")
        connection.commit()
    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 24
        retained = await upgraded.conversations.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
        assert retained.private_required
        assert (
            await upgraded.tax_cases.get(
                user_id=owner.user_id, conversation_id=conversation.conversation_id
            )
            is None
        )
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        upgraded.close()


async def test_year_change_is_explicit_user_only_revision_guarded_and_idempotent(state):
    case = await _create(state)
    with pytest.raises(TaxCaseError, match="user action"):
        await _update(state, case, "change_year", {"tax_year": 2024})
    unchanged = await _update(state, case, "change_year", {"tax_year": 2025}, actor="user")
    assert unchanged == case
    changed = await _update(state, case, "change_year", {"tax_year": 2024}, actor="user")
    assert changed["tax_year"] == 2024 and changed["revision"] == case["revision"] + 1
    assert changed["reference_release"] == case["reference_release"]
    with pytest.raises(TaxCaseConflictError):
        await _update(state, case, "change_year", {"tax_year": 2023}, actor="user")
    assert await _update(state, changed, "change_year", {"tax_year": 2024}, actor="user") == changed


@pytest.mark.parametrize("tax_year", [True, "2024", 1899, 2101, None])
async def test_year_change_rejects_implicit_or_out_of_range_values(state, tax_year):
    case = await _create(state)
    with pytest.raises(TaxCaseError):
        await _update(state, case, "change_year", {"tax_year": tax_year}, actor="user")


async def test_year_change_preserves_sources_and_requires_new_confirmation_and_calculation_review(
    state,
):
    case = await _create(state)
    fact_data = _fact(
        kind="extracted",
        value="100.00",
        provenance={
            "file_id": "file_1",
            "checksum": "a" * 64,
            "page": 1,
            "box": "1",
            "original_text": "100.00",
            "normalization": "100.00",
        },
    )
    case = await _update(state, case, "propose_fact", fact_data)
    fact_id = case["facts"][0]["id"]
    case = await _update(state, case, "confirm_fact", {"fact_id": fact_id}, actor="user")
    case = await _update(state, case, "record_evidence", _evidence())
    calc_data = {
        "label": "Original calculation",
        "inputs": {"income": "100.00"},
        "expression": "income",
        "rounding": "none",
        "result": "100",
        "fact_ids": [fact_id],
        "evidence_ids": [case["evidence"][0]["id"]],
    }
    case = await _update(state, case, "record_calculation", calc_data)
    case = await _update(
        state,
        case,
        "propose_return_item",
        {
            "label": "Original return item",
            "fact_ids": calc_data["fact_ids"],
            "evidence_ids": calc_data["evidence_ids"],
        },
    )
    original = copy.deepcopy(case)
    changed = await _update(state, case, "change_year", {"tax_year": 2024}, actor="user")
    assert changed["files"] == original["files"]
    assert changed["evidence"] == original["evidence"]
    assert changed["reference_release"] == original["reference_release"]
    fact = changed["facts"][0]
    assert fact["provenance"] == original["facts"][0]["provenance"]
    assert fact["case_tax_year"] == 2025 and fact["status"] == "proposed"
    assert not {"confirmed_at", "confirmed_by", "confirmed_tax_year"}.intersection(fact)
    for key in ("calculations", "return_items"):
        assert changed[key][0]["case_tax_year"] == 2025
        assert changed[key][0]["status"] == "needs_recheck"
        assert changed[key][0]["recheck_for_tax_year"] == 2024
    assert changed["calculations"][0]["result"] == original["calculations"][0]["result"]
    issues = tax_case_checklist(changed)["next_steps"]
    assert any("confirmation" in issue for issue in issues)
    assert any("different tax year" in issue for issue in issues)
    assert any("Recheck item" in issue for issue in issues)
    reconfirmed = await _update(state, changed, "confirm_fact", {"fact_id": fact_id}, actor="user")
    assert reconfirmed["facts"][0]["case_tax_year"] == 2025
    assert reconfirmed["facts"][0]["confirmed_tax_year"] == 2024
    assert reconfirmed["evidence"][0]["tax_year"] == 2025
    fresh = await _update(state, reconfirmed, "record_calculation", calc_data)
    assert len(fresh["calculations"]) == 2
    assert fresh["calculations"][-1]["case_tax_year"] == 2024
    assert fresh["calculations"][-1]["status"] == "recorded"
    assert fresh["calculations"][0]["superseded_by"] == fresh["calculations"][-1]["id"]
    assert fresh["calculations"][0]["case_tax_year"] == 2025
    assert not any(
        f"Recheck item {fresh['calculations'][0]['id']} " in issue
        for issue in tax_case_checklist(fresh)["next_steps"]
    )
    # A return to the original year still requires a fresh review; an old
    # same-year digest must not turn re-recording into a no-op.
    original_year = await _update(state, fresh, "change_year", {"tax_year": 2025}, actor="user")
    checked_again = await _update(state, original_year, "record_calculation", calc_data)
    assert len(checked_again["calculations"]) == 3
    assert checked_again["calculations"][-1]["case_tax_year"] == 2025
    assert all(
        item["superseded_by"] == checked_again["calculations"][-1]["id"]
        for item in checked_again["calculations"][:-1]
    )


async def test_year_change_preserves_origin_for_legacy_records_without_year_fields(state):
    store, _, _, _ = state
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact())
    case = await _update(
        state, case, "confirm_fact", {"fact_id": case["facts"][0]["id"]}, actor="user"
    )
    import json

    case["facts"][0].pop("case_tax_year")
    case["facts"][0].pop("confirmed_tax_year")
    with store._lock:
        store._conn.execute(
            "UPDATE app_tax_cases SET record_json=? WHERE case_id=?", (json.dumps(case), case["id"])
        )
        store._conn.commit()
    changed = await _update(state, case, "change_year", {"tax_year": 2024}, actor="user")
    assert changed["facts"][0]["case_tax_year"] == 2025
    assert changed["facts"][0]["status"] == "proposed"


async def test_rejected_facts_keep_history_and_block_dependencies_without_repeat_confirmation(
    state,
):
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact())
    fact = copy.deepcopy(case["facts"][0])
    with pytest.raises(TaxCaseError, match="user action"):
        await _update(state, case, "reject_fact", {"fact_id": fact["id"]})
    case = await _update(state, case, "reject_fact", {"fact_id": fact["id"]}, actor="user")
    rejected = case["facts"][0]
    assert rejected["status"] == "rejected" and rejected["value"] == fact["value"]
    assert rejected["rejected_by"] == "user" and rejected["rejected_tax_year"] == 2025
    assert await _update(state, case, "reject_fact", {"fact_id": fact["id"]}, actor="user") == case
    with pytest.raises(TaxCaseError, match="corrected proposal"):
        await _update(state, case, "confirm_fact", {"fact_id": fact["id"]}, actor="user")
    case = await _update(
        state,
        case,
        "propose_return_item",
        {"label": "Rejected dependency", "fact_ids": [fact["id"]]},
    )
    issues = tax_case_checklist(case)["next_steps"]
    assert not any("User confirmation is needed" in issue for issue in issues)
    assert any("unconfirmed fact" in issue for issue in issues)
    case = await _update(state, case, "change_year", {"tax_year": 2024}, actor="user")
    assert case["facts"][0]["status"] == "rejected"


async def _supported_review(state):
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact())
    case = await _update(
        state, case, "confirm_fact", {"fact_id": case["facts"][0]["id"]}, actor="user"
    )
    case = await _update(
        state, case, "record_evidence", _evidence(tax_year_status="verified"), actor="user"
    )
    calculation = {
        "label": "Supported sum",
        "inputs": {"a": "10"},
        "expression": "10",
        "rounding": "none",
        "result": "10",
        "fact_ids": [case["facts"][0]["id"]],
        "evidence_ids": [case["evidence"][0]["id"]],
    }
    case = await _update(state, case, "record_calculation", calculation)
    return_item = {
        "label": "Supported planning item",
        "fact_ids": calculation["fact_ids"],
        "evidence_ids": calculation["evidence_ids"],
    }
    case = await _update(state, case, "propose_return_item", return_item)
    return case, calculation, return_item


async def test_reference_update_requires_owner_user_action_revision_and_explicit_pin(state):
    store, owner, _, conversation = state
    case = await _create(state)
    update = {"reference_release": "rel_next"}
    with pytest.raises(TaxCaseError, match="user action"):
        await _update(state, case, "update_reference_release", update)
    assert (
        await _update(
            state,
            case,
            "update_reference_release",
            {"reference_release": "rel_current"},
            actor="user",
        )
        == case
    )
    other = await _owner(store, "other")
    with pytest.raises(TaxCaseAccessError):
        await store.tax_cases.update(
            user_id=other.user_id,
            conversation_id=conversation.conversation_id,
            expected_revision=case["revision"],
            operation="update_reference_release",
            data=update,
            actor="user",
        )
    changed = await _update(state, case, "update_reference_release", update, actor="user")
    assert changed["reference_release"] == "rel_next"
    assert changed["previous_reference_release"] == "rel_current"
    assert changed["reference_release_changed_at"]
    assert changed["revision"] == case["revision"] + 1
    assert (
        await _update(state, changed, "update_reference_release", update, actor="user") == changed
    )
    with pytest.raises(TaxCaseConflictError):
        await _update(state, case, "update_reference_release", update, actor="user")
    assert (
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
    ) == changed


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"reference_release": ""},
        {"reference_release": "x" * 201},
        {"reference_release": 123},
        {"reference_release": "rel_next", "confirm": True},
    ],
)
def test_reference_update_payload_is_strict_and_bounded(data):
    with pytest.raises(TaxCaseError):
        validate_operation("update_reference_release", data)


async def test_reference_update_preserves_facts_and_original_evidence_but_blocks_stale_dependencies(
    state,
):
    case, calculation, _ = await _supported_review(state)
    original = copy.deepcopy(case)
    case = await _update(
        state, case, "update_reference_release", {"reference_release": "rel_next"}, actor="user"
    )
    assert case["facts"] == original["facts"]
    assert case["files"] == original["files"] and case["tax_year"] == 2025
    old_evidence = case["evidence"][0]
    for field in (
        "document_id",
        "reference_release",
        "revision",
        "page",
        "passage",
        "tax_year",
        "tax_year_status",
    ):
        assert old_evidence[field] == original["evidence"][0][field]
    for category in ("evidence", "calculations", "return_items"):
        assert case[category][0]["case_reference_release"] == "rel_current"
        assert case[category][0]["status"] == "needs_recheck"
        assert case[category][0]["recheck_for_reference_release"] == "rel_next"
    issues = tax_case_checklist(case)["next_steps"]
    assert any("Recheck evidence" in issue for issue in issues)
    assert any("Recheck item" in issue for issue in issues)
    for operation, data in (
        ("record_calculation", calculation),
        (
            "propose_return_item",
            {"label": "Unsafe reuse", "evidence_ids": calculation["evidence_ids"]},
        ),
    ):
        with pytest.raises(TaxCaseError, match="rechecked reference evidence"):
            await _update(state, case, operation, data)
    with pytest.raises(TaxCaseError, match="pinned case release"):
        await _update(state, case, "record_evidence", _evidence())


async def test_rechecked_replacement_evidence_and_work_clear_stale_checklist_without_deleting_originals(
    state,
):
    case, calculation, return_item = await _supported_review(state)
    original = copy.deepcopy(case)
    case = await _update(
        state, case, "update_reference_release", {"reference_release": "rel_next"}, actor="user"
    )
    evidence_data = _evidence(
        reference_release="rel_next",
        revision="new_revision",
        passage="A changed retained rule passage.",
        tax_year_status="verified",
        supersedes=[case["evidence"][0]["id"]],
    )
    case = await _update(state, case, "record_evidence", evidence_data, actor="user")
    fresh_evidence = case["evidence"][-1]
    assert (
        fresh_evidence["status"] == "recorded"
        and fresh_evidence["case_reference_release"] == "rel_next"
    )
    assert case["evidence"][0]["superseded_by"] == fresh_evidence["id"]
    assert case["evidence"][0]["passage"] == original["evidence"][0]["passage"]
    assert await _update(state, case, "record_evidence", evidence_data, actor="user") == case
    calculation["evidence_ids"] = return_item["evidence_ids"] = [fresh_evidence["id"]]
    case = await _update(state, case, "record_calculation", calculation)
    case = await _update(state, case, "propose_return_item", return_item)
    for category in ("calculations", "return_items"):
        assert len(case[category]) == 2
        assert case[category][0]["superseded_by"] == case[category][-1]["id"]
        assert case[category][-1]["case_reference_release"] == "rel_next"
    assert tax_case_checklist(case)["status"] == "ready_for_review"
    with pytest.raises(TaxCaseError, match="awaiting recheck"):
        await _update(
            state,
            case,
            "record_evidence",
            _evidence(reference_release="rel_next", supersedes=[fresh_evidence["id"]]),
        )
    with pytest.raises(TaxCaseError, match="dependency"):
        await _update(
            state,
            case,
            "record_evidence",
            _evidence(reference_release="rel_next", supersedes=["missing"]),
        )


async def test_reference_roundtrip_requires_fresh_evidence_and_never_resurrects_old_work(state):
    case, calculation, return_item = await _supported_review(state)
    original_evidence_id = case["evidence"][0]["id"]
    for release in ("rel_next", "rel_current"):
        case = await _update(
            state, case, "update_reference_release", {"reference_release": release}, actor="user"
        )
        assert all(entry["status"] == "needs_recheck" for entry in case["evidence"])
        with pytest.raises(TaxCaseError, match="rechecked reference evidence"):
            await _update(state, case, "record_calculation", calculation)
        case = await _update(
            state,
            case,
            "record_evidence",
            _evidence(
                reference_release=release,
                tax_year_status="verified",
                supersedes=[item["id"] for item in case["evidence"]],
            ),
            actor="user",
        )
        calculation["evidence_ids"] = return_item["evidence_ids"] = [case["evidence"][-1]["id"]]
        case = await _update(state, case, "record_calculation", calculation)
        case = await _update(state, case, "propose_return_item", return_item)
        assert tax_case_checklist(case)["status"] == "ready_for_review"
    assert len(case["evidence"]) == len(case["calculations"]) == len(case["return_items"]) == 3
    assert case["evidence"][-1]["id"] != original_evidence_id
    assert case["evidence"][0]["status"] == "needs_recheck"
    assert case["facts"][0]["status"] == "confirmed"


async def test_multi_release_supersession_follows_current_terminal_replacements(state):
    case, calculation, return_item = await _supported_review(state)
    for index, release in enumerate(("rel_next", "rel_third"), start=1):
        case = await _update(
            state, case, "update_reference_release", {"reference_release": release}, actor="user"
        )
        case = await _update(
            state,
            case,
            "record_evidence",
            _evidence(
                reference_release=release,
                revision=str(index),
                passage=f"A distinct official passage for release {index}.",
                tax_year_status="verified",
                supersedes=[case["evidence"][-1]["id"]],
            ),
            actor="user",
        )
        calculation.update(
            result=str(10 + index),
            expression=f"10 + {index}",
            evidence_ids=[case["evidence"][-1]["id"]],
            supersedes=[case["calculations"][-1]["id"]],
        )
        return_item.update(
            label=f"Updated planning item {index}",
            evidence_ids=calculation["evidence_ids"],
            supersedes=[case["return_items"][-1]["id"]],
        )
        case = await _update(state, case, "record_calculation", calculation)
        case = await _update(state, case, "propose_return_item", return_item)
        assert tax_case_checklist(case)["status"] == "ready_for_review"
    assert case["evidence"][0]["superseded_by"] == case["evidence"][1]["id"]
    assert case["evidence"][1]["superseded_by"] == case["evidence"][2]["id"]
    assert case["calculations"][0]["result"] == "10"
    # Malformed persisted history fails safe; a cycle cannot hide the old evidence.
    case["evidence"][-1]["superseded_by"] = case["evidence"][0]["id"]
    assert tax_case_checklist(case)["status"] == "needs_information"


async def test_legacy_release_records_remain_idempotent_until_explicit_update(state):
    import hashlib
    import json

    store, _, _, _ = state
    case, calculation, _ = await _supported_review(state)
    for category in ("evidence", "calculations", "return_items"):
        for entry in case[category]:
            entry.pop("case_reference_release")
            entry.pop("supersedes")
            content = {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "id",
                    "operation_key",
                    "operation_digest",
                    "created_at",
                    "created_by",
                    "status",
                }
            }
            entry["operation_digest"] = hashlib.sha256(
                json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
    with store._lock:
        store._conn.execute(
            "UPDATE app_tax_cases SET record_json=? WHERE case_id=?", (json.dumps(case), case["id"])
        )
        store._conn.commit()
    assert (
        await _update(
            state, case, "record_evidence", _evidence(tax_year_status="verified"), actor="user"
        )
        == case
    )
    assert await _update(state, case, "record_calculation", calculation) == case
    changed = await _update(
        state, case, "update_reference_release", {"reference_release": "rel_next"}, actor="user"
    )
    assert all(
        item["case_reference_release"] == "rel_current" and item["status"] == "needs_recheck"
        for item in changed["evidence"]
    )


async def test_operation_key_cannot_change_which_retained_evidence_is_superseded(state):
    case = await _create(state)
    case = await _update(state, case, "record_evidence", _evidence(claim="First rule"))
    case = await _update(state, case, "record_evidence", _evidence(claim="Second rule"))
    case = await _update(
        state, case, "update_reference_release", {"reference_release": "rel_next"}, actor="user"
    )
    first, second = [item["id"] for item in case["evidence"]]
    payload = _evidence(
        reference_release="rel_next", operation_key="fresh-rule", supersedes=[first]
    )
    case = await _update(state, case, "record_evidence", payload)
    with pytest.raises(TaxCaseConflictError, match="operation key"):
        await _update(state, case, "record_evidence", {**payload, "supersedes": [first, second]})
    assert "superseded_by" not in case["evidence"][1]


def _saved_review_payload(case, **changes):
    return {
        "summary": "The recorded filing-status detail and selected rule passage are ready to discuss.",
        "findings": [
            {
                "label": "Recorded planning result",
                "detail": "This finding is limited to the records linked here.",
                "fact_ids": [item["id"] for item in case["facts"] if item["status"] != "rejected"],
                "evidence_ids": [item["id"] for item in case["evidence"]],
                "calculation_ids": [item["id"] for item in case["calculations"]],
                "return_item_ids": [item["id"] for item in case["return_items"]],
            }
        ],
        "next_question": "",
        **changes,
    }


async def test_saved_review_is_revision_bound_idempotent_and_survives_reopen(state):
    store, owner, _, conversation = state
    case, _, _ = await _supported_review(state)
    assert tax_case_review(case) is None
    payload = _saved_review_payload(case)
    previous_revision = case["revision"]
    case = await _update(state, case, "save_review", payload)
    assert case["revision"] == previous_revision + 1
    assert case["review"]["review_revision"] == case["revision"]
    assert case["review"]["tax_year"] == 2025
    assert case["review"]["reference_release"] == "rel_current"
    view = tax_case_review(case)
    assert view["status"] == "review_saved" and view["stale"] is False
    assert view["limitations"] == []
    assert "Selected passages only" in view["document_scope"]
    assert "does not establish" in view["notice"]
    assert await _update(state, case, "save_review", payload) == case
    reopened = ApplicationStore(store.path)
    try:
        current = await reopened.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
        assert current == case and tax_case_review(current) == view
    finally:
        reopened.close()
    with pytest.raises(TaxCaseConflictError):
        await store.tax_cases.update(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            expected_revision=previous_revision,
            operation="save_review",
            data=payload,
        )


async def test_changed_information_stales_review_and_identical_payload_requires_refresh(state):
    case, _, _ = await _supported_review(state)
    payload = _saved_review_payload(case)
    case = await _update(state, case, "save_review", payload)
    old_review = copy.deepcopy(case["review"])
    # A true no-op does not stale the existing report.
    same = await _update(
        state, case, "confirm_fact", {"fact_id": case["facts"][0]["id"]}, actor="user"
    )
    assert same == case and tax_case_review(same)["stale"] is False
    changed = await _update(state, case, "add_missing_item", {"label": "One additional statement"})
    assert changed["review"] == old_review
    assert tax_case_review(changed)["stale"] is True
    assert tax_case_review(changed)["status"] == "needs_input"
    refreshed = await _update(state, changed, "save_review", payload)
    assert refreshed["revision"] == changed["revision"] + 1
    assert refreshed["review"]["review_revision"] == refreshed["revision"]
    assert tax_case_review(refreshed)["stale"] is False
    assert tax_case_review(refreshed)["status"] == "needs_input"


@pytest.mark.parametrize(
    "field", ["fact_ids", "evidence_ids", "calculation_ids", "return_item_ids"]
)
async def test_review_references_must_belong_to_the_same_case_and_failure_is_atomic(state, field):
    store, owner, _, conversation = state
    case, _, _ = await _supported_review(state)
    case = await _update(state, case, "save_review", _saved_review_payload(case))
    other_conversation = await store.conversations.create(
        user_id=owner.user_id, private_required=True
    )
    other = await store.tax_cases.create(
        user_id=owner.user_id,
        conversation_id=other_conversation.conversation_id,
        tax_year=2025,
        goal="Other review",
        reference_release="rel_current",
    )
    other = await store.tax_cases.update(
        user_id=owner.user_id,
        conversation_id=other_conversation.conversation_id,
        expected_revision=other["revision"],
        operation="propose_fact",
        data=_fact(),
    )
    payload = _saved_review_payload(case)
    payload["findings"][0][field] = [other["facts"][0]["id"]]
    with pytest.raises(TaxCaseError, match="not found in this case"):
        await _update(state, case, "save_review", payload)
    assert (
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
        == case
    )


async def test_saved_review_cannot_refresh_using_rejected_nested_fact_support(state):
    store, owner, _, conversation = state
    case, _, _ = await _supported_review(state)
    payload = _saved_review_payload(case)
    # The rejected dependency is nested through saved work, not directly listed.
    payload["findings"][0]["fact_ids"] = []
    case = await _update(state, case, "save_review", payload)
    case = await _update(
        state, case, "reject_fact", {"fact_id": case["facts"][0]["id"]}, actor="user"
    )
    old_review = copy.deepcopy(case["review"])
    with pytest.raises(TaxCaseError, match="rejected fact"):
        await _update(state, case, "save_review", payload)
    current = await store.tax_cases.get(
        user_id=owner.user_id, conversation_id=conversation.conversation_id
    )
    assert current == case and current["review"] == old_review
    assert tax_case_review(current)["stale"] is True
    assert any("rejected fact" in line for line in tax_case_review(current)["limitations"])


@pytest.mark.parametrize("field", ["evidence_ids", "calculation_ids", "return_item_ids"])
async def test_review_cannot_reuse_support_after_library_change(state, field):
    case, _, _ = await _supported_review(state)
    payload = _saved_review_payload(case)
    payload["findings"][0] = {
        "label": "One retained dependency",
        "detail": "The dependency needs to be current.",
        field: payload["findings"][0][field],
    }
    case = await _update(state, case, "save_review", payload)
    case = await _update(
        state, case, "update_reference_release", {"reference_release": "rel_next"}, actor="user"
    )
    assert tax_case_review(case)["stale"] is True
    with pytest.raises(TaxCaseError, match="rechecked"):
        await _update(state, case, "save_review", payload)


async def test_year_change_preserves_report_but_stale_work_cannot_refresh_it(state):
    case, _, _ = await _supported_review(state)
    payload = _saved_review_payload(case)
    case = await _update(state, case, "save_review", payload)
    original = copy.deepcopy(case["review"])
    case = await _update(state, case, "change_year", {"tax_year": 2024}, actor="user")
    assert case["review"] == original and tax_case_review(case)["stale"] is True
    with pytest.raises(TaxCaseError, match="rechecked"):
        await _update(state, case, "save_review", payload)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"tax_year": None}, "unknown"),
        ({"tax_year": 2024, "tax_year_status": "verified"}, "different tax year"),
        ({"tax_year_status": "proposed"}, "not been verified"),
        (
            {"jurisdiction": "Other-jurisdiction", "tax_year_status": "verified"},
            "different jurisdiction",
        ),
    ],
)
async def test_uncertain_rule_applicability_is_preserved_as_review_limitation(
    state, changes, message
):
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact())
    case = await _update(
        state, case, "confirm_fact", {"fact_id": case["facts"][0]["id"]}, actor="user"
    )
    case = await _update(state, case, "record_evidence", _evidence(**changes), actor="user")
    case = await _update(state, case, "save_review", _saved_review_payload(case))
    view = tax_case_review(case)
    assert view["status"] == "limited_evidence" and view["stale"] is False
    assert any(message in limitation for limitation in view["limitations"])


async def test_proposed_facts_questions_and_missing_support_never_imply_completed_review(state):
    case = await _create(state)
    case = await _update(
        state,
        case,
        "save_review",
        {"summary": "There are no saved records to review yet.", "findings": []},
    )
    view = tax_case_review(case)
    assert view["status"] == "limited_evidence"
    assert any("No official rule passages" in line for line in view["limitations"])
    assert any("No saved facts" in line for line in view["limitations"])
    case = await _update(state, case, "propose_fact", _fact())
    case = await _update(
        state,
        case,
        "save_review",
        _saved_review_payload(case, next_question="Could you confirm your filing status?"),
    )
    view = tax_case_review(case)
    assert view["status"] == "needs_input"
    assert any("confirmation" in line for line in view["limitations"])
    assert view["next_question"] == "Could you confirm your filing status?"


@pytest.mark.parametrize(
    "changes",
    [
        {"summary": ""},
        {"summary": "x" * 2001},
        {"findings": [{}] * 13},
        {"next_question": "x" * 501},
        {"status": "correct"},
        {"findings": [{"label": "x" * 501, "detail": "Detail"}]},
        {"findings": [{"label": "Label", "detail": "x" * 2001}]},
        {"findings": [{"label": "Label", "detail": "Detail", "fact_ids": ["x"] * 101}]},
        {"findings": [{"label": "Label", "detail": "Detail", "verified": True}]},
    ],
)
def test_review_payload_bounds_cannot_be_overridden(changes):
    with pytest.raises(TaxCaseError):
        validate_operation(
            "save_review", {"summary": "A recorded review.", "findings": [], **changes}
        )


async def test_oversized_review_does_not_erase_prior_report_or_partial_case_update(state):
    store, owner, _, conversation = state
    case, _, _ = await _supported_review(state)
    payload = _saved_review_payload(case)
    case = await _update(state, case, "save_review", payload)
    finding = payload["findings"][0]
    finding.update(label="l" * 500, detail="d" * 2000)
    for field in ("fact_ids", "evidence_ids", "calculation_ids", "return_item_ids"):
        finding[field] = finding[field] * 100
    payload["findings"] = [copy.deepcopy(finding) for _ in range(12)]
    with pytest.raises(TaxCaseError, match="storage limit"):
        await _update(state, case, "save_review", payload)
    assert (
        await store.tax_cases.get(
            user_id=owner.user_id, conversation_id=conversation.conversation_id
        )
        == case
    )


async def test_owner_fact_correction_invalidates_old_calculations_and_review(state):
    case = await _create(state)
    case = await _update(state, case, "propose_fact", _fact(value="100.00"))
    fact_id = case["facts"][0]["id"]
    case = await _update(state, case, "confirm_fact", {"fact_id": fact_id}, actor="user")
    case = await _update(state, case, "record_evidence", _evidence())
    support = {"fact_ids": [fact_id], "evidence_ids": [case["evidence"][0]["id"]]}
    case = await _update(state, case, "record_calculation", {
        "label": "Wages", "inputs": {"wages": "100.00"}, "expression": "wages",
        "rounding": "none", "result": "100", **support})
    calculation_id = case["calculations"][0]["id"]
    case = await _update(state, case, "propose_return_item", {"label": "Wages", **support})
    candidate = {"summary": "Limited worksheet", "findings": [{"label": "Wages", "detail": "100.00",
                  "calculation_ids": [calculation_id], **support}]}
    case = await _update(state, case, "save_review", candidate)
    with pytest.raises(TaxCaseError, match="user action"):
        await _update(state, case, "edit_fact", {"fact_id": fact_id, "label": "Wages", "value": "200.00"})
    changed = await _update(state, case, "edit_fact", {
        "fact_id": fact_id, "label": "Wages", "value": "200.00"}, actor="user")
    assert len(changed["facts"]) == 1 and changed["facts"][0]["id"] == fact_id
    assert changed["facts"][0]["history"][0]["value"] == "100.00"
    assert changed["facts"][0]["value"] == "200.00"
    assert changed["facts"][0]["status"] == "confirmed"
    assert changed["calculations"][0]["status"] == "needs_recheck"
    assert changed["return_items"][0]["status"] == "needs_recheck"
    assert tax_case_review(changed)["stale"]
    assert any("after a saved fact changed" in issue for issue in tax_case_checklist(changed)["next_steps"])
    with pytest.raises(TaxCaseError, match="rechecked calculations"):
        await _update(state, changed, "save_review", candidate)
    unchanged = await _update(state, changed, "edit_fact", {
        "fact_id": fact_id, "label": "Wages", "value": "200.00"}, actor="user")
    assert unchanged == changed


async def test_unbound_repository_user_statement_is_not_an_authority(state):
    case = await _create(state)
    with pytest.raises(TaxCaseError, match="current user's own message"):
        await _update(state, case, "record_user_fact", {"label": "Wages", "value": "$50,000"})
    assert (await _create_read(state))["facts"] == []


async def _create_read(state):
    store, owner, _, conversation = state
    return await store.tax_cases.get(user_id=owner.user_id, conversation_id=conversation.conversation_id)


@pytest.mark.parametrize(("content", "excerpt", "valid"), [
    ("My wages were 50000.00", "0", False),
    ("My wages were $50,000.00", "000", False),
    ("My wages were $50,000.00", "$50,000", False),
    ("I was not blind", "blind", False),
    ("I was not legally blind", "legally blind", False),
    ("I cannot be claimed as a dependent", "claimed as a dependent", False),
    ("I was not blind", "not blind", True),
    ("I was not blind, and my wages were $50,000.00", "$50,000.00", True),
    ("I was 40 at the end of 2025", "40 at the end of 2025", True),
])
def test_current_statement_requires_intact_excerpt(content, excerpt, valid):
    from audrey.app_state.tax_cases import is_user_statement_excerpt

    assert is_user_statement_excerpt(content, excerpt) is valid
