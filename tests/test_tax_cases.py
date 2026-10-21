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
    assert store.schema_version == 23
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
        connection.execute("DROP TABLE app_tax_cases")
        connection.execute("DELETE FROM app_schema_migrations WHERE version=23")
        connection.commit()
    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 23
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
