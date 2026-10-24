"""Bounded, owner-bound Private tax review records and deterministic coverage checks.

Evidence authenticity and calculator execution are checked by the bound service
before a write. This repository enforces ownership, privacy and state transitions;
its checklist describes evidence coverage and never decides tax law or filing.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from audrey.app_state.private_tasks import PrivateTaskError, guard_tax_case_write
from audrey.app_state.projects import PROJECT_MAX_FILES

MAX_CASE_BYTES = 128 * 1024
MAX_CATEGORY_ENTRIES = 100
CHECKLIST_VERSION = 1
TaxYear = Annotated[int, Field(strict=True, ge=1900, le=2100)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Passage = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
Checksum = Annotated[str, StringConstraints(pattern=r"\A[0-9a-f]{64}\z")]
EntryIds = Annotated[list[Identifier], Field(max_length=MAX_CATEGORY_ENTRIES)]


class TaxCaseError(ValueError):
    """A tax case payload or transition violates the case contract."""


class TaxCaseConflictError(TaxCaseError):
    """A case already exists or its revision changed before this operation."""


class TaxCaseAccessError(TaxCaseError):
    """An active owner and canonical Private conversation are required."""


@dataclass
class TaxUserStatement:
    """Trusted native turn context; model payloads cannot supply this authority."""

    user_id: str
    conversation_id: str
    run_id: str
    message_id: str
    content: str
    active: bool = True


_USER_STATEMENT: ContextVar[TaxUserStatement | None] = ContextVar("tax_user_statement", default=None)


def current_user_statement() -> TaxUserStatement | None:
    statement = _USER_STATEMENT.get()
    return statement if statement is not None and statement.active else None


def is_user_statement_excerpt(content: str, value: str) -> bool:
    """Require an intact excerpt, not a number/token fragment or dropped negation.

    This validates provenance, not every semantic interpretation of natural text.
    The writer must still preserve complete assertions and leave ambiguity proposed.
    """
    if not value or not value.strip():
        return False
    for match in re.finditer(re.escape(value), content):
        before, after = content[:match.start()], content[match.end():]
        if before and before[-1].isalnum() and value[0].isalnum():
            continue
        if after and after[0].isalnum() and value[-1].isalnum():
            continue
        # Do not turn 50,000.00 into 50,000 or 50,000 into 000.
        if (value[0].isdigit() and re.search(r"[0-9][.,]$", before)) or (
            value[-1].isdigit() and re.match(r"[.,][0-9]", after)
        ):
            continue
        # A negated assertion must quote the negation as part of the fact value.
        if re.search(r"\b(?:not|no|never|without|cannot|[A-Za-z]+n't)\s+(?:[A-Za-z]+\s+){0,2}$",
                     before, re.IGNORECASE):
            continue
        return True
    return False


@contextmanager
def user_statement_scope(
    *, user_id: str, conversation_id: str, run_id: str, message_id: str, content: str,
):
    """Bind one canonical native user turn; copied contexts expire with its scope."""
    statement = TaxUserStatement(user_id, conversation_id, run_id, message_id, content)
    token = _USER_STATEMENT.set(statement)
    try:
        yield statement
    finally:
        statement.active = False
        _USER_STATEMENT.reset(token)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class FileSelection(_Payload):
    file_id: Identifier
    checksum: Checksum
    revision: Identifier
    filename: Annotated[str, Field(max_length=500)] = ""
    review_scope: Literal["selected_passages"] = "selected_passages"
    unread_pages: Annotated[
        list[Annotated[int, Field(strict=True, ge=1, le=100_000)]], Field(max_length=100)
    ] = Field(default_factory=list)


class FactProvenance(_Payload):
    file_id: Identifier
    checksum: Checksum
    page: Annotated[int, Field(strict=True, ge=1, le=100_000)]
    box: Annotated[str, Field(max_length=200)] = ""
    original_text: Passage
    normalization: Annotated[str, Field(max_length=2000)] = ""


class _AppendPayload(_Payload):
    operation_key: Identifier | None = None


class ProposeFactPayload(_AppendPayload):
    label: Label
    value: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    kind: Literal["user_statement", "extracted"]
    provenance: FactProvenance | None = None

    @model_validator(mode="after")
    def require_extraction_source(self) -> ProposeFactPayload:
        if self.kind == "extracted" and self.provenance is None:
            raise ValueError("extracted facts require document provenance")
        if self.kind == "user_statement" and self.provenance is not None:
            raise ValueError("only extracted facts may have document provenance")
        return self


class RecordUserFactPayload(_Payload):
    label: Label
    value: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    fact_id: Identifier | None = None
    addresses: EntryIds = Field(default_factory=list)

    @field_validator("addresses")
    @classmethod
    def unique_addresses(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("missing item IDs must be unique")
        return value


class EditFactPayload(_Payload):
    fact_id: Identifier
    label: Label
    value: Annotated[str, StringConstraints(min_length=1, max_length=4000)]


class ConfirmFactPayload(_Payload):
    fact_id: Identifier


class ConfirmFactsPayload(_Payload):
    fact_ids: EntryIds = Field(min_length=1)

    @field_validator("fact_ids")
    @classmethod
    def unique_fact_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("fact IDs must be unique")
        return value


class RejectFactPayload(_Payload):
    fact_id: Identifier


class ChangeYearPayload(_Payload):
    tax_year: TaxYear


class UpdateReferenceReleasePayload(_Payload):
    reference_release: Identifier


class _RecheckPayload(_AppendPayload):
    supersedes: EntryIds = Field(default_factory=list)


class RecordEvidencePayload(_RecheckPayload):
    claim: Label
    document_id: Identifier
    reference_release: Identifier
    revision: Identifier
    tax_year: TaxYear | None = None
    tax_year_status: Literal["unknown", "proposed", "verified"] = "unknown"
    jurisdiction: Identifier = "US-federal"
    catalog: Literal["current", "prior", "bulletins"] = "current"
    page: Annotated[int, Field(strict=True, ge=1, le=100_000)]
    offset: Annotated[int, Field(strict=True, ge=0, le=200_000)] = 0
    section: Annotated[str, Field(max_length=500)] = ""
    passage: Passage
    conflicts_with: EntryIds = Field(default_factory=list)


class RecordCalculationPayload(_RecheckPayload):
    label: Label
    inputs: Annotated[
        dict[Identifier, Annotated[str, Field(min_length=1, max_length=200)]], Field(max_length=100)
    ]
    expression: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    rounding: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    result: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    fact_ids: EntryIds = Field(default_factory=list)
    evidence_ids: EntryIds = Field(default_factory=list)

    @field_validator("inputs")
    @classmethod
    def decimal_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        for item in value.values():
            _finite_decimal(item)
        return value

    @field_validator("result")
    @classmethod
    def decimal_result(cls, value: str) -> str:
        _finite_decimal(value)
        return value


class AddMissingItemPayload(_AppendPayload):
    label: Label
    question: Annotated[str, Field(max_length=500)] = ""


class ResolveMissingItemPayload(_Payload):
    id: Identifier


class ProposeReturnItemPayload(_RecheckPayload):
    label: Label
    fact_ids: EntryIds = Field(default_factory=list)
    evidence_ids: EntryIds = Field(default_factory=list)


class ReviewFindingPayload(_Payload):
    label: Label
    detail: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    fact_ids: EntryIds = Field(default_factory=list)
    evidence_ids: EntryIds = Field(default_factory=list)
    calculation_ids: EntryIds = Field(default_factory=list)
    return_item_ids: EntryIds = Field(default_factory=list)


class SaveReviewPayload(_Payload):
    summary: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    findings: Annotated[list[ReviewFindingPayload], Field(max_length=12)]
    next_question: Annotated[str, Field(max_length=500)] = ""


class _CaseCreation(_Payload):
    tax_year: TaxYear
    jurisdiction: Identifier = "US-federal"
    goal: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    reference_release: Identifier
    files: Annotated[list[FileSelection], Field(max_length=100)] = Field(default_factory=list)


OPERATION_MODELS: dict[str, type[_Payload]] = {
    "propose_fact": ProposeFactPayload,
    "record_user_fact": RecordUserFactPayload,
    "edit_fact": EditFactPayload,
    "confirm_fact": ConfirmFactPayload,
    "confirm_facts": ConfirmFactsPayload,
    "reject_fact": RejectFactPayload,
    "change_year": ChangeYearPayload,
    "update_reference_release": UpdateReferenceReleasePayload,
    "record_evidence": RecordEvidencePayload,
    "record_calculation": RecordCalculationPayload,
    "add_missing_item": AddMissingItemPayload,
    "resolve_missing_item": ResolveMissingItemPayload,
    "propose_return_item": ProposeReturnItemPayload,
    "save_review": SaveReviewPayload,
}
_APPEND_CATEGORIES = {
    "propose_fact": "facts",
    "record_evidence": "evidence",
    "record_calculation": "calculations",
    "add_missing_item": "missing_items",
    "propose_return_item": "return_items",
}


def _finite_decimal(value: str) -> None:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("a decimal string is required") from exc
    if not parsed.is_finite() or abs(parsed.adjusted()) > 1000:
        raise ValueError("a bounded finite decimal is required")


def validate_operation(operation: str, data: dict[str, Any]) -> dict[str, Any]:
    """Validate typed data before the service uses any model-supplied fields."""
    model = OPERATION_MODELS.get(operation)
    if model is None:
        raise TaxCaseError("Unsupported tax case operation.")
    try:
        payload = model.model_validate(data).model_dump(mode="json")
        # Preserve historical idempotency digests when an old label-only gap is
        # replayed; optional presentation text need not change its stored shape.
        if operation == "add_missing_item" and not payload.get("question"):
            payload.pop("question", None)
        return payload
    except ValidationError as exc:
        raise TaxCaseError(f"Invalid {operation} payload.") from exc


class TaxCasesRepository:
    """One bounded case per Private conversation, with optimistic revisions."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    async def get(self, *, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_sync, user_id, conversation_id)

    def _get_sync(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            project_id = self._assert_context_locked(user_id, conversation_id)
            row = self._case_row_locked(user_id, conversation_id)
            return self._decode(row, project_id) if row is not None else None

    async def create(
        self,
        *,
        user_id: str,
        conversation_id: str,
        tax_year: int,
        jurisdiction: str = "US-federal",
        goal: str,
        reference_release: str,
        files: list[dict[str, Any]] | None = None,
        validated_private_file_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            payload = _CaseCreation.model_validate(
                {
                    "tax_year": tax_year,
                    "jurisdiction": jurisdiction,
                    "goal": goal,
                    "reference_release": reference_release,
                    "files": files if files is not None else [],
                }
            ).model_dump(mode="json")
        except ValidationError as exc:
            raise TaxCaseError("Invalid tax case creation payload.") from exc
        if len({item["file_id"] for item in payload["files"]}) != len(payload["files"]):
            raise TaxCaseError("Case files must be unique.")
        file_ids = self._project_file_ids(validated_private_file_ids)
        return await asyncio.to_thread(
            self._create_sync, user_id, conversation_id, payload, file_ids,
        )

    def _create_sync(
        self, user_id: str, conversation_id: str, payload: dict[str, Any],
        validated_private_file_ids: list[str],
    ) -> dict[str, Any]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                project_id = self._assert_context_locked(user_id, conversation_id)
                existing = self._case_row_locked(user_id, conversation_id)
                if existing is not None:
                    case = self._decode(existing, project_id)
                    if any(case[key] != payload[key] for key in ("tax_year", "jurisdiction", "goal")):
                        raise TaxCaseConflictError("A different tax case already exists for this conversation.")
                    # A retry must not reset facts, replace the release pin or
                    # recreate a project the owner intentionally removed.
                    self._conn.commit()
                    return case
                now = _utc_now()
                project_id = self._ensure_project_locked(
                    user_id, conversation_id, project_id, payload["tax_year"], now,
                )
                self._add_project_files_locked(project_id, validated_private_file_ids, now)
                case = {
                    "id": _new_id(),
                    "conversation_id": conversation_id,
                    "project_id": project_id,
                    **payload,
                    "revision": 1,
                    "checklist_version": CHECKLIST_VERSION,
                    "facts": [],
                    "evidence": [],
                    "calculations": [],
                    "missing_items": [],
                    "return_items": [],
                    "created_at": now,
                    "updated_at": now,
                }
                encoded = _encode(case)
                self._conn.execute(
                    "INSERT INTO app_tax_cases (case_id, conversation_id, user_id, project_id, "
                    "revision, record_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (case["id"], conversation_id, user_id, project_id, 1, encoded, now, now),
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return case

    async def ensure_project(
        self, *, user_id: str, conversation_id: str,
        validated_private_file_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Explicitly group a legacy case; reads never recreate deleted projects."""
        file_ids = self._project_file_ids(validated_private_file_ids)
        return await asyncio.to_thread(self._ensure_project_sync, user_id, conversation_id, file_ids)

    def _ensure_project_sync(
        self, user_id: str, conversation_id: str, file_ids: list[str],
    ) -> dict[str, Any]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                project_id = self._assert_context_locked(user_id, conversation_id)
                row = self._case_row_locked(user_id, conversation_id)
                if row is None:
                    raise TaxCaseError("No tax case exists for this conversation.")
                case = self._decode(row, project_id)
                now = _utc_now()
                project_id = self._ensure_project_locked(
                    user_id, conversation_id, project_id, case["tax_year"], now,
                )
                self._add_project_files_locked(project_id, file_ids, now)
                case["project_id"] = project_id
                self._conn.execute(
                    "UPDATE app_tax_cases SET project_id=?,record_json=? "
                    "WHERE user_id=? AND conversation_id=?",
                    (project_id, _encode(case), user_id, conversation_id),
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return case

    async def link_project_files(
        self, *, user_id: str, conversation_id: str,
        validated_private_file_ids: list[str],
    ) -> None:
        """Attach service-validated Private documents before capturing run inputs."""
        file_ids = self._project_file_ids(validated_private_file_ids)
        await asyncio.to_thread(self._link_project_files_sync, user_id, conversation_id, file_ids)

    def _link_project_files_sync(
        self, user_id: str, conversation_id: str, file_ids: list[str],
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                project_id = self._assert_context_locked(user_id, conversation_id)
                if project_id is None or self._case_row_locked(user_id, conversation_id) is None:
                    raise TaxCaseAccessError("An existing Private tax project is required.")
                if self._conn.execute(
                    "SELECT 1 FROM app_runs WHERE user_id=? AND conversation_id=? "
                    "AND status='running'", (user_id, conversation_id),
                ).fetchone() is not None:
                    raise TaxCaseConflictError("Stop the current run before adding project files.")
                self._add_project_files_locked(project_id, file_ids, _utc_now())
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    async def list_for_project(
        self, *, user_id: str, project_id: str,
    ) -> list[dict[str, Any]]:
        """Project views read current case records, never copied facts or outputs."""
        return await asyncio.to_thread(self._list_for_project_sync, user_id, project_id)

    def _list_for_project_sync(self, user_id: str, project_id: str) -> list[dict[str, Any]]:
        with self._lock:
            project = self._conn.execute(
                "SELECT 1 FROM app_projects p JOIN app_users u ON u.user_id=p.user_id "
                "WHERE p.project_id=? AND p.user_id=? AND p.private_required=1 "
                "AND u.status='active' AND NOT EXISTS "
                "(SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)",
                (project_id, user_id),
            ).fetchone()
            if project is None:
                raise TaxCaseAccessError("Private project not found.")
            rows = self._conn.execute(
                "SELECT t.case_id,t.revision,t.record_json FROM app_tax_cases t "
                "JOIN app_conversations c ON c.conversation_id=t.conversation_id "
                "AND c.user_id=t.user_id WHERE t.user_id=? AND c.project_id=? "
                "AND c.private_required=1 AND c.archived_at IS NULL "
                "ORDER BY t.updated_at DESC,t.case_id DESC", (user_id, project_id),
            ).fetchall()
            return [self._decode(row, project_id) for row in rows]

    @staticmethod
    def _project_file_ids(file_ids: list[str] | None) -> list[str]:
        # These are resolved against the owner's Private upload inventory by the
        # service. File bytes and their ownership live outside this database.
        if file_ids is None:
            return []
        if (not isinstance(file_ids, list) or len(file_ids) > PROJECT_MAX_FILES
                or any(not isinstance(item, str) or not item or len(item) > 200
                       for item in file_ids) or len(set(file_ids)) != len(file_ids)):
            raise TaxCaseError("Invalid Private project file selection.")
        return list(file_ids)

    def _ensure_project_locked(
        self, user_id: str, conversation_id: str, project_id: str | None,
        tax_year: int, now: str,
    ) -> str:
        if project_id is not None:
            return project_id  # _assert_context_locked already checked owner/privacy.
        # A running turn captured a project/source snapshot. Do not invalidate it
        # by silently moving its conversation during execution.
        if self._conn.execute(
            "SELECT 1 FROM app_runs WHERE user_id=? AND conversation_id=? "
            "AND status='running'", (user_id, conversation_id),
        ).fetchone() is not None or self._conn.execute(
            "SELECT 1 FROM app_private_tasks WHERE user_id=? AND conversation_id=? "
            "AND status IN ('queued','running','waiting_for_user','paused')",
            (user_id, conversation_id),
        ).fetchone() is not None:
            raise TaxCaseConflictError("Stop the current review before creating its project.")
        project_id = f"proj_{uuid.uuid4().hex}"
        self._conn.execute(
            "INSERT INTO app_projects "
            "(project_id,user_id,name,instructions,created_at,updated_at,private_required) "
            "VALUES (?,?,?,'',?,?,1)",
            (project_id, user_id, f"Tax review {tax_year}", now, now),
        )
        self._conn.execute(
            "UPDATE app_conversations SET project_id=?,updated_at=? "
            "WHERE user_id=? AND conversation_id=?", (project_id, now, user_id, conversation_id),
        )
        return project_id

    def _add_project_files_locked(self, project_id: str, file_ids: list[str], now: str) -> None:
        existing = {row[0] for row in self._conn.execute(
            "SELECT file_id FROM app_project_files WHERE project_id=?", (project_id,),
        )}
        if len(existing | set(file_ids)) > PROJECT_MAX_FILES:
            raise TaxCaseError(f"Private projects can contain at most {PROJECT_MAX_FILES} files.")
        self._conn.executemany(
            "INSERT OR IGNORE INTO app_project_files (project_id,file_id,added_at) VALUES (?,?,?)",
            [(project_id, file_id, now) for file_id in file_ids],
        )
        if set(file_ids) - existing:
            self._conn.execute(
                "UPDATE app_projects SET updated_at=? WHERE project_id=?", (now, project_id),
            )

    async def update(
        self,
        *,
        user_id: str,
        conversation_id: str,
        expected_revision: int,
        operation: str,
        data: dict[str, Any],
        actor: Literal["model", "user"] = "model",
        verification: dict[str, Any] | None = None,
        user_statement: TaxUserStatement | None = None,
    ) -> dict[str, Any]:
        if type(expected_revision) is not int or expected_revision < 1:
            raise TaxCaseError("A positive expected revision is required.")
        if actor not in {"model", "user"}:
            raise TaxCaseError("Unsupported tax case actor.")
        payload = validate_operation(operation, data)
        verified_report = None
        if verification is not None:
            if operation != "save_review" or actor != "model":
                raise TaxCaseError("Only a model worksheet save may retain trusted verification.")
            # This argument is supplied by the bound service, never by the model's
            # operation payload. Import lazily because the checker validates the
            # candidate with this module's SaveReviewPayload schema.
            from audrey.tax_verification import ReviewVerification, candidate_digest

            try:
                report = ReviewVerification.model_validate(verification)
            except ValidationError as exc:
                raise TaxCaseError("Invalid worksheet verification report.") from exc
            if (
                report.case_revision != expected_revision
                or report.candidate_sha256 != candidate_digest(payload)
                or sorted(finding.finding_index for finding in report.findings)
                != list(range(len(payload["findings"])))
            ):
                raise TaxCaseError("Worksheet verification does not match this candidate revision.")
            verified_report = report.model_dump(mode="json")
        if (
            operation == "record_evidence"
            and actor == "model"
            and payload["tax_year_status"] == "verified"
        ):
            raise TaxCaseError("A model cannot verify tax-year applicability.")
        if (
            operation
            in {
                "confirm_fact",
                "confirm_facts",
                "edit_fact",
                "reject_fact",
                "resolve_missing_item",
                "change_year",
                "update_reference_release",
            }
            and actor != "user"
        ):
            raise TaxCaseError("This tax case transition requires a user action.")
        return await asyncio.to_thread(
            self._update_sync,
            user_id,
            conversation_id,
            expected_revision,
            operation,
            payload,
            actor,
            verified_report,
            user_statement,
        )

    def _update_sync(
        self,
        user_id: str,
        conversation_id: str,
        expected_revision: int,
        operation: str,
        payload: dict[str, Any],
        actor: str,
        verification: dict[str, Any] | None,
        user_statement: TaxUserStatement | None,
    ) -> dict[str, Any]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                project_id = self._assert_context_locked(user_id, conversation_id)
                row = self._case_row_locked(user_id, conversation_id)
                if row is None:
                    raise TaxCaseError("No tax case exists for this conversation.")
                case = self._decode(row, project_id)
                if case["revision"] != expected_revision:
                    raise TaxCaseConflictError("The tax case changed; reload its current revision.")
                try:
                    task_write = guard_tax_case_write(
                        self._conn,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        actor=actor,
                    )
                except PrivateTaskError:
                    raise TaxCaseConflictError(
                        "The Private task lease or inputs changed."
                    ) from None
                if operation == "record_user_fact":
                    self._assert_user_statement_locked(
                        user_id, conversation_id, payload, user_statement,
                    )
                changed = _apply_operation(
                    case, operation, payload, actor, verification, user_statement,
                )
                if changed:
                    case["revision"] += 1
                    case["updated_at"] = _utc_now()
                    encoded = _encode(case)
                    cursor = self._conn.execute(
                        "UPDATE app_tax_cases SET project_id = ?, revision = ?, record_json = ?, "
                        "updated_at = ? WHERE user_id = ? AND conversation_id = ? AND revision = ?",
                        (
                            project_id,
                            case["revision"],
                            encoded,
                            case["updated_at"],
                            user_id,
                            conversation_id,
                            expected_revision,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise TaxCaseConflictError(
                            "The tax case changed; reload its current revision."
                        )
                    if task_write is not None:
                        # Case progress and the task's expected case revision commit together.
                        lease, task_record = task_write
                        task_record["case_revision"] = case["revision"]
                        # A verified extracted fact can add its selected source
                        # to the case inventory. This is worker progress, not an
                        # owner changing the selected documents. Keep the source
                        # refinement in the same transaction as the fact.
                        task_record["sources"]["files"] = sorted(
                            case.get("files", []), key=lambda source: source["file_id"]
                        )
                        lease.repository._write(task_record)
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return case

    def _assert_user_statement_locked(
        self, user_id: str, conversation_id: str, payload: dict[str, Any],
        statement: TaxUserStatement | None,
    ) -> None:
        if (
            statement is None or not statement.active
            or statement.user_id != user_id or statement.conversation_id != conversation_id
            or not is_user_statement_excerpt(statement.content, payload["value"])
        ):
            raise TaxCaseError("A fact must quote the current user's own message exactly.")
        row = self._conn.execute(
            "SELECT message_id,run_id,content FROM app_messages "
            "WHERE user_id=? AND conversation_id=? AND role='user' "
            "ORDER BY sequence_no DESC LIMIT 1", (user_id, conversation_id),
        ).fetchone()
        if (
            row is None or row["message_id"] != statement.message_id
            or row["run_id"] != statement.run_id or row["content"] != statement.content
        ):
            raise TaxCaseError("The current user-message source changed or is unavailable.")

    def _assert_context_locked(self, user_id: str, conversation_id: str) -> str | None:
        if (
            not isinstance(user_id, str)
            or not user_id
            or not isinstance(conversation_id, str)
            or not conversation_id
        ):
            raise TaxCaseAccessError("An active owner and Private conversation are required.")
        row = self._conn.execute(
            "SELECT c.private_required, c.project_id, p.private_required AS project_private, "
            "p.user_id AS project_owner FROM app_conversations AS c "
            "JOIN app_users AS u ON u.user_id = c.user_id "
            "LEFT JOIN app_projects AS p ON p.project_id = c.project_id "
            "WHERE c.user_id = ? AND c.conversation_id = ? AND u.status = 'active' "
            "AND c.archived_at IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM account_deletion_requests AS d WHERE d.user_id = u.user_id)",
            (user_id, conversation_id),
        ).fetchone()
        if row is None or not row["private_required"]:
            raise TaxCaseAccessError("An active owner and Private conversation are required.")
        project_id = str(row["project_id"]) if row["project_id"] is not None else None
        if project_id is not None and (
            not row["project_private"] or row["project_owner"] != user_id
        ):
            raise TaxCaseAccessError(
                "Tax cases require a Private project owned by the active user."
            )
        return project_id

    def _case_row_locked(self, user_id: str, conversation_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT case_id, revision, record_json FROM app_tax_cases "
            "WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        ).fetchone()

    @staticmethod
    def _decode(row: sqlite3.Row, project_id: str | None) -> dict[str, Any]:
        case = json.loads(str(row["record_json"]))
        case["id"] = str(row["case_id"])
        case["revision"] = int(row["revision"])
        # Membership is always derived from the canonical conversation, including
        # after a project is deleted while its conversations are retained.
        case["project_id"] = project_id
        return case


def _apply_operation(
    case: dict[str, Any],
    operation: str,
    payload: dict[str, Any],
    actor: str,
    verification: dict[str, Any] | None = None,
    user_statement: TaxUserStatement | None = None,
) -> bool:
    if operation in {"edit_fact", "record_user_fact"}:
        return _write_current_fact(case, operation, payload, actor, user_statement)
    if operation == "save_review":
        # Revalidate even an identical report: a later fact rejection, year or
        # library change must never refresh stale support merely by replaying it.
        _review_support(case, payload, strict=True)
        previous = case.get("review")
        if (
            previous
            and previous.get("review_revision") == case["revision"]
            and all(previous.get(key) == value for key, value in payload.items())
            and previous.get("verification") == verification
        ):
            return False
        case["review"] = {
            **payload,
            "tax_year": case["tax_year"],
            "reference_release": case["reference_release"],
            "review_revision": case["revision"] + 1,
            "saved_at": _utc_now(),
        }
        if verification is not None:
            case["review"]["verification"] = verification
        return True
    if operation == "change_year":
        old_year, new_year = case["tax_year"], payload["tax_year"]
        if old_year == new_year:
            return False
        for item in case["missing_items"]:
            item.setdefault("case_tax_year", old_year)
        for entry in case["facts"]:
            entry.setdefault("case_tax_year", old_year)
            if entry["status"] == "confirmed":
                entry["status"] = "proposed"
                for field in ("confirmed_at", "confirmed_by", "confirmed_tax_year"):
                    entry.pop(field, None)
        for entry in [*case["calculations"], *case["return_items"]]:
            entry.setdefault("case_tax_year", old_year)
            entry.update(status="needs_recheck", recheck_for_tax_year=new_year)
        case.update(tax_year=new_year, previous_tax_year=old_year, tax_year_changed_at=_utc_now())
        return True
    if operation == "update_reference_release":
        old_release, new_release = case["reference_release"], payload["reference_release"]
        if old_release == new_release:
            return False
        for entry in case["evidence"]:
            entry.setdefault("case_reference_release", old_release)
            entry.update(status="needs_recheck", recheck_for_reference_release=new_release)
        for entry in [*case["calculations"], *case["return_items"]]:
            entry.setdefault("case_reference_release", old_release)
            entry.update(status="needs_recheck", recheck_for_reference_release=new_release)
        case.update(
            reference_release=new_release,
            previous_reference_release=old_release,
            reference_release_changed_at=_utc_now(),
        )
        return True
    if operation in {"confirm_fact", "confirm_facts"}:
        identifiers = [payload["fact_id"]] if operation == "confirm_fact" else payload["fact_ids"]
        # Validate the entire selection before changing any proposed fact.
        entries = [_entry(case["facts"], identifier) for identifier in identifiers]
        if any(entry["status"] == "rejected" for entry in entries):
            raise TaxCaseError("A rejected fact requires a corrected proposal.")
        pending = [entry for entry in entries if entry["status"] != "confirmed"]
        if not pending:
            return False
        confirmed_at = _utc_now()
        for entry in pending:
            entry.setdefault("case_tax_year", case["tax_year"])
            entry.update(
                status="confirmed",
                confirmed_at=confirmed_at,
                confirmed_by="user",
                confirmed_tax_year=case["tax_year"],
            )
        return True
    if operation == "reject_fact":
        entry = _entry(case["facts"], payload["fact_id"])
        if entry["status"] == "rejected":
            return False
        entry.setdefault("case_tax_year", case["tax_year"])
        for field in ("confirmed_at", "confirmed_by", "confirmed_tax_year"):
            entry.pop(field, None)
        entry.update(
            status="rejected",
            rejected_at=_utc_now(),
            rejected_by="user",
            rejected_tax_year=case["tax_year"],
        )
        return True
    if operation == "resolve_missing_item":
        entry = _entry(case["missing_items"], payload["id"])
        if entry["status"] == "resolved":
            return False
        entry.update(status="resolved", resolved_at=_utc_now(), resolved_by="user")
        return True
    category = _APPEND_CATEGORIES[operation]
    entries = case[category]
    superseded_entries = []
    for entry_id in payload.get("supersedes", []):
        previous = _entry(entries, entry_id)
        if previous.get("status") != "needs_recheck":
            raise TaxCaseError("Only entries awaiting recheck may be superseded.")
        superseded_entries.append(previous)
    if operation == "propose_fact" and payload["provenance"] is not None:
        source = payload["provenance"]
        matching = next(
            (item for item in case["files"] if item["file_id"] == source["file_id"]), None
        )
        if matching is not None and matching["checksum"] != source["checksum"]:
            raise TaxCaseError("Fact provenance conflicts with the pinned case file checksum.")
        if matching is None:
            if len(case["files"]) >= MAX_CATEGORY_ENTRIES:
                raise TaxCaseError("The tax case file inventory has reached its entry limit.")
            # The bound service validates ownership, canonical membership, original
            # checksum and the exact extracted passage before this operation.
            case["files"].append(
                FileSelection(
                    file_id=source["file_id"],
                    checksum=source["checksum"],
                    revision=source["checksum"],
                ).model_dump(mode="json")
            )
    if operation == "record_evidence":
        if (
            payload["catalog"] == "current"
            and payload["reference_release"] != case["reference_release"]
        ):
            raise TaxCaseError("Current reference evidence must use the pinned case release.")
        for evidence_id in payload["conflicts_with"]:
            _entry(case["evidence"], evidence_id)
        for evidence_id in payload["supersedes"]:
            if evidence_id in payload["conflicts_with"]:
                raise TaxCaseError("Conflicting evidence cannot also be superseded.")
    if operation in {"record_calculation", "propose_return_item"}:
        for fact_id in payload["fact_ids"]:
            _entry(case["facts"], fact_id)
        for evidence_id in payload["evidence_ids"]:
            evidence = _entry(case["evidence"], evidence_id)
            if not _current_evidence(case, evidence):
                raise TaxCaseError("A tax case dependency requires rechecked reference evidence.")
    content = {
        key: value for key, value in payload.items() if key not in {"operation_key", "supersedes"}
    }
    legacy_digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    year_scoped = category in {"facts", "calculations", "return_items", "missing_items"}
    release_scoped = category in {"evidence", "calculations", "return_items"}
    payload_content = dict(content)
    if year_scoped:
        content["case_tax_year"] = case["tax_year"]
    prior_digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if release_scoped:
        content["case_reference_release"] = case["reference_release"]
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    key = payload.get("operation_key")
    for entry in entries:
        same_year = (
            not year_scoped or entry.get("case_tax_year", case["tax_year"]) == case["tax_year"]
        )
        same_release = (
            not release_scoped
            or entry.get("case_reference_release", case["reference_release"])
            == case["reference_release"]
        )
        needs_recheck = release_scoped and entry.get("status") == "needs_recheck"
        equivalent = all(entry.get(field) == value for field, value in payload_content.items())
        if (
            needs_recheck
            and category in {"calculations", "return_items"}
            and entry.get("recheck_for_reference_release")
        ):
            equivalent = all(
                entry.get(field) == value
                for field, value in payload_content.items()
                if field != "evidence_ids"
            )
        if needs_recheck and equivalent:
            superseded_entries.append(entry)
        if key is not None and entry.get("operation_key") == key:
            if (
                entry["operation_digest"] not in {digest, prior_digest, legacy_digest}
                or not same_year
                or not same_release
                or needs_recheck
                or entry.get("supersedes", []) != payload.get("supersedes", [])
            ):
                raise TaxCaseConflictError(
                    "The operation key was already used with different data or a different review context."
                )
            return _link_superseded(superseded_entries, entry["id"])
        if (
            same_year
            and same_release
            and not needs_recheck
            and entry["operation_digest"] in {digest, prior_digest, legacy_digest}
        ):
            return _link_superseded(superseded_entries, entry["id"])
    if len(entries) >= MAX_CATEGORY_ENTRIES:
        raise TaxCaseError("The tax case category has reached its entry limit.")
    entry = {
        "id": _new_id(),
        **payload,
        "operation_digest": digest,
        "created_at": _utc_now(),
        "created_by": actor,
    }
    if year_scoped:
        entry["case_tax_year"] = case["tax_year"]
    if release_scoped:
        entry["case_reference_release"] = case["reference_release"]
    if category in {"facts", "return_items"}:
        entry["status"] = "proposed"
    elif category in {"calculations", "evidence"}:
        entry["status"] = "recorded"
    elif category == "missing_items":
        entry["status"] = "unresolved"
    entries.append(entry)
    _link_superseded(superseded_entries, entry["id"])
    return True


def _write_current_fact(
    case: dict[str, Any], operation: str, payload: dict[str, Any], actor: str,
    statement: TaxUserStatement | None,
) -> bool:
    """Replace a current fact atomically; retained history is not another active row."""
    fact_id = payload.get("fact_id")
    if fact_id is not None:
        fact = _entry(case["facts"], fact_id)
    else:
        # Repeating the same quoted statement should not create duplicate rows.
        fact = next((item for item in case["facts"] if item.get("label") == payload["label"]
                     and item.get("value") == payload["value"]
                     and item.get("case_tax_year", case["tax_year"]) == case["tax_year"]
                     and item.get("status") != "rejected"), None)
    if fact is None and fact_id is None and any(
        item.get("label") == payload["label"] and item.get("status") != "rejected"
        and item.get("case_tax_year", case["tax_year"]) == case["tax_year"]
        for item in case["facts"]
    ):
        raise TaxCaseError("A current fact already has this label. Use its fact_id to correct it.")
    addressed = [_entry(case["missing_items"], identifier)
                 for identifier in payload.get("addresses", [])]
    if any(item.get("case_tax_year", case["tax_year"]) != case["tax_year"]
           for item in addressed):
        raise TaxCaseError("Only current-year missing items may be addressed.")
    if fact is not None and fact.get("case_tax_year", case["tax_year"]) != case["tax_year"]:
        raise TaxCaseError("A prior-year fact needs a new current-year record.")
    source = statement.message_id if operation == "record_user_fact" and statement else None
    confirmer = "user_message" if source else "user"
    fact_changed = fact is None or any(fact.get(key) != payload[key] for key in ("label", "value"))
    fact_changed = fact_changed or fact.get("status") != "confirmed"
    changed = fact_changed or any(
        item.get("status") != "resolved" for item in addressed
    )
    if not changed:
        return False
    now = _utc_now()
    if fact is None:
        if len(case["facts"]) >= MAX_CATEGORY_ENTRIES:
            raise TaxCaseError("The tax case category has reached its entry limit.")
        fact = {"id": _new_id(), "created_at": now, "created_by": actor,
                "case_tax_year": case["tax_year"]}
        case["facts"].append(fact)
    elif fact_changed:
        previous = {key: value for key, value in fact.items() if key != "history"}
        fact.setdefault("history", []).append(previous)
        for key in ("provenance", "rejected_at", "rejected_by", "rejected_tax_year",
                    "operation_key", "source_message_id"):
            fact.pop(key, None)
        for item in [*case["calculations"], *case["return_items"]]:
            if fact["id"] in item.get("fact_ids", []):
                item.update(status="needs_recheck", recheck_for_fact_ids=[fact["id"]])
    if fact_changed:
        fact.update(label=payload["label"], value=payload["value"], kind="user_statement",
                    provenance=None, status="confirmed", confirmed_at=now, confirmed_by=confirmer,
                    confirmed_tax_year=case["tax_year"], updated_at=now)
        if source:
            fact["source_message_id"] = source
        fact["operation_digest"] = hashlib.sha256(
            json.dumps({key: fact[key] for key in ("label", "value", "kind", "case_tax_year")},
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    for item in addressed:
        if item.get("status") != "resolved":
            item.update(status="resolved", resolved_at=now, resolved_by="user_message",
                        resolved_with_fact_id=fact["id"], source_message_id=source)
    return True


def _link_superseded(entries: list[dict[str, Any]], replacement_id: str) -> bool:
    changed = False
    for previous in entries:
        if previous.get("superseded_by") != replacement_id:
            previous["superseded_by"] = replacement_id
            changed = True
    return changed


def _terminal_replacement(
    entry: dict[str, Any], entries: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Follow bounded retained history without treating a stale intermediate as current."""
    visited = {entry["id"]}
    replacement = None
    while entry.get("superseded_by"):
        replacement_id = entry["superseded_by"]
        if replacement_id in visited or replacement_id not in entries:
            return None
        visited.add(replacement_id)
        replacement = entry = entries[replacement_id]
    return replacement


def _current_evidence(case: dict[str, Any], entry: dict[str, Any]) -> bool:
    return (
        entry.get("status") != "needs_recheck"
        and not entry.get("superseded_by")
        and entry.get("case_reference_release", case["reference_release"])
        == case["reference_release"]
        and (
            entry.get("catalog", "current") != "current"
            or entry.get("reference_release") == case["reference_release"]
        )
    )


def _entry(entries: list[dict[str, Any]], entry_id: str) -> dict[str, Any]:
    for entry in entries:
        if entry["id"] == entry_id:
            return entry
    raise TaxCaseError("A tax case dependency was not found in this case.")


def _encode(case: dict[str, Any]) -> str:
    encoded = json.dumps(
        case, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    if len(encoded.encode("utf-8")) > MAX_CASE_BYTES:
        raise TaxCaseError("The tax case exceeds its storage limit.")
    return encoded


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _new_id() -> str:
    return f"tax_{uuid.uuid4().hex}"


def _current_review_work(case: dict[str, Any], entry: dict[str, Any]) -> bool:
    return (
        entry.get("status") in {"recorded", "proposed"}
        and not entry.get("superseded_by")
        and not entry.get("recheck_for_tax_year")
        and not entry.get("recheck_for_reference_release")
        and entry.get("case_tax_year", case["tax_year"]) == case["tax_year"]
        and entry.get("case_reference_release", case["reference_release"])
        == case["reference_release"]
    )


def _review_support(
    case: dict[str, Any], review: dict[str, Any], *, strict: bool
) -> tuple[list[str], list[str]]:
    """Inspect only retained support; never infer the truth of a written finding."""
    limitations: list[str] = []
    needs_input: list[str] = []

    def add(target: list[str], message: str) -> None:
        if message not in target:
            target.append(message)

    def unsupported(message: str) -> None:
        if strict:
            raise TaxCaseError(message)
        add(limitations, message)

    records = {
        field: {entry["id"]: entry for entry in case.get(category, [])}
        for field, category in (
            ("fact_ids", "facts"),
            ("evidence_ids", "evidence"),
            ("calculation_ids", "calculations"),
            ("return_item_ids", "return_items"),
        )
    }
    selected = {field: set() for field in records}
    for finding in review.get("findings", []):
        if not any(finding.get(field) for field in records):
            add(limitations, "A finding has no linked saved support.")
        for field in records:
            selected[field].update(finding.get(field, []))

    for field in ("calculation_ids", "return_item_ids"):
        for identifier in sorted(selected[field]):
            item = records[field].get(identifier)
            if item is None:
                unsupported("A review dependency was not found in this case.")
                continue
            if not _current_review_work(case, item):
                unsupported("A review dependency requires rechecked calculations or return items.")
            for dependency in ("fact_ids", "evidence_ids"):
                selected[dependency].update(item.get(dependency, []))
                if not item.get(dependency):
                    add(
                        limitations,
                        "A saved calculation or return item lacks linked facts or official evidence.",
                    )

    for identifier in sorted(selected["fact_ids"]):
        fact = records["fact_ids"].get(identifier)
        if fact is None:
            unsupported("A review dependency was not found in this case.")
        elif fact.get("status") == "rejected":
            unsupported("A review cannot use a rejected fact as support.")
        elif fact.get("status") != "confirmed":
            add(needs_input, "Some saved facts still need your confirmation.")

    evidence_count = 0
    for identifier in sorted(selected["evidence_ids"]):
        evidence = records["evidence_ids"].get(identifier)
        if evidence is None:
            unsupported("A review dependency was not found in this case.")
            continue
        evidence_count += 1
        if not _current_evidence(case, evidence):
            unsupported("A review dependency requires rechecked reference evidence.")
        if evidence.get("tax_year") is None:
            add(
                limitations,
                "Tax-year applicability is unknown for some retained official evidence.",
            )
        elif evidence.get("tax_year") != case["tax_year"]:
            add(limitations, "Some retained official evidence applies to a different tax year.")
        if evidence.get("tax_year_status") != "verified":
            add(
                limitations,
                "Tax-year applicability has not been verified for some retained official evidence.",
            )
        if evidence.get("jurisdiction") != case.get("jurisdiction"):
            add(limitations, "Some retained official evidence concerns a different jurisdiction.")
        if evidence.get("conflicts_with"):
            add(limitations, "Conflicting official evidence remains unresolved.")
        if not all(
            evidence.get(key)
            for key in ("document_id", "revision", "page", "passage", "reference_release")
        ):
            add(limitations, "Some retained official evidence is incomplete.")
    if not evidence_count:
        add(limitations, "No official rule passages are linked to this review.")
    if not selected["fact_ids"]:
        add(limitations, "No saved facts are linked to this review.")
    if any(item.get("status") == "proposed" for item in case.get("facts", [])):
        add(needs_input, "Some saved facts still need your confirmation.")
    if any(item.get("status") != "resolved" for item in case.get("missing_items", [])):
        add(needs_input, "Some requested information has not been addressed.")
    if any(item.get("unread_pages") for item in case.get("files", [])):
        add(needs_input, "Some document pages still need to be read or clarified.")
    if review.get("next_question"):
        add(needs_input, "A follow-up question remains for you.")
    return needs_input, limitations


def tax_case_review(case: dict[str, Any]) -> dict[str, Any] | None:
    """Expose a saved goal-scoped worksheet without certifying tax correctness."""
    review = case.get("review")
    if not review:
        return None
    needs_input, limitations = _review_support(case, review, strict=False)
    stale = (
        review.get("review_revision") != case.get("revision")
        or review.get("tax_year") != case.get("tax_year")
        or review.get("reference_release") != case.get("reference_release")
    )
    if stale:
        limitations.insert(
            0, "Saved information changed after this review. Ask Audrey to refresh it."
        )
    projected = dict(review)
    verification = review.get("verification")
    if verification is not None:
        # Keep the retained report intact; a changed case cannot make an old
        # assessment look current merely because its candidate text is unchanged.
        projected["verification"] = {**verification, "stale": stale}
        if verification.get("status") != "checked":
            limitations.append(_verification_limitation(verification))
            for finding in verification.get("findings", []):
                if finding.get("verdict") != "supported" and finding.get("reason"):
                    message = f"Finding {finding['finding_index'] + 1}: {finding['reason']}"
                    if message not in limitations:
                        limitations.append(message)
    return {
        **projected,
        "status": "needs_input"
        if needs_input
        else "limited_evidence"
        if limitations
        else "review_saved",
        "stale": stale,
        "limitations": [*needs_input, *limitations],
        "document_scope": "Selected passages only. Remaining document pages have not been fully reviewed.",
        "notice": "This saved review does not establish that a tax return is complete or legally correct.",
    }


def _verification_limitation(verification: dict[str, Any]) -> str:
    reasons = {
        "needs_user_input": "Worksheet checking needs additional information from you.",
        "evidence_gap": "Some worksheet findings lack sufficient saved evidence.",
        "reviewer_unavailable": "The local worksheet check could not finish.",
        "verification_limit": "Worksheet checking reached its bounded review limit.",
        "no_new_evidence": "Further worksheet checking found no additional saved evidence.",
    }
    return reasons.get(
        verification.get("stop_reason"),
        "Some worksheet findings could not be confirmed from saved evidence.",
    )


def tax_case_checklist(case: dict[str, Any]) -> dict[str, Any]:
    """Describe versioned coverage gaps without deciding tax correctness."""
    facts = case.get("facts", [])
    evidence = case.get("evidence", [])
    facts_by_id = {item["id"]: item for item in facts}
    evidence_by_id = {item["id"]: item for item in evidence}
    checks: list[dict[str, Any]] = []

    def check(identifier: str, label: str, issues: list[str]) -> None:
        checks.append(
            {
                "id": identifier,
                "label": label,
                "status": "needs_information" if issues else "passed",
                "issues": issues,
            }
        )

    scope_issues = []
    if type(case.get("tax_year")) is not int or not 1900 <= case["tax_year"] <= 2100:
        scope_issues.append("Set the return tax year explicitly.")
    if not case.get("jurisdiction") or not case.get("goal"):
        scope_issues.append("Set the jurisdiction and review goal.")
    if not case.get("reference_release"):
        scope_issues.append("Pin the official reference release.")
    check("scope", "Return year and scope", scope_issues)
    active_facts = [fact for fact in facts if fact.get("status") != "rejected"]
    fact_issues = [] if active_facts else ["Record the facts required for this review."]
    source_issues = []
    for fact in active_facts:
        if fact.get("status") != "confirmed":
            fact_issues.append(f"User confirmation is needed for fact {fact['id']}.")
        if fact.get("kind") == "extracted":
            source = fact.get("provenance") or {}
            if not all(source.get(key) for key in ("file_id", "checksum", "page", "original_text")):
                source_issues.append(f"Document provenance is missing for fact {fact['id']}.")
            elif not any(
                item.get("file_id") == source["file_id"]
                and item.get("checksum") == source["checksum"]
                for item in case.get("files", [])
            ):
                source_issues.append(f"The pinned source is missing for fact {fact['id']}.")
        elif fact.get("kind") != "user_statement":
            source_issues.append(f"The source kind is unknown for fact {fact['id']}.")
    check("facts", "User confirmation of facts", fact_issues)
    check("fact_sources", "Fact provenance", source_issues)
    rule_issues = (
        [] if evidence else ["Read and retain official rule evidence for the return year."]
    )
    for item in evidence:
        replacement = _terminal_replacement(item, evidence_by_id)
        if replacement and _current_evidence(case, replacement):
            continue
        if not _current_evidence(case, item):
            rule_issues.append(
                f"Recheck evidence {item['id']} against the selected reference release."
            )
        if item.get("tax_year") is None:
            rule_issues.append(f"Tax-year applicability is unknown for evidence {item['id']}.")
        elif item["tax_year"] != case.get("tax_year"):
            rule_issues.append(f"Evidence {item['id']} applies to a different tax year.")
        if item.get("tax_year_status") != "verified":
            rule_issues.append(
                f"Tax-year applicability requires verification for evidence {item['id']}."
            )
        if item.get("jurisdiction") != case.get("jurisdiction"):
            rule_issues.append(f"Evidence {item['id']} has a different jurisdiction.")
        if not all(
            item.get(key)
            for key in ("document_id", "revision", "page", "passage", "reference_release")
        ):
            rule_issues.append(
                f"Official document support is incomplete for evidence {item['id']}."
            )
        if item.get("catalog", "current") == "current" and item.get(
            "reference_release"
        ) != case.get("reference_release"):
            rule_issues.append(f"Evidence {item['id']} uses a different reference release.")
        if item.get("conflicts_with"):
            rule_issues.append(f"Conflicting rule evidence remains unresolved for {item['id']}.")
    check("rule_evidence", "Dated official rule evidence", rule_issues)
    missing_issues = [
        f"Resolve missing item {item['id']}: {item['label']}"
        for item in case.get("missing_items", [])
        if item.get("status") != "resolved"
    ]
    for item in case.get("files", []):
        if item.get("unread_pages"):
            missing_issues.append(f"Read or clarify uncertain pages for file {item['file_id']}.")
    check("missing_items", "Missing records and uncertain pages", missing_issues)
    duplicate_issues = []
    files_by_checksum: dict[str, str] = {}
    for item in case.get("files", []):
        checksum = item.get("checksum")
        if checksum and checksum in files_by_checksum:
            duplicate_issues.append(
                f"Files {files_by_checksum[checksum]} and {item['file_id']} have identical content; check for duplicate records."
            )
        elif checksum:
            files_by_checksum[checksum] = item["file_id"]
    check("duplicate_records", "Duplicate document records", duplicate_issues)
    dependency_issues = []
    review_items = [*case.get("return_items", []), *case.get("calculations", [])]
    review_items_by_id = {item["id"]: item for item in review_items}
    for item in review_items:
        replacement = _terminal_replacement(item, review_items_by_id) or {}
        if (
            replacement.get("case_tax_year") == case.get("tax_year")
            and replacement.get("case_reference_release", case.get("reference_release"))
            == case.get("reference_release")
            and replacement.get("status") in {"recorded", "proposed"}
        ):
            continue
        if item.get("recheck_for_fact_ids"):
            dependency_issues.append(f"Recheck item {item['id']} after a saved fact changed.")
        if item.get("recheck_for_reference_release") or item.get(
            "case_reference_release", case.get("reference_release")
        ) != case.get("reference_release"):
            dependency_issues.append(
                f"Recheck item {item['id']} against the selected reference release."
            )
        if item.get("recheck_for_tax_year") or item.get(
            "case_tax_year", case.get("tax_year")
        ) != case.get("tax_year"):
            dependency_issues.append(
                f"Recheck item {item['id']} for the selected tax year; its recorded year is {item.get('case_tax_year', 'unknown')}."
            )
        if not item.get("fact_ids") or not item.get("evidence_ids"):
            dependency_issues.append(
                f"Link confirmed facts and rule evidence for item {item['id']}."
            )
        for fact_id in item.get("fact_ids", []):
            if facts_by_id.get(fact_id, {}).get("status") != "confirmed":
                dependency_issues.append(
                    f"Item {item['id']} depends on an absent or unconfirmed fact {fact_id}."
                )
        for evidence_id in item.get("evidence_ids", []):
            if evidence_id not in evidence_by_id:
                dependency_issues.append(
                    f"Item {item['id']} depends on absent evidence {evidence_id}."
                )
            elif not _current_evidence(case, evidence_by_id[evidence_id]):
                dependency_issues.append(
                    f"Item {item['id']} depends on evidence {evidence_id} that needs rechecking."
                )
    check("dependencies", "Calculation and proposed return item dependencies", dependency_issues)
    next_steps = [issue for item in checks for issue in item["issues"]]
    return {
        "checklist_version": CHECKLIST_VERSION,
        "status": "needs_information" if next_steps else "ready_for_review",
        "checks": checks,
        "next_steps": next_steps,
        "document_scope": "Selected passages only. Remaining document pages have not been fully reviewed.",
        "notice": "Review readiness describes recorded evidence coverage. It does not establish that a tax return is complete or legally correct.",
    }


def validate_tax_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic coverage validation; no tax-law engine is involved."""
    return tax_case_checklist(case)


__all__ = [
    "TaxCasesRepository",
    "TaxCaseError",
    "TaxCaseConflictError",
    "TaxCaseAccessError",
    "FactProvenance",
    "FileSelection",
    "ProposeFactPayload",
    "ConfirmFactPayload",
    "RejectFactPayload",
    "ChangeYearPayload",
    "UpdateReferenceReleasePayload",
    "RecordEvidencePayload",
    "RecordCalculationPayload",
    "AddMissingItemPayload",
    "ResolveMissingItemPayload",
    "ProposeReturnItemPayload",
    "ReviewFindingPayload",
    "SaveReviewPayload",
    "OPERATION_MODELS",
    "validate_operation",
    "tax_case_checklist",
    "tax_case_review",
    "validate_tax_case",
]
