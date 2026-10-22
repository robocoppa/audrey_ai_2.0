"""Durable Private checkpoints and native delivery; no model execution here.

Workers must construct verified local runtimes and validate live files/models
before supplying pins. This repository owns leases, revisions and committed
progress, never private reasoning traces. Native messages have explicit task bindings.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from audrey.user_data_visibility import remote_personal_reads_blocked

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Digest = Annotated[str, StringConstraints(pattern=r"\A[0-9a-f]{64}\z")]
Ids = Annotated[list[Identifier], Field(max_length=100)]
TERMINAL = frozenset({"completed", "cancelled", "failed"})


class PrivateTaskError(ValueError):
    """Invalid task input or transition."""


class PrivateTaskConflictError(PrivateTaskError):
    """A revision, lease or pinned input changed."""


class PrivateTaskLeaseExpiredError(PrivateTaskConflictError):
    """An attempt did not report progress before its lease expired."""


class PrivateTaskAccessError(PrivateTaskError):
    """An active owner and Private case are required."""


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class TaskPins(_Payload):
    model_digests: Annotated[dict[Identifier, Digest], Field(min_length=1, max_length=8)]
    policy_digest: Digest
    input_digest: Digest | None = None
    access_digest: Digest | None = None


class TaskLimits(_Payload):
    max_steps: int = Field(default=24, ge=1, le=100)
    max_wall_seconds: int = Field(default=600, ge=1, le=3600)
    max_model_tokens: int = Field(default=32000, ge=1, le=200000)
    max_tool_calls: int = Field(default=64, ge=1, le=256)


class TaskUsage(_Payload):
    model_tokens: int = Field(default=0, ge=0, le=200000)
    tool_calls: int = Field(default=0, ge=0, le=256)


class _DeliverySource(_Payload):
    source_id: Identifier
    title: Annotated[str, StringConstraints(max_length=500)]
    url: Annotated[str, StringConstraints(max_length=2048)]


class _DeliveryTool(_Payload):
    tool_call_id: Identifier
    name: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    status: Literal["succeeded", "failed", "incomplete"]
    arguments: dict[str, Any]
    result: Any | None
    error_code: Annotated[str, StringConstraints(max_length=64)]


class _DeliveryModel(_Payload):
    model: Identifier
    calls: int = Field(ge=1, le=10000)


class TaskDelivery(_Payload):
    """Bounded final native-message data, never instructions to execute later."""

    assistant_content: Annotated[str, StringConstraints(strip_whitespace=False, max_length=64000)]
    outcome: Literal["succeeded", "cancelled", "failed"]
    finish_reason: Annotated[str, StringConstraints(max_length=200)] = ""
    error_code: Annotated[str, StringConstraints(max_length=200)] = ""
    virtual_model: Annotated[str, StringConstraints(max_length=200)] = ""
    concrete_model: Annotated[str, StringConstraints(max_length=200)] = ""
    prompt_tokens: int = Field(default=0, ge=0, le=200000)
    completion_tokens: int = Field(default=0, ge=0, le=200000)
    sources: Annotated[list[_DeliverySource], Field(max_length=50)] = Field(default_factory=list)
    tool_calls: Annotated[list[_DeliveryTool], Field(max_length=50)] = Field(default_factory=list)
    models: Annotated[list[_DeliveryModel], Field(max_length=50)] = Field(default_factory=list)


class TaskCheckpoint(_Payload):
    stage: Literal["inspect", "facts", "evidence", "calculations", "review", "delivery"]
    summary: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    fact_ids: Ids = Field(default_factory=list)
    evidence_ids: Ids = Field(default_factory=list)
    calculation_ids: Ids = Field(default_factory=list)
    return_item_ids: Ids = Field(default_factory=list)
    missing_item_ids: Ids = Field(default_factory=list)


def _validated(model: type[_Payload], value: Any) -> dict:
    try:
        return model.model_validate(value).model_dump(mode="json", exclude_none=True)
    except ValidationError:
        raise PrivateTaskError("Invalid Private task payload.") from None


def _positive(value: int) -> None:
    if type(value) is not int or value < 1:
        raise PrivateTaskError("A positive revision or progress value is required.")


def _encoded(value: dict) -> str:
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise PrivateTaskError("Invalid Private task record.") from None
    if len(result.encode()) > 128 * 1024:
        raise PrivateTaskError("Private task storage limit exceeded.")
    return result


class PrivateTasksRepository:
    """Short, serialized transactions with a fresh fence for each worker lease."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn, self._lock = connection, lock
        self._clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC)

    def _transaction(self, work: Callable[[], Any]) -> Any:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = work()
                self._conn.commit()
                return result
            except BaseException:
                self._conn.rollback()
                raise

    def _case(self, user_id: str, conversation_id: str) -> tuple[dict, str | None]:
        row = self._conn.execute(
            "SELECT c.private_required,c.archived_at,c.project_id,p.private_required AS project_private,"
            "p.user_id AS project_owner,p.instructions AS project_instructions,"
            "u.storage_namespace,u.role,t.case_id,t.revision,t.record_json "
            "FROM app_conversations c JOIN app_users u ON u.user_id=c.user_id "
            "JOIN app_tax_cases t ON t.conversation_id=c.conversation_id AND t.user_id=c.user_id "
            "LEFT JOIN app_projects p ON p.project_id=c.project_id "
            "WHERE c.user_id=? AND c.conversation_id=? AND u.status='active' "
            "AND NOT EXISTS(SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)",
            (user_id, conversation_id),
        ).fetchone()
        if (
            row is None
            or not row["private_required"]
            or row["archived_at"] is not None
            or remote_personal_reads_blocked(row["storage_namespace"])
            or (
                row["project_id"] is not None
                and (not row["project_private"] or row["project_owner"] != user_id)
            )
        ):
            raise PrivateTaskAccessError("An active owner and Private case are required.")
        groups = sorted(item["group_id"] for item in self._conn.execute(
            "SELECT group_id FROM user_group_memberships WHERE user_id=?", (user_id,)
        ).fetchall())
        if row["role"] == "bot" or "bots" in groups:
            raise PrivateTaskAccessError("An active owner and Private case are required.")
        policy = self._conn.execute(
            "SELECT enabled,audience FROM model_access_policies WHERE model_id='private'"
        ).fetchone()
        profile = self._conn.execute(
            "SELECT visibility,roles_json FROM model_publication_profiles WHERE model_id='private'"
        ).fetchone()
        case = json.loads(row["record_json"])
        case.update(id=row["case_id"], revision=row["revision"])
        # The native adapter validates effective grants with the shared catalog.
        # Pin only permission-bearing state here, so a change between its guard
        # and a final case write cannot adopt the earlier permission decision.
        case["task_access_digest"] = hashlib.sha256(_encoded({
            "role": row["role"], "groups": groups, "namespace": row["storage_namespace"],
            "policy": dict(policy) if policy is not None else None,
            "profile": {"visibility": profile["visibility"],
                        "roles": sorted(json.loads(profile["roles_json"]))}
                       if profile is not None else None,
        }).encode()).hexdigest()
        project_files = (
            [
                item["file_id"]
                for item in self._conn.execute(
                    "SELECT file_id FROM app_project_files WHERE project_id=? ORDER BY file_id",
                    (row["project_id"],),
                ).fetchall()
            ]
            if row["project_id"] is not None
            else []
        )
        # Pin semantic context, so a title-only edit does not restart useful work.
        case["task_project_digest"] = hashlib.sha256(
            _encoded(
                {
                    "instructions": row["project_instructions"] or "",
                    "files": project_files,
                }
            ).encode()
        ).hexdigest()
        return case, row["project_id"]

    async def authorization_digest(self, *, user_id: str, conversation_id: str) -> str:
        """Snapshot grants for a previously catalog-authorized native task."""
        return await asyncio.to_thread(
            self._transaction,
            lambda: self._case(user_id, conversation_id)[0]["task_access_digest"],
        )

    @staticmethod
    def _access_binding(case: dict, pins: dict) -> None:
        if pins.get("access_digest") is not None and pins["access_digest"] != case["task_access_digest"]:
            raise PrivateTaskConflictError("Private task owner access changed.")

    def _load(self, user_id: str, task_id: str) -> tuple[dict, dict, str | None]:
        row = self._conn.execute(
            "SELECT * FROM app_private_tasks WHERE user_id=? AND task_id=?", (user_id, task_id)
        ).fetchone()
        if row is None:
            raise PrivateTaskAccessError("Private task not found.")
        case, project_id = self._case(user_id, row["conversation_id"])
        if case["id"] != row["case_id"]:
            raise PrivateTaskAccessError("Private task not found.")
        record = json.loads(row["record_json"])
        record.update(
            id=row["task_id"],
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            case_id=row["case_id"],
            status=row["status"],
            revision=row["revision"],
            fence=row["fence"],
            lease_token=row["lease_token"],
            lease_expires_at=row["lease_expires_at"],
        )
        record["project_id"] = project_id
        return record, case, project_id

    @staticmethod
    def _sources(case: dict, project_id: str | None) -> dict:
        return {
            "tax_year": case["tax_year"],
            "reference_release": case["reference_release"],
            "project_id": project_id,
            "project_digest": case["task_project_digest"],
            "files": sorted(case.get("files", []), key=lambda f: f["file_id"]),
        }

    @staticmethod
    def _revision(record: dict, expected: int) -> None:
        _positive(expected)
        if record["revision"] != expected:
            raise PrivateTaskConflictError("Private task changed; reload it.")

    def _bindings(
        self, record: dict, case: dict, project_id: str | None, pins: dict, case_revision: int
    ) -> None:
        _positive(case_revision)
        self._access_binding(case, pins)
        if (
            case["revision"] != case_revision
            or record["pins"] != pins
            or record["sources"] != self._sources(case, project_id)
        ):
            raise PrivateTaskConflictError(
                "Private task inputs changed; resume with fresh validation."
            )

    def _write(self, record: dict) -> dict:
        previous = record["revision"]
        record["revision"] += 1
        record["updated_at"] = self._clock().isoformat()
        cursor = self._conn.execute(
            "UPDATE app_private_tasks SET status=?,revision=?,fence=?,lease_token=?,"
            "lease_expires_at=?,record_json=?,updated_at=? WHERE task_id=? AND user_id=? AND revision=?",
            (
                record["status"],
                record["revision"],
                record["fence"],
                record["lease_token"],
                record["lease_expires_at"],
                _encoded(record),
                record["updated_at"],
                record["id"],
                record["user_id"],
                previous,
            ),
        )
        if cursor.rowcount != 1:
            raise PrivateTaskConflictError("Private task changed; reload it.")
        return record

    def _lease(self, record: dict, token: str, fence: int) -> None:
        if (
            record["status"] != "running"
            or not token
            or record["lease_token"] != token
            or type(fence) is not int
            or record["fence"] != fence
        ):
            raise PrivateTaskConflictError("Private task lease is no longer current.")
        if self._clock() >= dt.datetime.fromisoformat(record["lease_expires_at"]):
            raise PrivateTaskLeaseExpiredError("Private task lease expired without new progress.")

    @staticmethod
    def _clear_lease(record: dict) -> None:
        record.update(lease_token="", lease_expires_at=None, lease_accounted_at=None)

    def _account_time(self, record: dict, until: dt.datetime | None = None) -> None:
        if record.get("lease_accounted_at"):
            previous = dt.datetime.fromisoformat(record["lease_accounted_at"])
            now = max(previous, until or self._clock())
            record["usage"]["wall_seconds"] += (now - previous).total_seconds()
            record["lease_accounted_at"] = now.isoformat()

    @staticmethod
    def _exhausted(record: dict) -> bool:
        return any(
            record["usage"][key] >= record["limits"][limit]
            for key, limit in (
                ("steps", "max_steps"),
                ("wall_seconds", "max_wall_seconds"),
                ("model_tokens", "max_model_tokens"),
                ("tool_calls", "max_tool_calls"),
            )
        )

    @staticmethod
    def _lease_seconds(value: int) -> None:
        if type(value) is not int or not 1 <= value <= 120:
            raise PrivateTaskError("A lease must last from 1 to 120 seconds.")

    async def create(
        self,
        *,
        user_id: str,
        conversation_id: str,
        expected_case_revision: int,
        pins: dict,
        limits: dict | None = None,
        native_run_id: str | None = None,
    ) -> dict:
        if native_run_id is not None:
            native_run_id = self._native_identifier(native_run_id)
        pins, limits = (
            _validated(TaskPins, pins),
            _validated(TaskLimits, {} if limits is None else limits),
        )
        _positive(expected_case_revision)

        def work():
            case, project_id = self._case(user_id, conversation_id)
            self._access_binding(case, pins)
            if case["revision"] != expected_case_revision:
                raise PrivateTaskConflictError("The saved case changed; reload it.")
            if native_run_id is not None:
                self._native_run(user_id, conversation_id, native_run_id)
            now = self._clock().isoformat()
            record = {
                "id": "ptask_" + uuid.uuid4().hex,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "case_id": case["id"],
                "case_revision": case["revision"],
                "project_id": project_id,
                "kind": "tax_review",
                "native_run_id": native_run_id,
                "delivery": None,
                "status": "queued",
                "revision": 1,
                "fence": 0,
                "pins": pins,
                "sources": self._sources(case, project_id),
                "limits": limits,
                "usage": {"steps": 0, "wall_seconds": 0.0, "model_tokens": 0, "tool_calls": 0},
                "checkpoint": None,
                "checkpoint_sequence": 0,
                "attempts": 0,
                "progress": 0,
                "lease_token": "",
                "lease_expires_at": None,
                "lease_accounted_at": None,
                "stop_reason": "",
                "last_commit": None,
                "created_at": now,
                "updated_at": now,
            }
            try:
                self._conn.execute(
                    "INSERT INTO app_private_tasks(task_id,conversation_id,user_id,case_id,status,"
                    "revision,fence,lease_token,record_json,created_at,updated_at) "
                    "VALUES(?,?,?,?,'queued',1,0,'',?,?,?)",
                    (
                        record["id"],
                        conversation_id,
                        user_id,
                        case["id"],
                        _encoded(record),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                raise PrivateTaskConflictError("An active Private task already exists.") from None
            return record

        return await asyncio.to_thread(self._transaction, work)

    async def get(self, *, user_id: str, task_id: str) -> dict | None:
        def work():
            row = self._conn.execute(
                "SELECT 1 FROM app_private_tasks WHERE user_id=? AND task_id=?", (user_id, task_id)
            ).fetchone()
            return self._load(user_id, task_id)[0] if row else None

        return await asyncio.to_thread(self._transaction, work)

    @staticmethod
    def _native_identifier(run_id: str) -> str:
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 200:
            raise PrivateTaskError("A bounded native run id is required.")
        return run_id.strip()

    def _native_run(self, user_id: str, conversation_id: str, run_id: str,
                    task_id: str | None = None) -> None:
        row = self._conn.execute(
            "SELECT status,private_required,requested_model_id,skill_id FROM app_runs "
            "WHERE run_id=? AND user_id=? AND conversation_id=?",
            (run_id, user_id, conversation_id),
        ).fetchone()
        if row is None or not row["private_required"] or row["requested_model_id"] != "private" \
                or row["skill_id"] != "tax-review":
            raise PrivateTaskAccessError("An owned Private Tax review run is required.")
        if row["status"] != "running":
            raise PrivateTaskConflictError("Native task run is no longer active.")
        duplicate = self._conn.execute(
            "SELECT task_id FROM app_private_tasks "
            "WHERE json_extract(record_json,'$.native_run_id')=? AND task_id!=? LIMIT 1",
            (run_id, task_id or ""),
        ).fetchone()
        if duplicate is not None:
            raise PrivateTaskConflictError("Native run is already bound to a task.")

    async def bind_native_run(self, *, user_id: str, task_id: str,
                              expected_revision: int, run_id: str) -> dict:
        """Bind one trusted native attempt before execution, after revalidation."""
        run_id = self._native_identifier(run_id)

        def work():
            record, _, _ = self._load(user_id, task_id)
            self._revision(record, expected_revision)
            if record["status"] != "queued":
                raise PrivateTaskConflictError("Private task is not queued.")
            if (record.get("delivery") or {}).get("pending"):
                raise PrivateTaskConflictError("The previous task result is awaiting delivery.")
            previous = record.get("native_run_id")
            if previous and previous != run_id:
                active = self._conn.execute(
                    "SELECT 1 FROM app_runs WHERE run_id=? AND status='running'", (previous,),
                ).fetchone()
                if active is not None:
                    raise PrivateTaskConflictError("The previous native attempt is still active.")
            self._native_run(user_id, record["conversation_id"], run_id, task_id)
            if previous == run_id:
                return record
            record.update(native_run_id=run_id, delivery=None)
            return self._write(record)

        return await asyncio.to_thread(self._transaction, work)

    async def stop_unclaimed(self, *, user_id: str, task_id: str, expected_revision: int) -> bool:
        """Clean up exactly an unlaunched attempt, even if its owner was revoked."""
        def work():
            row = self._conn.execute(
                "SELECT record_json FROM app_private_tasks WHERE user_id=? AND task_id=? "
                "AND revision=? AND status='queued'", (user_id, task_id, expected_revision),
            ).fetchone()
            if row is None:
                return False
            record = json.loads(row["record_json"])
            record.update(status="paused", stop_reason="admission_failed", fence=record["fence"] + 1)
            self._clear_lease(record)
            self._write(record)
            return True

        return await asyncio.to_thread(self._transaction, work)

    async def latest_for_conversation(self, *, user_id: str, conversation_id: str) -> dict | None:
        def work():
            self._case(user_id, conversation_id)
            row = self._conn.execute(
                "SELECT task_id FROM app_private_tasks WHERE user_id=? AND conversation_id=? "
                "ORDER BY created_at DESC,rowid DESC LIMIT 1", (user_id, conversation_id),
            ).fetchone()
            return self._load(user_id, row["task_id"])[0] if row is not None else None

        return await asyncio.to_thread(self._transaction, work)

    async def get_for_run(self, *, user_id: str, run_id: str) -> dict | None:
        def work():
            row = self._conn.execute(
                "SELECT task_id FROM app_private_tasks WHERE user_id=? "
                "AND json_extract(record_json,'$.native_run_id')=? LIMIT 1", (user_id, run_id),
            ).fetchone()
            return self._load(user_id, row["task_id"])[0] if row is not None else None

        return await asyncio.to_thread(self._transaction, work)

    async def claim(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_revision: int,
        pins: dict,
        lease_seconds: int = 60,
    ) -> dict:
        pins = _validated(TaskPins, pins)
        self._lease_seconds(lease_seconds)

        def work():
            record, case, project_id = self._load(user_id, task_id)
            self._revision(record, expected_revision)
            self._bindings(record, case, project_id, pins, record["case_revision"])
            if record["status"] != "queued":
                raise PrivateTaskConflictError("Private task is not queued.")
            if (record.get("delivery") or {}).get("pending"):
                raise PrivateTaskConflictError("The previous task result is awaiting delivery.")
            if self._exhausted(record):
                raise PrivateTaskConflictError("Private task budget is exhausted.")
            now = self._clock()
            # Never let a lease authorize work beyond the remaining execution budget.
            remaining = record["limits"]["max_wall_seconds"] - record["usage"]["wall_seconds"]
            record.update(
                status="running",
                fence=record["fence"] + 1,
                lease_token=uuid.uuid4().hex,
                lease_accounted_at=now.isoformat(),
                lease_expires_at=(
                    now + dt.timedelta(seconds=min(lease_seconds, remaining))
                ).isoformat(),
                attempts=record["attempts"] + 1,
                stop_reason="",
            )
            return self._write(record)

        return await asyncio.to_thread(self._transaction, work)

    async def renew(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_revision: int,
        lease_token: str,
        fence: int,
        pins: dict,
        progress: int,
        lease_seconds: int = 60,
        usage: dict | None = None,
    ) -> dict:
        pins = _validated(TaskPins, pins)
        spent = _validated(TaskUsage, {} if usage is None else usage)
        _positive(progress)
        self._lease_seconds(lease_seconds)

        def work():
            record, case, project_id = self._load(user_id, task_id)
            self._revision(record, expected_revision)
            self._bindings(record, case, project_id, pins, record["case_revision"])
            self._lease(record, lease_token, fence)
            if progress <= record["progress"]:
                raise PrivateTaskConflictError("A lease renewal requires new work progress.")
            self._account_time(record)
            for key, value in spent.items():
                record["usage"][key] += value
            if any(
                record["usage"][key] > record["limits"][limit]
                for key, limit in (
                    ("model_tokens", "max_model_tokens"),
                    ("tool_calls", "max_tool_calls"),
                )
            ):
                record.update(
                    status="paused", stop_reason="budget_exhausted", fence=record["fence"] + 1
                )
                record["usage"]["steps"] += 1
                self._clear_lease(record)
                return self._write(record)
            remaining = record["limits"]["max_wall_seconds"] - record["usage"]["wall_seconds"]
            record.update(
                progress=progress,
                lease_expires_at=(
                    dt.datetime.fromisoformat(record["lease_accounted_at"])
                    + dt.timedelta(seconds=min(lease_seconds, remaining))
                ).isoformat(),
            )
            return self._write(record)

        return await asyncio.to_thread(self._transaction, work)

    async def checkpoint(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_revision: int,
        lease_token: str,
        fence: int,
        expected_case_revision: int,
        pins: dict,
        checkpoint_key: str,
        checkpoint: dict,
        status: str = "queued",
        usage: dict | None = None,
        delivery: dict | None = None,
    ) -> dict:
        checkpoint = _validated(TaskCheckpoint, checkpoint)
        if delivery is not None:
            delivery = _validated(TaskDelivery, delivery)
            for tool in delivery["tool_calls"]:
                tool.setdefault("result", None)
            if len(_encoded(delivery).encode()) > 96 * 1024:
                raise PrivateTaskError("Private task delivery limit exceeded.")
            if status not in {"waiting_for_user", "paused", "completed", "failed"}:
                raise PrivateTaskError("A native result requires a stopped task step.")
        pins, usage = (
            _validated(TaskPins, pins),
            _validated(TaskUsage, {} if usage is None else usage),
        )
        if status not in {"queued", "waiting_for_user", "paused", "completed", "failed"}:
            raise PrivateTaskError("Invalid Private checkpoint status.")
        if (
            not isinstance(checkpoint_key, str)
            or not checkpoint_key.strip()
            or len(checkpoint_key) > 200
        ):
            raise PrivateTaskError("A bounded checkpoint key is required.")
        receipt = {
            "key": checkpoint_key,
            "lease_token": lease_token,
            "fence": fence,
            "digest": hashlib.sha256(
                _encoded(
                    {
                        "checkpoint": checkpoint,
                        "status": status,
                        "case_revision": expected_case_revision,
                        "pins": pins,
                        "usage": usage,
                        **({"delivery": delivery} if delivery is not None else {}),
                    }
                ).encode()
            ).hexdigest(),
        }

        def work():
            record, case, project_id = self._load(user_id, task_id)
            previous = record.get("last_commit")
            if previous and previous["key"] == checkpoint_key:
                if previous == receipt:
                    return record  # Lost acknowledgement; do not count or deliver the stage twice.
                raise PrivateTaskConflictError(
                    "Checkpoint key was already used for different work."
                )
            self._revision(record, expected_revision)
            self._lease(record, lease_token, fence)
            self._bindings(record, case, project_id, pins, expected_case_revision)
            if delivery is not None:
                run_id = record.get("native_run_id")
                if not run_id:
                    raise PrivateTaskConflictError("A native task run must be bound before delivery.")
                self._native_run(user_id, record["conversation_id"], run_id, task_id)
                if (record.get("delivery") or {}).get("pending"):
                    raise PrivateTaskConflictError("The previous task result is awaiting delivery.")
            for field, category in (
                ("fact_ids", "facts"),
                ("evidence_ids", "evidence"),
                ("calculation_ids", "calculations"),
                ("return_item_ids", "return_items"),
                ("missing_item_ids", "missing_items"),
            ):
                owned = {item["id"] for item in case[category]}
                if not set(checkpoint[field]) <= owned:
                    raise PrivateTaskError("Checkpoint support must belong to the saved case.")
            self._account_time(record)
            record["usage"]["steps"] += 1
            for key, value in usage.items():
                record["usage"][key] += value
            record.update(
                checkpoint=checkpoint,
                checkpoint_sequence=record["checkpoint_sequence"] + 1,
                case_revision=case["revision"],
                status=status,
                last_commit=receipt,
                stop_reason={
                    "waiting_for_user": "user_input",
                    "paused": "worker_paused",
                    "completed": "finished",
                    "failed": "worker_failed",
                }.get(status, ""),
            )
            if self._exhausted(record) and status not in {"completed", "failed"}:
                record.update(status="paused", stop_reason="budget_exhausted")
            # A result obtained beyond a limit stays partial, never a completed task.
            if any(
                record["usage"][key] > record["limits"][limit]
                for key, limit in (
                    ("steps", "max_steps"),
                    ("wall_seconds", "max_wall_seconds"),
                    ("model_tokens", "max_model_tokens"),
                    ("tool_calls", "max_tool_calls"),
                )
            ):
                record.update(status="paused", stop_reason="budget_exhausted")
            if delivery is not None:
                record["delivery"] = {
                    "run_id": record["native_run_id"],
                    "checkpoint_sequence": record["checkpoint_sequence"],
                    "pending": True,
                    "payload": delivery,
                }
            self._clear_lease(record)
            return self._write(record)

        return await asyncio.to_thread(self._transaction, work)

    async def control(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_revision: int,
        action: str,
        expected_case_revision: int | None = None,
        pins: dict | None = None,
    ) -> dict:
        if action not in {"pause", "resume", "cancel"}:
            raise PrivateTaskError("Invalid Private task action.")
        if action == "resume":
            pins = _validated(TaskPins, pins)
            _positive(expected_case_revision)

        def work():
            record, case, project_id = self._load(user_id, task_id)
            if action == "cancel" and record["status"] == "cancelled":
                return record  # Safe retry after a lost cancellation acknowledgement.
            self._revision(record, expected_revision)
            if record["status"] in TERMINAL:
                raise PrivateTaskConflictError("Private task is already finished.")
            if action == "resume":
                if (record.get("delivery") or {}).get("pending"):
                    raise PrivateTaskConflictError("The previous task result is awaiting delivery.")
                self._access_binding(case, pins)
                if record["status"] not in {"paused", "waiting_for_user"}:
                    raise PrivateTaskConflictError("Private task is not paused.")
                if case["revision"] != expected_case_revision:
                    raise PrivateTaskConflictError("The saved case changed; reload it.")
                if self._exhausted(record):
                    raise PrivateTaskConflictError("Private task budget is exhausted.")
                sources = self._sources(case, project_id)
                if (
                    record["case_revision"] != case["revision"]
                    or record["sources"] != sources
                    or record["pins"] != pins
                ):
                    record.update(checkpoint=None, last_commit=None)
                record.update(
                    status="queued",
                    stop_reason="",
                    case_revision=case["revision"],
                    sources=sources,
                    pins=pins,
                )
            else:
                if record["status"] == "running":
                    expiry = dt.datetime.fromisoformat(record["lease_expires_at"])
                    self._account_time(record, min(self._clock(), expiry))
                record.update(
                    status="cancelled" if action == "cancel" else "paused",
                    stop_reason="owner_cancelled" if action == "cancel" else "owner_paused",
                )
            record["fence"] += 1
            self._clear_lease(record)
            return self._write(record)

        return await asyncio.to_thread(self._transaction, work)

    async def abandon(
        self,
        *,
        user_id: str,
        task_id: str,
        lease_token: str,
        fence: int,
        reason: str,
        usage: dict | None = None,
    ) -> dict | None:
        """Stop only this attempt, retaining partial work and charging it once.

        This internal cleanup does not authorize reads or execution for a revoked
        owner. It may stop their abandoned lease without opening the case again.
        """
        if reason not in {
            "step_timeout",
            "step_failed",
            "lease_expired",
            "inputs_changed",
            "access_changed",
            "budget_exhausted",
            "worker_stopped",
            "usage_unavailable",
        }:
            raise PrivateTaskError("Invalid Private task stop reason.")
        spent = _validated(TaskUsage, {} if usage is None else usage)

        def work():
            row = self._conn.execute(
                "SELECT * FROM app_private_tasks WHERE user_id=? AND task_id=?",
                (user_id, task_id),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["lease_token"] != lease_token
                or row["fence"] != fence
            ):
                return None  # Owner cancellation, deletion or a replacement already won.
            record = json.loads(row["record_json"])
            record.update(
                revision=row["revision"],
                fence=row["fence"] + 1,
                status="paused",
                stop_reason=reason,
            )
            self._account_time(
                record, min(self._clock(), dt.datetime.fromisoformat(row["lease_expires_at"]))
            )
            record["usage"]["steps"] += 1
            for key, value in spent.items():
                record["usage"][key] += value
            if self._exhausted(record):
                record["stop_reason"] = "budget_exhausted"
            self._clear_lease(record)
            return self._write(record)

        return await asyncio.to_thread(self._transaction, work)

    async def committed_delivery(self, *, user_id: str, task_id: str, run_id: str) -> dict | None:
        """Trusted native settlement lookup, independent of current owner access."""
        def work():
            row = self._conn.execute(
                "SELECT record_json FROM app_private_tasks WHERE user_id=? AND task_id=?",
                (user_id, task_id),
            ).fetchone()
            if row is None:
                return None
            delivery = json.loads(row["record_json"]).get("delivery")
            return delivery if delivery and delivery.get("run_id") == run_id else None

        return await asyncio.to_thread(self._transaction, work)

    async def pending_deliveries(self) -> list[dict]:
        """Trusted startup outbox, not an owner read or an HTTP projection.

        Revoked owners' already committed terminal attempts still need their run
        finalized before generic interrupted-run recovery. No generation occurs.
        """
        def work():
            rows = self._conn.execute(
                "SELECT record_json FROM app_private_tasks "
                "WHERE json_extract(record_json,'$.delivery.pending')=1 "
                "ORDER BY updated_at,task_id",
            ).fetchall()
            return [json.loads(row["record_json"]) for row in rows]

        return await asyncio.to_thread(self._transaction, work)

    async def ack_delivery(self, *, user_id: str, task_id: str, run_id: str,
                           checkpoint_sequence: int) -> bool:
        """Acknowledge exactly the committed result without changing task authority."""
        _positive(checkpoint_sequence)

        def work():
            row = self._conn.execute(
                "SELECT record_json FROM app_private_tasks WHERE user_id=? AND task_id=?",
                (user_id, task_id),
            ).fetchone()
            if row is None:
                return False
            record = json.loads(row["record_json"])
            delivery = record.get("delivery") or {}
            if delivery.get("run_id") != run_id \
                    or delivery.get("checkpoint_sequence") != checkpoint_sequence:
                raise PrivateTaskConflictError("Private task delivery changed.")
            if not delivery.get("pending"):
                return True
            delivery["pending"] = False
            self._conn.execute(
                "UPDATE app_private_tasks SET record_json=? WHERE user_id=? AND task_id=?",
                (_encoded(record), user_id, task_id),
            )
            return True

        return await asyncio.to_thread(self._transaction, work)

    async def recover_for_restart(self) -> int:
        """Pause all interrupted work; never dispatch it automatically after restart."""
        def work():
            now = self._clock()
            rows = self._conn.execute(
                "SELECT * FROM app_private_tasks WHERE status IN ('queued','running') "
                "ORDER BY created_at,task_id",
            ).fetchall()
            for row in rows:
                record = json.loads(row["record_json"])
                record.update(revision=row["revision"], fence=row["fence"] + 1,
                              status="paused", stop_reason="server_restart")
                if row["status"] == "running":
                    self._account_time(
                        record, min(now, dt.datetime.fromisoformat(row["lease_expires_at"])),
                    )
                self._clear_lease(record)
                self._write(record)
            return len(rows)

        return await asyncio.to_thread(self._transaction, work)

    async def recover_expired(self) -> int:
        """Pause abandoned attempts. Calling this never schedules or executes work."""

        def work():
            now = self._clock()
            rows = self._conn.execute(
                "SELECT * FROM app_private_tasks WHERE status='running' AND lease_expires_at<=? "
                "ORDER BY lease_expires_at LIMIT 100",
                (now.isoformat(),),
            ).fetchall()
            for row in rows:
                record = json.loads(row["record_json"])
                record.update(
                    revision=row["revision"],
                    fence=row["fence"] + 1,
                    status="paused",
                    stop_reason="lease_expired",
                )
                self._account_time(record, dt.datetime.fromisoformat(row["lease_expires_at"]))
                self._clear_lease(record)
                self._write(record)
            return len(rows)

        return await asyncio.to_thread(self._transaction, work)


@dataclass(slots=True)
class PrivateTaskLease:
    """Server-owned authority shared with tool callbacks and SQLite writes."""

    repository: PrivateTasksRepository
    user_id: str
    conversation_id: str
    task_id: str
    lease_token: str
    fence: int
    pins: dict
    active: bool = True
    work_active: bool = True

    def verify_locked(self) -> dict:
        if not self.active:
            raise PrivateTaskConflictError("Private task step has ended.")
        record, case, project_id = self.repository._load(self.user_id, self.task_id)
        self.repository._lease(record, self.lease_token, self.fence)
        self.repository._bindings(record, case, project_id, self.pins, record["case_revision"])
        return record

    async def verify(self) -> dict:
        return await asyncio.to_thread(self.repository._transaction, self.verify_locked)


_TASK_LEASE: ContextVar[PrivateTaskLease | None] = ContextVar(
    "audrey_private_task_lease", default=None
)


@contextmanager
def private_task_scope(
    repository: PrivateTasksRepository, task: dict
) -> Iterator[PrivateTaskLease]:
    """Invalidate inherited callbacks when the owning step exits."""
    if _TASK_LEASE.get() is not None:
        raise PrivateTaskAccessError("A Private task step cannot replace an active task scope.")
    lease = PrivateTaskLease(
        repository,
        task["user_id"],
        task["conversation_id"],
        task["id"],
        task["lease_token"],
        task["fence"],
        json.loads(_encoded(task["pins"])),
    )
    token = _TASK_LEASE.set(lease)
    try:
        yield lease
    finally:
        lease.active = False
        _TASK_LEASE.reset(token)


def guard_tax_case_write(
    connection: sqlite3.Connection, *, user_id: str, conversation_id: str, actor: str
) -> tuple[PrivateTaskLease, dict] | None:
    """Called inside the case transaction, before changing any case record."""
    lease = _TASK_LEASE.get()
    if lease is None:
        return None  # Ordinary foreground/user operations retain their existing authority.
    if (
        lease.repository._conn is not connection
        or not lease.work_active
        or lease.user_id != user_id
        or lease.conversation_id != conversation_id
        or actor != "model"
    ):
        raise PrivateTaskAccessError("Private task authority does not match this case write.")
    return lease, lease.verify_locked()
