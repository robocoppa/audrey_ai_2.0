"""Owner-bound research tasks with durable step boundaries and fenced leases.

Reservations charge uncertain model work before dispatch. Recovery reuses committed
steps and charges expired leases; it cannot promise exactly-once model billing.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Callable
from typing import Any

from audrey.app_state.transactions import write_transaction

DEFAULT_LIMITS = {
    "max_operations": 96,
    "max_active_seconds": 1800,
    "max_queued_per_user": 5,
    "max_queued_global": 50,
}
_LIMIT_BOUNDS = {
    "max_operations": (1, 1000),
    "max_active_seconds": (1, 14400),
    "max_queued_per_user": (1, 20),
    "max_queued_global": (1, 1000),
}
_STEP_LABELS = (("plan", "Plan"), ("research", "Research"), ("review", "Review"))
_PUBLIC_FIELDS = (
    "objective", "status", "revision", "created_at", "updated_at", "result", "error", "attempts",
)
_SAVED_FIELDS = (
    "status", "revision", "attempts", "fence", "lease_token", "lease_expires_at",
    "accounted_at", "active_seconds", "operations", "result", "error", "updated_at",
)


class AgentTaskError(ValueError):
    """Task input, transition or resource allowance is invalid."""


class AgentTaskConflictError(AgentTaskError):
    """The revision, lease or committed step has changed."""


class AgentTaskAccessError(AgentTaskError):
    """A task does not exist for an active owner."""


def _text(value: str, *, maximum: int, name: str, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AgentTaskError(f"Invalid {name}.")
    value = value.strip()
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise AgentTaskError(f"Invalid {name}.") from None
    if (not value and not empty) or size > maximum:
        raise AgentTaskError(f"Invalid {name}.")
    return value


def _integer(value: int, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise AgentTaskError("Invalid task limit or revision.")


def _limits(value: dict | None) -> dict:
    limits = DEFAULT_LIMITS.copy()
    if value is not None:
        if not isinstance(value, dict) or value.keys() - limits.keys():
            raise AgentTaskError("Invalid agent limits.")
        limits.update(value)
    for key, item in limits.items():
        _integer(item, *_LIMIT_BOUNDS[key])
    return limits


def _stamp(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat()


class AgentsRepository:
    """Serialize short writes; authenticate owner again at every worker boundary."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn, self._lock = connection, lock
        self._clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC)

    def _transaction(self, work: Callable[[], Any]) -> Any:
        with self._lock, write_transaction(self._conn):
            return work()

    def _owner(self, user_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM app_users u WHERE u.user_id=? AND u.status='active' "
            "AND NOT EXISTS (SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)",
            (user_id,),
        ).fetchone()
        if row is None:
            raise AgentTaskAccessError("Agent task is unavailable.")

    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["steps"] = json.loads(record.pop("steps_json"))
        record["limits"] = json.loads(record.pop("limits_json"))
        return record

    @staticmethod
    def _public(record: dict) -> dict:
        return {
            "id": record["task_id"],
            **{key: record[key] for key in _PUBLIC_FIELDS},
            "steps": [dict(step) for step in record["steps"]],
        }

    def _internal(self, record: dict) -> dict:
        return {
            **self._public(record),
            **{key: record[key] for key in (
                "user_id", "lease_token", "lease_expires_at", "fence", "limits",
                "operations", "active_seconds",
            )},
            "current_step": next(
                (step["id"] for step in record["steps"] if step["status"] != "completed"), "",
            ),
        }

    def _load(self, task_id: str, user_id: str | None = None) -> dict:
        row = self._conn.execute(
            "SELECT * FROM app_agent_tasks WHERE task_id=? AND (? IS NULL OR user_id=?)",
            (task_id, user_id, user_id),
        ).fetchone()
        if row is None:
            raise AgentTaskAccessError("Agent task is unavailable.")
        record = self._record(row)
        self._owner(record["user_id"])
        return record

    def _save(self, record: dict, *, revision: bool = True) -> None:
        if revision:
            record["revision"] += 1
            record["updated_at"] = _stamp(self._clock())
        steps_json = json.dumps(record["steps"], ensure_ascii=False)
        if len(steps_json.encode("utf-8")) > 200000:
            raise AgentTaskError("Agent saved step storage allowance exhausted.")
        self._conn.execute(
            "UPDATE app_agent_tasks SET status=?,revision=?,attempts=?,fence=?,lease_token=?,"
            "lease_expires_at=?,accounted_at=?,active_seconds=?,operations=?,result=?,error=?,"
            "updated_at=?,steps_json=? WHERE task_id=?",
            (*[record[key] for key in _SAVED_FIELDS],
             steps_json, record["task_id"]),
        )

    def _account(self, record: dict) -> None:
        if record["accounted_at"] is None:
            return
        start = dt.datetime.fromisoformat(record["accounted_at"])
        end = max(start, min(self._clock(), dt.datetime.fromisoformat(record["lease_expires_at"])))
        record["active_seconds"] += max(0.0, (end - start).total_seconds())
        record["accounted_at"] = _stamp(end)

    @staticmethod
    def _clear_lease(record: dict) -> None:
        record.update(lease_token="", lease_expires_at=None, accounted_at=None)
        for step in record["steps"]:
            if step["status"] == "running":
                step["status"] = "pending"

    def _lease(self, task_id: str, lease_token: str, fence: int) -> dict:
        record = self._load(task_id)
        if (
            record["status"] != "running" or not lease_token
            or record["lease_token"] != lease_token or record["fence"] != fence
            or dt.datetime.fromisoformat(record["lease_expires_at"]) <= self._clock()
        ):
            raise AgentTaskConflictError("Agent task lease changed or expired.")
        return record

    @staticmethod
    def _budget(record: dict) -> None:
        if record["active_seconds"] >= record["limits"]["max_active_seconds"]:
            raise AgentTaskError("Agent active execution allowance exhausted.")

    def _queue_capacity(self, user_id: str, limits: dict) -> None:
        queued = self._conn.execute(
            "SELECT COUNT(*) AS total,COALESCE(SUM(user_id=?),0) AS owned "
            "FROM app_agent_tasks WHERE status IN ('queued','running','paused')",
            (user_id,),
        ).fetchone()
        if (
            queued["owned"] >= limits["max_queued_per_user"]
            or queued["total"] >= limits["max_queued_global"]
        ):
            raise AgentTaskError("Agent task capacity reached.")

    async def create(
        self, *, user_id: str, objective: str, request_id: str, limits: dict | None = None,
    ) -> dict:
        objective = _text(objective, maximum=8000, name="objective")
        request_id = _text(request_id, maximum=200, name="request identity")
        limits = _limits(limits)

        def work() -> dict:
            self._owner(user_id)
            existing = self._conn.execute(
                "SELECT * FROM app_agent_tasks WHERE user_id=? AND request_id=?",
                (user_id, request_id),
            ).fetchone()
            if existing is not None:
                record = self._record(existing)
                if record["objective"] != objective:
                    raise AgentTaskConflictError("Request identity already used for another objective.")
                return self._public(record)
            self._queue_capacity(user_id, limits)
            saved = self._conn.execute(
                "SELECT COUNT(*) AS total,COALESCE(SUM(user_id=?),0) AS owned "
                "FROM app_agent_tasks", (user_id,),
            ).fetchone()
            if saved["owned"] >= 200 or saved["total"] >= 5000:
                raise AgentTaskError("Agent task capacity reached.")
            task_id, now = str(uuid.uuid4()), _stamp(self._clock())
            steps = [{"id": key, "label": label, "status": "pending", "result": ""}
                     for key, label in _STEP_LABELS]
            self._conn.execute(
                "INSERT INTO app_agent_tasks (task_id,user_id,request_id,objective,status,"
                "limits_json,steps_json,created_at,updated_at) VALUES (?,?,?,?,'queued',?,?,?,?)",
                (task_id, user_id, request_id, objective, json.dumps(limits), json.dumps(steps), now, now),
            )
            return self._public(self._load(task_id, user_id))

        return await asyncio.to_thread(self._transaction, work)

    async def list(self, *, user_id: str, limit: int = 50, summary: bool = False) -> list[dict]:
        _integer(limit, 1, 200)
        if type(summary) is not bool:
            raise AgentTaskError("Invalid task summary option.")

        def work() -> list[dict]:
            with self._lock:
                self._owner(user_id)
                if summary:
                    rows = self._conn.execute(
                        "SELECT task_id,objective,status,revision,created_at,updated_at,"
                        "'' AS result,error,attempts,"
                        "json_extract(steps_json,'$[0].status') AS plan_status,"
                        "json_extract(steps_json,'$[1].status') AS research_status,"
                        "json_extract(steps_json,'$[2].status') AS review_status "
                        "FROM app_agent_tasks WHERE user_id=? "
                        "ORDER BY created_at DESC,task_id DESC LIMIT ?", (user_id, limit),
                    ).fetchall()
                    return [{
                        "id": row["task_id"],
                        **{key: row[key] for key in _PUBLIC_FIELDS},
                        "summary": True,
                        "steps": [{
                            "id": key, "label": label, "status": row[f"{key}_status"], "result": "",
                        } for key, label in _STEP_LABELS],
                    } for row in rows]
                return [self._public(self._record(row)) for row in self._conn.execute(
                    "SELECT * FROM app_agent_tasks WHERE user_id=? "
                    "ORDER BY created_at DESC,task_id DESC LIMIT ?", (user_id, limit),
                ).fetchall()]

        return await asyncio.to_thread(work)

    async def get(self, *, user_id: str, task_id: str) -> dict:
        def work() -> dict:
            with self._lock:
                return self._public(self._load(task_id, user_id))

        return await asyncio.to_thread(work)

    async def delete(self, *, user_id: str, task_id: str, expected_revision: int) -> bool:
        """Erase terminal history and its receipts; cancel running work first."""
        _integer(expected_revision, 1, 2**63 - 1)

        def work() -> bool:
            record = self._load(task_id, user_id)
            if record["revision"] != expected_revision:
                raise AgentTaskConflictError("Agent task changed. Refresh before trying again.")
            if record["status"] not in {"completed", "cancelled", "failed"}:
                raise AgentTaskConflictError("Cancel active agent work before deleting it.")
            self._conn.execute("DELETE FROM app_agent_tasks WHERE task_id=?", (task_id,))
            return True

        return await asyncio.to_thread(self._transaction, work)

    async def claim(self, *, lease_seconds: int = 60, max_attempts: int = 6) -> dict | None:
        _integer(lease_seconds, 1, 3600)
        _integer(max_attempts, 1, 100)

        def work() -> dict | None:
            now = self._clock()
            rows = self._conn.execute(
                "SELECT t.* FROM app_agent_tasks t JOIN app_users u ON u.user_id=t.user_id "
                "WHERE u.status='active' AND NOT EXISTS (SELECT 1 FROM account_deletion_requests d "
                "WHERE d.user_id=u.user_id) AND (t.status='queued' OR "
                "(t.status='running' AND t.lease_expires_at<=?)) AND NOT EXISTS "
                "(SELECT 1 FROM app_agent_tasks other WHERE other.user_id=t.user_id "
                "AND other.status='running' AND other.task_id!=t.task_id) "
                "ORDER BY t.created_at,t.task_id LIMIT 1000", (_stamp(now),),
            ).fetchall()
            for row in rows:
                record = self._record(row)
                self._account(record)
                if (
                    record["attempts"] >= max_attempts
                    or record["active_seconds"] >= record["limits"]["max_active_seconds"]
                ):
                    record.update(status="failed", error="Agent execution allowance exhausted.")
                    self._clear_lease(record)
                    self._save(record)
                    continue
                record.update(
                    status="running", attempts=record["attempts"] + 1, fence=record["fence"] + 1,
                    lease_token=secrets.token_urlsafe(32),
                    lease_expires_at=_stamp(now + dt.timedelta(seconds=lease_seconds)),
                    accounted_at=_stamp(now), error="",
                )
                for step in record["steps"]:
                    if step["status"] != "completed":
                        step["status"] = "running"
                        break
                self._save(record)
                return self._internal(record)
            return None

        return await asyncio.to_thread(self._transaction, work)

    claim_next = claim

    async def validate_lease(self, *, task_id: str, lease_token: str, fence: int) -> bool:
        def work() -> bool:
            with self._lock:
                record = self._lease(task_id, lease_token, fence)
                self._account(record)
                self._budget(record)
                return True

        return await asyncio.to_thread(work)

    async def heartbeat(
        self, *, task_id: str, lease_token: str, fence: int, lease_seconds: int = 60,
    ) -> dict:
        _integer(lease_seconds, 1, 3600)

        def work() -> dict:
            record = self._lease(task_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            record["lease_expires_at"] = _stamp(max(
                dt.datetime.fromisoformat(record["lease_expires_at"]),
                self._clock() + dt.timedelta(seconds=lease_seconds),
            ))
            self._save(record, revision=False)
            return self._internal(record)

        return await asyncio.to_thread(self._transaction, work)

    async def reserve_operation(
        self, *, task_id: str, lease_token: str, fence: int, step_id: str,
    ) -> str:
        def work() -> str:
            record = self._lease(task_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            if step_id not in {key for key, _ in _STEP_LABELS} or self._internal(record)["current_step"] != step_id:
                raise AgentTaskConflictError("Agent step changed.")
            if record["operations"] >= record["limits"]["max_operations"]:
                raise AgentTaskError("Agent operation allowance exhausted.")
            operation_id = str(uuid.uuid4())
            self._conn.execute(
                "INSERT INTO app_agent_operations "
                "(operation_id,task_id,fence,step_id,status,created_at) VALUES (?,?,?,?,'reserved',?)",
                (operation_id, task_id, fence, step_id, _stamp(self._clock())),
            )
            record["operations"] += 1
            self._save(record)
            return operation_id

        return await asyncio.to_thread(self._transaction, work)

    async def checkpoint(
        self, *, task_id: str, lease_token: str, fence: int, step_id: str, result: str,
    ) -> dict:
        result = _text(result, maximum=64000, name="step result")

        def work() -> dict:
            record = self._lease(task_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            step = next((item for item in record["steps"] if item["id"] == step_id), None)
            if step is not None and step["status"] == "completed" and step["result"] == result:
                return self._internal(record)
            if step is None or self._internal(record)["current_step"] != step_id:
                raise AgentTaskConflictError("Agent step changed.")
            if self._conn.execute(
                "SELECT 1 FROM app_agent_operations WHERE task_id=? AND fence=? AND step_id=? "
                "AND status='reserved'", (task_id, fence, step_id),
            ).fetchone() is None:
                raise AgentTaskConflictError("Reserve the agent operation before committing its result.")
            step.update(status="completed", result=result)
            next_step = next((item for item in record["steps"] if item["status"] != "completed"), None)
            if next_step is not None:
                next_step["status"] = "running"
            self._conn.execute(
                "UPDATE app_agent_operations SET status='committed',committed_at=? "
                "WHERE task_id=? AND fence=? AND step_id=?",
                (_stamp(self._clock()), task_id, fence, step_id),
            )
            self._save(record)
            return self._internal(record)

        return await asyncio.to_thread(self._transaction, work)

    async def complete(self, *, task_id: str, lease_token: str, fence: int, result: str) -> dict:
        result = _text(result, maximum=64000, name="task result")

        def work() -> dict:
            record = self._lease(task_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            if any(step["status"] != "completed" for step in record["steps"]):
                raise AgentTaskConflictError("Complete all agent steps before saving the result.")
            record.update(status="completed", result=result, error="")
            self._clear_lease(record)
            self._save(record)
            return self._public(record)

        return await asyncio.to_thread(self._transaction, work)

    async def fail(
        self, *, task_id: str, lease_token: str, fence: int, error: str, retry: bool = False,
    ) -> dict:
        error = _text(error, maximum=2000, name="task error")
        if type(retry) is not bool:
            raise AgentTaskError("Invalid task retry.")

        def work() -> dict:
            record = self._lease(task_id, lease_token, fence)
            self._account(record)
            record.update(status="queued" if retry else "failed", error=error)
            self._clear_lease(record)
            self._save(record)
            return self._public(record)

        return await asyncio.to_thread(self._transaction, work)

    async def control(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_revision: int,
        action: str,
        max_attempts: int = 6,
        admission_limits: dict | None = None,
    ) -> dict:
        _integer(expected_revision, 1, 2**63 - 1)
        _integer(max_attempts, 1, 100)
        queue_limits = _limits(admission_limits) if admission_limits is not None else None
        if action not in {"pause", "resume", "cancel", "rerun"}:
            raise AgentTaskError("Invalid agent action.")

        def work() -> dict:
            record = self._load(task_id, user_id)
            if record["revision"] != expected_revision:
                raise AgentTaskConflictError("Agent task changed. Refresh before trying again.")
            allowed = {
                "pause": {"queued", "running"},
                "resume": {"paused"},
                "cancel": {"queued", "running", "paused"},
                "rerun": {"completed", "cancelled", "failed"},
            }
            if record["status"] not in allowed[action]:
                raise AgentTaskConflictError("Agent task cannot perform that action.")
            self._account(record)
            if action == "rerun":
                self._queue_capacity(user_id, queue_limits or record["limits"])
                self._budget(record)
                if record["attempts"] >= max_attempts:
                    raise AgentTaskError("Agent attempt allowance exhausted.")
                if record["operations"] >= record["limits"]["max_operations"]:
                    raise AgentTaskError("Agent operation allowance exhausted.")
                # Continue unfinished work; a fully finished report gets fresh
                # Research/Review while its committed Plan remains reusable.
                if all(step["status"] == "completed" for step in record["steps"]):
                    for step in record["steps"]:
                        if step["id"] != "plan":
                            step.update(status="pending", result="")
                record.update(result="", error="")
            record["status"] = {
                "pause": "paused",
                "resume": "queued",
                "cancel": "cancelled",
                "rerun": "queued",
            }[action]
            record["fence"] += 1
            self._clear_lease(record)
            self._save(record)
            return self._public(record)

        return await asyncio.to_thread(self._transaction, work)
