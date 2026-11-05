"""Owner-bound agent tasks with durable step boundaries and fenced leases.

Reservations charge uncertain model work before dispatch. Recovery reuses committed
steps and charges expired leases; it cannot promise exactly-once model billing.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Callable
from typing import Any

from audrey.app_state.agent_projects import (
    CONTEXT_CHANGED,
    AgentProjectAccessError,
    AgentProjectChangedError,
    AgentProjectContexts,
)
from audrey.app_state.notifications import _agent_notice_title, insert_agent_notice
from audrey.app_state.titles import fallback_conversation_title, normalize_agent_title
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
_STEP_LABELS = (("plan", "Plan"), ("research", "Work"), ("review", "Review"))
_EVENT_LIMIT = 200
_EVENT_FIELDS = (
    "seq", "attempt", "stage", "kind", "state", "tool", "created_at", "finished_at",
    "duration_ms", "input_tokens", "output_tokens",
)
_EVENT_TOOLS = frozenset({"web_search", "web_fetch", "create_file", "read_selected_file"})
_PUBLIC_FIELDS = (
    "objective", "title", "status", "revision", "created_at", "updated_at", "result", "error", "attempts",
)
_SAVED_FIELDS = (
    "status", "revision", "attempts", "fence", "lease_token", "lease_expires_at",
    "accounted_at", "active_seconds", "operations", "result", "error", "updated_at", "title", "title_attempted",
)


class AgentTaskError(ValueError):
    """Task input, transition or resource allowance is invalid."""


class AgentTaskConflictError(AgentTaskError):
    """The revision, lease or committed step has changed."""


class AgentTaskAccessError(AgentTaskError):
    """A task does not exist for an active owner."""


class AgentTaskContextError(AgentTaskConflictError):
    """A frozen Project input changed; cleanup and saved-report reads remain valid."""


class AgentOutputLimitError(AgentTaskError):
    """All eight durable output identities have already been used."""


def _output_links(result: str, outputs: list[dict]) -> str:
    """Retain canonical server links within the saved UTF-8 byte allowance."""
    links = []
    for item in outputs:
        url = item["download_url"]
        if re.search(r"\]\(" + re.escape(url) + r"\)", result):
            continue
        filename = item["filename"].replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        links.append(f"[Download {filename}]({url})")
    if not links:
        return result
    suffix = "\n\n" + "\n\n".join(links)
    if len((result + suffix).encode("utf-8")) > 64000:
        marker = "\n\n[Result shortened to retain file downloads.]"
        allowance = 64000 - len((marker + suffix).encode("utf-8"))
        result = result.encode("utf-8")[:max(0, allowance)].decode("utf-8", errors="ignore") + marker
    return result + suffix


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
        self._project_contexts = AgentProjectContexts(connection)
        self._file_input_service = None
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

    def _input_requests(self, task_id: str) -> list[dict]:
        return [{"id": row["request_id"], **{key: row[key] for key in (
            "step_id", "question", "answer", "created_at", "answered_at",
        )}} for row in self._conn.execute(
            "SELECT * FROM app_agent_input_requests WHERE task_id=? AND dismissed_at IS NULL "
            "ORDER BY sequence_no", (task_id,),
        ).fetchall()]

    def _record(self, row: sqlite3.Row) -> dict:
        record = dict(row)
        record["steps"] = json.loads(record.pop("steps_json"))
        # Stage identities/checkpoints survive presentation-label changes.
        labels = dict(_STEP_LABELS)
        for step in record["steps"]:
            step["label"] = labels.get(step["id"], step["label"])
        record["limits"] = json.loads(record.pop("limits_json"))
        record["input_requests"] = self._input_requests(record["task_id"])
        record["project_context"] = self._project_contexts.public(task_id=record["task_id"], user_id=record["user_id"])
        record["outputs"] = self._outputs(record["task_id"], record["user_id"])
        record["file_inputs"] = self._file_inputs_public(record["task_id"], record["user_id"])
        return record

    def _file_input_pins(self, task_id: str, user_id: str) -> list[tuple[str, dict]]:
        return [(row["project_id"], json.loads(row["pin_json"])) for row in self._conn.execute(
            "SELECT project_id,pin_json FROM app_agent_file_inputs WHERE task_id=? AND user_id=? ORDER BY position",
            (task_id, user_id),
        ).fetchall()]

    def _file_inputs_public(self, task_id: str, user_id: str) -> dict | None:
        from audrey.agent_file_inputs import public_file

        pins = self._file_input_pins(task_id, user_id)
        if not pins:
            return None
        service = self._file_input_service
        return {"current": bool(service and all(service.is_current(user_id, project, pin) for project, pin in pins)),
                "files": [public_file(pin) for _project, pin in pins]}

    def _fence_file_inputs(self, record: dict) -> None:
        from audrey.agent_file_inputs import FILES_CHANGED

        selected = self._file_inputs_public(record["task_id"], record["user_id"])
        if selected is not None and not selected["current"]:
            raise AgentTaskContextError(FILES_CHANGED)

    def _outputs(self, task_id: str, user_id: str) -> list[dict]:
        return [{"file_id": row["file_id"], "filename": row["filename"],
                 "download_url": f"/api/files/{row['file_id']}/download", "created_at": row["created_at"]}
                for row in self._conn.execute(
                    "SELECT file_id,filename,created_at FROM app_agent_outputs "
                    "WHERE task_id=? AND user_id=? AND status='ready' ORDER BY created_at,file_id",
                    (task_id, user_id),
                ).fetchall()]

    @staticmethod
    def _public(record: dict) -> dict:
        return {
            "id": record["task_id"],
            **{key: record[key] for key in _PUBLIC_FIELDS},
            "title": record["title"] or fallback_conversation_title(record["objective"]),
            "status": "waiting_for_input" if record["status"] == "paused" and any(
                item["answer"] is None for item in record["input_requests"]
            ) else record["status"],
            "steps": [dict(step) for step in record["steps"]],
            "input_requests": [dict(item) for item in record["input_requests"]],
            "project_context": record["project_context"],
            **({"outputs": [dict(item) for item in record["outputs"]]} if record["outputs"] else {}),
            **({"file_inputs": record["file_inputs"]} if record["file_inputs"] is not None else {}),
        }

    def _internal(self, record: dict) -> dict:
        return {
            **self._public(record),
            **{key: record[key] for key in (
                "user_id", "lease_token", "lease_expires_at", "fence", "limits",
                "operations", "active_seconds",
                "title_attempted",
            )},
            "title_generated": bool(record["title"]),
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
            "updated_at=?,title=?,title_attempted=?,steps_json=? WHERE task_id=?",
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

    def _interrupt_events(self, record: dict) -> None:
        self._conn.execute(
            "UPDATE app_agent_events SET state='interrupted',finished_at=? "
            "WHERE task_id=? AND state='started'", (_stamp(self._clock()), record["task_id"]),
        )

    def _clear_lease(self, record: dict) -> None:
        self._interrupt_events(record)
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

    def _context(self, record: dict) -> dict | None:
        self._fence_file_inputs(record)
        try:
            return self._project_contexts.resolve(task_id=record["task_id"], user_id=record["user_id"])
        except AgentProjectChangedError:
            raise AgentTaskContextError(CONTEXT_CHANGED) from None

    def fence_execution(self, *, task_id: str, lease_token: str, fence: int) -> None:
        """Check live owner, lease, inputs and cumulative time after the final await."""
        with self._lock:
            record = self._lease(task_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            self._context(record)

    async def resolve_project_context(self, *, task_id: str, lease_token: str, fence: int) -> dict | None:
        def work() -> dict | None:
            with self._lock:
                record = self._lease(task_id, lease_token, fence)
                self._account(record)
                self._budget(record)
                return self._context(record)
        return await asyncio.to_thread(work)

    @staticmethod
    def _event_identity(stage: str, kind: str, tool: str | None) -> None:
        if (not isinstance(stage, str) or stage not in {"plan", "research", "review", "title"}
                or not isinstance(kind, str) or kind not in {"stage", "model", "tool"}
                or (stage == "title" and kind != "model")
                or (kind == "tool" and (stage != "research" or not isinstance(tool, str) or tool not in _EVENT_TOOLS))
                or (kind != "tool" and tool is not None)):
            raise AgentTaskError("Invalid agent activity.")

    def _start_event(self, record: dict, *, stage: str, kind: str, tool: str | None = None) -> int:
        """Insert content-free activity inside an existing fenced transaction."""
        seq = self._conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM app_agent_events WHERE task_id=?", (record["task_id"],),
        ).fetchone()[0]
        self._conn.execute(
            "INSERT INTO app_agent_events(task_id,user_id,seq,attempt,fence,stage,kind,state,tool,created_at) "
            "VALUES (?,?,?,?,?,?,?,'started',?,?)",
            (record["task_id"], record["user_id"], seq, record["attempts"], record["fence"],
             stage, kind, tool, _stamp(self._clock())),
        )
        self._conn.execute(
            "DELETE FROM app_agent_events WHERE task_id=? AND seq<=?",
            (record["task_id"], seq - _EVENT_LIMIT),
        )
        return seq

    async def start_event(
        self, *, task_id: str, lease_token: str, fence: int,
        stage: str, kind: str, tool: str | None = None,
    ) -> int:
        """Observe a dispatch without changing task revisions, charges or notices."""
        self._event_identity(stage, kind, tool)

        def work() -> int:
            record = self._lease(task_id, lease_token, fence)
            self._context(record)
            self._account(record)
            self._budget(record)
            if stage != "title" and self._internal(record)["current_step"] != stage:
                raise AgentTaskConflictError("Agent step changed.")
            return self._start_event(record, stage=stage, kind=kind, tool=tool)

        return await asyncio.to_thread(self._transaction, work)

    async def finish_event(
        self, *, task_id: str, lease_token: str, fence: int, seq: int,
        state: str = "completed", duration_ms: int | None = None,
        input_tokens: int | None = None, output_tokens: int | None = None,
    ) -> None:
        _integer(seq, 1, 2**63 - 1)
        if not isinstance(state, str) or state not in {"completed", "failed"}:
            raise AgentTaskError("Invalid agent activity outcome.")
        for value in (duration_ms, input_tokens, output_tokens):
            if value is not None:
                _integer(value, 0, 2**63 - 1)

        def work() -> None:
            record = self._lease(task_id, lease_token, fence)
            self._context(record)
            self._account(record)
            self._budget(record)
            row = self._conn.execute(
                "SELECT * FROM app_agent_events WHERE task_id=? AND seq=? AND fence=?",
                (task_id, seq, fence),
            ).fetchone()
            if row is None:
                oldest = self._conn.execute(
                    "SELECT MIN(seq) FROM app_agent_events WHERE task_id=?", (task_id,),
                ).fetchone()[0]
                # Activity retention must never make otherwise authorized
                # useful work fail. A pruned identity has no row left to alter.
                if oldest is not None and seq < oldest:
                    return
                raise AgentTaskConflictError("Agent activity changed or expired.")
            if row["kind"] != "model" and (input_tokens is not None or output_tokens is not None):
                raise AgentTaskError("Invalid agent activity counts.")
            if row["state"] == "started" and row["stage"] != "title" and (
                row["stage"] != self._internal(record)["current_step"]
            ):
                raise AgentTaskConflictError("Agent step changed.")
            if row["state"] != "started":
                if all(row[key] == value for key, value in (
                    ("state", state), ("duration_ms", duration_ms), ("input_tokens", input_tokens),
                    ("output_tokens", output_tokens),
                )):
                    return
                raise AgentTaskConflictError("Agent activity already finished.")
            self._conn.execute(
                "UPDATE app_agent_events SET state=?,finished_at=?,duration_ms=?,input_tokens=?,output_tokens=? "
                "WHERE task_id=? AND seq=? AND fence=?",
                (state, _stamp(self._clock()), duration_ms, input_tokens, output_tokens, task_id, seq, fence),
            )

        return await asyncio.to_thread(self._transaction, work)

    async def events(
        self, *, user_id: str, task_id: str, before: int | None = None, limit: int = 50,
    ) -> dict:
        """Return newest-first bounded history, independently of task revisions."""
        _integer(limit, 1, 100)
        if before is not None:
            _integer(before, 1, 2**63 - 1)

        def work() -> dict:
            with self._lock:
                self._load(task_id, user_id)
                rows = self._conn.execute(
                    "SELECT * FROM app_agent_events WHERE task_id=? AND user_id=? "
                    "AND (? IS NULL OR seq<?) ORDER BY seq DESC LIMIT ?",
                    (task_id, user_id, before, before, limit + 1),
                ).fetchall()
                oldest = self._conn.execute(
                    "SELECT MIN(seq) FROM app_agent_events WHERE task_id=?", (task_id,),
                ).fetchone()[0]
                return {"items": [{key: row[key] for key in _EVENT_FIELDS} for row in rows[:limit]],
                        "has_more": len(rows) > limit, "truncated": oldest is not None and oldest > 1}

        return await asyncio.to_thread(work)

    def fence_events_response(self, *, user_id: str, task_id: str, page: dict) -> None:
        """Recheck owner/deletion and mutable or pruned rows after awaited reads."""
        with self._lock:
            self._load(task_id, user_id)
            for item in page["items"]:
                row = self._conn.execute(
                    "SELECT * FROM app_agent_events WHERE task_id=? AND user_id=? AND seq=?",
                    (task_id, user_id, item["seq"]),
                ).fetchone()
                if row is None or {key: row[key] for key in _EVENT_FIELDS} != item:
                    raise AgentTaskConflictError("Agent activity changed. Refresh before trying again.")
            oldest = self._conn.execute(
                "SELECT MIN(seq) FROM app_agent_events WHERE task_id=?", (task_id,),
            ).fetchone()[0]
            if page["truncated"] != (oldest is not None and oldest > 1):
                raise AgentTaskConflictError("Agent activity changed. Refresh before trying again.")

    async def reserve_output(
        self, *, task_id: str, lease_token: str, fence: int, file_id: str,
        filename: str, identity_sha256: str,
    ) -> dict:
        """Record content-free intent before bytes or catalog rows can be written."""
        if (not isinstance(file_id, str) or not re.fullmatch(r"[0-9a-f-]{36}", file_id)
                or not isinstance(identity_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", identity_sha256)):
            raise AgentTaskError("Invalid output identity.")
        filename = _text(filename, maximum=720, name="output filename")
        if len(filename) > 180:
            raise AgentTaskError("Invalid output filename.")

        def work():
            record = self._lease(task_id, lease_token, fence)
            self._context(record)
            self._account(record)
            self._budget(record)
            if self._internal(record)["current_step"] != "research":
                raise AgentTaskConflictError("Only the Work step can create files.")
            row = self._conn.execute("SELECT * FROM app_agent_outputs WHERE file_id=?", (file_id,)).fetchone()
            if row is not None:
                if (row["task_id"], row["user_id"], row["filename"], row["identity_sha256"]) != (
                    task_id, record["user_id"], filename, identity_sha256,
                ):
                    raise AgentTaskConflictError("The output identity changed.")
                return dict(row)
            count = self._conn.execute(
                "SELECT COUNT(*) FROM app_agent_outputs WHERE task_id=?", (task_id,),
            ).fetchone()[0]
            if count >= 8:
                raise AgentOutputLimitError("At most eight distinct files can be created by this agent.")
            now = _stamp(self._clock())
            self._conn.execute(
                "INSERT INTO app_agent_outputs(file_id,task_id,user_id,filename,identity_sha256,status,created_at) "
                "VALUES (?,?,?,?,?,'reserved',?)", (file_id, task_id, record["user_id"], filename, identity_sha256, now),
            )
            self._save(record)
            return dict(self._conn.execute("SELECT * FROM app_agent_outputs WHERE file_id=?", (file_id,)).fetchone())

        return await asyncio.to_thread(self._transaction, work)

    def _queue_capacity(self, user_id: str, limits: dict, *, exclude_task_id: str | None = None) -> None:
        queued = self._conn.execute(
            "SELECT COUNT(*) AS total,COALESCE(SUM(user_id=?),0) AS owned "
            "FROM app_agent_tasks WHERE status IN ('queued','running','paused') "
            "AND (? IS NULL OR task_id!=?)",
            (user_id, exclude_task_id, exclude_task_id),
        ).fetchone()
        if (
            queued["owned"] >= limits["max_queued_per_user"]
            or queued["total"] >= limits["max_queued_global"]
        ):
            raise AgentTaskError("Agent task capacity reached.")

    async def create(
        self, *, user_id: str, objective: str, request_id: str, limits: dict | None = None,
        project_id: str | None = None,
        knowledge_entry_id: str | None = None, knowledge_entry_revision: int | None = None,
        file_inputs=None, knowledge_file_sources=None,
    ) -> dict:
        objective = _text(objective, maximum=8000, name="objective")
        request_id = _text(request_id, maximum=200, name="request identity")
        limits = _limits(limits)
        from audrey.agent_file_inputs import FILES_CHANGED, CapturedFileInputs

        if file_inputs is not None and (
            not isinstance(file_inputs, CapturedFileInputs) or file_inputs.owner_id != user_id
            or file_inputs.project_id != project_id or len(file_inputs.pins) > 3
        ):
            raise AgentTaskError("Invalid selected Project files.")
        selected_pins = file_inputs.pins if file_inputs is not None else ()
        if knowledge_file_sources is not None:
            from audrey.agent_knowledge_files import CapturedKnowledgeFileSources

            if (not isinstance(knowledge_file_sources, CapturedKnowledgeFileSources)
                    or knowledge_file_sources.owner_id != user_id
                    or knowledge_file_sources.project_id != project_id
                    or len(knowledge_file_sources.proofs) > 96):
                raise AgentTaskError("Invalid Project knowledge sources.")
        if project_id is not None:
            project_id = _text(project_id, maximum=300, name="Project identity")
        if (knowledge_entry_id is None) != (knowledge_entry_revision is None):
            raise AgentTaskError("Project knowledge identity and revision must be supplied together.")
        if knowledge_entry_id is not None:
            knowledge_entry_id = _text(knowledge_entry_id, maximum=300, name="Project knowledge identity")
            _integer(knowledge_entry_revision, 1, 2**63 - 1)
            if project_id is None:
                raise AgentTaskError("Project knowledge tasks require a Project.")

        def work() -> dict:
            self._owner(user_id)
            existing = self._conn.execute(
                "SELECT * FROM app_agent_tasks WHERE user_id=? AND request_id=?",
                (user_id, request_id),
            ).fetchone()
            if existing is not None:
                record = self._record(existing)
                saved_project = record["project_context"]["project_id"] if record["project_context"] else None
                saved_origin = record["project_context"].get("origin_entry") if record["project_context"] else None
                requested_origin = {"id": knowledge_entry_id, "revision": knowledge_entry_revision} if knowledge_entry_id is not None else None
                saved_files = sorted(pin["file_id"] for _project, pin in self._file_input_pins(record["task_id"], user_id))
                requested_files = sorted(pin["file_id"] for pin in selected_pins)
                if (record["objective"] != objective or saved_project != project_id or saved_origin != requested_origin
                        or saved_files != requested_files):
                    raise AgentTaskConflictError("Request identity already used for another objective, Project or knowledge entry.")
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
            if project_id is not None:
                try:
                    self._project_contexts.capture(
                        task_id=task_id, user_id=user_id, project_id=project_id, objective=objective,
                        knowledge_entry_id=knowledge_entry_id, knowledge_entry_revision=knowledge_entry_revision,
                        file_sources=knowledge_file_sources,
                    )
                except AgentProjectAccessError:
                    raise AgentTaskAccessError("Project context is unavailable.") from None
                except AgentProjectChangedError as error:
                    raise AgentTaskConflictError(str(error)) from None
            for position, pin in enumerate(selected_pins):
                if self._file_input_service is None or not self._file_input_service.is_current(user_id, project_id, pin):
                    raise AgentTaskContextError(FILES_CHANGED)
                self._conn.execute(
                    "INSERT INTO app_agent_file_inputs(task_id,user_id,project_id,position,file_id,pin_json) VALUES (?,?,?,?,?,?)",
                    (task_id, user_id, project_id, position, pin["file_id"], json.dumps(pin, ensure_ascii=False, separators=(",", ":"))),
                )
            return self._public(self._load(task_id, user_id))

        return await asyncio.to_thread(self._transaction, work)

    async def list(self, *, user_id: str, limit: int = 50, summary: bool = False, project_id: str | None = None) -> list[dict]:
        _integer(limit, 1, 200)
        if type(summary) is not bool:
            raise AgentTaskError("Invalid task summary option.")
        if project_id is not None:
            project_id = _text(project_id, maximum=300, name="Project identity")

        def work() -> list[dict]:
            with self._lock:
                self._owner(user_id)
                if project_id is not None:
                    try:
                        self._project_contexts.require_project(user_id, project_id)
                    except AgentProjectAccessError:
                        raise AgentTaskAccessError("Project context is unavailable.") from None
                if summary:
                    rows = self._conn.execute(
                        "SELECT task_id,objective,title,status,revision,created_at,updated_at,"
                        "'' AS result,error,attempts,"
                        "json_extract(steps_json,'$[0].status') AS plan_status,"
                        "json_extract(steps_json,'$[1].status') AS research_status,"
                        "json_extract(steps_json,'$[2].status') AS review_status,"
                        "EXISTS(SELECT 1 FROM app_agent_input_requests q WHERE q.task_id=app_agent_tasks.task_id "
                        "AND q.answer IS NULL AND q.dismissed_at IS NULL) AS has_pending_input "
                        "FROM app_agent_tasks WHERE user_id=? AND (? IS NULL OR EXISTS "
                        "(SELECT 1 FROM app_agent_project_contexts c WHERE c.task_id=app_agent_tasks.task_id AND c.project_id=?)) "
                        "ORDER BY created_at DESC,task_id DESC LIMIT ?", (user_id, project_id, project_id, limit),
                    ).fetchall()
                    return [self._summary_payload(row, user_id) for row in rows]
                return [self._public(self._record(row)) for row in self._conn.execute(
                    "SELECT * FROM app_agent_tasks WHERE user_id=? AND (? IS NULL OR EXISTS "
                    "(SELECT 1 FROM app_agent_project_contexts c WHERE c.task_id=app_agent_tasks.task_id AND c.project_id=?)) "
                    "ORDER BY created_at DESC,task_id DESC LIMIT ?", (user_id, project_id, project_id, limit),
                ).fetchall()]

        return await asyncio.to_thread(work)

    def _summary_payload(self, row: sqlite3.Row, user_id: str) -> dict:
        return {
            "id": row["task_id"],
            **{key: row[key] for key in _PUBLIC_FIELDS},
            "title": row["title"] or fallback_conversation_title(row["objective"]),
            "status": "waiting_for_input" if row["status"] == "paused" and row["has_pending_input"] else row["status"],
            "summary": True,
            "input_requests": [],
            "project_context": self._project_contexts.public(task_id=row["task_id"], user_id=user_id),
            **({"file_inputs": selected} if (selected := self._file_inputs_public(row["task_id"], user_id)) is not None else {}),
            "steps": [{
                "id": key, "label": label, "status": row[f"{key}_status"], "result": "",
            } for key, label in _STEP_LABELS],
        }

    async def get(self, *, user_id: str, task_id: str) -> dict:
        def work() -> dict:
            with self._lock:
                return self._public(self._load(task_id, user_id))

        return await asyncio.to_thread(work)

    def response_snapshot(self, *, user_id: str, tasks: list[dict]) -> list[dict]:
        """Refresh normal progress after the last await, without replaying an action.

        The worker may claim or checkpoint between an asynchronous read/commit and
        its HTTP response. Read current bodies at one serialized boundary instead
        of treating that successful progress as an optimistic-write conflict.
        Erased owners/tasks and changed input metadata retain the strict fence.
        """
        if len(tasks) > 200:
            raise AgentTaskError("Agent response exceeds its allowance.")

        def work() -> list[dict]:
            self._owner(user_id)
            values = []
            for payload in tasks:
                try:
                    self.fence_response(user_id=user_id, tasks=[payload])
                except AgentTaskConflictError:
                    if payload.get("summary"):
                        row = self._conn.execute(
                            "SELECT task_id,objective,title,status,revision,created_at,updated_at,"
                            "'' AS result,error,attempts,"
                            "json_extract(steps_json,'$[0].status') AS plan_status,"
                            "json_extract(steps_json,'$[1].status') AS research_status,"
                            "json_extract(steps_json,'$[2].status') AS review_status,"
                            "EXISTS(SELECT 1 FROM app_agent_input_requests q WHERE q.task_id=t.task_id "
                            "AND q.answer IS NULL AND q.dismissed_at IS NULL) AS has_pending_input "
                            "FROM app_agent_tasks t WHERE user_id=? AND task_id=?",
                            (user_id, payload["id"]),
                        ).fetchone()
                        if row is None:
                            raise AgentTaskAccessError("Agent task is unavailable.") from None
                        current = self._summary_payload(row, user_id)
                    else:
                        current = self._public(self._load(payload["id"], user_id))
                    if any(current.get(key) != payload.get(key) for key in ("project_context", "file_inputs")):
                        raise AgentTaskConflictError("Agent task changed. Refresh before trying again.") from None
                    values.append(current)
                else:
                    values.append(payload)
            return values

        return self._transaction(work)

    def fence_response(self, *, user_id: str, tasks: list[dict] | tuple = ()) -> None:
        """Reject erased/revised task bodies synchronously after a route's last await."""
        if len(tasks) > 200:
            raise AgentTaskError("Agent response exceeds its allowance.")
        with self._lock:
            self._owner(user_id)
            for payload in tasks:
                row = self._conn.execute(
                    "SELECT revision,status,EXISTS(SELECT 1 FROM app_agent_input_requests q "
                    "WHERE q.task_id=t.task_id AND q.answer IS NULL AND q.dismissed_at IS NULL) AS waiting "
                    "FROM app_agent_tasks t WHERE task_id=? AND user_id=?", (payload["id"], user_id),
                ).fetchone()
                if row is None:
                    raise AgentTaskAccessError("Agent task is unavailable.")
                status = "waiting_for_input" if row["status"] == "paused" and row["waiting"] else row["status"]
                if row["revision"] != payload["revision"] or status != payload["status"] or (
                    payload.get("input_requests") and self._input_requests(payload["id"]) != payload["input_requests"]
                ) or (
                    self._project_contexts.public(task_id=payload["id"], user_id=user_id) != payload.get("project_context")
                ) or (
                    self._file_inputs_public(payload["id"], user_id) != payload.get("file_inputs")
                ):
                    raise AgentTaskConflictError("Agent task changed. Refresh before trying again.")

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
                "AND NOT EXISTS (SELECT 1 FROM app_watch_runs w WHERE w.user_id=t.user_id "
                "AND w.status='running') "
                "ORDER BY t.created_at,t.task_id LIMIT 1000", (_stamp(now),),
            ).fetchall()
            for row in rows:
                record = self._record(row)
                self._account(record)
                try:
                    self._context(record)
                except AgentTaskContextError:
                    record.update(status="failed", error=CONTEXT_CHANGED)
                    self._clear_lease(record)
                    self._save(record)
                    insert_agent_notice(self._conn, record, created_at=record["updated_at"])
                    continue
                if (
                    record["attempts"] >= max_attempts
                    or record["active_seconds"] >= record["limits"]["max_active_seconds"]
                ):
                    record.update(status="failed", error="Agent execution allowance exhausted.")
                    self._clear_lease(record)
                    self._save(record)
                    insert_agent_notice(self._conn, record, created_at=record["updated_at"])
                    continue
                # An expired lease has an unknown dispatch outcome. Preserve
                # its history without treating an unobserved response as success.
                self._interrupt_events(record)
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
                self._context(record)
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
            self._context(record)
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
            self._context(record)
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
            if self._conn.execute(
                "SELECT COUNT(*) FROM app_agent_operations WHERE task_id=? AND fence=? AND step_id=?",
                (task_id, fence, step_id),
            ).fetchone()[0] == 1:
                self._start_event(record, stage=step_id, kind="stage")
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
            self._context(record)
            self._account(record)
            self._budget(record)
            saved_result = _output_links(result, record["outputs"]) if step_id == "research" else result
            step = next((item for item in record["steps"] if item["id"] == step_id), None)
            if step is not None and step["status"] == "completed" and step["result"] == saved_result:
                return self._internal(record)
            if step is None or self._internal(record)["current_step"] != step_id:
                raise AgentTaskConflictError("Agent step changed.")
            if self._conn.execute(
                "SELECT 1 FROM app_agent_operations WHERE task_id=? AND fence=? AND step_id=? "
                "AND status='reserved'", (task_id, fence, step_id),
            ).fetchone() is None:
                raise AgentTaskConflictError("Reserve the agent operation before committing its result.")
            step.update(status="completed", result=saved_result)
            next_step = next((item for item in record["steps"] if item["status"] != "completed"), None)
            if next_step is not None:
                next_step["status"] = "running"
            self._conn.execute(
                "UPDATE app_agent_operations SET status='committed',committed_at=? "
                "WHERE task_id=? AND fence=? AND step_id=?",
                (_stamp(self._clock()), task_id, fence, step_id),
            )
            self._conn.execute(
                "UPDATE app_agent_events SET state='completed',finished_at=? "
                "WHERE task_id=? AND fence=? AND stage=? AND kind='stage' AND state='started'",
                (_stamp(self._clock()), task_id, fence, step_id),
            )
            self._save(record)
            return self._internal(record)

        return await asyncio.to_thread(self._transaction, work)

    async def reserve_title_operation(
        self, *, task_id: str, lease_token: str, fence: int, step_id: str,
    ) -> str | None:
        """Charge at most one title attempt, preserving room for useful model work."""
        def work() -> str | None:
            record = self._lease(task_id, lease_token, fence)
            self._context(record)
            self._account(record)
            self._budget(record)
            if self._internal(record)["current_step"] != step_id:
                raise AgentTaskConflictError("Agent step changed.")
            if record["title"] or record["title_attempted"]:
                return None
            record["title_attempted"] = 1
            if record["operations"] + 1 >= record["limits"]["max_operations"]:
                self._save(record)
                return None
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

    async def save_title(
        self, *, task_id: str, lease_token: str, fence: int, step_id: str,
        operation_id: str, title: str,
    ) -> str:
        normalized = normalize_agent_title(title)
        if not normalized:
            raise AgentTaskError("Invalid generated agent title.")

        def work() -> str:
            record = self._lease(task_id, lease_token, fence)
            self._context(record)
            self._account(record)
            self._budget(record)
            operation = self._conn.execute(
                "SELECT 1 FROM app_agent_operations WHERE operation_id=? AND task_id=? "
                "AND fence=? AND step_id=? AND status='reserved'",
                (operation_id, task_id, fence, step_id),
            ).fetchone()
            if not record["title_attempted"] or operation is None or self._internal(record)["current_step"] != step_id:
                raise AgentTaskConflictError("The title generation operation changed.")
            if not record["title"]:
                record["title"] = normalized
                self._save(record)
            self._conn.execute(
                "UPDATE app_agent_operations SET status='committed',committed_at=? WHERE operation_id=?",
                (_stamp(self._clock()), operation_id),
            )
            return record["title"]
        return await asyncio.to_thread(self._transaction, work)

    async def request_input(
        self, *, task_id: str, lease_token: str, fence: int, question: str, step_id: str = "plan",
    ) -> dict:
        question = _text(question, maximum=2000, name="agent question")
        if step_id != "plan":
            raise AgentTaskError("Agent clarification is supported during Plan only.")

        def work() -> dict:
            record = self._lease(task_id, lease_token, fence)
            self._context(record)
            self._account(record)
            self._budget(record)
            if self._internal(record)["current_step"] != "plan":
                raise AgentTaskConflictError("The agent has already completed its Plan.")
            count = self._conn.execute(
                "SELECT COUNT(*) FROM app_agent_input_requests WHERE task_id=?", (task_id,),
            ).fetchone()[0]
            if count >= 3:
                raise AgentTaskError("Agent clarification allowance exhausted.")
            now, request_id = _stamp(self._clock()), str(uuid.uuid4())
            # The model's question is a durable result of its reserved work. It
            # consumes its existing reservation and never resets cumulative use.
            self._conn.execute(
                "UPDATE app_agent_operations SET status='committed',committed_at=? "
                "WHERE task_id=? AND fence=? AND step_id='plan' AND status='reserved'",
                (now, task_id, fence),
            )
            record.update(status="paused", error="", fence=record["fence"] + 1)
            self._clear_lease(record)
            self._save(record)
            self._conn.execute(
                "INSERT INTO app_agent_input_requests (request_id,user_id,task_id,task_revision,"
                "sequence_no,step_id,question,created_at,notice_title) VALUES (?,?,?,?,?,'plan',?,?,?)",
                (request_id, record["user_id"], task_id, record["revision"], count + 1, question, now,
                 _agent_notice_title(record)),
            )
            return self._public(self._load(task_id, record["user_id"]))

        return await asyncio.to_thread(self._transaction, work)

    async def answer_input(
        self, *, user_id: str, task_id: str, request_id: str, expected_revision: int,
        answer: str, max_attempts: int = 6, admission_limits: dict | None = None,
    ) -> dict:
        answer = _text(answer, maximum=8000, name="agent answer")
        request_id = _text(request_id, maximum=200, name="agent question identity")
        _integer(expected_revision, 1, 2**63 - 1)
        _integer(max_attempts, 1, 100)
        queue_limits = _limits(admission_limits) if admission_limits is not None else None

        def work() -> dict:
            record = self._load(task_id, user_id)
            self._context(record)
            request = self._conn.execute(
                "SELECT * FROM app_agent_input_requests WHERE request_id=? AND task_id=? AND user_id=?",
                (request_id, task_id, user_id),
            ).fetchone()
            if request is None or request["dismissed_at"] is not None:
                raise AgentTaskConflictError("This agent question is no longer pending. Refresh before trying again.")
            if request["answer"] is not None:
                if (request["answer"] == answer and record["status"] == "queued"
                        and record["revision"] == request["answer_revision"]
                        and expected_revision in {request["task_revision"], request["answer_revision"]}):
                    return self._public(record)
                raise AgentTaskConflictError("This agent question was already answered. Refresh before trying again.")
            if record["revision"] != expected_revision or record["status"] != "paused":
                raise AgentTaskConflictError("Agent task changed. Refresh before trying again.")
            self._queue_capacity(user_id, queue_limits or record["limits"], exclude_task_id=task_id)
            self._budget(record)
            if record["attempts"] >= max_attempts:
                raise AgentTaskError("Agent attempt allowance exhausted.")
            if record["operations"] >= record["limits"]["max_operations"]:
                raise AgentTaskError("Agent operation allowance exhausted.")
            now = _stamp(self._clock())
            self._conn.execute(
                "UPDATE app_agent_input_requests SET answer=?,answered_at=?,answer_revision=? "
                "WHERE request_id=?", (answer, now, record["revision"] + 1, request_id),
            )
            record.update(status="queued", error="", fence=record["fence"] + 1)
            self._clear_lease(record)
            self._save(record)
            return self._public(self._load(task_id, user_id))

        return await asyncio.to_thread(self._transaction, work)

    async def complete(self, *, task_id: str, lease_token: str, fence: int, result: str) -> dict:
        result = _text(result, maximum=64000, name="task result")

        def work() -> dict:
            record = self._lease(task_id, lease_token, fence)
            self._context(record)
            self._account(record)
            self._budget(record)
            if any(step["status"] != "completed" for step in record["steps"]):
                raise AgentTaskConflictError("Complete all agent steps before saving the result.")
            record.update(status="completed", result=_output_links(result, record["outputs"]), error="")
            self._clear_lease(record)
            self._save(record)
            insert_agent_notice(self._conn, record, created_at=record["updated_at"])
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
            if not retry:
                insert_agent_notice(self._conn, record, created_at=record["updated_at"])
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
            if action == "resume" and any(item["answer"] is None for item in record["input_requests"]):
                raise AgentTaskConflictError("Answer the agent's pending question before continuing.")
            if action in {"resume", "rerun"}:
                self._context(record)
            self._account(record)
            if action == "rerun":
                self._queue_capacity(user_id, queue_limits or record["limits"])
                self._budget(record)
                if record["attempts"] >= max_attempts:
                    raise AgentTaskError("Agent attempt allowance exhausted.")
                if record["operations"] >= record["limits"]["max_operations"]:
                    raise AgentTaskError("Agent operation allowance exhausted.")
                # Continue unfinished work; a fully finished report gets fresh
                # Work/Review while its committed Plan remains reusable.
                if all(step["status"] == "completed" for step in record["steps"]):
                    for step in record["steps"]:
                        if step["id"] != "plan":
                            step.update(status="pending", result="")
                record.update(result="", error="")
                if not record["title"]:
                    record["title_attempted"] = 0
                # Run again replaces the current report. Its former notice
                # cannot remain a link that silently opens a different result.
                self._conn.execute("DELETE FROM app_agent_notices WHERE task_id=?", (task_id,))
                self._conn.execute(
                    "UPDATE app_agent_input_requests SET dismissed_at=? "
                    "WHERE task_id=? AND answer IS NULL AND dismissed_at IS NULL",
                    (_stamp(self._clock()), task_id),
                )
                record["input_requests"] = self._input_requests(task_id)
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
