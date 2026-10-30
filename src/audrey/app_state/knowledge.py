"""Canonical Project knowledge with explicit approval and immutable revisions.

Sources are bounded references, never copied evidence. The route/service verifies
their accessibility; this repository fences Project ownership again at commit.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from audrey.app_state.transactions import write_transaction

MAX_PROJECT_ENTRIES = 200
MAX_ENTRY_CHARS = 8_000
MAX_ENTRY_REVISIONS = 100
MAX_SOURCES = 8
_KINDS = frozenset({"fact", "decision", "question", "action"})
_APPROVAL_STATES = frozenset({"approved", "proposed", "rejected", "superseded"})


class KnowledgeError(ValueError):
    """An entry or transition is outside the Project knowledge contract."""


class KnowledgeAccessError(KnowledgeError):
    """An active owner and an owned Project/entry are required."""


class KnowledgeConflictError(KnowledgeError):
    """An entry or replacement target changed before the write."""


class KnowledgeLimitError(KnowledgeError):
    """The bounded Project entry or revision allowance is exhausted."""


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 300:
        raise KnowledgeError(f"Invalid {name}.")
    return value.strip()


def _integer(value: int, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise KnowledgeError(f"Invalid {name}.")
    return value


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ENTRY_CHARS:
        raise KnowledgeError(f"Knowledge text must contain 1–{MAX_ENTRY_CHARS} characters.")
    return value.strip()


def _kind(value: str) -> str:
    if value not in _KINDS:
        raise KnowledgeError("Invalid knowledge kind.")
    return value


def _status(kind: str, value: str) -> str:
    allowed = {"open"}
    if kind == "question":
        allowed.add("resolved")
    if kind == "action":
        allowed.add("completed")
    if value not in allowed:
        raise KnowledgeError("Work status does not apply to this knowledge kind.")
    return value


def _sources(value: Sequence[dict]) -> list[dict]:
    if not isinstance(value, (list, tuple)) or len(value) > MAX_SOURCES:
        raise KnowledgeError(f"Knowledge supports at most {MAX_SOURCES} source references.")
    result: list[dict] = []
    seen: set[str] = set()
    for source in value:
        if not isinstance(source, dict) or set(source) != {"source_id", "revision"}:
            raise KnowledgeError("Sources contain only source_id and revision.")
        source_id = _identifier(source["source_id"], "source id")
        revision = _identifier(source["revision"], "source revision")
        if source_id.split(":", 1)[0] not in {"message", "file", "agent"} or ":" not in source_id:
            raise KnowledgeError("Unsupported knowledge source type.")
        if source_id in seen:
            raise KnowledgeError("Knowledge source references must be unique.")
        seen.add(source_id)
        result.append({"source_id": source_id, "revision": revision})
    return result


class KnowledgeRepository:
    """Serialize owner-scoped knowledge changes with optimistic revisions."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn, self._lock = connection, lock

    def _transaction(self, work: Callable[[], Any]) -> Any:
        with self._lock, write_transaction(self._conn):
            return work()

    def _project(self, user_id: str, project_id: str) -> None:
        _identifier(user_id, "user id")
        _identifier(project_id, "Project id")
        row = self._conn.execute(
            "SELECT 1 FROM app_projects p JOIN app_users u ON u.user_id=p.user_id "
            "WHERE p.user_id=? AND p.project_id=? AND u.status='active' "
            "AND NOT EXISTS (SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)",
            (user_id, project_id),
        ).fetchone()
        if row is None:
            raise KnowledgeAccessError("Project knowledge is unavailable.")

    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        value = dict(row)
        value["id"] = value.pop("entry_id")
        value["sources"] = json.loads(value.pop("sources_json"))
        value["pinned"] = bool(value["pinned"])
        for key in ("user_id", "request_id", "request_digest"):
            value.pop(key, None)
        return value

    def _load(self, user_id: str, project_id: str, entry_id: str) -> dict:
        self._project(user_id, project_id)
        _identifier(entry_id, "entry id")
        row = self._conn.execute(
            "SELECT * FROM app_project_knowledge WHERE user_id=? AND project_id=? AND entry_id=?",
            (user_id, project_id, entry_id),
        ).fetchone()
        if row is None:
            raise KnowledgeAccessError("Project knowledge is unavailable.")
        return self._record(row)

    @staticmethod
    def _expected(record: dict, revision: int) -> None:
        _integer(revision, 1, MAX_ENTRY_REVISIONS, "expected revision")
        if record["revision"] != revision:
            raise KnowledgeConflictError("Knowledge changed. Refresh before trying again.")

    def _save(self, user_id: str, record: dict) -> dict:
        if record["revision"] >= MAX_ENTRY_REVISIONS:
            raise KnowledgeLimitError("This entry has reached its revision allowance.")
        record["revision"] += 1
        record["updated_at"] = dt.datetime.now(dt.UTC).isoformat()
        self._conn.execute(
            "UPDATE app_project_knowledge SET kind=?,text=?,approval_state=?,work_status=?,"
            "revision=?,pinned=?,sources_json=?,updated_at=?,approved_at=? WHERE entry_id=?",
            (record["kind"], record["text"], record["approval_state"], record["work_status"],
             record["revision"], int(record["pinned"]), json.dumps(record["sources"], ensure_ascii=False),
             record["updated_at"], record["approved_at"], record["id"]),
        )
        self._snapshot(user_id, record)
        return record

    def _snapshot(self, user_id: str, record: dict) -> None:
        self._conn.execute(
            "INSERT INTO app_project_knowledge_revisions "
            "(entry_id,revision,user_id,project_id,record_json,created_at) VALUES (?,?,?,?,?,?)",
            (record["id"], record["revision"], user_id, record["project_id"],
             json.dumps(record, ensure_ascii=False), record["updated_at"]),
        )

    async def create(
        self, *, user_id: str, project_id: str, kind: str, text: str,
        pinned: bool = False, sources: Sequence[dict] = (), origin: str = "user",
        source_task_id: str | None = None, replaces_id: str | None = None,
        base_revision: int | None = None, request_id: str | None = None,
    ) -> dict:
        kind, text, sources = _kind(kind), _text(text), _sources(sources)
        if type(pinned) is not bool or origin not in {"user", "agent_report"}:
            raise KnowledgeError("Invalid knowledge origin or pin.")
        if (replaces_id is None) != (base_revision is None):
            raise KnowledgeError("A replacement requires its target and base revision.")
        if origin == "user" and (source_task_id is not None or replaces_id is not None):
            raise KnowledgeError("User entries are edited directly; only agent proposals replace entries.")
        if origin == "agent_report" and source_task_id is None:
            raise KnowledgeError("An agent proposal requires its saved task.")
        if request_id is not None:
            request_id = _identifier(request_id, "request id")
        if source_task_id is not None:
            source_task_id = _identifier(source_task_id, "task id")
        if replaces_id is not None:
            replaces_id = _identifier(replaces_id, "replacement entry id")
            _integer(base_revision, 1, MAX_ENTRY_REVISIONS, "base revision")
        digest = hashlib.sha256(json.dumps(
            [kind, text, pinned, sources, origin, source_task_id, replaces_id, base_revision],
            ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()

        def work() -> dict:
            self._project(user_id, project_id)
            if request_id is not None:
                previous = self._conn.execute(
                    "SELECT * FROM app_project_knowledge WHERE user_id=? AND project_id=? AND request_id=?",
                    (user_id, project_id, request_id),
                ).fetchone()
                if previous is not None:
                    if previous["request_digest"] != digest:
                        raise KnowledgeConflictError("This creation request already has different content.")
                    return self._record(previous)
            if source_task_id is not None:
                task = self._conn.execute(
                    "SELECT 1 FROM app_agent_tasks WHERE user_id=? AND task_id=? "
                    "AND status='completed' AND result!=''", (user_id, source_task_id),
                ).fetchone()
                if task is None:
                    raise KnowledgeAccessError("The saved agent report is unavailable.")
            if replaces_id is not None:
                target = self._load(user_id, project_id, replaces_id)
                self._expected(target, base_revision)
                if target["approval_state"] != "approved":
                    raise KnowledgeConflictError("Only current approved entries can be replaced.")
            count = self._conn.execute(
                "SELECT COUNT(*) FROM app_project_knowledge WHERE project_id=?", (project_id,),
            ).fetchone()[0]
            if count >= MAX_PROJECT_ENTRIES:
                raise KnowledgeLimitError("This Project has reached its knowledge entry allowance.")
            stamp = dt.datetime.now(dt.UTC).isoformat()
            state = "approved" if origin == "user" else "proposed"
            record = {
                "id": f"kn_{uuid.uuid4().hex}", "project_id": project_id, "kind": kind,
                "text": text, "approval_state": state, "work_status": "open", "revision": 1,
                "pinned": pinned, "origin": origin, "sources": sources,
                "created_at": stamp, "updated_at": stamp,
                "approved_at": stamp if state == "approved" else None,
                "source_task_id": source_task_id, "replaces_id": replaces_id,
                "base_revision": base_revision,
            }
            self._conn.execute(
                "INSERT INTO app_project_knowledge "
                "(entry_id,project_id,user_id,kind,text,approval_state,work_status,revision,pinned,"
                "origin,sources_json,created_at,updated_at,approved_at,source_task_id,replaces_id,"
                "base_revision,request_id,request_digest) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record["id"], project_id, user_id, kind, text, state, "open", 1, int(pinned),
                 origin, json.dumps(sources, ensure_ascii=False), stamp, stamp, record["approved_at"], source_task_id,
                 replaces_id, base_revision, request_id, digest),
            )
            self._snapshot(user_id, record)
            return record

        return await asyncio.to_thread(self._transaction, work)

    async def get(self, *, user_id: str, project_id: str, entry_id: str) -> dict:
        return await asyncio.to_thread(
            self.get_sync, user_id=user_id, project_id=project_id, entry_id=entry_id,
        )

    def get_sync(self, *, user_id: str, project_id: str, entry_id: str) -> dict:
        with self._lock:
            return self._load(user_id, project_id, entry_id)

    async def list(
        self, *, user_id: str, project_id: str, limit: int = 100,
        before_updated_at: str | None = None, before_id: str | None = None,
        approval_state: str | None = None,
    ) -> list[dict]:
        _integer(limit, 1, 100, "page limit")
        if (before_updated_at is None) != (before_id is None):
            raise KnowledgeError("Incomplete knowledge cursor.")
        if approval_state is not None and approval_state not in _APPROVAL_STATES:
            raise KnowledgeError("Invalid approval state.")

        def work() -> list[dict]:
            with self._lock:
                self._project(user_id, project_id)
                rows = self._conn.execute(
                    "SELECT * FROM app_project_knowledge WHERE user_id=? AND project_id=? "
                    "AND (? IS NULL OR approval_state=?) AND (? IS NULL OR updated_at<? "
                    "OR (updated_at=? AND entry_id<?)) ORDER BY updated_at DESC,entry_id DESC LIMIT ?",
                    (user_id, project_id, approval_state, approval_state, before_updated_at,
                     before_updated_at, before_updated_at, before_id, limit),
                ).fetchall()
                return [self._record(row) for row in rows]
        return await asyncio.to_thread(work)

    def list_approved_sync(
        self, *, user_id: str, project_id: str, limit: int = 100,
    ) -> list[dict]:
        """Read one bounded approved snapshot; caller still validates source freshness."""
        _integer(limit, 1, 100, "snapshot limit")
        with self._lock:
            self._project(user_id, project_id)
            rows = self._conn.execute(
                "SELECT * FROM app_project_knowledge WHERE user_id=? AND project_id=? "
                "AND approval_state='approved' ORDER BY pinned DESC, "
                "CASE WHEN kind='decision' THEN 0 WHEN work_status='open' AND kind IN "
                "('question','action') THEN 1 ELSE 2 END, updated_at DESC,entry_id DESC LIMIT ?",
                (user_id, project_id, limit),
            ).fetchall()
            return [self._record(row) for row in rows]

    async def list_approved(
        self, *, user_id: str, project_id: str, limit: int = 100,
    ) -> list[dict]:
        return await asyncio.to_thread(
            self.list_approved_sync, user_id=user_id, project_id=project_id, limit=limit,
        )

    async def update(
        self, *, user_id: str, project_id: str, entry_id: str, expected_revision: int,
        kind: str | None = None, text: str | None = None, pinned: bool | None = None,
        work_status: str | None = None, sources: Sequence[dict] | None = None,
    ) -> dict:
        if all(value is None for value in (kind, text, pinned, work_status, sources)):
            raise KnowledgeError("Knowledge update has no fields.")
        if kind is not None:
            kind = _kind(kind)
        if text is not None:
            text = _text(text)
        if pinned is not None and type(pinned) is not bool:
            raise KnowledgeError("Knowledge pin must be a boolean.")
        if sources is not None:
            sources = _sources(sources)

        def work() -> dict:
            record = self._load(user_id, project_id, entry_id)
            self._expected(record, expected_revision)
            if record["approval_state"] in {"rejected", "superseded"}:
                raise KnowledgeConflictError("Rejected or superseded entries cannot be edited.")
            if kind is not None and kind != record["kind"]:
                record.update(kind=kind, work_status="open")
            for key, value in (("text", text), ("pinned", pinned), ("sources", sources)):
                if value is not None:
                    record[key] = value
            if work_status is not None:
                record["work_status"] = _status(record["kind"], work_status)
            return self._save(user_id, record)
        return await asyncio.to_thread(self._transaction, work)

    async def decide(
        self, *, user_id: str, project_id: str, entry_id: str,
        expected_revision: int, action: str,
    ) -> dict:
        if action not in {"approve", "reject"}:
            raise KnowledgeError("Unknown knowledge approval action.")

        def work() -> dict:
            record = self._load(user_id, project_id, entry_id)
            self._expected(record, expected_revision)
            if record["approval_state"] != "proposed":
                raise KnowledgeConflictError("Only a pending proposal can be approved or rejected.")
            if action == "approve" and record["replaces_id"] is not None:
                target = self._load(user_id, project_id, record["replaces_id"])
                self._expected(target, record["base_revision"])
                if target["approval_state"] != "approved":
                    raise KnowledgeConflictError("The replacement target is no longer approved.")
                target["approval_state"] = "superseded"
                self._save(user_id, target)
            record["approval_state"] = "approved" if action == "approve" else "rejected"
            if action == "approve":
                record["approved_at"] = dt.datetime.now(dt.UTC).isoformat()
            return self._save(user_id, record)
        return await asyncio.to_thread(self._transaction, work)

    async def delete(
        self, *, user_id: str, project_id: str, entry_id: str, expected_revision: int,
    ) -> None:
        def work() -> None:
            record = self._load(user_id, project_id, entry_id)
            self._expected(record, expected_revision)
            self._conn.execute("DELETE FROM app_project_knowledge WHERE entry_id=?", (entry_id,))
        await asyncio.to_thread(self._transaction, work)

    async def history(
        self, *, user_id: str, project_id: str, entry_id: str,
        limit: int = 100, before_revision: int | None = None,
    ) -> list[dict]:
        _integer(limit, 1, 100, "history page limit")
        if before_revision is not None:
            _integer(before_revision, 1, MAX_ENTRY_REVISIONS + 1, "history cursor")

        def work() -> list[dict]:
            with self._lock:
                self._load(user_id, project_id, entry_id)
                rows = self._conn.execute(
                    "SELECT record_json FROM app_project_knowledge_revisions WHERE entry_id=? "
                    "AND (? IS NULL OR revision<?) ORDER BY revision DESC LIMIT ?",
                    (entry_id, before_revision, before_revision, limit),
                ).fetchall()
                return [json.loads(row["record_json"]) for row in rows]
        return await asyncio.to_thread(work)

    def _run(self, user_id: str, project_id: str, run_id: str) -> sqlite3.Row:
        self._project(user_id, project_id)
        _identifier(run_id, "run id")
        row = self._conn.execute(
            "SELECT r.run_id,r.status,c.project_id FROM app_runs r "
            "JOIN app_conversations c ON c.conversation_id=r.conversation_id AND c.user_id=r.user_id "
            "WHERE r.run_id=? AND r.user_id=?", (run_id, user_id),
        ).fetchone()
        if row is None:
            raise KnowledgeAccessError("Run knowledge is unavailable.")
        return row

    async def record_run_snapshot(
        self, *, user_id: str, run_id: str, project_id: str, entries: Sequence[dict],
    ) -> None:
        if not isinstance(entries, (list, tuple)) or len(entries) > 100:
            raise KnowledgeError("Invalid run knowledge snapshot.")
        normalized: list[dict] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "revision"}:
                raise KnowledgeError("Run knowledge stores entry ids and revisions only.")
            normalized.append({
                "id": _identifier(entry["id"], "entry id"),
                "revision": _integer(entry["revision"], 1, MAX_ENTRY_REVISIONS, "entry revision"),
            })
        if len({entry["id"] for entry in normalized}) != len(normalized):
            raise KnowledgeError("Run knowledge entries must be unique.")
        normalized.sort(key=lambda entry: entry["id"])

        def work() -> None:
            run = self._run(user_id, project_id, run_id)
            if run["project_id"] != project_id:
                raise KnowledgeConflictError("The conversation's Project changed.")
            for entry in normalized:
                record = self._load(user_id, project_id, entry["id"])
                self._expected(record, entry["revision"])
                if record["approval_state"] != "approved":
                    raise KnowledgeConflictError("Run knowledge must still be approved.")
            previous = self._conn.execute(
                "SELECT project_id FROM app_run_project_knowledge_snapshots WHERE run_id=? AND user_id=?",
                (run_id, user_id),
            ).fetchone()
            if previous is not None:
                rows = self._conn.execute(
                    "SELECT entry_id,revision FROM app_run_project_knowledge "
                    "WHERE run_id=? AND user_id=? ORDER BY entry_id", (run_id, user_id),
                ).fetchall()
                saved = [{"id": row["entry_id"], "revision": row["revision"]} for row in rows]
                if previous["project_id"] != project_id or saved != normalized:
                    raise KnowledgeConflictError("This run already has a different knowledge snapshot.")
                return
            if run["status"] != "running":
                raise KnowledgeConflictError("A finished run cannot acquire new Project knowledge.")
            self._conn.execute(
                "INSERT INTO app_run_project_knowledge_snapshots(run_id,user_id,project_id) VALUES (?,?,?)",
                (run_id, user_id, project_id),
            )
            self._conn.executemany(
                "INSERT INTO app_run_project_knowledge(run_id,user_id,project_id,entry_id,revision) "
                "VALUES (?,?,?,?,?)",
                [(run_id, user_id, project_id, entry["id"], entry["revision"]) for entry in normalized],
            )
        await asyncio.to_thread(self._transaction, work)

    async def get_run_snapshot(
        self, *, user_id: str, run_id: str, project_id: str,
    ) -> list[dict]:
        def work() -> list[dict]:
            with self._lock:
                self._run(user_id, project_id, run_id)
                rows = self._conn.execute(
                    "SELECT entry_id,revision FROM app_run_project_knowledge "
                    "WHERE run_id=? AND user_id=? AND project_id=? ORDER BY entry_id",
                    (run_id, user_id, project_id),
                ).fetchall()
                return [{"id": row["entry_id"], "revision": row["revision"]} for row in rows]
        return await asyncio.to_thread(work)


__all__ = [
    "KnowledgeRepository", "KnowledgeError", "KnowledgeAccessError", "KnowledgeConflictError",
    "KnowledgeLimitError", "MAX_PROJECT_ENTRIES", "MAX_ENTRY_CHARS", "MAX_ENTRY_REVISIONS",
]
