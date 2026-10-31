"""Bounded in-app activity linked to current canonical tasks and watch checks.

Agent notices describe the saved result of the current task revision. They are
removed on Run again because that action replaces the task's saved report; this
repository does not create a second report history or copy provider text.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
import threading
import uuid
from collections.abc import Callable
from typing import Any

from audrey.app_state.titles import fallback_conversation_title
from audrey.app_state.transactions import write_transaction

_AGENT_BODIES = {
    "completed": "The agent completed its task. Open it to view the saved result.",
    "failure": "The agent could not finish its task. Open it to review the saved progress.",
}
# This one canonical relation supplies feed bodies and the full unread count in
# one SQL snapshot, including writes made by another ApplicationStore connection.
_VALID_NOTICES = """
WITH valid_notices AS (
  SELECT n.notice_id,'agent' AS origin,n.kind,n.title,n.body,n.created_at,n.read_at,
         n.task_id AS target_id,NULL AS run_id,n.task_revision AS task_revision
  FROM app_agent_notices n JOIN app_agent_tasks t
    ON t.task_id=n.task_id AND t.user_id=n.user_id AND t.revision=n.task_revision
  WHERE n.user_id=? AND ((n.kind='completed' AND t.status='completed')
                     OR (n.kind='failure' AND t.status='failed'))
  UNION ALL
  SELECT n.notice_id,'watch' AS origin,n.kind,n.title,n.body,n.created_at,n.read_at,
         n.watch_id AS target_id,n.run_id,NULL AS task_revision
  FROM app_watch_notices n
  JOIN app_watches w ON w.watch_id=n.watch_id AND w.user_id=n.user_id
  JOIN app_watch_runs r ON r.run_id=n.run_id AND r.watch_id=n.watch_id
                       AND r.user_id=n.user_id
  WHERE n.user_id=?
)
"""
_GET_SQL = "\n".join((_VALID_NOTICES,
    "SELECT * FROM valid_notices WHERE origin=? AND notice_id=?"))
_FEED_SQL = "\n".join((_VALID_NOTICES, """
SELECT n.*,(SELECT COUNT(*) FROM valid_notices WHERE read_at IS NULL) AS unread_count
FROM (SELECT 1) LEFT JOIN (
  SELECT * FROM valid_notices
  ORDER BY (read_at IS NOT NULL),created_at DESC,notice_id DESC,origin DESC LIMIT ?
) n ON 1=1
ORDER BY (n.read_at IS NOT NULL),n.created_at DESC,n.notice_id DESC,n.origin DESC
"""))
_COUNT_SQL = "\n".join((_VALID_NOTICES,
    "SELECT COUNT(*) FROM valid_notices WHERE read_at IS NULL"))
_ACKNOWLEDGE_SQL = {
    "agent": "UPDATE app_agent_notices SET read_at=COALESCE(read_at,?) WHERE notice_id=? AND user_id=?",
    "watch": "UPDATE app_watch_notices SET read_at=COALESCE(read_at,?) WHERE notice_id=? AND user_id=?",
}


class NotificationError(ValueError):
    """Activity input exceeds its bounded supported contract."""


class NotificationAccessError(NotificationError):
    """The notice or its canonical target is unavailable for this owner."""


class NotificationConflictError(NotificationError):
    """A notice changed after its response was read."""


def _origin(value: str) -> None:
    if not isinstance(value, str) or value not in {"agent", "watch"}:
        raise NotificationError("Invalid activity origin.")


def _identity(value: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 200:
        raise NotificationError("Invalid activity identity.")


def _agent_notice_title(record: dict) -> str:
    # Provider titles already satisfy the title contract. A legacy prompt-derived
    # fallback is still plain display text; remove controls and markup delimiters.
    value = record["title"] or fallback_conversation_title(record["objective"])
    value = "".join(char for char in value if ord(char) >= 32 and not 127 <= ord(char) <= 159
                    and char not in "<>")
    return " ".join(value.split())[:100] or "Agent task"


def insert_agent_notice(connection: sqlite3.Connection, record: dict, *, created_at: str) -> None:
    """Commit a terminal delivery intent inside the caller's task transaction."""
    kind = {"completed": "completed", "failed": "failure"}.get(record["status"])
    if kind is None:
        raise NotificationError("Only terminal agent results create activity.")
    connection.execute(
        "INSERT OR IGNORE INTO app_agent_notices "
        "(notice_id,user_id,task_id,task_revision,kind,title,body,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), record["user_id"], record["task_id"], record["revision"], kind,
         _agent_notice_title(record), _AGENT_BODIES[kind], created_at),
    )


class NotificationsRepository:
    """Unify current agent results and retained watch notices without hydration."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn, self._lock = connection, lock
        self._clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC)

    def _transaction(self, work: Callable[[], Any]) -> Any:
        with self._lock, write_transaction(self._conn):
            return work()

    def _owner(self, user_id: str) -> None:
        if self._conn.execute(
            "SELECT 1 FROM app_users u WHERE u.user_id=? AND u.status='active' AND NOT EXISTS "
            "(SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)", (user_id,),
        ).fetchone() is None:
            raise NotificationAccessError("Activity is unavailable.")

    @staticmethod
    def _public(row: sqlite3.Row) -> dict:
        target = (
            {"kind": "agent", "task_id": row["target_id"], "revision": row["task_revision"]}
            if row["origin"] == "agent" else
            {"kind": "watch", "watch_id": row["target_id"], "run_id": row["run_id"]}
        )
        return {
            "id": row["notice_id"],
            **{key: row[key] for key in (
                "origin", "kind", "title", "body", "created_at", "read_at",
            )},
            "target": target,
        }

    def _get(self, user_id: str, origin: str, notice_id: str) -> dict:
        row = self._conn.execute(
            _GET_SQL,
            (user_id, user_id, origin, notice_id),
        ).fetchone()
        if row is None:
            raise NotificationAccessError("Activity is unavailable.")
        return self._public(row)

    async def feed(self, *, user_id: str, limit: int = 50) -> dict:
        if type(limit) is not int or not 1 <= limit <= 50:
            raise NotificationError("Invalid activity limit.")

        def work() -> dict:
            with self._lock:
                self._owner(user_id)
                rows = self._conn.execute(
                    _FEED_SQL,
                    (user_id, user_id, limit),
                ).fetchall()
                return {
                    "items": [self._public(row) for row in rows if row["notice_id"] is not None],
                    "unread_count": int(rows[0]["unread_count"]),
                }

        return await asyncio.to_thread(work)

    async def unread_count(self, *, user_id: str) -> int:
        def work() -> int:
            with self._lock:
                self._owner(user_id)
                return int(self._conn.execute(
                    _COUNT_SQL,
                    (user_id, user_id),
                ).fetchone()[0])

        return await asyncio.to_thread(work)

    async def get(self, *, user_id: str, origin: str, notice_id: str) -> dict:
        _origin(origin)
        _identity(notice_id)

        def work() -> dict:
            with self._lock:
                self._owner(user_id)
                return self._get(user_id, origin, notice_id)

        return await asyncio.to_thread(work)

    async def mark_read(self, *, user_id: str, origin: str, notice_id: str) -> dict:
        _origin(origin)
        _identity(notice_id)

        def work() -> dict:
            self._owner(user_id)
            self._get(user_id, origin, notice_id)
            self._conn.execute(
                _ACKNOWLEDGE_SQL[origin],
                (self._clock().astimezone(dt.UTC).isoformat(), notice_id, user_id),
            )
            return self._get(user_id, origin, notice_id)

        return await asyncio.to_thread(self._transaction, work)

    def fence_response(self, *, user_id: str, notices: list[dict] | tuple = ()) -> None:
        """Synchronously reject erased/revised response bodies after the last await."""
        if len(notices) > 50:
            raise NotificationError("Activity response exceeds its allowance.")
        with self._lock:
            self._owner(user_id)
            for payload in notices:
                _origin(payload["origin"])
                current = self._get(user_id, payload["origin"], payload["id"])
                # Another tab may mark the notice read while it is being served;
                # that acknowledgement does not change the body or its authority.
                if any(current[key] != payload[key] for key in (
                    "origin", "kind", "title", "body", "created_at", "target",
                )):
                    raise NotificationConflictError("Activity changed. Refresh before trying again.")
