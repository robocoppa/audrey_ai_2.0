"""Shared canonical SQLite source identities, independent of Search services.

Callers hold the application store lock and authorize the live owner. Full rows
produce existing Search revisions; metadata rows support cheap final fences.
No body, resolver cache, file access or search projection is an authority here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

MAX_DB_SOURCE_CHARS = 500_000


def source_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def database_source_rows(
    connection: sqlite3.Connection,
    *,
    owner_id: str,
    kind: str,
    after: str = "",
    limit: int = 1,
    source_id: str | None = None,
    include_text: bool = True,
    completed_report: bool = False,
) -> list[dict]:
    """Read the same source shape for revisions and metadata-only fences."""
    if kind == "message":
        identity_filter = "AND m.message_id=? " if source_id is not None else ""
        body = (
            f"CASE WHEN length(m.content)<={MAX_DB_SOURCE_CHARS} "
            "THEN m.content ELSE NULL END AS content,"
            if include_text else ""
        )
        rows = connection.execute(
            "SELECT m.message_id,m.conversation_id,m.sequence_no,m.role,m.status,"
            + body
            + "m.updated_at,c.title,c.project_id,c.archived_at,c.private_required "
            "FROM app_messages m JOIN app_conversations c "
            "ON c.conversation_id=m.conversation_id AND c.user_id=m.user_id "
            "WHERE m.user_id=? AND m.role IN ('user','assistant') "
            "AND m.status IN ('completed','incomplete') AND m.message_id>? "
            + identity_filter + "ORDER BY m.message_id LIMIT ?",
            (owner_id, after, *((source_id,) if source_id is not None else ()), limit),
        ).fetchall()
    elif kind == "agent":
        identity_filter = "AND t.task_id=? " if source_id is not None else ""
        body = (
            f"CASE WHEN length(t.result)<={MAX_DB_SOURCE_CHARS} "
            "THEN t.result ELSE NULL END AS result,"
            if include_text else ""
        )
        completed = "AND t.status='completed' " if completed_report else ""
        rows = connection.execute(
            "SELECT t.task_id,t.objective,t.title,"
            + body
            + "t.revision,t.updated_at,p.project_id "
            "FROM app_agent_tasks t LEFT JOIN app_agent_project_contexts x "
            "ON x.task_id=t.task_id AND x.user_id=t.user_id "
            "LEFT JOIN app_projects p ON p.project_id=x.project_id AND p.user_id=t.user_id "
            "AND p.private_required=0 WHERE t.user_id=? AND t.result!='' "
            + completed
            + "AND t.task_id>? " + identity_filter + "ORDER BY t.task_id LIMIT ?",
            (owner_id, after, *((source_id,) if source_id is not None else ()), limit),
        ).fetchall()
    else:
        return []
    return [dict(row) for row in rows]


def database_source_identity(source_id: object) -> tuple[str, str] | None:
    """Accept only canonical message and saved-report source identifiers."""
    if not isinstance(source_id, str) or len(source_id) > 256:
        return None
    parts = source_id.split(":")
    if len(parts) == 2 and parts[0] == "message":
        kind, identifier = parts
    elif len(parts) == 3 and parts[0] == "agent" and parts[2] == "report":
        kind, identifier = parts[:2]
    else:
        return None
    if not (
        0 < len(identifier) <= 200
        and not any(char in identifier for char in ("/", "\\", "\x00", ":"))
        and identifier not in {".", ".."}
    ):
        return None
    try:
        identifier.encode("utf-8")
    except UnicodeError:
        return None
    return kind, identifier
