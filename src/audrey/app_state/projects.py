"""Owner-scoped project storage for grouped conversations and file references."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
import threading
import uuid

from audrey.app_state.records import ProjectFileRecord, ProjectRecord

PROJECT_NAME_MAX_CHARS = 100
PROJECT_INSTRUCTIONS_MAX_CHARS = 4_000
PROJECT_MAX_FILES = 20


class InvalidProjectStateError(ValueError):
    """A project write is empty, malformed, or outside the server limits."""


class ProjectNotFoundError(LookupError):
    """The requested project is not owned by the authenticated user."""


class ProjectFileConflictError(RuntimeError):
    """The file is already attached to the project."""


class ProjectFileLimitError(RuntimeError):
    """The project already has the maximum number of file references."""


class ProjectsRepository:
    """Transactional authority for projects and their durable memberships."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    async def create(
        self,
        *,
        user_id: str,
        name: str,
        instructions: str = "",
    ) -> ProjectRecord:
        return await asyncio.to_thread(self._create_sync, user_id, name, instructions)

    def _create_sync(self, user_id: str, name: str, instructions: str) -> ProjectRecord:
        user_id = _required(user_id, "user id")
        name = _normalize_name(name)
        instructions = _normalize_instructions(instructions)
        project_id = _new_id("proj")
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                owner = self._conn.execute(
                    "SELECT 1 FROM app_users WHERE user_id = ? AND status = 'active'",
                    (user_id,),
                ).fetchone()
                if owner is None:
                    raise InvalidProjectStateError("project owner does not exist")
                self._conn.execute(
                    "INSERT INTO app_projects "
                    "(project_id, user_id, name, instructions, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (project_id, user_id, name, instructions, now, now),
                )
                row = self._project_row_locked(user_id, project_id)
                assert row is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _project_from_row(row)

    async def get(self, *, user_id: str, project_id: str) -> ProjectRecord | None:
        return await asyncio.to_thread(self._get_sync, user_id, project_id)

    def _get_sync(self, user_id: str, project_id: str) -> ProjectRecord | None:
        user_id = _required(user_id, "user id")
        project_id = _required(project_id, "project id")
        with self._lock:
            row = self._project_row_locked(user_id, project_id)
        return _project_from_row(row) if row is not None else None

    async def list_page(
        self,
        *,
        user_id: str,
        limit: int,
        before_updated_at: str | None = None,
        before_project_id: str | None = None,
    ) -> tuple[ProjectRecord, ...]:
        return await asyncio.to_thread(
            self._list_page_sync,
            user_id,
            limit,
            before_updated_at,
            before_project_id,
        )

    def _list_page_sync(
        self,
        user_id: str,
        limit: int,
        before_updated_at: str | None,
        before_project_id: str | None,
    ) -> tuple[ProjectRecord, ...]:
        user_id = _required(user_id, "user id")
        if not 1 <= limit <= 101:
            raise InvalidProjectStateError("project limit must be between 1 and 101")
        if (before_updated_at is None) != (before_project_id is None):
            raise InvalidProjectStateError("project cursor is incomplete")
        with self._lock:
            rows = self._conn.execute(
                "SELECT project_id, user_id, name, instructions, created_at, updated_at "
                "FROM app_projects WHERE user_id = ? "
                "AND (? IS NULL OR updated_at < ? "
                "OR (updated_at = ? AND project_id < ?)) "
                "ORDER BY updated_at DESC, project_id DESC LIMIT ?",
                (
                    user_id,
                    before_updated_at,
                    before_updated_at,
                    before_updated_at,
                    before_project_id,
                    limit,
                ),
            ).fetchall()
        return tuple(_project_from_row(row) for row in rows)

    async def update(
        self,
        *,
        user_id: str,
        project_id: str,
        name: str | None = None,
        instructions: str | None = None,
    ) -> ProjectRecord | None:
        return await asyncio.to_thread(
            self._update_sync,
            user_id,
            project_id,
            name,
            instructions,
        )

    def _update_sync(
        self,
        user_id: str,
        project_id: str,
        name: str | None,
        instructions: str | None,
    ) -> ProjectRecord | None:
        user_id = _required(user_id, "user id")
        project_id = _required(project_id, "project id")
        if name is None and instructions is None:
            raise InvalidProjectStateError("project update has no fields")
        normalized_name = _normalize_name(name) if name is not None else ""
        normalized_instructions = (
            _normalize_instructions(instructions) if instructions is not None else ""
        )
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    "UPDATE app_projects SET "
                    "name = CASE WHEN ? = 1 THEN ? ELSE name END, "
                    "instructions = CASE WHEN ? = 1 THEN ? ELSE instructions END, "
                    "updated_at = ? WHERE user_id = ? AND project_id = ?",
                    (
                        int(name is not None),
                        normalized_name,
                        int(instructions is not None),
                        normalized_instructions,
                        now,
                        user_id,
                        project_id,
                    ),
                )
                if cursor.rowcount == 0:
                    self._conn.rollback()
                    return None
                row = self._project_row_locked(user_id, project_id)
                assert row is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _project_from_row(row)

    async def delete(self, *, user_id: str, project_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, user_id, project_id)

    def _delete_sync(self, user_id: str, project_id: str) -> bool:
        user_id = _required(user_id, "user id")
        project_id = _required(project_id, "project id")
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._project_row_locked(user_id, project_id) is None:
                    self._conn.rollback()
                    return False
                self._conn.execute(
                    "UPDATE app_conversations SET project_id = NULL, updated_at = ? "
                    "WHERE user_id = ? AND project_id = ?",
                    (now, user_id, project_id),
                )
                self._conn.execute(
                    "DELETE FROM app_project_files WHERE project_id = ?",
                    (project_id,),
                )
                cursor = self._conn.execute(
                    "DELETE FROM app_projects WHERE user_id = ? AND project_id = ?",
                    (user_id, project_id),
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return cursor.rowcount == 1

    async def list_files(
        self,
        *,
        user_id: str,
        project_id: str,
    ) -> tuple[ProjectFileRecord, ...] | None:
        return await asyncio.to_thread(self._list_files_sync, user_id, project_id)

    def _list_files_sync(
        self,
        user_id: str,
        project_id: str,
    ) -> tuple[ProjectFileRecord, ...] | None:
        user_id = _required(user_id, "user id")
        project_id = _required(project_id, "project id")
        with self._lock:
            if self._project_row_locked(user_id, project_id) is None:
                return None
            rows = self._conn.execute(
                "SELECT project_id, file_id, added_at FROM app_project_files "
                "WHERE project_id = ? ORDER BY added_at DESC, file_id",
                (project_id,),
            ).fetchall()
        return tuple(_project_file_from_row(row) for row in rows)

    async def add_file(
        self,
        *,
        user_id: str,
        project_id: str,
        file_id: str,
    ) -> ProjectFileRecord:
        return await asyncio.to_thread(self._add_file_sync, user_id, project_id, file_id)

    def _add_file_sync(
        self,
        user_id: str,
        project_id: str,
        file_id: str,
    ) -> ProjectFileRecord:
        user_id = _required(user_id, "user id")
        project_id = _required(project_id, "project id")
        file_id = _required(file_id, "file id")
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._project_row_locked(user_id, project_id) is None:
                    raise ProjectNotFoundError("project not found")
                duplicate = self._conn.execute(
                    "SELECT 1 FROM app_project_files WHERE project_id = ? AND file_id = ?",
                    (project_id, file_id),
                ).fetchone()
                if duplicate is not None:
                    raise ProjectFileConflictError("file is already in this project")
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM app_project_files WHERE project_id = ?",
                    (project_id,),
                ).fetchone()
                if count is not None and int(count[0]) >= PROJECT_MAX_FILES:
                    raise ProjectFileLimitError(
                        f"projects can contain at most {PROJECT_MAX_FILES} files"
                    )
                self._conn.execute(
                    "INSERT INTO app_project_files (project_id, file_id, added_at) "
                    "VALUES (?, ?, ?)",
                    (project_id, file_id, now),
                )
                self._conn.execute(
                    "UPDATE app_projects SET updated_at = ? WHERE user_id = ? AND project_id = ?",
                    (now, user_id, project_id),
                )
                row = self._conn.execute(
                    "SELECT project_id, file_id, added_at FROM app_project_files "
                    "WHERE project_id = ? AND file_id = ?",
                    (project_id, file_id),
                ).fetchone()
                assert row is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _project_file_from_row(row)

    async def remove_file(
        self,
        *,
        user_id: str,
        project_id: str,
        file_id: str,
    ) -> bool | None:
        return await asyncio.to_thread(
            self._remove_file_sync,
            user_id,
            project_id,
            file_id,
        )

    def _remove_file_sync(
        self,
        user_id: str,
        project_id: str,
        file_id: str,
    ) -> bool | None:
        user_id = _required(user_id, "user id")
        project_id = _required(project_id, "project id")
        file_id = _required(file_id, "file id")
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._project_row_locked(user_id, project_id) is None:
                    self._conn.rollback()
                    return None
                cursor = self._conn.execute(
                    "DELETE FROM app_project_files WHERE project_id = ? AND file_id = ?",
                    (project_id, file_id),
                )
                if cursor.rowcount:
                    self._conn.execute(
                        "UPDATE app_projects SET updated_at = ? "
                        "WHERE user_id = ? AND project_id = ?",
                        (now, user_id, project_id),
                    )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return cursor.rowcount == 1

    async def prune_file(
        self,
        *,
        user_id: str,
        file_id: str,
        project_id: str | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self._prune_file_sync,
            user_id,
            file_id,
            project_id,
        )

    def _prune_file_sync(
        self,
        user_id: str,
        file_id: str,
        project_id: str | None,
    ) -> int:
        user_id = _required(user_id, "user id")
        file_id = _required(file_id, "file id")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                query = (
                    "DELETE FROM app_project_files WHERE file_id = ? "
                    "AND project_id IN (SELECT project_id FROM app_projects WHERE user_id = ?)"
                )
                params: tuple[str, ...] = (file_id, user_id)
                if project_id is not None:
                    query += " AND project_id = ?"
                    params = (*params, _required(project_id, "project id"))
                cursor = self._conn.execute(query, params)
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return max(0, cursor.rowcount)

    def _project_row_locked(
        self,
        user_id: str,
        project_id: str,
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT project_id, user_id, name, instructions, created_at, updated_at "
            "FROM app_projects WHERE user_id = ? AND project_id = ?",
            (user_id, project_id),
        ).fetchone()


def _project_from_row(row: sqlite3.Row) -> ProjectRecord:
    return ProjectRecord(
        project_id=str(row["project_id"]),
        user_id=str(row["user_id"]),
        name=str(row["name"]),
        instructions=str(row["instructions"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _project_file_from_row(row: sqlite3.Row) -> ProjectFileRecord:
    return ProjectFileRecord(
        project_id=str(row["project_id"]),
        file_id=str(row["file_id"]),
        added_at=str(row["added_at"]),
    )


def _required(value: str | None, label: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise InvalidProjectStateError(f"{label} is required")
    if len(value) > 200:
        raise InvalidProjectStateError(f"{label} is too long")
    return value


def _normalize_name(value: str | None) -> str:
    value = str(value or "").strip()
    if not value:
        raise InvalidProjectStateError("project name is required")
    if len(value) > PROJECT_NAME_MAX_CHARS:
        raise InvalidProjectStateError(
            f"project name must be at most {PROJECT_NAME_MAX_CHARS} characters"
        )
    return value


def _normalize_instructions(value: str | None) -> str:
    value = str(value or "").strip()
    if len(value) > PROJECT_INSTRUCTIONS_MAX_CHARS:
        raise InvalidProjectStateError(
            f"project instructions must be at most {PROJECT_INSTRUCTIONS_MAX_CHARS} characters"
        )
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


__all__ = [
    "InvalidProjectStateError",
    "PROJECT_INSTRUCTIONS_MAX_CHARS",
    "PROJECT_MAX_FILES",
    "PROJECT_NAME_MAX_CHARS",
    "ProjectFileConflictError",
    "ProjectFileLimitError",
    "ProjectNotFoundError",
    "ProjectsRepository",
]
