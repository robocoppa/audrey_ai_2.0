"""Repositories for Audrey-owned preferences, conversations, messages, and runs."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
import threading
import uuid
from collections.abc import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from audrey.app_state.records import (
    ChatProjectionDeletionRecord,
    ChatProjectionRecord,
    ConversationRecord,
    FinishedRun,
    MessageRecord,
    RunRecord,
    StartedRun,
    UserPreferences,
)

_ALLOWED_MODES = frozenset(
    {"auto", "fast", "deep", "research", "local", "cloud", "video"}
)
_TERMINAL_RUN_STATUSES = frozenset({"succeeded", "cancelled", "failed"})


class InvalidApplicationStateError(ValueError):
    """A canonical application-state write is incomplete or outside policy."""


class RunAlreadyTerminalError(RuntimeError):
    """A caller attempted a second terminal transition for the same run."""


class ConversationHasActiveRunError(RuntimeError):
    """A destructive conversation mutation raced an active generation."""


class ConversationArchivedError(RuntimeError):
    """A caller attempted to start a run in an archived conversation."""


class PreferencesRepository:
    """Owner-keyed access to durable user preferences."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    async def get(self, *, user_id: str) -> UserPreferences | None:
        return await asyncio.to_thread(self._get_sync, user_id)

    def _get_sync(self, user_id: str) -> UserPreferences | None:
        user_id = _required(user_id, "user id")
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, timezone, persona, response_preferences_json, "
                "created_at, updated_at FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _preferences_from_row(row) if row is not None else None

    async def replace(
        self,
        *,
        user_id: str,
        timezone: str,
        persona: str,
        response_preferences: Mapping[str, object],
    ) -> UserPreferences | None:
        """Replace one owner's preferences, returning ``None`` for no such owner."""

        return await asyncio.to_thread(
            self._replace_sync,
            user_id,
            timezone,
            persona,
            response_preferences,
        )

    def _replace_sync(
        self,
        user_id: str,
        timezone: str,
        persona: str,
        response_preferences: Mapping[str, object],
    ) -> UserPreferences | None:
        user_id = _required(user_id, "user id")
        timezone = _normalize_timezone(timezone)
        persona = str(persona).strip()
        if len(persona) > 4_000:
            raise InvalidApplicationStateError("persona must be at most 4000 characters")
        preferences_json = _encode_preferences(response_preferences)
        now = _utc_now()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    "UPDATE user_preferences SET timezone = ?, persona = ?, "
                    "response_preferences_json = ?, updated_at = ? WHERE user_id = ?",
                    (timezone, persona, preferences_json, now, user_id),
                )
                if cursor.rowcount == 0:
                    self._conn.rollback()
                    return None
                row = self._preference_row_locked(user_id)
                assert row is not None
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return _preferences_from_row(row)

    def _preference_row_locked(self, user_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT user_id, timezone, persona, response_preferences_json, "
            "created_at, updated_at FROM user_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()


class ConversationsRepository:
    """Transactional authority for conversations, messages, and run lifecycle."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    async def create(
        self,
        *,
        user_id: str,
        title: str = "",
        default_mode: str = "auto",
    ) -> ConversationRecord:
        return await asyncio.to_thread(self._create_sync, user_id, title, default_mode)

    def _create_sync(
        self,
        user_id: str,
        title: str,
        default_mode: str,
    ) -> ConversationRecord:
        user_id = _required(user_id, "user id")
        title = _normalize_title(title)
        default_mode = _normalize_mode(default_mode)
        conversation_id = _new_id("con")
        now = _utc_now()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                owner = self._conn.execute(
                    "SELECT 1 FROM app_users WHERE user_id = ? AND status = 'active'",
                    (user_id,),
                ).fetchone()
                if owner is None:
                    raise InvalidApplicationStateError("conversation owner does not exist")
                self._conn.execute(
                    "INSERT INTO app_conversations "
                    "(conversation_id, user_id, title, default_mode, created_at, "
                    "updated_at, last_message_at, archived_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                    (conversation_id, user_id, title, default_mode, now, now),
                )
                row = self._conversation_row_locked(user_id, conversation_id)
                assert row is not None
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return _conversation_from_row(row)

    async def get(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> ConversationRecord | None:
        return await asyncio.to_thread(self._get_sync, user_id, conversation_id)

    def _get_sync(
        self,
        user_id: str,
        conversation_id: str,
    ) -> ConversationRecord | None:
        user_id = _required(user_id, "user id")
        conversation_id = _required(conversation_id, "conversation id")
        with self._lock:
            row = self._conversation_row_locked(user_id, conversation_id)
        return _conversation_from_row(row) if row is not None else None

    async def list_for_user(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> tuple[ConversationRecord, ...]:
        return await asyncio.to_thread(self._list_for_user_sync, user_id, limit)

    async def list_page(
        self,
        *,
        user_id: str,
        archived: bool,
        limit: int,
        search: str = "",
        before_activity_at: str | None = None,
        before_conversation_id: str | None = None,
    ) -> tuple[ConversationRecord, ...]:
        """List one stable keyset page for an owner-scoped title search."""

        return await asyncio.to_thread(
            self._list_page_sync,
            user_id,
            archived,
            limit,
            search,
            before_activity_at,
            before_conversation_id,
        )

    def _list_page_sync(
        self,
        user_id: str,
        archived: bool,
        limit: int,
        search: str,
        before_activity_at: str | None,
        before_conversation_id: str | None,
    ) -> tuple[ConversationRecord, ...]:
        user_id = _required(user_id, "user id")
        if not 1 <= limit <= 200:
            raise InvalidApplicationStateError("conversation limit must be between 1 and 200")
        search = str(search).strip()
        if len(search) > 200:
            raise InvalidApplicationStateError(
                "conversation search must be at most 200 characters"
            )
        if (before_activity_at is None) != (before_conversation_id is None):
            raise InvalidApplicationStateError("conversation cursor is incomplete")

        if before_activity_at is not None and before_conversation_id is not None:
            before_activity_at = _required(before_activity_at, "cursor activity")
            before_conversation_id = _required(
                before_conversation_id,
                "cursor conversation id",
            )

        with self._lock:
            rows = self._conn.execute(
                "SELECT conversation_id, user_id, title, default_mode, created_at, "
                "updated_at, last_message_at, archived_at FROM app_conversations "
                "WHERE user_id = ? "
                "AND ((? = 1 AND archived_at IS NOT NULL) "
                "OR (? = 0 AND archived_at IS NULL)) "
                "AND (? = '' OR instr(lower(title), lower(?)) > 0) "
                "AND (? IS NULL OR COALESCE(last_message_at, created_at) < ? "
                "OR (COALESCE(last_message_at, created_at) = ? "
                "AND conversation_id < ?)) "
                "ORDER BY COALESCE(last_message_at, created_at) DESC, "
                "conversation_id DESC LIMIT ?",
                (
                    user_id,
                    int(archived),
                    int(archived),
                    search,
                    search,
                    before_activity_at,
                    before_activity_at,
                    before_activity_at,
                    before_conversation_id,
                    limit,
                ),
            ).fetchall()
        return tuple(_conversation_from_row(row) for row in rows)

    async def update(
        self,
        *,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
        default_mode: str | None = None,
        archived: bool | None = None,
    ) -> ConversationRecord | None:
        return await asyncio.to_thread(
            self._update_sync,
            user_id,
            conversation_id,
            title,
            default_mode,
            archived,
        )

    def _update_sync(
        self,
        user_id: str,
        conversation_id: str,
        title: str | None,
        default_mode: str | None,
        archived: bool | None,
    ) -> ConversationRecord | None:
        user_id = _required(user_id, "user id")
        conversation_id = _required(conversation_id, "conversation id")
        if title is None and default_mode is None and archived is None:
            raise InvalidApplicationStateError("conversation update has no fields")

        normalized_title = _normalize_title(title) if title is not None else ""
        normalized_mode = _normalize_mode(default_mode) if default_mode is not None else "auto"
        now = _utc_now()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if archived is True and self._has_active_run_locked(user_id, conversation_id):
                    raise ConversationHasActiveRunError(
                        "conversation cannot be archived while a run is active"
                    )
                cursor = self._conn.execute(
                    "UPDATE app_conversations SET "
                    "title = CASE WHEN ? = 1 THEN ? ELSE title END, "
                    "default_mode = CASE WHEN ? = 1 THEN ? ELSE default_mode END, "
                    "archived_at = CASE WHEN ? = 1 THEN ? ELSE archived_at END, "
                    "updated_at = ? "
                    "WHERE user_id = ? AND conversation_id = ?",
                    (
                        int(title is not None),
                        normalized_title,
                        int(default_mode is not None),
                        normalized_mode,
                        int(archived is not None),
                        now if archived else None,
                        now,
                        user_id,
                        conversation_id,
                    ),
                )
                if cursor.rowcount == 0:
                    self._conn.rollback()
                    return None
                row = self._conversation_row_locked(user_id, conversation_id)
                assert row is not None
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return _conversation_from_row(row)

    async def delete(self, *, user_id: str, conversation_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, user_id, conversation_id)

    def _delete_sync(self, user_id: str, conversation_id: str) -> bool:
        user_id = _required(user_id, "user id")
        conversation_id = _required(conversation_id, "conversation id")
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if self._has_active_run_locked(user_id, conversation_id):
                    raise ConversationHasActiveRunError(
                        "conversation cannot be deleted while a run is active"
                    )
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO app_chat_projection_deletions
                      (deletion_id, user_id, conversation_id, requested_at,
                       completed_at, attempts, last_attempt_at, last_error,
                       next_attempt_at)
                    SELECT 'conversation:' || conversation_id, user_id,
                           conversation_id, ?, NULL, 0, NULL, '', ?
                    FROM app_conversations
                    WHERE user_id = ? AND conversation_id = ?
                    """,
                    (now, now, user_id, conversation_id),
                )
                cursor = self._conn.execute(
                    "DELETE FROM app_conversations WHERE user_id = ? AND conversation_id = ?",
                    (user_id, conversation_id),
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return cursor.rowcount == 1

    def _list_for_user_sync(
        self,
        user_id: str,
        limit: int,
    ) -> tuple[ConversationRecord, ...]:
        user_id = _required(user_id, "user id")
        if not 1 <= limit <= 200:
            raise InvalidApplicationStateError("conversation limit must be between 1 and 200")
        with self._lock:
            rows = self._conn.execute(
                "SELECT conversation_id, user_id, title, default_mode, created_at, "
                "updated_at, last_message_at, archived_at FROM app_conversations "
                "WHERE user_id = ? ORDER BY COALESCE(last_message_at, created_at) DESC, "
                "conversation_id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return tuple(_conversation_from_row(row) for row in rows)

    async def list_messages(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> tuple[MessageRecord, ...] | None:
        return await asyncio.to_thread(self._list_messages_sync, user_id, conversation_id)

    def _list_messages_sync(
        self,
        user_id: str,
        conversation_id: str,
    ) -> tuple[MessageRecord, ...] | None:
        user_id = _required(user_id, "user id")
        conversation_id = _required(conversation_id, "conversation id")
        with self._lock:
            if self._conversation_row_locked(user_id, conversation_id) is None:
                return None
            rows = self._conn.execute(
                "SELECT message_id, conversation_id, user_id, run_id, sequence_no, "
                "role, status, content, created_at, updated_at FROM app_messages "
                "WHERE user_id = ? AND conversation_id = ? ORDER BY sequence_no",
                (user_id, conversation_id),
            ).fetchall()
        return tuple(_message_from_row(row) for row in rows)

    async def list_message_page(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
        limit: int,
    ) -> tuple[MessageRecord, ...] | None:
        """List one ascending message page, hidden when the owner does not match."""

        return await asyncio.to_thread(
            self._list_message_page_sync,
            user_id,
            conversation_id,
            after_sequence,
            limit,
        )

    def _list_message_page_sync(
        self,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
        limit: int,
    ) -> tuple[MessageRecord, ...] | None:
        user_id = _required(user_id, "user id")
        conversation_id = _required(conversation_id, "conversation id")
        if after_sequence < 0:
            raise InvalidApplicationStateError("message cursor cannot be negative")
        if not 1 <= limit <= 200:
            raise InvalidApplicationStateError("message limit must be between 1 and 200")
        with self._lock:
            if self._conversation_row_locked(user_id, conversation_id) is None:
                return None
            rows = self._conn.execute(
                "SELECT message_id, conversation_id, user_id, run_id, sequence_no, "
                "role, status, content, created_at, updated_at FROM app_messages "
                "WHERE user_id = ? AND conversation_id = ? AND sequence_no > ? "
                "ORDER BY sequence_no LIMIT ?",
                (user_id, conversation_id, after_sequence, limit),
            ).fetchall()
        return tuple(_message_from_row(row) for row in rows)

    async def get_run(self, *, user_id: str, run_id: str) -> RunRecord | None:
        return await asyncio.to_thread(self._get_run_sync, user_id, run_id)

    def _get_run_sync(self, user_id: str, run_id: str) -> RunRecord | None:
        user_id = _required(user_id, "user id")
        run_id = _required(run_id, "run id")
        with self._lock:
            row = self._run_row_locked(user_id, run_id)
        return _run_from_row(row) if row is not None else None

    async def begin_run(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_content: str,
        mode: str | None = None,
    ) -> StartedRun | None:
        """Create run plus user/assistant messages in one write transaction."""

        return await asyncio.to_thread(
            self._begin_run_sync,
            user_id,
            conversation_id,
            user_content,
            mode,
        )

    def _begin_run_sync(
        self,
        user_id: str,
        conversation_id: str,
        user_content: str,
        mode: str | None,
    ) -> StartedRun | None:
        user_id = _required(user_id, "user id")
        conversation_id = _required(conversation_id, "conversation id")
        user_content = _required(user_content, "user content")
        if len(user_content) > 1_000_000:
            raise InvalidApplicationStateError("user content is too large")
        now = _utc_now()
        run_id = _new_id("run")
        user_message_id = _new_id("msg")
        assistant_message_id = _new_id("msg")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                conversation_row = self._conversation_row_locked(user_id, conversation_id)
                if conversation_row is None:
                    self._conn.rollback()
                    return None
                if conversation_row["archived_at"] is not None:
                    raise ConversationArchivedError(
                        "conversation must be unarchived before starting a run"
                    )
                if self._has_active_run_locked(user_id, conversation_id):
                    raise ConversationHasActiveRunError(
                        "conversation already has an active run"
                    )
                selected_mode = _normalize_mode(mode or str(conversation_row["default_mode"]))
                next_row = self._conn.execute(
                    "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence "
                    "FROM app_messages WHERE user_id = ? AND conversation_id = ?",
                    (user_id, conversation_id),
                ).fetchone()
                first_sequence = int(next_row["next_sequence"])
                self._conn.execute(
                    "INSERT INTO app_runs "
                    "(run_id, conversation_id, user_id, mode, status, started_at, "
                    "completed_at, finish_reason, error_code, virtual_model, "
                    "concrete_model, prompt_tokens, completion_tokens) "
                    "VALUES (?, ?, ?, ?, 'running', ?, NULL, '', '', '', '', 0, 0)",
                    (run_id, conversation_id, user_id, selected_mode, now),
                )
                self._conn.execute(
                    "INSERT INTO app_messages "
                    "(message_id, conversation_id, user_id, run_id, sequence_no, "
                    "role, status, content, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 'user', 'completed', ?, ?, ?)",
                    (
                        user_message_id,
                        conversation_id,
                        user_id,
                        run_id,
                        first_sequence,
                        user_content,
                        now,
                        now,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO app_messages "
                    "(message_id, conversation_id, user_id, run_id, sequence_no, "
                    "role, status, content, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 'assistant', 'in_progress', '', ?, ?)",
                    (
                        assistant_message_id,
                        conversation_id,
                        user_id,
                        run_id,
                        first_sequence + 1,
                        now,
                        now,
                    ),
                )
                self._conn.execute(
                    "UPDATE app_conversations SET updated_at = ?, last_message_at = ? "
                    "WHERE user_id = ? AND conversation_id = ?",
                    (now, now, user_id, conversation_id),
                )
                conversation_row = self._conversation_row_locked(user_id, conversation_id)
                run_row = self._run_row_locked(user_id, run_id)
                message_rows = self._conn.execute(
                    "SELECT message_id, conversation_id, user_id, run_id, sequence_no, "
                    "role, status, content, created_at, updated_at FROM app_messages "
                    "WHERE run_id = ? ORDER BY sequence_no",
                    (run_id,),
                ).fetchall()
                assert conversation_row is not None and run_row is not None
                assert len(message_rows) == 2
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

        return StartedRun(
            conversation=_conversation_from_row(conversation_row),
            run=_run_from_row(run_row),
            user_message=_message_from_row(message_rows[0]),
            assistant_message=_message_from_row(message_rows[1]),
        )

    async def finish_run(
        self,
        *,
        user_id: str,
        run_id: str,
        outcome: str,
        assistant_content: str,
        finish_reason: str = "",
        error_code: str = "",
        virtual_model: str = "",
        concrete_model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> FinishedRun | None:
        """Atomically finalize assistant content and the run's sole outcome."""

        return await asyncio.to_thread(
            self._finish_run_sync,
            user_id,
            run_id,
            outcome,
            assistant_content,
            finish_reason,
            error_code,
            virtual_model,
            concrete_model,
            prompt_tokens,
            completion_tokens,
        )

    def _finish_run_sync(
        self,
        user_id: str,
        run_id: str,
        outcome: str,
        assistant_content: str,
        finish_reason: str,
        error_code: str,
        virtual_model: str,
        concrete_model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> FinishedRun | None:
        user_id = _required(user_id, "user id")
        run_id = _required(run_id, "run id")
        outcome = str(outcome).strip().lower()
        if outcome not in _TERMINAL_RUN_STATUSES:
            raise InvalidApplicationStateError("unsupported terminal run outcome")
        if prompt_tokens < 0 or completion_tokens < 0:
            raise InvalidApplicationStateError("token counts cannot be negative")
        finish_reason = _bounded_metadata(finish_reason, "finish reason")
        error_code = _bounded_metadata(error_code, "error code")
        virtual_model = _bounded_metadata(virtual_model, "virtual model")
        concrete_model = _bounded_metadata(concrete_model, "concrete model")
        assistant_content = str(assistant_content)
        now = _utc_now()
        message_status = "completed" if outcome == "succeeded" else "incomplete"

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._run_row_locked(user_id, run_id)
                if existing is None:
                    self._conn.rollback()
                    return None
                if str(existing["status"]) != "running":
                    raise RunAlreadyTerminalError("run already has a terminal outcome")
                cursor = self._conn.execute(
                    "UPDATE app_runs SET status = ?, completed_at = ?, finish_reason = ?, "
                    "error_code = ?, virtual_model = ?, concrete_model = ?, "
                    "prompt_tokens = ?, completion_tokens = ? "
                    "WHERE user_id = ? AND run_id = ? AND status = 'running'",
                    (
                        outcome,
                        now,
                        finish_reason,
                        error_code,
                        virtual_model,
                        concrete_model,
                        prompt_tokens,
                        completion_tokens,
                        user_id,
                        run_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RunAlreadyTerminalError("run already has a terminal outcome")
                message_cursor = self._conn.execute(
                    "UPDATE app_messages SET status = ?, content = ?, updated_at = ? "
                    "WHERE user_id = ? AND run_id = ? AND role = 'assistant' "
                    "AND status = 'in_progress'",
                    (message_status, assistant_content, now, user_id, run_id),
                )
                if message_cursor.rowcount != 1:
                    raise InvalidApplicationStateError(
                        "run does not have one in-progress assistant message"
                    )
                run_row = self._run_row_locked(user_id, run_id)
                message_row = self._conn.execute(
                    "SELECT message_id, conversation_id, user_id, run_id, sequence_no, "
                    "role, status, content, created_at, updated_at FROM app_messages "
                    "WHERE user_id = ? AND run_id = ? AND role = 'assistant'",
                    (user_id, run_id),
                ).fetchone()
                assert run_row is not None and message_row is not None
                self._insert_native_projection_locked(run_id)
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return FinishedRun(
            run=_run_from_row(run_row),
            assistant_message=_message_from_row(message_row),
        )

    async def recover_interrupted_runs(self) -> int:
        """Fail rows left running by a previous process before serving traffic."""

        return await asyncio.to_thread(self._recover_interrupted_runs_sync)

    def _recover_interrupted_runs_sync(self) -> int:
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    "SELECT run_id FROM app_runs WHERE status = 'running'"
                ).fetchall()
                if not rows:
                    self._conn.commit()
                    return 0
                self._conn.execute(
                    "UPDATE app_messages SET status = 'incomplete', updated_at = ? "
                    "WHERE role = 'assistant' AND status = 'in_progress' "
                    "AND run_id IN (SELECT run_id FROM app_runs WHERE status = 'running')",
                    (now,),
                )
                cursor = self._conn.execute(
                    "UPDATE app_runs SET status = 'failed', completed_at = ?, "
                    "finish_reason = 'error', error_code = 'server_restart' "
                    "WHERE status = 'running'",
                    (now,),
                )
                for row in rows:
                    self._insert_native_projection_locked(str(row["run_id"]))
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return cursor.rowcount

    def _insert_native_projection_locked(self, run_id: str) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO app_chat_projections
              (projection_id, user_id, conversation_id, user_message_id,
               assistant_message_id, partial, virtual_model, concrete_model,
               prompt_tokens, completion_tokens, created_at, enqueued_at,
               attempts, last_attempt_at, last_error, next_attempt_at)
            SELECT
              'native:' || r.run_id,
              r.user_id,
              r.conversation_id,
              user_message.message_id,
              assistant_message.message_id,
              CASE WHEN r.status = 'succeeded' THEN 0 ELSE 1 END,
              r.virtual_model,
              r.concrete_model,
              r.prompt_tokens,
              r.completion_tokens,
              r.started_at,
              NULL,
              0,
              NULL,
              '',
              COALESCE(r.completed_at, r.started_at)
            FROM app_runs AS r
            JOIN app_messages AS user_message
              ON user_message.run_id = r.run_id AND user_message.role = 'user'
            JOIN app_messages AS assistant_message
              ON assistant_message.run_id = r.run_id
             AND assistant_message.role = 'assistant'
            WHERE r.run_id = ? AND r.status != 'running'
            """,
            (run_id,),
        )

    def _conversation_row_locked(
        self,
        user_id: str,
        conversation_id: str,
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT conversation_id, user_id, title, default_mode, created_at, "
            "updated_at, last_message_at, archived_at FROM app_conversations "
            "WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        ).fetchone()

    def _run_row_locked(self, user_id: str, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT run_id, conversation_id, user_id, mode, status, started_at, "
            "completed_at, finish_reason, error_code, virtual_model, concrete_model, "
            "prompt_tokens, completion_tokens FROM app_runs "
            "WHERE user_id = ? AND run_id = ?",
            (user_id, run_id),
        ).fetchone()

    def _has_active_run_locked(self, user_id: str, conversation_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM app_runs WHERE user_id = ? AND conversation_id = ? "
            "AND status = 'running' LIMIT 1",
            (user_id, conversation_id),
        ).fetchone() is not None


class ChatProjectionsRepository:
    """Durable promotion receipts derived from canonical conversation turns."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    async def due(self, *, limit: int = 50) -> tuple[ChatProjectionRecord, ...]:
        return await asyncio.to_thread(self._due_sync, limit)

    def _due_sync(self, limit: int) -> tuple[ChatProjectionRecord, ...]:
        if not 1 <= limit <= 200:
            raise InvalidApplicationStateError(
                "chat projection limit must be between 1 and 200"
            )
        now = _utc_now()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.projection_id, p.user_id, u.storage_namespace,
                       p.conversation_id, COALESCE(user_message.content, ''),
                       COALESCE(assistant_message.content, ''), p.partial,
                       p.virtual_model, p.concrete_model, p.prompt_tokens,
                       p.completion_tokens, p.created_at, p.attempts
                FROM app_chat_projections AS p
                JOIN app_users AS u ON u.user_id = p.user_id
                LEFT JOIN app_messages AS user_message
                  ON user_message.message_id = p.user_message_id
                LEFT JOIN app_messages AS assistant_message
                  ON assistant_message.message_id = p.assistant_message_id
                WHERE p.enqueued_at IS NULL AND p.next_attempt_at <= ?
                ORDER BY p.next_attempt_at, p.created_at, p.projection_id
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return tuple(_chat_projection_from_row(row) for row in rows)

    async def mark_enqueued(self, *, projection_id: str) -> bool:
        return await asyncio.to_thread(self._mark_enqueued_sync, projection_id)

    def _mark_enqueued_sync(self, projection_id: str) -> bool:
        projection_id = _required(projection_id, "projection id")
        now = _utc_now()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE app_chat_projections SET enqueued_at = ?, attempts = attempts + 1, "
                "last_attempt_at = ?, last_error = '' "
                "WHERE projection_id = ? AND enqueued_at IS NULL",
                (now, now, projection_id),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    async def mark_failed(
        self,
        *,
        projection_id: str,
        error: str,
        retry_interval_s: float,
    ) -> bool:
        return await asyncio.to_thread(
            self._mark_failed_sync,
            projection_id,
            error,
            retry_interval_s,
        )

    def _mark_failed_sync(
        self,
        projection_id: str,
        error: str,
        retry_interval_s: float,
    ) -> bool:
        projection_id = _required(projection_id, "projection id")
        if retry_interval_s <= 0:
            raise InvalidApplicationStateError(
                "chat projection retry interval must be positive"
            )
        now = dt.datetime.now(dt.UTC)
        retry_at = (now + dt.timedelta(seconds=retry_interval_s)).isoformat(
            timespec="microseconds"
        )
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE app_chat_projections SET attempts = attempts + 1, "
                "last_attempt_at = ?, last_error = ?, next_attempt_at = ? "
                "WHERE projection_id = ? AND enqueued_at IS NULL",
                (
                    now.isoformat(timespec="microseconds"),
                    str(error)[:500],
                    retry_at,
                    projection_id,
                ),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    async def retry_now(self, *, limit: int = 200) -> int:
        return await asyncio.to_thread(self._retry_now_sync, limit)

    def _retry_now_sync(self, limit: int) -> int:
        if not 1 <= limit <= 10_000:
            raise InvalidApplicationStateError(
                "chat projection retry limit must be between 1 and 10000"
            )
        now = _utc_now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT projection_id FROM app_chat_projections "
                "WHERE enqueued_at IS NULL "
                "ORDER BY next_attempt_at, created_at, projection_id LIMIT ?",
                (limit,),
            ).fetchall()
            if rows:
                self._conn.executemany(
                    "UPDATE app_chat_projections SET next_attempt_at = ? "
                    "WHERE projection_id = ? AND enqueued_at IS NULL",
                    [(now, str(row[0])) for row in rows],
                )
                self._conn.commit()
        return len(rows)

    async def reset_all(self) -> int:
        return await asyncio.to_thread(self._reset_all_sync)

    def _reset_all_sync(self) -> int:
        now = _utc_now()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE app_chat_projections SET enqueued_at = NULL, attempts = 0, "
                "last_attempt_at = NULL, last_error = '', next_attempt_at = ?",
                (now,),
            )
            self._conn.commit()
        return max(0, int(cursor.rowcount))

    async def due_deletions(
        self,
        *,
        limit: int = 50,
    ) -> tuple[ChatProjectionDeletionRecord, ...]:
        return await asyncio.to_thread(self._due_deletions_sync, limit)

    def _due_deletions_sync(
        self,
        limit: int,
    ) -> tuple[ChatProjectionDeletionRecord, ...]:
        if not 1 <= limit <= 200:
            raise InvalidApplicationStateError(
                "chat projection deletion limit must be between 1 and 200"
            )
        now = _utc_now()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT d.deletion_id, d.user_id, u.storage_namespace,
                       d.conversation_id, d.requested_at, d.attempts
                FROM app_chat_projection_deletions AS d
                JOIN app_users AS u ON u.user_id = d.user_id
                WHERE d.completed_at IS NULL AND d.next_attempt_at <= ?
                ORDER BY d.next_attempt_at, d.requested_at, d.deletion_id
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return tuple(_chat_projection_deletion_from_row(row) for row in rows)

    async def mark_deletion_completed(self, *, deletion_id: str) -> bool:
        return await asyncio.to_thread(
            self._mark_deletion_completed_sync,
            deletion_id,
        )

    def _mark_deletion_completed_sync(self, deletion_id: str) -> bool:
        deletion_id = _required(deletion_id, "chat projection deletion id")
        now = _utc_now()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE app_chat_projection_deletions "
                "SET completed_at = ?, attempts = attempts + 1, "
                "last_attempt_at = ?, last_error = '' "
                "WHERE deletion_id = ? AND completed_at IS NULL",
                (now, now, deletion_id),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    async def mark_deletion_failed(
        self,
        *,
        deletion_id: str,
        error: str,
        retry_interval_s: float,
    ) -> bool:
        return await asyncio.to_thread(
            self._mark_deletion_failed_sync,
            deletion_id,
            error,
            retry_interval_s,
        )

    def _mark_deletion_failed_sync(
        self,
        deletion_id: str,
        error: str,
        retry_interval_s: float,
    ) -> bool:
        deletion_id = _required(deletion_id, "chat projection deletion id")
        if retry_interval_s <= 0:
            raise InvalidApplicationStateError(
                "chat projection deletion retry interval must be positive"
            )
        now = dt.datetime.now(dt.UTC)
        retry_at = (now + dt.timedelta(seconds=retry_interval_s)).isoformat(
            timespec="microseconds"
        )
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE app_chat_projection_deletions "
                "SET attempts = attempts + 1, last_attempt_at = ?, "
                "last_error = ?, next_attempt_at = ? "
                "WHERE deletion_id = ? AND completed_at IS NULL",
                (
                    now.isoformat(timespec="microseconds"),
                    str(error)[:500],
                    retry_at,
                    deletion_id,
                ),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    async def retry_deletions_now(self, *, limit: int = 200) -> int:
        return await asyncio.to_thread(self._retry_deletions_now_sync, limit)

    def _retry_deletions_now_sync(self, limit: int) -> int:
        if not 1 <= limit <= 10_000:
            raise InvalidApplicationStateError(
                "chat projection deletion retry limit must be between 1 and 10000"
            )
        now = _utc_now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT deletion_id FROM app_chat_projection_deletions "
                "WHERE completed_at IS NULL "
                "ORDER BY next_attempt_at, requested_at, deletion_id LIMIT ?",
                (limit,),
            ).fetchall()
            if rows:
                self._conn.executemany(
                    "UPDATE app_chat_projection_deletions SET next_attempt_at = ? "
                    "WHERE deletion_id = ? AND completed_at IS NULL",
                    [(now, str(row[0])) for row in rows],
                )
                self._conn.commit()
        return len(rows)

    async def deletion_stats(
        self,
        *,
        user_id: str | None = None,
    ) -> dict[str, int | str]:
        return await asyncio.to_thread(self._deletion_stats_sync, user_id)

    def _deletion_stats_sync(
        self,
        user_id: str | None,
    ) -> dict[str, int | str]:
        columns = (
            "COALESCE(SUM(CASE WHEN completed_at IS NULL THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN completed_at IS NULL THEN attempts ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN completed_at IS NULL AND length(last_error) > 0 "
            "THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN completed_at IS NOT NULL THEN 1 ELSE 0 END), 0), "
            "COALESCE(MIN(CASE WHEN completed_at IS NULL THEN requested_at END), '') "
        )
        with self._lock:
            if user_id is None:
                row = self._conn.execute(
                    "SELECT " + columns + "FROM app_chat_projection_deletions"
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT " + columns + "FROM app_chat_projection_deletions "
                    "WHERE user_id = ?",
                    (_required(user_id, "user id"),),
                ).fetchone()
        return {
            "pending": int(row[0]),
            "attempts": int(row[1]),
            "with_error": int(row[2]),
            "exhausted": 0,
            "completed": int(row[3]),
            "oldest_created_at": str(row[4]),
        }

    async def stats(self, *, user_id: str | None = None) -> dict[str, int | str]:
        return await asyncio.to_thread(self._stats_sync, user_id)

    def _stats_sync(self, user_id: str | None) -> dict[str, int | str]:
        columns = (
            "COALESCE(SUM(CASE WHEN enqueued_at IS NULL THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN enqueued_at IS NULL THEN attempts ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN enqueued_at IS NULL AND length(last_error) > 0 "
            "THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN enqueued_at IS NOT NULL THEN 1 ELSE 0 END), 0), "
            "COALESCE(MIN(CASE WHEN enqueued_at IS NULL THEN created_at END), '') "
        )
        with self._lock:
            if user_id is None:
                row = self._conn.execute(
                    "SELECT " + columns + "FROM app_chat_projections"
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT " + columns + "FROM app_chat_projections WHERE user_id = ?",
                    (_required(user_id, "user id"),),
                ).fetchone()
        return {
            "pending": int(row[0]),
            "attempts": int(row[1]),
            "with_error": int(row[2]),
            "exhausted": 0,
            "completed": int(row[3]),
            "oldest_created_at": str(row[4]),
        }


def _required(value: str | None, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise InvalidApplicationStateError(f"{label} is required")
    return clean


def _normalize_title(value: str) -> str:
    title = str(value).strip()
    if len(title) > 200:
        raise InvalidApplicationStateError("conversation title must be at most 200 characters")
    return title


def _normalize_mode(value: str) -> str:
    mode = _required(value, "mode").lower()
    if mode not in _ALLOWED_MODES:
        raise InvalidApplicationStateError(f"unsupported Audrey mode {mode!r}")
    return mode


def _normalize_timezone(value: str) -> str:
    timezone = _required(value, "timezone")
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise InvalidApplicationStateError("timezone must be a valid IANA name") from exc
    return timezone


def _encode_preferences(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping):
        raise InvalidApplicationStateError("response preferences must be an object")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise InvalidApplicationStateError("response preferences must be valid JSON") from exc
    if len(encoded.encode("utf-8")) > 16_384:
        raise InvalidApplicationStateError("response preferences are too large")
    return encoded


def _bounded_metadata(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if len(clean) > 200:
        raise InvalidApplicationStateError(f"{label} must be at most 200 characters")
    return clean


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _preferences_from_row(row: sqlite3.Row) -> UserPreferences:
    decoded = json.loads(str(row["response_preferences_json"]))
    if not isinstance(decoded, dict):
        raise InvalidApplicationStateError("stored response preferences are not an object")
    return UserPreferences(
        user_id=str(row["user_id"]),
        timezone=str(row["timezone"]),
        persona=str(row["persona"]),
        response_preferences=decoded,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _conversation_from_row(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=str(row["conversation_id"]),
        user_id=str(row["user_id"]),
        title=str(row["title"]),
        default_mode=str(row["default_mode"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_message_at=str(row["last_message_at"]) if row["last_message_at"] else None,
        archived_at=str(row["archived_at"]) if row["archived_at"] else None,
    )


def _message_from_row(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        message_id=str(row["message_id"]),
        conversation_id=str(row["conversation_id"]),
        user_id=str(row["user_id"]),
        run_id=str(row["run_id"]) if row["run_id"] else None,
        sequence_no=int(row["sequence_no"]),
        role=str(row["role"]),
        status=str(row["status"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        conversation_id=str(row["conversation_id"]),
        user_id=str(row["user_id"]),
        mode=str(row["mode"]),
        status=str(row["status"]),
        started_at=str(row["started_at"]),
        completed_at=str(row["completed_at"]) if row["completed_at"] else None,
        finish_reason=str(row["finish_reason"]),
        error_code=str(row["error_code"]),
        virtual_model=str(row["virtual_model"]),
        concrete_model=str(row["concrete_model"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
    )


def _chat_projection_from_row(row: sqlite3.Row) -> ChatProjectionRecord:
    return ChatProjectionRecord(
        projection_id=str(row[0]),
        user_id=str(row[1]),
        storage_namespace=str(row[2]),
        conversation_id=str(row[3]),
        user_content=str(row[4]),
        assistant_content=str(row[5]),
        partial=bool(row[6]),
        virtual_model=str(row[7]),
        concrete_model=str(row[8]),
        prompt_tokens=int(row[9]),
        completion_tokens=int(row[10]),
        created_at=str(row[11]),
        attempts=int(row[12]),
    )


def _chat_projection_deletion_from_row(
    row: sqlite3.Row,
) -> ChatProjectionDeletionRecord:
    return ChatProjectionDeletionRecord(
        deletion_id=str(row[0]),
        user_id=str(row[1]),
        storage_namespace=str(row[2]),
        conversation_id=str(row[3]),
        requested_at=str(row[4]),
        attempts=int(row[5]),
    )


__all__ = [
    "ChatProjectionsRepository",
    "ConversationArchivedError",
    "ConversationHasActiveRunError",
    "ConversationsRepository",
    "InvalidApplicationStateError",
    "PreferencesRepository",
    "RunAlreadyTerminalError",
]
