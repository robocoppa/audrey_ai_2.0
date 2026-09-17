"""One-time, owner-bound import of Audrey's portable chat archive export.

This is an operator migration path, not a live OWUI dependency. Imported
messages do not re-enter the archive projection queue. Source mappings outlive
individual conversation deletion so a rerun cannot resurrect deleted history.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from audrey.app_state.titles import fallback_conversation_title

_SOURCE = "audrey_archive_v1"
_MAX_EXPORT_BYTES = 100 * 1024 * 1024


class HistoryImportError(ValueError):
    """The export, target account, or existing provenance is unsafe to import."""


class ArchiveMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1, max_length=200)
    conversation_id: str = Field(min_length=1, max_length=200)
    conversation_title: str = Field(max_length=1000)
    conversation_created_at: str
    conversation_updated_at: str
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1_000_000)
    created_at: str
    archived_at: str
    partial: bool
    virtual_model: str = Field(max_length=200)
    concrete_model: str = Field(max_length=200)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    @field_validator(
        "created_at", "archived_at", "conversation_created_at",
        "conversation_updated_at",
    )
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if not value:
            return ""
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("archive timestamp is not ISO 8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("archive timestamp must include a timezone")
        return parsed.astimezone(dt.UTC).isoformat(timespec="microseconds")


class ArchiveExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    items: list[ArchiveMessage] = Field(max_length=100_000)
    exported_at: str = ""
    audrey_user_id: str = ""
    account_email: str = ""

    @field_validator("exported_at")
    @classmethod
    def validate_exported_at(cls, value: str) -> str:
        return ArchiveMessage.validate_timestamp(value)


@dataclass(frozen=True, slots=True)
class ImportSummary:
    source_conversations: int
    source_messages: int
    new_conversations: int
    new_messages: int
    existing_messages: int
    skipped_native_conversations: int
    skipped_deleted_conversations: int


def load_archive_export(path: Path) -> ArchiveExport:
    if not path.is_file():
        raise HistoryImportError("archive export file does not exist")
    if path.stat().st_size > _MAX_EXPORT_BYTES:
        raise HistoryImportError("archive export exceeds 100 MiB")
    try:
        export = ArchiveExport.model_validate_json(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise HistoryImportError("invalid Audrey chat export format") from exc
    seen: set[str] = set()
    for item in export.items:
        if item.message_id in seen:
            raise HistoryImportError("archive export contains duplicate message ids")
        if not item.created_at:
            raise HistoryImportError("archive message has no creation timestamp")
        if item.role == "user" and item.partial:
            raise HistoryImportError("archive user message cannot be partial")
        seen.add(item.message_id)
    return export


def _groups(export: ArchiveExport) -> dict[str, list[ArchiveMessage]]:
    groups: dict[str, list[ArchiveMessage]] = defaultdict(list)
    for item in export.items:
        groups[item.conversation_id].append(item)
    for messages in groups.values():
        messages.sort(
            key=lambda item: (
                item.created_at,
                0 if item.role == "user" else 1,
                item.message_id,
            )
        )
    return dict(groups)


def _content_hash(item: ArchiveMessage) -> str:
    fields = (
        item.role, item.content, item.created_at, item.partial,
        item.virtual_model, item.concrete_model,
        item.prompt_tokens, item.completion_tokens,
    )
    encoded = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


class HistoryImportRepository:
    """Preview and apply an export against one exact Audrey account."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    def preview(
        self,
        *,
        user_id: str,
        email: str,
        export: ArchiveExport,
    ) -> ImportSummary:
        with self._lock:
            self._validate_owner(user_id, email, export)
            has_provenance = self._has_provenance()
            counts = {
                "new_conversations": 0,
                "new_messages": 0,
                "existing_messages": 0,
                "skipped_native_conversations": 0,
                "skipped_deleted_conversations": 0,
            }
            groups = _groups(export)
            for source_id, messages in groups.items():
                state, new, existing = self._inspect(
                    user_id, source_id, messages, has_provenance=has_provenance
                )
                if state == "new":
                    counts["new_conversations"] += 1
                elif state == "native":
                    counts["skipped_native_conversations"] += 1
                elif state == "deleted":
                    counts["skipped_deleted_conversations"] += 1
                counts["new_messages"] += len(new)
                counts["existing_messages"] += existing
            return ImportSummary(
                source_conversations=len(groups),
                source_messages=len(export.items),
                **counts,
            )

    def apply(
        self,
        *,
        user_id: str,
        email: str,
        export: ArchiveExport,
    ) -> ImportSummary:
        plan = self.preview(user_id=user_id, email=email, export=export)
        with self._lock:
            if not self._has_provenance():
                raise HistoryImportError("history import schema is not installed")
            for source_id, messages in _groups(export).items():
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._validate_owner(user_id, email, export)
                    state, new, _existing = self._inspect(
                        user_id, source_id, messages, has_provenance=True
                    )
                    if state in {"native", "deleted"} or not new:
                        self._conn.commit()
                        continue
                    if state == "new":
                        target_id = _new_id("con")
                        self._create_conversation(
                            user_id, source_id, target_id, messages
                        )
                    else:
                        target_id = state
                    self._append_messages(user_id, source_id, target_id, new)
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
        return plan

    def _validate_owner(
        self, user_id: str, email: str, export: ArchiveExport,
    ) -> None:
        row = self._conn.execute(
            "SELECT current_email, status FROM app_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None or row["status"] != "active":
            raise HistoryImportError("target Audrey account is not active")
        if str(row["current_email"]).casefold() != email.casefold():
            raise HistoryImportError("target email does not match the exact Audrey user id")
        if export.audrey_user_id and export.audrey_user_id != user_id:
            raise HistoryImportError("export belongs to a different Audrey user id")
        if export.account_email and export.account_email.casefold() != email.casefold():
            raise HistoryImportError("export account email does not match the target")

    def _has_provenance(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'app_history_import_conversations'"
        ).fetchone()
        return row is not None

    def _inspect(
        self,
        user_id: str,
        source_id: str,
        messages: list[ArchiveMessage],
        *,
        has_provenance: bool,
    ) -> tuple[str, list[ArchiveMessage], int]:
        target_id = ""
        if has_provenance:
            mapped = self._conn.execute(
                "SELECT conversation_id FROM app_history_import_conversations "
                "WHERE user_id = ? AND source = ? AND source_conversation_id = ?",
                (user_id, _SOURCE, source_id),
            ).fetchone()
            if mapped is not None:
                target_id = str(mapped["conversation_id"])
                target = self._conn.execute(
                    "SELECT 1 FROM app_conversations "
                    "WHERE user_id = ? AND conversation_id = ?",
                    (user_id, target_id),
                ).fetchone()
                if target is None:
                    return "deleted", [], 0
        if not target_id:
            native = self._conn.execute(
                "SELECT 1 FROM app_conversations WHERE conversation_id = ?",
                (source_id,),
            ).fetchone()
            if native is not None:
                return "native", [], 0

        new: list[ArchiveMessage] = []
        existing = 0
        for item in messages:
            prior = (
                self._conn.execute(
                    "SELECT source_conversation_id, content_sha256 "
                    "FROM app_history_import_messages "
                    "WHERE user_id = ? AND source = ? AND source_message_id = ?",
                    (user_id, _SOURCE, item.message_id),
                ).fetchone()
                if has_provenance else None
            )
            if prior is None:
                new.append(item)
            elif (
                prior["source_conversation_id"] != source_id
                or prior["content_sha256"] != _content_hash(item)
            ):
                raise HistoryImportError(
                    f"source message {item.message_id} changed after import"
                )
            else:
                existing += 1
        if target_id and new:
            active = self._conn.execute(
                "SELECT 1 FROM app_runs WHERE user_id = ? AND conversation_id = ? "
                "AND status = 'running' LIMIT 1",
                (user_id, target_id),
            ).fetchone()
            if active is not None:
                raise HistoryImportError(
                    f"conversation {source_id} has an active native run"
                )
            latest = self._conn.execute(
                "SELECT MAX(source_created_at) AS latest "
                "FROM app_history_import_messages "
                "WHERE user_id = ? AND source = ? AND source_conversation_id = ?",
                (user_id, _SOURCE, source_id),
            ).fetchone()["latest"]
            if latest and new[0].created_at < latest:
                raise HistoryImportError(
                    f"conversation {source_id} has older late-arriving messages"
                )
        return target_id or "new", new, existing

    def _create_conversation(
        self,
        user_id: str,
        source_id: str,
        target_id: str,
        messages: list[ArchiveMessage],
    ) -> None:
        first = messages[0]
        latest = max(item.created_at for item in messages)
        title = first.conversation_title.strip()[:200]
        if not title:
            first_user = next((item.content for item in messages if item.role == "user"), "")
            title = fallback_conversation_title(first_user)
        created = first.conversation_created_at or first.created_at
        updated = first.conversation_updated_at or latest
        now = _utc_now()
        self._conn.execute(
            "INSERT INTO app_conversations "
            "(conversation_id, user_id, title, default_mode, default_model_id, "
            "created_at, updated_at, last_message_at, archived_at) "
            "VALUES (?, ?, ?, 'auto', 'auto', ?, ?, ?, ?)",
            (target_id, user_id, title, created, updated, latest, now),
        )
        self._conn.execute(
            "INSERT INTO app_history_import_conversations "
            "(user_id, source, source_conversation_id, conversation_id, imported_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, _SOURCE, source_id, target_id, now),
        )

    def _append_messages(
        self,
        user_id: str,
        source_id: str,
        target_id: str,
        messages: list[ArchiveMessage],
    ) -> None:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(sequence_no), 0) AS last_sequence "
            "FROM app_messages WHERE user_id = ? AND conversation_id = ?",
            (user_id, target_id),
        ).fetchone()
        sequence = int(row["last_sequence"])
        now = _utc_now()
        for item in messages:
            sequence += 1
            message_id = _new_id("msg")
            self._conn.execute(
                "INSERT INTO app_messages "
                "(message_id, conversation_id, user_id, run_id, sequence_no, "
                "role, status, content, created_at, updated_at) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
                (
                    message_id, target_id, user_id, sequence, item.role,
                    "incomplete" if item.partial else "completed",
                    item.content, item.created_at, item.created_at,
                ),
            )
            self._conn.execute(
                "INSERT INTO app_history_import_messages "
                "(user_id, source, source_message_id, source_conversation_id, "
                "message_id, content_sha256, role, source_created_at, partial, "
                "virtual_model, concrete_model, prompt_tokens, completion_tokens, "
                "imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id, _SOURCE, item.message_id, source_id, message_id,
                    _content_hash(item), item.role, item.created_at, int(item.partial),
                    item.virtual_model, item.concrete_model,
                    item.prompt_tokens, item.completion_tokens, now,
                ),
            )
        last = messages[-1].created_at
        self._conn.execute(
            "UPDATE app_conversations SET "
            "last_message_at = CASE WHEN last_message_at IS NULL OR last_message_at < ? "
            "THEN ? ELSE last_message_at END, "
            "updated_at = CASE WHEN updated_at < ? THEN ? ELSE updated_at END "
            "WHERE user_id = ? AND conversation_id = ?",
            (last, last, last, last, user_id, target_id),
        )


__all__ = [
    "ArchiveExport", "ArchiveMessage", "HistoryImportError",
    "HistoryImportRepository", "ImportSummary", "load_archive_export",
]
