"""Typed records owned by Audrey's canonical application database."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserPreferences:
    """Durable, server-owned preferences for one Audrey user."""

    user_id: str
    timezone: str
    persona: str
    response_preferences: dict[str, object]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """One canonical conversation owned by a stable Audrey user id."""

    conversation_id: str
    user_id: str
    title: str
    default_mode: str
    created_at: str
    updated_at: str
    last_message_at: str | None
    archived_at: str | None


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """One ordered canonical message, including an in-progress assistant row."""

    message_id: str
    conversation_id: str
    user_id: str
    run_id: str | None
    sequence_no: int
    role: str
    status: str
    content: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One durable generation attempt with exactly one terminal transition."""

    run_id: str
    conversation_id: str
    user_id: str
    mode: str
    status: str
    started_at: str
    completed_at: str | None
    finish_reason: str
    error_code: str
    virtual_model: str
    concrete_model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class StartedRun:
    """Records created atomically before Audrey starts streaming a response."""

    conversation: ConversationRecord
    run: RunRecord
    user_message: MessageRecord
    assistant_message: MessageRecord


@dataclass(frozen=True, slots=True)
class FinishedRun:
    """Terminal run metadata and the assistant message finalized with it."""

    run: RunRecord
    assistant_message: MessageRecord


@dataclass(frozen=True, slots=True)
class LocalUserDataPurge:
    """Authoritative local rows erased or reset before remote purge delivery."""

    tokens_deleted: int
    conversations_deleted: int
    messages_deleted: int
    runs_deleted: int
    preferences_reset: bool


__all__ = [
    "ConversationRecord",
    "FinishedRun",
    "LocalUserDataPurge",
    "MessageRecord",
    "RunRecord",
    "StartedRun",
    "UserPreferences",
]
