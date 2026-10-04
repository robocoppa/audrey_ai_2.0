"""Typed records owned by Audrey's canonical application database."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AttachmentSnapshot:
    """Safe file metadata captured when an owner attaches a file to a message."""

    file_id: str
    filename: str
    mime: str
    kind: str
    bytes: int


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """A source observed during a run, attached to its saved assistant message."""

    source_id: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class ToolCallSnapshot:
    """Safe owner-visible activity captured for one server-side tool call."""

    tool_call_id: str
    name: str
    status: str
    arguments: dict[str, Any]
    result: Any | None
    error_code: str


@dataclass(frozen=True, slots=True)
class ModelUsageSnapshot:
    """Generation model activity aggregated for one assistant message."""

    model: str
    calls: int


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
    default_model_id: str
    created_at: str
    updated_at: str
    last_message_at: str | None
    archived_at: str | None
    project_id: str | None


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    """One owner-scoped workspace for conversations and shared file context."""

    project_id: str
    user_id: str
    name: str
    instructions: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ProjectFileRecord:
    """One durable reference from a project to an existing My Files item."""

    project_id: str
    file_id: str
    added_at: str


@dataclass(frozen=True, slots=True)
class FileVersionRecord:
    """Immutable metadata for one owner-scoped source or derived file version."""

    version_id: str
    user_id: str
    file_id: str
    version_number: int
    parent_version_id: str | None
    filename: str
    mime: str
    bytes: int
    content_sha256: str
    origin: str
    created_at: str


@dataclass(frozen=True, slots=True)
class DocumentJobRecord:
    """One exact document operation moving through approval and worker states."""

    job_id: str
    user_id: str
    input_version_id: str
    operation: str
    arguments: dict[str, Any]
    operation_digest: str
    output_mime: str
    summary: str
    preview: str
    requested_by_kind: str
    requested_by_id: str
    idempotency_key: str
    status: str
    attempts: int
    lease_id: str
    lease_expires_at: str | None
    output_version_id: str | None
    error_code: str
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class DocumentApprovalRecord:
    """Single-use approval for one exact document job digest."""

    approval_id: str
    job_id: str
    user_id: str
    operation_digest: str
    expires_at: str
    decision: str
    actor_user_id: str | None
    created_at: str
    decided_at: str | None
    used_at: str | None


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
    attachments: tuple[AttachmentSnapshot, ...] = ()
    sources: tuple[SourceSnapshot, ...] = ()
    tool_calls: tuple[ToolCallSnapshot, ...] = ()
    models: tuple[ModelUsageSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One durable generation attempt with exactly one terminal transition."""

    run_id: str
    conversation_id: str
    user_id: str
    mode: str
    requested_model_id: str
    status: str
    started_at: str
    completed_at: str | None
    finish_reason: str
    error_code: str
    virtual_model: str
    concrete_model: str
    prompt_tokens: int
    completion_tokens: int
    skill_id: str = ""
    skill_version: int = 0
    skill_digest: str = ""
    skill_reason: str = ""


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
class ChatProjectionRecord:
    """One canonical turn awaiting promotion to the search delivery queue."""

    projection_id: str
    user_id: str
    storage_namespace: str
    conversation_id: str
    user_content: str
    assistant_content: str
    partial: bool
    virtual_model: str
    concrete_model: str
    prompt_tokens: int
    completion_tokens: int
    skill_id: str
    skill_version: int
    skill_digest: str
    skill_reason: str
    created_at: str
    attempts: int


@dataclass(frozen=True, slots=True)
class ChatProjectionDeletionRecord:
    """One canonical conversation deletion awaiting search cleanup handoff."""

    deletion_id: str
    user_id: str
    storage_namespace: str
    conversation_id: str
    requested_at: str
    attempts: int


@dataclass(frozen=True, slots=True)
class LocalUserDataPurge:
    """Authoritative local rows erased or reset before remote purge delivery."""

    tokens_deleted: int
    conversations_deleted: int
    messages_deleted: int
    runs_deleted: int
    preferences_reset: bool


@dataclass(frozen=True, slots=True)
class AdminUserRecord:
    """Safe account and authorization fields shown in the admin application."""

    user_id: str
    email: str
    display_name: str
    role: str
    status: str
    groups: tuple[str, ...]
    auth_provider: str
    created_at: str
    updated_at: str
    last_seen_at: str
    deletion_pending: bool = False


@dataclass(frozen=True, slots=True)
class AccessRoleRecord:
    """A managed user role and its membership count."""

    role_id: str
    label: str
    description: str
    system: bool
    user_count: int


@dataclass(frozen=True, slots=True)
class ModelPublicationProfile:
    """Audrey-owned presentation and role grants for one model."""

    model_id: str
    visibility: str
    roles: tuple[str, ...]
    display_name: str
    portrait_mime: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ModelAccessPolicy:
    """Mutable audience overlay for one deployment-defined model."""

    model_id: str
    enabled: bool
    audience: str
    updated_at: str


__all__ = [
    "AccessRoleRecord",
    "AdminUserRecord",
    "ModelPublicationProfile",
    "AttachmentSnapshot",
    "ChatProjectionDeletionRecord",
    "ChatProjectionRecord",
    "ConversationRecord",
    "DocumentApprovalRecord",
    "DocumentJobRecord",
    "FileVersionRecord",
    "FinishedRun",
    "LocalUserDataPurge",
    "MessageRecord",
    "ModelAccessPolicy",
    "ProjectFileRecord",
    "ProjectRecord",
    "RunRecord",
    "StartedRun",
    "UserPreferences",
]
