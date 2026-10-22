"""Fresh server-owned admission for leased Private tax work.

This service is internal authority, not HTTP authentication. Callers must already
be authorized to act for the owner; no request-supplied principal, digest, model
roster or resource limit establishes permission. It does not schedule work.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from audrey.app_state.private_tasks import (
    PrivateTaskAccessError,
    PrivateTaskConflictError,
    PrivateTaskError,
    TaskLimits,
)
from audrey.identity.models import Principal
from audrey.kb.private_documents import private_document_source
from audrey.model_catalog import resolve_model
from audrey.private_processing import prepare_native_private_runtime
from audrey.private_task_runner import (
    PreparedStep,
    prepared_private_step,
    runtime_task_pins,
)
from audrey.processing_policy import ProcessingPolicyError, processing_scope
from audrey.tax_review import require_tax_case
from audrey.user_data_visibility import remote_personal_reads_blocked


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _stat(path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _hash_file(path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    file_id: str
    original: str
    text: str
    source_stat: tuple[int, int, int, int, int]
    text_stat: tuple[int, int, int, int, int]

    def binding(self) -> dict[str, str]:
        return {"file_id": self.file_id, "original": self.original, "text": self.text}


async def _file_identity(app: Any, principal: Principal, file_id: str,
                         previous: _FileIdentity | None = None) -> _FileIdentity:
    # The existing document boundary holds the file lock and verifies owner,
    # ready status, Private collection, path, retained source and source size.
    async with private_document_source(app, principal, file_id) as source:
        path = source.path.with_suffix(".ocr.txt")
        try:
            source_stat = _stat(source.path)
            if path.is_symlink() or path.resolve() != path.absolute():
                raise ValueError
            text_stat = _stat(path)
            if text_stat[2] > 8_000_000:
                raise ValueError
            if (previous is not None and source_stat == previous.source_stat
                    and text_stat == previous.text_stat):
                return previous

            def read() -> tuple[str, str]:
                # Validate the same nonempty UTF-8 text the local reader serves,
                # while pinning its exact bytes independently of the original.
                text = path.read_text(encoding="utf-8")
                if not text.strip() or len(text) > 2_000_000:
                    raise ValueError
                return _hash_file(source.path), _hash_file(path)

            original, text = await asyncio.to_thread(read)
            if source_stat != _stat(source.path) or text_stat != _stat(path):
                raise ValueError
        except (OSError, UnicodeError, ValueError):
            raise PrivateTaskConflictError("A selected Private document changed or is unavailable.") from None
        return _FileIdentity(file_id, original, text, source_stat, text_stat)


class PrivateTaskAdmission:
    """Prepare current owner authority and exact local inputs before a task claim.

    ``prepare`` is compatible with ``PrivateTaskStepRunner``. The returned pins
    carry only hashes; document text is never added to a durable task record.
    Server limits are supplied at construction, never by create/resume payloads.
    """

    def __init__(self, app: Any, *, limits: Mapping[str, int] | None = None) -> None:
        self.app = app
        self.store = app.state.application_store
        if limits is not None and not isinstance(limits, Mapping):
            raise PrivateTaskError("Invalid Private task server limits.")
        try:
            self.limits = TaskLimits.model_validate(dict(limits or {})).model_dump(mode="json")
        except (TypeError, ValueError, ValidationError):
            raise PrivateTaskError("Invalid Private task server limits.") from None

    async def _authorize(self, user_id: str, conversation_id: str):
        principal = await self.store.get_principal(user_id=user_id)
        if (principal is None or principal.status != "active" or principal.role == "bot"
                or "bots" in principal.groups
                or remote_personal_reads_blocked(principal.storage_namespace)):
            raise PrivateTaskAccessError("An active owner and Private case are required.")
        selected = await resolve_model(self.app.state.cfg, self.store, principal, "private")
        if selected is None:
            raise PrivateTaskAccessError("Private model access is unavailable.")
        try:
            case = await require_tax_case(self.app, principal, conversation_id)
        except HTTPException as error:
            if error.status_code in {403, 404}:
                raise PrivateTaskAccessError("An active owner and Private case are required.") from None
            raise PrivateTaskConflictError("The saved Tax review context is unavailable or changed.") from None
        conversation = await self.store.conversations.get(
            user_id=user_id, conversation_id=conversation_id,
        )
        if conversation is None:
            raise PrivateTaskAccessError("Private conversation not found.")
        return principal, conversation, case

    async def _manifest(self, principal: Principal, conversation, case: dict,
                        files: dict[str, _FileIdentity]) -> dict:
        records = await self.store.conversations.list_messages(
            user_id=principal.user_id, conversation_id=conversation.conversation_id,
        )
        if records is None:
            raise PrivateTaskAccessError("Private conversation not found.")
        conversation_ids = {a.file_id for message in records for a in message.attachments}
        snapshot = await self.store.projects.snapshot_for_conversation(
            user_id=principal.user_id, conversation_id=conversation.conversation_id,
        )
        project_ids: set[str] = set()
        instructions = ""
        if conversation.project_id is not None:
            if (snapshot is None or snapshot[0].project_id != conversation.project_id
                    or not snapshot[0].private_required):
                raise PrivateTaskConflictError("The Private Project changed.")
            project_ids = {entry.file_id for entry in snapshot[1]}
            instructions = snapshot[0].instructions
        elif snapshot is not None:
            raise PrivateTaskConflictError("The Private Project changed.")
        selected_ids = conversation_ids | project_ids
        if len(selected_ids) > 100:
            raise PrivateTaskError("Choose at most one hundred Private documents for a task.")
        for file_id in sorted(selected_ids):
            try:
                files[file_id] = await _file_identity(self.app, principal, file_id, files.get(file_id))
            except ProcessingPolicyError:
                raise PrivateTaskConflictError("A selected Private document changed or is unavailable.") from None
        # A case checksum must resolve to the exact still-selected original.
        # New extracted facts may add this metadata without changing the input.
        for binding in case.get("files", []):
            item = files.get(binding["file_id"])
            if (binding["file_id"] not in selected_ids or item is None
                    or binding["checksum"] != item.original):
                raise PrivateTaskConflictError("A saved case document no longer matches its source.")
        return {
            "tax_year": case["tax_year"],
            "reference_release": case["reference_release"],
            "project_id": conversation.project_id,
            "project_instructions": instructions,
            "project_files": sorted(project_ids),
            "conversation_files": sorted(conversation_ids),
            "files": [files[file_id].binding() for file_id in sorted(selected_ids)],
        }

    @asynccontextmanager
    async def prepare(self, user_id: str, conversation_id: str) -> AsyncIterator[PreparedStep]:
        principal, conversation, case = await self._authorize(user_id, conversation_id)
        runtime = await prepare_native_private_runtime(self.app, tax_review=True)
        files: dict[str, _FileIdentity] = {}
        # Only local file reads occur while assembling the manifest. Model
        # metadata is verified by runtime preparation and again on scope entry;
        # every later content call retains the runtime's digest enforcement.
        with processing_scope(runtime.policy):
            manifest = await self._manifest(principal, conversation, case, files)
        access_digest = await self.store.private_tasks.authorization_digest(
            user_id=user_id, conversation_id=conversation_id,
        )
        pins = {**runtime_task_pins(runtime), "input_digest": _digest(manifest),
                "access_digest": access_digest}

        async def guard() -> None:
            current, current_conversation, current_case = await self._authorize(user_id, conversation_id)
            if (current.storage_namespace != principal.storage_namespace
                    or current.role != principal.role or current.groups != principal.groups):
                raise PrivateTaskConflictError("Private task owner access changed.")
            # prepared_private_step invokes the guard before entering its
            # runtime scope too. Bind the already verified policy explicitly
            # for these local reads; this does not open a model transport.
            with processing_scope(runtime.policy):
                current_manifest = await self._manifest(current, current_conversation, current_case, files)
            if _digest(current_manifest) != pins["input_digest"]:
                raise PrivateTaskConflictError("Private task inputs changed; resume with fresh validation.")

        async with prepared_private_step(
            app=self.app, runtime=runtime, principal=principal,
            conversation_id=conversation_id, pins=pins, guard=guard,
        ) as prepared:
            yield prepared

    async def create(self, *, user_id: str, conversation_id: str,
                     expected_case_revision: int) -> dict:
        async with self.prepare(user_id, conversation_id) as prepared:
            await prepared.guard()
            return await self.store.private_tasks.create(
                user_id=user_id, conversation_id=conversation_id,
                expected_case_revision=expected_case_revision,
                pins=prepared.pins, limits=self.limits,
            )

    async def resume(self, *, user_id: str, task_id: str, expected_revision: int,
                     expected_case_revision: int) -> dict:
        task = await self.store.private_tasks.get(user_id=user_id, task_id=task_id)
        if task is None:
            raise PrivateTaskAccessError("Private task not found.")
        async with self.prepare(user_id, task["conversation_id"]) as prepared:
            await prepared.guard()
            return await self.store.private_tasks.control(
                user_id=user_id, task_id=task_id, expected_revision=expected_revision,
                action="resume", expected_case_revision=expected_case_revision,
                pins=prepared.pins,
            )
