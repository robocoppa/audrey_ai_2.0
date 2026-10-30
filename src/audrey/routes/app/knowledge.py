"""Owned Project knowledge, explicit proposal review, and local source previews."""

from __future__ import annotations

import base64
import hmac
import json
import logging
import secrets
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from audrey.app_state.agents import AgentTaskError
from audrey.app_state.knowledge import (
    MAX_ENTRY_CHARS,
    MAX_PROJECT_ENTRIES,
    MAX_SOURCES,
    KnowledgeAccessError,
    KnowledgeConflictError,
    KnowledgeError,
    KnowledgeLimitError,
)
from audrey.auth import require_scope
from audrey.content_search import ContentSearchError
from audrey.identity import Principal
from audrey.knowledge_sources import get_knowledge_resolver, source_in_project

router = APIRouter(prefix="/projects/{project_id}/knowledge", tags=["application-knowledge"])
_access = require_scope("compat:full")
_key = secrets.token_bytes(32)
Kind = Literal["fact", "decision", "question", "action"]
log = logging.getLogger(__name__)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=1, max_length=256)


class CreateEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Kind
    text: str = Field(min_length=1, max_length=MAX_ENTRY_CHARS)
    pinned: bool = Field(default=False, strict=True)
    sources: list[SourceRef] = Field(default_factory=list, max_length=MAX_SOURCES)
    request_id: str = Field(min_length=1, max_length=200)


class CreateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    task_id: str = Field(min_length=1, max_length=256)
    task_revision: int = Field(ge=1, strict=True)
    kind: Kind
    text: str = Field(min_length=1, max_length=MAX_ENTRY_CHARS)
    pinned: bool = Field(default=False, strict=True)
    replaces_id: str | None = Field(default=None, min_length=1, max_length=256)
    base_revision: int | None = Field(default=None, ge=1, strict=True)
    request_id: str = Field(min_length=1, max_length=200)


class UpdateEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_revision: int = Field(ge=1, strict=True)
    kind: Kind | None = None
    text: str | None = Field(default=None, min_length=1, max_length=MAX_ENTRY_CHARS)
    pinned: bool | None = Field(default=None, strict=True)
    work_status: Literal["open", "resolved", "completed"] | None = None
    sources: list[SourceRef] | None = Field(default=None, max_length=MAX_SOURCES)


class DecideEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]
    expected_revision: int = Field(ge=1, strict=True)


def _error(error: KnowledgeError) -> HTTPException:
    code = (
        404
        if isinstance(error, KnowledgeAccessError)
        else (
            409
            if isinstance(error, KnowledgeConflictError)
            else 429
            if isinstance(error, KnowledgeLimitError)
            else 422
        )
    )
    return HTTPException(code, str(error))


async def _resources(request: Request, principal: Principal, project_id: str):
    store = getattr(request.app.state, "application_store", None)
    if store is None or not hasattr(store, "knowledge"):
        raise HTTPException(503, "Project knowledge is unavailable.")
    current = await store.get_principal(user_id=principal.user_id)
    project = await store.projects.get(user_id=principal.user_id, project_id=project_id)
    if current is None or current.status != "active" or project is None:
        raise HTTPException(404, "Project knowledge is unavailable.")
    try:
        resolver = get_knowledge_resolver(request.app)
    except (AttributeError, OSError, sqlite3.Error, ContentSearchError, RuntimeError):
        raise HTTPException(503, "Project knowledge sources are unavailable.") from None
    return store, project, resolver


def _page_cursor(owner: str, project: str, record: dict) -> str:
    data = json.dumps([owner, project, record["updated_at"], record["id"]]).encode()
    return base64.urlsafe_b64encode(hmac.digest(_key, data, "sha256") + data).decode().rstrip("=")


def _decode(cursor: str | None, owner: str, project: str):
    if cursor is None:
        return None, None
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        signature, data = raw[:32], raw[32:]
        if not hmac.compare_digest(signature, hmac.digest(_key, data, "sha256")):
            raise ValueError
        bound_owner, bound_project, updated, entry_id = json.loads(data)
        if (
            bound_owner != owner
            or bound_project != project
            or not isinstance(updated, str)
            or not isinstance(entry_id, str)
        ):
            raise ValueError
        return updated, entry_id
    except (ValueError, TypeError):
        raise HTTPException(422, "Knowledge cursor is invalid. Refresh the list.") from None


async def _hydrate(principal, project, resolver, entry):
    try:
        resolution = await resolver.resolve(principal, project, entry)
    except (OSError, sqlite3.Error, ContentSearchError, RuntimeError):
        raise HTTPException(503, "Project knowledge sources are unavailable.") from None
    current = resolver.store.knowledge.get_sync(
        user_id=principal.user_id,
        project_id=project.project_id,
        entry_id=entry["id"],
    )
    if current["revision"] != entry["revision"]:
        raise KnowledgeConflictError("Knowledge changed. Refresh before trying again.")
    stale = resolution.stale or not resolver.is_current(resolution)
    warnings = list(resolution.warnings)
    if stale and not warnings:
        warnings = ["A referenced source changed. Refresh its reference before reuse."]
    return {**entry, "stale": stale, "source_warnings": warnings}, resolution


async def _read_source(principal, resolver, source_id):
    try:
        return await resolver.reader.resolve(
            principal.user_id,
            principal.storage_namespace,
            source_id,
        )
    except (OSError, sqlite3.Error, ContentSearchError, RuntimeError):
        raise HTTPException(503, "Project knowledge sources are unavailable.") from None


def _project_current(store, principal, project):
    """Fence Project/privacy changes after asynchronous local source reads."""
    with store._lock:
        return (
            store._conn.execute(
                "SELECT 1 FROM app_projects p JOIN app_users u ON u.user_id=p.user_id "
                "WHERE p.user_id=? AND u.storage_namespace=? AND p.project_id=? "
                "AND p.updated_at=? AND p.private_required=? AND u.status='active' "
                "AND NOT EXISTS (SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)",
                (
                    principal.user_id,
                    principal.storage_namespace,
                    project.project_id,
                    project.updated_at,
                    int(project.private_required),
                ),
            ).fetchone()
            is not None
        )


async def _refresh_search(request, principal, entry_ids):
    """Refresh a rebuildable projection after the canonical write has committed."""
    service = getattr(request.app.state, "content_search", None)
    refresh = getattr(service, "refresh_knowledge", None)
    if refresh is None:
        return
    try:
        await refresh(principal.user_id, principal.storage_namespace, entry_ids)
    except (ContentSearchError, OSError, sqlite3.Error):
        # Canonical edits remain successful; the bounded indexer repairs Search.
        # Exception messages may contain private source paths or content.
        log.warning("Project knowledge search refresh is unavailable.")


async def _inspect_refs(principal, project, resolver, refs):
    documents = []
    for ref in refs:
        if ref["source_id"].split(":", 1)[0] not in {"message", "file", "agent"}:
            raise HTTPException(422, "Unsupported knowledge source.")
        document = await _read_source(principal, resolver, ref["source_id"])
        if (
            document is None
            or str(document.revision) != ref["revision"]
            or not source_in_project(document, project)
        ):
            raise HTTPException(409, "A source changed or is unavailable in this Project.")
        documents.append(document)
    if not _project_current(resolver.store, principal, project) or any(
        not resolver.reader.is_current(document) for document in documents
    ):
        raise HTTPException(409, "A source changed while it was being read.")


@router.get("")
async def list_knowledge(
    request: Request,
    response: Response,
    project_id: str,
    limit: int = Query(default=50, ge=1, le=50),
    cursor: str | None = Query(default=None, max_length=1024),
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    before, before_id = _decode(cursor, principal.user_id, project_id)
    try:
        entries = await store.knowledge.list(
            user_id=principal.user_id,
            project_id=project_id,
            limit=limit + 1,
            before_updated_at=before,
            before_id=before_id,
        )
        page = entries[:limit]
        hydrated = [await _hydrate(principal, project, resolver, entry) for entry in page]
        # Recheck native rows after all awaited provenance reads. Stale source
        # metadata may remain on a user note; deleted notes must never return.
        items = []
        for value, proof in hydrated:
            current = store.knowledge.get_sync(
                user_id=principal.user_id, project_id=project_id, entry_id=value["id"]
            )
            if current["revision"] != value["revision"]:
                raise KnowledgeConflictError("Knowledge changed. Refresh the list.")
            if not proof.stale and not resolver.is_current(proof):
                value = {
                    **value,
                    "stale": True,
                    "source_warnings": [
                        "A referenced source changed. Refresh its reference before reuse."
                    ],
                }
            items.append(value)
        return {
            "items": items,
            "next_cursor": _page_cursor(principal.user_id, project_id, page[-1])
            if len(entries) > limit
            else None,
            "limits": {
                "max_entries": MAX_PROJECT_ENTRIES,
                "max_text_chars": MAX_ENTRY_CHARS,
                "max_sources": MAX_SOURCES,
            },
        }
    except KnowledgeError as error:
        raise _error(error) from None


@router.post("", status_code=201)
async def create_knowledge(
    body: CreateEntry,
    request: Request,
    response: Response,
    project_id: str,
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    refs = [ref.model_dump() for ref in body.sources]
    await _inspect_refs(principal, project, resolver, refs)
    try:
        entry = await store.knowledge.create(
            user_id=principal.user_id,
            project_id=project_id,
            kind=body.kind,
            text=body.text,
            pinned=body.pinned,
            sources=refs,
            request_id=body.request_id,
        )
        await _refresh_search(request, principal, [entry["id"]])
        return (await _hydrate(principal, project, resolver, entry))[0]
    except KnowledgeError as error:
        raise _error(error) from None


@router.post("/proposals", status_code=201)
async def propose_knowledge(
    body: CreateProposal,
    request: Request,
    response: Response,
    project_id: str,
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    try:
        task = await store.agents.get(user_id=principal.user_id, task_id=body.task_id)
    except AgentTaskError:
        raise HTTPException(404, "Saved report is unavailable.") from None
    if (
        task["status"] != "completed"
        or not task["result"]
        or task["revision"] != body.task_revision
    ):
        raise HTTPException(409, "The saved report changed or is not complete.")
    document = await _read_source(principal, resolver, f"agent:{body.task_id}:report")
    if document is None or not resolver.reader.is_current(document):
        raise HTTPException(409, "The saved report changed.")
    try:
        current_task = await store.agents.get(user_id=principal.user_id, task_id=body.task_id)
    except AgentTaskError:
        raise HTTPException(404, "Saved report is unavailable.") from None
    if (
        current_task["revision"] != body.task_revision
        or current_task["status"] != "completed"
        or not resolver.reader.is_current(document)
    ):
        raise HTTPException(409, "The saved report changed.")
    try:
        entry = await store.knowledge.create(
            user_id=principal.user_id,
            project_id=project_id,
            kind=body.kind,
            text=body.text,
            pinned=body.pinned,
            origin="agent_report",
            source_task_id=body.task_id,
            sources=[{"source_id": document.source_id, "revision": str(document.revision)}],
            replaces_id=body.replaces_id,
            base_revision=body.base_revision,
            request_id=body.request_id,
        )
        await _refresh_search(request, principal, [entry["id"]])
        return (await _hydrate(principal, project, resolver, entry))[0]
    except KnowledgeError as error:
        raise _error(error) from None


@router.get("/{entry_id}/history")
async def knowledge_history(
    request: Request,
    response: Response,
    project_id: str,
    entry_id: str,
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    try:
        entries = await store.knowledge.history(
            user_id=principal.user_id, project_id=project_id, entry_id=entry_id, limit=100
        )
        current = store.knowledge.get_sync(
            user_id=principal.user_id, project_id=project_id, entry_id=entry_id
        )
        hydrated, _ = await _hydrate(principal, project, resolver, current)
        return {
            "items": [
                {
                    **entry,
                    "stale": entry["revision"] != current["revision"] or hydrated["stale"],
                    "source_warnings": [
                        "Historical references are checked when opening their current entry."
                    ]
                    if entry["revision"] != current["revision"]
                    else hydrated["source_warnings"],
                }
                for entry in entries
            ]
        }
    except KnowledgeError as error:
        raise _error(error) from None


@router.get("/{entry_id}/sources/{source_index}")
async def knowledge_source(
    request: Request,
    response: Response,
    project_id: str,
    entry_id: str,
    source_index: int,
    expected_revision: int = Query(ge=1),
    offset: int = Query(default=0, ge=0, le=500_000),
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    try:
        entry = await store.knowledge.get(
            user_id=principal.user_id, project_id=project_id, entry_id=entry_id
        )
        if entry["revision"] != expected_revision:
            raise KnowledgeConflictError("Knowledge changed. Refresh before opening its source.")
        if not 0 <= source_index < len(entry["sources"]):
            raise KnowledgeAccessError("Knowledge source is unavailable.")
        ref = entry["sources"][source_index]
        document = await _read_source(principal, resolver, ref["source_id"])
        current = store.knowledge.get_sync(
            user_id=principal.user_id, project_id=project_id, entry_id=entry_id
        )
        if (
            current["revision"] != expected_revision
            or document is None
            or str(document.revision) != ref["revision"]
            or not source_in_project(document, project)
            or not resolver.reader.is_current(document)
            or not _project_current(store, principal, project)
        ):
            raise KnowledgeConflictError("This source changed, was detached, or is unavailable.")
        if offset > len(document.text):
            raise HTTPException(422, "Text offset exceeds source length.")
        end = min(len(document.text), offset + 4000)
        return {
            "source_id": document.source_id,
            "revision": str(document.revision),
            "kind": document.kind,
            "title": document.title,
            "text": document.text[offset:end],
            "offset": offset,
            "next_offset": end if end < len(document.text) else None,
            "total_chars": len(document.text),
            "target": document.target,
            "availability": document.availability,
            "private_required": document.private_required,
        }
    except KnowledgeError as error:
        raise _error(error) from None


@router.get("/{entry_id}")
async def get_knowledge(
    request: Request,
    response: Response,
    project_id: str,
    entry_id: str,
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    try:
        entry = await store.knowledge.get(
            user_id=principal.user_id, project_id=project_id, entry_id=entry_id
        )
        value, _ = await _hydrate(principal, project, resolver, entry)
        current = store.knowledge.get_sync(
            user_id=principal.user_id, project_id=project_id, entry_id=entry_id
        )
        if current["revision"] != entry["revision"]:
            raise KnowledgeConflictError("Knowledge changed. Refresh before trying again.")
        return value
    except KnowledgeError as error:
        raise _error(error) from None


@router.patch("/{entry_id}")
async def update_knowledge(
    body: UpdateEntry,
    request: Request,
    response: Response,
    project_id: str,
    entry_id: str,
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    changes = body.model_dump(exclude_unset=True, exclude={"expected_revision"})
    if any(value is None for value in changes.values()):
        raise HTTPException(422, "Knowledge fields cannot be null.")
    if "sources" in changes:
        await _inspect_refs(principal, project, resolver, changes["sources"])
    try:
        entry = await store.knowledge.update(
            user_id=principal.user_id,
            project_id=project_id,
            entry_id=entry_id,
            expected_revision=body.expected_revision,
            **changes,
        )
        await _refresh_search(request, principal, [entry["id"]])
        return (await _hydrate(principal, project, resolver, entry))[0]
    except KnowledgeError as error:
        raise _error(error) from None


@router.post("/{entry_id}/decision")
async def decide_knowledge(
    body: DecideEntry,
    request: Request,
    response: Response,
    project_id: str,
    entry_id: str,
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, project, resolver = await _resources(request, principal, project_id)
    try:
        current = await store.knowledge.get(
            user_id=principal.user_id, project_id=project_id, entry_id=entry_id
        )
        _, proof = await _hydrate(principal, project, resolver, current)
        if body.decision == "approve" and (proof.stale or not resolver.is_current(proof)):
            raise KnowledgeConflictError(
                "Refresh unavailable sources before approving this proposal."
            )
        entry = await store.knowledge.decide(
            user_id=principal.user_id,
            project_id=project_id,
            entry_id=entry_id,
            expected_revision=body.expected_revision,
            action=body.decision,
        )
        refresh_ids = [entry["id"]]
        if body.decision == "approve" and entry["replaces_id"] is not None:
            refresh_ids.append(entry["replaces_id"])
        await _refresh_search(request, principal, refresh_ids)
        return (await _hydrate(principal, project, resolver, entry))[0]
    except KnowledgeError as error:
        raise _error(error) from None


@router.delete("/{entry_id}")
async def delete_knowledge(
    request: Request,
    response: Response,
    project_id: str,
    entry_id: str,
    expected_revision: int = Query(ge=1),
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    store, _, _ = await _resources(request, principal, project_id)
    try:
        await store.knowledge.delete(
            user_id=principal.user_id,
            project_id=project_id,
            entry_id=entry_id,
            expected_revision=expected_revision,
        )
        await _refresh_search(request, principal, [entry_id])
        return {"deleted": True}
    except KnowledgeError as error:
        raise _error(error) from None
