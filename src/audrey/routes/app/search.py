"""Authenticated local lexical search over canonically revalidated sources."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import UTC, date, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from audrey.auth import require_scope
from audrey.content_search import ContentSearchError, ContentSearchQueryError, _query_terms
from audrey.content_search_service import ContentSearchService
from audrey.identity import Principal

router = APIRouter(tags=["application-search"])
_access = require_scope("compat:full")
_cursor_key = secrets.token_bytes(32)


def _service(request: Request) -> ContentSearchService:
    service = getattr(request.app.state, "content_search", None)
    if service is None:
        raise HTTPException(503, "Local content search is unavailable.")
    return service


def _fingerprint(
    owner: str,
    query: str,
    kind: str,
    project: str | None,
    archived: str,
    since: date | None,
    until: date | None,
    generation: int,
) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                owner,
                query,
                kind,
                project,
                archived,
                since.isoformat() if since else None,
                until.isoformat() if until else None,
                generation,
            ]
        ).encode()
    ).hexdigest()


def _updated_date(value: str) -> date | None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return timestamp.replace(tzinfo=timestamp.tzinfo or UTC).astimezone(UTC).date()
    except (ValueError, TypeError):
        return None


def _cursor(offset: int, fingerprint: str) -> str:
    payload = json.dumps([offset, fingerprint], separators=(",", ":")).encode()
    signature = hmac.digest(_cursor_key, payload, "sha256")
    return base64.urlsafe_b64encode(signature + payload).decode().rstrip("=")


def _offset(cursor: str | None, fingerprint: str) -> int:
    if cursor is None:
        return 0
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        signature, payload = raw[:32], raw[32:]
        if not hmac.compare_digest(signature, hmac.digest(_cursor_key, payload, "sha256")):
            raise ValueError
        offset, bound = json.loads(payload)
        if bound != fingerprint or type(offset) is not int or not 0 <= offset < 200:
            raise ValueError
        return offset
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(422, "Search cursor is invalid. Search again.") from None


@router.get("/search")
async def search_content(
    request: Request,
    response: Response,
    q: str = Query(min_length=1, max_length=256),
    kind: Literal["all", "message", "file", "agent"] = "all",
    project_id: str | None = Query(default=None, min_length=1, max_length=256),
    archived: Literal["all", "active", "archived"] = "all",
    since: date | None = None,
    until: date | None = None,
    cursor: str | None = Query(default=None, max_length=512),
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    service = _service(request)
    try:
        _query_terms(q)
    except ContentSearchQueryError as error:
        raise HTTPException(422, str(error)) from None
    if since is not None and until is not None and since > until:
        raise HTTPException(422, "The start date must be on or before the end date.")
    if project_id is not None:
        project = await service.store.projects.get(user_id=principal.user_id, project_id=project_id)
        if project is None:
            raise HTTPException(404, "Project not found.")
    async with service.requests:
        try:
            scan = await service.prepare(principal.user_id, principal.storage_namespace)
            generation = scan.generation
            fingerprint = _fingerprint(
                principal.user_id, q, kind, project_id, archived, since, until, generation
            )
            offset = _offset(cursor, fingerprint)
            # Project membership is canonical and may be many-to-many for files;
            # apply it after hydration rather than trusting a singular FTS field.
            candidates = await service.index.search(
                principal.user_id,
                q,
                kind=None if kind == "all" else kind,
                archived=None if archived == "all" else archived == "archived",
                offset=offset,
                limit=min(50, 200 - offset),
            )
            hydrated = []
            scanned = 0
            stale = False
            for candidate in candidates:
                scanned += 1
                document = await service.hydrate(
                    principal.user_id, principal.storage_namespace, candidate
                )
                if document is None:
                    stale = True
                    continue
                if project_id is not None and project_id not in {
                    document.project_id,
                    *document.target.get("project_ids", []),
                }:
                    continue
                if archived != "all" and document.archived != (archived == "archived"):
                    continue
                if since is not None or until is not None:
                    updated = _updated_date(document.updated_at)
                    if (
                        updated is None
                        or (since is not None and updated < since)
                        or (until is not None and updated > until)
                    ):
                        continue
                hydrated.append((candidate, document))
                if len(hydrated) == 25:
                    break
            # No awaits after this final ownership/deletion/revision fence.
            items = []
            for candidate, document in hydrated:
                if not service.reader.is_current(document):
                    stale = True
                    continue
                items.append(
                    {
                        key: value
                        for key, value in {
                            **candidate,
                            "id": document.source_id,
                            "revision": str(document.revision),
                        }.items()
                        if key not in {"owner_id", "body_hash", "archived"}
                    }
                )
            next_offset = offset + scanned
            more = (
                bool(candidates)
                and scanned == len(candidates)
                and len(candidates) == min(50, 200 - offset)
            )
            more |= scanned < len(candidates)
            warnings = sorted(scan.warnings)[:20]
            changed = stale or generation != scan.generation
            if changed:
                warnings.append(
                    "Some sources changed during search. Search again for fresh matches."
                )
            if next_offset >= 200 and more:
                warnings.append(
                    "Showing the first 200 matching sources. Refine your search for more."
                )
            return {
                "items": items,
                "next_cursor": _cursor(next_offset, fingerprint)
                if more and next_offset < 200 and not changed
                else None,
                "partial": bool(warnings) or len(scan.done) < 3,
                "indexing": len(scan.done) < 3,
                "warnings": warnings,
            }
        except (ContentSearchError, sqlite3.Error, OSError):
            raise HTTPException(
                503, "Local content search could not finish. Please try again."
            ) from None


@router.get("/search/source")
async def open_search_source(
    request: Request,
    response: Response,
    source_id: str = Query(min_length=1, max_length=256),
    revision: str = Query(min_length=1, max_length=256),
    offset: int = Query(default=0, ge=0, le=500_000),
    principal: Principal = Depends(_access),
):
    response.headers["Cache-Control"] = "private, no-store"
    service = _service(request)
    async with service.requests:
        try:
            document = await service.reader.resolve(
                principal.user_id,
                principal.storage_namespace,
                source_id,
            )
            current = document is not None and service.reader.is_current(document)
        except (ContentSearchError, sqlite3.Error, OSError):
            raise HTTPException(
                503, "Local source text could not be read. Please try again."
            ) from None
        if document is None:
            raise HTTPException(404, "This source is no longer available. Search again.")
        if str(document.revision) != revision or not current:
            raise HTTPException(409, "This source changed. Search again to open its current text.")
        if offset > len(document.text):
            raise HTTPException(422, "Text offset exceeds the source length.")
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
