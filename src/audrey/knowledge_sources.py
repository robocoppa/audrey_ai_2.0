"""Local, revocable provenance for Project knowledge; no source text is copied."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from audrey.app_state.knowledge import KnowledgeAccessError
from audrey.content_search import SearchDocument
from audrey.content_sources import ContentSourceReader
from audrey.processing_policy import current_processing_policy


def _fingerprint(entry: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            entry,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def source_in_project(document: SearchDocument, project) -> bool:
    """Files/messages require membership; owned reports identify provenance only."""
    return bool(
        not (document.private_required and not project.private_required)
        and (
            (document.kind == "message" and document.project_id == project.project_id)
            or (
                document.kind == "file"
                and project.project_id in document.target.get("project_ids", [])
            )
            or document.kind == "agent"
        )
    )


@dataclass(frozen=True, slots=True)
class KnowledgeResolution:
    stale: bool
    warnings: tuple[str, ...]
    documents: tuple[SearchDocument, ...]
    owner_id: str
    storage_namespace: str
    project_id: str
    project_updated_at: str
    private_required: bool
    entry_id: str
    revision: int
    fingerprint: str


class KnowledgeSourceResolver:
    """Resolve stored references against the owner's current canonical sources."""

    def __init__(self, store, reader: ContentSourceReader) -> None:
        self.store = store
        self.reader = reader

    async def resolve(self, principal, project, entry: dict) -> KnowledgeResolution:
        documents = []
        warnings = []
        sources = entry.get("sources", [])
        if len(sources) > 8:
            warnings.append("Some source references exceed the supported limit.")
        for source in sources[:8]:
            document = await self.reader.resolve(
                principal.user_id,
                principal.storage_namespace,
                source.get("source_id", ""),
            )
            if (
                document is None
                or str(document.revision) != str(source.get("revision", ""))
                or not source_in_project(document, project)
                or not self.reader.is_current(document)
            ):
                warnings.append("A referenced source changed, was detached, or is unavailable.")
                continue
            documents.append(document)
        result = KnowledgeResolution(
            stale=bool(warnings),
            warnings=tuple(dict.fromkeys(warnings)),
            documents=tuple(documents),
            owner_id=principal.user_id,
            storage_namespace=principal.storage_namespace,
            project_id=project.project_id,
            project_updated_at=project.updated_at,
            private_required=project.private_required,
            entry_id=entry["id"],
            revision=entry["revision"],
            fingerprint=_fingerprint(entry),
        )
        if not result.stale and not self.is_current(result):
            return KnowledgeResolution(
                **{
                    name: getattr(result, name)
                    for name in result.__dataclass_fields__
                    if name not in {"stale", "warnings"}
                },
                stale=True,
                warnings=("This Project knowledge changed while its sources were read.",),
            )
        return result

    def is_current(self, resolution: KnowledgeResolution) -> bool:
        """Final synchronous fence, also usable immediately before model dispatch."""
        if resolution.stale or any(
            not self.reader.is_current(document) for document in resolution.documents
        ):
            return False
        with self.store._lock:
            owner = self.store._conn.execute(
                "SELECT 1 FROM app_users WHERE user_id=? AND storage_namespace=? "
                "AND status='active' AND NOT EXISTS "
                "(SELECT 1 FROM account_deletion_requests WHERE user_id=app_users.user_id)",
                (resolution.owner_id, resolution.storage_namespace),
            ).fetchone()
            project = self.store._conn.execute(
                "SELECT updated_at,private_required FROM app_projects WHERE user_id=? AND project_id=?",
                (resolution.owner_id, resolution.project_id),
            ).fetchone()
            try:
                entry = self.store.knowledge.get_sync(
                    user_id=resolution.owner_id,
                    project_id=resolution.project_id,
                    entry_id=resolution.entry_id,
                )
            except KnowledgeAccessError:
                return False
            return bool(
                owner is not None
                and project is not None
                and project["updated_at"] == resolution.project_updated_at
                and bool(project["private_required"]) == resolution.private_required
                and entry is not None
                and _fingerprint(entry) == resolution.fingerprint
            )


def get_knowledge_resolver(app, *, isolated: bool = False) -> KnowledgeSourceResolver:
    """Reuse search's local reader, without making Search a runtime dependency."""
    protected = current_processing_policy() is not None
    standalone = isolated or protected
    # The isolated Private execution state intentionally disallows ordinary
    # app-service lookups and mutation. Build only the local canonical reader.
    cached = None if standalone else getattr(app.state, "knowledge_source_resolver", None)
    if cached is None:
        search = None if standalone else getattr(app.state, "content_search", None)
        reader = getattr(search, "reader", None)
        if reader is None:
            reader = ContentSourceReader(
                app.state.application_store,
                getattr(app.state, "uploads_db", None),
                (app.state.cfg.raw.get("kb") or {}).get("upload_root", "/data/uploads"),
            )
        cached = KnowledgeSourceResolver(app.state.application_store, reader)
        if not standalone:
            app.state.knowledge_source_resolver = cached
    return cached
