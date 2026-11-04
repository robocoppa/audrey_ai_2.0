"""Durable metadata proofs for ordinary file-backed approved Agent notes.

Only the approved note is context. Source bodies are decoded once to verify
canonical reference revisions, then discarded; no implicit file tool is granted.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from audrey.app_state.agent_projects import (
    AgentProjectAccessError,
    AgentProjectChangedError,
    knowledge_source_identity,
)
from audrey.canonical_db_sources import source_digest
from audrey.processing_policy import current_processing_policy
from audrey.user_data_visibility import remote_personal_reads_blocked

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CapturedKnowledgeFileSources:
    owner_id: str
    project_id: str
    proofs: tuple[tuple[str, str, str], ...]


class AgentKnowledgeFiles:
    def __init__(self, app, inputs) -> None:
        self.app, self.inputs = app, inputs
        self.store, self.reader = inputs.store, inputs.reader
        self.store.agents._project_contexts.file_sources = self

    def _metadata(self, user_id: str, project_id: str, source_id: str) -> dict | None:
        """Bounded source metadata; caller owns app lock before any upload lock.

        inputs._row also honors the existing publication thread's marker when
        app and upload locks are already held. Never reacquire that upload lock.
        """
        identity = knowledge_source_identity(source_id)
        if identity is None or identity[0] != "file" or current_processing_policy() is not None:
            return None
        file_id, artifact = identity[1].split(":")
        owner = self.store._conn.execute(
            "SELECT storage_namespace FROM app_users WHERE user_id=? AND status='active' "
            "AND NOT EXISTS (SELECT 1 FROM account_deletion_requests WHERE user_id=app_users.user_id)",
            (user_id,),
        ).fetchone()
        if owner is None or remote_personal_reads_blocked(owner[0]):
            return None
        namespace = owner[0]
        membership = self.inputs._scope(user_id, project_id, namespace, file_id)
        if membership is None:
            return None
        row = self.inputs._row(namespace, file_id)
        if row is None:
            return None
        expected = ({"text", "summary"} if row["kind"] in {"text", "image"} or row["origin"] == "generated"
                    else {"transcript", "summary"} if row["kind"] == "audio"
                    else {"transcript", "visual", "summary"} if row["kind"] == "video" else set())
        if artifact not in expected:
            return None
        self.inputs._directory(namespace)
        projects = self.reader._file_context(user_id, namespace, file_id)
        if projects is None or project_id not in projects:
            return None
        pins = self.reader.file_proof_metadata(namespace, row, artifact)
        return {"namespace": namespace, "catalog": row, "project_ids": projects,
                "membership": membership, "pins": pins}

    def proof(self, row, source: dict, *, verify_revision: bool, captured=None) -> str | None:
        """Recompute the same content-free proof after restart and at every fence."""
        try:
            with self.store._lock:
                metadata = self._metadata(row["user_id"], row["project_id"], source["source_id"])
                if metadata is None:
                    return None
                current = source_digest(metadata)
                if verify_revision:
                    if (not isinstance(captured, CapturedKnowledgeFileSources)
                            or captured.owner_id != row["user_id"] or captured.project_id != row["project_id"]
                            or (source["source_id"], source["revision"], current) not in captured.proofs):
                        return None
                return current
        except (KeyError, TypeError, ValueError, OSError, sqlite3.Error):
            return None

    async def prepare(
        self, *, user_id: str, project_id: str, objective: str,
        knowledge_entry_id: str | None = None, knowledge_entry_revision: int | None = None,
    ) -> CapturedKnowledgeFileSources:
        """Verify at most 12 sourced candidates, with at most 96 unique refs.

        The later creation transaction chooses its context again. Any new or
        changed file reference lacks this trusted prepared proof and is skipped;
        an explicitly assigned origin must still qualify or creation fails.
        """
        empty = CapturedKnowledgeFileSources(user_id, project_id, ())
        with self.store._lock:
            try:
                sources = self.store.agents._project_contexts.file_source_candidates(
                    user_id=user_id, project_id=project_id, objective=objective,
                    knowledge_entry_id=knowledge_entry_id, knowledge_entry_revision=knowledge_entry_revision,
                )
            except (AgentProjectAccessError, AgentProjectChangedError):
                # Creation owns the canonical Project/origin error response.
                return empty
        wanted = {}
        for source in sources:
            wanted.setdefault(source["source_id"], set()).add(source["revision"])
        proofs = []
        for source_id, revisions in wanted.items():
            try:
                with self.store._lock:
                    before = self._metadata(user_id, project_id, source_id)
                if before is None:
                    continue
                document = await self.reader.resolve(user_id, before["namespace"], source_id)
                if (document is None or document.private_required or str(document.revision) not in revisions
                        or not self.reader.is_current(document)):
                    continue
                with self.store._lock:
                    after = self._metadata(user_id, project_id, source_id)
                    if after is None or after != before:
                        continue
                    # This is exactly the canonical file revision shape. It
                    # couples the decoded body to the current chosen paths, not
                    # a stale parser cache or a newly appeared OCR branch.
                    revision = source_digest([after["catalog"], after["project_ids"],
                                              [pin[1] for pin in after["pins"]], document.text])
                    if revision != str(document.revision):
                        continue
                    proofs.append((source_id, revision, source_digest(after)))
            except Exception as error:  # noqa: BLE001 - parser/storage failures expose no source content or paths
                logger.debug("Agent Knowledge file source unavailable (%s)", type(error).__name__)
        return CapturedKnowledgeFileSources(user_id, project_id, tuple(proofs))


def get_agent_knowledge_files(app, *, inputs=None) -> AgentKnowledgeFiles | None:
    if (getattr(app.state, "application_store", None) is None
            or getattr(app.state, "uploads_db", None) is None
            or not isinstance(getattr(getattr(app.state, "cfg", None), "raw", None), dict)):
        return None
    if inputs is None:
        from audrey.agent_file_inputs import get_agent_file_inputs

        inputs = get_agent_file_inputs(app)
    if inputs is None:
        return None
    service = getattr(app.state, "agent_knowledge_files", None)
    if service is None or service.inputs is not inputs:
        service = AgentKnowledgeFiles(app, inputs)
        app.state.agent_knowledge_files = service
    return service


__all__ = ["CapturedKnowledgeFileSources", "AgentKnowledgeFiles", "get_agent_knowledge_files"]
