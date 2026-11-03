"""Frozen Project manifests for ordinary public-research agents.

Only hashes, revision identities and prefix lengths are persisted here. Canonical
Project/knowledge rows remain the sole authority for context bodies. A deleted
Project leaves saved reports accessible, but never supplies context again.
Callers hold the application store lock and transaction for capture and writes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3

from audrey.canonical_db_sources import (
    database_source_identity,
    database_source_rows,
    source_digest,
)
from audrey.pipeline.complexity import count_tokens

CONTEXT_CHANGED = "Project context changed. Create a new agent to use its current instructions and knowledge."
SUPPORTED_KNOWLEDGE_SOURCE_KINDS = frozenset({"message", "agent"})


class AgentProjectAccessError(ValueError):
    """Only an owned ordinary Project can supply public research context."""


class AgentProjectChangedError(ValueError):
    """A pinned context body no longer matches canonical authority."""


def token_count(text: str) -> int:
    try:
        return count_tokens([{"role": "system", "content": text}])
    except ValueError:
        # Literal tokenizer control-token spellings are still ordinary user
        # text. UTF-8 bytes conservatively bound tokens without interpreting them.
        return len(text.encode("utf-8"))


def _prefix(text: str, limit: int) -> str:
    if token_count(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if token_count(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_refs(row: sqlite3.Row) -> list[dict] | None:
    try:
        sources = json.loads(row["sources_json"])
    except (ValueError, TypeError):
        return None
    if not isinstance(sources, list) or len(sources) > 8:
        return None
    seen = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"source_id", "revision"}:
            return None
        if database_source_identity(source["source_id"]) is None:
            return None
        if source["source_id"] in seen or not isinstance(source["revision"], str) or not source["revision"]:
            return None
        seen.add(source["source_id"])
    return sources


class AgentProjectContexts:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection

    def _project(self, user_id: str, project_id: str):
        return self._conn.execute(
            "SELECT name,instructions FROM app_projects "
            "WHERE user_id=? AND project_id=? AND private_required=0", (user_id, project_id),
        ).fetchone()

    def require_project(self, user_id: str, project_id: str) -> None:
        if self._project(user_id, project_id) is None:
            raise AgentProjectAccessError("Project research context is unavailable.")

    def _manifest(self, task_id: str, user_id: str):
        return self._conn.execute(
            "SELECT * FROM app_agent_project_contexts WHERE task_id=? AND user_id=?",
            (task_id, user_id),
        ).fetchone()

    def _source_proof(
        self, row: sqlite3.Row, *, verify_revision: bool,
        cache: dict | None = None,
    ) -> str | None:
        """Freeze provenance metadata without retaining or dispatching its body.

        The pinned Knowledge revision already freezes the canonical source IDs
        and revisions. A single aggregate metadata hash avoids copying those
        references into the bounded Agent manifest. Full source revisions are
        checked at capture only; every later fence rechecks live metadata.
        """
        sources = _source_refs(row)
        if sources is None:
            return None
        metadata = []
        for source in sources:
            identity = database_source_identity(source["source_id"])
            assert identity is not None
            kind, identifier = identity
            cache_key = (row["user_id"], source["source_id"])
            if cache is not None and cache_key in cache:
                saved = cache[cache_key]
            else:
                rows = database_source_rows(
                    self._conn, owner_id=row["user_id"], kind=kind, source_id=identifier,
                    include_text=verify_revision, completed_report=True,
                )
                saved = None
                if rows:
                    current = rows[0]
                    revision = None
                    if verify_revision:
                        body_key = "content" if kind == "message" else "result"
                        body = current[body_key]
                        if body is not None and body.strip():
                            revision = source_digest(current)
                        current.pop(body_key)
                    saved = (current, revision)
                if cache is not None:
                    # Bodies never enter even this capture-local cache.
                    cache[cache_key] = saved
            if saved is None:
                return None
            current, revision = saved
            if kind == "message" and (
                current["private_required"] or current["project_id"] != row["project_id"]
            ):
                return None
            if verify_revision and revision != source["revision"]:
                return None
            metadata.append({"source_id": source["source_id"], "metadata": current})
        return source_digest(metadata)

    def capture(self, *, task_id: str, user_id: str, project_id: str, objective: str) -> None:
        project = self._project(user_id, project_id)
        if project is None:
            raise AgentProjectAccessError("Project research context is unavailable.")
        terms = set(re.findall(r"\w+", objective.casefold()))
        rows = self._conn.execute(
            "SELECT * FROM app_project_knowledge WHERE user_id=? AND project_id=? "
            "AND approval_state='approved' ORDER BY pinned DESC,updated_at DESC,entry_id DESC LIMIT 200",
            (user_id, project_id),
        ).fetchall()

        def rank(row):
            overlap = len(terms.intersection(re.findall(r"\w+", row["text"].casefold())))
            return (bool(row["pinned"]), overlap > 0, overlap,
                    row["kind"] in {"question", "action"} and row["work_status"] == "open",
                    row["kind"] == "decision", row["updated_at"], row["entry_id"])

        selected, remaining = [], 1000
        # Unsupported refs never crowd out notes. Inspect at most twelve sourced
        # candidates; stale candidates do not prevent later sourceless selection.
        candidates = (row for row in rows if _source_refs(row) is not None)
        sourced_candidates, cache = 0, {}
        for row in sorted(candidates, key=rank, reverse=True):
            if json.loads(row["sources_json"]):
                if sourced_candidates >= 12:
                    continue
                sourced_candidates += 1
            proof = self._source_proof(row, verify_revision=True, cache=cache)
            if proof is None:
                continue
            text = _prefix(row["text"], remaining)
            used = token_count(text)
            if text and used > 0:
                pin = {"id": row["entry_id"], "revision": row["revision"], "chars": len(text)}
                if json.loads(row["sources_json"]):
                    pin["sources_sha256"] = proof
                # Preserve schema32's bound, including unusual long canonical IDs.
                if len(json.dumps([*selected, pin], separators=(",", ":"))) > 4000:
                    continue
                selected.append(pin)
                remaining -= used
            if remaining <= 0 or len(selected) >= 12:
                break
        instructions = _prefix(project["instructions"], 2000)
        self._conn.execute(
            "INSERT INTO app_agent_project_contexts "
            "(task_id,user_id,project_id,instructions_sha256,instructions_chars,knowledge_json) "
            "VALUES (?,?,?,?,?,?)",
            (task_id, user_id, project_id, _digest(project["instructions"]), len(instructions),
             json.dumps(selected, separators=(",", ":"))),
        )

    def _resolve(self, manifest):
        project = self._project(manifest["user_id"], manifest["project_id"])
        if project is None or _digest(project["instructions"]) != manifest["instructions_sha256"]:
            raise AgentProjectChangedError(CONTEXT_CHANGED)
        selected = []
        for pin in json.loads(manifest["knowledge_json"]):
            row = self._conn.execute(
                "SELECT * FROM app_project_knowledge WHERE entry_id=? AND user_id=? AND project_id=?",
                (pin["id"], manifest["user_id"], manifest["project_id"]),
            ).fetchone()
            if row is None or row["revision"] != pin["revision"] or row["approval_state"] != "approved":
                raise AgentProjectChangedError(CONTEXT_CHANGED)
            proof = self._source_proof(row, verify_revision=False)
            if proof is None:
                raise AgentProjectChangedError(CONTEXT_CHANGED)
            sources = json.loads(row["sources_json"])
            if sources and proof != pin.get("sources_sha256"):
                raise AgentProjectChangedError(CONTEXT_CHANGED)
            selected.append({"id": pin["id"], "revision": pin["revision"],
                             **{key: row[key] for key in ("kind", "work_status", "origin")},
                             "text": row["text"][:pin["chars"]]})
        return {"project_id": manifest["project_id"], "project_name": project["name"],
                "instructions": project["instructions"][:manifest["instructions_chars"]], "knowledge": selected}

    def public(self, *, task_id: str, user_id: str) -> dict | None:
        manifest = self._manifest(task_id, user_id)
        if manifest is None:
            return None
        project = self._project(user_id, manifest["project_id"])
        try:
            self._resolve(manifest)
            current = True
        except AgentProjectChangedError:
            current = False
        return {"project_id": manifest["project_id"], "project_name": project["name"] if project else None,
                "available": project is not None, "current": current,
                "instructions_included": manifest["instructions_chars"] > 0,
                "knowledge": [{"id": pin["id"], "revision": pin["revision"]}
                              for pin in json.loads(manifest["knowledge_json"])]}

    def resolve(self, *, task_id: str, user_id: str) -> dict | None:
        manifest = self._manifest(task_id, user_id)
        return self._resolve(manifest) if manifest is not None else None
