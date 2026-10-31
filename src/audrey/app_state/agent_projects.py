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

from audrey.pipeline.complexity import count_tokens

CONTEXT_CHANGED = "Project context changed. Create a new agent to use its current instructions and knowledge."


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


def _sourceless(row: sqlite3.Row) -> bool:
    try:
        return json.loads(row["sources_json"]) == []
    except (ValueError, TypeError):
        return False


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
        for row in sorted((row for row in rows if _sourceless(row)), key=rank, reverse=True)[:12]:
            text = _prefix(row["text"], remaining)
            used = token_count(text)
            if text and used > 0:
                selected.append({"id": row["entry_id"], "revision": row["revision"], "chars": len(text)})
                remaining -= used
            if remaining <= 0:
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
            if row is None or row["revision"] != pin["revision"] or row["approval_state"] != "approved" or not _sourceless(row):
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
