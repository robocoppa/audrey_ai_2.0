"""Explicit, bounded ordinary Project file inputs with durable revocation pins.

Only content hashes, canonical identities and relative stat pins are persisted.
Text is read locally for each step; hot-path fences never parse source bodies.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from audrey.app_state.agent_projects import _prefix
from audrey.app_state.agents import AgentTaskAccessError, AgentTaskContextError, AgentTaskError
from audrey.content_sources import ContentSourceReader, _identifier, _presence_pin
from audrey.kb.user_store import sanitize_user

FILES_CHANGED = "Selected files changed or are unavailable. Create a new agent to use their current text."
_PUBLIC = ("file_id", "filename", "artifact", "chars", "total_chars", "truncated", "origin")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CapturedFileInputs:
    owner_id: str
    project_id: str
    pins: tuple[dict, ...]


def public_file(pin: dict) -> dict:
    return {key: pin[key] for key in _PUBLIC}


class AgentFileInputs:
    def __init__(self, app) -> None:
        self.store = app.state.application_store
        self.uploads = getattr(app.state, "uploads_db", None)
        self.reader = ContentSourceReader(self.store, self.uploads,
            (app.state.cfg.raw.get("kb") or {}).get("upload_root", "/data/uploads"))
        self._catalog_lock_state = threading.local()
        self.store.agents._file_input_service = self

    @contextmanager
    def catalog_lock_already_held(self):
        """Publication owns app→upload locks; nested input reads must not lock again."""
        previous = getattr(self._catalog_lock_state, "held", False)
        self._catalog_lock_state.held = True
        try:
            yield
        finally:
            self._catalog_lock_state.held = previous

    def _scope(self, owner_id: str, project_id: str, namespace: str, file_id: str) -> dict | None:
        """Caller holds the application lock, then acquires the upload lock."""
        row = self.store._conn.execute(
            "SELECT f.added_at,f.origin FROM app_users u JOIN app_projects p ON p.user_id=u.user_id "
            "JOIN app_project_files f ON f.project_id=p.project_id "
            "WHERE u.user_id=? AND u.storage_namespace=? AND u.status='active' "
            "AND p.project_id=? AND p.private_required=0 AND f.file_id=? "
            "AND NOT EXISTS (SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)",
            (owner_id, namespace, project_id, file_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def _row(self, namespace: str, file_id: str) -> dict | None:
        if self.uploads is None:
            return None
        read = (self.uploads._search_rows_locked if getattr(self._catalog_lock_state, "held", False)
                else self.uploads._search_rows_sync)
        rows = read(namespace, "", 1, file_id)
        row = rows[0] if rows else None
        return row if (row and row["status"] == "ready" and not row["private_required"]
                       and row["origin"] in {"upload", "generated"}) else None

    def _directory(self, namespace: str) -> Path:
        directory = self.reader.root / sanitize_user(namespace)
        if self.reader.root.is_symlink() or directory.is_symlink():
            raise ValueError("Unsafe source directory")
        return directory

    def is_current(self, owner_id: str, project_id: str, pin: dict) -> bool:
        """Bounded synchronous fence; no source bodies or Search cache required."""
        try:
            with self.store._lock:
                namespace, file_id = pin["namespace"], pin["file_id"]
                membership = self._scope(owner_id, project_id, namespace, file_id)
                if not _identifier(file_id) or membership is None or membership != pin["membership"]:
                    return False
                row = self._row(namespace, file_id)
                if row is None or _digest(row) != pin["catalog_sha256"]:
                    return False
                directory = self._directory(namespace)
                # Names originate from the canonical reader, but validate again
                # when decoding durable manifests rather than trusting SQLite.
                for name, expected in pin["pins"]:
                    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                        return False
                    actual = _presence_pin(directory / name)
                    if (list(actual) if actual is not None else None) != expected:
                        return False
                return bool(pin["pins"])
        except (KeyError, TypeError, ValueError, OSError):
            return False

    async def capture(self, principal, project_id: str | None, file_ids: list[str]) -> CapturedFileInputs:
        if (not isinstance(file_ids, list) or len(file_ids) > 3
                or any(not _identifier(value) for value in file_ids) or len(set(file_ids)) != len(file_ids)):
            raise AgentTaskError("Select at most three distinct Project files.")
        if not project_id:
            raise AgentTaskError("Selected files require a Project.")
        pins = []
        for file_id in sorted(file_ids):
            with self.store._lock:
                membership = self._scope(principal.user_id, project_id, principal.storage_namespace, file_id)
                if membership is None:
                    raise AgentTaskAccessError("Selected Project file is unavailable.")
                row = self._row(principal.storage_namespace, file_id)
            if row is None:
                raise AgentTaskAccessError("Selected Project file is unavailable.")
            artifact = "transcript" if row["kind"] in {"audio", "video"} else "text"
            if row["kind"] not in {"text", "image", "audio", "video"}:
                raise AgentTaskError("Selected Project file has no supported saved text.")
            try:
                text, disk_pins = await self.reader._disk(self.reader._read_file, principal.storage_namespace, row, artifact)
                prefix = await self.reader._disk(_prefix, text, 1500)
                if not prefix:
                    raise ValueError
                directory = self._directory(principal.storage_namespace)
                pin = {
                    "file_id": file_id, "filename": row["filename"][:1000], "artifact": artifact,
                    "chars": len(prefix), "total_chars": len(text), "truncated": len(prefix) < len(text),
                    "origin": row["origin"], "namespace": principal.storage_namespace,
                    "catalog_sha256": _digest(row), "text_sha256": _digest(text),
                    "membership": membership,
                    "pins": [[str(path.relative_to(directory)), list(value) if value is not None else None]
                             for path, value in disk_pins],
                }
            except Exception:  # noqa: BLE001 - parser failures must not expose source paths or text
                raise AgentTaskError("Selected Project file has no readable saved text within the source limits.") from None
            if not self.is_current(principal.user_id, project_id, pin):
                raise AgentTaskContextError(FILES_CHANGED)
            pins.append(pin)
        # No await follows the final fence; creation repeats it transactionally.
        if any(not self.is_current(principal.user_id, project_id, pin) for pin in pins):
            raise AgentTaskContextError(FILES_CHANGED)
        return CapturedFileInputs(principal.user_id, project_id, tuple(pins))

    async def resolve(self, task: dict, lease: dict) -> list[dict]:
        repository = self.store.agents
        await repository.validate_lease(**lease)
        with self.store._lock:
            rows = repository._file_input_pins(task["id"], task["user_id"])
        resolved = []
        for project_id, pin in rows:
            if not self.is_current(task["user_id"], project_id, pin):
                raise AgentTaskContextError(FILES_CHANGED)
            with self.store._lock:
                row = self._row(pin["namespace"], pin["file_id"])
            if row is None:
                raise AgentTaskContextError(FILES_CHANGED)
            try:
                text, _pins = await self.reader._disk(self.reader._read_file, pin["namespace"], row, pin["artifact"])
            except Exception:  # noqa: BLE001 - untrusted saved text has only content-free failure details
                raise AgentTaskContextError(FILES_CHANGED) from None
            if _digest(text) != pin["text_sha256"] or not self.is_current(task["user_id"], project_id, pin):
                raise AgentTaskContextError(FILES_CHANGED)
            resolved.append({**public_file(pin), "text": text[:pin["chars"]]})
        repository.fence_execution(**lease)
        return resolved


def get_agent_file_inputs(app) -> AgentFileInputs | None:
    if (getattr(app.state, "application_store", None) is None
            or getattr(app.state, "uploads_db", None) is None
            or not isinstance(getattr(getattr(app.state, "cfg", None), "raw", None), dict)):
        return None
    service = getattr(app.state, "agent_file_inputs", None)
    if (service is None or service.store is not app.state.application_store
            or service.uploads is not app.state.uploads_db):
        service = AgentFileInputs(app)
        app.state.agent_file_inputs = service
    return service
