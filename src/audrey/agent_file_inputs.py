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

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state.agent_projects import _prefix
from audrey.app_state.agents import AgentTaskAccessError, AgentTaskContextError, AgentTaskError
from audrey.content_sources import ContentSourceReader, _identifier, _presence_pin
from audrey.kb.user_store import sanitize_user
from audrey.model_catalog import _apply_policy, _apply_profile, _model_allows, configured_models
from audrey.processing_policy import current_processing_policy
from audrey.user_data_visibility import remote_personal_reads_blocked

FILES_CHANGED = "Selected files changed or are unavailable. Create a new agent to use their current text."
_PUBLIC = ("file_id", "filename", "artifact", "chars", "total_chars", "truncated", "origin")
_PAGE_RESULT_CHARS = 5000


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CapturedFileInputs:
    owner_id: str
    project_id: str
    pins: tuple[dict, ...]


def public_file(pin: dict) -> dict:
    return {key: pin[key] for key in _PUBLIC}


def _page_arguments(arguments) -> tuple[str, int, int] | None:
    if (not isinstance(arguments, dict) or set(arguments) - {"file_id", "offset", "limit"}
            or not _identifier(arguments.get("file_id"))):
        return None
    offset, limit = arguments.get("offset", 0), arguments.get("limit", 3000)
    if type(offset) is not int or not 0 <= offset <= 500000 or type(limit) is not int or not 1 <= limit <= 6000:
        return None
    return arguments["file_id"], offset, limit


def _page_result(pin: dict, text: str, offset: int, limit: int) -> dict:
    """Keep the JSON intact and report the actual Unicode slice delivered.

    ReAct's generic truncation cannot preserve offsets in an escaped string.
    Fit the entire payload here, before it reaches that generic boundary.
    """
    result = {key: pin[key] for key in ("file_id", "filename", "artifact", "origin")}
    result["filename"] = result["filename"][:240]
    result.update(text="", start=offset, end=offset, total_chars=len(text), next_offset=None, has_more=False)

    def candidate(count):
        end = min(len(text), offset + count)
        more = end < len(text)
        return {**result, "text": text[offset:end], "end": end,
                "next_offset": end if more else None, "has_more": more}

    low, high = 0, min(limit, len(text) - offset)
    while low < high:
        count = (low + high + 1) // 2
        if len(json.dumps(candidate(count), ensure_ascii=False)) <= _PAGE_RESULT_CHARS:
            low = count
        else:
            high = count - 1
    return candidate(low)


class AgentFileInputs:
    def __init__(self, app) -> None:
        self.app = app
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

    async def _read_pin(self, task: dict, project_id: str, pin: dict, *, scope_guard=None) -> str:
        """Reread one frozen source; other selections need only their stat fences."""
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
        if scope_guard is not None:
            scope_guard()
        if _digest(text) != pin["text_sha256"] or not self.is_current(task["user_id"], project_id, pin):
            raise AgentTaskContextError(FILES_CHANGED)
        return text

    async def resolve(self, task: dict, lease: dict) -> list[dict]:
        repository = self.store.agents
        await repository.validate_lease(**lease)
        with self.store._lock:
            rows = repository._file_input_pins(task["id"], task["user_id"])
        resolved = []
        for project_id, pin in rows:
            text = await self._read_pin(task, project_id, pin)
            resolved.append({**public_file(pin), "text": text[:pin["chars"]]})
        repository.fence_execution(**lease)
        return resolved

    def _read_authority(self, principal, task, scope_guard=None):
        """Live synchronous read authority; source text never enters this fence."""
        from audrey.agent_executor import agent_settings

        if scope_guard is not None:
            scope_guard()
        if current_processing_policy() is not None or not agent_settings(self.app).enabled:
            raise AgentDispatchStoppedError("Agents are unavailable in this processing mode.")
        lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
        try:
            self.store.agents.fence_execution(**lease)
            current = self.store._get_principal_sync(principal.user_id)
            record = self.store.agents._lease(**lease)
        except AgentTaskError:
            raise AgentDispatchStoppedError("Agent access, inputs or execution limits changed.") from None
        if (current is None or current.status != "active" or current.role == "bot" or "bots" in current.groups
                or current.storage_namespace != principal.storage_namespace or record["user_id"] != principal.user_id
                or task.get("user_id") != principal.user_id
                or self.store.agents._internal(record)["current_step"] != "research"
                or remote_personal_reads_blocked(principal.storage_namespace)):
            raise AgentDispatchStoppedError("Selected file reading is unavailable.")
        model = next((item for item in configured_models(self.app.state.cfg) if item.id == "local"), None)
        policy = next((item for item in self.store._list_model_access_policies_sync() if item.model_id == "local"), None)
        profile = next((item for item in self.store._list_model_publication_profiles_sync() if item.model_id == "local"), None)
        if model is None:
            raise AgentDispatchStoppedError("Local model access is unavailable.")
        model = _apply_profile(_apply_policy(model, policy), profile)
        if not model.enabled or not _model_allows(model, current):
            raise AgentDispatchStoppedError("Local model access is unavailable.")

    async def read_selected(self, principal, task: dict, arguments: dict, *, scope_guard=None) -> dict:
        """Read one bounded page from an explicitly selected, unchanged file."""
        with self.store._lock:
            self._read_authority(principal, task, scope_guard)
            request = _page_arguments(arguments)
            if request is None:
                return {"error": "invalid_arguments", "detail": "Supply a selected file ID, a character offset from 0 to 500000 and a page limit from 1 to 6000."}
            file_id, offset, limit = request
            selected = next(((project, pin) for project, pin in self.store.agents._file_input_pins(task["id"], task["user_id"])
                             if pin["file_id"] == file_id), None)
            if selected is None:
                return {"error": "file_not_selected", "detail": "Read only a file explicitly selected for this agent."}
        project_id, pin = selected
        try:
            text = await self._read_pin(task, project_id, pin, scope_guard=scope_guard)
        except AgentTaskError:
            raise AgentDispatchStoppedError(FILES_CHANGED) from None
        with self.store._lock:
            self._read_authority(principal, task, scope_guard)
            if offset > len(text):
                return {"error": "offset_out_of_range", "detail": "The offset exceeds this file's character count."}
            return _page_result(pin, text, offset, limit)


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
    from audrey.agent_knowledge_files import get_agent_knowledge_files

    get_agent_knowledge_files(app, inputs=service)
    return service
