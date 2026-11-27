"""Owner-scoped, local fingerprints for a watched Project filename."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audrey.app_state.projects import PROJECT_MAX_FILES
from audrey.app_state.watches import (
    WATCH_SOURCE_MAX_BYTES,
    WatchAccessError,
    WatchConflictError,
    WatchError,
    normalize_watch_source_filename,
)
from audrey.content_sources import ContentSourceReader, _presence_pin
from audrey.identity import Principal

_TEXT_SUFFIXES = frozenset({".txt", ".md", ".csv", ".tsv", ".json"})
_UNAVAILABLE = "The watched Project file is unavailable."
_NOT_READY = "The latest matching Project file must be ready and contain supported UTF-8 text."
_CHANGED = "The watched Project file changed while checking."
_SIZE_LIMIT = "The watched Project file exceeds the 64,000-byte text limit."

SourceGuard = Callable[[], AbstractContextManager[Callable[[str | None], bool]]]
FilePin = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CapturedWatchFile:
    file_id: str
    digest: str
    source_guard: SourceGuard


@dataclass(frozen=True, slots=True)
class _ProjectFile:
    added_at: str
    attachment_origin: str
    row: dict


class ProjectFileTextWatchSource:
    def __init__(self, app: Any) -> None:
        self.store = app.state.application_store
        self.uploads = getattr(app.state, "uploads_db", None)
        self.reader = ContentSourceReader(
            self.store,
            self.uploads,
            (app.state.cfg.raw.get("kb") or {}).get("upload_root", "/data/uploads"),
        )

    async def validate(self, principal: Principal, project_id: str | None, source_filename: str) -> None:
        """Validate the latest exact-name attachment without reading its text."""
        await self.reader._disk(self._capture_selection, principal, project_id, source_filename)

    async def capture(
        self, principal: Principal, project_id: str | None, source_filename: str,
    ) -> CapturedWatchFile:
        selected, source, pin, digest = await self.reader._disk(
            self._capture_file, principal, project_id, source_filename,
        )

        @contextmanager
        def source_guard() -> Iterator[Callable[[str | None], bool]]:
            # The repository holds the application lock before entering this
            # context and keeps both locks through the publication commit.
            with self.uploads._lock:
                def check_source(previous_file_id: str | None) -> bool:
                    matches = self._matches_locked(principal, project_id, source_filename)
                    if not matches or matches[0] != selected:
                        raise WatchConflictError(_CHANGED)
                    if self._source_pin(principal, selected) != (source, pin):
                        raise WatchConflictError(_CHANGED)
                    if previous_file_id == selected.row["file_id"]:
                        return True
                    previous = next((match for match in matches if match.row["file_id"] == previous_file_id), None)
                    if previous is None:
                        return False
                    try:
                        self._source_pin(principal, previous)
                    except WatchError:
                        return False
                    return True

                yield check_source

        return CapturedWatchFile(selected.row["file_id"], digest, source_guard)

    def _capture_file(
        self, principal: Principal, project_id: str | None, source_filename: str,
    ) -> tuple[_ProjectFile, Path, FilePin, str]:
        selected, source, pin = self._capture_selection(principal, project_id, source_filename)
        return selected, source, pin, self._fingerprint(source, pin)

    def _capture_selection(
        self, principal: Principal, project_id: str | None, source_filename: str,
    ) -> tuple[_ProjectFile, Path, FilePin]:
        source_filename = normalize_watch_source_filename(source_filename)
        if Path(source_filename).suffix.lower() not in _TEXT_SUFFIXES:
            raise WatchError("Choose a supported text filename in the Project.")
        if self.uploads is None:
            raise WatchAccessError(_UNAVAILABLE)
        with self.store._lock, self.uploads._lock:
            matches = self._matches_locked(principal, project_id, source_filename)
            if not matches:
                raise WatchAccessError(_UNAVAILABLE)
            selected = matches[0]
            source, pin = self._source_pin(principal, selected)
            return selected, source, pin

    def _matches_locked(
        self, principal: Principal, project_id: str | None, filename: str,
    ) -> list[_ProjectFile]:
        members = self.store._conn.execute(
            "SELECT f.file_id,f.added_at,f.origin FROM app_users u JOIN app_projects p ON p.user_id=u.user_id "
            "JOIN app_project_files f ON f.project_id=p.project_id "
            "WHERE u.user_id=? AND u.storage_namespace=? AND u.status='active' "
            "AND p.project_id=? AND p.private_required=0 "
            "AND NOT EXISTS (SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id) "
            "AND NOT EXISTS (SELECT 1 FROM user_group_memberships g WHERE g.user_id=u.user_id AND g.group_id='bots') "
            "ORDER BY f.added_at DESC,f.file_id LIMIT ?",
            (principal.user_id, principal.storage_namespace, project_id, PROJECT_MAX_FILES + 1),
        ).fetchall()
        if not members:
            return []
        if len(members) > PROJECT_MAX_FILES:
            raise WatchError("The Project file inventory exceeds the watch allowance.")

        # Include non-ready rows in the ordering. A pending or deleting newest
        # copy must fail this check instead of falling back to older text.
        rows = self.uploads._conn.execute(
            "SELECT u.file_id,u.filename,u.kind,u.status,u.bytes,u.uploaded_at,u.completed_at,"
            "u.source_freed_at,u.private_required,u.origin,EXISTS (SELECT 1 FROM file_deletions d "
            "WHERE d.file_id=u.file_id AND d.user=u.user) AS tombstoned FROM uploads u "
            "WHERE u.user=? AND u.filename=? AND u.file_id IN (SELECT value FROM json_each(?))",
            (principal.storage_namespace, filename, json.dumps([member["file_id"] for member in members])),
        ).fetchall()
        by_id = {row["file_id"]: dict(row) for row in rows}
        return [
            _ProjectFile(member["added_at"], member["origin"], by_id[member["file_id"]])
            for member in members
            if member["file_id"] in by_id
        ]

    def _source_pin(self, principal: Principal, selected: _ProjectFile) -> tuple[Path, FilePin]:
        row = selected.row
        if row["private_required"] or row["tombstoned"] or row["status"] == "deleting":
            raise WatchAccessError(_UNAVAILABLE)
        if (
            row["status"] != "ready" or row["kind"] != "text"
            or row["origin"] not in {"upload", "generated"} or row["origin"] != selected.attachment_origin
            or Path(row["filename"]).suffix.lower() not in _TEXT_SUFFIXES
        ):
            raise WatchError(_NOT_READY)
        try:
            if row["source_freed_at"] or self.reader.root.is_symlink():
                raise WatchAccessError(_UNAVAILABLE)
            source, _, _ = self.reader._file_paths(principal.storage_namespace, row, "text")
            pin = _presence_pin(source)
        except (ValueError, OSError):
            raise WatchAccessError(_UNAVAILABLE) from None
        if pin is None:
            raise WatchAccessError(_UNAVAILABLE)
        if pin[2] > WATCH_SOURCE_MAX_BYTES:
            raise WatchError(_SIZE_LIMIT)
        if pin[2] == 0 or pin[2] != row["bytes"]:
            raise WatchError(_NOT_READY)
        return source, pin

    @staticmethod
    def _fingerprint(source: Path, expected: FilePin) -> str:
        def verified(info: os.stat_result) -> bool:
            return stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == expected

        try:
            descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                if not verified(os.fstat(stream.fileno())):
                    raise WatchConflictError(_CHANGED)
                data = stream.read(WATCH_SOURCE_MAX_BYTES + 1)
                if len(data) != expected[2] or not verified(os.fstat(stream.fileno())):
                    raise WatchConflictError(_CHANGED)
            value = data.decode("utf-8-sig")
        except OSError:
            raise WatchAccessError(_UNAVAILABLE) from None
        except UnicodeDecodeError:
            raise WatchError(_NOT_READY) from None
        if not value.strip() or "\x00" in value:
            raise WatchError(_NOT_READY)
        # Ignore the UTF-8 BOM and line ending changes when comparing text.
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
