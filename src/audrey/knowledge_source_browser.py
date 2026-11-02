"""Bounded canonical metadata browsing for Project knowledge references.

Listing never extracts or reads source bodies, and does not depend on Search's
projection. Previewing still goes through the revocable canonical source reader.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import secrets
import stat
from pathlib import Path

from audrey.content_search import MAX_SOURCE_CHARS
from audrey.content_sources import _identifier
from audrey.kb.extract import ocr_text_path
from audrey.kb.user_store import sanitize_user

_KEY = secrets.token_bytes(32)
_ARTIFACTS = {
    "text": (None, "Text"), "summary": ("summary.txt", "Summary"),
    "transcript": ("transcript.txt", "Transcript"), "visual": ("frames.txt", "Visual notes"),
}


class SourceBrowserConflictError(ValueError):
    """The owner, Project or source changed during a bounded metadata read."""


def valid_source_id(value: str) -> bool:
    parts = value.split(":")
    return bool(
        len(parts) in {2, 3} and _identifier(parts[1])
        and ((parts[0] == "message" and len(parts) == 2)
             or (parts[0] == "agent" and len(parts) == 3 and parts[2] == "report")
             or (parts[0] == "file" and len(parts) == 3 and parts[2] in _ARTIFACTS))
    )


def _binding(principal, project, kind):
    return [principal.user_id, principal.storage_namespace, project.project_id,
            project.updated_at, project.private_required, kind]


def decode_cursor(cursor, principal, project, kind):
    if cursor is None:
        return ""
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        signature, payload = raw[:32], raw[32:]
        if not hmac.compare_digest(signature, hmac.digest(_KEY, payload, "sha256")):
            raise ValueError
        binding, after = json.loads(payload)
        if (binding != _binding(principal, project, kind) or not isinstance(after, str)
                or len(after) > 256 or not valid_source_id(after) or not after.startswith(kind + ":")):
            raise ValueError
        return after
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("Source cursor is invalid. Refresh the source list.") from None


def encode_cursor(after, principal, project, kind):
    payload = json.dumps([_binding(principal, project, kind), after], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(hmac.digest(_KEY, payload, "sha256") + payload).decode().rstrip("=")


def _pin(path):
    try:
        info = path.lstat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) if stat.S_ISREG(info.st_mode) else None
    except FileNotFoundError:
        return None


class KnowledgeSourceBrowser:
    def __init__(self, store, reader):
        self.store, self.reader = store, reader

    def project_current(self, principal, project):
        with self.store._lock:
            return self.store._conn.execute(
                "SELECT 1 FROM app_projects p JOIN app_users u ON u.user_id=p.user_id "
                "WHERE p.user_id=? AND p.project_id=? AND p.updated_at=? AND p.private_required=? "
                "AND u.storage_namespace=? AND u.status='active' AND NOT EXISTS "
                "(SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)",
                (principal.user_id, project.project_id, project.updated_at,
                 int(project.private_required), principal.storage_namespace),
            ).fetchone() is not None

    async def page(self, principal, project, kind, after, limit):
        return await asyncio.to_thread(self.snapshot, principal, project, kind, after, limit)

    def snapshot(self, principal, project, kind, after, limit):
        if not self.project_current(principal, project):
            raise SourceBrowserConflictError("Project sources changed. Refresh the source list.")
        if kind == "file":
            candidates = self._files(principal, project)
        else:
            with self.store._lock:
                if kind == "message":
                    rows = self.store._conn.execute(
                        "SELECT m.message_id,m.role,m.sequence_no,c.title,c.private_required,m.updated_at,"
                        "c.updated_at AS conversation_updated "
                        "FROM app_messages m JOIN app_conversations c ON c.conversation_id=m.conversation_id "
                        "AND c.user_id=m.user_id WHERE m.user_id=? AND c.project_id=? "
                        "AND m.role IN ('user','assistant') AND m.status IN ('completed','incomplete') "
                        "AND length(m.content)<=? AND length(trim(m.content))>0 AND c.private_required<=? "
                        "AND m.message_id>? ORDER BY m.message_id LIMIT ?",
                        (principal.user_id, project.project_id, MAX_SOURCE_CHARS,
                         int(project.private_required), after.split(":")[1] if after else "", limit + 1),
                    ).fetchall()
                    candidates = [(f"message:{row['message_id']}",
                                   f"{row['title'] or 'Untitled conversation'} · {row['role'].capitalize()} message {row['sequence_no']}",
                                   bool(row["private_required"]), tuple(row)) for row in rows]
                else:
                    rows = self.store._conn.execute(
                        "SELECT task_id,CASE WHEN title!='' THEN title ELSE substr(objective,1,100) END AS title,"
                        "revision,updated_at FROM app_agent_tasks "
                        "WHERE user_id=? AND length(result)<=? AND length(trim(result))>0 AND task_id>? "
                        "ORDER BY task_id LIMIT ?",
                        (principal.user_id, MAX_SOURCE_CHARS, after.split(":")[1] if after else "", limit + 1),
                    ).fetchall()
                    candidates = [(f"agent:{row['task_id']}:report", row["title"],
                                   False, tuple(row)) for row in rows]
        values = sorted((value for value in candidates if value[0] > after and valid_source_id(value[0])),
                        key=lambda value: value[0])[:limit + 1]
        if not self.project_current(principal, project):
            raise SourceBrowserConflictError("Project sources changed. Refresh the source list.")
        return values

    def _files(self, principal, project):
        uploads = getattr(self.reader, "uploads", None)
        if uploads is None:
            return []
        with self.store._lock:
            ids = [row[0] for row in self.store._conn.execute(
                "SELECT file_id FROM app_project_files WHERE project_id=? ORDER BY file_id LIMIT 21",
                (project.project_id,),
            ).fetchall()]
        if len(ids) > 20:
            raise SourceBrowserConflictError("Project source allowance exceeded.")
        directory = self.reader.root / sanitize_user(principal.storage_namespace)
        if directory.is_symlink():
            return []
        candidates = []
        for file_id in ids:
            if not _identifier(file_id):
                continue
            rows = uploads._search_rows_sync(principal.storage_namespace, "", 1, file_id)
            if not rows:
                continue
            row = rows[0]
            if row["status"] != "ready" or (row["private_required"] and not project.private_required):
                continue
            source = directory / f"{file_id}{Path(row['filename']).suffix.lower()}"
            source_pin, cached_pin = _pin(source), _pin(ocr_text_path(source))
            artifacts = (("text", "summary") if row["kind"] in {"text", "image"} or row["origin"] == "generated"
                         else ("transcript", "summary") if row["kind"] == "audio"
                         else ("transcript", "visual", "summary") if row["kind"] == "video" else ())
            for artifact in artifacts:
                suffix, label = _ARTIFACTS[artifact]
                if artifact == "text":
                    present = (bool(source_pin and cached_pin) if row["origin"] == "generated" else
                               bool(cached_pin or (source_pin and not row["private_required"] and row["kind"] != "image")))
                    pins = (source_pin, cached_pin)
                else:
                    pins = (_pin(directory / f"{file_id}.{suffix}"),)
                    present = bool(pins[0])
                if present:
                    candidates.append((f"file:{file_id}:{artifact}", f"{row['filename'][:1000]} · {label}",
                                       bool(row["private_required"]), (row, pins)))
        return candidates
