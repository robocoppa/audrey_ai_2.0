"""Canonical, owner-scoped local sources for the rebuildable search projection.

No model, remote retrieval, embedding, or ordinary memory/history write occurs
here. File I/O is bounded and runs off-loop; authority is checked after it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anyio

from audrey.content_search import MAX_SOURCE_CHARS, SearchDocument
from audrey.kb.extract import EmptyExtractionError, extract_text, ocr_text_path
from audrey.kb.user_store import sanitize_user

MAX_FILE_BYTES = 8 * 1024 * 1024
_ARTIFACTS = {"transcript": "transcript.txt", "visual": "frames.txt", "summary": "summary.txt"}
_OWNER_SQL = (
    "SELECT 1 FROM app_users u WHERE u.user_id=? AND u.storage_namespace=? "
    "AND u.status='active' AND NOT EXISTS "
    "(SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)"
)


class _UnavailableSourceError(ValueError):
    pass


class _SourceSizeLimitError(_UnavailableSourceError):
    pass


@dataclass(frozen=True)
class SearchCoverageIssue:
    """Content-free coverage metadata, scoped to the same filters as results."""

    source_id: str
    kind: Literal["message", "file", "agent"]
    reason: Literal["size_limit", "text_unavailable"]
    updated_at: str
    project_ids: tuple[str, ...] = ()
    archived: bool = False


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _identifier(value: str) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 200
        and not any(char in value for char in ("/", "\\", "\x00", ":"))
        and value not in {".", ".."}
    )


def _presence_pin(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _pin(path: Path) -> tuple[int, int, int, int]:
    result = _presence_pin(path)
    if result is None:
        raise _UnavailableSourceError
    if result[2] > MAX_FILE_BYTES:
        raise _SourceSizeLimitError
    return result


def _validate_archive(path: Path) -> None:
    """Refuse small Office inputs that expand beyond the local read budget."""
    if path.suffix.lower() not in {".docx", ".xlsx"}:
        return
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 2000 or sum(member.file_size for member in members) > MAX_FILE_BYTES:
                raise _SourceSizeLimitError
    except zipfile.BadZipFile:
        raise _UnavailableSourceError from None


def _extract_search_text(path: Path) -> str:
    """Keep canonical PDF separators while bounding expanded streams/pages."""
    _validate_archive(path)
    if path.suffix.lower() != ".pdf":
        return extract_text(path)
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if not 0 < len(reader.pages) <= 2000:
            raise _SourceSizeLimitError
        parts, characters, expanded = [], 0, 0
        for page in reader.pages:
            contents = page.get_contents()
            if contents is not None:
                expanded += len(contents.get_data())
                if expanded > MAX_FILE_BYTES:
                    raise _SourceSizeLimitError
            text = page.extract_text() or ""
            characters += len(text) + 2
            if characters > MAX_SOURCE_CHARS:
                raise _SourceSizeLimitError
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts)
    except _SourceSizeLimitError:
        raise
    except Exception:  # noqa: BLE001 - untrusted PDF parser failures are content-free refusals
        # Parsers must not return a partial extraction or expose input details.
        raise _UnavailableSourceError from None


def _read_text(path: Path, expected: tuple[int, int, int, int]) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != expected:
            raise _UnavailableSourceError
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise _SourceSizeLimitError
    return data.decode("utf-8")


class ContentSourceReader:
    """Enumerate canonical pages and resolve exact, live source identities."""

    def __init__(self, app_store, uploads_db, upload_root: str | Path) -> None:
        self.store = app_store
        self.uploads = uploads_db
        self.root = Path(upload_root)
        self._limiter: anyio.CapacityLimiter | None = None
        self._proofs: OrderedDict[tuple, tuple] = OrderedDict()
        self._proof_lock = threading.RLock()

    def _remember(self, document: SearchDocument, namespace: str, proof: tuple) -> SearchDocument:
        key = (document.owner_id, document.source_id, document.revision)
        with self._proof_lock:
            self._proofs[key] = (namespace, proof)
            self._proofs.move_to_end(key)
            while len(self._proofs) > 4096:
                self._proofs.popitem(last=False)
        return document

    def is_current(self, document: SearchDocument) -> bool:
        """Final synchronous fence; never parse text or await another source."""
        with self._proof_lock:
            saved = self._proofs.get((document.owner_id, document.source_id, document.revision))
        if saved is None:
            return False
        namespace, proof = saved
        kind, expected = proof[:2]
        if kind == "file":
            file_id, projects, pins = proof[2:]
            if self._file_context(document.owner_id, namespace, file_id) != projects:
                return False
            rows = self.uploads._search_rows_sync(namespace, "", 1, file_id)
            if not rows or rows[0] != expected or not self._pins_current(pins):
                return False
            with self.store._lock:
                return self._owner(document.owner_id, namespace)
        with self.store._lock:
            if not self._owner(document.owner_id, namespace):
                return False
            if kind == "message":
                row = self.store._conn.execute(
                    "SELECT m.message_id,m.conversation_id,m.sequence_no,m.role,m.status,"
                    "m.updated_at,c.title,c.project_id,c.archived_at,c.private_required,"
                    "c.updated_at AS conversation_updated_at "
                    "FROM app_messages m JOIN app_conversations c "
                    "ON c.conversation_id=m.conversation_id AND c.user_id=m.user_id "
                    "WHERE m.user_id=? AND m.message_id=?",
                    (document.owner_id, expected["message_id"]),
                ).fetchone()
            else:
                row = self.store._conn.execute(
                    "SELECT task_id,objective,revision,updated_at FROM app_agent_tasks "
                    "WHERE user_id=? AND task_id=?",
                    (document.owner_id, expected["task_id"]),
                ).fetchone()
            return row is not None and dict(row) == expected

    async def _disk(self, function, *args):
        if self._limiter is None:
            self._limiter = anyio.CapacityLimiter(2)
        task = asyncio.create_task(anyio.to_thread.run_sync(function, *args, limiter=self._limiter))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep slots/SQLite alive until an already-started worker settles.
            try:
                await task
            finally:
                raise

    def _owner(self, owner_id: str, namespace: str) -> bool:
        return self.store._conn.execute(_OWNER_SQL, (owner_id, namespace)).fetchone() is not None

    def _rows(
        self,
        owner_id: str,
        namespace: str,
        kind: str,
        after: str,
        limit: int,
        source_id: str | None = None,
    ) -> list[dict]:
        with self.store._lock:
            if not self._owner(owner_id, namespace):
                return []
            if kind == "message":
                rows = self.store._conn.execute(
                    "SELECT m.message_id,m.conversation_id,m.sequence_no,m.role,m.status,"
                    "CASE WHEN length(m.content)<=? THEN m.content ELSE NULL END AS content,"
                    "m.updated_at,c.title,c.project_id,c.archived_at,c.private_required,"
                    "c.updated_at AS conversation_updated_at "
                    "FROM app_messages m JOIN app_conversations c "
                    "ON c.conversation_id=m.conversation_id AND c.user_id=m.user_id "
                    "WHERE m.user_id=? AND m.role IN ('user','assistant') "
                    "AND m.status IN ('completed','incomplete') AND m.message_id>? "
                    "AND (? IS NULL OR m.message_id=?) ORDER BY m.message_id LIMIT ?",
                    (MAX_SOURCE_CHARS, owner_id, after, source_id, source_id, limit),
                ).fetchall()
            else:
                rows = self.store._conn.execute(
                    "SELECT task_id,objective,result,revision,updated_at FROM app_agent_tasks "
                    "WHERE user_id=? AND result!='' AND task_id>? "
                    "AND (? IS NULL OR task_id=?) ORDER BY task_id LIMIT ?",
                    (owner_id, after, source_id, source_id, limit),
                ).fetchall()
            return [dict(row) for row in rows]

    def _document(
        self, owner_id: str, namespace: str, kind: str, row: dict
    ) -> SearchDocument | None:
        if kind == "message":
            text = row["content"]
            if text is None or not text.strip():
                return None
            document = SearchDocument(
                owner_id=owner_id,
                source_id=f"message:{row['message_id']}",
                kind=kind,
                revision=_digest(row),
                title=row["title"] or "Untitled conversation",
                text=text,
                updated_at=row["updated_at"],
                project_id=row["project_id"],
                archived=bool(row["archived_at"]),
                private_required=bool(row["private_required"]),
                target={
                    "kind": kind,
                    "conversation_id": row["conversation_id"],
                    "message_id": row["message_id"],
                    "sequence": row["sequence_no"],
                },
            )
            return self._remember(
                document,
                namespace,
                (kind, {key: value for key, value in row.items() if key != "content"}),
            )
        if len(row["result"]) > MAX_SOURCE_CHARS:
            return None
        document = SearchDocument(
            owner_id=owner_id,
            source_id=f"agent:{row['task_id']}:report",
            kind=kind,
            revision=_digest(row),
            title=row["objective"][:1000],
            text=row["result"],
            updated_at=row["updated_at"],
            target={"kind": kind, "task_id": row["task_id"], "origin": "generated"},
        )
        return self._remember(
            document,
            namespace,
            (kind, {key: value for key, value in row.items() if key != "result"}),
        )

    def _file_context(self, owner_id: str, namespace: str, file_id: str) -> list[str] | None:
        with self.store._lock:
            if not self._owner(owner_id, namespace):
                return None
            rows = self.store._conn.execute(
                "SELECT p.project_id FROM app_projects p JOIN app_project_files f "
                "ON f.project_id=p.project_id WHERE p.user_id=? AND f.file_id=? "
                "ORDER BY p.project_id LIMIT 201",
                (owner_id, file_id),
            ).fetchall()
        if len(rows) > 200:
            return None
        return [row[0] for row in rows]

    def _read_file(self, namespace: str, row: dict, artifact: str):
        file_id = row["file_id"]
        if not _identifier(file_id):
            raise _UnavailableSourceError
        directory = self.root / sanitize_user(namespace)
        if directory.is_symlink():
            raise _UnavailableSourceError
        source = directory / f"{file_id}{Path(row['filename']).suffix.lower()}"
        if artifact == "text":
            cached = ocr_text_path(source)
            if row["origin"] == "generated":
                paths = (source, cached)
                text_path = cached
            elif cached.exists():
                paths = (cached,)
                text_path = cached
            elif row["private_required"] or row["kind"] == "image":
                # Private indexing only reads the verified upload's saved text.
                raise _UnavailableSourceError
            else:
                paths = (source,)
                text_path = None
        else:
            paths = (directory / f"{file_id}.{_ARTIFACTS[artifact]}",)
            text_path = paths[0]
        pins = tuple((path, _pin(path)) for path in paths)
        if source not in paths:
            pins += ((source, _presence_pin(source)),)
        text = (
            _read_text(text_path, dict(pins)[text_path])
            if text_path is not None
            else _extract_search_text(source)
        )
        if not text.strip():
            raise _UnavailableSourceError
        if len(text) > MAX_SOURCE_CHARS:
            raise _SourceSizeLimitError
        if not self._pins_current(pins):
            raise _UnavailableSourceError
        return text, pins

    @staticmethod
    def _pins_current(pins) -> bool:
        try:
            return all(_presence_pin(path) == pin for path, pin in pins)
        except (OSError, _UnavailableSourceError):
            return False

    async def _file_document(
        self,
        owner_id: str,
        namespace: str,
        row: dict,
        artifact: str,
        failures: list[str] | None = None,
    ) -> SearchDocument | None:
        if row["status"] != "ready":
            return None
        expected = (
            ("text", "summary")
            if row["kind"] in {"text", "image"} or row["origin"] == "generated"
            else (
                ("transcript", "summary")
                if row["kind"] == "audio"
                else ("transcript", "visual", "summary")
                if row["kind"] == "video"
                else ()
            )
        )
        if artifact not in expected:
            return None
        projects = await self._disk(self._file_context, owner_id, namespace, row["file_id"])
        if projects is None:
            return None
        try:
            text, pins = await self._disk(self._read_file, namespace, row, artifact)
            fresh = await self.uploads.get_owned_search_row(user=namespace, file_id=row["file_id"])
            if fresh != row or not await self._disk(self._pins_current, pins):
                return None
            if projects != await self._disk(
                self._file_context, owner_id, namespace, row["file_id"]
            ):
                return None
        except (OSError, UnicodeError, EmptyExtractionError, _UnavailableSourceError) as error:
            if failures is not None:
                failures.append(
                    "size_limit" if isinstance(error, _SourceSizeLimitError) else "text_unavailable"
                )
            return None
        document = SearchDocument(
            owner_id=owner_id,
            source_id=f"file:{row['file_id']}:{artifact}",
            kind="file",
            revision=_digest(
                [row, projects, [list(pin) if pin else None for _, pin in pins], text]
            ),
            title=row["filename"][:1000],
            text=text,
            updated_at=row["completed_at"] or row["uploaded_at"],
            private_required=bool(row["private_required"]),
            availability="text_only"
            if row["source_freed_at"] or pins[-1][1] is None
            else "available",
            target={
                "kind": "file",
                "file_id": row["file_id"],
                "artifact": artifact,
                "origin": "generated" if artifact in {"visual", "summary"} else row["origin"],
                "project_ids": projects,
            },
        )
        return self._remember(document, namespace, ("file", row, row["file_id"], projects, pins))

    async def list_page(
        self,
        owner_id: str,
        storage_namespace: str,
        kind: Literal["message", "file", "agent"],
        after: str | None = None,
        limit: int = 50,
    ) -> tuple[list[SearchDocument], str | None, list[SearchCoverageIssue]]:
        if (
            kind not in {"message", "file", "agent"}
            or type(limit) is not int
            or not 1 <= limit <= 200
        ):
            raise ValueError("Invalid canonical search page.")
        if after is not None and (not isinstance(after, str) or len(after) > 256):
            raise ValueError("Invalid canonical search cursor.")
        warnings = []
        if kind != "file":
            rows = await self._disk(
                self._rows, owner_id, storage_namespace, kind, after or "", limit
            )
            documents = []
            for row in rows:
                document = self._document(owner_id, storage_namespace, kind, row)
                if document is not None:
                    documents.append(document)
                elif row.get("content") is None and kind == "message":
                    warnings.append(SearchCoverageIssue(
                        source_id=f"message:{row['message_id']}",
                        kind="message",
                        reason="size_limit",
                        updated_at=row["updated_at"],
                        project_ids=(row["project_id"],) if row["project_id"] else (),
                        archived=bool(row["archived_at"]),
                    ))
                elif kind == "agent" and len(row["result"]) > MAX_SOURCE_CHARS:
                    warnings.append(SearchCoverageIssue(
                        source_id=f"agent:{row['task_id']}:report",
                        kind="agent",
                        reason="size_limit",
                        updated_at=row["updated_at"],
                    ))
            key = "message_id" if kind == "message" else "task_id"
            return documents, rows[-1][key] if len(rows) == limit else None, warnings
        if await self._disk(self._file_context, owner_id, storage_namespace, "") is None:
            return [], None, []
        rows = await self.uploads.list_search_page(
            user=storage_namespace, after_file_id=after or "", limit=limit
        )
        documents = []
        for row in rows:
            # Pending/failed uploads have not produced searchable content yet;
            # they are not failed reads of completed saved content.
            if row["status"] != "ready":
                continue
            artifacts = (
                ("text", "summary")
                if row["kind"] in {"text", "image"} or row["origin"] == "generated"
                else (
                    ("transcript", "summary")
                    if row["kind"] == "audio"
                    else ("transcript", "visual", "summary")
                    if row["kind"] == "video"
                    else ()
                )
            )
            found = []
            failures = []
            for artifact in artifacts:
                document = await self._file_document(
                    owner_id, storage_namespace, row, artifact, failures
                )
                if document is not None:
                    found.append(document)
            documents.extend(found)
            # A missing optional sidecar does not make a readable file partial.
            # An existing oversized artifact does leave a real coverage gap,
            # even when a shorter summary of that file can still be searched.
            if failures and (not found or "size_limit" in failures):
                fresh = await self.uploads.get_owned_search_row(
                    user=storage_namespace, file_id=row["file_id"]
                )
                projects = await self._disk(
                    self._file_context, owner_id, storage_namespace, row["file_id"]
                )
                if fresh == row and projects is not None:
                    warnings.append(SearchCoverageIssue(
                        source_id=f"file:{row['file_id']}",
                        kind="file",
                        reason="size_limit" if "size_limit" in failures else "text_unavailable",
                        updated_at=row["completed_at"] or row["uploaded_at"],
                        project_ids=tuple(projects),
                    ))
        return documents, rows[-1]["file_id"] if len(rows) == limit else None, warnings

    async def resolve(
        self, owner_id: str, storage_namespace: str, source_id: str
    ) -> SearchDocument | None:
        if not isinstance(source_id, str) or len(source_id) > 256:
            return None
        parts = source_id.split(":")
        if len(parts) == 2 and parts[0] == "message" and _identifier(parts[1]):
            rows = await self._disk(
                self._rows, owner_id, storage_namespace, "message", "", 1, parts[1]
            )
            return self._document(owner_id, storage_namespace, "message", rows[0]) if rows else None
        if (
            len(parts) == 3
            and parts[0] == "agent"
            and parts[2] == "report"
            and _identifier(parts[1])
        ):
            rows = await self._disk(
                self._rows, owner_id, storage_namespace, "agent", "", 1, parts[1]
            )
            return self._document(owner_id, storage_namespace, "agent", rows[0]) if rows else None
        if (
            len(parts) == 3
            and parts[0] == "file"
            and _identifier(parts[1])
            and parts[2] in {"text", *_ARTIFACTS}
        ):
            row = await self.uploads.get_owned_search_row(user=storage_namespace, file_id=parts[1])
            return (
                await self._file_document(owner_id, storage_namespace, row, parts[2])
                if row
                else None
            )
        return None
