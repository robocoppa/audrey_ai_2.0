"""Bounded reads of actual, ready generated files owned by one saved Agent.

The output receipt grants no access to other user files. Source bytes, not OCR
sidecars or model claims, supply the text. Office extraction checks text only.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

import anyio

from audrey.agent_file_inputs import _page_arguments, _page_result
from audrey.agent_files import _authority, agent_files_available
from audrey.kb.user_store import sanitize_user

MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_TEXT_CHARS = 500000
_FORMATS = frozenset({".txt", ".md", ".csv", ".json", ".docx", ".xlsx"})
_UNAVAILABLE = {"error": "output_unavailable", "detail": "This Agent's saved output is unavailable."}


class _UnavailableOutputError(ValueError):
    pass


def _arguments(arguments):
    if not isinstance(arguments, dict):
        return None
    revision = arguments.get("expected_revision")
    if "expected_revision" in arguments and (
        not isinstance(revision, str) or re.fullmatch("[0-9a-f]{64}", revision) is None
    ):
        return None
    page = _page_arguments({key: value for key, value in arguments.items() if key != "expected_revision"})
    return (*page, revision) if page is not None else None


def _pin(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _source(root: Path, namespace: str, file_id: str, suffix: str):
    """Open every directory and the source without following symlinks."""
    descriptors = []
    try:
        root = root.absolute()
        if ".." in root.parts or sanitize_user(namespace) != namespace:
            raise _UnavailableOutputError
        descriptor = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(descriptor)
        for component in (*root.parts[1:], namespace):
            descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            descriptors.append(descriptor)
        descriptor = os.open(file_id + suffix, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        descriptors.append(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_OUTPUT_BYTES:
            raise _UnavailableOutputError
        yield descriptor, _pin(info)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _physical_pin(root, namespace, file_id, suffix):
    with _source(root, namespace, file_id, suffix) as (_, pin):
        return pin


def _office_text(data: bytes, suffix: str) -> str:
    """Parse bounded local Office bytes without opening paths or following links."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 2000 or sum(member.file_size for member in members) > 8 * 1024 * 1024:
            raise _UnavailableOutputError
    if suffix == ".docx":
        from docx import Document
        from docx.text.paragraph import Paragraph

        document = Document(io.BytesIO(data))
        parts = []
        for index, block in enumerate(document.iter_inner_content()):
            if index >= 10000:
                raise _UnavailableOutputError
            if isinstance(block, Paragraph):
                parts.append(block.text)
            else:
                if len(block.rows) > 1000 or len(block.columns) > 32:
                    raise _UnavailableOutputError
                parts.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
        return "\n".join(parts)
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=False, keep_links=False)
    try:
        if len(workbook.worksheets) > 8:
            raise _UnavailableOutputError
        parts, total_rows = [], 0
        for sheet in workbook.worksheets:
            if (sheet.max_row or 0) > 1000 or (sheet.max_column or 0) > 32:
                raise _UnavailableOutputError
            parts.append("# Sheet: " + sheet.title)
            for row in sheet.iter_rows(values_only=True):
                total_rows += 1
                if total_rows > 1000:
                    raise _UnavailableOutputError
                parts.append("\t".join("" if value is None else str(value) for value in row))
        return "\n".join(parts)
    finally:
        workbook.close()


def _read_source(root, namespace, file_id, suffix):
    with _source(root, namespace, file_id, suffix) as (descriptor, pin):
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            data = stream.read(MAX_OUTPUT_BYTES + 1)
        if len(data) != pin[2] or len(data) > MAX_OUTPUT_BYTES or _pin(os.fstat(descriptor)) != pin:
            raise _UnavailableOutputError
    try:
        text = _office_text(data, suffix) if suffix in {".docx", ".xlsx"} else data.decode("utf-8")
    except Exception:  # noqa: BLE001 - untrusted parser errors can contain source bodies
        raise _UnavailableOutputError from None
    if not text.strip() or len(text) > MAX_TEXT_CHARS or "\x00" in text:
        raise _UnavailableOutputError
    if _physical_pin(root, namespace, file_id, suffix) != pin:
        raise _UnavailableOutputError
    return text, hashlib.sha256(data).hexdigest(), pin


def _snapshot_locked(app, principal, task, file_id, scope_guard):
    """Caller holds app then upload locks and the selected-input lock bridge."""
    store, uploads = app.state.application_store, app.state.uploads_db
    _authority(app, principal, task, scope_guard)
    receipt = store._conn.execute(
        "SELECT * FROM app_agent_outputs WHERE file_id=? AND task_id=? AND user_id=? AND status='ready'",
        (file_id, task["id"], principal.user_id),
    ).fetchone()
    row = uploads._conn.execute(
        "SELECT * FROM uploads WHERE file_id=? AND user=? AND status='ready' "
        "AND origin='generated' AND private_required=0 AND NOT EXISTS "
        "(SELECT 1 FROM file_deletions d WHERE d.file_id=uploads.file_id AND d.user=uploads.user)",
        (file_id, principal.storage_namespace),
    ).fetchone()
    if receipt is None or row is None or receipt["filename"] != row["filename"]:
        raise _UnavailableOutputError
    suffix = Path(row["filename"]).suffix.lower()
    if suffix not in _FORMATS:
        raise _UnavailableOutputError
    return dict(receipt), dict(row), suffix


def _snapshot(app, principal, task, file_id, scope_guard):
    """Hold app then upload locks, using the existing selected-input lock bridge."""
    store, uploads = app.state.application_store, app.state.uploads_db
    inputs = store.agents._file_input_service
    with (store._lock, uploads._lock,
          inputs.catalog_lock_already_held() if inputs else nullcontext()):
        return _snapshot_locked(app, principal, task, file_id, scope_guard)


async def _disk(app, function, *args):
    """Bound concurrent local parsing and settle started workers on cancellation."""
    limiter = getattr(app.state, "agent_output_read_limiter", None)
    if limiter is None:
        limiter = anyio.CapacityLimiter(2)
        app.state.agent_output_read_limiter = limiter
    worker = asyncio.create_task(anyio.to_thread.run_sync(function, *args, limiter=limiter))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancelled:
        try:
            await worker
        finally:
            # A failing worker must not replace cancellation with a recoverable
            # tool error after the caller has stopped waiting for this read.
            raise cancelled


def _result(file_id, filename, text, offset, limit, revision, representation):
    pin = {"file_id": file_id, "filename": filename, "artifact": "text", "origin": "generated"}

    def page(count):
        return {**_page_result(pin, text, offset, count), "revision": revision, "representation": representation}

    low, high = 0, min(limit, len(text) - offset)
    while low < high:
        count = (low + high + 1) // 2
        if len(json.dumps(page(count), ensure_ascii=False)) <= 5000:
            low = count
        else:
            high = count - 1
    return page(low)


async def read_agent_output(app, principal, task, arguments, *, scope_guard=None):
    """Read this task's ready output; no cache, inferred source scope or HTTP path."""
    if not agent_files_available(app):
        return dict(_UNAVAILABLE)
    with app.state.application_store._lock:
        _authority(app, principal, task, scope_guard)
    parsed = _arguments(arguments)
    if parsed is None:
        return {"error": "arguments_invalid", "detail": (
            "Use file_id, optional Unicode offset (0–500000), limit (1–6000) "
            "and expected_revision from a previous read."
        )}
    file_id, offset, limit, expected_revision = parsed
    try:
        snapshot = _snapshot(app, principal, task, file_id, scope_guard)
        receipt, _row, suffix = snapshot
        root = Path((app.state.cfg.raw.get("kb") or {}).get("upload_root", "/data/uploads"))
        text, revision, pin = await _disk(app, _read_source, root, principal.storage_namespace, file_id, suffix)
        # This final synchronous boundary checks both canonical data and disk
        # identity before any text can leave the live attempt's authority.
        store, uploads = app.state.application_store, app.state.uploads_db
        inputs = store.agents._file_input_service
        with (store._lock, uploads._lock,
              inputs.catalog_lock_already_held() if inputs else nullcontext()):
            if _snapshot_locked(app, principal, task, file_id, scope_guard) != snapshot:
                raise _UnavailableOutputError
            if _physical_pin(root, principal.storage_namespace, file_id, suffix) != pin:
                raise _UnavailableOutputError
            if expected_revision is not None and expected_revision != revision:
                return {"error": "output_changed", "detail": "The saved output changed; begin a fresh read at offset zero."}
            if offset > len(text):
                return {"error": "offset_invalid", "detail": "The offset is past the end of this output's text."}
            return _result(file_id, receipt["filename"], text, offset, limit, revision,
                           "extracted_text" if suffix in {".docx", ".xlsx"} else "plain_text")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        # Parser errors may carry document content or paths. Never expose them.
        with app.state.application_store._lock:
            _authority(app, principal, task, scope_guard)
        return dict(_UNAVAILABLE)


__all__ = ["read_agent_output"]
