"""Read named public reference archives without modifying them or searching uploads.

This first library surface provides catalog lookup and a bounded PDF text-page
reader. It does not infer tax-year applicability or create a vector index.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audrey.references.irs_catalog import CatalogParseError, normalize_irs_url

MAX_PDF_BYTES = 256 * 1024 * 1024
MAX_PAGE_CHARS = 200_000
MAX_READER_SECONDS = 20
_LIBRARY_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")
_DOCUMENT_ID = re.compile(r"[0-9a-f]{64}")
_CATALOGS = frozenset({"current", "prior", "bulletins"})
_PUBLIC_COLUMNS = """
    d.source_url, d.error,
    substr(o.product_number,1,128) AS product_number,
    substr(o.title,1,500) AS title, o.catalog, o.language, o.document_kind,
    substr(o.revision_date,1,100) AS revision_date,
    substr(o.posted_date,1,100) AS posted_date, o.catalog_url,
    reference_edition_year(o.catalog,o.revision_date,o.product_number,o.title) AS catalog_edition_year,
    r.sha256, r.bytes, r.relative_path, r.downloaded_at,
    substr(r.metadata_json,1,8192) AS metadata_json
"""


class ReferenceLibraryError(Exception):
    """A reference operation could not return a trustworthy result."""


class ReferenceNotFound(ReferenceLibraryError):  # noqa: N818 — shared API exception name
    """The requested library document or retained original is unavailable."""


class ReferenceUnavailable(ReferenceLibraryError):  # noqa: N818 — shared API exception name
    """The configured archive cannot currently be read."""


class ReferenceQueryError(ReferenceLibraryError, ValueError):
    """The requested filter, page or bound is unsupported."""


class ReferenceReadError(ReferenceLibraryError):
    """A bounded reference PDF inspection could not extract its text."""


@dataclass(frozen=True)
class _Library:
    id: str
    name: str
    description: str
    root: Path


def document_id(source_url: str) -> str:
    """A stable public document identity, independent of its retained revision."""
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()


def _edition_year(catalog: str, revision_date: str, product: str, title: str) -> int | None:
    """Catalog edition identity only, never tax-year applicability."""
    if catalog == "bulletins":
        found = set(re.findall(
            r"\b(?:Internal\s+Revenue\s+Bulletin:?\s*|IRB\s*)([12]\d{3})-\d+\b",
            f"{product} {title}", re.I,
        ))
    else:
        found = set(re.findall(r"\b([12]\d{3})\b", revision_date))
    return int(next(iter(found))) if len(found) == 1 else None


def _safe_path(root: Path, relative: str, *, required: bool = True) -> Path:
    path_part = Path(relative)
    if path_part.is_absolute() or ".." in path_part.parts or not path_part.parts:
        raise ReferenceNotFound("Reference document not found")
    path = root / path_part
    current = Path(root.anchor)
    for part in root.parts[1:] + path_part.parts:
        current /= part
        if current.is_symlink():
            raise ReferenceNotFound("Reference document not found")
    if not path.resolve(strict=False).is_relative_to(root.resolve(strict=False)):
        raise ReferenceNotFound("Reference document not found")
    if required and not path.is_file():
        raise ReferenceNotFound("Reference document not found")
    return path


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ReferenceQueryError(f"{name} must be between {minimum} and {maximum}")
    return value


class ReferenceLibraryRegistry:
    """Server-owned roots; callers select only registered IDs and bounded filters."""

    def __init__(self, config: Mapping[str, Any] | None):
        config = config or {}
        enabled = config.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ReferenceQueryError("Reference library enabled must be boolean")
        self.enabled = enabled
        self.asof_year = _integer(config.get("asof_year", 2026), "asof_year", 2000, 2100)
        self.history_years = _integer(config.get("history_years", 5), "history_years", 1, 5)
        self.first_history_year = self.asof_year - self.history_years + 1
        entries = config.get("libraries", [])
        if not isinstance(entries, list) or len(entries) > 50:
            raise ReferenceQueryError("Reference libraries must be a bounded list")
        self._libraries: dict[str, _Library] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ReferenceQueryError("Reference library configuration must be an object")
            identifier = entry.get("id", "")
            name = entry.get("name", "")
            description = entry.get("description", "")
            root_text = entry.get("root", "")
            if not isinstance(identifier, str) or not _LIBRARY_ID.fullmatch(identifier):
                raise ReferenceQueryError("Reference library ID is invalid")
            if identifier in self._libraries:
                raise ReferenceQueryError("Reference library IDs must be unique")
            if not isinstance(name, str) or not name.strip() or len(name) > 100:
                raise ReferenceQueryError("Reference library name is required and bounded")
            if not isinstance(description, str) or len(description) > 1000:
                raise ReferenceQueryError("Reference library description is too long")
            if not isinstance(root_text, str) or not root_text or len(root_text) > 4096:
                raise ReferenceQueryError("Reference library root is required")
            root = Path(root_text)
            if not root.is_absolute() or ".." in root.parts:
                raise ReferenceQueryError("Reference library root must be an absolute server path")
            self._libraries[identifier] = _Library(identifier, name.strip(), description, root)

    def _library(self, library_id: str) -> _Library:
        if not self.enabled or library_id not in self._libraries:
            raise ReferenceNotFound("Reference library not found")
        return self._libraries[library_id]

    @contextlib.contextmanager
    def _connection(self, library: _Library) -> Iterator[sqlite3.Connection]:
        """Read committed archive rows without creating SQLite WAL/SHM files.

        Existing readable WAL+SHM permit crash-retained committed rows. A clean
        checkpointed database uses immutable reads. An active collector's lock
        and unfinished rollback recovery cause an explicit unavailable result.
        """
        connection = None
        lock = None
        try:
            database = _safe_path(library.root, "catalogs/archive.sqlite")
            lock_path = _safe_path(library.root, ".collect.lock", required=False)
            if lock_path.exists():
                lock = lock_path.open("rb")
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ReferenceUnavailable("Reference archive update is in progress") from None
            journal = _safe_path(library.root, "catalogs/archive.sqlite-journal", required=False)
            if journal.exists() and journal.stat().st_size:
                raise ReferenceUnavailable("Reference archive requires rollback recovery")
            wal = _safe_path(library.root, "catalogs/archive.sqlite-wal", required=False)
            if wal.exists() and wal.stat().st_size:
                shared_memory = _safe_path(library.root, "catalogs/archive.sqlite-shm")
                if not wal.is_file() or not shared_memory.is_file():
                    raise ReferenceUnavailable("Reference archive WAL state is unavailable")
                if os.access(shared_memory, os.W_OK):
                    raise ReferenceUnavailable("Reference archive WAL requires a read-only mount")
                uri = database.as_uri() + "?mode=ro"
            else:
                uri = database.as_uri() + "?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True, timeout=2)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.create_function("reference_doc_id", 1, document_id, deterministic=True)
            connection.create_function("reference_edition_year", 4, _edition_year, deterministic=True)
            yield connection
        except ReferenceNotFound:
            raise ReferenceUnavailable("Reference archive is unavailable") from None
        except (sqlite3.Error, OSError):
            raise ReferenceUnavailable("Reference archive is unavailable") from None
        finally:
            if connection is not None:
                connection.close()
            if lock is not None:
                lock.close()

    def list_libraries(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": "disabled", "items": []}
        items = []
        for library in self._libraries.values():
            try:
                counts = {}
                with self._connection(library) as connection:
                    for catalog in ("current", "prior", "bulletins"):
                        sql, params = self._select(catalog=catalog)
                        count = connection.execute(
                            f"SELECT COUNT(*) AS catalogued,COUNT(sha256) AS retained FROM ({sql})",  # noqa: S608
                            params,
                        ).fetchone()
                        counts[catalog] = dict(count)
                status = "available"
            except ReferenceUnavailable:
                status = "unavailable"
                counts = {}
            items.append({
                "id": library.id, "name": library.name, "description": library.description,
                "status": status, "search_status": "not_indexed",
                "tax_year_applicability": "unknown", "default_catalog": "current",
                "historical_edition_window": [self.first_history_year, self.asof_year],
                "accessible_catalog_counts": counts,
            })
        return {"enabled": True, "status": "ready", "items": items}

    @staticmethod
    def _metadata(library: _Library, row: sqlite3.Row) -> dict[str, Any]:
        original_available = False
        if row["relative_path"]:
            try:
                path = _safe_path(library.root, row["relative_path"])
                original_available = path.stat().st_size == row["bytes"]
            except (ReferenceNotFound, OSError):
                pass
        try:
            source_url = normalize_irs_url(row["source_url"])
            catalog_url = normalize_irs_url(row["catalog_url"])
            archived = json.loads(row["metadata_json"]) if row["metadata_json"] else None
        except (CatalogParseError, ValueError, TypeError):
            raise ReferenceUnavailable("Reference archive metadata is invalid") from None
        if archived is not None and not isinstance(archived, dict):
            raise ReferenceUnavailable("Reference archive metadata is invalid")
        metadata_matches = None if archived is None else all(
            str(archived.get(field, "")) == row[field]
            for field in ("product_number", "title", "revision_date", "posted_date")
        )
        return {
            "library_id": library.id, "document_id": document_id(source_url),
            "revision_sha256": row["sha256"],
            "source_url": source_url, "catalog_url": catalog_url,
            "product_number": row["product_number"], "title": row["title"],
            "catalog": row["catalog"], "language": row["language"],
            "catalog_edition_year": row["catalog_edition_year"],
            "document_kind": row["document_kind"],
            "revision_date": row["revision_date"], "posted_date": row["posted_date"],
            "tax_year_applicability": "unknown", "applicable_tax_years": None,
            "original_available": original_available, "bytes": row["bytes"] or 0,
            "downloaded_at": row["downloaded_at"],
            "catalog_metadata_matches_original": metadata_matches,
            "edition_status": (
                "not_downloaded" if archived is None else
                "catalog_and_original_match" if metadata_matches else
                "catalog_changed_since_download"
            ),
            "status": "failed" if row["error"] else ("downloaded" if row["sha256"] else "pending"),
            "archived_edition": {
                "title": str(archived.get("title", ""))[:500],
                "product_number": str(archived.get("product_number", ""))[:128],
                "revision_date": str(archived.get("revision_date", ""))[:100],
                "posted_date": str(archived.get("posted_date", ""))[:100],
                "tax_year_applicability": "unknown",
            } if archived else None,
        }

    def _select(self, *, catalog: str | None = None, revision: bool = False) -> tuple[str, list[Any]]:
        scope = "current" if catalog is None else catalog
        if scope not in _CATALOGS:
            raise ReferenceQueryError("Unsupported reference catalog")
        params: list[Any] = [scope]
        revision_join = (
            "JOIN revisions r ON r.id=d.latest_revision_id AND r.source_url=d.source_url AND r.sha256=?"
            if revision else
            "LEFT JOIN revisions r ON r.id=d.latest_revision_id AND r.source_url=d.source_url"
        )
        # Only constant SQL fragments are interpolated; every caller value binds.
        sql = (
            f"SELECT {_PUBLIC_COLUMNS} FROM documents d {revision_join} "  # noqa: S608
            "JOIN observations o ON o.id=("
            "SELECT candidate.id FROM observations candidate WHERE candidate.source_url=d.source_url "
            "AND candidate.catalog=? AND EXISTS("
            "SELECT 1 FROM catalog_pages p WHERE p.catalog=candidate.catalog "
            "AND p.url=candidate.catalog_url AND p.snapshot_sha256=candidate.snapshot_sha256"
            ") ORDER BY candidate.observed_at DESC,candidate.id DESC LIMIT 1"
            ") WHERE o.language='en'"
        )
        if scope != "current":
            sql += " AND reference_edition_year(o.catalog,o.revision_date,o.product_number,o.title) BETWEEN ? AND ?"
            params.extend([self.first_history_year, self.asof_year])
        return sql, params

    def catalog(
        self, library_id: str, *, product_number: str | None = None, query: str | None = None,
        catalog: str | None = None, limit: int = 20, offset: int = 0, year: int | None = None,
    ) -> dict[str, Any]:
        library = self._library(library_id)
        _integer(limit, "limit", 1, 20)
        _integer(offset, "offset", 0, 100_000)
        if year is not None:
            raise ReferenceQueryError("Tax-year filtering is unavailable; applicability is unknown")
        if catalog is not None and catalog not in _CATALOGS:
            raise ReferenceQueryError("Unsupported reference catalog")
        for name, value in (("product_number", product_number), ("query", query)):
            maximum = 200 if name == "query" else 128
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > maximum
            ):
                raise ReferenceQueryError(f"{name} must contain between 1 and {maximum} characters")
        sql, params = self._select(catalog=catalog)
        if product_number is not None:
            sql += " AND o.product_number = ? COLLATE NOCASE"
            params.append(product_number.strip())
        if query is not None:
            sql += " AND instr(lower(o.title),lower(?))>0"
            params.append(query.strip())
        sql += " ORDER BY o.product_number COLLATE NOCASE,d.source_url LIMIT ? OFFSET ?"
        params.extend([limit + 1, offset])
        with self._connection(library) as connection:
            rows = connection.execute(sql, params).fetchall()
            items = [self._metadata(library, row) for row in rows[:limit]]
        has_more = len(rows) > limit
        return {
            "library_id": library_id, "items": items, "offset": offset,
            "returned": len(items), "has_more": has_more,
            "next_offset": offset + len(items) if has_more else None,
            "tax_year_applicability": "unknown", "search_status": "catalog_lookup_only",
            "catalog": "current" if catalog is None else catalog,
            "historical_edition_window": [self.first_history_year, self.asof_year],
        }

    def iter_index_documents(self, library_id: str, *, catalog: str = "current"):
        """Page through the permitted manifest; asset paths remain internal.

        No database transaction stays open while the caller embeds a document.
        The same archived-edition guard used by exact reads applies here.
        """
        library = self._library(library_id)
        after = ""
        while True:
            sql, params = self._select(catalog=catalog)
            sql += " AND d.source_url>? ORDER BY d.source_url LIMIT 64"
            with self._connection(library) as connection:
                rows = connection.execute(sql, [*params, after]).fetchall()
            if not rows:
                return
            for row in rows:
                after = row["source_url"]
                try:
                    self._check_archived_scope(row, catalog)
                except ReferenceNotFound:
                    continue
                metadata = self._metadata(library, row)
                metadata["_asset"] = {
                    "root": str(library.root), "relative_path": row["relative_path"],
                    "bytes": row["bytes"], "sha256": row["sha256"],
                }
                yield metadata

    def _lookup(
        self, library: _Library, identifier: str, revision_sha256: str | None = None,
        catalog: str | None = None,
    ) -> sqlite3.Row:
        if not isinstance(identifier, str) or not _DOCUMENT_ID.fullmatch(identifier):
            raise ReferenceNotFound("Reference document not found")
        if revision_sha256 is not None and (
            not isinstance(revision_sha256, str) or not _DOCUMENT_ID.fullmatch(revision_sha256)
        ):
            raise ReferenceNotFound("Reference document not found")
        sql, params = self._select(catalog=catalog, revision=revision_sha256 is not None)
        if revision_sha256 is not None:
            params.insert(0, revision_sha256)
        sql += " AND reference_doc_id(d.source_url)=? LIMIT 1"
        params.append(identifier)
        with self._connection(library) as connection:
            row = connection.execute(sql, params).fetchone()
        if row is None:
            raise ReferenceNotFound("Reference document not found")
        self._check_archived_scope(row, catalog)
        return row

    def _check_archived_scope(self, row: sqlite3.Row, catalog: str | None) -> None:
        scope = "current" if catalog is None else catalog
        if scope != "current" and row["sha256"]:
            try:
                archived = json.loads(row["metadata_json"])
                if not isinstance(archived, dict):
                    raise ValueError
                archived_year = _edition_year(
                    scope, str(archived.get("revision_date", "")),
                    str(archived.get("product_number", "")), str(archived.get("title", "")),
                )
            except (TypeError, ValueError):
                archived_year = None
            if archived_year is None or not self.first_history_year <= archived_year <= self.asof_year:
                raise ReferenceNotFound("Reference document not found")

    def lookup(
        self, library_id: str, document_id: str, *, catalog: str | None = None,
    ) -> dict[str, Any]:
        library = self._library(library_id)
        return self._metadata(library, self._lookup(library, document_id, catalog=catalog))

    def read(
        self, library_id: str, document_id: str, *, page: int = 1, offset: int = 0,
        limit: int = 4000, revision_sha256: str | None = None, catalog: str | None = None,
    ) -> dict[str, Any]:
        library = self._library(library_id)
        _integer(page, "page", 1, 10_000)
        _integer(offset, "offset", 0, MAX_PAGE_CHARS)
        _integer(limit, "limit", 1, 4000)
        row = self._lookup(library, document_id, revision_sha256, catalog=catalog)
        if row["error"]:
            raise ReferenceUnavailable(
                "Reference edition update failed; retained original is not available for reading"
            )
        if not row["sha256"] or not row["relative_path"]:
            raise ReferenceNotFound("Reference document not found")
        _safe_path(library.root, row["relative_path"])
        if row["bytes"] > MAX_PDF_BYTES or row["bytes"] <= 0:
            raise ReferenceReadError("Reference PDF exceeds the 256 MiB reader limit")
        payload = {
            "root": str(library.root), "relative_path": row["relative_path"],
            "sha256": row["sha256"], "bytes": row["bytes"],
            "page": page, "offset": offset, "limit": limit,
        }
        with tempfile.TemporaryFile() as output:
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "audrey.references.library", "--_reader"],
                    input=json.dumps(payload).encode("utf-8"), stdout=output,
                    stderr=subprocess.DEVNULL, timeout=MAX_READER_SECONDS, check=False,
                )
            except subprocess.TimeoutExpired:
                raise ReferenceReadError("Reference page inspection exceeded its 20-second limit") from None
            output.seek(0)
            encoded = output.read(32_769)
        if len(encoded) > 32_768 or completed.returncode != 0:
            raise ReferenceReadError("Reference PDF could not be read within its resource limits")
        try:
            result = json.loads(encoded)
        except (UnicodeError, ValueError):
            raise ReferenceReadError("Reference page inspection returned an invalid result") from None
        if not isinstance(result, dict):
            raise ReferenceReadError("Reference page inspection returned an invalid result")
        error = result.pop("error", None)
        if error == "not_found":
            raise ReferenceNotFound("Reference document not found")
        if error == "query":
            raise ReferenceQueryError(result["message"])
        if error:
            raise ReferenceReadError(result.get("message", "Reference PDF text is unavailable"))
        metadata = self._metadata(library, row)
        return {
            **metadata, **result,
            "limitations": [
                "Text layer only; no OCR, vision or verification of visual layout.",
                "Reading one page does not establish review of the full document.",
                "Tax-year applicability is unknown and must be established from the source.",
            ],
        }


def _reader_worker(payload: dict[str, Any]) -> dict[str, Any]:
    from pypdf import PdfReader

    root = Path(payload["root"])
    path = _safe_path(root, payload["relative_path"])
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        if os.fstat(source.fileno()).st_size != payload["bytes"]:
            return {"error": "not_found"}
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != payload["sha256"]:
            return {"error": "not_found"}
        source.seek(0)
        reader = PdfReader(source, strict=False)
        if reader.is_encrypted:
            return {"error": "read", "message": "Encrypted reference PDF text is unavailable"}
        total_pages = len(reader.pages)
        if total_pages > 10_000:
            return {"error": "read", "message": "Reference PDF exceeds the 10,000-page reader limit"}
        if payload["page"] > total_pages:
            return {"error": "query", "message": "Requested page is outside this reference PDF"}
        text = reader.pages[payload["page"] - 1].extract_text() or ""
    total_chars = len(text)
    text = text[:MAX_PAGE_CHARS]
    offset = payload["offset"]
    if offset > len(text):
        return {"error": "query", "message": "Requested offset is outside the available page text"}
    selected = text[offset:offset + payload["limit"]]
    next_offset = offset + len(selected) if offset + len(selected) < len(text) else None
    return {
        "page": payload["page"], "total_pages": total_pages,
        "text": selected, "offset": offset, "next_offset": next_offset,
        "has_more": next_offset is not None, "total_chars": total_chars,
        "available_chars": len(text), "page_text_truncated": total_chars > len(text),
        "text_layer_available": bool(text),
        "extraction_status": "text" if text else "no_text_layer",
    }


def _reader_main() -> int:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024,) * 2)
        encoded = sys.stdin.buffer.read(8193)
        if len(encoded) > 8192:
            return 2
        payload = json.loads(encoded)
        try:
            result = _reader_worker(payload)
        except ReferenceNotFound:
            result = {"error": "not_found"}
        except Exception:  # noqa: BLE001 — child returns a bounded safe error for corrupt PDFs
            result = {"error": "read", "message": "Reference PDF text could not be extracted"}
        sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        return 0
    except Exception:  # noqa: BLE001 — bounded worker never prints parser internals
        return 2


if __name__ == "__main__":
    raise SystemExit(_reader_main() if sys.argv[1:] == ["--_reader"] else 2)
