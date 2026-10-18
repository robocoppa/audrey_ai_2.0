"""Durable, bounded checkpoints and publication pointers for public reference indexes."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from audrey.references.irs_catalog import normalize_irs_url

_LIBRARY = re.compile(r"[a-z][a-z0-9-]{0,63}")
_RELEASE = re.compile(r"rel_[0-9a-f]{32}")
_SHA = re.compile(r"[0-9a-f]{64}")
_CATALOGS = frozenset({"current", "prior", "bulletins"})
_DOCUMENT_STATUSES = frozenset({"pending", "indexed", "no_text", "failed"})
_PUBLIC_METADATA = frozenset(
    {
        "library_id",
        "document_id",
        "revision_sha256",
        "source_url",
        "catalog_url",
        "product_number",
        "title",
        "catalog",
        "language",
        "document_kind",
        "catalog_edition_year",
        "revision_date",
        "posted_date",
        "tax_year_applicability",
        "applicable_tax_years",
        "original_available",
        "bytes",
        "downloaded_at",
        "status",
        "catalog_metadata_matches_original",
        "edition_status",
        "archived_edition",
        "applicability_basis",
        "verification_page",
        "verification_quote",
        "verification_sha256",
        "applicability_revision_sha256",
        "extraction_status",
        "limitations",
    }
)
_PAGE_SIZE = 64


class ReleaseError(Exception):
    """A derivative reference release could not be read or changed safely."""


class ReleaseStateError(ReleaseError):
    """A release operation does not satisfy its publication/checkpoint contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _library_id(value: str) -> str:
    if not isinstance(value, str) or not _LIBRARY.fullmatch(value):
        raise ReleaseStateError("Reference library ID is invalid")
    return value


def _catalog(value: str) -> str:
    if value not in _CATALOGS:
        raise ReleaseStateError("Reference catalog is invalid")
    return value


def _release_id(value: str) -> str:
    if not isinstance(value, str) or not _RELEASE.fullmatch(value):
        raise ReleaseStateError("Reference release ID is invalid")
    return value


def _document_id(value: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ReleaseStateError("Reference document ID is invalid")
    return value


def validate_extraction_gaps(values: Iterable[str]) -> tuple[str, ...]:
    """Validate a small, explicit operator approval without consuming unbounded input."""
    if isinstance(values, (str, bytes, Mapping)):
        raise ReleaseStateError("Reference extraction gaps must be explicit document IDs")
    try:
        iterator = iter(values)
    except TypeError:
        raise ReleaseStateError("Reference extraction gaps must be explicit document IDs") from None
    approved: list[str] = []
    for value in iterator:
        if len(approved) == 20:
            raise ReleaseStateError("Reference extraction gap approval exceeds 20 documents")
        identifier = _document_id(value)
        if identifier in approved:
            raise ReleaseStateError("Reference extraction gap approvals must not contain duplicates")
        approved.append(identifier)
    return tuple(approved)


def _extraction_gap_reason(error: str) -> str:
    """Only existing safe worker messages may enter public coverage metadata."""
    from audrey.references.extract import ERROR_MESSAGES

    if error in ERROR_MESSAGES.values() or re.fullmatch(
        r"PDF text extraction (?:exceeded its (?:CPU|[0-9]{1,5}-second) time limit|"
        r"worker (?:stopped \(signal [0-9]{1,3}\)|failed \(exit [0-9]{1,3}\)|could not run)|"
        r"returned invalid page data)",
        error,
    ):
        return error
    return "PDF text extraction is unavailable"


def reference_collection_name(library_id: str, embedding_identity: str | None = None) -> str:
    """One collection per library and immutable embedding space; older releases survive."""
    name = "audrey_reference_" + _library_id(library_id)
    if embedding_identity is not None:
        if not isinstance(embedding_identity, str) or not _SHA.fullmatch(embedding_identity):
            raise ReleaseStateError("Reference embedding identity is invalid")
        name += "_" + embedding_identity
    return name


def _bounded_json(value: Mapping[str, Any], *, maximum: int = 32768) -> str:
    if not isinstance(value, Mapping):
        raise ReleaseStateError("Reference metadata must be an object")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, UnicodeError):
        raise ReleaseStateError("Reference metadata is not JSON compatible") from None
    if len(encoded.encode("utf-8")) > maximum:
        raise ReleaseStateError("Reference metadata exceeds its size limit")
    return encoded


def _metadata(value: Mapping[str, Any], library_id: str, catalog: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseStateError("Reference document metadata must be an object")
    public = {key: value[key] for key in _PUBLIC_METADATA if key in value}
    identifier = _document_id(public.get("document_id"))
    sha = public.get("revision_sha256")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ReleaseStateError("Reference original revision checksum is invalid")
    try:
        source_url = normalize_irs_url(public.get("source_url"))
    except (TypeError, ValueError):
        raise ReleaseStateError("Reference source must be an official IRS PDF") from None
    if not urlsplit(source_url).path.lower().endswith(".pdf"):
        raise ReleaseStateError("Reference source must be an official IRS PDF")
    if identifier != hashlib.sha256(source_url.encode()).hexdigest():
        raise ReleaseStateError("Reference document identity does not match its source")
    if public.get("library_id", library_id) != library_id or public.get("catalog") != catalog:
        raise ReleaseStateError("Reference document does not match the release scope")
    if public.get("language", "en") != "en":
        raise ReleaseStateError("Reference releases contain English documents only")
    public["library_id"] = library_id
    public["source_url"] = source_url
    _bounded_json(public, maximum=16384)
    return public


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ReleaseStateError(f"Reference {name} is outside its supported range")
    return value


class ReleaseStore:
    """Keep derivative index state; publishing atomically swaps one scope pointer.

    Each operation opens its own SQLite connection, so service thread workers do
    not share a connection. Iterators fetch bounded pages and close the database
    before yielding, avoiding long read transactions during remote embedding work.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        if not self.root.is_absolute() or ".." in self.root.parts:
            raise ReleaseStateError("Reference index root must be an absolute server path")
        self._check_path(self.root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._path = self.root / "index.sqlite"
        self._write_lock = threading.RLock()
        for suffix in ("", "-wal", "-shm", "-journal"):
            self._check_path(Path(str(self._path) + suffix))
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS releases (
                    id TEXT PRIMARY KEY, library_id TEXT NOT NULL, catalog TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('building','published','failed')),
                    fingerprint TEXT NOT NULL, collection TEXT NOT NULL,
                    provenance_json TEXT NOT NULL, coverage_json TEXT NOT NULL,
                    expected_documents INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    published_at TEXT, error TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS release_scope ON releases(library_id,catalog,created_at);
                CREATE TABLE IF NOT EXISTS documents (
                    release_id TEXT NOT NULL REFERENCES releases(id),
                    document_id TEXT NOT NULL, metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','indexed','no_text','failed')),
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
                    PRIMARY KEY(release_id,document_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    release_id TEXT NOT NULL, document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL, page INTEGER NOT NULL,
                    offset INTEGER NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY(release_id,document_id,chunk_index),
                    FOREIGN KEY(release_id,document_id)
                        REFERENCES documents(release_id,document_id)
                );
                CREATE TABLE IF NOT EXISTS published (
                    library_id TEXT NOT NULL, catalog TEXT NOT NULL,
                    release_id TEXT NOT NULL REFERENCES releases(id),
                    PRIMARY KEY(library_id,catalog)
                );
            """)

    @staticmethod
    def _check_path(path: Path) -> None:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ReleaseStateError("Symbolic links are not allowed in reference index paths")

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._check_path(self._path)
        for suffix in ("-wal", "-shm", "-journal"):
            self._check_path(Path(str(self._path) + suffix))
        try:
            connection = sqlite3.connect(self._path, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                yield connection
        except sqlite3.Error:
            raise ReleaseError("Reference index state is unavailable") from None
        finally:
            if "connection" in locals():
                connection.close()

    @staticmethod
    def _require(connection: sqlite3.Connection, identifier: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM releases WHERE id=?", (_release_id(identifier),)
        ).fetchone()
        if row is None:
            raise ReleaseStateError("Reference release not found")
        return row

    @staticmethod
    def _describe(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        descriptor = dict(row)
        descriptor["provenance"] = json.loads(descriptor.pop("provenance_json"))
        descriptor["coverage"] = json.loads(descriptor.pop("coverage_json"))
        counts = {
            item["status"]: item["count"]
            for item in connection.execute(
                "SELECT status,COUNT(*) AS count FROM documents WHERE release_id=? GROUP BY status",
                (row["id"],),
            )
        }
        descriptor.update(
            {
                "completed_documents": counts.get("indexed", 0) + counts.get("no_text", 0),
                "indexed_documents": counts.get("indexed", 0),
                "no_text_documents": counts.get("no_text", 0),
                "failed_documents": counts.get("failed", 0),
                "pending_documents": counts.get("pending", 0),
                "chunk_count": connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE release_id=?",
                    (row["id"],),
                ).fetchone()[0],
            }
        )
        return descriptor

    def begin(
        self,
        library_id: str,
        catalog: str,
        *,
        fingerprint: str,
        provenance: dict[str, Any],
        coverage: dict[str, Any],
        documents: Iterable[dict[str, Any]],
        release_id: str | None = None,
    ) -> dict[str, Any]:
        _library_id(library_id)
        _catalog(catalog)
        if not isinstance(fingerprint, str) or not _SHA.fullmatch(fingerprint):
            raise ReleaseStateError("Reference release fingerprint must be a SHA-256")
        provenance_json = _bounded_json(provenance)
        if not coverage:
            raise ReleaseStateError("Reference coverage must be explicit")
        coverage = dict(coverage)
        if release_id is not None:
            _release_id(release_id)
        now = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if release_id is None:
                existing = connection.execute(
                    """SELECT * FROM releases WHERE library_id=? AND catalog=?
                       AND fingerprint=? AND status IN ('building','failed')
                       ORDER BY created_at DESC LIMIT 1""",
                    (library_id, catalog, fingerprint),
                ).fetchone()
            else:
                existing = self._require(connection, release_id)
                if (
                    existing["library_id"] != library_id
                    or existing["catalog"] != catalog
                    or existing["fingerprint"] != fingerprint
                    or existing["status"] == "published"
                ):
                    raise ReleaseStateError("Reference release cannot resume under this policy")
            identifier = existing["id"] if existing is not None else "rel_" + uuid.uuid4().hex
            if existing is not None and existing["provenance_json"] != provenance_json:
                raise ReleaseStateError("Reference release provenance changed")
            if existing is None:
                connection.execute(
                    """INSERT INTO releases(id,library_id,catalog,status,fingerprint,collection,
                       provenance_json,coverage_json,created_at,updated_at)
                       VALUES(?,?,?,'building',?,?,?,?,?,?)""",
                    (
                        identifier,
                        library_id,
                        catalog,
                        fingerprint,
                        reference_collection_name(library_id, provenance.get("embedding_identity")),
                        provenance_json,
                        "{}",
                        now,
                        now,
                    ),
                )
            connection.execute("CREATE TEMP TABLE incoming_documents(document_id TEXT PRIMARY KEY)")
            count = 0
            for supplied in documents:
                metadata = _metadata(supplied, library_id, catalog)
                document_id = metadata["document_id"]
                connection.execute("INSERT INTO incoming_documents VALUES(?)", (document_id,))
                if existing is None:
                    connection.execute(
                        """INSERT INTO documents(release_id,document_id,metadata_json,status,updated_at)
                           VALUES(?,?,?,'pending',?)""",
                        (identifier, document_id, _bounded_json(metadata), now),
                    )
                else:
                    planned = connection.execute(
                        "SELECT metadata_json FROM documents WHERE release_id=? AND document_id=?",
                        (identifier, document_id),
                    ).fetchone()
                    if planned is None:
                        raise ReleaseStateError("Reference release selected documents changed")
                    prior = json.loads(planned["metadata_json"])
                    if any(
                        prior[field] != metadata[field]
                        for field in ("source_url", "revision_sha256")
                    ):
                        raise ReleaseStateError("Reference release original revision changed")
                count += 1
            if count == 0:
                raise ReleaseStateError("Reference release requires selected documents")
            if existing is not None and count != existing["expected_documents"]:
                raise ReleaseStateError("Reference release selected documents changed")
            if "selected_documents" in coverage and coverage["selected_documents"] != count:
                raise ReleaseStateError("Reference coverage selected count does not match")
            coverage.update({"selected_documents": count, "complete_for_selected_documents": False})
            connection.execute(
                """UPDATE releases SET status='building',error='',expected_documents=?,
                   coverage_json=?,updated_at=? WHERE id=?""",
                (count, _bounded_json(coverage), now, identifier),
            )
            return self._describe(connection, self._require(connection, identifier))

    def release(self, release_id: str) -> dict[str, Any] | None:
        _release_id(release_id)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM releases WHERE id=?", (release_id,)).fetchone()
            return self._describe(connection, row) if row is not None else None

    def published(self, library_id: str, catalog: str = "current") -> dict[str, Any] | None:
        _library_id(library_id)
        _catalog(catalog)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT r.* FROM published p JOIN releases r ON r.id=p.release_id
                   WHERE p.library_id=? AND p.catalog=? AND r.status='published'""",
                (library_id, catalog),
            ).fetchone()
            return self._describe(connection, row) if row is not None else None

    def latest_build(self, library_id: str, catalog: str = "current") -> dict[str, Any] | None:
        _library_id(library_id)
        _catalog(catalog)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM releases WHERE library_id=? AND catalog=?
                   AND status IN ('building','failed') ORDER BY created_at DESC LIMIT 1""",
                (library_id, catalog),
            ).fetchone()
            return self._describe(connection, row) if row is not None else None

    @staticmethod
    def _document_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        metadata = json.loads(data.pop("metadata_json"))
        return {**metadata, **data, "metadata": metadata}

    def document(self, release_id: str, document_id: str) -> dict[str, Any] | None:
        _release_id(release_id)
        _document_id(document_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE release_id=? AND document_id=?",
                (release_id, document_id),
            ).fetchone()
            return self._document_row(row) if row is not None else None

    def documents(self, release_id: str, status: str | None = None) -> Iterator[dict[str, Any]]:
        _release_id(release_id)
        if status is not None and status not in _DOCUMENT_STATUSES:
            raise ReleaseStateError("Reference document checkpoint status is invalid")
        after = ""
        while True:
            with self._connect() as connection:
                rows = connection.execute(
                    """SELECT * FROM documents WHERE release_id=? AND document_id>?
                       AND (? IS NULL OR status=?) ORDER BY document_id LIMIT ?""",
                    (release_id, after, status, status, _PAGE_SIZE),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                after = row["document_id"]
                yield self._document_row(row)

    def put_document(
        self,
        release_id: str,
        document_id: str,
        *,
        metadata: dict[str, Any],
        page_count: int,
        status: str,
        chunks: Iterable[dict[str, Any]],
        error: str = "",
    ) -> None:
        _release_id(release_id)
        _document_id(document_id)
        if status not in {"indexed", "no_text", "failed"}:
            raise ReleaseStateError("Reference document completion status is invalid")
        _integer(page_count, "page count", 0 if status == "failed" else 1, 10000)
        if not isinstance(error, str) or len(error) > 1000 or (status == "failed" and not error):
            raise ReleaseStateError("Reference document failure must have a bounded reason")
        if status != "failed" and error:
            raise ReleaseStateError("Successful reference checkpoint cannot have an error")
        now = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            release = self._require(connection, release_id)
            if release["status"] != "building":
                raise ReleaseStateError("Only building reference releases accept checkpoints")
            planned = connection.execute(
                "SELECT metadata_json FROM documents WHERE release_id=? AND document_id=?",
                (release_id, document_id),
            ).fetchone()
            if planned is None:
                raise ReleaseStateError("Reference document was not selected for this release")
            metadata = _metadata(metadata, release["library_id"], release["catalog"])
            previous = json.loads(planned["metadata_json"])
            if metadata["document_id"] != document_id or any(
                previous[field] != metadata[field] for field in ("source_url", "revision_sha256")
            ):
                raise ReleaseStateError("Reference checkpoint changed its original identity")
            connection.execute(
                "DELETE FROM chunks WHERE release_id=? AND document_id=?",
                (release_id, document_id),
            )
            count = 0
            for chunk in chunks:
                if status != "indexed" or not isinstance(chunk, Mapping):
                    raise ReleaseStateError("Only indexed documents may contain chunks")
                if chunk.get("document_id", document_id) != document_id:
                    raise ReleaseStateError("Reference chunk belongs to a different document")
                if _integer(chunk.get("chunk_index"), "chunk index", 0, 1000000) != count:
                    raise ReleaseStateError("Reference chunks must have contiguous indexes")
                page = _integer(chunk.get("page"), "chunk page", 1, page_count)
                offset = _integer(chunk.get("offset"), "chunk offset", 0, 200000)
                text = chunk.get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > 8000:
                    raise ReleaseStateError("Reference chunk text must be nonempty and bounded")
                if offset + len(text) > 200000:
                    raise ReleaseStateError("Reference chunk exceeds its bounded page")
                connection.execute(
                    "INSERT INTO chunks VALUES(?,?,?,?,?,?)",
                    (release_id, document_id, count, page, offset, text),
                )
                count += 1
            if status == "indexed" and not count:
                raise ReleaseStateError("Indexed reference document requires chunks")
            connection.execute(
                """UPDATE documents SET metadata_json=?,status=?,page_count=?,chunk_count=?,
                   error=?,updated_at=? WHERE release_id=? AND document_id=?""",
                (
                    _bounded_json(metadata),
                    status,
                    page_count,
                    count,
                    error,
                    now,
                    release_id,
                    document_id,
                ),
            )
            connection.execute("UPDATE releases SET updated_at=? WHERE id=?", (now, release_id))

    @staticmethod
    def _chunk_row(row: sqlite3.Row) -> dict[str, Any]:
        chunk = dict(row)
        chunk["char_end"] = chunk["offset"] + len(chunk["text"])
        return chunk

    def chunk(self, release_id: str, document_id: str, chunk_index: int) -> dict[str, Any] | None:
        _release_id(release_id)
        _document_id(document_id)
        _integer(chunk_index, "chunk index", 0, 1000000)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chunks WHERE release_id=? AND document_id=? AND chunk_index=?",
                (release_id, document_id, chunk_index),
            ).fetchone()
            return self._chunk_row(row) if row is not None else None

    def chunks(self, release_id: str, document_id: str | None = None) -> Iterator[dict[str, Any]]:
        _release_id(release_id)
        if document_id is not None:
            _document_id(document_id)
        after_document, after_index = "", -1
        while True:
            with self._connect() as connection:
                rows = connection.execute(
                    """SELECT * FROM chunks WHERE release_id=?
                       AND (? IS NULL OR document_id=?)
                       AND (document_id>? OR (document_id=? AND chunk_index>?))
                       ORDER BY document_id,chunk_index LIMIT ?""",
                    (
                        release_id,
                        document_id,
                        document_id,
                        after_document,
                        after_document,
                        after_index,
                        _PAGE_SIZE,
                    ),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                after_document, after_index = row["document_id"], row["chunk_index"]
                yield self._chunk_row(row)

    def publish(
        self,
        release_id: str,
        *,
        verified_chunks: int,
        allowed_extraction_gaps: Iterable[str] = (),
    ) -> dict[str, Any]:
        _release_id(release_id)
        _integer(verified_chunks, "verified chunk count", 0, 100000000)
        approved = validate_extraction_gaps(allowed_extraction_gaps)
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            release = self._require(connection, release_id)
            if release["status"] != "building":
                raise ReleaseStateError("Only a building reference release can be published")
            descriptor = self._describe(connection, release)
            gaps = []
            for document_id in sorted(approved):
                row = connection.execute(
                    "SELECT metadata_json,status,error FROM documents "
                    "WHERE release_id=? AND document_id=?",
                    (release_id, document_id),
                ).fetchone()
                if row is None or row["status"] != "failed":
                    raise ReleaseStateError(
                        "Reference extraction gap approval must name a selected failed document"
                    )
                metadata = json.loads(row["metadata_json"])
                gaps.append(
                    {
                        "document_id": document_id,
                        "product_number": metadata.get("product_number", ""),
                        "title": metadata.get("title", ""),
                        "source_url": metadata["source_url"],
                        "revision_sha256": metadata["revision_sha256"],
                        "reason": _extraction_gap_reason(row["error"]),
                        "original_available": metadata.get("original_available") is True,
                    }
                )
            if (
                descriptor["completed_documents"] + len(gaps) != descriptor["expected_documents"]
                or descriptor["failed_documents"] != len(gaps)
                or descriptor["pending_documents"]
                or descriptor["chunk_count"] != verified_chunks
            ):
                raise ReleaseStateError(
                    "Reference release coverage or remote chunk verification is incomplete"
                )
            coverage = descriptor["coverage"]
            coverage["complete_for_selected_documents"] = not gaps
            if gaps:
                coverage.update({"content_search_gaps": gaps, "gap_count": len(gaps)})
            now = _now()
            connection.execute(
                """UPDATE releases SET status='published',published_at=?,updated_at=?,
                   coverage_json=? WHERE id=?""",
                (now, now, _bounded_json(coverage), release_id),
            )
            connection.execute(
                """INSERT INTO published VALUES(?,?,?) ON CONFLICT(library_id,catalog)
                   DO UPDATE SET release_id=excluded.release_id""",
                (release["library_id"], release["catalog"], release_id),
            )
            return self._describe(connection, self._require(connection, release_id))

    def fail(self, release_id: str, error: str) -> None:
        _release_id(release_id)
        if not isinstance(error, str) or not error or len(error) > 1000:
            raise ReleaseStateError("Reference release failure requires a bounded reason")
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            release = self._require(connection, release_id)
            if release["status"] != "building":
                raise ReleaseStateError("Only a building reference release can fail")
            connection.execute(
                "UPDATE releases SET status='failed',error=?,updated_at=? WHERE id=?",
                (error, _now(), release_id),
            )
