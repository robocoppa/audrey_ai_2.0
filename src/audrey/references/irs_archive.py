"""Durable public IRS originals and catalog observations, independent of search indexes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audrey.references.irs_catalog import normalize_irs_url

_ELIGIBLE_KINDS = frozenset({"form", "instruction", "publication", "bulletin", "other"})
_ENTRY_FIELDS = (
    "catalog", "source_url", "product_number", "title", "revision_date",
    "posted_date", "language", "document_kind", "catalog_url",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _official_url(url: str) -> str:
    return normalize_irs_url(url)


def _slug(value: str, *, fallback: str = "document") -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-")
    return cleaned[:80] or fallback


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


class ArchiveStore:
    """Append-only inventory of public catalog metadata and PDF revisions.

    The caller serializes archive operations with the CLI's process lock.
    Downloads must be staged beneath root/.staging. Original bytes and HTML
    snapshots are never replaced. SQLite commits follow their fsync.
    """

    def __init__(self, root: Path):
        root = Path(root).expanduser()
        if root.is_symlink():
            raise ValueError("The archive root must not be a symbolic link.")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)
        for directory in ("catalogs", "catalogs/snapshots", "originals", ".staging", "reports"):
            self._directory(directory)
        db_path = self._path("catalogs/archive.sqlite")
        for name in ("archive.sqlite", "archive.sqlite-wal", "archive.sqlite-shm",
                     "archive.sqlite-journal"):
            self._path(f"catalogs/{name}")
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                source_url TEXT PRIMARY KEY,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                latest_revision_id INTEGER REFERENCES revisions(id),
                last_checked_at TEXT,
                error TEXT,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_failed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS catalog_pages (
                catalog TEXT NOT NULL,
                url TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                next_urls_json TEXT NOT NULL,
                total_rows INTEGER,
                PRIMARY KEY(catalog,url)
            );
            CREATE TABLE IF NOT EXISTS page_snapshots (
                catalog TEXT NOT NULL,
                url TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                PRIMARY KEY(catalog,url,snapshot_sha256)
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY,
                source_url TEXT NOT NULL REFERENCES documents(source_url),
                catalog TEXT NOT NULL,
                catalog_url TEXT NOT NULL,
                product_number TEXT NOT NULL,
                title TEXT NOT NULL,
                revision_date TEXT NOT NULL,
                posted_date TEXT NOT NULL,
                language TEXT NOT NULL,
                document_kind TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                UNIQUE(source_url,catalog,catalog_url,product_number,title,
                       revision_date,posted_date,language,document_kind,snapshot_sha256)
            );
            CREATE INDEX IF NOT EXISTS observations_source ON observations(source_url,id);
            CREATE TABLE IF NOT EXISTS revisions (
                id INTEGER PRIMARY KEY,
                source_url TEXT NOT NULL REFERENCES documents(source_url),
                sha256 TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                relative_path TEXT NOT NULL,
                final_url TEXT NOT NULL,
                etag TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                downloaded_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(source_url,sha256)
            );
        """)
        self._db.commit()

    def __enter__(self) -> ArchiveStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _path(self, relative: str | Path) -> Path:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Archive paths must remain beneath the archive root.")
        path = self.root / relative
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Symbolic links are not allowed inside the archive.")
        if not path.resolve(strict=False).is_relative_to(self.root):
            raise ValueError("Archive path escaped its root.")
        return path

    def _directory(self, relative: str | Path) -> Path:
        path = self._path(relative)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError("An archive directory is occupied by a file.")
        return path

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _immutable_bytes(self, relative: str, content: bytes) -> Path:
        path = self._path(relative)
        self._copy_original(io.BytesIO(content), path, hashlib.sha256(content).hexdigest())
        return path

    def record_catalog_page(
        self,
        catalog: str,
        url: str,
        html: bytes,
        entries: Iterable[object],
        next_urls: Sequence[str] = (),
        total_rows: int | None = None,
    ) -> None:
        """Retain all metadata, including non-English or ambiguous entries."""
        _official_url(url)
        if not catalog or len(catalog) > 100:
            raise ValueError("A catalog name is required.")
        if not isinstance(html, bytes):
            raise TypeError("Catalog snapshots must be bytes.")
        if total_rows is not None and total_rows < 0:
            raise ValueError("Catalog total rows must not be negative.")
        next_urls = tuple(_official_url(item) for item in next_urls)
        prepared: list[dict[str, str]] = []
        for entry in entries:
            data = {
                name: str(entry.get(name, "") if isinstance(entry, Mapping) else
                          getattr(entry, name, ""))
                for name in _ENTRY_FIELDS
            }
            _official_url(data["source_url"])
            if data["catalog"] != catalog:
                raise ValueError("Catalog entry belongs to a different catalog.")
            data["catalog_url"] = data["catalog_url"] or url
            _official_url(data["catalog_url"])
            if data["language"] not in {"en", "non_english", "unknown"}:
                raise ValueError("Catalog language must be explicit.")
            prepared.append(data)
        snapshot_hash = hashlib.sha256(html).hexdigest()
        snapshot_path = f"catalogs/snapshots/{snapshot_hash}.html"
        self._immutable_bytes(snapshot_path, html)
        now = _now()
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO page_snapshots VALUES(?,?,?,?,?)",
                (catalog, url, snapshot_hash, snapshot_path, now),
            )
            self._db.execute(
                """INSERT INTO catalog_pages VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(catalog,url) DO UPDATE SET
                   snapshot_sha256=excluded.snapshot_sha256,
                   snapshot_path=excluded.snapshot_path,fetched_at=excluded.fetched_at,
                   next_urls_json=excluded.next_urls_json,total_rows=excluded.total_rows""",
                (catalog, url, snapshot_hash, snapshot_path, now, json.dumps(next_urls), total_rows),
            )
            for entry in prepared:
                self._db.execute(
                    """INSERT INTO documents(source_url,first_seen_at,last_seen_at)
                       VALUES(?,?,?) ON CONFLICT(source_url)
                       DO UPDATE SET last_seen_at=excluded.last_seen_at""",
                    (entry["source_url"], now, now),
                )
                self._db.execute(
                    """INSERT INTO observations(
                       source_url,catalog,catalog_url,product_number,title,revision_date,
                       posted_date,language,document_kind,snapshot_sha256,observed_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(source_url,catalog,catalog_url,product_number,title,
                       revision_date,posted_date,language,document_kind,snapshot_sha256)
                       DO UPDATE SET observed_at=excluded.observed_at""",
                    (*(entry[name] for name in (
                        "source_url", "catalog", "catalog_url", "product_number", "title",
                        "revision_date", "posted_date", "language", "document_kind",
                    )), snapshot_hash, now),
                )

    def catalog_urls(self, catalog: str) -> list[str]:
        return [page["url"] for page in self.catalog_pages(catalog)]

    def catalog_pages(self, catalog: str | None = None) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM catalog_pages WHERE (? IS NULL OR catalog=?) ORDER BY catalog,url",
            (catalog, catalog),
        )
        result = []
        for row in rows:
            page = dict(row)
            page["next_urls"] = json.loads(page.pop("next_urls_json"))
            result.append(page)
        return result

    def _document(
        self, source_row: sqlite3.Row, requested: set[str] | None = None,
    ) -> dict[str, Any] | None:
        source = dict(source_row)
        latest: dict[str, dict[str, Any]] = {}
        for row in self._db.execute(
            "SELECT * FROM observations WHERE source_url=? ORDER BY observed_at DESC,id DESC",
            (source["source_url"],),
        ):
            observation = dict(row)
            latest.setdefault(observation["catalog"], observation)
        selected = [item for name, item in latest.items()
                    if requested is None or name in requested]
        if not selected:
            return None
        selected.sort(key=lambda item: (
            item["language"] == "en" and item["document_kind"] in _ELIGIBLE_KINDS,
            item["language"] == "en", item["observed_at"], item["id"],
        ), reverse=True)
        preferred = selected[0]
        document = {key: preferred[key] for key in _ENTRY_FIELDS}
        document.update(source)
        document["catalogs"] = sorted(latest)
        document["eligible"] = (
            preferred["language"] == "en" and
            preferred["document_kind"] in _ELIGIBLE_KINDS
        )
        revision = self.latest_revision(source["source_url"])
        document["latest_revision"] = revision
        document["etag"] = revision["etag"] if revision else ""
        document["last_modified"] = revision["last_modified"] if revision else ""
        document["status"] = (
            "skipped" if not document["eligible"] else
            "failed" if source["error"] else
            "downloaded" if revision else "pending"
        )
        return document

    def documents(self, catalogs: Sequence[str] | None = None) -> list[dict[str, Any]]:
        requested = None if catalogs is None else set(catalogs)
        result = []
        for source_row in self._db.execute("SELECT * FROM documents ORDER BY source_url"):
            document = self._document(source_row, requested)
            if document is not None:
                result.append(document)
        return result

    def latest_revision(self, source_url: str) -> dict[str, Any] | None:
        row = self._db.execute(
            """SELECT r.* FROM revisions r JOIN documents d ON d.latest_revision_id=r.id
               WHERE d.source_url=?""", (source_url,),
        ).fetchone()
        if row is None:
            return None
        revision = dict(row)
        revision["metadata"] = json.loads(revision.pop("metadata_json"))
        return revision

    def record_download(
        self, source_url: str, temp_path: Path, final_url: str,
        etag: str = "", last_modified: str = "",
    ) -> dict[str, Any]:
        _official_url(source_url)
        _official_url(final_url)
        row = self._db.execute("SELECT * FROM documents WHERE source_url=?", (source_url,)).fetchone()
        source = self._document(row) if row is not None else None
        if source is None or not source["eligible"]:
            raise ValueError("Only eligible discovered English documents may be archived.")
        temp_path = Path(temp_path)
        if not temp_path.is_absolute():
            temp_path = self.root / temp_path
        try:
            relative = temp_path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Downloads must be staged inside the archive.") from error
        temp_path = self._path(relative)
        if not relative.parts or relative.parts[0] != ".staging":
            raise ValueError("Downloads must be staged beneath .staging.")
        descriptor = os.open(temp_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as incoming:
            if incoming.read(5) != b"%PDF-":
                raise ValueError("Downloaded original does not have a PDF signature.")
            incoming.seek(0)
            digest = hashlib.sha256()
            byte_count = 0
            while chunk := incoming.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            sha256 = digest.hexdigest()
            existing = self._db.execute(
                "SELECT * FROM revisions WHERE source_url=? AND sha256=?",
                (source_url, sha256),
            ).fetchone()
            if existing is None:
                directory = self._directory(f"originals/{_slug(source['catalog'])}")
                original_name = f"{_slug(source['product_number'])}-{sha256}.pdf"
                target = self._path(directory.relative_to(self.root) / original_name)
                relative_path = str(target.relative_to(self.root))
                incoming.seek(0)
                self._copy_original(incoming, target, sha256)
            else:
                relative_path = existing["relative_path"]
                target = self._path(relative_path)
                self._directory(target.parent.relative_to(self.root))
                incoming.seek(0)
                self._copy_original(incoming, target, sha256)
        now = _now()
        metadata = {field: source[field] for field in _ENTRY_FIELDS}
        with self._db:
            self._db.execute(
                """INSERT OR IGNORE INTO revisions(
                   source_url,sha256,bytes,relative_path,final_url,etag,last_modified,
                   downloaded_at,metadata_json) VALUES(?,?,?,?,?,?,?,?,?)""",
                (source_url, sha256, byte_count, relative_path, final_url, etag,
                 last_modified, now, json.dumps(metadata, sort_keys=True)),
            )
            revision_id = self._db.execute(
                "SELECT id FROM revisions WHERE source_url=? AND sha256=?",
                (source_url, sha256),
            ).fetchone()[0]
            self._db.execute(
                """UPDATE documents SET latest_revision_id=?,last_checked_at=?,error=NULL
                   WHERE source_url=?""", (revision_id, now, source_url),
            )
        revision = self.latest_revision(source_url)
        assert revision is not None
        return revision

    def _copy_original(self, incoming: Any, target: Path, expected_hash: str) -> None:
        # An exclusive link publishes only complete bytes. A crash can leave a hidden
        # staging object, but never a partial checksum-named original.
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent,
        )
        temporary = Path(name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as outgoing:
                os.fchmod(outgoing.fileno(), 0o644)
                while chunk := incoming.read(1024 * 1024):
                    outgoing.write(chunk)
                    digest.update(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if digest.hexdigest() != expected_hash:
                raise ValueError("A staged original changed while it was copied.")
            self._path(target.relative_to(self.root))
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                if not target.is_file() or _hash_file(target)[0] != expected_hash:
                    raise ValueError("An existing original has unexpected content.") from None
            self._sync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def record_failure(self, source_url: str, error: str) -> None:
        _official_url(source_url)
        with self._db:
            cursor = self._db.execute(
                """UPDATE documents SET error=?,failure_count=failure_count+1,
                   last_failed_at=? WHERE source_url=?""",
                (str(error)[:1000], _now(), source_url),
            )
            if cursor.rowcount != 1:
                raise ValueError("Cannot record failure for an undiscovered document.")

    def record_not_modified(self, source_url: str) -> dict[str, Any]:
        revision = self.latest_revision(source_url)
        if revision is None:
            raise ValueError("No archived original exists for this document.")
        path = self._path(revision["relative_path"])
        if not path.is_file() or _hash_file(path) != (revision["sha256"], revision["bytes"]):
            raise ValueError("An existing original is missing or has changed.")
        with self._db:
            self._db.execute(
                "UPDATE documents SET error=NULL,last_checked_at=? WHERE source_url=?",
                (_now(), source_url),
            )
        return revision

    @staticmethod
    def _counts(documents: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "discovered": len(documents),
            "eligible": sum(item["eligible"] for item in documents),
            "skipped_non_english": sum(item["language"] == "non_english" for item in documents),
            "skipped_unknown_language": sum(item["language"] == "unknown" for item in documents),
            "skipped_document_kind": sum(
                item["language"] == "en" and item["document_kind"] not in _ELIGIBLE_KINDS
                for item in documents
            ),
            "downloaded": sum(item["latest_revision"] is not None for item in documents),
            "failed": sum(item["status"] == "failed" for item in documents),
            "pending": sum(item["status"] == "pending" for item in documents),
            "downloaded_bytes": sum(
                item["latest_revision"]["bytes"] for item in documents if item["latest_revision"]
            ),
        }

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"schema": 1, **self._counts(self.documents())}
        summary["revisions"] = self._db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
        summary["revision_bytes"] = self._db.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM revisions"
        ).fetchone()[0]
        summary["catalog_pages"] = self._db.execute(
            "SELECT COUNT(*) FROM catalog_pages"
        ).fetchone()[0]
        summary["catalog_snapshots"] = self._db.execute(
            "SELECT COUNT(*) FROM page_snapshots"
        ).fetchone()[0]
        catalogs = [row[0] for row in self._db.execute(
            "SELECT DISTINCT catalog FROM catalog_pages ORDER BY catalog"
        )]
        summary["per_catalog"] = {name: self._counts(self.documents([name])) for name in catalogs}
        return summary

    def save_report(self, name: str, data: object) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Report names must be a single file name.")
        path = self._path(f"reports/{name}")
        return self._write_json(path, data)

    def _write_json(self, path: Path, data: object) -> Path:
        content = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        temp = self._path(path.relative_to(self.root).parent / f".{path.name}-{os.urandom(8).hex()}.tmp")
        try:
            with temp.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            self._path(path.relative_to(self.root))
            temp.replace(path)
            self._sync_directory(path.parent)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def export_manifest(self) -> Path:
        """Write a readable derivative; SQLite and immutable originals are authoritative."""
        revisions = []
        for row in self._db.execute("SELECT * FROM revisions ORDER BY id"):
            revision = dict(row)
            revision["metadata"] = json.loads(revision.pop("metadata_json"))
            revisions.append(revision)
        return self._write_json(self._path("catalogs/manifest.json"), {
            "schema": 1, "generated_at": _now(), "summary": self.summary(),
            "catalog_pages": self.catalog_pages(), "documents": self.documents(),
            "revisions": revisions,
        })
