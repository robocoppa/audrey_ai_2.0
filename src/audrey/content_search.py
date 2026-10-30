"""Rebuildable local lexical projection; canonical sources remain authoritative.

This module never resolves identity or trusts a projection as permission to read a
source. Its callers revalidate ownership, revision and deletion fences against
the application store before returning any indexed excerpt.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

MAX_QUERY_CHARS = 256
MAX_QUERY_TERMS = 12
MAX_SOURCE_CHARS = 500_000
MAX_OWNER_CHARS = 10_000_000
MAX_GLOBAL_CHARS = 64_000_000
MAX_OWNER_SOURCES = 5_000
MAX_GLOBAL_SOURCES = 50_000
MAX_OWNERS = 512
MAX_CANDIDATES = 200
CHUNK_CHARS = 4_000
CHUNK_OVERLAP = 200
EXCERPT_CHARS = 400
_KINDS = frozenset({"message", "file", "agent", "knowledge"})
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


class ContentSearchError(RuntimeError):
    """A local projection operation could not complete."""


class ContentSearchUnavailableError(ContentSearchError):
    """SQLite/FTS5 is unavailable, or the projection requires rebuilding."""


class ContentSearchQueryError(ContentSearchError, ValueError):
    """The query or its filters exceed supported bounds."""


class ContentSearchQuotaError(ContentSearchError):
    """The projection's bounded storage allowance has been reached."""


@dataclass(frozen=True)
class SearchDocument:
    owner_id: str
    source_id: str
    kind: str
    revision: str | int
    title: str
    text: str
    updated_at: str
    project_id: str | None = None
    archived: bool = False
    private_required: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    availability: str = "available"


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256 or "\x00" in value:
        raise ContentSearchQueryError(f"Invalid {name}.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ContentSearchQueryError(f"Invalid {name}.") from exc


def _query_terms(query: str) -> list[str]:
    if not isinstance(query, str) or len(query) > MAX_QUERY_CHARS or "\x00" in query:
        raise ContentSearchQueryError("Search accepts at most 256 characters.")
    terms = _WORDS.findall(query)
    if not terms or len(terms) > MAX_QUERY_TERMS:
        raise ContentSearchQueryError("Search needs between 1 and 12 literal terms.")
    return list(dict.fromkeys(term.casefold() for term in terms))


def _fold(text: str) -> tuple[str, list[int]]:
    """Fold accents while retaining a map to original Python character offsets."""
    chars: list[str] = []
    offsets: list[int] = []
    for offset, char in enumerate(text):
        for folded in unicodedata.normalize("NFKD", char).casefold():
            category = unicodedata.category(folded)
            if category.startswith("M"):
                continue
            chars.append(folded if category[0] in {"L", "N"} or category == "Co" else " ")
            offsets.append(offset)
    return "".join(chars), offsets


def _excerpt(text: str, terms: list[str]) -> tuple[str, int, list[dict[str, int]]]:
    folded, offsets = _fold(text)
    folded_terms = {_fold(term)[0] for term in terms}
    ranges = []
    for match in _WORDS.finditer(folded):
        if match.group() in folded_terms:
            start, end = offsets[match.start()], offsets[match.end() - 1] + 1
            while end < len(text) and unicodedata.category(text[end]).startswith("M"):
                end += 1
            ranges.append((start, end))
    excerpt_start = max(0, (ranges[0][0] - 140) if ranges else 0)
    excerpt_end = min(len(text), excerpt_start + EXCERPT_CHARS)
    highlights = [
        {"start": start - excerpt_start, "end": min(end, excerpt_end) - excerpt_start}
        for start, end in ranges
        if excerpt_start <= start < excerpt_end
    ][:64]
    return text[excerpt_start:excerpt_end], excerpt_start, highlights


class ContentSearchIndex:
    """One dedicated SQLite database, with bounded writes and off-loop I/O."""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.RLock()
        self._closed = False
        try:
            if str(path) != ":memory:":
                database = Path(path)
                if database.is_symlink() or database.parent.is_symlink():
                    raise ContentSearchUnavailableError("The local search path is unsafe.")
                database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                str(path), check_same_thread=False, isolation_level=None
            )
            self._connection.row_factory = sqlite3.Row
            if str(path) != ":memory:":
                # SQLite derives WAL/SHM permissions from the database file.
                Path(path).chmod(0o600)
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA secure_delete=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA journal_size_limit=8388608")
            page_size = self._connection.execute("PRAGMA page_size").fetchone()[0]
            self._connection.execute(f"PRAGMA max_page_count={134_217_728 // page_size}")
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise ContentSearchUnavailableError("The search projection needs rebuilding.")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS search_sources (
                    owner_id TEXT NOT NULL, source_id TEXT NOT NULL, kind TEXT NOT NULL,
                    revision_json TEXT NOT NULL, title TEXT NOT NULL, updated_at TEXT NOT NULL,
                    project_id TEXT, archived INTEGER NOT NULL, private_required INTEGER NOT NULL,
                    target_json TEXT NOT NULL, availability TEXT NOT NULL,
                    body_hash TEXT NOT NULL, char_count INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, source_id)
                );
                CREATE TABLE IF NOT EXISTS search_chunks (
                    id INTEGER PRIMARY KEY, owner_id TEXT NOT NULL, source_id TEXT NOT NULL,
                    owner_key TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    FOREIGN KEY (owner_id, source_id) REFERENCES search_sources(owner_id, source_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS search_chunks_source
                    ON search_chunks(owner_id, source_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                    owner_key, title, text, content='search_chunks', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );
                PRAGMA user_version=1;
                """
            )
            try:
                self._connection.execute(
                    "INSERT INTO search_fts(search_fts, rank) VALUES ('secure-delete', 1)"
                )
            except sqlite3.OperationalError as exc:
                if exc.sqlite_errorcode != sqlite3.SQLITE_ERROR:
                    raise
                # FTS5 secure-delete first shipped in SQLite 3.42. The raw
                # content table still uses secure_delete on older runtimes.
            if str(path) != ":memory:":
                for suffix in ("-wal", "-shm"):
                    sidecar = Path(f"{path}{suffix}")
                    if sidecar.exists():
                        sidecar.chmod(0o600)
        except (sqlite3.Error, OSError, ContentSearchUnavailableError) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise ContentSearchUnavailableError("Local full-text search is unavailable.") from exc

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._connection.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ContentSearchUnavailableError("Local full-text search is closed.")

    @staticmethod
    def _validate_document(document: SearchDocument) -> tuple[str, str]:
        _identifier(document.owner_id, "owner")
        _identifier(document.source_id, "source")
        if document.project_id is not None:
            _identifier(document.project_id, "Project")
        if document.kind not in _KINDS:
            raise ContentSearchQueryError("Unsupported source kind.")
        if not isinstance(document.text, str) or len(document.text) > MAX_SOURCE_CHARS:
            raise ContentSearchQuotaError("A search source exceeds 500,000 characters.")
        if not isinstance(document.title, str) or len(document.title) > 1_000:
            raise ContentSearchQueryError("A search title exceeds 1,000 characters.")
        if not isinstance(document.updated_at, str) or len(document.updated_at) > 100:
            raise ContentSearchQueryError("Invalid source update date.")
        if not isinstance(document.availability, str) or len(document.availability) > 100:
            raise ContentSearchQueryError("Invalid source availability.")
        if type(document.archived) is not bool or type(document.private_required) is not bool:
            raise ContentSearchQueryError("Invalid source policy flags.")
        if type(document.revision) not in {str, int}:
            raise ContentSearchQueryError("Invalid source revision.")
        try:
            revision = json.dumps(document.revision, ensure_ascii=False, allow_nan=False)
            target = json.dumps(document.target, ensure_ascii=False, allow_nan=False)
            for value in (
                document.title,
                document.text,
                document.updated_at,
                document.availability,
                revision,
                target,
            ):
                value.encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ContentSearchQueryError("Invalid source metadata.") from exc
        if not isinstance(document.target, dict) or len(target) > 4_000 or len(revision) > 256:
            raise ContentSearchQueryError("Source metadata exceeds supported bounds.")
        return revision, target

    async def replace_source(self, document: SearchDocument) -> None:
        revision, target = self._validate_document(document)
        await anyio.to_thread.run_sync(self._replace_source, document, revision, target)

    def _remove_rows(self, owner_id: str, source_id: str) -> None:
        connection = self._connection
        chunks = connection.execute(
            "SELECT id, owner_key, title, text FROM search_chunks WHERE owner_id=? AND source_id=?",
            (owner_id, source_id),
        ).fetchall()
        connection.executemany(
            "INSERT INTO search_fts(search_fts, rowid, owner_key, title, text) "
            "VALUES ('delete', ?, ?, ?, ?)",
            [(row["id"], row["owner_key"], row["title"], row["text"]) for row in chunks],
        )
        connection.execute(
            "DELETE FROM search_sources WHERE owner_id=? AND source_id=?", (owner_id, source_id)
        )

    def _replace_source(self, document: SearchDocument, revision: str, target: str) -> None:
        with self._lock:
            self._check_open()
            connection = self._connection
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._remove_rows(document.owner_id, document.source_id)
                stats = connection.execute(
                    "SELECT COUNT(*) AS sources, COUNT(DISTINCT owner_id) AS owners, "
                    "COALESCE(SUM(char_count),0) AS chars, "
                    "COALESCE(SUM(owner_id=?),0) AS owner_sources, "
                    "COALESCE(SUM(CASE WHEN owner_id=? THEN char_count ELSE 0 END),0) AS owner_chars "
                    "FROM search_sources",
                    (document.owner_id, document.owner_id),
                ).fetchone()
                if (
                    stats["sources"] >= MAX_GLOBAL_SOURCES
                    or stats["owner_sources"] >= MAX_OWNER_SOURCES
                    or (stats["owner_sources"] == 0 and stats["owners"] >= MAX_OWNERS)
                    or stats["chars"] + len(document.text) > MAX_GLOBAL_CHARS
                    or stats["owner_chars"] + len(document.text) > MAX_OWNER_CHARS
                ):
                    raise ContentSearchQuotaError("Local search storage allowance reached.")
                connection.execute(
                    "INSERT INTO search_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.owner_id,
                        document.source_id,
                        document.kind,
                        revision,
                        document.title,
                        document.updated_at,
                        document.project_id,
                        int(document.archived),
                        int(document.private_required),
                        target,
                        document.availability,
                        hashlib.sha256(document.text.encode()).hexdigest(),
                        len(document.text),
                    ),
                )
                owner_key = hashlib.sha256(document.owner_id.encode()).hexdigest()
                for offset in range(0, max(1, len(document.text)), CHUNK_CHARS - CHUNK_OVERLAP):
                    text = document.text[offset : offset + CHUNK_CHARS]
                    cursor = connection.execute(
                        "INSERT INTO search_chunks(owner_id, source_id, owner_key, title, text, offset) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            document.owner_id,
                            document.source_id,
                            owner_key,
                            document.title,
                            text,
                            offset,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO search_fts(rowid, owner_key, title, text) VALUES (?, ?, ?, ?)",
                        (cursor.lastrowid, owner_key, document.title, text),
                    )
                connection.execute("COMMIT")
            except (sqlite3.Error, ContentSearchError) as exc:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                if isinstance(exc, ContentSearchError):
                    raise
                if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
                    raise ContentSearchQuotaError(
                        "Local search storage allowance reached."
                    ) from exc
                raise ContentSearchUnavailableError("Local search indexing failed.") from exc

    async def remove_source(self, owner_id: str, source_id: str) -> None:
        _identifier(owner_id, "owner")
        _identifier(source_id, "source")
        await anyio.to_thread.run_sync(self._remove_source, owner_id, source_id)

    def _remove_source(self, owner_id: str, source_id: str) -> None:
        with self._lock:
            self._check_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._remove_rows(owner_id, source_id)
                self._connection.execute("COMMIT")
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise ContentSearchUnavailableError("Local search deletion failed.") from exc

    @staticmethod
    def _metadata(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "owner_id": row["owner_id"],
            "source_id": row["source_id"],
            "kind": row["kind"],
            "revision": json.loads(row["revision_json"]),
            "title": row["title"],
            "updated_at": row["updated_at"],
            "project_id": row["project_id"],
            "archived": bool(row["archived"]),
            "private_required": bool(row["private_required"]),
            "target": json.loads(row["target_json"]),
            "availability": row["availability"],
            "body_hash": row["body_hash"],
        }

    async def list_sources(self, owner_id: str) -> list[dict[str, Any]]:
        _identifier(owner_id, "owner")
        return await anyio.to_thread.run_sync(self._list_sources, owner_id)

    def _list_sources(self, owner_id: str) -> list[dict[str, Any]]:
        with self._lock:
            self._check_open()
            try:
                rows = self._connection.execute(
                    "SELECT * FROM search_sources WHERE owner_id=? ORDER BY source_id", (owner_id,)
                ).fetchall()
                return [self._metadata(row) for row in rows]
            except sqlite3.Error as exc:
                raise ContentSearchUnavailableError(
                    "Local search metadata is unavailable."
                ) from exc

    async def list_owners(self) -> list[str]:
        """Internal reconciliation inventory; never an endpoint identity selector."""
        return await anyio.to_thread.run_sync(self._list_owners)

    def _list_owners(self) -> list[str]:
        with self._lock:
            self._check_open()
            try:
                rows = self._connection.execute(
                    "SELECT DISTINCT owner_id FROM search_sources ORDER BY owner_id LIMIT ?",
                    (MAX_OWNERS,),
                ).fetchall()
                return [row["owner_id"] for row in rows]
            except sqlite3.Error as exc:
                raise ContentSearchUnavailableError(
                    "Local search owner inventory is unavailable."
                ) from exc

    async def remove_owner(self, owner_id: str) -> None:
        _identifier(owner_id, "owner")
        await anyio.to_thread.run_sync(self._remove_owner, owner_id)

    def _remove_owner(self, owner_id: str) -> None:
        with self._lock:
            self._check_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                sources = self._connection.execute(
                    "SELECT source_id FROM search_sources WHERE owner_id=?", (owner_id,)
                ).fetchall()
                for source in sources:
                    self._remove_rows(owner_id, source["source_id"])
                self._connection.execute("COMMIT")
                # Owner-data purge also discards old WAL frames, which may
                # otherwise retain pre-deletion copies of private text pages.
                self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise ContentSearchUnavailableError("Local search owner cleanup failed.") from exc

    async def search(
        self,
        owner_id: str,
        query: str,
        *,
        kind: str | None = None,
        project_id: str | None = None,
        archived: bool | None = False,
        offset: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        _identifier(owner_id, "owner")
        terms = _query_terms(query)
        if kind is not None and kind not in _KINDS:
            raise ContentSearchQueryError("Unsupported source kind.")
        if project_id is not None:
            _identifier(project_id, "Project")
        if archived is not None and type(archived) is not bool:
            raise ContentSearchQueryError("Invalid archive filter.")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ContentSearchQueryError("Search offset exceeds supported bounds.")
        if type(limit) is not int or not 1 <= limit <= MAX_CANDIDATES:
            raise ContentSearchQueryError("Search page accepts between 1 and 200 results.")
        return await anyio.to_thread.run_sync(
            self._search,
            owner_id,
            terms,
            kind,
            project_id,
            archived,
            offset,
            limit,
        )

    def _search(
        self,
        owner_id: str,
        terms: list[str],
        kind: str | None,
        project_id: str | None,
        archived: bool | None,
        offset: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        owner_key = hashlib.sha256(owner_id.encode()).hexdigest()
        literal_terms = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)
        match = f'owner_key : "{owner_key}" AND {{title text}} : ({literal_terms})'
        parameters = (
            match,
            owner_id,
            kind,
            kind,
            project_id,
            project_id,
            archived,
            archived,
            limit,
            offset,
        )
        # SQLite's single MIN aggregate chooses bare chunk columns from a row
        # attaining that minimum. Deduplicate before LIMIT so one long source
        # cannot occupy every candidate slot.
        sql = (
            "SELECT s.*, c.text, c.offset, MIN(search_fts.rank) AS relevance "
            "FROM search_fts JOIN search_chunks c ON c.id=search_fts.rowid "
            "JOIN search_sources s ON s.owner_id=c.owner_id AND s.source_id=c.source_id "
            "WHERE search_fts MATCH ? AND s.owner_id=? "
            "AND (? IS NULL OR s.kind=?) AND (? IS NULL OR s.project_id=?) "
            "AND (? IS NULL OR s.archived=?) GROUP BY s.owner_id, s.source_id "
            "ORDER BY relevance, s.updated_at DESC, s.source_id LIMIT ? OFFSET ?"
        )
        with self._lock:
            self._check_open()
            progress_calls = 0

            def bounded_work() -> int:
                nonlocal progress_calls
                progress_calls += 1
                return int(progress_calls > 10_000)

            self._connection.set_progress_handler(bounded_work, 1_000)
            try:
                rows = self._connection.execute(sql, parameters).fetchall()
            except sqlite3.Error as exc:
                raise ContentSearchUnavailableError(
                    "Local search could not finish within its bounds."
                ) from exc
            finally:
                self._connection.set_progress_handler(None, 0)
            results = []
            for row in rows:
                excerpt, start, highlights = _excerpt(row["text"], terms)
                results.append(
                    {
                        **self._metadata(row),
                        "excerpt": excerpt,
                        "highlights": highlights,
                        "offset": row["offset"] + start,
                    }
                )
            return results
