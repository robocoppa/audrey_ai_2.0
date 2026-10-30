"""Lifecycle-owned, bounded local indexing and canonical search hydration."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from audrey.content_search import (
    EXCERPT_CHARS,
    ContentSearchError,
    ContentSearchIndex,
    SearchDocument,
)
from audrey.content_sources import ContentSourceReader, SearchCoverageIssue

log = logging.getLogger(__name__)
SOURCE_KINDS = ("message", "file", "agent", "knowledge")
_KINDS = SOURCE_KINDS


@dataclass
class _OwnerScan:
    namespace: str
    cursors: dict[str, str | None] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)
    warnings: set[str] = field(default_factory=set)
    coverage: dict[str, SearchCoverageIssue] = field(default_factory=dict)
    completed_at: float = 0
    indexed: dict[str, dict] | None = None
    generation: int = field(default_factory=lambda: secrets.randbits(64))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def document_hash(document: SearchDocument) -> str:
    return hashlib.sha256(document.text.encode()).hexdigest()


def _same_document(document: SearchDocument, metadata: dict) -> bool:
    return (
        document.owner_id == metadata["owner_id"]
        and document.source_id == metadata["source_id"]
        and document.kind == metadata["kind"]
        and str(document.revision) == str(metadata["revision"])
        and document_hash(document) == metadata["body_hash"]
        and document.title == metadata["title"]
        and document.updated_at == metadata["updated_at"]
        and document.target == metadata["target"]
        and document.project_id == metadata["project_id"]
        and document.archived == metadata["archived"]
        and document.private_required == metadata["private_required"]
        and document.availability == metadata["availability"]
    )


def _valid_excerpt(document: SearchDocument, candidate: dict) -> bool:
    offset, excerpt, highlights = (
        candidate.get(key) for key in ("offset", "excerpt", "highlights")
    )
    if (
        type(offset) is not int
        or not isinstance(excerpt, str)
        or not 0 <= offset <= len(document.text)
        or len(excerpt) > EXCERPT_CHARS
        or document.text[offset : offset + len(excerpt)] != excerpt
        or not isinstance(highlights, list)
        or len(highlights) > 64
    ):
        return False
    return all(
        isinstance(span, dict)
        and type(span.get("start")) is int
        and type(span.get("end")) is int
        and 0 <= span["start"] < span["end"] <= len(excerpt)
        for span in highlights
    )


class ContentSearchService:
    """One small indexing worker, independent of model and embedding services.

    Owners are admitted by an authenticated request or recovered from the existing
    projection. Each pass advances bounded keyset pages, then removes obsolete
    rows. Query results are always hydrated from their canonical sources.
    """

    def __init__(self, store: Any, reader: ContentSourceReader, index: ContentSearchIndex):
        self.store = store
        self.reader = reader
        self.index = index
        self._owners: dict[str, _OwnerScan] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.requests = asyncio.Semaphore(4)

    async def start(self) -> None:
        for owner in await self.index.list_owners():
            principal = await self.store.get_principal(user_id=owner)
            if principal is None or principal.status != "active":
                await self.index.remove_owner(owner)
            elif len(self._owners) < 512:
                self._owners[owner] = _OwnerScan(principal.storage_namespace)
        self._task = asyncio.create_task(self._run(), name="audrey.content_search")
        self._wake.set()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def purge_owner(self, owner: str) -> None:
        """Purge cached personal text after canonical deletion fences are installed."""
        scan = self._owners.get(owner)
        if scan is None:
            await self.index.remove_owner(owner)
            return
        async with scan.lock:
            await self.index.remove_owner(owner)
            self._restart_scan(scan)
            scan.generation += 1
            if self._owners.get(owner) is scan:
                self._owners.pop(owner)

    async def refresh_knowledge(self, owner: str, namespace: str, entry_ids: list[str]) -> None:
        """Refresh at most a changed entry and its superseded predecessor.

        Canonical writes already succeeded. Never admit a new search owner here,
        and never let a read begun before purge repopulate the removed projection.
        """
        if (
            not isinstance(entry_ids, (list, tuple)) or len(entry_ids) > 2
            or any(not isinstance(entry_id, str) or not 0 < len(entry_id) <= 200
                   for entry_id in entry_ids)
        ):
            raise ContentSearchError("Invalid knowledge search refresh.")
        scan = self._owners.get(owner)
        if scan is None or scan.namespace != namespace:
            return
        documents = [
            (f"knowledge:{entry_id}", await self.reader.resolve(owner, namespace, f"knowledge:{entry_id}"))
            for entry_id in dict.fromkeys(entry_ids)
        ]
        async with scan.lock:
            if self._owners.get(owner) is not scan:
                return
            for source_id, document in documents:
                if document is not None and self.reader.is_current(document):
                    await self.index.replace_source(document)
                    scan.seen.add(source_id)
                else:
                    await self.index.remove_source(owner, source_id)
                    scan.seen.discard(source_id)
                self._changed(owner)

    @staticmethod
    def _restart_scan(scan: _OwnerScan) -> None:
        scan.cursors.clear()
        scan.done.clear()
        scan.seen.clear()
        scan.warnings.clear()
        scan.coverage.clear()
        scan.indexed = None
        scan.completed_at = 0

    def _changed(self, owner: str) -> None:
        scan = self._owners.get(owner)
        if scan is not None:
            scan.generation += 1
            # Hydration and backfill can run concurrently; reload metadata on
            # the next bounded pass after any out-of-band projection repair.
            scan.indexed = None

    async def prepare(self, owner: str, namespace: str) -> _OwnerScan:
        scan = self._owners.get(owner)
        if scan is None:
            if len(self._owners) >= 512:
                raise ContentSearchError("Local search owner allowance reached.")
            scan = self._owners[owner] = _OwnerScan(namespace)
        if scan.completed_at and time.monotonic() - scan.completed_at >= 30:
            self._restart_scan(scan)
        # Give an empty projection its first bounded page without waiting for a
        # whole history or for document extraction across the corpus.
        if not scan.cursors and not scan.done:
            await self._advance(owner, scan)
        self._wake.set()
        return scan

    async def _advance(self, owner: str, scan: _OwnerScan) -> None:
        async with scan.lock:
            if self._owners.get(owner) is not scan:
                return
            if len(scan.done) == len(_KINDS):
                return
            principal = await self.store.get_principal(user_id=owner)
            if principal is None or principal.status != "active":
                await self.index.remove_owner(owner)
                scan.generation += 1
                scan.done.update(_KINDS)
                scan.completed_at = time.monotonic()
                self._owners.pop(owner, None)
                return
            if scan.indexed is None:
                scan.indexed = {
                    source["source_id"]: source for source in await self.index.list_sources(owner)
                }
            for kind in _KINDS:
                if kind in scan.done:
                    continue
                documents, cursor, warnings = await self.reader.list_page(
                    owner,
                    principal.storage_namespace,
                    kind,
                    after=scan.cursors.get(kind),
                    limit=5 if kind in {"file", "knowledge"} else 50,
                )
                for warning in warnings:
                    if isinstance(warning, SearchCoverageIssue):
                        if warning.source_id in scan.coverage or len(scan.coverage) < 5_000:
                            scan.coverage[warning.source_id] = warning
                        else:
                            scan.warnings.add("Some sources exceed the local search allowance.")
                    else:
                        scan.warnings.add(warning)
                for document in documents:
                    if len(scan.seen) >= 5_000:
                        scan.warnings.add("Some sources exceed the local search allowance.")
                        scan.done.update(_KINDS)
                        break
                    if not self.reader.is_current(document):
                        continue
                    scan.seen.add(document.source_id)
                    saved = (
                        scan.indexed.get(document.source_id) if scan.indexed is not None else None
                    )
                    if saved is not None and _same_document(document, saved):
                        continue
                    try:
                        await self.index.replace_source(document)
                        scan.generation += 1
                    except ContentSearchError:
                        scan.warnings.add("Some sources exceed the local search allowance.")
                        # A failed replacement must not retain a previous text
                        # revision indefinitely merely because indexing is full.
                        if saved is not None:
                            await self.index.remove_source(owner, document.source_id)
                            scan.generation += 1
                scan.cursors[kind] = cursor
                if cursor is None:
                    scan.done.add(kind)
            if len(scan.done) == len(_KINDS):
                for source in await self.index.list_sources(owner):
                    if source["source_id"] not in scan.seen:
                        await self.index.remove_source(owner, source["source_id"])
                        scan.generation += 1
                scan.completed_at = time.monotonic()

    async def _run(self) -> None:
        while True:
            self._wake.clear()
            more = False
            for owner, scan in list(self._owners.items()):
                if scan.completed_at and time.monotonic() - scan.completed_at >= 30:
                    self._restart_scan(scan)
                if len(scan.done) < len(_KINDS):
                    try:
                        await self._advance(owner, scan)
                    except Exception:  # noqa: BLE001 — isolated local projection failure
                        log.warning("Local search indexing pass failed", exc_info=False)
                        scan.warnings.add("Some sources could not be indexed. Search will retry.")
                        scan.done.update(_KINDS)
                        scan.completed_at = time.monotonic()
                    more |= len(scan.done) < len(_KINDS)
            if more:
                await asyncio.sleep(0.1)
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=30)
            except TimeoutError:
                pass

    async def hydrate(self, owner: str, namespace: str, candidate: dict) -> SearchDocument | None:
        document = await self.reader.resolve(owner, namespace, candidate["source_id"])
        if (
            document is not None
            and _same_document(document, candidate)
            and _valid_excerpt(document, candidate)
        ):
            return document
        scan = self._owners.get(owner)
        if scan is None:
            return None
        # Serialize repairs with canonical-data purge as well as backfill. A
        # read started before purge must never repopulate the cleared cache.
        async with scan.lock:
            if self._owners.get(owner) is not scan:
                return None
            if document is None or not self.reader.is_current(document):
                await self.index.remove_source(owner, candidate["source_id"])
            else:
                await self.index.replace_source(document)
            self._changed(owner)
        return None
