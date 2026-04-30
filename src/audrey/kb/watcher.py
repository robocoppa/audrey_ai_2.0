"""Debounced filesystem watcher for automatic KB ingest.

Watchdog runs its observer on a native thread. We bridge to asyncio by
pushing events onto a queue from the watchdog callback and consuming
them from an async worker task. The worker debounces per-(kind, path):
if a file changes N times within `debounce_s` seconds, we only act
once after the last change settles. Bulk operations (e.g. `cp -r`)
thus become O(files) ingests, not O(events).

Phase 27: handles deletes and renames properly. Pre-Phase-27 the
watcher only knew about `on_created` / `on_modified` / `on_moved` and
treated `on_moved` as "ingest the new path" — leaving stale vectors
behind for both deleted files and the source side of renames. Now:

- `on_deleted`         → enqueue ("delete", path) → `delete_by_source`
- `on_moved` (rename)  → enqueue ("delete", src), then ("ingest", dest)
- `on_created`         → enqueue ("ingest", path) → ingest as before
- `on_modified`        → enqueue ("ingest", path) → re-ingest

The watcher is gated behind `KB_WATCHER_ENABLED=1` so tests and the
smoke-test container don't start it.

On shutdown, `stop()` cancels the worker and stops the observer; the
orchestrator awaits it in its lifespan teardown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from audrey.kb.chunk import IMAGE_SUFFIXES, TEXT_SUFFIXES
from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.ingest import ingest_image_file, ingest_text_file
from audrey.kb.qdrant import QdrantKB

log = logging.getLogger(__name__)

_DOC_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx", ".html", ".htm"}
_ALL_SUFFIXES = _DOC_SUFFIXES | IMAGE_SUFFIXES

EventKind = Literal["ingest", "delete"]


class _QueueHandler(FileSystemEventHandler):
    """Forward filesystem events onto an asyncio queue (thread-safe).

    Each enqueued item is a `(kind, path)` tuple where `kind` is
    `"ingest"` (create/modify, or the destination side of a rename) or
    `"delete"` (delete, or the source side of a rename).
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[tuple[EventKind, Path]],
    ) -> None:
        self._loop = loop
        self._queue = queue

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue("ingest", event.src_path, event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue("ingest", event.src_path, event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._enqueue("delete", event.src_path, event)

    def on_moved(self, event: FileSystemEvent) -> None:
        # Rename: delete vectors for the source, ingest at the destination.
        # Either side may have an unsupported suffix (e.g. mv image.png
        # image.png.bak) — _enqueue filters per side independently.
        if event.is_directory:
            return
        self._enqueue("delete", event.src_path, event)
        dest = getattr(event, "dest_path", None)
        if dest:
            self._enqueue("ingest", dest, event)

    def _enqueue(self, kind: EventKind, raw_path: str, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if not raw_path:
            return
        path = Path(raw_path)
        if path.name.startswith("."):
            return
        if path.suffix.lower() not in _ALL_SUFFIXES:
            return
        # asyncio.Queue isn't thread-safe; schedule the put on the loop.
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, path))


class KBWatcher:
    def __init__(
        self,
        *,
        roots: list[Path],
        qdrant: QdrantKB,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder | None,
        debounce_s: float = 2.0,
        chunk_tokens: int = 1000,
        overlap_tokens: int = 100,
    ) -> None:
        self._roots = [r for r in roots if r.exists()]
        self._qdrant = qdrant
        self._text = text_embedder
        self._image = image_embedder
        self._debounce_s = max(0.25, debounce_s)
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._observer: Observer | None = None
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue[tuple[EventKind, Path]] | None = None

    async def start(self) -> None:
        if not self._roots:
            log.info("kb.watcher: no valid roots, not starting")
            return
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        handler = _QueueHandler(loop, self._queue)
        self._observer = Observer()
        for root in self._roots:
            self._observer.schedule(handler, str(root), recursive=True)
        self._observer.start()
        self._task = asyncio.create_task(self._run(), name="kb-watcher")
        log.info("kb.watcher: watching %d root(s): %s", len(self._roots), [str(r) for r in self._roots])

    async def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        assert self._queue is not None
        # Debounce by (kind, path) so a delete-then-ingest sequence
        # (typical for `mv` or save-as-rename editors) doesn't collapse
        # into one event.
        pending: dict[tuple[EventKind, Path], float] = {}
        while True:
            try:
                timeout = self._debounce_s if pending else None
                kind, path = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                pending[(kind, path)] = time.monotonic()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                return
            now = time.monotonic()
            due = [k for k, t in pending.items() if now - t >= self._debounce_s]
            # Process deletes before ingests for the same path. Without
            # this, a `mv old.pdf new.pdf` could ingest new.pdf first,
            # then the delete for old.pdf races with the ingest's
            # `delete_by_source` no-op-on-missing.
            due.sort(key=lambda k: 0 if k[0] == "delete" else 1)
            for key in due:
                pending.pop(key, None)
                kind, path = key
                if kind == "delete":
                    await self._delete_vectors(path)
                else:
                    await self._handle_ingest(path)

    async def _handle_ingest(self, path: Path) -> None:
        if not path.exists():
            # The file may have been deleted between the event firing
            # and the debounce expiring. Treat as a no-op.
            return
        suffix = path.suffix.lower()
        try:
            if suffix in _DOC_SUFFIXES:
                n = await ingest_text_file(
                    path, qdrant=self._qdrant, embedder=self._text,
                    chunk_tokens=self._chunk_tokens, overlap_tokens=self._overlap_tokens,
                )
                log.info("kb.watcher: reingested text %s -> %d chunks", path, n)
            elif suffix in IMAGE_SUFFIXES and self._image is not None:
                ok = await ingest_image_file(path, qdrant=self._qdrant, embedder=self._image)
                log.info("kb.watcher: reingested image %s -> %s", path, "ok" if ok else "failed")
        except Exception as e:  # noqa: BLE001 — watcher must stay alive
            log.warning("kb.watcher: %s failed: %s", path, e)

    async def _delete_vectors(self, path: Path) -> None:
        """Remove KB vectors keyed off this source path.

        We don't know whether a deleted file was text or image without
        consulting qdrant, and probing is more work than just calling
        `delete_by_source` on both collections — qdrant treats a no-op
        delete as a successful empty operation. The qdrant-client
        UpdateResult doesn't expose deleted-count, so we can't quiet
        the log on no-op deletes; routed at debug to avoid noise from
        editor swap-file churn (.tmp, ~ etc are already filtered by
        the suffix allowlist, so we mostly only get here for real
        deletes).
        """
        src = str(path)
        try:
            await self._qdrant.delete_by_source(src, collection="kb_text")
            if self._image is not None:
                await self._qdrant.delete_by_source(src, collection="kb_images")
            log.info("kb.watcher: requested delete of vectors for %s", path)
        except Exception as e:  # noqa: BLE001 — watcher must stay alive
            log.warning("kb.watcher: delete %s failed: %s", path, e)


__all__ = ["KBWatcher"]
