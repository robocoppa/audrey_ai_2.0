"""Debounced filesystem watcher for automatic KB ingest.

Watchdog runs its observer on a native thread. We bridge to asyncio by
pushing events onto a queue from the watchdog callback and consuming
them from an async worker task. The worker debounces per-path — if a
file changes N times within `debounce_s` seconds, we only re-ingest it
once after the last change settles. Bulk operations (e.g. `cp -r`) thus
become O(files) ingests, not O(events).

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

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from audrey.kb.chunk import IMAGE_SUFFIXES, TEXT_SUFFIXES
from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.ingest import ingest_image_file, ingest_text_file
from audrey.kb.qdrant import QdrantKB

log = logging.getLogger(__name__)

_DOC_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx", ".html", ".htm"}


class _QueueHandler(FileSystemEventHandler):
    """Forward filesystem events onto an asyncio queue (thread-safe)."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[Path]) -> None:
        self._loop = loop
        self._queue = queue

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def _enqueue(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = getattr(event, "dest_path", None) or event.src_path
        if not src:
            return
        path = Path(src)
        if path.name.startswith("."):
            return
        if path.suffix.lower() not in _DOC_SUFFIXES and path.suffix.lower() not in IMAGE_SUFFIXES:
            return
        # asyncio.Queue isn't thread-safe; schedule the put on the loop.
        self._loop.call_soon_threadsafe(self._queue.put_nowait, path)


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
        self._queue: asyncio.Queue[Path] | None = None

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
        pending: dict[Path, float] = {}
        while True:
            try:
                timeout = self._debounce_s if pending else None
                path = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                pending[path] = time.monotonic()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                return
            now = time.monotonic()
            due = [p for p, t in pending.items() if now - t >= self._debounce_s]
            for p in due:
                pending.pop(p, None)
                await self._handle_path(p)

    async def _handle_path(self, path: Path) -> None:
        if not path.exists():
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


__all__ = ["KBWatcher"]
