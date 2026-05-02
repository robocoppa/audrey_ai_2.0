"""KB reconcile pass — drift catch-up for the global text/image collections.

The Phase 27 watcher fixes the *steady-state* leak: while running, a
deleted file's vectors get cleaned up via `on_deleted` / `on_moved`.
But anything that happens while the watcher isn't running — container
restart, `KB_WATCHER_ENABLED=0` for a stretch, bulk `rm` on
`/mnt/user/knowledge/...`, an atomic-rename pattern watchdog struggles
with on certain filesystems — leaves stale vectors in `kb_text` and
`kb_images` forever.

Phase 30 adds a periodic reconciler that scrolls every point in the
two global collections, groups by `payload.source`, checks each unique
source against `Path.exists()`, and calls `delete_by_source` for any
orphan. Returns a summary dict suitable for logging or admin response.

Excludes per-user collections (`kb_user_text_*` / `kb_user_images_*`)
on purpose — Phase 15's sqlite uploads index already does its own
reconcile against qdrant at startup. Mixing the two paths would let
this reconciler delete legitimately-uploaded files just because the
host can't see the user's filename (which it can't — uploads live in
qdrant only, no on-disk source).

Lifecycle wrapper `KBReconciler` mirrors `KBWatcher`'s shape: `start()`
spawns a long-lived asyncio task, `stop()` cancels it. The task sleeps
for `interval_s` between sweeps. Pass `interval_s=0` for "single-shot,
exit after one run" — useful for a one-off CLI sweep.

The same `reconcile_once()` function backs both the periodic loop and
the admin endpoint `POST /v1/admin/kb/reconcile`. No code path runs
the reconcile in two places at once — the periodic task and the admin
trigger share the same qdrant client and serialize naturally through
qdrant's per-collection locking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audrey.kb.qdrant import QdrantKB

log = logging.getLogger(__name__)

# Page size for the qdrant scroll. 256 matches the value used in
# `_list_user_files_sync` — large enough to keep the round-trip count
# down, small enough to bound memory on a single page.
_SCROLL_PAGE_SIZE = 256


@dataclass(slots=True)
class ReconcileSummary:
    """Per-collection breakdown of one reconcile pass."""

    checked: int = 0           # unique source paths examined
    orphans_deleted: int = 0   # source paths whose Path.exists() was False
    points_in_orphans: int = 0 # qdrant points covered by deleted sources
    elapsed_s: float = 0.0
    error: str = ""            # non-empty on failure; the loop keeps going


@dataclass(slots=True)
class ReconcileResult:
    """Top-level summary returned to the admin endpoint and logger."""

    by_collection: dict[str, ReconcileSummary] = field(default_factory=dict)
    total_orphans_deleted: int = 0
    total_elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_collection": {
                name: {
                    "checked": s.checked,
                    "orphans_deleted": s.orphans_deleted,
                    "points_in_orphans": s.points_in_orphans,
                    "elapsed_s": round(s.elapsed_s, 3),
                    "error": s.error,
                }
                for name, s in self.by_collection.items()
            },
            "total_orphans_deleted": self.total_orphans_deleted,
            "total_elapsed_s": round(self.total_elapsed_s, 3),
        }


async def _scroll_sources(qdrant: QdrantKB, *, collection: str) -> dict[str, int]:
    """Scroll a collection and return `{source_path: point_count}`.

    Best-effort — if a point has no `payload.source` we skip it (uploaded
    files don't have one; only ingested-from-disk content does). Pagination
    via qdrant's scroll offset; runs to completion before returning.
    """
    by_source: dict[str, int] = {}
    next_page: Any = None
    client = qdrant._client

    if not await qdrant.collection_exists(collection):
        return by_source

    while True:
        # qdrant-client's scroll is sync; offload to a thread to keep the
        # event loop responsive on collections with many pages.
        points, next_page = await asyncio.to_thread(
            client.scroll,
            collection_name=collection,
            limit=_SCROLL_PAGE_SIZE,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            source = str(payload.get("source") or "")
            if not source:
                continue
            by_source[source] = by_source.get(source, 0) + 1
        if next_page is None:
            break
    return by_source


async def reconcile_collection(
    qdrant: QdrantKB,
    *,
    collection: str,
) -> ReconcileSummary:
    """Reconcile one collection. Never raises — errors land in `summary.error`."""
    summary = ReconcileSummary()
    start = time.monotonic()
    try:
        by_source = await _scroll_sources(qdrant, collection=collection)
        summary.checked = len(by_source)
        for source, point_count in by_source.items():
            # `Path.exists` is synchronous; call directly. We're not on a
            # tight latency budget — this loop runs every `interval_s`,
            # not per-request. ASYNC240 doesn't apply at the same severity
            # here as it does in the request hot path.
            if Path(source).exists():
                continue
            try:
                await qdrant.delete_by_source(source, collection=collection)
            except Exception as e:  # noqa: BLE001 — one orphan failing shouldn't kill the sweep
                log.warning("kb.reconcile: delete failed for %s in %s: %s",
                            source, collection, e)
                continue
            summary.orphans_deleted += 1
            summary.points_in_orphans += point_count
            log.info("kb.reconcile: deleted orphan %s (%d points) from %s",
                     source, point_count, collection)
    except Exception as e:  # noqa: BLE001 — bubble up via summary, not exception
        summary.error = str(e)[:300]
        log.warning("kb.reconcile: %s sweep failed: %s", collection, e)
    summary.elapsed_s = time.monotonic() - start
    return summary


async def reconcile_once(
    qdrant: QdrantKB,
    *,
    text_collection: str = "kb_text",
    image_collection: str = "kb_images",
) -> ReconcileResult:
    """Run one full reconcile pass over both global collections.

    Returns a structured summary. Per-user collections are NOT touched —
    they're managed by Phase 15's sqlite + startup reconcile.
    """
    result = ReconcileResult()
    start = time.monotonic()
    for name in (text_collection, image_collection):
        summary = await reconcile_collection(qdrant, collection=name)
        result.by_collection[name] = summary
        result.total_orphans_deleted += summary.orphans_deleted
    result.total_elapsed_s = time.monotonic() - start
    log.info(
        "kb.reconcile: pass complete; orphans_deleted=%d elapsed=%.2fs (%s)",
        result.total_orphans_deleted, result.total_elapsed_s,
        ", ".join(
            f"{n}: {s.orphans_deleted}/{s.checked}"
            for n, s in result.by_collection.items()
        ),
    )
    return result


class KBReconciler:
    """Long-lived background task that runs `reconcile_once()` every `interval_s`.

    Mirrors `KBWatcher`'s lifecycle shape — `start()` spawns the task,
    `stop()` cancels it cleanly. `interval_s=0` disables the loop (the
    admin endpoint can still call `reconcile_once()` directly).
    """

    def __init__(
        self,
        *,
        qdrant: QdrantKB,
        interval_s: float = 1800.0,
        text_collection: str = "kb_text",
        image_collection: str = "kb_images",
    ) -> None:
        self._qdrant = qdrant
        self._interval_s = max(0.0, interval_s)
        self._text_collection = text_collection
        self._image_collection = image_collection
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._interval_s <= 0:
            log.info("kb.reconcile: interval_s=0, periodic loop disabled")
            return
        self._task = asyncio.create_task(self._run(), name="kb-reconcile")
        log.info("kb.reconcile: periodic sweep every %.0fs", self._interval_s)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        # First sweep waits one full interval so we don't compete with
        # startup's qdrant init. The admin endpoint is available
        # immediately for ad-hoc sweeps if you need one sooner.
        try:
            while True:
                await asyncio.sleep(self._interval_s)
                try:
                    await reconcile_once(
                        self._qdrant,
                        text_collection=self._text_collection,
                        image_collection=self._image_collection,
                    )
                except Exception as e:  # noqa: BLE001 — loop must survive a bad sweep
                    log.warning("kb.reconcile: sweep raised: %s", e)
        except asyncio.CancelledError:
            return


__all__ = [
    "KBReconciler",
    "ReconcileResult",
    "ReconcileSummary",
    "reconcile_collection",
    "reconcile_once",
]
