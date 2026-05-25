"""Tests for the KB reconcile pass (Phase 30).

Stubs the qdrant client surface so tests run offline. Uses pytest's
`tmp_path` fixture so `Path.exists()` is real — that's the function
we're testing the *reaction to*; mocking it would just be testing the
mock.

The reconciler logic:
  - scroll all points in kb_text + kb_images, group by payload.source
  - for each unique source, check Path.exists()
  - if absent, call delete_by_source(source, collection=...)
  - return structured summary
"""

from collections import defaultdict
from pathlib import Path

from audrey.kb.reconcile import (
    KBReconciler,
    reconcile_once,
)

# ─── Fakes ─────────────────────────────────────────────────────────────

class _FakeQdrantKB:
    """Stand-in for `QdrantKB`. Only the methods reconcile uses.

    Models the public facade: `collection_exists`, `scroll_collection`,
    `delete_by_source`. Reconcile shouldn't reach into qdrant-client
    primitives, so the fake doesn't expose any either.
    """

    def __init__(
        self,
        points_by_collection: dict[str, list[dict]],
        existing_collections: set[str] | None = None,
    ):
        self._points = points_by_collection
        self._existing = (
            existing_collections
            if existing_collections is not None
            else set(points_by_collection.keys())
        )
        self.deletes: list[tuple[str, str]] = []  # (collection, source)

    async def collection_exists(self, name: str) -> bool:
        return name in self._existing

    async def scroll_collection(
        self, collection: str, *, page_size: int = 256,
    ) -> list[tuple[str, dict]]:
        if collection not in self._existing:
            return []
        # Return `(point_id, payload)` tuples — same shape the real method
        # returns; payloads are the dicts passed in via `_make_points`.
        return [(f"pt-{i}", p) for i, p in enumerate(self._points.get(collection, []))]

    async def delete_by_source(self, source: str, *, collection: str) -> None:
        self.deletes.append((collection, source))


def _make_points(*sources_with_counts: tuple[str, int]) -> list[dict]:
    """Build a flat list of point payloads from `(source, n)` tuples."""
    out: list[dict] = []
    for source, n in sources_with_counts:
        for i in range(n):
            out.append({"source": source, "kind": "text", "chunk_idx": i})
    return out


# ─── reconcile_once ────────────────────────────────────────────────────

async def test_reconcile_no_orphans(tmp_path: Path):
    # All sources exist on disk → zero deletes.
    real_file = tmp_path / "geology.md"
    real_file.write_text("hello")
    qdrant = _FakeQdrantKB({
        "kb_text": _make_points((str(real_file), 3)),
        "kb_images": [],
    })

    result = await reconcile_once(qdrant)

    assert result.total_orphans_deleted == 0
    assert qdrant.deletes == []
    assert result.by_collection["kb_text"].checked == 1
    assert result.by_collection["kb_text"].orphans_deleted == 0
    assert result.by_collection["kb_images"].checked == 0


async def test_reconcile_deletes_orphan(tmp_path: Path):
    # File was ingested, then deleted while watcher was off → vectors
    # remain in qdrant but Path.exists() is False → must be deleted.
    real_file = tmp_path / "exists.md"
    real_file.write_text("present")
    ghost_file = tmp_path / "deleted-on-disk.md"
    # Don't create ghost_file.

    qdrant = _FakeQdrantKB({
        "kb_text": _make_points(
            (str(real_file), 2),
            (str(ghost_file), 5),
        ),
        "kb_images": [],
    })

    result = await reconcile_once(qdrant)

    assert result.total_orphans_deleted == 1
    assert qdrant.deletes == [("kb_text", str(ghost_file))]
    summary = result.by_collection["kb_text"]
    assert summary.checked == 2
    assert summary.orphans_deleted == 1
    assert summary.points_in_orphans == 5  # 5 points covered by the orphan source


async def test_reconcile_handles_both_collections(tmp_path: Path):
    real_text = tmp_path / "doc.md"
    real_text.write_text("x")
    real_img = tmp_path / "rock.jpg"
    real_img.write_bytes(b"\x00")
    ghost_text = tmp_path / "ghost.md"
    ghost_img = tmp_path / "ghost.jpg"

    qdrant = _FakeQdrantKB({
        "kb_text": _make_points((str(real_text), 1), (str(ghost_text), 1)),
        "kb_images": _make_points((str(real_img), 1), (str(ghost_img), 1)),
    })

    result = await reconcile_once(qdrant)

    assert result.total_orphans_deleted == 2
    deleted = set(qdrant.deletes)
    assert ("kb_text", str(ghost_text)) in deleted
    assert ("kb_images", str(ghost_img)) in deleted


async def test_reconcile_skips_points_with_no_source(tmp_path: Path):
    # Per-user uploads have no `payload.source` (their content lives only
    # in qdrant). Reconcile must skip them rather than crash.
    qdrant = _FakeQdrantKB({
        "kb_text": [
            {"kind": "text"},  # no source field
            {"source": "", "kind": "text"},  # empty source
        ],
        "kb_images": [],
    })

    result = await reconcile_once(qdrant)

    assert result.total_orphans_deleted == 0
    assert qdrant.deletes == []
    assert result.by_collection["kb_text"].checked == 0


async def test_reconcile_handles_missing_collection_silently(tmp_path: Path):
    # Fresh install: kb_images may not exist yet. Reconcile should not
    # raise; the collection_exists check guards the scroll call.
    real_file = tmp_path / "doc.md"
    real_file.write_text("x")
    qdrant = _FakeQdrantKB(
        points_by_collection={"kb_text": _make_points((str(real_file), 1))},
        existing_collections={"kb_text"},  # kb_images deliberately missing
    )

    result = await reconcile_once(qdrant)

    assert result.total_orphans_deleted == 0
    assert result.by_collection["kb_images"].checked == 0
    assert result.by_collection["kb_images"].error == ""


async def test_reconcile_collapses_duplicate_sources(tmp_path: Path):
    # Multiple chunks share one source path. The reconciler should call
    # delete_by_source ONCE per unique source, not once per point.
    ghost = tmp_path / "ghost.md"
    qdrant = _FakeQdrantKB({
        "kb_text": _make_points((str(ghost), 12)),  # 12 chunks of one ghost file
        "kb_images": [],
    })

    result = await reconcile_once(qdrant)

    assert result.total_orphans_deleted == 1
    # delete_by_source called ONCE for the unique source.
    by_source: dict[str, int] = defaultdict(int)
    for _coll, src in qdrant.deletes:
        by_source[src] += 1
    assert by_source[str(ghost)] == 1
    assert result.by_collection["kb_text"].points_in_orphans == 12


async def test_reconcile_summary_to_dict_shape(tmp_path: Path):
    # Lock down the response shape returned to the admin endpoint.
    real_file = tmp_path / "x.md"
    real_file.write_text("x")
    qdrant = _FakeQdrantKB({
        "kb_text": _make_points((str(real_file), 1)),
        "kb_images": [],
    })

    result = await reconcile_once(qdrant)
    payload = result.to_dict()

    assert set(payload.keys()) == {"by_collection", "total_orphans_deleted", "total_elapsed_s"}
    assert set(payload["by_collection"].keys()) == {"kb_text", "kb_images"}
    for s in payload["by_collection"].values():
        assert set(s.keys()) == {
            "checked", "orphans_deleted", "points_in_orphans", "elapsed_s", "error",
        }


# ─── KBReconciler lifecycle ────────────────────────────────────────────

async def test_reconciler_disabled_when_interval_zero(tmp_path: Path):
    # `interval_s=0` is the documented "disable the loop" knob. The
    # admin endpoint can still call reconcile_once() directly.
    qdrant = _FakeQdrantKB({"kb_text": [], "kb_images": []})
    rec = KBReconciler(qdrant=qdrant, interval_s=0)

    await rec.start()
    assert rec._task is None  # type: ignore[attr-defined]

    # stop() with no task is a no-op — must not raise.
    await rec.stop()


async def test_reconciler_clean_shutdown_on_stop(tmp_path: Path):
    # start → stop without crashing or leaking the task. The interval is
    # huge so the task spends its whole life in `await asyncio.sleep`
    # *after* the startup sweep returns.
    qdrant = _FakeQdrantKB({"kb_text": [], "kb_images": []})
    rec = KBReconciler(qdrant=qdrant, interval_s=3600)

    await rec.start()
    assert rec._task is not None  # type: ignore[attr-defined]
    assert not rec._task.done()  # type: ignore[attr-defined]

    await rec.stop()
    assert rec._task is None  # type: ignore[attr-defined]


async def test_reconciler_runs_one_sweep_immediately_at_startup(tmp_path: Path):
    # Regression for the "first sweep waits one full interval" finding.
    # Pre-fix: with `interval_s=1800` (default) the first sweep didn't
    # fire until 30 minutes after boot, leaving stale state to accumulate
    # if `KB_WATCHER_ENABLED=0` for a stretch beforehand.
    # Post-fix: one sweep runs immediately, *then* the periodic cadence
    # takes over.
    #
    # We count scroll_collection calls — a sweep scrolls both collections,
    # so the first sweep should produce 2 calls. We give it a long
    # interval so the periodic loop hasn't fired yet by the time we
    # inspect, and use asyncio.wait_for + an event to know the sweep
    # actually ran.
    import asyncio as _asyncio

    sweep_done = _asyncio.Event()

    class _SignalingQdrantKB(_FakeQdrantKB):
        def __init__(self) -> None:
            super().__init__({"kb_text": [], "kb_images": []})
            self.scroll_calls: list[str] = []

        async def scroll_collection(
            self, collection: str, *, page_size: int = 256,
        ) -> list[tuple[str, dict]]:
            self.scroll_calls.append(collection)
            # Signal once both global collections have been scrolled.
            if len(self.scroll_calls) >= 2:
                sweep_done.set()
            return await super().scroll_collection(collection, page_size=page_size)

    qdrant = _SignalingQdrantKB()
    rec = KBReconciler(qdrant=qdrant, interval_s=3600)

    await rec.start()
    try:
        # If the startup sweep didn't run, this will time out — that's
        # the pre-fix behavior we're guarding against.
        await _asyncio.wait_for(sweep_done.wait(), timeout=2.0)
        assert qdrant.scroll_calls == ["kb_text", "kb_images"]
    finally:
        await rec.stop()
