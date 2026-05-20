"""Tests for KBWatcher's event-filtering and delete-collection wiring.

The watcher's thread-bridge + debounce loop is awkward to drive end-to-end
in unit tests (real watchdog observer, real filesystem events, real
asyncio task lifecycle). These tests cover the directly-testable
surfaces: the event filter that decides which paths enter the queue,
and the `_delete_vectors` call that has to honor `QdrantKB`'s
configured collection names.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audrey.kb.watcher import KBWatcher, _QueueHandler


def _make_event(path: str, *, is_directory: bool = False, dest_path: str | None = None) -> MagicMock:
    ev = MagicMock()
    ev.src_path = path
    ev.is_directory = is_directory
    if dest_path is not None:
        ev.dest_path = dest_path
    return ev


def _drain(queue: asyncio.Queue) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


@pytest.mark.asyncio
async def test_enqueue_admits_supported_text_file():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = _QueueHandler(loop, queue)

    handler.on_modified(_make_event("/datasets/topic/notes.md"))

    # call_soon_threadsafe is scheduled on the loop; yield once.
    await asyncio.sleep(0)
    items = _drain(queue)
    assert items == [("ingest", Path("/datasets/topic/notes.md"))]


@pytest.mark.asyncio
async def test_enqueue_skips_unsupported_suffix():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = _QueueHandler(loop, queue)

    handler.on_modified(_make_event("/datasets/topic/notes.zip"))

    await asyncio.sleep(0)
    assert _drain(queue) == []


@pytest.mark.asyncio
async def test_enqueue_skips_dotfiles_at_any_depth():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = _QueueHandler(loop, queue)

    handler.on_modified(_make_event("/datasets/topic/.hidden.md"))
    handler.on_created(_make_event("/datasets/.also_hidden.md"))

    await asyncio.sleep(0)
    assert _drain(queue) == []


@pytest.mark.asyncio
async def test_enqueue_skips_files_inside_dot_directory():
    # Regression test for the same bug class fixed in `_iter_files`:
    # a stray `.git/HEAD` or `.cache/tmp.md` under a watched root must
    # NOT enter the queue just because the leaf filename is non-dot.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = _QueueHandler(loop, queue)

    handler.on_modified(_make_event("/datasets/topic/.git/objects/some.md"))
    handler.on_modified(_make_event("/datasets/topic/.cache/tmp.md"))
    handler.on_modified(_make_event("/.config/notes.md"))
    # Sanity: the non-dot sibling still admits.
    handler.on_modified(_make_event("/datasets/topic/real.md"))

    await asyncio.sleep(0)
    items = _drain(queue)
    assert items == [("ingest", Path("/datasets/topic/real.md"))]


@pytest.mark.asyncio
async def test_enqueue_skips_directory_events():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = _QueueHandler(loop, queue)

    handler.on_modified(_make_event("/datasets/topic", is_directory=True))

    await asyncio.sleep(0)
    assert _drain(queue) == []


@pytest.mark.asyncio
async def test_on_moved_enqueues_delete_then_ingest():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = _QueueHandler(loop, queue)

    handler.on_moved(_make_event(
        "/datasets/topic/old.md", dest_path="/datasets/topic/new.md",
    ))

    await asyncio.sleep(0)
    items = _drain(queue)
    assert items == [
        ("delete", Path("/datasets/topic/old.md")),
        ("ingest", Path("/datasets/topic/new.md")),
    ]


# ─── _delete_vectors honors QdrantKB's configured collection names ───

class _FakeQdrant:
    """Stand-in that exposes the collection-name attributes plus a
    delete_by_source spy."""

    def __init__(self, text: str, image: str) -> None:
        self.text_collection = text
        self.image_collection = image
        self.deletes: list[tuple[str, str]] = []  # (source, collection)

    async def delete_by_source(self, source: str, *, collection: str) -> None:
        self.deletes.append((source, collection))


@pytest.mark.asyncio
async def test_delete_vectors_uses_qdrant_supplied_collection_names():
    # Regression for the bug where the watcher hardcoded "kb_text" /
    # "kb_images" instead of using qdrant.text_collection /
    # .image_collection. A deployment that renames collections via
    # config would otherwise have deletes silently miss the actual
    # collections.
    qdrant = _FakeQdrant(text="custom_text", image="custom_images")
    watcher = KBWatcher(
        roots=[], qdrant=qdrant,  # type: ignore[arg-type]
        text_embedder=MagicMock(), image_embedder=MagicMock(),
    )

    await watcher._delete_vectors(Path("/datasets/topic/x.md"))

    assert qdrant.deletes == [
        ("/datasets/topic/x.md", "custom_text"),
        ("/datasets/topic/x.md", "custom_images"),
    ]


@pytest.mark.asyncio
async def test_delete_vectors_skips_image_collection_when_no_image_embedder():
    qdrant = _FakeQdrant(text="kb_text", image="kb_images")
    watcher = KBWatcher(
        roots=[], qdrant=qdrant,  # type: ignore[arg-type]
        text_embedder=MagicMock(), image_embedder=None,
    )

    await watcher._delete_vectors(Path("/datasets/topic/x.md"))

    assert qdrant.deletes == [("/datasets/topic/x.md", "kb_text")]
