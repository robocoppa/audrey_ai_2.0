"""Tests for `QdrantKB._ensure_user_indexes_sync` exception handling.

The method is called from per-user upload setup and must be idempotent:
re-running over an already-indexed collection should not raise. Before the
narrowing pass it caught `Exception` blindly, which silently swallowed
real failures (5xx, schema mismatch). The current contract:

  - 4xx "already exists" → log at DEBUG, continue.
  - Anything else (5xx, transport errors, unrelated 4xx) → propagate.

Tests stub `_client.create_payload_index` to raise the relevant variants.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from audrey.kb import qdrant as qdrant_mod
from audrey.kb.qdrant import QdrantKB


def _make_kb(monkeypatch) -> tuple[QdrantKB, MagicMock]:
    """Construct a QdrantKB whose internal client is a MagicMock."""
    fake_client = MagicMock()
    monkeypatch.setattr(qdrant_mod, "QdrantClient", lambda **_: fake_client)
    kb = QdrantKB(host="x", port=0)
    return kb, fake_client


def _unexpected(status: int, body: bytes) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status,
        reason_phrase="",
        content=body,
        headers=httpx.Headers({}),
    )


def test_ensure_user_indexes_swallows_already_exists_4xx(monkeypatch):
    # Qdrant's idempotent path: index already present → 4xx with "exists"
    # in the body. We log at DEBUG and continue to the next field.
    kb, client = _make_kb(monkeypatch)
    client.create_payload_index.side_effect = _unexpected(
        409, b'{"status":{"error":"index already exists"}}'
    )

    kb._ensure_user_indexes_sync("kb_user_text_alice")

    # Both fields ("user", "file_id") still attempted despite the first raising.
    assert client.create_payload_index.call_count == 2


def test_ensure_user_indexes_propagates_5xx(monkeypatch):
    # 5xx is a real server failure, not an idempotency signal. Must surface
    # so the upload route 503s instead of silently writing un-indexed data.
    kb, client = _make_kb(monkeypatch)
    client.create_payload_index.side_effect = _unexpected(
        503, b"service unavailable"
    )

    with pytest.raises(UnexpectedResponse):
        kb._ensure_user_indexes_sync("kb_user_text_alice")


def test_ensure_user_indexes_propagates_unrelated_4xx(monkeypatch):
    # 4xx whose body doesn't mention "exist" — likely a schema or validation
    # error. Must surface; we'd rather see the failure than ship a collection
    # missing its indexes.
    kb, client = _make_kb(monkeypatch)
    client.create_payload_index.side_effect = _unexpected(
        400, b'{"status":{"error":"invalid field schema"}}'
    )

    with pytest.raises(UnexpectedResponse):
        kb._ensure_user_indexes_sync("kb_user_text_alice")


def test_ensure_user_indexes_propagates_non_qdrant_exception(monkeypatch):
    # Transport errors (connection refused, timeout) bubble up as plain
    # exceptions, not UnexpectedResponse. They must surface unchanged.
    kb, client = _make_kb(monkeypatch)
    client.create_payload_index.side_effect = ConnectionError("no qdrant")

    with pytest.raises(ConnectionError):
        kb._ensure_user_indexes_sync("kb_user_text_alice")


# ─── build_*_point extras-clobber guard ──────────────────────────────────────

@pytest.mark.parametrize("reserved_key", ["source", "kind", "text", "chunk_idx", "mtime"])
def test_build_text_point_rejects_extras_that_clobber_reserved_keys(reserved_key):
    # The base payload pins the deterministic fields (source, kind, chunk_idx,
    # mtime). An `extras` dict that includes one of those would silently
    # override it after the .update() — which would corrupt the search-merge
    # invariants. The guard turns that into a loud ValueError instead.
    from audrey.kb.qdrant import build_text_point

    with pytest.raises(ValueError, match="reserved payload keys"):
        build_text_point(
            source="/x.md", chunk_idx=0, text="hello",
            vector=[0.1] * 768, mtime=0.0,
            extra={reserved_key: "evil"},
        )


def test_build_image_point_rejects_reserved_caption_override():
    from audrey.kb.qdrant import build_image_point

    with pytest.raises(ValueError, match="reserved payload keys"):
        build_image_point(
            source="/x.jpg", chunk_idx=0, caption="ok",
            vector=[0.1] * 512, mtime=0.0,
            extra={"caption": "shadow"},
        )


def test_build_text_point_allows_non_reserved_extras():
    # The user-upload path passes `user`, `file_id`, `filename`, `mime`,
    # `bytes`, `uploaded_at` in extras — none of which are reserved. They
    # must still pass through into the final payload.
    from audrey.kb.qdrant import build_text_point

    pt = build_text_point(
        source="/x.md", chunk_idx=0, text="hello",
        vector=[0.1] * 768, mtime=0.0,
        extra={"user": "alice@example.com", "file_id": "uuid", "filename": "x.md"},
    )
    assert pt.payload["user"] == "alice@example.com"
    assert pt.payload["file_id"] == "uuid"
    assert pt.payload["filename"] == "x.md"
    assert pt.payload["source"] == "/x.md"  # untouched


# ─── list_collections + scroll_collection facade methods ────────────────

class _FakeRecord:
    def __init__(self, point_id: str, payload: dict):
        self.id = point_id
        self.payload = payload


class _FakeCollDescriptor:
    def __init__(self, name: str):
        self.name = name


class _FakeCollectionsResponse:
    def __init__(self, names: list[str]):
        self.collections = [_FakeCollDescriptor(n) for n in names]


async def test_list_collections_returns_every_collection_name(monkeypatch):
    kb, client = _make_kb(monkeypatch)
    client.get_collections.return_value = _FakeCollectionsResponse(
        ["kb_text", "kb_images", "kb_user_text_alice_example_com"],
    )

    out = await kb.list_collections()

    assert out == ["kb_text", "kb_images", "kb_user_text_alice_example_com"]


async def test_scroll_collection_returns_empty_when_collection_missing(monkeypatch):
    # The facade's collection-existence check fires before scroll; missing
    # collections return [] rather than raising.
    kb, client = _make_kb(monkeypatch)
    client.collection_exists.return_value = False

    out = await kb.scroll_collection("does_not_exist")

    assert out == []
    client.scroll.assert_not_called()


async def test_scroll_collection_walks_pages_and_returns_id_payload_tuples(monkeypatch):
    kb, client = _make_kb(monkeypatch)
    client.collection_exists.return_value = True

    # Two pages: first returns next_page="cursor", second returns None.
    page_one = ([_FakeRecord("pt-1", {"source": "/a.md"})], "cursor")
    page_two = ([_FakeRecord("pt-2", {"source": "/b.md"})], None)
    client.scroll.side_effect = [page_one, page_two]

    out = await kb.scroll_collection("kb_text", page_size=1)

    assert out == [
        ("pt-1", {"source": "/a.md"}),
        ("pt-2", {"source": "/b.md"}),
    ]
    assert client.scroll.call_count == 2


async def test_scroll_collection_normalizes_missing_payload_to_empty_dict(monkeypatch):
    # Qdrant can return a point with payload=None; we materialize {} so
    # callers don't have to .get with a default.
    kb, client = _make_kb(monkeypatch)
    client.collection_exists.return_value = True
    record_with_none_payload = MagicMock()
    record_with_none_payload.id = "pt-3"
    record_with_none_payload.payload = None
    client.scroll.return_value = ([record_with_none_payload], None)

    out = await kb.scroll_collection("kb_text")

    assert out == [("pt-3", {})]
