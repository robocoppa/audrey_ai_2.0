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
