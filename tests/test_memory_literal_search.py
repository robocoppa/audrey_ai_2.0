"""Literal memory keys remain retrievable when dense similarity is low."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

sys.path.insert(0, str(Path(__file__).parents[1] / "tools-server"))
from db import EmbedError, MemoryStore, _point_id


@pytest.fixture
async def memory():
    store = MemoryStore.__new__(MemoryStore)
    store._collection = "memory"
    store._qdrant = AsyncQdrantClient(":memory:")
    store._threshold = .60
    store._embed = AsyncMock(return_value=[1., 0.])
    await store._qdrant.create_collection(
        collection_name=store._collection,
        vectors_config=qm.VectorParams(size=2, distance=qm.Distance.COSINE),
    )
    try:
        yield store
    finally:
        await store._qdrant.close()


async def save(memory, *, user="owner", key="migration_test_phrase", value="blue canoe",
               updated_at="2026-10-16T12:00:00+00:00", score=.51565):
    await memory._qdrant.upsert(
        collection_name=memory._collection,
        points=[qm.PointStruct(
            id=_point_id(user, key),
            vector=[score, math.sqrt(1 - score**2)],
            payload={"user": user, "key": key, "value": value,
                     "tags": f"user:{user}", "created_at": updated_at,
                     "updated_at": updated_at},
        )],
    )


@pytest.mark.parametrize("query", [
    "migration test phrase", "migration_test_phrase", "migration-test-phrase",
    "  Migration   Test Phrase  ",
])
async def test_whole_key_spelling_finds_low_cosine_memory_without_embedding(memory, query):
    await save(memory)
    raw = await memory._qdrant.query_points(
        collection_name=memory._collection, query=[1., 0.], limit=1,
    )
    assert raw.points[0].score < memory._threshold
    memory._embed.side_effect = EmbedError("embedder must not be needed for a literal key")

    hits = await memory.search(user="owner", query=query)

    assert [(hit.key, hit.value) for hit in hits] == [("migration_test_phrase", "blue canoe")]
    memory._embed.assert_not_awaited()


async def test_literal_keys_remain_owner_scoped(memory):
    await save(memory, user="other")
    assert await memory.search(user="owner", query="migration test phrase") == []
    memory._embed.assert_awaited_once()


async def test_purge_cutoff_hides_old_literal_key_but_allows_new_activity(memory):
    await save(memory, updated_at="2026-10-16T10:00:00+00:00")
    cutoff = "2026-10-16T11:00:00+00:00"
    assert await memory.search(user="owner", query="migration test phrase", visible_after=cutoff) == []
    memory._embed.assert_awaited_once()
    memory._embed.reset_mock()
    await save(memory, updated_at="2026-10-16T12:00:00+00:00")
    assert (await memory.search(user="owner", query="migration test phrase", visible_after=cutoff))[0].value == "blue canoe"
    memory._embed.assert_not_awaited()


@pytest.mark.parametrize("query", ["migration", "test phrase", "unrelated topic"])
async def test_partial_or_unrelated_query_still_obeys_dense_threshold(memory, query):
    await save(memory)
    assert await memory.search(user="owner", query=query) == []
    memory._embed.assert_awaited_once_with(query)


async def test_dense_semantic_match_and_embed_error_behavior_remain(memory):
    await save(memory, score=.8)
    assert (await memory.search(user="owner", query="what phrase did I save"))[0].value == "blue canoe"
    memory._embed.side_effect = EmbedError("unavailable")
    with pytest.raises(EmbedError, match="unavailable"):
        await memory.search(user="owner", query="another topic")


async def test_literal_spellings_are_deduplicated_ordered_and_bounded(memory):
    await save(memory, key="migration test phrase", value="exact spelling")
    await save(memory)
    hits = await memory.search(user="owner", query="migration test phrase", top_k=1)
    assert [(hit.key, hit.value) for hit in hits] == [("migration test phrase", "exact spelling")]
    memory._embed.assert_not_awaited()
    hits = await memory.search(user="owner", query="migration test phrase", top_k=5)
    assert [hit.key for hit in hits] == ["migration test phrase", "migration_test_phrase"]
