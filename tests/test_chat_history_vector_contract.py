"""History retrieval against real SQLite and a persistent local Qdrant index.

Only embedding HTTP is mocked. Fixed vectors test filtering and score handling;
they do not measure a model's semantic understanding.
"""
from __future__ import annotations

import json
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from audrey.embedding_profile import QWEN_QUERY_PREFIX, EmbeddingProfile

sys.path.insert(0, str(Path(__file__).parents[1] / "tools-server"))
import chat_archive

PROFILE = EmbeddingProfile("qwen3-embedding:4b", "c" * 64, 3, "", QWEN_QUERY_PREFIX)


@pytest.fixture
def persisted_archive(tmp_path, monkeypatch):
    real_qdrant = AsyncQdrantClient
    monkeypatch.setattr(
        chat_archive, "AsyncQdrantClient",
        lambda **_kwargs: real_qdrant(path=str(tmp_path / "vectors")),
    )

    def embed_http(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{
                "name": PROFILE.model, "digest": PROFILE.digest,
            }]})
        body = json.loads(request.content)
        assert body["truncate"] is False
        text = body["input"][0]
        if text.startswith(PROFILE.query_prefix):
            vector = [1.0, 0.0, 0.0]
        else:
            score = (0.43342 if "1250" in text else
                     0.8 if "Estimate" in text else
                     0.4 if "sunflowers" in text else 0.99)
            vector = [score, math.sqrt(1 - score * score), 0.0]
        return httpx.Response(200, json={"embeddings": [vector]})

    @asynccontextmanager
    async def open_archive():
        store = chat_archive.ChatArchiveStore(
            sqlite_path=tmp_path / "archive.sqlite",
            qdrant_url="http://unused.invalid", ollama_url="http://unused.invalid",
            collection="history", embed_model=PROFILE.model, embed_dim=3,
            embed_timeout_s=1, chunk_max_chars=1000, chunk_overlap_chars=100,
            search_threshold=0.62, retention_days=0, max_bytes=0,
            embedding_profile=PROFILE,
        )
        try:
            await store._http.aclose()
            store._http = httpx.AsyncClient(
                base_url="http://unused.invalid", transport=httpx.MockTransport(embed_http),
            )
            await store.init()
            yield store
        finally:
            await store.aclose()

    return open_archive


async def test_real_history_filters_scores_literal_fallback_and_reopen(persisted_archive):
    for reopened in (False, True):
        async with persisted_archive() as store:
            if not reopened:
                for user, conversation, question, answer in [
                    ("owner", "calculation", "Calculate 1250 × 1.08", "1350"),
                    ("owner", "semantic", "Estimate the order total", "ORDER_TOTAL"),
                    ("owner", "below-cutoff", "Plant sunflowers", "GARDEN"),
                    ("different-owner", "private", "A confidential budget", "PRIVATE_OTHER_OWNER"),
                ]:
                    await store.archive_turn(
                        user=user, conversation_id=conversation,
                        user_content=question, assistant_content=answer,
                        archive_id=conversation,
                    )

            # The real Qdrant engine applies the owner filter and .62 cutoff.
            # The .99 foreign point and .4/.43342 owner points must stay out.
            dense = await store.search(user="owner", query="prior arithmetic", limit=5)
            assert [hit.conversation_id for hit in dense] == ["semantic"]
            assert dense[0].score == pytest.approx(0.8, abs=1e-6)
            assert dense[0].match_type == "semantic"

            # The below-cutoff calculation is still retrievable by its words.
            combined = await store.search(user="owner", query="calculate", limit=5)
            assert {hit.conversation_id for hit in combined} == {"calculation", "semantic"}
            calculation = next(hit for hit in combined if hit.conversation_id == "calculation")
            assert "1250" in calculation.snippet
            assert calculation.match_type == "keyword" and calculation.score is None
            assert len({hit.chunk_id for hit in combined}) == len(combined)
            assert all("PRIVATE_OTHER_OWNER" not in hit.snippet for hit in combined)

            # A genuinely absent owner never receives another owner's points.
            assert await store.search(user="missing-owner", query="calculate", limit=5) == []



async def test_semantic_history_uses_source_text_over_stale_vector_payload(persisted_archive):
    async with persisted_archive() as store:
        created_at = "2026-10-15T12:00:00+00:00"
        await store.archive_turn(
            user="owner", conversation_id="source-conversation",
            user_content="Estimate the order total", assistant_content="AUTHORITATIVE_TOTAL",
            archive_id="source-turn", created_at=created_at,
        )
        points, _ = await store._qdrant.scroll(
            collection_name=store._collection, limit=1, with_payload=True,
        )
        assert len(points) == 1
        await store._qdrant.set_payload(
            collection_name=store._collection, points=[points[0].id],
            payload={"text": "STALE_VECTOR_CONTENT", "conversation_id": "stale-conversation",
                     "created_at": "2099-01-01T00:00:00+00:00"},
        )

        hits = await store.search(user="owner", query="prior arithmetic", limit=5)

        assert len(hits) == 1
        assert hits[0].conversation_id == "source-conversation"
        assert hits[0].created_at == created_at
        assert hits[0].snippet == "User: Estimate the order total\nAssistant: AUTHORITATIVE_TOTAL"
        assert hits[0].match_type == "semantic"
        assert hits[0].score == pytest.approx(.8, abs=1e-6)
        assert "STALE_VECTOR_CONTENT" not in hits[0].snippet


@pytest.mark.parametrize("gate", ["conversation_delete", "purge"])
async def test_semantic_history_rechecks_visibility_after_real_vector_query(
    persisted_archive, monkeypatch, gate,
):
    async with persisted_archive() as store:
        await store.archive_turn(
            user="owner", conversation_id="removed-during-query",
            user_content="Estimate the order total", assistant_content="HIDDEN_TOTAL",
            archive_id="removed-turn", created_at="2026-10-15T12:00:00+00:00",
        )
        real_query = store._qdrant.query_points
        queried_points = []

        async def query_then_hide(**kwargs):
            result = await real_query(**kwargs)
            queried_points.extend(result.points)
            assert len(result.points) == 1
            if gate == "conversation_delete":
                deletion = await store.request_conversation_deletion(
                    user="owner", conversation_id="removed-during-query",
                )
                assert deletion["chunks_queued"] == 1
            else:
                # Install just the authoritative purge receipt. The real
                # vector remains present, so only the final source gate can
                # prevent this already-fetched semantic hit from escaping.
                cutoff = chat_archive._now_iso()
                async with store._db_lock:
                    await store._db.execute(
                        "INSERT INTO archive_user_purges "
                        "(purge_id, user, cutoff_at, requested_at) VALUES (?, ?, ?, ?)",
                        ("during-query", "owner", cutoff, cutoff),
                    )
                    await store._db.commit()
            return result

        monkeypatch.setattr(store._qdrant, "query_points", query_then_hide)
        hits = await store.search(user="owner", query="prior arithmetic", limit=5)

        assert len(queried_points) == 1
        assert queried_points[0].score == pytest.approx(.8, abs=1e-6)
        assert hits == []
        retained = await real_query(
            collection_name=store._collection, query=[1., 0., 0.], limit=5,
        )
        assert len(retained.points) == 1
