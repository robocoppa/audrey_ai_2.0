"""Real local stores and HTTP contracts for unified text embeddings."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client import models as qm

from audrey.embedding_profile import QWEN_QUERY_PREFIX, EmbeddingProfile
from audrey.kb.embed import TextEmbedder
from audrey.kb.qdrant import QdrantKB
from audrey.kb.user_store import ensure_user_collections

sys.path.insert(0, str(Path(__file__).parents[1]/"tools-server"))
from chat_archive import ChatArchiveStore, _embed
from db import MemoryStore
from text_embedding import TextEmbeddingError, embed_text

PROFILE = EmbeddingProfile("qwen3-embedding:4b", "c"*64, 3, "", QWEN_QUERY_PREFIX)


def transport(seen, *, bad_vector=None, wrong_digest=False):
    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": PROFILE.model,
                "digest": "x"*64 if wrong_digest else PROFILE.digest}]})
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [bad_vector if bad_vector is not None else [1., 0., 0.]]})
    return httpx.MockTransport(handle)


async def test_sidecar_memory_documents_queries_and_owner_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr("db.AsyncQdrantClient", lambda **kw: AsyncQdrantClient(":memory:"))
    memory = MemoryStore(qdrant_url="http://unused", ollama_url="http://unused", collection="memory",
        embed_model="retired", embed_dim=768, similarity_threshold=.6, embed_timeout_s=8,
        legacy_sqlite_path=tmp_path/"missing.db", embedding_profile=PROFILE, embed_keep_alive="24h")
    await memory._qdrant.close()
    await memory._http.aclose()
    memory._qdrant = AsyncQdrantClient(":memory:")
    seen = []
    memory._http = httpx.AsyncClient(base_url="http://unused", transport=transport(seen))
    try:
        await memory.init_qdrant()
        await memory.store("desk", "oak", "user:owner furniture")
        assert len(await memory.search(user="owner", query="furniture")) == 1
        assert await memory.search(user="different", query="furniture") == []
        assert seen[0]["input"] == ["desk: oak [tags: furniture]"]
        assert seen[1]["input"] == [PROFILE.query("furniture")]
        assert all(body["truncate"] is False for body in seen)
        assert (await memory._qdrant.get_collection("memory")).config.metadata == PROFILE.metadata
    finally:
        await memory.aclose()


@pytest.mark.parametrize("vector", [[0.,0.,0.], [float("nan"),0.,0.], [1.], [True,0.,0.]])
async def test_sidecar_refuses_invalid_vectors(vector):
    async with httpx.AsyncClient(base_url="http://unused", transport=transport([],bad_vector=vector)) as http:
        with pytest.raises(TextEmbeddingError):
            await embed_text(http, PROFILE, "text")


async def test_sidecar_wrong_digest_prevents_embedding():
    seen = []
    async with httpx.AsyncClient(base_url="http://unused", transport=transport(seen,wrong_digest=True)) as http:
        with pytest.raises(TextEmbeddingError, match="digest"):
            await embed_text(http, PROFILE, "text")
    assert seen == []


async def test_archive_formats_documents_and_queries_in_same_space():
    seen = []
    async with httpx.AsyncClient(base_url="http://unused", transport=transport(seen)) as http:
        await _embed(http, PROFILE.model, "User: old chat", profile=PROFILE)
        await _embed(http, PROFILE.model, "old chat", profile=PROFILE, query=True)
    assert seen[0]["input"] == ["User: old chat"]
    assert seen[1]["input"] == [PROFILE.query("old chat")]


async def test_new_global_and_user_collections_use_profile_and_reject_old_schema(monkeypatch):
    client = QdrantClient(":memory:")
    monkeypatch.setattr("audrey.kb.qdrant.QdrantClient",lambda **kw: client)
    kb = QdrantKB("unused",6333,embedding_profile=PROFILE)
    await kb.ensure_collections()
    text, image = await ensure_user_collections(kb,"owner")
    assert client.get_collection(text).config.params.vectors.size == 3
    assert client.get_collection(text).config.metadata == PROFILE.metadata
    assert "bm25" in client.get_collection(text).config.params.sparse_vectors
    assert client.get_collection(image).config.params.vectors.size == 512
    client.delete_collection(text)
    client.create_collection(text,vectors_config=qm.VectorParams(size=768,distance=qm.Distance.COSINE))
    with pytest.raises(ValueError,match="identity"):
        await kb.search_text([1.,0.,0.],collection=text)
    with pytest.raises(ValueError):
        await ensure_user_collections(kb,"owner")
    kb.close()


async def test_backend_documents_and_query_formats_are_different():
    client = SimpleNamespace(tags=AsyncMock(return_value=[{"name":PROFILE.model,"digest":PROFILE.digest}]),
        embed=AsyncMock(return_value=[[1.,0.,0.]]))
    embedder = TextEmbedder(client,profile=PROFILE)
    await embedder.embed_many(["raw document"])
    await embedder.embed_one("question")
    assert client.embed.await_args_list[0].kwargs["texts"] == ["raw document"]
    assert client.embed.await_args_list[1].kwargs["texts"] == [PROFILE.query("question")]
    assert all(call.kwargs["truncate"] is False for call in client.embed.await_args_list)


async def test_empty_history_index_repairs_live_receipts_only(tmp_path, monkeypatch):
    monkeypatch.setattr("chat_archive.AsyncQdrantClient", lambda **kw: AsyncQdrantClient(":memory:"))
    store = ChatArchiveStore(sqlite_path=tmp_path/"archive.db",qdrant_url="http://unused",
        ollama_url="http://unused",collection="history",embed_model="retired",embed_dim=768,
        embed_timeout_s=10,chunk_max_chars=1000,chunk_overlap_chars=100,search_threshold=.62,
        retention_days=0,max_bytes=0,embedding_profile=PROFILE)
    try:
        await store.init_source()
        for ident in ("live","deleted"):
            await store._db.execute("INSERT INTO archive_chunks(chunk_id,conversation_id,user,message_ids_json,text,created_at,indexed_at) VALUES(?,?,?,?,?,?,?)",
                (ident,ident,"owner","m1","chat","2026-01-01T00:00:00+00:00","receipt"))
        await store._db.execute("INSERT INTO archive_conversation_deletions VALUES(?,?,?,?,?)",
            ("owner","deleted","2026-01-01T00:00:00+00:00","now",None))
        await store._db.commit()
        store._qdrant.count = AsyncMock(return_value=SimpleNamespace(count=0))
        await store._repair_empty_index()
        rows = await (await store._db.execute("SELECT chunk_id,indexed_at FROM archive_chunks ORDER BY chunk_id")).fetchall()
        assert rows == [("deleted","receipt"),("live",None)]
    finally:
        await store.aclose()
