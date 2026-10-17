"""Independent history profile, application wiring and actual retrieval boundary.

Fixed HTTP vectors check transport and index contracts, not model quality.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from starlette.datastructures import State

from audrey.embedding_profile import load_profile, load_settings
from audrey.tools.discovery import TOOL_DECLARATIONS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools-server"))
import app as tools_app  # noqa: E402
import chat_archive  # noqa: E402
from capabilities import CapabilityRegistry  # noqa: E402
from text_embedding import TextEmbeddingError  # noqa: E402

CONFIG = ROOT / "config.yaml"
NOMIC = load_profile(CONFIG, role="history")


def test_real_config_pins_two_distinct_embedding_spaces():
    text = load_profile(CONFIG)
    history = load_settings(CONFIG, role="history")
    assert text.model == "qwen3-embedding:4b" and text.dimensions == 2560
    assert NOMIC.model == "nomic-embed-text:latest" and NOMIC.dimensions == 768
    assert NOMIC.document("chunk") == "search_document: chunk"
    assert NOMIC.query("question") == "search_query: question"
    assert NOMIC.identity != text.identity
    assert history["min_score"] == .4
    assert load_settings(CONFIG)["memory_min_score"] == .6


def test_missing_history_profile_fails_instead_of_inheriting_text(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"text_embedding": load_settings(CONFIG)}))
    assert load_profile(path).model == "qwen3-embedding:4b"
    with pytest.raises(ValueError, match="history_embedding"):
        load_profile(path, role="history")
    with pytest.raises(ValueError, match="role"):
        load_settings(path, role="misspelled")


@pytest.mark.parametrize("key,value", [
    ("min_score", True), ("min_score", float("nan")), ("min_score", 1.1),
    ("timeout_s", 0), ("timeout_s", float("inf")), ("timeout_s", "8"),
    ("keep_alive", 24),
])
def test_invalid_history_policy_is_refused(tmp_path, key, value):
    path = tmp_path / "config.yaml"
    config = {"history_embedding": {**load_settings(CONFIG, role="history"), key: value}}
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="history embedding"):
        load_settings(path, role="history")


async def test_lifespan_constructs_memory_and_history_from_their_own_profiles(monkeypatch):
    created = {}
    monkeypatch.setattr(tools_app.app, "state", State())

    def store(name):
        def create(**kwargs):
            created[name] = kwargs
            return SimpleNamespace(aclose=AsyncMock())
        return create

    monkeypatch.setattr(tools_app, "MemoryStore", store("memory"))
    monkeypatch.setattr(tools_app, "ChatArchiveStore", store("history"))
    monkeypatch.setattr(tools_app.settings, "embedding_config_path", CONFIG)
    monkeypatch.setattr(tools_app.settings, "brave_api_key", "test-placeholder")
    # Retired env-backed fields cannot select a different runtime history space.
    monkeypatch.setattr(tools_app.settings, "memory_embed_model", "wrong-memory")
    monkeypatch.setattr(tools_app.settings, "chat_archive_search_threshold", .99)
    monkeypatch.setattr(tools_app, "CapabilitySupervisor", lambda **_: SimpleNamespace(
        start=AsyncMock(), stop=AsyncMock(),
    ))
    monkeypatch.setattr(tools_app, "MemoryBackupScheduler", lambda *_, **__: SimpleNamespace(
        start=lambda: None, stop=AsyncMock(),
    ))
    async with tools_app.lifespan(tools_app.app):
        memory, history = created["memory"], created["history"]
        assert memory["embedding_profile"] == load_profile(CONFIG)
        assert memory["similarity_threshold"] == .6
        assert history["embedding_profile"] == NOMIC
        assert history["embed_model"] == NOMIC.model
        assert history["embed_dim"] == 768
        assert history["search_threshold"] == .4
        assert history["embed_timeout_s"] == 8
        assert history["embed_keep_alive"] == "24h"


def test_history_route_and_discovery_require_history_readiness_only(monkeypatch):
    state = CapabilityRegistry.all_available()
    state.set_unavailable("text_embedding", "warmup_failed")
    monkeypatch.setattr(tools_app.app.state, "capabilities", state)
    archive = object()
    monkeypatch.setattr(tools_app.app.state, "chat_archive", archive, raising=False)
    assert tools_app._searchable_archive() is archive
    deps = TOOL_DECLARATIONS["chat_history_search"].dependencies
    assert "history_embedding" in deps and "text_embedding" not in deps
    assert "text_embedding" in TOOL_DECLARATIONS["memory_search"].dependencies
    assert "history_embedding" not in TOOL_DECLARATIONS["kb_search"].dependencies
    state.set_unavailable("history_embedding", "warmup_failed")
    with pytest.raises(HTTPException) as error:
        tools_app._searchable_archive()
    assert error.value.status_code == 503
    assert error.value.detail["components"] == ["history_embedding"]


async def test_nomic_source_query_cutoff_and_owner_date_delete_gates(tmp_path, monkeypatch):
    real_qdrant = AsyncQdrantClient
    monkeypatch.setattr(chat_archive, "AsyncQdrantClient", lambda **_: real_qdrant(":memory:"))
    seen = []

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": NOMIC.model, "digest": NOMIC.digest}]})
        body = json.loads(request.content)
        seen.append(body)
        text = body["input"][0]
        assert body["model"] == NOMIC.model and body["truncate"] is False
        if text.startswith(NOMIC.query_prefix):
            vector = [1., 0.] + [0.] * 766
        else:
            assert text.startswith(NOMIC.document_prefix)
            score = .5788 if "Evaluate" in text else .2 if "orchard" in text else .99
            vector = [score, math.sqrt(1 - score * score)] + [0.] * 766
        return httpx.Response(200, json={"embeddings": [vector]})

    store = chat_archive.ChatArchiveStore(
        sqlite_path=tmp_path / "source.sqlite", qdrant_url="http://unused.invalid",
        ollama_url="http://unused.invalid", collection="history", embed_model="retired",
        embed_dim=2560, embedding_profile=NOMIC, embed_timeout_s=8,
        chunk_max_chars=2500, chunk_overlap_chars=100, search_threshold=.4,
        retention_days=0, max_bytes=0, embed_keep_alive="24h",
    )
    await store._http.aclose()
    store._http = httpx.AsyncClient(base_url="http://unused.invalid", transport=httpx.MockTransport(handle))
    try:
        await store.init()
        for user, conversation, question, answer in [
            ("owner", "math", "Evaluate 90 times 2", "180"),
            ("owner", "garden", "Plant an orchard", "GARDEN"),
            ("foreign", "private", "A private subject", "PRIVATE_OTHER_USER"),
        ]:
            await store.archive_turn(user=user, conversation_id=conversation,
                                     user_content=question, assistant_content=answer,
                                     archive_id=conversation, created_at="2026-10-15T12:00:00+00:00")
        info = await store._qdrant.get_collection("history")
        assert info.config.params.vectors.size == 768
        assert info.config.metadata == NOMIC.metadata
        hits = await store.search(user="owner", query="prior arithmetic expression", limit=5)
        assert [hit.conversation_id for hit in hits] == ["math"]
        assert hits[0].score == pytest.approx(.5788, abs=1e-6)
        assert hits[0].match_type == "semantic" and "180" in hits[0].snippet
        assert seen[-1]["input"] == ["search_query: prior arithmetic expression"]
        assert await store.search(user="owner", query="prior arithmetic expression", limit=5,
                                  date_from="2026-10-16T00:00:00+00:00") == []
        await store.request_conversation_deletion(user="owner", conversation_id="math")
        assert await store.search(user="owner", query="prior arithmetic expression", limit=5) == []

        # A Qwen collection cannot be read or reused as Nomic just because its name matches.
        await store._qdrant.delete_collection("history")
        await store._qdrant.create_collection("history", vectors_config=qm.VectorParams(
            size=2560, distance=qm.Distance.COSINE,
        ), metadata=load_profile(CONFIG).metadata)
        with pytest.raises(TextEmbeddingError, match="differs"):
            await store.init_index()
    finally:
        await store.aclose()
