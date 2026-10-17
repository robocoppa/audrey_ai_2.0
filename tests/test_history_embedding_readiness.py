"""History warms its own model and degrades independently of memory embeddings."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from audrey.embedding_profile import EmbeddingProfile

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import chat_archive  # noqa: E402
from capabilities import CapabilityRegistry, CapabilitySupervisor  # noqa: E402

NOMIC = EmbeddingProfile("nomic-embed-text:latest", "b" * 64, 3, "", "")


def supervisor_parts():
    registry = CapabilityRegistry()
    memory = SimpleNamespace(
        probe_qdrant=AsyncMock(), init_qdrant=AsyncMock(),
        warm_embedder=AsyncMock(return_value=True),
    )
    archive = SimpleNamespace(
        init_source=AsyncMock(), init_index=AsyncMock(),
        warm_embedder=AsyncMock(return_value=True),
    )
    supervisor = CapabilitySupervisor(
        registry=registry, memory=memory, archive=archive,
        archive_maintainer=SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
        retry_interval_s=60, probe_timeout_s=.5,
    )
    return supervisor, registry, memory, archive


@pytest.mark.parametrize("failed", ["memory", "history"])
async def test_models_warm_and_recover_independently(failed):
    supervisor, registry, memory, archive = supervisor_parts()
    broken = memory if failed == "memory" else archive
    other = archive if failed == "memory" else memory
    broken.warm_embedder.return_value = False

    await supervisor.refresh()

    states = registry.snapshot()
    assert states["text_embedding"].available is (failed != "memory")
    assert states["history_embedding"].available is (failed != "history")
    assert states["memory"].available
    assert states["chat_archive"].available
    assert states["chat_archive_source"].available
    assert states["audrey_kb"].available
    assert states["web_search"].available

    broken.warm_embedder.return_value = True
    await supervisor.refresh()
    states = registry.snapshot()
    assert states["text_embedding"].available
    assert states["history_embedding"].available
    assert broken.warm_embedder.await_count == 2
    other.warm_embedder.assert_awaited_once()
    memory.init_qdrant.assert_awaited_once()
    archive.init_index.assert_awaited_once()


async def test_history_profile_mismatch_does_not_warm_or_disable_memory():
    supervisor, registry, memory, archive = supervisor_parts()
    archive.init_index.side_effect = ValueError("collection uses another embedding profile")

    await supervisor.refresh()

    assert registry.snapshot()["text_embedding"].available
    assert not registry.snapshot()["chat_archive"].available
    assert not registry.snapshot()["history_embedding"].available
    archive.warm_embedder.assert_not_awaited()

    archive.init_index.side_effect = None
    await supervisor.refresh()

    assert registry.snapshot()["chat_archive"].available
    assert registry.snapshot()["history_embedding"].available
    archive.warm_embedder.assert_awaited_once()
    memory.warm_embedder.assert_awaited_once()


async def test_unexpected_history_warmup_error_remains_isolated_and_private(caplog):
    supervisor, registry, _memory, archive = supervisor_parts()
    archive.warm_embedder.side_effect = RuntimeError("PRIVATE_PROVIDER_BODY")

    await supervisor.refresh()

    assert registry.snapshot()["text_embedding"].available
    assert not registry.snapshot()["history_embedding"].available
    assert registry.snapshot()["history_embedding"].reason == "warmup_failed"
    assert "RuntimeError" in caplog.text
    assert "PRIVATE_PROVIDER_BODY" not in caplog.text


def history_store(http):
    store = chat_archive.ChatArchiveStore.__new__(chat_archive.ChatArchiveStore)
    store._http = http
    store._embedding_profile = NOMIC
    store._embed_model = NOMIC.model
    store._embed_keep_alive = "24h"
    return store


async def test_history_warmup_uses_and_verifies_nomic_instead_of_memory_model():
    seen = []

    def transport(request):
        seen.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3-embedding:4b", "digest": "a" * 64},
                {"name": NOMIC.model, "digest": NOMIC.digest},
            ]})
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[1., 0., 0.]]})

    async with httpx.AsyncClient(
        base_url="http://ollama.invalid", transport=httpx.MockTransport(transport),
    ) as http:
        assert await history_store(http).warm_embedder()

    assert [request.url.path for request in seen] == ["/api/tags", "/api/embed", "/api/tags"]
    assert json.loads(seen[1].content) == {
        "model": NOMIC.model, "input": ["warm"], "truncate": False, "keep_alive": "24h",
    }


@pytest.mark.parametrize("failure", ["missing", "wrong_digest", "provider_error"])
async def test_history_warmup_cannot_be_satisfied_by_qwen_and_logs_no_provider_text(
    failure, caplog,
):
    posts = []

    def transport(request):
        if request.url.path == "/api/tags":
            models = [{"name": "qwen3-embedding:4b", "digest": "a" * 64}]
            if failure != "missing":
                models.append({"name": NOMIC.model, "digest": (
                    "c" * 64 if failure == "wrong_digest" else NOMIC.digest
                )})
            return httpx.Response(200, json={"models": models})
        posts.append(request)
        return httpx.Response(500, text="PRIVATE_PROVIDER_BODY")

    async with httpx.AsyncClient(
        base_url="http://ollama.invalid", transport=httpx.MockTransport(transport),
    ) as http:
        assert not await history_store(http).warm_embedder()

    assert len(posts) == int(failure == "provider_error")
    assert "PRIVATE_PROVIDER_BODY" not in caplog.text


async def test_history_warmup_has_bounded_deadline_and_preserves_cancellation(monkeypatch):
    monkeypatch.setattr(chat_archive, "_WARM_TIMEOUT_S", .01)

    async def transport(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": NOMIC.model, "digest": NOMIC.digest},
            ]})
        await asyncio.Event().wait()
        raise AssertionError("a stalled provider must time out")

    async with httpx.AsyncClient(
        base_url="http://ollama.invalid", transport=httpx.MockTransport(transport),
    ) as http:
        store = history_store(http)
        assert not await store.warm_embedder()
        monkeypatch.setattr(chat_archive, "_embed", AsyncMock(side_effect=asyncio.CancelledError))
        with pytest.raises(asyncio.CancelledError):
            await store.warm_embedder()
