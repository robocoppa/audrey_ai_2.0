"""Optional search providers must not govern the sidecar's other tools."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from starlette.datastructures import State

_TOOLS = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import app as tools_app  # noqa: E402
from brave import SearchResult  # noqa: E402


@pytest.fixture
def resources(monkeypatch):
    monkeypatch.setattr(tools_app.app, "state", State())
    memory = SimpleNamespace(probe_qdrant=AsyncMock(), init_qdrant=AsyncMock(),
        warm_embedder=AsyncMock(return_value=True), aclose=AsyncMock())
    archive = SimpleNamespace(init_source=AsyncMock(), init_index=AsyncMock(),
        warm_embedder=AsyncMock(return_value=True), aclose=AsyncMock())
    maintainer = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    backup = SimpleNamespace(start=Mock(), stop=AsyncMock())
    monkeypatch.setattr(tools_app, "MemoryStore", lambda **_: memory)
    monkeypatch.setattr(tools_app, "ChatArchiveStore", lambda **_: archive)
    monkeypatch.setattr(tools_app, "ChatArchiveMaintainer", lambda *_, **__: maintainer)
    monkeypatch.setattr(tools_app, "MemoryBackup", lambda **_: object())
    monkeypatch.setattr(tools_app, "MemoryBackupScheduler", lambda *_, **__: backup)
    created = {}

    class Brave:
        def __init__(self, *, api_key, **_):
            assert api_key == "test-key"
            self.search = AsyncMock(return_value=[SearchResult("Brave", "https://brave.example", "hit")])
            self.aclose = AsyncMock()
            created["brave"] = self

    class Searxng:
        def __init__(self, url):
            assert url == "http://search.example"
            self.search = AsyncMock(return_value=[SearchResult("SearXNG", "https://searxng.example", "hit")])
            self.aclose = AsyncMock()
            created["searxng"] = self

    monkeypatch.setattr(tools_app, "BraveClient", Brave)
    monkeypatch.setattr(tools_app, "SearxngClient", Searxng)
    monkeypatch.setattr(tools_app.settings, "capability_retry_interval_s", 60)
    return SimpleNamespace(memory=memory, archive=archive, maintainer=maintainer,
                           backup=backup, created=created)


@pytest.mark.parametrize("providers", [(), ("brave",), ("searxng",), ("brave", "searxng")])
async def test_optional_provider_startup_routes_and_shutdown(resources, monkeypatch, providers):
    # Whitespace is an unset optional credential too, never an invalid client.
    monkeypatch.setattr(tools_app.settings, "brave_api_key", " test-key " if "brave" in providers else "  ")
    monkeypatch.setattr(tools_app.settings, "searxng_url", " http://search.example " if "searxng" in providers else "  ")
    monkeypatch.setattr(tools_app.settings, "web_search_alternate", False)
    async with tools_app.lifespan(tools_app.app):
        assert set(resources.created) == set(providers)
        states = tools_app.app.openapi()["x-audrey-capabilities"]
        assert states["web_search"]["available"] is bool(providers)
        assert states["memory"]["available"] and states["text_embedding"]["available"]
        assert states["chat_archive_source"]["available"] and states["history_embedding"]["available"]
        assert states["web_fetch"]["available"] and states["audrey_kb"]["available"]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=tools_app.app), base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
            response = await client.post("/web_search", json={"query": "test query"})
        if providers:
            expected = "Brave" if "brave" in providers else "SearXNG"
            assert response.status_code == 200
            assert response.json()["results"][0]["title"] == expected
        else:
            assert response.status_code == 503
            assert response.json()["detail"] == {"error": "capability_unavailable", "components": ["web_search"]}
            assert states["web_search"]["reason"] == "not_configured"
    for provider in resources.created.values():
        provider.aclose.assert_awaited_once()
    resources.memory.aclose.assert_awaited_once()
    resources.archive.aclose.assert_awaited_once()
    resources.maintainer.stop.assert_awaited_once()
    resources.backup.stop.assert_awaited_once()


@pytest.mark.parametrize("prefer", [True, False])
async def test_keyless_searxng_search_works_for_either_query_hash(resources, monkeypatch, prefer):
    monkeypatch.setattr(tools_app.settings, "brave_api_key", "")
    monkeypatch.setattr(tools_app.settings, "searxng_url", "http://search.example")
    monkeypatch.setattr(tools_app.settings, "web_search_alternate", True)
    monkeypatch.setattr(tools_app, "_prefer_searxng", lambda _: prefer)
    async with tools_app.lifespan(tools_app.app):
        response = await tools_app.web_search(tools_app.WebSearchRequest(query="test query"))
        assert response.results[0].title == "SearXNG"
        resources.created["searxng"].search.assert_awaited_once_with(query="test query", count=5)
