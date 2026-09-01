"""The real custom-tools lifespan must yield while Qdrant is unavailable."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import app as tools_app  # noqa: E402


class _DeadQdrantMemory:
    async def probe_qdrant(self, *, timeout_s: float) -> None:
        assert timeout_s > 0
        raise ConnectionError("injected qdrant outage")

    async def init_qdrant(self) -> None:
        raise AssertionError("memory initialization must wait for Qdrant")

    async def warm_embedder(self) -> bool:
        raise AssertionError("embedding warm-up must wait for Qdrant")

    async def aclose(self) -> None:
        pass


class _ArchiveSource:
    def __init__(self) -> None:
        self.source_ready = False

    async def init_source(self) -> None:
        self.source_ready = True

    async def init_index(self) -> None:
        raise AssertionError("archive index initialization must wait for Qdrant")

    async def aclose(self) -> None:
        pass


class _Maintainer:
    def __init__(self) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


class _WebClient:
    async def aclose(self) -> None:
        pass


async def test_lifespan_yields_with_qdrant_down(monkeypatch):
    memory = _DeadQdrantMemory()
    archive = _ArchiveSource()
    maintainer = _Maintainer()
    brave = _WebClient()

    monkeypatch.setattr(tools_app, "MemoryStore", lambda **_kwargs: memory)
    monkeypatch.setattr(tools_app, "ChatArchiveStore", lambda **_kwargs: archive)
    monkeypatch.setattr(
        tools_app,
        "ChatArchiveMaintainer",
        lambda *_args, **_kwargs: maintainer,
    )
    monkeypatch.setattr(tools_app, "BraveClient", lambda **_kwargs: brave)
    monkeypatch.setattr(tools_app.settings, "searxng_url", "")
    monkeypatch.setattr(
        tools_app.settings, "capability_retry_interval_s", 60.0
    )
    monkeypatch.setattr(tools_app.settings, "capability_probe_timeout_s", 0.5)

    async with tools_app.lifespan(tools_app.app):
        states = tools_app.app.state.capabilities.snapshot()
        assert await tools_app.health() == {"status": "ok"}
        assert states["web_search"].available is True
        assert states["web_fetch"].available is True
        assert states["chat_archive_source"].available is True
        assert states["qdrant"].available is False
        assert maintainer.started is True

        schema_state = tools_app.app.openapi()["x-audrey-capabilities"]
        assert schema_state["qdrant"]["available"] is False

    assert maintainer.started is False
    tools_app.app.state.capabilities = (
        tools_app.CapabilityRegistry.all_available()
    )
