"""Campaign 3 Wave 1D.3 — custom-tools degrades per capability."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import app as tools_app  # noqa: E402
from capabilities import CapabilityRegistry, CapabilitySupervisor  # noqa: E402


class _Memory:
    def __init__(self) -> None:
        self.qdrant_available = False
        self.probes = 0
        self.initializations = 0
        self.warmups = 0

    async def probe_qdrant(self, *, timeout_s: float) -> None:
        assert timeout_s == 0.5
        self.probes += 1
        if not self.qdrant_available:
            raise ConnectionError("injected qdrant outage")

    async def init_qdrant(self) -> None:
        self.initializations += 1

    async def warm_embedder(self) -> bool:
        self.warmups += 1
        return True


class _Archive:
    def __init__(self) -> None:
        self.source_initializations = 0
        self.index_initializations = 0

    async def init_source(self) -> None:
        self.source_initializations += 1

    async def init_index(self) -> None:
        self.index_initializations += 1


class _Maintainer:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1


async def test_qdrant_outage_leaves_stateless_and_archive_source_available():
    registry = CapabilityRegistry()
    memory = _Memory()
    archive = _Archive()
    maintainer = _Maintainer()
    supervisor = CapabilitySupervisor(
        registry=registry,
        memory=memory,
        archive=archive,
        archive_maintainer=maintainer,
        retry_interval_s=60,
        probe_timeout_s=0.5,
    )

    await supervisor.refresh()

    states = registry.snapshot()
    assert states["web_search"].available is True
    assert states["web_fetch"].available is True
    assert states["chat_archive_source"].available is True
    assert states["qdrant"].available is False
    assert states["memory"].available is False
    assert states["chat_archive"].available is False
    assert archive.source_initializations == 1
    assert archive.index_initializations == 0
    assert maintainer.starts == 1

    memory.qdrant_available = True
    await supervisor.refresh()

    states = registry.snapshot()
    assert states["qdrant"].available is True
    assert states["memory"].available is True
    assert states["text_embedding"].available is True
    assert states["chat_archive"].available is True
    assert memory.initializations == 1
    assert memory.warmups == 1
    assert archive.index_initializations == 1
    # Recovery reuses the initialized source and its one maintainer.
    assert archive.source_initializations == 1
    assert maintainer.starts == 1


def test_dynamic_openapi_publishes_current_capability_state(monkeypatch):
    registry = CapabilityRegistry.all_available()
    registry.set_unavailable("qdrant", "connection_failed")
    monkeypatch.setattr(
        tools_app.app.state, "capabilities", registry, raising=False
    )

    state = tools_app.app.openapi()["x-audrey-capabilities"]

    assert state["qdrant"] == {
        "available": False,
        "reason": "connection_failed",
    }
    assert state["web_fetch"]["available"] is True


def test_unavailable_route_returns_component_specific_503(monkeypatch):
    registry = CapabilityRegistry.all_available()
    registry.set_unavailable("qdrant", "connection_failed")
    monkeypatch.setattr(
        tools_app.app.state, "capabilities", registry, raising=False
    )

    with pytest.raises(HTTPException) as caught:
        tools_app._require_capabilities("audrey_kb", "qdrant")

    assert caught.value.status_code == 503
    assert caught.value.detail == {
        "error": "capability_unavailable",
        "components": ["qdrant"],
    }
    assert caught.value.headers == {"Retry-After": "5"}


async def test_shallow_health_ignores_optional_capability_failure(monkeypatch):
    registry = CapabilityRegistry()
    registry.set_unavailable("qdrant", "connection_failed")
    monkeypatch.setattr(
        tools_app.app.state, "capabilities", registry, raising=False
    )

    assert await tools_app.health() == {"status": "ok"}
