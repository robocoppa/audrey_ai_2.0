"""Healthy sidecar dependencies can degrade and recover after startup."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from audrey.tools.discovery import ToolRegistry, discover_one

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import app as tools_app  # noqa: E402
from capabilities import CapabilityRegistry, CapabilitySupervisor  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _parts():
    clock = _Clock()
    registry = CapabilityRegistry()
    memory = SimpleNamespace(
        probe_qdrant=AsyncMock(), init_qdrant=AsyncMock(),
        warm_embedder=AsyncMock(return_value=True),
    )
    archive = SimpleNamespace(
        init_source=AsyncMock(), init_index=AsyncMock(),
        warm_embedder=AsyncMock(return_value=True),
    )
    maintainer = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    supervisor = CapabilitySupervisor(
        registry=registry, memory=memory, archive=archive,
        archive_maintainer=maintainer, retry_interval_s=5, probe_timeout_s=.5,
        clock=clock,
    )
    return supervisor, registry, memory, archive, maintainer, clock


async def test_fast_recovery_polls_do_not_rewarm_healthy_models():
    supervisor, registry, memory, archive, maintainer, clock = _parts()
    await supervisor.refresh()

    for second in (5, 10, 299):
        clock.now = second
        await supervisor.refresh()

    memory.init_qdrant.assert_awaited_once()
    archive.init_index.assert_awaited_once()
    memory.warm_embedder.assert_awaited_once()
    archive.warm_embedder.assert_awaited_once()
    assert memory.probe_qdrant.await_count == 4

    clock.now = 300
    await supervisor.refresh()

    assert memory.init_qdrant.await_count == 2
    assert archive.init_index.await_count == 2
    assert memory.warm_embedder.await_count == 2
    assert archive.warm_embedder.await_count == 2
    assert all(state.available for state in registry.snapshot().values())
    archive.init_source.assert_awaited_once()
    maintainer.start.assert_awaited_once()


@pytest.mark.parametrize(
    "component, client_name, method_name, unavailable_tools, unaffected_tool",
    [
        ("memory", "memory", "init_qdrant", {"memory_recall", "memory_search", "memory_store"}, "chat_history_search"),
        ("chat_archive", "archive", "init_index", {"chat_history_search"}, "memory_search"),
        ("text_embedding", "memory", "warm_embedder", {"kb_search", "memory_search", "memory_store", "reference_search"}, "chat_history_search"),
        ("history_embedding", "archive", "warm_embedder", {"chat_history_search"}, "memory_search"),
    ],
)
async def test_ready_component_failure_hides_its_tools_and_recovers(
    monkeypatch, caplog, component, client_name, method_name,
    unavailable_tools, unaffected_tool,
):
    supervisor, registry, memory, archive, _maintainer, clock = _parts()
    monkeypatch.setattr(tools_app.app.state, "capabilities", registry, raising=False)
    await supervisor.refresh()
    backend = memory if client_name == "memory" else archive
    method = getattr(backend, method_name)
    method.side_effect = ValueError("PRIVATE_PROVIDER_BODY")
    clock.now = 300

    await supervisor.refresh()

    assert not registry.snapshot()[component].available
    assert "PRIVATE_PROVIDER_BODY" not in caplog.text
    assert "ValueError" in caplog.text

    def transport(_request):
        return httpx.Response(200, json=tools_app.app.openapi())

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        failed = ToolRegistry(by_name={
            spec.name: spec for spec in await discover_one(client, "http://custom-tools:8001")
        })
        assert unavailable_tools.isdisjoint(failed.names())
        assert unaffected_tool in failed.names()
        assert {"web_search", "web_fetch", "calculate"}.issubset(failed.names())
        assert set(failed.by_name) - set(failed.names()) == unavailable_tools
        assert tools_app.app.openapi()["x-audrey-capabilities"][component]["available"] is False

        method.side_effect = None
        # Failed dependencies recover on the normal retry cadence, rather than
        # waiting another full healthy revalidation interval.
        clock.now = 305
        await supervisor.refresh()
        recovered = ToolRegistry(by_name={
            spec.name: spec for spec in await discover_one(client, "http://custom-tools:8001")
        })
    assert registry.snapshot()[component].available
    assert unavailable_tools.issubset(recovered.names())
    assert set(recovered.by_name) == set(recovered.names())
    assert tools_app.app.openapi()["x-audrey-capabilities"][component]["available"] is True


async def test_qdrant_outage_forces_collection_identity_revalidation_on_recovery():
    supervisor, registry, memory, archive, maintainer, clock = _parts()
    await supervisor.refresh()
    clock.now = 5
    memory.probe_qdrant.side_effect = ConnectionError("injected outage")

    await supervisor.refresh()

    assert not registry.snapshot()["qdrant"].available
    assert not registry.snapshot()["memory"].available
    assert not registry.snapshot()["chat_archive"].available
    memory.init_qdrant.assert_awaited_once()
    archive.init_index.assert_awaited_once()

    memory.probe_qdrant.side_effect = None
    memory.init_qdrant.side_effect = ValueError("collection identity changed")
    clock.now = 10
    await supervisor.refresh()

    assert registry.snapshot()["qdrant"].available
    assert not registry.snapshot()["memory"].available
    assert registry.snapshot()["chat_archive"].available
    assert memory.init_qdrant.await_count == 2
    assert archive.init_index.await_count == 2
    # A Qdrant incident does not pretend the independently healthy models failed.
    memory.warm_embedder.assert_awaited_once()
    archive.warm_embedder.assert_awaited_once()
    maintainer.start.assert_awaited_once()
