"""Audrey refreshes a partially degraded tool registry in place."""

from __future__ import annotations

from audrey import main as main_module
from audrey.tools.discovery import ToolRegistry, ToolSpec


def _spec(name: str, *, available: bool) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        server_url="http://custom-tools:8001",
        path=f"/{name}",
        available=available,
        unavailable_reason=None if available else "dependency_unavailable:qdrant",
    )


async def test_bounded_retry_waits_for_partial_capability_recovery(monkeypatch):
    live = ToolRegistry(by_name={
        "web_search": _spec("web_search", available=True),
        "kb_search": _spec("kb_search", available=False),
    })
    partial = ToolRegistry(by_name={
        "web_search": _spec("web_search", available=True),
        "kb_search": _spec("kb_search", available=False),
    })
    recovered = ToolRegistry(by_name={
        "web_search": _spec("web_search", available=True),
        "kb_search": _spec("kb_search", available=True),
    })
    discoveries = [partial, recovered]

    async def fake_discover_all(_servers):
        return discoveries.pop(0)

    monkeypatch.setattr(main_module, "discover_all", fake_discover_all)

    await main_module._retry_tool_discovery(
        live,
        ["http://custom-tools:8001"],
        attempts=2,
        interval_s=0,
    )

    assert discoveries == []
    assert live.names() == ["kb_search", "web_search"]
    assert live.get("kb_search") is not None
