"""A Qdrant outage removes only the six declared dependent model tools."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from audrey.tools.discovery import ToolRegistry, discover_one

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import app as tools_app  # noqa: E402
from capabilities import CapabilityRegistry  # noqa: E402


async def test_real_openapi_projects_only_qdrant_independent_tools(monkeypatch):
    capabilities = CapabilityRegistry.all_available()
    capabilities.set_unavailable("qdrant", "connection_failed")
    monkeypatch.setattr(
        tools_app.app.state, "capabilities", capabilities, raising=False
    )
    schema = tools_app.app.openapi()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=schema)
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        specs = await discover_one(client, "http://custom-tools:8001")

    registry = ToolRegistry(by_name={spec.name: spec for spec in specs})
    assert registry.names() == [
        "get_file_text",
        "list_my_files",
        "web_fetch",
        "web_search",
    ]
    assert {
        spec.name
        for spec in registry.policy_records()
        if not spec.available
    } == {
        "chat_history_search",
        "kb_image_search",
        "kb_search",
        "memory_recall",
        "memory_search",
        "memory_store",
    }
    assert all(
        "qdrant" in (spec.unavailable_reason or "")
        for spec in registry.policy_records()
        if not spec.available
    )
