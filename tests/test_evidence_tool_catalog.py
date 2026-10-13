"""Real sidecar discovery, identity binding, and evidence proxy boundaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from audrey.tools.discovery import (
    ToolRegistry,
    ToolUserScope,
    discover_one,
)
from audrey.tools.dispatch import dispatch_one

_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

import app as sidecar  # noqa: E402
from capabilities import CapabilityRegistry  # noqa: E402

_OWNER = "alice@example.com"
_SCOPED = {
    "analyze_table": {"filename": "costs.csv", "action": "inspect"},
    "inspect_pdf_page": {"filename": "drawing.pdf", "page": 2},
    "inspect_video_frame": {"filename": "clip.mp4", "timestamp_s": 2.5},
    "open_chat_history": {"conversation_id": "con_saved"},
}
_PATHS = {
    "analyze_table": "/v1/files/table",
    "inspect_pdf_page": "/v1/files/inspect-pdf-page",
    "inspect_video_frame": "/v1/files/inspect-video-frame",
    "open_chat_history": "/v1/chat/history/conversation",
}
_HISTORY = {
    "conversation_id": "con_saved", "title": "A saved decision",
    "text": "[user] Saved question\n[assistant] Saved answer",
    "offset": 0, "next_offset": None, "total_chars": 49,
}


@pytest.fixture
def available(monkeypatch):
    registry = CapabilityRegistry.all_available()
    monkeypatch.setattr(sidecar.app.state, "capabilities", registry)
    archive = SimpleNamespace(conversation_read_cutoff=AsyncMock(return_value=""))
    monkeypatch.setattr(sidecar.app.state, "chat_archive", archive, raising=False)
    return registry, archive


async def _discover():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sidecar.app), base_url="http://sidecar",
    ) as client:
        specs = await discover_one(client, "http://sidecar")
    return ToolRegistry(by_name={spec.name: spec for spec in specs})


async def _call_proxy(monkeypatch, name, handler, *, extra=None):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://audrey",
    ) as backend:
        monkeypatch.setattr(sidecar.app.state, "audrey", backend, raising=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=sidecar.app), base_url="http://sidecar",
        ) as client:
            return await client.post(
                f"/{name}", json={"user": _OWNER, **_SCOPED[name], **(extra or {})},
            )


async def test_actual_openapi_discovers_declared_scope_and_boundaries(available):
    registry = await _discover()
    calculate = registry.get("calculate")
    assert calculate is not None
    assert calculate.user_scope is ToolUserScope.NONE
    assert not calculate.dependencies
    assert "user" not in calculate.parameters["properties"]
    for name in _SCOPED:
        spec = registry.get(name)
        assert spec is not None, name
        assert spec.user_scope is ToolUserScope.ARGUMENT
        assert "user" in spec.parameters["required"]
        assert spec.dependencies == (
            frozenset({"audrey_history", "chat_archive_source"}) if name == "open_chat_history"
            else frozenset({"audrey_files"})
        )
    history = registry.get("open_chat_history")
    assert history is not None
    assert history.purge_gated
    assert "limit" not in history.parameters["properties"]
    assert "visible_after" not in history.parameters["properties"]


@pytest.mark.parametrize("name", list(_SCOPED))
async def test_dispatch_overrides_forged_owner_and_uses_visual_deadline(available, name):
    registry = await _discover()
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["timeout"] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await dispatch_one(
            client, registry,
            {"id": "call", "function": {"name": name, "arguments": {
                "user": "bob@example.com", **_SCOPED[name],
            }}},
            user_id=_OWNER, max_result_chars=4000, timeout_s=30,
        )
    assert not result.is_error
    assert captured["body"]["user"] == _OWNER
    assert captured["timeout"] == (150 if name.startswith("inspect_") else 30)


async def test_stateless_calculation_survives_stateful_component_failure(available):
    capabilities, _ = available
    for component in ("qdrant", "memory", "text_embedding", "audrey_files", "chat_archive_source"):
        capabilities.set_unavailable(component, "offline")
    registry = await _discover()
    assert registry.get("calculate") is not None
    assert registry.get("analyze_table") is None
    assert registry.get("open_chat_history") is None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sidecar.app), base_url="http://sidecar",
    ) as client:
        good = await client.post("/calculate", json={"expression": "250 * 12%"})
        assert good.status_code == 200 and good.json()["result"] == "30"
        forged = await client.post("/calculate", json={"expression": "1", "user": "bob"})
        assert forged.status_code == 422


@pytest.mark.parametrize("name", list(_SCOPED))
async def test_successful_proxy_keeps_owned_request_and_backend_metadata(available, monkeypatch, name):
    captured = {}
    response = _HISTORY if name == "open_chat_history" else {
        "filename": _SCOPED[name]["filename"], "file_id": "owned-id",
        "artifact": "table" if name == "analyze_table" else "visual",
        "description": "Visible detail only.",
    }

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["timeout"] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json=response)

    result = await _call_proxy(monkeypatch, name, handler)
    assert result.status_code == 200, result.text
    assert result.json() == response
    assert captured["path"] == _PATHS[name]
    assert captured["body"]["user"] == _OWNER
    assert captured["timeout"] == (140 if name.startswith("inspect_") else 25)
    if name == "open_chat_history":
        assert captured["body"]["limit"] == 2000
        assert captured["body"]["visible_after"] is None


@pytest.mark.parametrize(("upstream", "expected"), [
    (404, 404), (409, 409), (422, 422), (503, 503),
])
async def test_proxy_preserves_explicit_backend_errors(available, monkeypatch, upstream, expected):
    result = await _call_proxy(
        monkeypatch, "inspect_pdf_page",
        lambda _: httpx.Response(upstream, json={"detail": "Original file is unavailable."}),
    )
    assert result.status_code == expected
    assert result.json() == {"detail": "Original file is unavailable."}


@pytest.mark.parametrize(("failure", "expected"), [
    ("timeout", 504), ("network", 502), ("html", 502), ("list", 502),
])
async def test_proxy_transport_and_malformed_data_are_safe(available, monkeypatch, failure, expected):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("private-provider-url", request=request)
        if failure == "network":
            raise httpx.ConnectError("private-provider-url", request=request)
        if failure == "html":
            return httpx.Response(200, text="<html>provider secret</html>")
        return httpx.Response(200, json=["provider secret"])

    result = await _call_proxy(monkeypatch, "inspect_pdf_page", handler)
    assert result.status_code == expected
    assert "private-provider-url" not in result.text and "provider secret" not in result.text


async def test_history_passes_durable_cutoff_and_rechecks_mapped_canonical_id(available, monkeypatch):
    _, archive = available
    cutoffs = {
        "con_saved": "2026-10-12T00:00:00+00:00",
        "con_canonical": "2026-10-13T00:00:00+00:00",
    }
    archive.conversation_read_cutoff.side_effect = lambda **fields: cutoffs[fields["conversation_id"]]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        text = "VISIBLE RECENT" if len(requests) == 2 else "OLD CONTENT BEFORE CANONICAL CUTOFF"
        return httpx.Response(200, json={**_HISTORY, "conversation_id": "con_canonical", "text": text})

    result = await _call_proxy(monkeypatch, "open_chat_history", handler, extra={"offset": 12})
    assert result.status_code == 200, result.text
    assert len(requests) == 2
    assert requests[0]["visible_after"] == cutoffs["con_saved"]
    assert requests[1]["visible_after"] == cutoffs["con_canonical"]
    assert requests[1]["offset"] == 12
    assert result.json()["text"] == "VISIBLE RECENT"
    assert "OLD CONTENT" not in result.text
    assert archive.conversation_read_cutoff.await_args_list[0].kwargs == {
        "user": _OWNER, "conversation_id": "con_saved",
    }
    assert archive.conversation_read_cutoff.await_args_list[1].kwargs == {
        "user": _OWNER, "conversation_id": "con_canonical",
    }
