"""Every Qdrant-dependent model route fails explicitly while degraded."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.testclient import TestClient

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import app as tools_app  # noqa: E402
from capabilities import CapabilityRegistry  # noqa: E402


def test_qdrant_dependent_routes_return_precise_503(monkeypatch):
    @asynccontextmanager
    async def no_lifespan(_app):
        yield

    capabilities = CapabilityRegistry.all_available()
    capabilities.set_unavailable("qdrant", "connection_failed")
    monkeypatch.setattr(tools_app.app.router, "lifespan_context", no_lifespan)
    monkeypatch.setattr(
        tools_app.app.state, "capabilities", capabilities, raising=False
    )

    requests = {
        "/kb_search": {"query": "x", "user": "alice@example.com"},
        "/kb_image_search": {"query": "x", "user": "alice@example.com"},
        "/memory_store": {
            "key": "k",
            "value": "v",
            "tags": "user:alice@example.com",
        },
        "/memory_recall": {"key": "k", "user": "alice@example.com"},
        "/memory_search": {"query": "x", "user": "alice@example.com"},
        "/chat_history_search": {
            "query": "x",
            "user": "alice@example.com",
        },
    }

    with TestClient(tools_app.app) as client:
        for path, body in requests.items():
            response = client.post(path, json=body)
            assert response.status_code == 503, path
            assert response.json()["detail"] == {
                "error": "capability_unavailable",
                "components": ["qdrant"],
            }
            assert response.headers["retry-after"] == "5"
        assert client.get("/health").status_code == 200
