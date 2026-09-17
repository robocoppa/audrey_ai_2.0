"""Public-safe capability health and disabled skills contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.auth import require_principal
from audrey.readiness import ComponentReadiness, ToolReadiness
from audrey.routes.app import router
from audrey.routes.app.capabilities import capability_response


def _snapshot(
    *,
    ollama: str = "available",
    custom_tools: str = "available",
    qdrant: str = "available",
    discovered: int = 10,
    available: int = 10,
):
    return SimpleNamespace(
        generated_at="2026-09-16T12:00:00+00:00",
        components={
            "ollama": ComponentReadiness(status=ollama),
            "custom_tools": ComponentReadiness(status=custom_tools),
            "qdrant": ComponentReadiness(status=qdrant),
        },
        tools=ToolReadiness(
            policy_count=10,
            discovered_count=discovered,
            available_count=available,
        ),
    )


def test_capability_projection_is_sanitized_and_distinguishes_degradation():
    healthy = capability_response(_snapshot())
    assert healthy.status == "ready"
    assert healthy.chat.status == "available"
    assert healthy.tools.status == "available"
    assert healthy.knowledge.status == "available"
    assert healthy.skills.status == "disabled"

    degraded = capability_response(_snapshot(qdrant="unavailable", available=8))
    assert degraded.status == "degraded"
    assert degraded.chat.status == "available"
    assert degraded.tools.status == "degraded"
    assert degraded.knowledge.status == "unavailable"

    offline = capability_response(_snapshot(ollama="unavailable", custom_tools="disabled"))
    assert offline.status == "unavailable"
    assert offline.tools.status == "disabled"
    assert "host" not in offline.model_dump_json()


def test_capabilities_and_skills_require_native_identity_and_expose_no_admin_detail():
    for path in ("/api/capabilities", "/api/skills"):
        route = next(route for route in router.routes if route.path == path)
        assert require_principal in {
            dependency.call for dependency in route.dependant.dependencies
        }

    app = FastAPI()
    collector = SimpleNamespace(collect=AsyncMock(return_value=_snapshot(
        ollama="unavailable",
        discovered=0,
        available=0,
    )))
    app.state.readiness = collector
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: object()

    with TestClient(app) as client:
        health = client.get("/api/capabilities")
        skills = client.get("/api/skills")

    assert health.status_code == 200
    assert health.json() == {
        "status": "unavailable",
        "generated_at": "2026-09-16T12:00:00+00:00",
        "chat": {"status": "unavailable"},
        "tools": {"status": "degraded"},
        "knowledge": {"status": "degraded"},
        "skills": {"status": "disabled"},
    }
    assert skills.status_code == 200
    assert skills.json() == {"enabled": False, "items": []}
    collector.collect.assert_awaited_once_with()


def test_capabilities_fail_closed_if_readiness_not_initialized():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: object()

    with TestClient(app) as client:
        response = client.get("/api/capabilities")

    assert response.status_code == 503
    assert response.json()["detail"] == "capability_health_unavailable"
