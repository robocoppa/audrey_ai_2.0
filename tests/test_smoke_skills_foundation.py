"""The built-in skill registry smoke is narrow and contract-focused."""

from __future__ import annotations

import json
from typing import Any

from scripts import smoke_skills_foundation as smoke


def _ready_counts() -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "ready",
        "loaded_count": 2,
        "available_count": 2,
        "degraded_count": 0,
        "invalid_count": 0,
    }


def _catalog_items() -> list[dict[str, Any]]:
    return [
        {
            "id": "grounded-document-analysis",
            "name": "Grounded document analysis",
            "description": (
                "Read, compare, and explain the user's uploaded documents "
                "with explicit evidence coverage."
            ),
            "version": 1,
            "supported_modes": ["auto", "deep", "fast"],
            "availability": "available",
        },
        {
            "id": "video-analysis",
            "name": "Video analysis",
            "description": (
                "Analyze uploaded videos and documents from the user's own evidence."
            ),
            "version": 1,
            "supported_modes": ["auto", "deep", "fast"],
            "availability": "available",
        },
    ]


def test_smoke_checks_the_complete_builtin_skill_registry(monkeypatch, capsys):
    calls: list[tuple[str, str, str]] = []

    def fake_json_request(
        path: str,
        *,
        token: str,
        method: str = "GET",
        expected: frozenset[int] = frozenset({200}),
    ) -> tuple[int, dict[str, Any]]:
        calls.append((method, path, token))
        if path == "/api/skills":
            return 200, {
                "enabled": True,
                "status": "ready",
                "items": _catalog_items(),
            }
        if path == "/api/capabilities":
            return 200, {"skills": {"status": "available"}}
        if path == "/v1/admin/readiness":
            assert expected == frozenset({200, 503})
            return 200, {
                "skills": _ready_counts(),
                "components": {"skills": {"status": "available"}},
            }
        if path == "/v1/admin/skills/rediscover":
            assert method == "POST"
            return 200, {**_ready_counts(), "invalid": []}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(smoke, "USER_TOKEN", "user-evidence")
    monkeypatch.setattr(smoke, "ADMIN_TOKEN", "admin-evidence")
    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    assert smoke.main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["catalog"]["items"] == [
        "grounded-document-analysis",
        "video-analysis",
    ]
    assert calls == [
        ("POST", "/v1/admin/skills/rediscover", "admin-evidence"),
        ("GET", "/api/skills", "user-evidence"),
        ("GET", "/api/capabilities", "user-evidence"),
        ("GET", "/v1/admin/readiness", "admin-evidence"),
    ]


def test_smoke_fails_on_the_wrong_catalog(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "USER_TOKEN", "user-evidence")
    monkeypatch.setattr(smoke, "ADMIN_TOKEN", "admin-evidence")
    def fake_json_request(path, **_kwargs):
        if path == "/v1/admin/skills/rediscover":
            return 200, {**_ready_counts(), "invalid": []}
        if path == "/api/skills":
            return 200, {
                "enabled": True,
                "status": "ready",
                "items": [{"id": "unexpected"}],
            }
        raise AssertionError(f"unexpected request after wrong catalog: {path}")

    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    assert smoke.main() == 1

    assert "skill catalog mismatch" in capsys.readouterr().err
