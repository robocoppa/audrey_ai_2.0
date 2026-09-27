"""The 3A.1 smoke is narrow, non-mutating, and contract-focused."""

from __future__ import annotations

import json
from typing import Any

from scripts import smoke_skills_foundation as smoke


def _disabled_counts() -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "disabled",
        "loaded_count": 0,
        "available_count": 0,
        "degraded_count": 0,
        "invalid_count": 0,
    }


def test_smoke_checks_only_the_registry_foundation(monkeypatch, capsys):
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
            return 200, {"enabled": False, "status": "disabled", "items": []}
        if path == "/api/capabilities":
            return 200, {"skills": {"status": "disabled"}}
        if path == "/v1/admin/readiness":
            assert expected == frozenset({200, 503})
            return 200, {
                "skills": _disabled_counts(),
                "components": {"skills": {"status": "disabled"}},
            }
        if path == "/v1/admin/skills/rediscover":
            assert method == "POST"
            return 200, {**_disabled_counts(), "invalid": []}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(smoke, "USER_TOKEN", "user-evidence")
    monkeypatch.setattr(smoke, "ADMIN_TOKEN", "admin-evidence")
    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    assert smoke.main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert calls == [
        ("GET", "/api/skills", "user-evidence"),
        ("GET", "/api/capabilities", "user-evidence"),
        ("GET", "/v1/admin/readiness", "admin-evidence"),
        ("POST", "/v1/admin/skills/rediscover", "admin-evidence"),
    ]


def test_smoke_fails_on_enabled_or_nonempty_catalog(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "USER_TOKEN", "user-evidence")
    monkeypatch.setattr(smoke, "ADMIN_TOKEN", "admin-evidence")
    monkeypatch.setattr(
        smoke,
        "_json_request",
        lambda *_args, **_kwargs: (
            200,
            {"enabled": True, "status": "ready", "items": [{"id": "unexpected"}]},
        ),
    )

    assert smoke.main() == 1

    assert "catalog was not disabled and empty" in capsys.readouterr().err
