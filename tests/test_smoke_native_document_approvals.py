"""Contract tests for the two-stage document-approval smoke."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SMOKE_DIR = Path(__file__).resolve().parent / "smoke"
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

import smoke_native_document_approvals as smoke  # noqa: E402


def test_readiness_uses_backend_health_and_ui_capabilities(monkeypatch):
    calls: list[tuple[str, str, str]] = []

    def fake_json_request(path, *, token, absolute_url=""):
        calls.append((path, token, absolute_url))
        if absolute_url:
            return 200, {"status": "ok"}
        return 200, {"status": "ready"}

    monkeypatch.setattr(smoke, "BACKEND_HEALTH_URL", "http://audrey:8000/health")
    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    assert smoke._wait_ready() == {
        "health": "ok",
        "capabilities": "ready",
    }
    assert calls == [
        ("", smoke.USER_TOKEN, "http://audrey:8000/health"),
        ("/api/capabilities", smoke.USER_TOKEN, ""),
    ]


def test_readiness_fails_immediately_when_access_assertion_is_rejected(monkeypatch):
    calls: list[str] = []

    def fake_json_request(path, *, token, absolute_url=""):
        calls.append(absolute_url or path)
        if absolute_url:
            return 200, {"status": "ok"}
        raise smoke.SmokeCredentialError(
            "HTTP 401; refresh application assertions in .env.smoke.local"
        )

    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    with pytest.raises(smoke.SmokeCredentialError, match=r"\.env\.smoke\.local"):
        smoke._wait_ready()

    assert calls == [
        smoke.BACKEND_HEALTH_URL,
        "/api/capabilities",
    ]
