"""Contract tests for the two-stage restart-persistence smoke."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_SMOKE_DIR = Path(__file__).resolve().parent / "smoke"
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

import smoke_native_restart_persistence as smoke  # noqa: E402


def _state() -> dict:
    return {
        "accounts": {
            "owner": {
                "id": "usr_admin",
                "email": "admin@example.com",
                "role": "admin",
                "status": "active",
                "groups": ["admins", "users"],
                "auth_provider": "cloudflare_access",
            },
            "ordinary": {
                "id": "usr_user",
                "email": "alice@example.com",
                "role": "user",
                "status": "active",
                "groups": ["users"],
                "auth_provider": "cloudflare_access",
            },
        },
        "models": [
            {
                "id": "auto",
                "kind": "workflow",
                "enabled": True,
                "audience": "users",
                "visibility": "public",
                "roles": ["users"],
                "policy_overridden": False,
                "access_policy_overridden": False,
                "publication_profile_overridden": False,
                "profile_display_name": "",
            },
            {
                "id": "direct/qwen:latest",
                "kind": "direct",
                "enabled": True,
                "audience": "testers",
                "visibility": "public",
                "roles": ["testers"],
                "policy_overridden": True,
                "access_policy_overridden": True,
                "publication_profile_overridden": True,
                "profile_display_name": "Qwen",
            },
        ],
        "conversation": {
            "id": "con_keep",
            "default_mode": "auto",
            "default_model_id": "auto",
            "archived_at": None,
        },
    }


def test_capture_then_verify_preserves_every_stable_surface(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.json"
    state = _state()
    monkeypatch.setattr(smoke, "SNAPSHOT_PATH", snapshot)
    monkeypatch.setattr(
        smoke,
        "_wait_until_ready",
        lambda: {"health": "ok", "capabilities": "ready"},
    )
    monkeypatch.setattr(smoke, "_collect_state", lambda **_kwargs: deepcopy(state))

    captured = smoke.capture()
    verified = smoke.verify()

    assert captured["status"] == "captured"
    assert verified["status"] == "passed"
    assert all(verified["checks"].values())
    assert snapshot.stat().st_mode & 0o777 == 0o600
    assert json.loads(snapshot.read_text())["state"] == state


def test_verify_reports_model_policy_or_order_drift(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.json"
    state = _state()
    monkeypatch.setattr(smoke, "SNAPSHOT_PATH", snapshot)
    monkeypatch.setattr(
        smoke,
        "_wait_until_ready",
        lambda: {"health": "ok", "capabilities": "ready"},
    )
    monkeypatch.setattr(smoke, "_collect_state", lambda **_kwargs: deepcopy(state))
    smoke.capture()

    changed = deepcopy(state)
    changed["models"].reverse()
    monkeypatch.setattr(smoke, "_collect_state", lambda **_kwargs: changed)

    with pytest.raises(smoke.SmokeError, match="model_policy_and_order"):
        smoke.verify()


def test_stable_account_rejects_non_active_state():
    with pytest.raises(smoke.SmokeError, match="not active"):
        smoke._stable_account(
            {
                "id": "usr_pending",
                "email": "alice@example.com",
                "role": "pending",
                "status": "pending",
                "groups": [],
                "auth_provider": "cloudflare_access",
            },
            label="ordinary",
        )


def test_readiness_uses_backend_health_and_ui_capabilities(monkeypatch):
    calls = []

    def fake_json_request(path, *, token="", absolute_url=""):
        calls.append((path, token, absolute_url))
        if absolute_url:
            return {"status": "ok"}
        return {"status": "ready"}

    monkeypatch.setattr(smoke, "BACKEND_HEALTH_URL", "http://audrey:8000/health")
    monkeypatch.setattr(smoke, "_json_request", fake_json_request)

    assert smoke._wait_until_ready() == {
        "health": "ok",
        "capabilities": "ready",
    }
    assert calls == [
        ("", "", "http://audrey:8000/health"),
        ("/api/capabilities", smoke.USER_TOKEN, ""),
    ]
