"""Control-flow contract for the self-restoring 2D.5 live smoke."""

from __future__ import annotations

import json
from email.message import Message
from typing import Any

from scripts import smoke_native_access_models as smoke

_ADMIN_EVIDENCE = "admin-provider"
_USER_EVIDENCE = "user-provider"
_PAT_EVIDENCE = "pat-secret"


class _FakeDeployment:
    def __init__(self) -> None:
        self.user = {
            "id": "usr_test",
            "status": "active",
            "groups": ["testers", "users"],
        }
        self.admin = {
            "id": "usr_admin",
            "status": "active",
            "groups": ["admins", "users"],
        }
        self.model = {
            "id": smoke.DIRECT_MODEL_ID,
            "kind": "direct",
            "enabled": False,
            "audience": "admins",
            "concrete_model": "qwen3.8:latest",
            "policy_overridden": False,
        }
        self.messages: list[dict[str, Any]] = []
        self.pat_revoked = False
        self.conversation_deleted = False
        self.archive_deleted = False

    def json_request(
        self,
        path: str,
        *,
        token: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        expected: frozenset[int] = frozenset({200}),
        timeout: float = 300,
    ) -> tuple[int, dict[str, Any]]:
        del expected, timeout
        if path == "/api/me":
            record = self.admin if token == _ADMIN_EVIDENCE else self.user
            return 200, record.copy()
        if path == "/api/admin/users" and method == "GET":
            return 200, {"items": [self.user.copy(), self.admin.copy()]}
        if path == "/api/admin/models":
            return 200, {"items": [self.model.copy()]}
        if path == "/api/tokens" and method == "POST":
            return 201, {"id": "pat_smoke", "token": "pat-secret"}
        if path == f"/api/admin/users/{self.user['id']}" and method == "PATCH":
            assert payload is not None
            self.user["status"] = payload["status"]
            self.user["groups"] = sorted(payload["groups"])
            return 200, self.user.copy()
        if path == f"/api/admin/models/{smoke.DIRECT_MODEL_ID}" and method == "PATCH":
            assert payload is not None
            self.model["enabled"] = payload["enabled"]
            self.model["audience"] = payload["audience"]
            self.model["policy_overridden"] = True
            return 200, self.model.copy()
        if (
            path == f"/api/admin/model-policies/{smoke.DIRECT_MODEL_ID}"
            and method == "DELETE"
        ):
            self.model["enabled"] = False
            self.model["audience"] = "admins"
            self.model["policy_overridden"] = False
            return 200, self.model.copy()
        if path == "/api/models":
            visible = (
                self.model["enabled"]
                and "testers" in self.user["groups"]
                and self.model["audience"] == "testers"
            )
            model = {
                key: value
                for key, value in self.model.items()
                if key not in {"concrete_model", "policy_overridden"}
            }
            return 200, {"items": [model] if visible else []}
        if path == "/api/conversations" and method == "POST":
            assert payload is not None
            assert payload["model_id"] == smoke.DIRECT_MODEL_ID
            return 201, {
                "id": "con_smoke",
                "default_mode": "direct",
                "default_model_id": smoke.DIRECT_MODEL_ID,
            }
        if path == "/api/runs/run_smoke":
            return 200, {
                "status": "succeeded",
                "mode": "direct",
                "requested_model_id": smoke.DIRECT_MODEL_ID,
                "concrete_model": "qwen3.8:latest",
                "virtual_model": "audrey_passthrough/qwen3.8:latest",
            }
        if path == "/api/conversations/con_smoke/messages?limit=100":
            return 200, {"items": list(self.messages)}
        raise AssertionError(f"unexpected JSON request: {method} {path}")

    def request(
        self,
        path: str,
        *,
        token: str = "",
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        expected: frozenset[int] = frozenset({200}),
        timeout: float = 300,
    ) -> tuple[int, bytes, Message]:
        del expected, timeout
        headers = Message()
        if path == f"/api/admin/users/{self.admin['id']}" and method == "PATCH":
            assert token == _ADMIN_EVIDENCE
            detail = (
                "an administrator cannot disable their own account"
                if payload is not None and payload.get("status") == "disabled"
                else "an administrator cannot remove their own admin access"
            )
            return 409, json.dumps({"detail": detail}).encode(), headers
        if path == "/api/admin/users" and token == _PAT_EVIDENCE:
            detail = "External provider authentication is required for this operation."
            return 403, json.dumps({"detail": detail}).encode(), headers
        if path == "/api/conversations" and method == "POST":
            return 404, b'{"detail":"Model is not available."}', headers
        if path == "/api/agent" and method == "POST":
            return 404, b'{"detail":"Model is not available."}', headers
        if path == "/api/runs/run_smoke/cancel" and method == "POST":
            return 200, b"{}", headers
        if path == "/api/conversations/con_smoke" and method == "DELETE":
            self.conversation_deleted = True
            return 204, b"", headers
        if path == "/v1/me/chat-history/con_smoke" and method == "DELETE":
            self.archive_deleted = True
            return 202, b"{}", headers
        if path == "/api/tokens/pat_smoke" and method == "DELETE":
            self.pat_revoked = True
            return 200, b"{}", headers
        raise AssertionError(f"unexpected request: {method} {path}")

    def direct_turn(
        self,
        *,
        conversation_id: str,
        marker: str,
    ) -> tuple[str, list[dict[str, Any]], str, str]:
        assert conversation_id == "con_smoke"
        prompt = f"Reply with this exact verification marker and nothing else: {marker}"
        answer = marker
        self.messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        events = [
            {"type": "RUN_STARTED"},
            {"type": "TEXT_MESSAGE_START"},
            {"type": "TEXT_MESSAGE_CONTENT", "delta": answer},
            {"type": "TEXT_MESSAGE_END"},
            {"type": "RUN_FINISHED", "outcome": {"type": "success"}},
        ]
        return "run_smoke", events, prompt, answer


def test_smoke_restores_state_and_removes_disposable_records(monkeypatch, capsys):
    deployment = _FakeDeployment()
    original_user = deployment.user.copy()
    original_user["groups"] = list(deployment.user["groups"])
    original_model = deployment.model.copy()
    monkeypatch.setattr(smoke, "USER_TOKEN", _USER_EVIDENCE)
    monkeypatch.setattr(smoke, "ADMIN_TOKEN", _ADMIN_EVIDENCE)
    monkeypatch.setattr(smoke, "_json_request", deployment.json_request)
    monkeypatch.setattr(smoke, "_request", deployment.request)
    monkeypatch.setattr(smoke, "_direct_agent_turn", deployment.direct_turn)
    monkeypatch.setattr(smoke, "_repair_until_ready", lambda: "ready")

    assert smoke.main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["access"] == {
        "disabled_agent_http": 404,
        "disabled_denial_preserved_messages": True,
        "ordinary_create_http": 404,
        "tester_catalog_visible": True,
    }
    assert result["cleanup"]["model_policy_restored"] is True
    assert result["cleanup"]["user_access_restored"] is True
    assert deployment.user == original_user
    assert deployment.model == original_model
    assert deployment.conversation_deleted
    assert deployment.archive_deleted
    assert deployment.pat_revoked
