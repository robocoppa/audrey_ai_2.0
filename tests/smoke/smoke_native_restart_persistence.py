#!/usr/bin/env python3
"""Capture and verify native application state across an Audrey restart.

Run ``capture`` before restarting the Audrey container, then run ``verify``
afterward. The snapshot contains only stable application-owned fields and is
written mode 600 to the runner's persistent state mount.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SNAPSHOT_PATH = Path(
    os.getenv(
        "AUDREY_PERSISTENCE_SNAPSHOT_PATH",
        "/state/c3-restart-persistence.json",
    )
)
READY_TIMEOUT_SECONDS = float(
    os.getenv("AUDREY_RESTART_SMOKE_TIMEOUT_SECONDS", "180")
)
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin


class SmokeError(RuntimeError):
    """A deployed restart-persistence contract was not satisfied."""


def _auth_headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        token,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


def _request(
    path: str,
    *,
    token: str = "",
    timeout: float = 30,
    expected: frozenset[int] = frozenset({200}),
) -> tuple[int, bytes, Message]:
    headers = {"Accept": "application/json"}
    if token:
        headers.update(_auth_headers(token))
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}",
        headers=headers,
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = response.status
            content = response.read()
            response_headers = response.headers
    except HTTPError as exc:
        status = exc.code
        content = exc.read()
        response_headers = exc.headers
    if status not in expected:
        excerpt = content.decode(errors="replace")[:500]
        raise SmokeError(f"GET {path}: HTTP {status}: {excerpt}")
    return status, content, response_headers


def _json_request(path: str, *, token: str = "") -> dict[str, Any]:
    _status, content, _headers = _request(path, token=token)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SmokeError(f"GET {path} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"GET {path} returned a non-object JSON document")
    return value


def _stable_account(record: dict[str, Any], *, label: str) -> dict[str, Any]:
    account = {
        "id": str(record.get("id") or ""),
        "email": str(record.get("email") or ""),
        "role": str(record.get("role") or ""),
        "status": str(record.get("status") or ""),
        "groups": sorted(str(group) for group in record.get("groups", [])),
        "auth_provider": str(record.get("auth_provider") or ""),
    }
    if not account["id"] or not account["email"]:
        raise SmokeError(f"{label} account omitted its stable identity")
    if account["status"] != "active":
        raise SmokeError(f"{label} account is not active: {account['status']!r}")
    return account


def _stable_model(record: dict[str, Any]) -> dict[str, Any]:
    model = {
        "id": str(record.get("id") or ""),
        "kind": str(record.get("kind") or ""),
        "enabled": record.get("enabled"),
        "audience": str(record.get("audience") or ""),
        "visibility": str(record.get("visibility") or ""),
        "roles": sorted(str(role) for role in record.get("roles", [])),
        "policy_overridden": record.get("policy_overridden"),
        "access_policy_overridden": record.get("access_policy_overridden"),
        "publication_profile_overridden": record.get(
            "publication_profile_overridden"
        ),
        "profile_display_name": str(record.get("profile_display_name") or ""),
    }
    if not model["id"] or model["kind"] not in {"workflow", "direct"}:
        raise SmokeError(f"admin model catalog returned an invalid model: {record}")
    if not isinstance(model["enabled"], bool):
        raise SmokeError(f"model {model['id']} omitted its enabled policy")
    if any(
        not isinstance(model[field], bool)
        for field in (
            "policy_overridden",
            "access_policy_overridden",
            "publication_profile_overridden",
        )
    ):
        raise SmokeError(f"model {model['id']} omitted override provenance")
    return model


def _stable_conversation(record: dict[str, Any]) -> dict[str, Any]:
    conversation = {
        "id": str(record.get("id") or ""),
        "default_mode": str(record.get("default_mode") or ""),
        "default_model_id": str(record.get("default_model_id") or ""),
        "archived_at": record.get("archived_at"),
    }
    if not conversation["id"] or not conversation["default_model_id"]:
        raise SmokeError("conversation omitted its id or selected model")
    return conversation


def _account_from_admin_page(
    page: dict[str, Any],
    *,
    user_id: str,
    label: str,
) -> dict[str, Any]:
    record = next(
        (item for item in page.get("items", []) if item.get("id") == user_id),
        None,
    )
    if not isinstance(record, dict):
        raise SmokeError(f"admin account catalog omitted the {label} account")
    return _stable_account(record, label=label)


def _collect_state(*, conversation_id: str = "") -> dict[str, Any]:
    user_me = _json_request("/api/me", token=USER_TOKEN)
    admin_me = _json_request("/api/me", token=ADMIN_TOKEN)
    user_id = str(user_me.get("id") or "")
    admin_id = str(admin_me.get("id") or "")
    if not user_id or not admin_id or user_id == admin_id:
        raise SmokeError("smoke credentials must resolve to two distinct accounts")

    account_page = _json_request("/api/admin/users", token=ADMIN_TOKEN)
    accounts = {
        "owner": _account_from_admin_page(
            account_page,
            user_id=admin_id,
            label="owner",
        ),
        "ordinary": _account_from_admin_page(
            account_page,
            user_id=user_id,
            label="ordinary",
        ),
    }
    if "admins" not in accounts["owner"]["groups"]:
        raise SmokeError("administrator smoke identity is not an Audrey administrator")
    if "admins" in accounts["ordinary"]["groups"]:
        raise SmokeError("ordinary smoke identity unexpectedly has administrator access")

    model_page = _json_request("/api/admin/models", token=ADMIN_TOKEN)
    models = [_stable_model(item) for item in model_page.get("items", [])]
    if not models:
        raise SmokeError("admin model catalog is empty")
    if not any(model["kind"] == "workflow" for model in models):
        raise SmokeError("admin model catalog omitted workflow models")

    if conversation_id:
        encoded_id = quote(conversation_id, safe="")
        conversation = _stable_conversation(
            _json_request(f"/api/conversations/{encoded_id}", token=USER_TOKEN)
        )
    else:
        conversation_page = _json_request(
            "/api/conversations?limit=100",
            token=USER_TOKEN,
        )
        items = conversation_page.get("items", [])
        record = next(
            (
                item
                for item in items
                if isinstance(item, dict) and item.get("last_message_at")
            ),
            items[0] if items else None,
        )
        if not isinstance(record, dict):
            raise SmokeError(
                "ordinary smoke account has no conversation to verify; "
                "send one native chat message and rerun capture"
            )
        conversation = _stable_conversation(record)

    return {
        "accounts": accounts,
        "models": models,
        "conversation": conversation,
    }


def _wait_until_ready() -> dict[str, Any]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = _json_request("/health")
            capabilities = _json_request("/api/capabilities", token=USER_TOKEN)
            if health.get("status") == "ok" and capabilities.get("status") == "ready":
                return {
                    "health": "ok",
                    "capabilities": "ready",
                }
            last_error = (
                f"health={health.get('status')!r}, "
                f"capabilities={capabilities.get('status')!r}"
            )
        except Exception as exc:  # noqa: BLE001 - startup failures are retried
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise SmokeError(
        f"Audrey did not become ready within {READY_TIMEOUT_SECONDS:g}s: {last_error}"
    )


def _write_snapshot(payload: dict[str, Any]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SNAPSHOT_PATH.with_suffix(SNAPSHOT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(SNAPSHOT_PATH)


def capture() -> dict[str, Any]:
    readiness = _wait_until_ready()
    state = _collect_state()
    payload = {
        "schema": 1,
        "captured_at": dt.datetime.now(dt.UTC).isoformat(),
        "state": state,
    }
    _write_snapshot(payload)
    workflow_order = [
        model["id"] for model in state["models"] if model["kind"] == "workflow"
    ]
    direct_order = [
        model["id"] for model in state["models"] if model["kind"] == "direct"
    ]
    return {
        "schema": 1,
        "status": "captured",
        "snapshot": str(SNAPSHOT_PATH),
        "accounts": {
            "owner_id": state["accounts"]["owner"]["id"],
            "ordinary_id": state["accounts"]["ordinary"]["id"],
        },
        "models": {
            "workflow_count": len(workflow_order),
            "direct_count": len(direct_order),
            "workflow_order": workflow_order,
            "direct_order": direct_order,
        },
        "conversation": state["conversation"],
        "readiness": readiness,
    }


def _load_snapshot() -> dict[str, Any]:
    try:
        payload = json.loads(SNAPSHOT_PATH.read_text())
    except FileNotFoundError as exc:
        raise SmokeError(
            f"snapshot does not exist: {SNAPSHOT_PATH}; run capture first"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SmokeError(f"snapshot is not valid JSON: {SNAPSHOT_PATH}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise SmokeError("snapshot schema is unsupported")
    if not isinstance(payload.get("state"), dict):
        raise SmokeError("snapshot omitted its state object")
    return payload


def verify() -> dict[str, Any]:
    payload = _load_snapshot()
    before = payload["state"]
    conversation_id = str(before.get("conversation", {}).get("id") or "")
    if not conversation_id:
        raise SmokeError("snapshot conversation omitted its id")
    readiness = _wait_until_ready()
    after = _collect_state(conversation_id=conversation_id)

    checks = {
        "owner_account": before.get("accounts", {}).get("owner")
        == after["accounts"]["owner"],
        "ordinary_account": before.get("accounts", {}).get("ordinary")
        == after["accounts"]["ordinary"],
        "model_policy_and_order": before.get("models") == after["models"],
        "conversation_model": before.get("conversation") == after["conversation"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SmokeError(
            "restart changed persistent state: " + ", ".join(failed)
        )
    return {
        "schema": 1,
        "status": "passed",
        "snapshot": str(SNAPSHOT_PATH),
        "captured_at": payload.get("captured_at"),
        "checks": checks,
        "conversation": after["conversation"],
        "model_count": len(after["models"]),
        "readiness": readiness,
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(MISSING_CREDENTIALS, file=sys.stderr)
        return 2
    if READY_TIMEOUT_SECONDS <= 0:
        print("AUDREY_RESTART_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2
    if len(sys.argv) != 2 or sys.argv[1] not in {"capture", "verify"}:
        print(
            "usage: smoke_native_restart_persistence.py capture|verify",
            file=sys.stderr,
        )
        return 2

    try:
        result = capture() if sys.argv[1] == "capture" else verify()
    except Exception as exc:  # noqa: BLE001 - emit one structured smoke failure
        result = {
            "schema": 1,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
