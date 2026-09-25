#!/usr/bin/env python3
"""Exercise deployed native account groups and model access controls.

Run this through the standalone Audrey UI proxy with an ordinary-user provider
token and a different provider-authenticated administrator token. The script
temporarily makes the ordinary account a tester, runs one direct model turn,
then restores the account state plus the model policy's source and values.

A replayed Access assertion verifies origin auth, but the first browser login
and Cloudflare handoff remain a separate live gate.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin


def _auth_headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(token, user_token=USER_TOKEN, admin_token=ADMIN_TOKEN)


DIRECT_MODEL_ID = os.getenv(
    "AUDREY_DIRECT_SMOKE_MODEL_ID",
    "direct/qwen3.8:latest",
).strip()
RUN_TIMEOUT_SECONDS = float(os.getenv("AUDREY_DIRECT_SMOKE_TIMEOUT_SECONDS", "600"))


class SmokeError(RuntimeError):
    """A deployed access or model-publication contract was not satisfied."""


def _request(
    path: str,
    *,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 300,
) -> tuple[int, bytes, Message]:
    headers = {"Accept": "application/json"}
    if token:
        headers.update(_auth_headers(token))
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
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
        raise SmokeError(f"{method} {path}: HTTP {status}: {excerpt}")
    return status, content, response_headers


def _json_request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 300,
) -> tuple[int, dict[str, Any]]:
    status, content, _headers = _request(
        path,
        token=token,
        method=method,
        payload=payload,
        expected=expected,
        timeout=timeout,
    )
    return status, json.loads(content) if content else {}


def _admin_user(user_id: str) -> dict[str, Any]:
    _, page = _json_request("/api/admin/users", token=ADMIN_TOKEN)
    user = next(
        (item for item in page.get("items", []) if item.get("id") == user_id),
        None,
    )
    if user is None:
        raise SmokeError(f"admin catalog omitted Audrey user {user_id}")
    return user


def _admin_model(model_id: str) -> dict[str, Any]:
    _, page = _json_request("/api/admin/models", token=ADMIN_TOKEN)
    model = next(
        (item for item in page.get("items", []) if item.get("id") == model_id),
        None,
    )
    if model is None:
        direct_ids = sorted(
            str(item.get("id")) for item in page.get("items", []) if item.get("kind") == "direct"
        )
        raise SmokeError(f"admin catalog omitted {model_id}; deployed direct models: {direct_ids}")
    return model


def _patch_user(
    user_id: str,
    *,
    status: str,
    groups: list[str],
) -> dict[str, Any]:
    _, updated = _json_request(
        f"/api/admin/users/{user_id}",
        token=ADMIN_TOKEN,
        method="PATCH",
        payload={"status": status, "groups": groups},
    )
    return updated


def _patch_model(
    model_id: str,
    *,
    enabled: bool,
    audience: str,
) -> dict[str, Any]:
    _, updated = _json_request(
        f"/api/admin/models/{model_id}",
        token=ADMIN_TOKEN,
        method="PATCH",
        payload={"enabled": enabled, "audience": audience},
    )
    return updated


def _delete_model_policy(model_id: str) -> dict[str, Any]:
    _, updated = _json_request(
        f"/api/admin/model-policies/{model_id}",
        token=ADMIN_TOKEN,
        method="DELETE",
    )
    return updated


def _catalog_model(*, token: str, model_id: str) -> dict[str, Any] | None:
    _, page = _json_request("/api/models", token=token)
    return next(
        (item for item in page.get("items", []) if item.get("id") == model_id),
        None,
    )


def _detail(content: bytes) -> str:
    try:
        value = json.loads(content).get("detail", "")
    except (json.JSONDecodeError, AttributeError):
        value = content.decode(errors="replace")
    return str(value)


def _assert_admin_safeguards(admin_id: str) -> dict[str, int]:
    disabled, disabled_content, _ = _request(
        f"/api/admin/users/{admin_id}",
        token=ADMIN_TOKEN,
        method="PATCH",
        payload={"status": "disabled"},
        expected=frozenset({409}),
    )
    demoted, demoted_content, _ = _request(
        f"/api/admin/users/{admin_id}",
        token=ADMIN_TOKEN,
        method="PATCH",
        payload={"groups": ["users"]},
        expected=frozenset({409}),
    )
    if "own account" not in _detail(disabled_content):
        raise SmokeError("self-disable was refused without the expected safeguard")
    if "own admin access" not in _detail(demoted_content):
        raise SmokeError("self-demotion was refused without the expected safeguard")
    return {"self_disable_http": disabled, "self_demote_http": demoted}


def _issue_admin_pat() -> tuple[str, str]:
    _, issued = _json_request(
        "/api/tokens",
        token=ADMIN_TOKEN,
        method="POST",
        payload={
            "name": f"2D5 access smoke {uuid.uuid4().hex[:12]}",
            "scopes": ["compat:full"],
            "expires_in_days": 1,
        },
        expected=frozenset({201}),
    )
    token_id = str(issued.get("id") or "")
    token = str(issued.get("token") or "")
    if not token_id or not token:
        raise SmokeError("personal-token creation omitted its id or one-time secret")
    return token_id, token


def _assert_pat_cannot_admin(token: str) -> int:
    status, content, _ = _request(
        "/api/admin/users",
        token=token,
        expected=frozenset({403}),
    )
    if "provider authentication" not in _detail(content).lower():
        raise SmokeError("admin PAT was rejected without the provider-only boundary")
    return status


def _direct_agent_turn(
    *,
    conversation_id: str,
    marker: str,
) -> tuple[str, list[dict[str, Any]], str, str]:
    prompt = f"Reply with this exact verification marker and nothing else: {marker}"
    body = json.dumps(
        {
            "threadId": conversation_id,
            "runId": f"native-direct-smoke-{uuid.uuid4().hex}",
            "messages": [
                {
                    "id": f"native-direct-message-{uuid.uuid4().hex}",
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
    ).encode()
    query = urlencode({"model": DIRECT_MODEL_ID})
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}/api/agent?{query}",
        data=body,
        headers={
            "Accept": "text/event-stream",
            **_auth_headers(USER_TOKEN),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    events: list[dict[str, Any]] = []
    run_id = ""
    try:
        with urlopen(request, timeout=RUN_TIMEOUT_SECONDS) as response:  # noqa: S310
            if response.status != 200:
                raise SmokeError(f"POST /api/agent: HTTP {response.status}")
            run_id = str(response.headers.get("X-Audrey-Run-ID") or "")
            for raw_line in response:
                line = raw_line.decode(errors="replace").rstrip("\r\n")
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    except HTTPError as exc:
        excerpt = exc.read().decode(errors="replace")[:500]
        raise SmokeError(f"POST /api/agent: HTTP {exc.code}: {excerpt}") from exc
    if not run_id:
        raise SmokeError("POST /api/agent omitted X-Audrey-Run-ID")

    counts = Counter(str(event.get("type") or "") for event in events)
    terminal = events[-1] if events else {}
    answer = "".join(
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "TEXT_MESSAGE_CONTENT"
    )
    if counts["RUN_STARTED"] != 1 or counts["TEXT_MESSAGE_START"] != 1:
        raise SmokeError(f"direct AG-UI start events were invalid: {dict(counts)}")
    if not (
        terminal.get("type") == "RUN_FINISHED"
        and terminal.get("outcome", {}).get("type") == "success"
    ):
        raise SmokeError(f"direct AG-UI run did not finish successfully: {terminal}")
    tool_events = sorted(name for name in counts if name.startswith("TOOL_CALL_"))
    if tool_events:
        raise SmokeError(f"direct model unexpectedly emitted tool events: {tool_events}")
    if marker not in answer:
        raise SmokeError(f"direct model answer omitted {marker}: {answer!r}")
    return run_id, events, prompt, answer


def _repair_until_ready() -> str:
    _json_request(
        "/v1/admin/repair",
        token=ADMIN_TOKEN,
        method="POST",
        expected=frozenset({202}),
    )
    deadline = time.monotonic() + 180
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, last = _json_request("/v1/admin/repair-status", token=ADMIN_TOKEN)
        if last.get("status") == "ready":
            return "ready"
        time.sleep(0.5)
    raise SmokeError(f"repair queues did not become ready: {last}")


def _cleanup_step(
    result: dict[str, Any],
    errors: list[str],
    name: str,
    action: Callable[[], Any],
) -> None:
    try:
        result[name] = action()
    except Exception as exc:  # noqa: BLE001 - every cleanup step must still run
        message = f"{type(exc).__name__}: {exc}"
        result[f"{name}_error"] = message
        errors.append(f"{name}: {message}")


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(MISSING_CREDENTIALS, file=sys.stderr)
        return 2
    if not DIRECT_MODEL_ID:
        print("AUDREY_DIRECT_SMOKE_MODEL_ID must not be empty.", file=sys.stderr)
        return 2
    if RUN_TIMEOUT_SECONDS <= 0:
        print("AUDREY_DIRECT_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"schema": 1, "direct_model_id": DIRECT_MODEL_ID}
    original_user: dict[str, Any] | None = None
    original_model: dict[str, Any] | None = None
    user_restore_needed = False
    model_restore_needed = False
    pat_id = ""
    conversation_id = ""
    run_id = ""
    primary_error: Exception | None = None
    try:
        _, user = _json_request("/api/me", token=USER_TOKEN)
        _, admin = _json_request("/api/me", token=ADMIN_TOKEN)
        user_id = str(user.get("id") or "")
        admin_id = str(admin.get("id") or "")
        if not user_id or not admin_id or user_id == admin_id:
            raise SmokeError("smoke tokens must resolve to two different Audrey users")

        original_user = _admin_user(user_id)
        admin_record = _admin_user(admin_id)
        original_model = _admin_model(DIRECT_MODEL_ID)
        if original_user.get("status") != "active":
            raise SmokeError("the ordinary smoke account must already be active")
        if "users" not in original_user.get("groups", []):
            raise SmokeError("the ordinary smoke account must belong to users")
        if "admins" in original_user.get("groups", []):
            raise SmokeError("the user smoke credential must belong to a non-admin account")
        if "admins" not in admin_record.get("groups", []):
            raise SmokeError("the admin smoke credential does not resolve to an Audrey admin")
        if original_model.get("kind") != "direct":
            raise SmokeError(f"{DIRECT_MODEL_ID} is not a direct model")
        if not isinstance(original_model.get("policy_overridden"), bool):
            raise SmokeError("admin model catalog omitted its policy source")

        safeguards = _assert_admin_safeguards(admin_id)
        pat_id, pat = _issue_admin_pat()
        pat_rejection = _assert_pat_cannot_admin(pat)

        user_restore_needed = True
        ordinary = _patch_user(user_id, status="active", groups=["users"])
        if ordinary.get("groups") != ["users"]:
            raise SmokeError(f"ordinary group update did not round-trip: {ordinary}")

        model_restore_needed = True
        tester_policy = _patch_model(
            DIRECT_MODEL_ID,
            enabled=True,
            audience="testers",
        )
        if not tester_policy.get("enabled") or tester_policy.get("audience") != "testers":
            raise SmokeError(f"tester model policy did not round-trip: {tester_policy}")
        if _catalog_model(token=USER_TOKEN, model_id=DIRECT_MODEL_ID) is not None:
            raise SmokeError("ordinary account discovered the tester-only direct model")
        ordinary_denial, _, _ = _request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={"model_id": DIRECT_MODEL_ID},
            expected=frozenset({404}),
        )

        tester = _patch_user(
            user_id,
            status="active",
            groups=["users", "testers"],
        )
        if set(tester.get("groups", [])) != {"users", "testers"}:
            raise SmokeError(f"tester group update did not round-trip: {tester}")
        visible_model = _catalog_model(token=USER_TOKEN, model_id=DIRECT_MODEL_ID)
        if visible_model is None:
            raise SmokeError("tester catalog omitted its enabled direct model")
        if visible_model.get("kind") != "direct" or not visible_model.get("enabled"):
            raise SmokeError(f"tester catalog returned an invalid model: {visible_model}")
        if "concrete_model" in visible_model:
            raise SmokeError("ordinary model catalog exposed the admin-only concrete field")

        _, conversation = _json_request(
            "/api/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={
                "title": "C3 2D5 DIRECT MODEL SMOKE",
                "model_id": DIRECT_MODEL_ID,
            },
            expected=frozenset({201}),
        )
        conversation_id = str(conversation.get("id") or "")
        if conversation.get("default_mode") != "direct":
            raise SmokeError(f"direct conversation persisted the wrong mode: {conversation}")
        if conversation.get("default_model_id") != DIRECT_MODEL_ID:
            raise SmokeError(f"direct conversation lost its stable model id: {conversation}")

        marker = f"AUDREY-DIRECT-{uuid.uuid4().hex.upper()}"
        started_at = time.monotonic()
        run_id, events, prompt, answer = _direct_agent_turn(
            conversation_id=conversation_id,
            marker=marker,
        )
        _, persisted = _json_request(f"/api/runs/{run_id}", token=USER_TOKEN)
        concrete_model = str(original_model.get("concrete_model") or "")
        expected_virtual = f"audrey_passthrough/{concrete_model}"
        if persisted.get("status") != "succeeded":
            raise SmokeError(f"direct run did not persist success: {persisted}")
        if persisted.get("mode") != "direct":
            raise SmokeError(f"direct run persisted mode {persisted.get('mode')!r}")
        if persisted.get("requested_model_id") != DIRECT_MODEL_ID:
            raise SmokeError("direct run lost its requested stable model id")
        if persisted.get("concrete_model") != concrete_model:
            raise SmokeError("direct run persisted the wrong concrete model")
        if persisted.get("virtual_model") != expected_virtual:
            raise SmokeError("direct run did not use its passthrough protocol model")

        _, message_page = _json_request(
            f"/api/conversations/{conversation_id}/messages?limit=100",
            token=USER_TOKEN,
        )
        messages = message_page.get("items", [])
        if len(messages) != 2:
            raise SmokeError(f"direct conversation was not exactly one turn: {messages}")
        if messages[0].get("content") != prompt or messages[1].get("content") != answer:
            raise SmokeError("canonical messages differed from the direct AG-UI turn")

        disabled_policy = _patch_model(
            DIRECT_MODEL_ID,
            enabled=False,
            audience="testers",
        )
        if disabled_policy.get("enabled") is not False:
            raise SmokeError(f"disabled model policy did not round-trip: {disabled_policy}")
        if _catalog_model(token=USER_TOKEN, model_id=DIRECT_MODEL_ID) is not None:
            raise SmokeError("disabled direct model remained visible to its tester")
        disabled_denial, _, _ = _request(
            "/api/agent",
            token=USER_TOKEN,
            method="POST",
            payload={
                "threadId": conversation_id,
                "runId": f"disabled-model-{uuid.uuid4().hex}",
                "messages": [
                    {
                        "id": f"disabled-model-message-{uuid.uuid4().hex}",
                        "role": "user",
                        "content": "This turn must be refused.",
                    }
                ],
            },
            expected=frozenset({404}),
        )
        _, messages_after_denial = _json_request(
            f"/api/conversations/{conversation_id}/messages?limit=100",
            token=USER_TOKEN,
        )
        if messages_after_denial.get("items") != messages:
            raise SmokeError("disabled-model denial appended canonical messages")

        result.update(
            {
                "identity": {
                    "user_id": user_id,
                    "admin_id": admin_id,
                    "provider_admin_route": True,
                    "pat_admin_http": pat_rejection,
                },
                "safeguards": safeguards,
                "access": {
                    "ordinary_create_http": ordinary_denial,
                    "tester_catalog_visible": True,
                    "disabled_agent_http": disabled_denial,
                    "disabled_denial_preserved_messages": True,
                },
                "run": {
                    "run_id": run_id,
                    "concrete_model": concrete_model,
                    "elapsed_seconds": round(time.monotonic() - started_at, 3),
                    "answer_chars": len(answer),
                    "event_counts": dict(Counter(str(event.get("type") or "") for event in events)),
                    "tool_events": 0,
                    "canonical_messages": len(messages),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup: dict[str, Any] = {}
        cleanup_errors: list[str] = []
        if run_id:
            _cleanup_step(
                cleanup,
                cleanup_errors,
                "run_cancel_http",
                lambda: _request(
                    f"/api/runs/{run_id}/cancel",
                    token=USER_TOKEN,
                    method="POST",
                    expected=frozenset({200, 404}),
                )[0],
            )
        if conversation_id:
            _cleanup_step(
                cleanup,
                cleanup_errors,
                "canonical_delete_http",
                lambda: _request(
                    f"/api/conversations/{conversation_id}",
                    token=USER_TOKEN,
                    method="DELETE",
                    expected=frozenset({204, 404}),
                )[0],
            )
            _cleanup_step(
                cleanup,
                cleanup_errors,
                "archive_delete_http",
                lambda: _request(
                    f"/v1/me/chat-history/{conversation_id}",
                    token=USER_TOKEN,
                    method="DELETE",
                    expected=frozenset({202, 404}),
                )[0],
            )
            _cleanup_step(
                cleanup,
                cleanup_errors,
                "repair_status",
                _repair_until_ready,
            )
        if model_restore_needed and original_model is not None:

            def restore_model() -> bool:
                if original_model.get("policy_overridden"):
                    restored = _patch_model(
                        DIRECT_MODEL_ID,
                        enabled=bool(original_model.get("enabled")),
                        audience=str(original_model.get("audience") or "admins"),
                    )
                else:
                    restored = _delete_model_policy(DIRECT_MODEL_ID)
                return (
                    restored.get("enabled") == original_model.get("enabled")
                    and restored.get("audience") == original_model.get("audience")
                    and restored.get("policy_overridden") == original_model.get("policy_overridden")
                )

            _cleanup_step(
                cleanup,
                cleanup_errors,
                "model_policy_restored",
                restore_model,
            )
            if cleanup.get("model_policy_restored") is False:
                cleanup_errors.append("model_policy_restored: effective policy differs")
        if user_restore_needed and original_user is not None:

            def restore_user() -> bool:
                restored = _patch_user(
                    str(original_user["id"]),
                    status=str(original_user["status"]),
                    groups=[str(group) for group in original_user.get("groups", [])],
                )
                return restored.get("status") == original_user.get("status") and restored.get(
                    "groups"
                ) == original_user.get("groups")

            _cleanup_step(
                cleanup,
                cleanup_errors,
                "user_access_restored",
                restore_user,
            )
            if cleanup.get("user_access_restored") is False:
                cleanup_errors.append("user_access_restored: account state differs")
        if pat_id:
            _cleanup_step(
                cleanup,
                cleanup_errors,
                "pat_revoke_http",
                lambda: _request(
                    f"/api/tokens/{pat_id}",
                    token=ADMIN_TOKEN,
                    method="DELETE",
                    expected=frozenset({200, 404}),
                )[0],
            )
        result["cleanup"] = cleanup
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            if primary_error is None:
                primary_error = SmokeError("; ".join(cleanup_errors))

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
