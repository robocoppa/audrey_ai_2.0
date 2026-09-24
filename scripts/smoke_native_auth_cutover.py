#!/usr/bin/env python3
"""Prove the deployed native identity paths no longer depend on Open WebUI.

The smoke requires an active Cloudflare Access application assertion. It checks
that an unknown legacy bearer is rejected locally, then creates a short-lived
Audrey personal token, exercises both native account and protected `/v1/files`
reads, revokes the token, and proves revocation is immediate.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import SmokeCredentials
else:
    from smoke_native_auth import SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin
_DISABLED_DETAIL = "Open WebUI bearer authentication is disabled."
_REVOKED_DETAIL = "Personal access token is invalid."


def _auth_headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        token,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


class SmokeError(RuntimeError):
    """A deployed authentication-cutover contract was not satisfied."""


def _request(
    path: str,
    *,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 60,
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
) -> tuple[int, dict[str, Any]]:
    status, content, _headers = _request(
        path,
        token=token,
        method=method,
        payload=payload,
        expected=expected,
    )
    return status, json.loads(content) if content else {}


def main() -> int:
    if not USER_TOKEN or not _CREDENTIALS.user_access:
        print(
            "Set AUDREY_SMOKE_USER_ACCESS_JWT to an active Cloudflare Access "
            "application assertion.",
            file=sys.stderr,
        )
        return 2

    result: dict[str, Any] = {"schema": 1}
    token_id = ""
    primary_error: Exception | None = None
    try:
        _, access_me = _json_request("/api/me", token=USER_TOKEN)
        if access_me.get("auth_provider") != "cloudflare_access":
            raise SmokeError(
                "the user assertion did not resolve through Cloudflare Access"
            )

        legacy_probe = f"owui-cutover-probe-{uuid.uuid4().hex}"
        legacy_status, legacy_body = _json_request(
            "/api/me",
            token=legacy_probe,
            expected=frozenset({401}),
        )
        if legacy_body.get("detail") != _DISABLED_DETAIL:
            raise SmokeError(
                "unknown bearer was not rejected by the local OWUI cutover gate: "
                f"{legacy_body}"
            )

        _, issued = _json_request(
            "/api/tokens",
            token=USER_TOKEN,
            method="POST",
            payload={
                "name": f"2F cutover smoke {uuid.uuid4().hex[:8]}",
                "scopes": ["account:read", "compat:full"],
                "expires_in_days": 1,
            },
            expected=frozenset({201}),
        )
        token_id = str(issued.get("id") or "")
        personal_token = str(issued.get("token") or "")
        if not token_id or not personal_token.startswith("aud_pat_"):
            raise SmokeError("token creation omitted the Audrey token id or secret")

        _, token_me = _json_request("/api/me", token=personal_token)
        if token_me.get("id") != access_me.get("id"):
            raise SmokeError("personal token resolved to a different Audrey account")
        if token_me.get("auth_provider") != "audrey":
            raise SmokeError("personal token did not resolve through Audrey authentication")

        _, files = _json_request("/v1/files", token=personal_token)
        file_items = files.get("files")
        if not isinstance(file_items, list):
            raise SmokeError("compatibility file listing returned an invalid shape")

        _, revoked = _json_request(
            f"/api/tokens/{token_id}",
            token=USER_TOKEN,
            method="DELETE",
        )
        if revoked != {"id": token_id, "revoked": True}:
            raise SmokeError(f"token revocation response was invalid: {revoked}")
        token_id = ""

        revoked_status, revoked_body = _json_request(
            "/api/me",
            token=personal_token,
            expected=frozenset({401}),
        )
        if revoked_body.get("detail") != _REVOKED_DETAIL:
            raise SmokeError(f"revoked Audrey token was not rejected: {revoked_body}")

        result["identity"] = {
            "user_id": access_me["id"],
            "access_provider": access_me["auth_provider"],
            "personal_token_provider": token_me["auth_provider"],
        }
        result["cutover"] = {
            "legacy_bearer_http": legacy_status,
            "legacy_bearer_rejected_locally": True,
            "compatibility_file_count": len(file_items),
            "revoked_token_http": revoked_status,
        }
    except Exception as exc:  # noqa: BLE001 - retain error across cleanup
        primary_error = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if token_id:
            try:
                _, revoked = _json_request(
                    f"/api/tokens/{token_id}",
                    token=USER_TOKEN,
                    method="DELETE",
                    expected=frozenset({200, 404}),
                )
                result["cleanup"] = {
                    "token_id": token_id,
                    "revoked": bool(revoked.get("revoked")),
                }
            except Exception as exc:  # noqa: BLE001 - report cleanup separately
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if primary_error is None:
                    primary_error = exc

    result["status"] = "passed" if primary_error is None else "failed"
    print(json.dumps(result, indent=2))
    return 0 if primary_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
