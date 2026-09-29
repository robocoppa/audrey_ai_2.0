#!/usr/bin/env python3
"""Verify the deployed built-in skill catalog and registry health.

This targeted smoke performs read-only catalog/readiness requests plus one
admin rediscovery. It creates no user data, conversations, tokens, files, or
model calls.
"""

from __future__ import annotations

import json
import os
import sys
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__:
    from .smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials
else:
    from smoke_native_auth import MISSING_CREDENTIALS, SmokeCredentials

BASE_URL = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin


class SmokeError(RuntimeError):
    """The deployed built-in skills violated their registry contract."""


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
    method: str = "GET",
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 60,
) -> tuple[int, bytes, Message]:
    headers = {"Accept": "application/json"}
    if token:
        headers.update(_auth_headers(token))
    request = Request(  # noqa: S310 - base URL is operator-controlled
        f"{BASE_URL}{path}",
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
    expected: frozenset[int] = frozenset({200}),
) -> tuple[int, dict[str, Any]]:
    status, content, _headers = _request(
        path,
        token=token,
        method=method,
        expected=expected,
    )
    try:
        payload = json.loads(content) if content else {}
    except json.JSONDecodeError as exc:
        raise SmokeError(f"{method} {path}: response was not JSON") from exc
    if not isinstance(payload, dict):
        raise SmokeError(f"{method} {path}: response was not an object")
    return status, payload


def _expect_ready_skill_registry(
    payload: dict[str, Any],
    *,
    source: str,
) -> None:
    expected = {
        "enabled": True,
        "status": "ready",
        "loaded_count": 2,
        "available_count": 2,
        "degraded_count": 0,
        "invalid_count": 0,
    }
    mismatches = {
        key: payload.get(key) for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise SmokeError(f"{source}: skill registry mismatch: {mismatches}")


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(MISSING_CREDENTIALS, file=sys.stderr)
        return 2

    try:
        _, catalog = _json_request("/api/skills", token=USER_TOKEN)
        expected_items = [
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
        if catalog != {
            "enabled": True,
            "status": "ready",
            "items": expected_items,
        }:
            raise SmokeError(f"skill catalog mismatch: {catalog}")

        _, capabilities = _json_request("/api/capabilities", token=USER_TOKEN)
        if (capabilities.get("skills") or {}).get("status") != "available":
            raise SmokeError("public capabilities did not report skills available")

        readiness_http, readiness = _json_request(
            "/v1/admin/readiness",
            token=ADMIN_TOKEN,
            expected=frozenset({200, 503}),
        )
        _expect_ready_skill_registry(
            readiness.get("skills") or {},
            source="readiness",
        )
        skill_component = (readiness.get("components") or {}).get("skills") or {}
        if skill_component.get("status") != "available":
            raise SmokeError("admin readiness did not report skills available")

        _, rediscovery = _json_request(
            "/v1/admin/skills/rediscover",
            token=ADMIN_TOKEN,
            method="POST",
        )
        _expect_ready_skill_registry(rediscovery, source="rediscovery")
        if rediscovery.get("invalid") != []:
            raise SmokeError("skill rediscovery returned invalid bundle diagnostics")
    except (OSError, SmokeError) as exc:
        print(f"skills foundation smoke failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema": 1,
                "catalog": {
                    "enabled": True,
                    "status": "ready",
                    "items": ["grounded-document-analysis", "video-analysis"],
                },
                "capabilities": {"skills": "available"},
                "readiness": {"http": readiness_http, "skills": "ready"},
                "rediscovery": {"status": "ready", "invalid_count": 0},
                "status": "passed",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
