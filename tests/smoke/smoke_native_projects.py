#!/usr/bin/env python3
"""Exercise owner-scoped Projects and verify them across an Audrey restart.

Run ``capture`` from the laptop before restarting Audrey, then run ``verify``
from the same checkout afterward. The smoke references one existing Ready file
but never changes or deletes that file. Temporary projects and conversations
are removed during verification or after a failed capture.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
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
        "AUDREY_PROJECTS_SNAPSHOT_PATH",
        "testing-out/smokes/c3-projects-restart.json",
    )
)
READY_TIMEOUT_SECONDS = float(os.getenv("AUDREY_PROJECTS_SMOKE_TIMEOUT_SECONDS", "180"))
_CREDENTIALS = SmokeCredentials.from_env()
USER_TOKEN = _CREDENTIALS.user
ADMIN_TOKEN = _CREDENTIALS.admin


class SmokeError(RuntimeError):
    """A deployed Projects contract was not satisfied."""


def _auth_headers(token: str) -> dict[str, str]:
    return _CREDENTIALS.headers_for(
        token,
        user_token=USER_TOKEN,
        admin_token=ADMIN_TOKEN,
    )


def _request(
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected: frozenset[int] = frozenset({200}),
    timeout: float = 60,
) -> tuple[int, bytes, Message]:
    headers = {"Accept": "application/json", **_auth_headers(token)}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    request = Request(  # noqa: S310 - the operator controls the smoke base URL
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
    if not content:
        return status, {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SmokeError(f"{method} {path} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"{method} {path} returned a non-object JSON document")
    return status, value


def _wait_until_ready() -> dict[str, str]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error = ""
    while time.monotonic() < deadline:
        try:
            _, health = _json_request("/health", token=USER_TOKEN)
            _, capabilities = _json_request("/api/capabilities", token=USER_TOKEN)
            if health.get("status") == "ok" and capabilities.get("status") == "ready":
                return {"health": "ok", "capabilities": "ready"}
            last_error = (
                f"health={health.get('status')!r}, capabilities={capabilities.get('status')!r}"
            )
        except Exception as exc:  # noqa: BLE001 - startup failures are retried
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise SmokeError(f"Audrey did not become ready within {READY_TIMEOUT_SECONDS:g}s: {last_error}")


def _write_snapshot(payload: dict[str, Any]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SNAPSHOT_PATH.with_suffix(SNAPSHOT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(SNAPSHOT_PATH)


def _load_snapshot() -> dict[str, Any]:
    try:
        value = json.loads(SNAPSHOT_PATH.read_text())
    except FileNotFoundError as exc:
        raise SmokeError(f"snapshot does not exist: {SNAPSHOT_PATH}; run capture first") from exc
    except json.JSONDecodeError as exc:
        raise SmokeError(f"snapshot is not valid JSON: {SNAPSHOT_PATH}") from exc
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise SmokeError("snapshot schema is unsupported")
    return value


def _delete(path: str, *, token: str) -> int:
    status, _content, _headers = _request(
        path,
        token=token,
        method="DELETE",
        expected=frozenset({204, 404}),
    )
    return status


def _cleanup_ids(ids: dict[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    conversation_id = ids.get("conversation_id", "")
    first_project_id = ids.get("first_project_id", "")
    second_project_id = ids.get("second_project_id", "")
    foreign_project_id = ids.get("foreign_project_id", "")
    if conversation_id:
        result["conversation_http"] = _delete(
            f"/api/conversations/{quote(conversation_id, safe='')}",
            token=USER_TOKEN,
        )
    if first_project_id:
        result["first_project_http"] = _delete(
            f"/api/projects/{quote(first_project_id, safe='')}",
            token=USER_TOKEN,
        )
    if second_project_id:
        result["second_project_http"] = _delete(
            f"/api/projects/{quote(second_project_id, safe='')}",
            token=USER_TOKEN,
        )
    if foreign_project_id:
        result["foreign_project_http"] = _delete(
            f"/api/projects/{quote(foreign_project_id, safe='')}",
            token=ADMIN_TOKEN,
        )
    return result


def _first_ready_file() -> dict[str, Any]:
    _, listing = _json_request("/api/files", token=USER_TOKEN)
    items = listing.get("items")
    if not isinstance(items, list):
        raise SmokeError("native Files listing returned an invalid shape")
    ready = next(
        (
            item
            for item in items
            if isinstance(item, dict) and item.get("status") == "ready" and item.get("id")
        ),
        None,
    )
    if not isinstance(ready, dict):
        raise SmokeError(
            "the smoke user has no Ready file; upload one small document in "
            "My Files, wait for Ready, and rerun capture"
        )
    return ready


def capture() -> dict[str, Any]:
    if SNAPSHOT_PATH.exists():
        raise SmokeError(f"snapshot already exists: {SNAPSHOT_PATH}; run cleanup before capture")
    readiness = _wait_until_ready()
    suffix = uuid.uuid4().hex[:8]
    ids: dict[str, str] = {}
    try:
        _, user_me = _json_request("/api/me", token=USER_TOKEN)
        _, admin_me = _json_request("/api/me", token=ADMIN_TOKEN)
        if not user_me.get("id") or user_me.get("id") == admin_me.get("id"):
            raise SmokeError("smoke credentials must resolve to two distinct accounts")

        _, first = _json_request(
            "/api/projects",
            token=USER_TOKEN,
            method="POST",
            payload={
                "name": f"Restart proof {suffix}",
                "instructions": "Answer with concise project context.",
            },
            expected=frozenset({201}),
        )
        ids["first_project_id"] = str(first.get("id") or "")
        _, second = _json_request(
            "/api/projects",
            token=USER_TOKEN,
            method="POST",
            payload={"name": f"Move target {suffix}"},
            expected=frozenset({201}),
        )
        ids["second_project_id"] = str(second.get("id") or "")
        _, foreign = _json_request(
            "/api/projects",
            token=ADMIN_TOKEN,
            method="POST",
            payload={"name": f"Foreign project {suffix}"},
            expected=frozenset({201}),
        )
        ids["foreign_project_id"] = str(foreign.get("id") or "")
        if not all(ids.values()):
            raise SmokeError("project creation omitted a project id")

        first_id = quote(ids["first_project_id"], safe="")
        second_id = ids["second_project_id"]
        foreign_id = ids["foreign_project_id"]
        cross_owner_http, _ = _json_request(
            f"/api/projects/{first_id}",
            token=ADMIN_TOKEN,
            expected=frozenset({404}),
        )

        _, project_page = _json_request("/api/projects?limit=1", token=USER_TOKEN)
        if project_page.get("limits") != {
            "max_name_chars": 100,
            "max_instructions_chars": 4000,
            "max_files": 20,
        }:
            raise SmokeError(f"project limits changed: {project_page.get('limits')!r}")
        if not project_page.get("next_cursor"):
            raise SmokeError("two projects did not produce a one-item pagination cursor")

        _, conversation = _json_request(
            f"/api/projects/{first_id}/conversations",
            token=USER_TOKEN,
            method="POST",
            payload={"title": f"Project persistence {suffix}"},
            expected=frozenset({201}),
        )
        ids["conversation_id"] = str(conversation.get("id") or "")
        if not ids["conversation_id"]:
            raise SmokeError("project conversation creation omitted its id")
        conversation_path = f"/api/conversations/{quote(ids['conversation_id'], safe='')}"
        for project_id in (second_id, None, ids["first_project_id"]):
            _, moved = _json_request(
                conversation_path,
                token=USER_TOKEN,
                method="PATCH",
                payload={"project_id": project_id},
            )
            if moved.get("project_id") != project_id:
                raise SmokeError("conversation move did not persist the requested project")
        cross_project_http, _ = _json_request(
            conversation_path,
            token=USER_TOKEN,
            method="PATCH",
            payload={"project_id": foreign_id},
            expected=frozenset({404}),
        )

        ready_file = _first_ready_file()
        file_id = str(ready_file["id"])
        _, attached = _json_request(
            f"/api/projects/{first_id}/files",
            token=USER_TOKEN,
            method="POST",
            payload={"file_id": file_id},
            expected=frozenset({201}),
        )
        duplicate_http, _ = _json_request(
            f"/api/projects/{first_id}/files",
            token=USER_TOKEN,
            method="POST",
            payload={"file_id": file_id},
            expected=frozenset({409}),
        )
        _, renamed = _json_request(
            f"/api/projects/{first_id}",
            token=USER_TOKEN,
            method="PATCH",
            payload={
                "name": f"Persisted project {suffix}",
                "instructions": "Keep the persisted project response concise.",
            },
        )
        snapshot = {
            "schema": 1,
            "ids": ids,
            "project": renamed,
            "file": {
                "id": file_id,
                "filename": str(attached.get("filename") or ""),
            },
            "conversation": {
                "id": ids["conversation_id"],
                "project_id": ids["first_project_id"],
                "title": str(conversation.get("title") or ""),
            },
        }
        _write_snapshot(snapshot)
        return {
            "schema": 1,
            "status": "captured",
            "snapshot": str(SNAPSHOT_PATH),
            "readiness": readiness,
            "project": {
                "id": ids["first_project_id"],
                "name": renamed.get("name"),
            },
            "conversation": snapshot["conversation"],
            "file_reference": snapshot["file"],
            "checks": {
                "cross_owner_project_http": cross_owner_http,
                "cross_owner_move_http": cross_project_http,
                "duplicate_file_http": duplicate_http,
                "pagination_cursor": True,
            },
            "next": "restart Audrey, then run verify from this same checkout",
        }
    except Exception:
        _cleanup_ids(ids)
        raise


def verify() -> dict[str, Any]:
    snapshot = _load_snapshot()
    ids = snapshot.get("ids")
    if not isinstance(ids, dict):
        raise SmokeError("snapshot omitted cleanup ids")
    readiness = _wait_until_ready()
    first_id = str(ids.get("first_project_id") or "")
    conversation_id = str(ids.get("conversation_id") or "")
    file_id = str(snapshot.get("file", {}).get("id") or "")
    if not first_id or not conversation_id or not file_id:
        raise SmokeError("snapshot omitted a project, conversation, or file id")

    primary_error: Exception | None = None
    result: dict[str, Any] = {}
    try:
        _, project = _json_request(
            f"/api/projects/{quote(first_id, safe='')}",
            token=USER_TOKEN,
        )
        if project != snapshot.get("project"):
            raise SmokeError("project fields changed across restart")
        _, conversation = _json_request(
            f"/api/conversations/{quote(conversation_id, safe='')}",
            token=USER_TOKEN,
        )
        if conversation.get("project_id") != first_id:
            raise SmokeError("conversation lost its project across restart")
        _, conversations = _json_request(
            f"/api/projects/{quote(first_id, safe='')}/conversations",
            token=USER_TOKEN,
        )
        if conversation_id not in {
            str(item.get("id") or "")
            for item in conversations.get("items", [])
            if isinstance(item, dict)
        }:
            raise SmokeError("project conversation list lost the saved conversation")
        _, files = _json_request(
            f"/api/projects/{quote(first_id, safe='')}/files",
            token=USER_TOKEN,
        )
        if file_id not in {
            str(item.get("id") or "") for item in files.get("items", []) if isinstance(item, dict)
        }:
            raise SmokeError("project file list lost the saved file reference")

        deleted_project_http = _delete(
            f"/api/projects/{quote(first_id, safe='')}",
            token=USER_TOKEN,
        )
        ids["first_project_id"] = ""
        _, retained_conversation = _json_request(
            f"/api/conversations/{quote(conversation_id, safe='')}",
            token=USER_TOKEN,
        )
        if retained_conversation.get("project_id") is not None:
            raise SmokeError("project deletion did not ungroup its conversation")
        retained_file_http, _ = _json_request(
            f"/api/files/{quote(file_id, safe='')}",
            token=USER_TOKEN,
        )
        deleted_read_http, _ = _json_request(
            f"/api/projects/{quote(first_id, safe='')}",
            token=USER_TOKEN,
            expected=frozenset({404}),
        )
        result = {
            "schema": 1,
            "status": "passed",
            "snapshot": str(SNAPSHOT_PATH),
            "readiness": readiness,
            "persistence": {
                "project": True,
                "conversation_membership": True,
                "file_reference": True,
            },
            "deletion": {
                "project_http": deleted_project_http,
                "project_read_http": deleted_read_http,
                "conversation_retained": True,
                "conversation_project_id": retained_conversation.get("project_id"),
                "file_retained_http": retained_file_http,
            },
        }
    except Exception as exc:  # noqa: BLE001 - cleanup follows every failed check
        primary_error = exc
    cleanup_error: Exception | None = None
    cleanup: dict[str, int] = {}
    try:
        cleanup = _cleanup_ids({str(key): str(value) for key, value in ids.items()})
    except Exception as exc:  # noqa: BLE001 - report cleanup beside primary error
        cleanup_error = exc
    if primary_error is None and cleanup_error is None:
        SNAPSHOT_PATH.unlink(missing_ok=True)
        result["cleanup"] = cleanup
        return result
    if primary_error is not None and cleanup_error is not None:
        raise SmokeError(f"{primary_error}; cleanup also failed: {cleanup_error}")
    if primary_error is not None:
        raise primary_error
    raise SmokeError(f"cleanup failed: {cleanup_error}")


def cleanup() -> dict[str, Any]:
    snapshot = _load_snapshot()
    ids = snapshot.get("ids")
    if not isinstance(ids, dict):
        raise SmokeError("snapshot omitted cleanup ids")
    cleaned = _cleanup_ids({str(key): str(value) for key, value in ids.items()})
    SNAPSHOT_PATH.unlink(missing_ok=True)
    return {
        "schema": 1,
        "status": "cleaned",
        "snapshot": str(SNAPSHOT_PATH),
        "cleanup": cleaned,
    }


def main() -> int:
    if not USER_TOKEN or not ADMIN_TOKEN:
        print(MISSING_CREDENTIALS, file=sys.stderr)
        return 2
    if READY_TIMEOUT_SECONDS <= 0:
        print("AUDREY_PROJECTS_SMOKE_TIMEOUT_SECONDS must be positive.", file=sys.stderr)
        return 2
    if len(sys.argv) != 2 or sys.argv[1] not in {"capture", "verify", "cleanup"}:
        print("usage: smoke_native_projects.py capture|verify|cleanup", file=sys.stderr)
        return 2
    try:
        action = sys.argv[1]
        if action == "capture":
            result = capture()
        elif action == "verify":
            result = verify()
        else:
            result = cleanup()
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
