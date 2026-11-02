#!/usr/bin/env python3
"""Check deployed Project Watch APIs using disposable, paused fixtures only.

Requires AUDREY_USER_JWT and AUDREY_SMOKE_BASE_URL. No successful Run/Resume,
source fetch, upload or model generation is requested. All disposable resources
are removed in finally, including after a failed contract check.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from urllib.parse import quote, urlsplit

import httpx

if __package__:
    from .smoke_native_auth import SmokeCredentials
else:
    from smoke_native_auth import SmokeCredentials


class SmokeSetupError(Exception):
    """A prerequisite failed before creating disposable fixtures."""

    def __init__(self, message, *, safe=False):
        super().__init__(message)
        self.public_message = message if safe else "Project Watches smoke setup failed."


class SmokeFindingError(Exception):
    """A native Project Watch API contract failed."""

    def __init__(self, message, *, safe=False):
        super().__init__(message)
        self.public_message = message if safe else "Project Watches API smoke found a contract or cleanup failure."


def require(condition, explanation):
    if not condition:
        raise SmokeFindingError(explanation, safe=True)


def resource_id(payload):
    value = payload.get("id")
    require(isinstance(value, str) and 0 < len(value) <= 300, "Creation omitted a valid resource identity.")
    return value


def revision(watch):
    value = watch.get("revision")
    require(type(value) is int and value > 0, "Watch omitted a valid revision.")
    return value


def paused(watch):
    require(watch.get("enabled") is False and watch.get("next_run_at") is None
            and watch.get("active_run_id") is None and watch.get("latest_run") is None,
            "A paused disposable watch unexpectedly scheduled or ran a check.")


async def request(client, method, path, *, payload=None, expected=200):
    try:
        response = await client.request(method, path, json=payload)
    except httpx.HTTPError as error:
        raise SmokeFindingError("The native API request failed.", safe=True) from error
    kind = "Project" if path.startswith("/api/projects") else "Watch"
    if path == "/api/me":
        kind = "Account"
    elif path.endswith("/actions"):
        kind = "Watch action"
    elif path.endswith("/runs"):
        kind = "Watch history"
    elif "?project_id=" in path:
        kind = "Project Watch list"
    verb = method if method in {"GET", "POST", "PATCH", "DELETE"} else "Request"
    require(response.status_code == expected,
            f"{verb} {kind}: expected HTTP {expected}, received HTTP {response.status_code}.")
    if expected in {204, 404, 409}:
        return {}
    try:
        value = response.json()
    except ValueError as error:
        raise SmokeFindingError("The native API returned invalid JSON.", safe=True) from error
    require(isinstance(value, dict), "The native API returned an invalid response shape.")
    return value


async def cleanup(client, watch_ids, project_ids):
    clean = True
    for watch_id in reversed(watch_ids):
        path = "/api/watches/" + quote(watch_id, safe="")
        removed = False
        for _ in range(3):
            try:
                current = await client.get(path)
                if current.status_code == 404:
                    removed = True
                    break
                require(current.status_code == 200, "Disposable watch cleanup failed.")
                watch = current.json()
                response = await client.delete(path, params={"expected_revision": revision(watch)})
                if response.status_code in {200, 404}:
                    removed = True
                    break
                require(response.status_code == 409, "Disposable watch cleanup failed.")
            except (httpx.HTTPError, SmokeFindingError, ValueError, TypeError, AttributeError):
                break
        clean = clean and removed
    for project_id in reversed(project_ids):
        try:
            response = await client.delete("/api/projects/" + quote(project_id, safe=""))
            clean = response.status_code in {204, 404} and clean
        except httpx.HTTPError:
            clean = False
    return clean


async def run_smoke(client):
    """Inject an AsyncClient/ASGI transport to verify the exact deployed flow."""
    watch_ids, project_ids, checks = [], [], []
    suffix = uuid.uuid4().hex[:12]
    try:
        try:
            me = await request(client, "GET", "/api/me")
            require(bool(me.get("id")), "Native user identity is unavailable.")
            listing = await request(client, "GET", "/api/watches")
            require(isinstance(listing.get("items"), list) and len(listing["items"]) < 20,
                    "The smoke account needs one free Watch slot.")
        except SmokeFindingError as error:
            raise SmokeSetupError("Native preflight failed: " + error.public_message, safe=True) from error
        await request(client, "GET", "/api/watches?project_id=" + suffix, expected=404)

        projects = []
        for label in ("first", "second"):
            project = await request(client, "POST", "/api/projects", payload={
                "name": f"Audrey Watch smoke {label} {suffix}", "model_id": "auto",
            }, expected=201)
            project_id = resource_id(project)
            require(project_id not in project_ids, "Disposable Project creation reused an identity.")
            project_ids.append(project_id)
            require(project.get("private_required") is False, "An ordinary disposable Project was not created.")
            projects.append(project)
        first, second = projects
        body = {
            "title": "Audrey Project Watch smoke", "url": "https://example.invalid/audrey-watch-smoke",
            "criterion": "Report public release changes.", "request_id": "project-watch-smoke-" + suffix,
            "schedule": {"kind": "interval", "timezone": "UTC", "interval_minutes": 60},
            "enabled": False, "project_id": first["id"],
        }
        watch = await request(client, "POST", "/api/watches", payload=body, expected=201)
        watch_ids.append(resource_id(watch))
        paused(watch)
        expected_project = {"id": first["id"], "name": first["name"], "available": True}
        require(watch.get("project") == expected_project, "Watch creation lost its canonical Project link.")
        duplicate = await request(client, "POST", "/api/watches", payload=body, expected=201)
        duplicate_id = resource_id(duplicate)
        if duplicate_id not in watch_ids:
            watch_ids.append(duplicate_id)
        require(duplicate_id == watch["id"], "Watch request replay created another watch.")
        paused(duplicate)
        changed = {**body, "project_id": second["id"]}
        await request(client, "POST", "/api/watches", payload=changed, expected=409)
        checks.append("paused_project_definition_and_idempotency")

        path = "/api/watches/" + quote(watch["id"], safe="")
        first_path = "/api/projects/" + quote(first["id"], safe="")
        scoped = await request(client, "GET", "/api/watches?project_id=" + quote(first["id"], safe=""))
        require([item.get("id") for item in scoped.get("items", [])] == [watch["id"]], "Project Watch filtering failed.")
        renamed = await request(client, "PATCH", first_path, payload={"name": "Renamed Watch smoke " + suffix})
        watch = await request(client, "GET", path)
        require(watch.get("project") == {**expected_project, "name": renamed.get("name")},
                "Watch metadata did not follow the canonical Project rename.")
        paused(watch)
        checks.append("project_filter_and_canonical_rename")

        watch = await request(client, "PATCH", path, payload={
            "expected_revision": revision(watch), "project_id": None, "enabled": False,
        })
        require(watch.get("project") is None, "Explicit Project unlink failed.")
        paused(watch)
        scoped = await request(client, "GET", "/api/watches?project_id=" + quote(first["id"], safe=""))
        require(scoped.get("items") == [], "An unlinked watch remained in its old Project.")
        watch = await request(client, "PATCH", path, payload={
            "expected_revision": revision(watch), "project_id": second["id"], "enabled": False,
        })
        require(watch.get("project") == {"id": second["id"], "name": second["name"], "available": True},
                "Watch Project reassignment failed.")
        paused(watch)
        scoped = await request(client, "GET", "/api/watches?project_id=" + quote(second["id"], safe=""))
        require([item.get("id") for item in scoped.get("items", [])] == [watch["id"]],
                "A reassigned watch was absent from its new Project.")
        checks.append("explicit_unlink_and_reassignment")

        await request(client, "DELETE", "/api/projects/" + quote(second["id"], safe=""), expected=204)
        watch = await request(client, "GET", path)
        require(watch.get("project") == {"id": second["id"], "name": None, "available": False},
                "A deleted Project remained available through its Watch.")
        paused(watch)
        for action in ("run", "resume"):
            await request(client, "POST", path + "/actions", payload={
                "expected_revision": revision(watch), "action": action,
            }, expected=404)
        watch = await request(client, "GET", path)
        paused(watch)
        await request(client, "GET", "/api/watches?project_id=" + quote(second["id"], safe=""), expected=404)
        history = await request(client, "GET", path + "/runs")
        require(history.get("items") == [], "The smoke unexpectedly created a Watch check.")
        checks.append("deleted_project_blocks_run_and_resume")
        watch = await request(client, "PATCH", path, payload={
            "expected_revision": revision(watch), "project_id": None, "enabled": False,
        })
        require(watch.get("project") is None, "A deleted Project could not be explicitly unlinked.")
        paused(watch)
        history = await request(client, "GET", path + "/runs")
        require(history.get("items") == [], "Deleted-Project unlink unexpectedly created a Watch check.")
        checks.append("deleted_project_unlink_recovery")
    finally:
        cleaned = await cleanup(client, watch_ids, project_ids)
        if not cleaned:
            raise SmokeFindingError("Disposable smoke resources could not all be cleaned up.", safe=True)
    checks.append("disposable_cleanup")
    return checks


async def execute(base_url, credentials, *, transport=None):
    headers = {"Accept": "application/json", **credentials.headers_for(
        credentials.user, user_token=credentials.user, admin_token=credentials.admin,
    )}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, transport=transport,
                                 timeout=10.0, follow_redirects=False) as client:
        return await run_smoke(client)


def settings():
    try:
        credentials = SmokeCredentials.from_env()
        base_url = os.getenv("AUDREY_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        parsed = urlsplit(base_url)
        if (not credentials.user_access or parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path):
            raise ValueError
    except ValueError as error:
        raise SmokeSetupError("Native user credentials or origin configuration is invalid.", safe=True) from error
    return base_url, credentials


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv not in ([], ["--json"]):
        print("SETUP: usage: smoke_native_project_watches.py [--json]", file=sys.stderr)
        return 2
    checks, code, message = [], 0, "Project Watches API smoke passed; disposable paused fixtures cleaned; no source/model calls."
    try:
        base_url, credentials = settings()
        checks = asyncio.run(execute(base_url, credentials))
    except SmokeSetupError as error:
        code, message = 2, error.public_message
    except SmokeFindingError as error:
        code, message = 1, error.public_message
    except (KeyboardInterrupt, asyncio.CancelledError):
        code, message = 130, "Project Watches API smoke interrupted."
    except Exception:  # noqa: BLE001 - never echo provider bodies, URLs or credentials
        code, message = 3, "Project Watches API smoke harness crashed."
    if argv == ["--json"]:
        print(json.dumps({"exit_code": code, "checks": checks, "message": message}, sort_keys=True))
    else:
        for check in checks:
            print("PASS " + check)
        print(("PASS" if code == 0 else "FAIL") + f" {message} (exit {code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
