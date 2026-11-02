"""Exercise the deployed smoke against real ASGI routes and cleanup failures."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from audrey.app_state import ApplicationStore
from audrey.auth import require_account_principal, require_principal
from audrey.routes.app import router

_SMOKE_DIR = Path(__file__).resolve().parent / "smoke"
if str(_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(_SMOKE_DIR))

import smoke_native_project_watches as smoke  # noqa: E402
from smoke_native_auth import SmokeCredentials  # noqa: E402


class ObservedTransport(httpx.AsyncBaseTransport):
    def __init__(self, app, *, mode=None):
        self.inner = httpx.ASGITransport(app=app)
        self.mode, self.requests, self.creates = mode, [], 0

    async def handle_async_request(self, request):
        payload = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, dict(request.headers), payload))
        if self.mode == "auth" and request.url.path == "/api/me":
            return httpx.Response(401, json={"detail": "secret provider body"})
        if self.mode == "cleanup" and request.method == "DELETE" and request.url.path.startswith("/api/watches/"):
            return httpx.Response(503, json={"detail": "secret storage body"})
        if self.mode == "interrupt" and request.method == "PATCH" and request.url.path.startswith("/api/projects/"):
            raise asyncio.CancelledError
        if request.method == "POST" and request.url.path == "/api/watches":
            self.creates += 1
            if self.mode == "duplicate" and self.creates == 2:
                payload["request_id"] += "-unexpected-duplicate"
                request = httpx.Request(request.method, request.url, headers=request.headers, json=payload)
        response = await self.inner.handle_async_request(request)
        if self.mode == "metadata" and request.method == "POST" and request.url.path == "/api/watches":
            await response.aread()
            value = response.json()
            value["project"] = None
            return httpx.Response(response.status_code, json=value)
        return response


@pytest.fixture
async def state(tmp_path, monkeypatch):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await store.resolve_external_identity(provider="test", subject="alice", email="alice@example.com",
                                                   display_name="Alice", role="user", auth_method="test")
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}})
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner
    app.dependency_overrides[require_account_principal] = lambda: owner

    def forbidden(*_args, **_kwargs):
        raise AssertionError("The smoke must not enqueue a source/model check.")

    monkeypatch.setattr(store.watches, "_enqueue", forbidden)
    yield SimpleNamespace(store=store, app=app, owner=owner,
                          credentials=SmokeCredentials("user-fake-secret", "admin-unused-secret", True, True))
    store.close()


def counts(store):
    return tuple(store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
                 for table in ("app_projects", "app_watches", "app_watch_runs", "app_watch_operations"))


async def test_real_asgi_smoke_proves_full_flow_without_queue_source_or_model_calls(state):
    transport = ObservedTransport(state.app)
    checks = await smoke.execute("http://test", state.credentials, transport=transport)
    assert checks == ["paused_project_definition_and_idempotency", "project_filter_and_canonical_rename",
                      "explicit_unlink_and_reassignment", "deleted_project_blocks_run_and_resume",
                      "deleted_project_unlink_recovery", "disposable_cleanup"]
    assert counts(state.store) == (0, 0, 0, 0)
    for method, path, headers, payload in transport.requests:
        assert headers["cf-access-jwt-assertion"] == state.credentials.user
        assert "authorization" not in headers
        assert path.startswith("/api/")
        if method in {"POST", "PATCH"} and path.startswith("/api/watches") and not path.endswith("/actions"):
            assert payload.get("enabled") is False
    actions = [payload["action"] for method, path, _, payload in transport.requests if path.endswith("/actions")]
    assert actions == ["run", "resume"]


@pytest.mark.parametrize("mode", ["metadata", "duplicate"])
async def test_contract_failure_cleans_all_created_resources_including_unexpected_duplicate(state, mode):
    transport = ObservedTransport(state.app, mode=mode)
    with pytest.raises(smoke.SmokeFindingError):
        await smoke.execute("http://test", state.credentials, transport=transport)
    assert counts(state.store) == (0, 0, 0, 0)
    if mode == "duplicate":
        assert sum(method == "DELETE" and path.startswith("/api/watches/")
                   for method, path, *_ in transport.requests) == 2


async def test_cleanup_failure_is_reported_and_other_disposable_resources_still_removed(state):
    transport = ObservedTransport(state.app, mode="cleanup")
    with pytest.raises(smoke.SmokeFindingError, match="could not all be cleaned"):
        await smoke.execute("http://test", state.credentials, transport=transport)
    assert counts(state.store) == (0, 1, 0, 0)
    assert (await state.store.watches.list(user_id=state.owner.user_id))[0]["enabled"] is False


async def test_interruption_still_cleans_every_created_disposable_resource(state):
    with pytest.raises(asyncio.CancelledError):
        await smoke.execute("http://test", state.credentials, transport=ObservedTransport(state.app, mode="interrupt"))
    assert counts(state.store) == (0, 0, 0, 0)


async def test_authentication_setup_failure_makes_no_write_requests(state):
    transport = ObservedTransport(state.app, mode="auth")
    with pytest.raises(smoke.SmokeSetupError):
        await smoke.execute("http://test", state.credentials, transport=transport)
    assert [(method, path) for method, path, *_ in transport.requests] == [("GET", "/api/me")]
    assert counts(state.store) == (0, 0, 0, 0)


async def test_status_failure_retains_safe_request_kind_without_resource_ids_or_response_body():
    async def respond(_request):
        return httpx.Response(503, json={"detail": "secret provider URL and token"})

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(smoke.SmokeFindingError) as raised:
            await smoke.request(client, "GET", "/api/watches/secret-resource-id")
    assert raised.value.public_message == "GET Watch: expected HTTP 200, received HTTP 503."


async def test_existing_owner_resources_survive_the_disposable_smoke(state):
    existing = await state.store.projects.create(user_id=state.owner.user_id, name="Keep this Project")
    watch = await state.store.watches.create(user_id=state.owner.user_id, title="Keep this watch", url="https://example.com/keep",
        criterion="Existing scope", enabled=False, request_id="keep", project_id=existing.project_id,
        schedule={"kind": "interval", "timezone": "UTC", "interval_minutes": 60})
    await smoke.execute("http://test", state.credentials, transport=ObservedTransport(state.app))
    assert counts(state.store) == (1, 1, 0, 0)
    assert (await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])) == watch


@pytest.mark.parametrize("url", ["file:///tmp/private", "http://alice:secret@example.com", "https://example.com/?token=secret", "invalid"])
def test_invalid_local_setup_fails_before_any_client_and_never_prints_credentials(monkeypatch, capsys, url):
    monkeypatch.setenv("AUDREY_USER_JWT", "fake-user-secret")
    monkeypatch.delenv("AUDREY_ADMIN_JWT", raising=False)
    monkeypatch.setenv("AUDREY_SMOKE_BASE_URL", url)
    assert smoke.main(["--json"]) == 2
    output = capsys.readouterr()
    assert "fake-user-secret" not in output.out + output.err and "alice:secret" not in output.out + output.err
    assert json.loads(output.out)["checks"] == []


def test_user_assertion_is_required_but_administrator_is_optional(monkeypatch, capsys):
    monkeypatch.delenv("AUDREY_USER_JWT", raising=False)
    monkeypatch.delenv("AUDREY_ADMIN_JWT", raising=False)
    assert smoke.main([]) == 2
    monkeypatch.setenv("AUDREY_USER_JWT", "fake-user-secret")
    monkeypatch.setenv("AUDREY_SMOKE_BASE_URL", "http://test")
    origin, credentials = smoke.settings()
    assert origin == "http://test" and credentials.user_access and not credentials.admin_access
    assert "fake-user-secret" not in capsys.readouterr().out


@pytest.mark.parametrize("error, expected", [
    (smoke.SmokeFindingError("secret finding"), 1), (smoke.SmokeSetupError("secret setup"), 2),
    (RuntimeError("secret crash"), 3), (KeyboardInterrupt("secret interruption"), 130),
])
def test_main_sanitizes_failure_exit_codes(monkeypatch, capsys, error, expected):
    monkeypatch.setattr(smoke, "settings", lambda: ("http://test", SmokeCredentials("secret-token", "", True, False)))

    def fail(coroutine):
        coroutine.close()
        raise error

    monkeypatch.setattr(asyncio, "run", fail)
    assert smoke.main(["--json"]) == expected
    output = capsys.readouterr()
    receipt = json.loads(output.out)
    assert receipt["exit_code"] == expected and "secret" not in output.out + output.err


def test_main_reports_sanitized_success_checks(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "settings", lambda: ("http://test", SmokeCredentials("secret-token", "", True, False)))

    def done(coroutine):
        coroutine.close()
        return ["disposable_cleanup"]

    monkeypatch.setattr(asyncio, "run", done)
    assert smoke.main([]) == 0
    output = capsys.readouterr()
    assert "PASS disposable_cleanup" in output.out and "exit 0" in output.out
    assert "secret-token" not in output.out + output.err
