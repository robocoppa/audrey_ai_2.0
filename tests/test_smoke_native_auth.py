"""Native live smokes can authenticate without OWUI bearer tokens."""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from email.message import Message
from pathlib import Path
from urllib.request import Request

import pytest

from scripts.smoke_native_auth import SmokeCredentials

_SCRIPTS = (
    "smoke_native_ui",
    "smoke_native_files",
    "smoke_native_chat_projection",
    "smoke_native_access_models",
    "smoke_native_preferences",
    "smoke_native_tool_events",
    "smoke_native_modes",
)


def test_access_assertions_take_precedence_and_secrets_stay_out_of_repr(monkeypatch):
    monkeypatch.setenv("AUDREY_SMOKE_USER_ACCESS_JWT", " user-jwt ")
    monkeypatch.setenv("AUDREY_SMOKE_ADMIN_ACCESS_JWT", " admin-jwt ")
    monkeypatch.setenv("TEST_OWUI_TOKEN", "legacy-user")
    monkeypatch.setenv("ADMIN_OWUI_TOKEN", "legacy-admin")
    credentials = SmokeCredentials.from_env()
    assert credentials.user == "user-jwt"
    assert credentials.admin == "admin-jwt"
    assert credentials.headers_for(
        credentials.user, user_token=credentials.user, admin_token=credentials.admin
    ) == {"Cf-Access-Jwt-Assertion": "user-jwt"}
    assert credentials.headers_for(
        credentials.admin, user_token=credentials.user, admin_token=credentials.admin
    ) == {"Cf-Access-Jwt-Assertion": "admin-jwt"}
    assert credentials.headers_for(
        "pat-one-time", user_token=credentials.user, admin_token=credentials.admin
    ) == {"Authorization": "Bearer pat-one-time"}
    assert "user-jwt" not in repr(credentials)
    assert "admin-jwt" not in repr(credentials)


def test_legacy_bearers_remain_a_transition_fallback(monkeypatch):
    monkeypatch.delenv("AUDREY_SMOKE_USER_ACCESS_JWT", raising=False)
    monkeypatch.delenv("AUDREY_SMOKE_ADMIN_ACCESS_JWT", raising=False)
    monkeypatch.setenv("TEST_OWUI_TOKEN", "legacy-user")
    monkeypatch.setenv("ADMIN_OWUI_TOKEN", "legacy-admin")
    credentials = SmokeCredentials.from_env()
    assert credentials.headers_for(
        credentials.user, user_token=credentials.user, admin_token=credentials.admin
    ) == {"Authorization": "Bearer legacy-user"}
    assert credentials.headers_for(
        credentials.admin, user_token=credentials.user, admin_token=credentials.admin
    ) == {"Authorization": "Bearer legacy-admin"}


def test_same_credential_is_refused_before_any_live_write(monkeypatch):
    monkeypatch.setenv("AUDREY_SMOKE_USER_ACCESS_JWT", "same-jwt")
    monkeypatch.setenv("AUDREY_SMOKE_ADMIN_ACCESS_JWT", "same-jwt")
    with pytest.raises(ValueError, match="must differ"):
        SmokeCredentials.from_env()


class _Response:
    status = 200
    headers = Message()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"{}"


@pytest.mark.parametrize("module_name", _SCRIPTS)
def test_every_native_smoke_uses_access_for_rest_and_stream(
    monkeypatch,
    module_name,
):
    smoke = importlib.import_module(f"scripts.{module_name}")
    credentials = SmokeCredentials("user-jwt", "admin-jwt", True, True)
    monkeypatch.setattr(smoke, "_CREDENTIALS", credentials)
    monkeypatch.setattr(smoke, "USER_TOKEN", credentials.user)
    monkeypatch.setattr(smoke, "ADMIN_TOKEN", credentials.admin)
    seen: list[Request] = []

    def fake_urlopen(request: Request, *, timeout: float):
        del timeout
        seen.append(request)
        return _Response()

    monkeypatch.setattr(smoke, "urlopen", fake_urlopen)
    for token in (credentials.user, credentials.admin, "pat-one-time"):
        smoke._request("/api/me", token=token)

    def headers(request: Request) -> dict[str, str]:
        return {key.casefold(): value for key, value in request.header_items()}

    assert headers(seen[0])["cf-access-jwt-assertion"] == "user-jwt"
    assert "authorization" not in headers(seen[0])
    assert headers(seen[1])["cf-access-jwt-assertion"] == "admin-jwt"
    assert "authorization" not in headers(seen[1])
    assert headers(seen[2])["authorization"] == "Bearer pat-one-time"
    assert "cf-access-jwt-assertion" not in headers(seen[2])
    source = inspect.getsource(smoke)
    assert "**_auth_headers(USER_TOKEN)" in source or "**_auth_headers(token)" in source


def test_direct_script_execution_still_resolves_helper(tmp_path):
    del tmp_path
    env = os.environ.copy()
    for key in (
        "AUDREY_SMOKE_USER_ACCESS_JWT",
        "AUDREY_SMOKE_ADMIN_ACCESS_JWT",
        "TEST_OWUI_TOKEN",
        "ADMIN_OWUI_TOKEN",
    ):
        env.pop(key, None)
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "smoke_native_ui.py")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "AUDREY_SMOKE_USER_ACCESS_JWT" in result.stderr
