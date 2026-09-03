"""Contract tests for strict Cloudflare Access application-token verification."""

from __future__ import annotations

import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from audrey.config import EnvOverrides
from audrey.identity import (
    CloudflareAccessTokenError,
    CloudflareAccessUnavailableError,
    CloudflareAccessVerifier,
    build_cloudflare_access_verifier,
)
from audrey.identity import cloudflare_access as cloudflare_module

_TEAM_DOMAIN = "https://audrey-test.cloudflareaccess.com"
_AUDIENCE = "audrey-application-audience"


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _patch_jwks(monkeypatch, *results):
    pending = iter(results)
    calls: list[str] = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def get(self, url):
            calls.append(url)
            result = next(pending)
            if isinstance(result, Exception):
                raise result
            return _Response(result)

    monkeypatch.setattr(cloudflare_module.httpx, "AsyncClient", lambda **kwargs: _Client())
    return calls


def _key(kid: str):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    public_jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return private, public_jwk


def _token(private, kid: str, **overrides):
    now = int(time.time())
    payload = {
        "aud": [_AUDIENCE],
        "email": "alice@example.com",
        "exp": now + 300,
        "iat": now - 1,
        "iss": _TEAM_DOMAIN,
        "nbf": now - 1,
        "sub": "cloudflare-subject-1",
        "type": "app",
    }
    payload.update(overrides)
    return jwt.encode(payload, private, algorithm="RS256", headers={"kid": kid})


async def test_valid_application_token_is_verified_and_jwks_is_cached(monkeypatch):
    private, jwk = _key("key-1")
    calls = _patch_jwks(monkeypatch, {"keys": [jwk]})
    verifier = CloudflareAccessVerifier(team_domain=_TEAM_DOMAIN, audience=_AUDIENCE)
    token = _token(private, "key-1")

    first = await verifier.verify(token)
    second = await verifier.verify(token)

    assert first == second
    assert first.subject == "cloudflare-subject-1"
    assert first.email == "alice@example.com"
    assert calls == [f"{_TEAM_DOMAIN}/cdn-cgi/access/certs"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"aud": ["wrong"]}, "validation failed"),
        ({"iss": "https://other.cloudflareaccess.com"}, "validation failed"),
        ({"exp": int(time.time()) - 60}, "validation failed"),
        ({"type": "service"}, "not an application token"),
        ({"sub": ""}, "no user subject"),
        ({"email": ""}, "no user email"),
    ],
)
async def test_invalid_identity_or_security_claims_are_rejected(monkeypatch, overrides, expected):
    private, jwk = _key("key-1")
    _patch_jwks(monkeypatch, {"keys": [jwk]})
    verifier = CloudflareAccessVerifier(team_domain=_TEAM_DOMAIN, audience=_AUDIENCE)

    with pytest.raises(CloudflareAccessTokenError, match=expected):
        await verifier.verify(_token(private, "key-1", **overrides))


async def test_service_token_without_user_claims_is_rejected(monkeypatch):
    private, jwk = _key("key-1")
    _patch_jwks(monkeypatch, {"keys": [jwk]})
    verifier = CloudflareAccessVerifier(team_domain=_TEAM_DOMAIN, audience=_AUDIENCE)
    now = int(time.time())
    token = jwt.encode(
        {
            "aud": [_AUDIENCE],
            "exp": now + 300,
            "iat": now - 1,
            "iss": _TEAM_DOMAIN,
            "nbf": now - 1,
            "type": "app",
        },
        private,
        algorithm="RS256",
        headers={"kid": "key-1"},
    )

    with pytest.raises(CloudflareAccessTokenError, match="validation failed"):
        await verifier.verify(token)


async def test_unknown_key_refreshes_once_and_accepts_rotated_key(monkeypatch):
    private_1, jwk_1 = _key("key-1")
    private_2, jwk_2 = _key("key-2")
    calls = _patch_jwks(
        monkeypatch,
        {"keys": [jwk_1]},
        {"keys": [jwk_1, jwk_2]},
    )
    verifier = CloudflareAccessVerifier(team_domain=_TEAM_DOMAIN, audience=_AUDIENCE)

    await verifier.verify(_token(private_1, "key-1"))
    claims = await verifier.verify(_token(private_2, "key-2", sub="rotated-subject"))

    assert claims.subject == "rotated-subject"
    assert len(calls) == 2


async def test_repeated_unknown_keys_do_not_amplify_jwks_requests(monkeypatch):
    _, jwk_1 = _key("key-1")
    private_2, _ = _key("key-2")
    calls = _patch_jwks(monkeypatch, {"keys": [jwk_1]}, {"keys": [jwk_1]})
    verifier = CloudflareAccessVerifier(team_domain=_TEAM_DOMAIN, audience=_AUDIENCE)
    token = _token(private_2, "key-2")

    with pytest.raises(CloudflareAccessTokenError, match="unknown"):
        await verifier.verify(token)
    with pytest.raises(CloudflareAccessTokenError, match="unknown"):
        await verifier.verify(token)

    assert len(calls) == 2


async def test_jwks_network_failure_is_typed_as_provider_unavailable(monkeypatch):
    private, _ = _key("key-1")
    _patch_jwks(monkeypatch, httpx.ConnectError("offline"))
    verifier = CloudflareAccessVerifier(team_domain=_TEAM_DOMAIN, audience=_AUDIENCE)

    with pytest.raises(CloudflareAccessUnavailableError, match="unavailable"):
        await verifier.verify(_token(private, "key-1"))


@pytest.mark.parametrize(
    "team_domain",
    [
        "",
        "http://team.cloudflareaccess.com",
        "https://cloudflareaccess.com",
        "https://team.cloudflareaccess.com.evil.example",
        "https://team.cloudflareaccess.com/path",
        "https://user@team.cloudflareaccess.com",
    ],
)
def test_team_domain_configuration_fails_closed(team_domain):
    with pytest.raises(ValueError):
        CloudflareAccessVerifier(team_domain=team_domain, audience=_AUDIENCE)


def test_adapter_is_off_by_default_and_required_settings_fail_when_enabled():
    assert build_cloudflare_access_verifier(enabled=False, team_domain="", audience="") is None
    with pytest.raises(ValueError, match="team domain"):
        build_cloudflare_access_verifier(enabled=True, team_domain="", audience="")
    with pytest.raises(ValueError, match="audience"):
        build_cloudflare_access_verifier(
            enabled=True,
            team_domain=_TEAM_DOMAIN,
            audience="",
        )


def test_environment_configuration_defaults_to_disabled(monkeypatch):
    for name in (
        "CLOUDFLARE_ACCESS_ENABLED",
        "CLOUDFLARE_ACCESS_TEAM_DOMAIN",
        "CLOUDFLARE_ACCESS_AUDIENCE",
    ):
        monkeypatch.delenv(name, raising=False)

    env = EnvOverrides(_env_file=None)

    assert env.cloudflare_access_enabled is False
    assert env.cloudflare_access_team_domain == ""
    assert env.cloudflare_access_audience == ""
