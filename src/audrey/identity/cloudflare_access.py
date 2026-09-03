"""Strict Cloudflare Access application-token verification.

Cloudflare Access proves an external identity; it does not choose Audrey
roles, user ids, or storage namespaces. Tokens are accepted only after an
RS256 signature check against the team's JWKS plus issuer, audience, time,
token-type, subject, and email validation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt


class CloudflareAccessTokenError(ValueError):
    """The presented token is invalid or outside Audrey's identity policy."""


class CloudflareAccessUnavailableError(RuntimeError):
    """Cloudflare's signing keys cannot currently be obtained or used."""


@dataclass(frozen=True, slots=True)
class CloudflareAccessClaims:
    """The minimal verified identity evidence Audrey consumes."""

    subject: str
    email: str


class CloudflareAccessVerifier:
    """Verify Access JWTs with a bounded, refreshable JWKS cache."""

    def __init__(
        self,
        *,
        team_domain: str,
        audience: str,
        cache_ttl_s: float = 3600.0,
        timeout_s: float = 5.0,
        unknown_kid_refresh_interval_s: float = 30.0,
    ) -> None:
        self.team_domain = _normalize_team_domain(team_domain)
        self.audience = _required(audience, "Cloudflare Access audience")
        if cache_ttl_s <= 0:
            raise ValueError("Cloudflare Access JWKS cache TTL must be positive")
        if timeout_s <= 0:
            raise ValueError("Cloudflare Access timeout must be positive")
        if unknown_kid_refresh_interval_s <= 0:
            raise ValueError("Cloudflare Access unknown-key refresh interval must be positive")
        self.cache_ttl_s = float(cache_ttl_s)
        self.timeout_s = float(timeout_s)
        self.unknown_kid_refresh_interval_s = float(unknown_kid_refresh_interval_s)
        self.jwks_url = f"{self.team_domain}/cdn-cgi/access/certs"
        self._keys: dict[str, jwt.PyJWK] = {}
        self._keys_loaded_at = 0.0
        self._last_unknown_kid_refresh_at: float | None = None
        self._refresh_lock = asyncio.Lock()

    async def verify(self, token: str) -> CloudflareAccessClaims:
        """Return verified identity claims or raise a typed failure."""

        token = token.strip()
        if not token:
            raise CloudflareAccessTokenError("Cloudflare Access token is empty")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise CloudflareAccessTokenError("Cloudflare Access token header is invalid") from exc

        if header.get("alg") != "RS256":
            raise CloudflareAccessTokenError("Cloudflare Access token must use RS256")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid.strip():
            raise CloudflareAccessTokenError("Cloudflare Access token is missing a key id")

        keys = await self._get_keys()
        key = keys.get(kid)
        if key is None:
            # Access rotates signing keys. Refresh immediately on an unknown
            # kid so a still-live cache cannot reject a newly rotated token.
            keys = await self._get_keys(force=True)
            key = keys.get(kid)
        if key is None:
            raise CloudflareAccessTokenError("Cloudflare Access signing key is unknown")

        try:
            payload = await asyncio.to_thread(
                jwt.decode,
                token,
                key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.team_domain,
                leeway=30,
                options={
                    "require": [
                        "aud",
                        "email",
                        "exp",
                        "iat",
                        "iss",
                        "nbf",
                        "sub",
                        "type",
                    ]
                },
            )
        except jwt.PyJWTError as exc:
            raise CloudflareAccessTokenError("Cloudflare Access token validation failed") from exc

        if payload.get("type") != "app":
            raise CloudflareAccessTokenError("Cloudflare Access token is not an application token")
        subject = payload.get("sub")
        email = payload.get("email")
        if not isinstance(subject, str) or not subject.strip():
            raise CloudflareAccessTokenError("Cloudflare Access token has no user subject")
        if not isinstance(email, str) or not email.strip():
            raise CloudflareAccessTokenError("Cloudflare Access token has no user email")
        return CloudflareAccessClaims(subject=subject.strip(), email=email.strip())

    async def _get_keys(self, *, force: bool = False) -> dict[str, jwt.PyJWK]:
        now = time.monotonic()
        if not force and self._keys and now - self._keys_loaded_at < self.cache_ttl_s:
            return self._keys

        async with self._refresh_lock:
            now = time.monotonic()
            if not force and self._keys and now - self._keys_loaded_at < self.cache_ttl_s:
                return self._keys
            if (
                force
                and self._keys
                and self._last_unknown_kid_refresh_at is not None
                and now - self._last_unknown_kid_refresh_at
                < self.unknown_kid_refresh_interval_s
            ):
                return self._keys
            if force:
                # Set before the network request so an outage plus attacker-
                # controlled kid values cannot turn this endpoint into a JWKS
                # request amplifier.
                self._last_unknown_kid_refresh_at = now
            keys = await self._fetch_keys()
            self._keys = keys
            self._keys_loaded_at = time.monotonic()
            return keys

    async def _fetch_keys(self) -> dict[str, jwt.PyJWK]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.get(self.jwks_url)
                response.raise_for_status()
                payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CloudflareAccessUnavailableError(
                "Cloudflare Access signing keys are unavailable"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("keys"), list):
            raise CloudflareAccessUnavailableError("Cloudflare Access JWKS is malformed")

        keys: dict[str, jwt.PyJWK] = {}
        try:
            for item in payload["keys"]:
                if not isinstance(item, dict):
                    continue
                kid = item.get("kid")
                if (
                    isinstance(kid, str)
                    and kid
                    and item.get("kty") == "RSA"
                    and item.get("alg") == "RS256"
                    and item.get("use", "sig") == "sig"
                ):
                    keys[kid] = jwt.PyJWK.from_dict(item, algorithm="RS256")
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            raise CloudflareAccessUnavailableError("Cloudflare Access JWKS is unusable") from exc
        if not keys:
            raise CloudflareAccessUnavailableError(
                "Cloudflare Access JWKS contains no signing keys"
            )
        return keys


def _normalize_team_domain(value: str) -> str:
    value = _required(value, "Cloudflare Access team domain")
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".cloudflareaccess.com")
        or hostname == ".cloudflareaccess.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Cloudflare Access team domain must be an HTTPS *.cloudflareaccess.com origin"
        )
    return f"https://{hostname}"


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    return normalized


def build_cloudflare_access_verifier(
    *,
    enabled: bool,
    team_domain: str,
    audience: str,
) -> CloudflareAccessVerifier | None:
    """Build the optional adapter, validating required settings when enabled."""

    if not enabled:
        return None
    return CloudflareAccessVerifier(team_domain=team_domain, audience=audience)


__all__ = [
    "CloudflareAccessClaims",
    "CloudflareAccessTokenError",
    "CloudflareAccessUnavailableError",
    "CloudflareAccessVerifier",
    "build_cloudflare_access_verifier",
]
