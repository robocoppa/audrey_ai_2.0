"""Credentials for live native smoke scripts during the OWUI cutover.

Cloudflare Access assertions take precedence. Legacy OWUI bearer tokens remain
an explicit fallback while the old client is still deployed. Never print the
tokens: the application JWT is an account credential, not a service token.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class SmokeCredentials:
    user: str
    admin: str
    user_access: bool
    admin_access: bool

    @classmethod
    def from_env(cls) -> SmokeCredentials:
        user_access_jwt = os.getenv("AUDREY_SMOKE_USER_ACCESS_JWT", "").strip()
        admin_access_jwt = os.getenv("AUDREY_SMOKE_ADMIN_ACCESS_JWT", "").strip()
        credentials = cls(
            user=user_access_jwt or os.getenv("TEST_OWUI_TOKEN", "").strip(),
            admin=admin_access_jwt or os.getenv("ADMIN_OWUI_TOKEN", "").strip(),
            user_access=bool(user_access_jwt),
            admin_access=bool(admin_access_jwt),
        )
        if credentials.user and credentials.user == credentials.admin:
            raise ValueError("user and administrator smoke credentials must differ")
        return credentials

    def headers_for(
        self,
        token: str,
        *,
        user_token: str,
        admin_token: str,
    ) -> dict[str, str]:
        """Use an Access assertion only for its selected account evidence."""
        if token == user_token and self.user_access:
            return {"Cf-Access-Jwt-Assertion": token}
        if token == admin_token and self.admin_access:
            return {"Cf-Access-Jwt-Assertion": token}
        return {"Authorization": f"Bearer {token}"}


MISSING_CREDENTIALS = (
    "Set distinct user/admin credentials with AUDREY_SMOKE_USER_ACCESS_JWT "
    "and AUDREY_SMOKE_ADMIN_ACCESS_JWT (or legacy TEST_OWUI_TOKEN and "
    "ADMIN_OWUI_TOKEN)."
)


__all__ = ["MISSING_CREDENTIALS", "SmokeCredentials"]
