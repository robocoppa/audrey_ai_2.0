"""Provider-neutral authenticated identity models and adapters."""

from audrey.identity.cloudflare_access import (
    CloudflareAccessClaims,
    CloudflareAccessTokenError,
    CloudflareAccessUnavailableError,
    CloudflareAccessVerifier,
    build_cloudflare_access_verifier,
)
from audrey.identity.models import (
    TOKEN_SCOPES,
    IssuedPersonalToken,
    PersonalTokenSummary,
    Principal,
)

__all__ = [
    "CloudflareAccessClaims",
    "CloudflareAccessTokenError",
    "CloudflareAccessUnavailableError",
    "CloudflareAccessVerifier",
    "build_cloudflare_access_verifier",
    "IssuedPersonalToken",
    "PersonalTokenSummary",
    "Principal",
    "TOKEN_SCOPES",
]
