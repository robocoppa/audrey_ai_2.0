"""Stable Audrey principals resolved from external authentication evidence."""

from __future__ import annotations

from dataclasses import dataclass

TOKEN_SCOPES = frozenset({"account:read", "compat:full"})


@dataclass(frozen=True, slots=True)
class Principal:
    """One authenticated Audrey account, independent of its login provider.

    ``user_id`` and ``storage_namespace`` are Audrey-owned, durable identifiers.
    Email and display name are mutable profile attributes. Provider subject is
    retained for audit and adapter decisions, but native resource ownership is
    always keyed by ``user_id``.
    """

    user_id: str
    storage_namespace: str
    provider: str
    provider_subject: str
    email: str
    display_name: str
    role: str
    status: str
    auth_method: str
    token_id: str | None = None
    scopes: frozenset[str] = frozenset()

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass(frozen=True, slots=True)
class PersonalTokenSummary:
    """Safe token metadata; the bearer secret is never retained here."""

    token_id: str
    name: str
    scopes: tuple[str, ...]
    created_at: str
    expires_at: str
    last_used_at: str
    revoked_at: str


@dataclass(frozen=True, slots=True)
class IssuedPersonalToken:
    """One-time bearer secret paired with its safe persistent metadata."""

    token: str
    record: PersonalTokenSummary


__all__ = [
    "IssuedPersonalToken",
    "PersonalTokenSummary",
    "Principal",
    "TOKEN_SCOPES",
]
