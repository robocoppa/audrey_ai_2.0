"""Stable Audrey principals resolved from external authentication evidence."""

from __future__ import annotations

from dataclasses import dataclass


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

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


__all__ = ["Principal"]
