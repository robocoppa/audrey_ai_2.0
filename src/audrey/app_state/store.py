"""Versioned SQLite authority for Audrey users and provider identities.

The native application database is authoritative for stable user ownership.
External providers prove who authenticated; they do not choose Audrey resource
ids or private-storage namespaces. SQLite runs in WAL mode because Audrey is a
single-node deployment and reads must continue while short identity writes
commit.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
import threading
import uuid
from pathlib import Path

from audrey.identity import Principal

_ALLOWED_ROLES = frozenset({"user", "admin"})

_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS app_users (
          user_id            TEXT PRIMARY KEY,
          storage_namespace  TEXT NOT NULL UNIQUE,
          current_email      TEXT NOT NULL,
          display_name       TEXT NOT NULL DEFAULT '',
          role               TEXT NOT NULL CHECK (role IN ('user', 'admin')),
          status             TEXT NOT NULL DEFAULT 'active'
                             CHECK (status IN ('active', 'disabled')),
          created_at         TEXT NOT NULL,
          updated_at         TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS external_identities (
          provider       TEXT NOT NULL,
          subject        TEXT NOT NULL,
          user_id        TEXT NOT NULL,
          email          TEXT NOT NULL,
          created_at     TEXT NOT NULL,
          last_seen_at   TEXT NOT NULL,
          PRIMARY KEY (provider, subject),
          FOREIGN KEY (user_id) REFERENCES app_users(user_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_external_identities_user
          ON external_identities(user_id);
        """,
    ),
)


class InvalidIdentityError(ValueError):
    """Authentication evidence is incomplete or outside Audrey policy."""


class IdentityConflictError(RuntimeError):
    """A provider binding would implicitly merge two Audrey accounts."""


class ApplicationStore:
    """Own versioned application records behind short SQLite transactions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._migrate_locked()

    def _migrate_locked(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS app_schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {
            int(row["version"])
            for row in self._conn.execute(
                "SELECT version FROM app_schema_migrations"
            ).fetchall()
        }
        for version, sql in _MIGRATIONS:
            if version in applied:
                continue
            stamp = _utc_now()
            try:
                self._conn.executescript(
                    "BEGIN IMMEDIATE;\n"
                    f"{sql}\n"
                    "INSERT INTO app_schema_migrations(version, applied_at) "
                    f"VALUES ({version}, '{stamp}');\n"
                    "COMMIT;"
                )
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS version "
                "FROM app_schema_migrations"
            ).fetchone()
            return int(row["version"])

    async def resolve_external_identity(
        self,
        *,
        provider: str,
        subject: str,
        email: str,
        display_name: str,
        role: str,
        auth_method: str,
        legacy_storage_namespace: str | None = None,
    ) -> Principal:
        """Resolve or create one stable principal from provider evidence.

        Existing OWUI users pass their exact current email as
        ``legacy_storage_namespace`` so deployed collections and disk paths are
        not renamed. The provider subject, never a similar-looking email,
        controls whether a later login is the same Audrey account.
        """

        return await asyncio.to_thread(
            self._resolve_external_identity_sync,
            provider,
            subject,
            email,
            display_name,
            role,
            auth_method,
            legacy_storage_namespace,
        )

    def _resolve_external_identity_sync(
        self,
        provider: str,
        subject: str,
        email: str,
        display_name: str,
        role: str,
        auth_method: str,
        legacy_storage_namespace: str | None,
    ) -> Principal:
        provider = _required(provider, "provider").lower()
        subject = _required(subject, "provider subject")
        email = _required(email, "email")
        auth_method = _required(auth_method, "auth method")
        display_name = display_name.strip()
        role = role.strip().lower()
        if role not in _ALLOWED_ROLES:
            raise InvalidIdentityError(f"unsupported Audrey role {role!r}")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._identity_row_locked(provider, subject)
                now = _utc_now()
                if row is not None:
                    user_id = str(row["user_id"])
                    self._conn.execute(
                        "UPDATE app_users SET current_email = ?, display_name = ?, "
                        "role = ?, updated_at = ? WHERE user_id = ?",
                        (email, display_name, role, now, user_id),
                    )
                    self._conn.execute(
                        "UPDATE external_identities SET email = ?, last_seen_at = ? "
                        "WHERE provider = ? AND subject = ?",
                        (email, now, provider, subject),
                    )
                else:
                    namespace = (
                        _required(legacy_storage_namespace, "storage namespace")
                        if legacy_storage_namespace is not None
                        else f"ns_{uuid.uuid4().hex}"
                    )
                    collision = self._conn.execute(
                        "SELECT user_id FROM app_users WHERE storage_namespace = ?",
                        (namespace,),
                    ).fetchone()
                    if collision is not None:
                        raise IdentityConflictError(
                            "storage namespace already belongs to another identity"
                        )
                    user_id = f"usr_{uuid.uuid4().hex}"
                    self._conn.execute(
                        "INSERT INTO app_users "
                        "(user_id, storage_namespace, current_email, display_name, "
                        "role, status, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
                        (user_id, namespace, email, display_name, role, now, now),
                    )
                    self._conn.execute(
                        "INSERT INTO external_identities "
                        "(provider, subject, user_id, email, created_at, last_seen_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (provider, subject, user_id, email, now, now),
                    )
                row = self._identity_row_locked(provider, subject)
                assert row is not None
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

        return _principal_from_row(row, auth_method=auth_method)

    def _identity_row_locked(
        self,
        provider: str,
        subject: str,
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT u.user_id, u.storage_namespace, u.current_email, "
            "u.display_name, u.role, u.status, i.provider, i.subject "
            "FROM external_identities AS i "
            "JOIN app_users AS u ON u.user_id = i.user_id "
            "WHERE i.provider = ? AND i.subject = ?",
            (provider, subject),
        ).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _required(value: str | None, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise InvalidIdentityError(f"{label} is required")
    return clean


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


def _principal_from_row(row: sqlite3.Row, *, auth_method: str) -> Principal:
    return Principal(
        user_id=str(row["user_id"]),
        storage_namespace=str(row["storage_namespace"]),
        provider=str(row["provider"]),
        provider_subject=str(row["subject"]),
        email=str(row["current_email"]),
        display_name=str(row["display_name"]),
        role=str(row["role"]),
        status=str(row["status"]),
        auth_method=auth_method,
    )


__all__ = [
    "ApplicationStore",
    "IdentityConflictError",
    "InvalidIdentityError",
]
