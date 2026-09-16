"""Versioned SQLite authority for Audrey's canonical application records.

The native application database is authoritative for stable user ownership.
External providers prove who authenticated; they do not choose Audrey resource
ids or private-storage namespaces. SQLite runs in WAL mode because Audrey is a
single-node deployment and reads must continue while short identity writes
commit.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path

from audrey.app_state.migrations import MIGRATIONS
from audrey.app_state.records import AdminUserRecord, LocalUserDataPurge, ModelAccessPolicy
from audrey.app_state.repositories import (
    ChatProjectionsRepository,
    ConversationsRepository,
    PreferencesRepository,
)
from audrey.identity import (
    TOKEN_SCOPES,
    IssuedPersonalToken,
    PersonalTokenSummary,
    Principal,
)

_ALLOWED_ROLES = frozenset({"user", "admin"})
_ALLOWED_ACCOUNT_STATUSES = frozenset({"pending", "active", "disabled"})
_ALLOWED_GROUPS = frozenset({"users", "testers", "admins"})
_ALLOWED_MODEL_AUDIENCES = frozenset({"users", "testers", "admins"})
_TOKEN_RE = re.compile(r"\Aaud_(pat_[0-9a-f]{32})\.([A-Za-z0-9_-]{32,})\Z")
_LAST_USED_WRITE_INTERVAL = dt.timedelta(minutes=5)
_FOREIGN_KEYS_OFF_MIGRATIONS = frozenset({5, 7, 8})

class InvalidIdentityError(ValueError):
    """Authentication evidence is incomplete or outside Audrey policy."""


class IdentityConflictError(RuntimeError):
    """A provider binding would implicitly merge two Audrey accounts."""


class PersonalTokenAuthenticationError(ValueError):
    """A personal token is malformed, invalid, expired, revoked, or disabled."""


class AccountAdministrationError(ValueError):
    """An administrative account or authorization mutation is unsafe."""


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
        self.preferences = PreferencesRepository(self._conn, self._lock)
        self.conversations = ConversationsRepository(self._conn, self._lock)
        self.chat_projections = ChatProjectionsRepository(self._conn, self._lock)

    def _migrate_locked(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS app_schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {
            int(row["version"])
            for row in self._conn.execute("SELECT version FROM app_schema_migrations").fetchall()
        }
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            stamp = _utc_now()
            if version in _FOREIGN_KEYS_OFF_MIGRATIONS:
                self._apply_table_rebuild_migration_locked(version, sql, stamp)
                continue
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

    def _apply_table_rebuild_migration_locked(
        self,
        version: int,
        sql: str,
        stamp: str,
    ) -> None:
        """Apply a table rebuild without cascading through canonical children."""

        self._conn.execute("PRAGMA foreign_keys = OFF")
        try:
            self._conn.executescript(f"BEGIN IMMEDIATE;\n{sql}\n")
            violations = self._conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise sqlite3.IntegrityError(
                    f"migration {version} produced foreign-key violations"
                )
            self._conn.execute(
                "INSERT INTO app_schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, stamp),
            )
            self._conn.commit()
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        finally:
            self._conn.execute("PRAGMA legacy_alter_table = OFF")
            self._conn.execute("PRAGMA foreign_keys = ON")

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM app_schema_migrations"
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
        sync_role: bool = True,
        sync_display_name: bool = True,
        initial_status: str = "active",
    ) -> Principal:
        """Resolve or create one stable principal from provider evidence.

        Existing OWUI users pass their exact current email as
        ``legacy_storage_namespace`` so deployed collections and disk paths are
        not renamed. The provider subject, never a similar-looking email,
        controls whether a later login is the same Audrey account. Providers
        without Audrey role authority pass ``sync_role=False`` so a login can
        refresh profile evidence without granting or removing local access.
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
            sync_role,
            sync_display_name,
            initial_status,
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
        sync_role: bool,
        sync_display_name: bool,
        initial_status: str,
    ) -> Principal:
        provider = _required(provider, "provider").lower()
        subject = _required(subject, "provider subject")
        email = _required(email, "email")
        auth_method = _required(auth_method, "auth method")
        display_name = display_name.strip()
        role = role.strip().lower()
        if role not in _ALLOWED_ROLES:
            raise InvalidIdentityError(f"unsupported Audrey role {role!r}")
        initial_status = initial_status.strip().lower()
        if initial_status not in {"pending", "active"}:
            raise InvalidIdentityError(
                f"unsupported initial Audrey account status {initial_status!r}"
            )

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._identity_row_locked(provider, subject)
                now = _utc_now()
                if row is not None:
                    user_id = str(row["user_id"])
                    self._conn.execute(
                        "UPDATE app_users SET current_email = ?, "
                        "display_name = CASE WHEN ? THEN ? ELSE display_name END, "
                        "role = CASE WHEN ? THEN ? ELSE role END, updated_at = ? "
                        "WHERE user_id = ?",
                        (
                            email,
                            sync_display_name,
                            display_name,
                            sync_role,
                            role,
                            now,
                            user_id,
                        ),
                    )
                    self._conn.execute(
                        "UPDATE external_identities SET email = ?, last_seen_at = ? "
                        "WHERE provider = ? AND subject = ?",
                        (email, now, provider, subject),
                    )
                    if sync_role:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO user_group_memberships "
                            "(user_id, group_id, granted_by_user_id, granted_at) "
                            "VALUES (?, 'users', NULL, ?)",
                            (user_id, now),
                        )
                        if role == "admin":
                            self._conn.execute(
                                "INSERT OR IGNORE INTO user_group_memberships "
                                "(user_id, group_id, granted_by_user_id, granted_at) "
                                "VALUES (?, 'admins', NULL, ?)",
                                (user_id, now),
                            )
                        else:
                            self._conn.execute(
                                "DELETE FROM user_group_memberships "
                                "WHERE user_id = ? AND group_id = 'admins'",
                                (user_id,),
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
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            user_id,
                            namespace,
                            email,
                            display_name,
                            role,
                            initial_status,
                            now,
                            now,
                        ),
                    )
                    if initial_status == "active":
                        self._conn.execute(
                            "INSERT INTO user_group_memberships "
                            "(user_id, group_id, granted_by_user_id, granted_at) "
                            "VALUES (?, 'users', NULL, ?)",
                            (user_id, now),
                        )
                        if role == "admin":
                            self._conn.execute(
                                "INSERT INTO user_group_memberships "
                                "(user_id, group_id, granted_by_user_id, granted_at) "
                                "VALUES (?, 'admins', NULL, ?)",
                                (user_id, now),
                            )
                    self._conn.execute(
                        "INSERT INTO user_preferences "
                        "(user_id, timezone, persona, response_preferences_json, "
                        "created_at, updated_at) VALUES (?, 'UTC', '', '{}', ?, ?)",
                        (user_id, now, now),
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
            "u.display_name, u.role, u.status, i.provider, i.subject, "
            "COALESCE((SELECT group_concat(m.group_id, ',') "
            "FROM user_group_memberships AS m WHERE m.user_id = u.user_id), '') "
            "AS groups_csv "
            "FROM external_identities AS i "
            "JOIN app_users AS u ON u.user_id = i.user_id "
            "WHERE i.provider = ? AND i.subject = ?",
            (provider, subject),
        ).fetchone()

    async def update_display_name(
        self,
        *,
        user_id: str,
        display_name: str,
    ) -> str:
        """Persist one account-owned profile name without touching auth evidence."""

        return await asyncio.to_thread(
            self._update_display_name_sync,
            user_id,
            display_name,
        )

    def _update_display_name_sync(
        self,
        user_id: str,
        display_name: str,
    ) -> str:
        user_id = _required(user_id, "user id")
        display_name = _required(display_name, "display name")
        if len(display_name) > 100:
            raise InvalidIdentityError(
                "display name must be at most 100 characters"
            )

        with self._lock:
            cursor = self._conn.execute(
                "UPDATE app_users SET display_name = ?, updated_at = ? "
                "WHERE user_id = ? AND status = 'active'",
                (display_name, _utc_now(), user_id),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise InvalidIdentityError("profile owner does not exist")
            self._conn.commit()
        return display_name

    async def create_personal_token(
        self,
        *,
        user_id: str,
        name: str,
        scopes: Iterable[str],
        expires_at: str,
    ) -> IssuedPersonalToken:
        """Create a high-entropy bearer token and persist only its digest."""

        return await asyncio.to_thread(
            self._create_personal_token_sync,
            user_id,
            name,
            scopes,
            expires_at,
        )

    def _create_personal_token_sync(
        self,
        user_id: str,
        name: str,
        scopes: Iterable[str],
        expires_at: str,
    ) -> IssuedPersonalToken:
        user_id = _required(user_id, "user id")
        name = _required(name, "token name")
        if len(name) > 80:
            raise InvalidIdentityError("token name must be at most 80 characters")
        normalized_scopes = _normalize_scopes(scopes)
        normalized_expiry = _normalize_expiry(expires_at, require_future=True)
        token_id = f"pat_{uuid.uuid4().hex}"
        raw_token = f"aud_{token_id}.{secrets.token_urlsafe(32)}"
        secret_hash = _token_hash(raw_token)
        now = _utc_now()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                owner = self._conn.execute(
                    "SELECT status FROM app_users WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if owner is None:
                    raise InvalidIdentityError("token owner does not exist")
                if str(owner["status"]) != "active":
                    raise InvalidIdentityError("token owner is disabled")
                self._conn.execute(
                    "INSERT INTO personal_access_tokens "
                    "(token_id, user_id, name, secret_hash, scopes_json, "
                    "created_at, expires_at, last_used_at, revoked_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                    (
                        token_id,
                        user_id,
                        name,
                        secret_hash,
                        json.dumps(normalized_scopes, separators=(",", ":")),
                        now,
                        normalized_expiry,
                    ),
                )
                row = self._token_row_for_owner_locked(user_id, token_id)
                assert row is not None
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

        return IssuedPersonalToken(
            token=raw_token,
            record=_token_summary_from_row(row),
        )

    async def list_personal_tokens(
        self,
        *,
        user_id: str,
    ) -> tuple[PersonalTokenSummary, ...]:
        return await asyncio.to_thread(self._list_personal_tokens_sync, user_id)

    def _list_personal_tokens_sync(
        self,
        user_id: str,
    ) -> tuple[PersonalTokenSummary, ...]:
        user_id = _required(user_id, "user id")
        with self._lock:
            rows = self._conn.execute(
                "SELECT token_id, name, scopes_json, created_at, expires_at, "
                "last_used_at, revoked_at FROM personal_access_tokens "
                "WHERE user_id = ? ORDER BY created_at DESC, token_id DESC",
                (user_id,),
            ).fetchall()
        return tuple(_token_summary_from_row(row) for row in rows)

    async def revoke_personal_token(
        self,
        *,
        user_id: str,
        token_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._revoke_personal_token_sync,
            user_id,
            token_id,
        )

    def _revoke_personal_token_sync(
        self,
        user_id: str,
        token_id: str,
    ) -> bool:
        user_id = _required(user_id, "user id")
        token_id = _required(token_id, "token id")
        with self._lock:
            row = self._token_row_for_owner_locked(user_id, token_id)
            if row is None:
                return False
            if not str(row["revoked_at"] or ""):
                self._conn.execute(
                    "UPDATE personal_access_tokens SET revoked_at = ? "
                    "WHERE user_id = ? AND token_id = ?",
                    (_utc_now(), user_id, token_id),
                )
                self._conn.commit()
        return True

    async def authenticate_personal_token(self, token: str) -> Principal:
        """Resolve a bearer secret without caching revocation or account state."""

        return await asyncio.to_thread(
            self._authenticate_personal_token_sync,
            token,
        )

    def _authenticate_personal_token_sync(self, token: str) -> Principal:
        match = _TOKEN_RE.fullmatch(str(token or ""))
        if match is None:
            raise PersonalTokenAuthenticationError("invalid personal token")
        token_id = match.group(1)
        now_dt = dt.datetime.now(dt.UTC)

        with self._lock:
            row = self._conn.execute(
                "SELECT t.token_id, t.secret_hash, t.scopes_json, t.expires_at, "
                "t.last_used_at, t.revoked_at, u.user_id, u.storage_namespace, "
                "u.current_email, u.display_name, u.role, u.status, "
                "COALESCE((SELECT group_concat(m.group_id, ',') "
                "FROM user_group_memberships AS m WHERE m.user_id = u.user_id), '') "
                "AS groups_csv "
                "FROM personal_access_tokens AS t "
                "JOIN app_users AS u ON u.user_id = t.user_id "
                "WHERE t.token_id = ?",
                (token_id,),
            ).fetchone()
            if row is None or not hmac.compare_digest(
                str(row["secret_hash"]),
                _token_hash(token),
            ):
                raise PersonalTokenAuthenticationError("invalid personal token")
            if str(row["revoked_at"] or ""):
                raise PersonalTokenAuthenticationError("invalid personal token")
            try:
                expiry = _parse_utc(str(row["expires_at"] or ""))
                scopes = frozenset(_scopes_from_json(str(row["scopes_json"])))
                last_used = _parse_utc(str(row["last_used_at"] or ""))
            except ValueError as exc:
                raise PersonalTokenAuthenticationError("invalid personal token") from exc
            if expiry is None or expiry <= now_dt:
                raise PersonalTokenAuthenticationError("invalid personal token")
            if str(row["status"]) != "active":
                raise PersonalTokenAuthenticationError("invalid personal token")
            if last_used is None or now_dt - last_used >= _LAST_USED_WRITE_INTERVAL:
                self._conn.execute(
                    "UPDATE personal_access_tokens SET last_used_at = ? WHERE token_id = ?",
                    (now_dt.isoformat(timespec="microseconds"), token_id),
                )
                self._conn.commit()

        return Principal(
            user_id=str(row["user_id"]),
            storage_namespace=str(row["storage_namespace"]),
            provider="audrey",
            provider_subject=token_id,
            email=str(row["current_email"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            status=str(row["status"]),
            auth_method="personal_token",
            groups=_groups_from_csv(str(row["groups_csv"])),
            token_id=token_id,
            scopes=scopes,
        )

    def _token_row_for_owner_locked(
        self,
        user_id: str,
        token_id: str,
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT token_id, name, scopes_json, created_at, expires_at, "
            "last_used_at, revoked_at FROM personal_access_tokens "
            "WHERE user_id = ? AND token_id = ?",
            (user_id, token_id),
        ).fetchone()

    async def delete_personal_tokens(self, *, user_id: str) -> int:
        """Erase every personal-token record owned by one Audrey account."""

        return await asyncio.to_thread(self._delete_personal_tokens_sync, user_id)

    def _delete_personal_tokens_sync(self, user_id: str) -> int:
        user_id = _required(user_id, "user id")
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM personal_access_tokens WHERE user_id = ?",
                (user_id,),
            )
            self._conn.commit()
            return max(0, int(cursor.rowcount))

    async def purge_local_user_data(self, *, user_id: str) -> LocalUserDataPurge:
        """Atomically erase tokens and canonical app data while retaining identity."""

        return await asyncio.to_thread(self._purge_local_user_data_sync, user_id)

    def _purge_local_user_data_sync(self, user_id: str) -> LocalUserDataPurge:
        user_id = _required(user_id, "user id")
        now = _utc_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                owner = self._conn.execute(
                    "SELECT 1 FROM app_users WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if owner is None:
                    raise InvalidIdentityError("purge owner does not exist")
                messages_deleted = _owned_count(
                    self._conn,
                    "app_messages",
                    user_id,
                )
                runs_deleted = _owned_count(self._conn, "app_runs", user_id)
                conversations_deleted = _owned_count(
                    self._conn,
                    "app_conversations",
                    user_id,
                )
                token_cursor = self._conn.execute(
                    "DELETE FROM personal_access_tokens WHERE user_id = ?",
                    (user_id,),
                )
                self._conn.execute(
                    "DELETE FROM app_chat_projection_deletions WHERE user_id = ?",
                    (user_id,),
                )
                self._conn.execute(
                    "DELETE FROM app_conversations WHERE user_id = ?",
                    (user_id,),
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO user_preferences "
                    "(user_id, timezone, persona, response_preferences_json, "
                    "created_at, updated_at) VALUES (?, 'UTC', '', '{}', ?, ?)",
                    (user_id, now, now),
                )
                self._conn.execute(
                    "UPDATE user_preferences SET timezone = 'UTC', persona = '', "
                    "response_preferences_json = '{}', updated_at = ? WHERE user_id = ?",
                    (now, user_id),
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return LocalUserDataPurge(
            tokens_deleted=max(0, int(token_cursor.rowcount)),
            conversations_deleted=conversations_deleted,
            messages_deleted=messages_deleted,
            runs_deleted=runs_deleted,
            preferences_reset=True,
        )

    async def list_admin_users(
        self,
        *,
        status: str = "",
        group: str = "",
        search: str = "",
        limit: int = 200,
    ) -> tuple[AdminUserRecord, ...]:
        """Return safe account rows for the native administration surface."""

        return await asyncio.to_thread(
            self._list_admin_users_sync,
            status,
            group,
            search,
            limit,
        )

    def _list_admin_users_sync(
        self,
        status: str,
        group: str,
        search: str,
        limit: int,
    ) -> tuple[AdminUserRecord, ...]:
        status = str(status).strip().lower()
        group = str(group).strip().lower()
        search = str(search).strip()
        if status and status not in _ALLOWED_ACCOUNT_STATUSES:
            raise AccountAdministrationError("unsupported account status filter")
        if group and group not in _ALLOWED_GROUPS:
            raise AccountAdministrationError("unsupported account group filter")
        if len(search) > 200:
            raise AccountAdministrationError("account search is too long")
        if not 1 <= limit <= 200:
            raise AccountAdministrationError("account limit must be between 1 and 200")

        with self._lock:
            rows = self._conn.execute(
                "SELECT u.user_id, u.current_email, u.display_name, u.role, "
                "u.status, u.created_at, u.updated_at, "
                "COALESCE((SELECT group_concat(m.group_id, ',') "
                "FROM user_group_memberships AS m WHERE m.user_id = u.user_id), '') "
                "AS groups_csv, "
                "COALESCE((SELECT i.provider FROM external_identities AS i "
                "WHERE i.user_id = u.user_id ORDER BY i.last_seen_at DESC LIMIT 1), '') "
                "AS auth_provider, "
                "COALESCE((SELECT max(i.last_seen_at) FROM external_identities AS i "
                "WHERE i.user_id = u.user_id), '') AS last_seen_at "
                "FROM app_users AS u "
                "WHERE (? = '' OR u.status = ?) "
                "AND (? = '' OR EXISTS (SELECT 1 FROM user_group_memberships AS gm "
                "WHERE gm.user_id = u.user_id AND gm.group_id = ?)) "
                "AND (? = '' OR instr(lower(u.current_email), lower(?)) > 0 "
                "OR instr(lower(u.display_name), lower(?)) > 0) "
                "ORDER BY CASE u.status WHEN 'pending' THEN 0 WHEN 'active' THEN 1 "
                "ELSE 2 END, u.created_at DESC, u.user_id DESC LIMIT ?",
                (
                    status,
                    status,
                    group,
                    group,
                    search,
                    search,
                    search,
                    limit,
                ),
            ).fetchall()
        return tuple(_admin_user_from_row(row) for row in rows)

    async def get_admin_user(self, *, user_id: str) -> AdminUserRecord | None:
        return await asyncio.to_thread(self._get_admin_user_sync, user_id)

    def _get_admin_user_sync(self, user_id: str) -> AdminUserRecord | None:
        user_id = _required(user_id, "user id")
        with self._lock:
            row = self._admin_user_row_locked(user_id)
        return _admin_user_from_row(row) if row is not None else None

    def _admin_user_row_locked(self, user_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT u.user_id, u.current_email, u.display_name, u.role, "
            "u.status, u.created_at, u.updated_at, "
            "COALESCE((SELECT group_concat(m.group_id, ',') "
            "FROM user_group_memberships AS m WHERE m.user_id = u.user_id), '') "
            "AS groups_csv, "
            "COALESCE((SELECT i.provider FROM external_identities AS i "
            "WHERE i.user_id = u.user_id ORDER BY i.last_seen_at DESC LIMIT 1), '') "
            "AS auth_provider, "
            "COALESCE((SELECT max(i.last_seen_at) FROM external_identities AS i "
            "WHERE i.user_id = u.user_id), '') AS last_seen_at "
            "FROM app_users AS u WHERE u.user_id = ?",
            (user_id,),
        ).fetchone()

    async def admin_update_user(
        self,
        *,
        actor_user_id: str,
        target_user_id: str,
        status: str | None = None,
        groups: Iterable[str] | None = None,
        action: str = "update_user",
    ) -> AdminUserRecord:
        """Atomically change account state/memberships with admin safeguards."""

        return await asyncio.to_thread(
            self._admin_update_user_sync,
            actor_user_id,
            target_user_id,
            status,
            groups,
            action,
        )

    def _admin_update_user_sync(
        self,
        actor_user_id: str,
        target_user_id: str,
        status: str | None,
        groups: Iterable[str] | None,
        action: str,
    ) -> AdminUserRecord:
        actor_user_id = _required(actor_user_id, "actor user id")
        target_user_id = _required(target_user_id, "target user id")
        requested_status = str(status).strip().lower() if status is not None else None
        if requested_status is not None and requested_status not in _ALLOWED_ACCOUNT_STATUSES:
            raise AccountAdministrationError("unsupported account status")
        requested_groups = _normalize_groups(groups) if groups is not None else None
        action = _required(action, "admin action")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._assert_admin_locked(actor_user_id)
                before_row = self._admin_user_row_locked(target_user_id)
                if before_row is None:
                    raise AccountAdministrationError("account does not exist")
                before = _admin_user_from_row(before_row)
                next_status = requested_status or before.status
                next_groups = (
                    set(requested_groups)
                    if requested_groups is not None
                    else set(before.groups)
                )

                if next_status == "pending" and next_groups:
                    raise AccountAdministrationError(
                        "pending accounts cannot have access groups"
                    )
                if next_status == "active" and "users" not in next_groups:
                    raise AccountAdministrationError(
                        "active accounts must belong to users"
                    )
                if actor_user_id == target_user_id and next_status != "active":
                    raise AccountAdministrationError(
                        "an administrator cannot disable their own account"
                    )
                if (
                    actor_user_id == target_user_id
                    and "admins" in before.groups
                    and "admins" not in next_groups
                ):
                    raise AccountAdministrationError(
                        "an administrator cannot remove their own admin access"
                    )
                if "admins" in before.groups and "admins" not in next_groups:
                    remaining = self._conn.execute(
                        "SELECT COUNT(*) AS total FROM user_group_memberships AS m "
                        "JOIN app_users AS u ON u.user_id = m.user_id "
                        "WHERE m.group_id = 'admins' AND u.status = 'active' "
                        "AND m.user_id != ?",
                        (target_user_id,),
                    ).fetchone()
                    if int(remaining["total"]) < 1:
                        raise AccountAdministrationError(
                            "the last active administrator cannot be removed"
                        )

                now = _utc_now()
                role = "admin" if "admins" in next_groups else "user"
                self._conn.execute(
                    "UPDATE app_users SET status = ?, role = ?, updated_at = ? "
                    "WHERE user_id = ?",
                    (next_status, role, now, target_user_id),
                )
                self._conn.execute(
                    "DELETE FROM user_group_memberships WHERE user_id = ?",
                    (target_user_id,),
                )
                self._conn.executemany(
                    "INSERT INTO user_group_memberships "
                    "(user_id, group_id, granted_by_user_id, granted_at) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (target_user_id, group_id, actor_user_id, now)
                        for group_id in sorted(next_groups)
                    ],
                )
                after_row = self._admin_user_row_locked(target_user_id)
                assert after_row is not None
                after = _admin_user_from_row(after_row)
                self._insert_audit_locked(
                    actor_user_id=actor_user_id,
                    target_type="user",
                    target_id=target_user_id,
                    action=action,
                    before=_admin_user_snapshot(before),
                    after=_admin_user_snapshot(after),
                    now=now,
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return after

    async def bootstrap_admin(self, *, user_id: str) -> AdminUserRecord:
        """One-time operator path for granting a native identity admin access."""

        return await asyncio.to_thread(self._bootstrap_admin_sync, user_id)

    async def bootstrap_admin_by_email(self, *, email: str) -> AdminUserRecord:
        """Grant admin access to the one account with an exact email match."""

        return await asyncio.to_thread(self._bootstrap_admin_sync, None, email=email)

    def _bootstrap_admin_sync(
        self, user_id: str | None, *, email: str | None = None
    ) -> AdminUserRecord:
        if email is None:
            user_id = _required(user_id, "user id")
        else:
            email = _required(email, "email")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if email is not None:
                    rows = self._conn.execute(
                        "SELECT user_id FROM app_users "
                        "WHERE lower(current_email) = lower(?) ORDER BY user_id",
                        (email,),
                    ).fetchall()
                    if not rows:
                        raise AccountAdministrationError("account does not exist")
                    if len(rows) > 1:
                        raise AccountAdministrationError(
                            "multiple accounts use that email; grant by exact user id"
                        )
                    user_id = str(rows[0]["user_id"])
                assert user_id is not None
                before_row = self._admin_user_row_locked(user_id)
                if before_row is None:
                    raise AccountAdministrationError("account does not exist")
                before = _admin_user_from_row(before_row)
                now = _utc_now()
                self._conn.execute(
                    "UPDATE app_users SET status = 'active', role = 'admin', updated_at = ? "
                    "WHERE user_id = ?",
                    (now, user_id),
                )
                for group_id in ("users", "admins"):
                    self._conn.execute(
                        "INSERT OR IGNORE INTO user_group_memberships "
                        "(user_id, group_id, granted_by_user_id, granted_at) "
                        "VALUES (?, ?, NULL, ?)",
                        (user_id, group_id, now),
                    )
                after_row = self._admin_user_row_locked(user_id)
                assert after_row is not None
                after = _admin_user_from_row(after_row)
                self._insert_audit_locked(
                    actor_user_id=None,
                    target_type="user",
                    target_id=user_id,
                    action="bootstrap_admin",
                    before=_admin_user_snapshot(before),
                    after=_admin_user_snapshot(after),
                    now=now,
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return after

    async def list_model_access_policies(self) -> tuple[ModelAccessPolicy, ...]:
        return await asyncio.to_thread(self._list_model_access_policies_sync)

    def _list_model_access_policies_sync(self) -> tuple[ModelAccessPolicy, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_id, enabled, audience, updated_at "
                "FROM model_access_policies ORDER BY model_id"
            ).fetchall()
        return tuple(_model_policy_from_row(row) for row in rows)

    async def set_model_access_policy(
        self,
        *,
        actor_user_id: str,
        model_id: str,
        enabled: bool,
        audience: str,
    ) -> ModelAccessPolicy:
        return await asyncio.to_thread(
            self._set_model_access_policy_sync,
            actor_user_id,
            model_id,
            enabled,
            audience,
        )

    def _set_model_access_policy_sync(
        self,
        actor_user_id: str,
        model_id: str,
        enabled: bool,
        audience: str,
    ) -> ModelAccessPolicy:
        actor_user_id = _required(actor_user_id, "actor user id")
        model_id = _required(model_id, "model id")
        if len(model_id) > 200:
            raise AccountAdministrationError("model id is too long")
        audience = str(audience).strip().lower()
        if audience not in _ALLOWED_MODEL_AUDIENCES:
            raise AccountAdministrationError("unsupported model audience")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._assert_admin_locked(actor_user_id)
                before_row = self._conn.execute(
                    "SELECT model_id, enabled, audience, updated_at "
                    "FROM model_access_policies WHERE model_id = ?",
                    (model_id,),
                ).fetchone()
                now = _utc_now()
                self._conn.execute(
                    "INSERT INTO model_access_policies "
                    "(model_id, enabled, audience, updated_by_user_id, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(model_id) DO UPDATE SET enabled = excluded.enabled, "
                    "audience = excluded.audience, "
                    "updated_by_user_id = excluded.updated_by_user_id, "
                    "updated_at = excluded.updated_at",
                    (model_id, int(bool(enabled)), audience, actor_user_id, now),
                )
                row = self._conn.execute(
                    "SELECT model_id, enabled, audience, updated_at "
                    "FROM model_access_policies WHERE model_id = ?",
                    (model_id,),
                ).fetchone()
                assert row is not None
                policy = _model_policy_from_row(row)
                self._insert_audit_locked(
                    actor_user_id=actor_user_id,
                    target_type="model",
                    target_id=model_id,
                    action="set_model_policy",
                    before=(
                        _model_policy_snapshot(_model_policy_from_row(before_row))
                        if before_row is not None
                        else {}
                    ),
                    after=_model_policy_snapshot(policy),
                    now=now,
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return policy

    async def delete_model_access_policy(
        self,
        *,
        actor_user_id: str,
        model_id: str,
    ) -> bool:
        """Remove a database override so deployment configuration applies again."""

        return await asyncio.to_thread(
            self._delete_model_access_policy_sync,
            actor_user_id,
            model_id,
        )

    def _delete_model_access_policy_sync(
        self,
        actor_user_id: str,
        model_id: str,
    ) -> bool:
        actor_user_id = _required(actor_user_id, "actor user id")
        model_id = _required(model_id, "model id")
        if len(model_id) > 200:
            raise AccountAdministrationError("model id is too long")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._assert_admin_locked(actor_user_id)
                before_row = self._conn.execute(
                    "SELECT model_id, enabled, audience, updated_at "
                    "FROM model_access_policies WHERE model_id = ?",
                    (model_id,),
                ).fetchone()
                if before_row is None:
                    self._conn.commit()
                    return False
                now = _utc_now()
                self._conn.execute(
                    "DELETE FROM model_access_policies WHERE model_id = ?",
                    (model_id,),
                )
                self._insert_audit_locked(
                    actor_user_id=actor_user_id,
                    target_type="model",
                    target_id=model_id,
                    action="delete_model_policy",
                    before=_model_policy_snapshot(_model_policy_from_row(before_row)),
                    after={},
                    now=now,
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return True

    def _assert_admin_locked(self, user_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM app_users AS u "
            "JOIN user_group_memberships AS m ON m.user_id = u.user_id "
            "WHERE u.user_id = ? AND u.status = 'active' AND m.group_id = 'admins'",
            (user_id,),
        ).fetchone()
        if row is None:
            raise AccountAdministrationError("active administrator required")

    def _insert_audit_locked(
        self,
        *,
        actor_user_id: str | None,
        target_type: str,
        target_id: str,
        action: str,
        before: dict[str, object],
        after: dict[str, object],
        now: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO admin_audit_events "
            "(event_id, actor_user_id, target_type, target_id, action, before_json, "
            "after_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"audit_{uuid.uuid4().hex}",
                actor_user_id,
                target_type,
                target_id,
                action,
                json.dumps(before, separators=(",", ":"), sort_keys=True),
                json.dumps(after, separators=(",", ":"), sort_keys=True),
                now,
            ),
        )

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
        groups=_groups_from_csv(str(row["groups_csv"])),
    )


def _groups_from_csv(value: str) -> frozenset[str]:
    return frozenset(group for group in value.split(",") if group)


def _normalize_groups(groups: Iterable[str]) -> tuple[str, ...]:
    if isinstance(groups, (str, bytes)):
        raise AccountAdministrationError("account groups must be a collection")
    clean = tuple(
        sorted({str(group).strip().lower() for group in groups if str(group).strip()})
    )
    unknown = sorted(set(clean) - _ALLOWED_GROUPS)
    if unknown:
        raise AccountAdministrationError(
            "unsupported account group: " + ", ".join(unknown)
        )
    return clean


def _admin_user_from_row(row: sqlite3.Row) -> AdminUserRecord:
    return AdminUserRecord(
        user_id=str(row["user_id"]),
        email=str(row["current_email"]),
        display_name=str(row["display_name"]),
        role=str(row["role"]),
        status=str(row["status"]),
        groups=tuple(sorted(_groups_from_csv(str(row["groups_csv"])))),
        auth_provider=str(row["auth_provider"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_seen_at=str(row["last_seen_at"]),
    )


def _admin_user_snapshot(record: AdminUserRecord) -> dict[str, object]:
    return {
        "status": record.status,
        "role": record.role,
        "groups": list(record.groups),
    }


def _model_policy_from_row(row: sqlite3.Row) -> ModelAccessPolicy:
    return ModelAccessPolicy(
        model_id=str(row["model_id"]),
        enabled=bool(row["enabled"]),
        audience=str(row["audience"]),
        updated_at=str(row["updated_at"]),
    )


def _model_policy_snapshot(record: ModelAccessPolicy) -> dict[str, object]:
    return {
        "enabled": record.enabled,
        "audience": record.audience,
    }


def _normalize_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    if isinstance(scopes, (str, bytes)):
        raise InvalidIdentityError("token scopes must be a collection")
    clean = tuple(sorted({str(scope).strip() for scope in scopes if str(scope).strip()}))
    if not clean:
        raise InvalidIdentityError("at least one token scope is required")
    unknown = sorted(set(clean) - TOKEN_SCOPES)
    if unknown:
        raise InvalidIdentityError("unsupported token scope: " + ", ".join(unknown))
    return clean


def _normalize_expiry(value: str, *, require_future: bool) -> str:
    if not str(value or "").strip():
        raise InvalidIdentityError("token expiry is required")
    try:
        parsed = _parse_utc(str(value))
    except ValueError as exc:
        raise InvalidIdentityError("token expiry must be an ISO-8601 UTC timestamp") from exc
    assert parsed is not None
    if require_future and parsed <= dt.datetime.now(dt.UTC):
        raise InvalidIdentityError("token expiry must be in the future")
    return parsed.isoformat(timespec="microseconds")


def _parse_utc(value: str) -> dt.datetime | None:
    clean = value.strip()
    if not clean:
        return None
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(dt.UTC)


def _scopes_from_json(value: str) -> tuple[str, ...]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError("token scopes are not a list")
    try:
        return _normalize_scopes(decoded)
    except InvalidIdentityError as exc:
        raise ValueError("invalid stored token scopes") from exc


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_summary_from_row(row: sqlite3.Row) -> PersonalTokenSummary:
    return PersonalTokenSummary(
        token_id=str(row["token_id"]),
        name=str(row["name"]),
        scopes=_scopes_from_json(str(row["scopes_json"])),
        created_at=str(row["created_at"]),
        expires_at=str(row["expires_at"] or ""),
        last_used_at=str(row["last_used_at"] or ""),
        revoked_at=str(row["revoked_at"] or ""),
    )


def _owned_count(connection: sqlite3.Connection, table: str, user_id: str) -> int:
    if table not in {"app_conversations", "app_messages", "app_runs"}:
        raise ValueError("unsupported application-state table")
    row = connection.execute(
        f"SELECT COUNT(*) AS total FROM {table} WHERE user_id = ?",  # noqa: S608
        (user_id,),
    ).fetchone()
    return int(row["total"])


__all__ = [
    "AccountAdministrationError",
    "ApplicationStore",
    "IdentityConflictError",
    "InvalidIdentityError",
    "PersonalTokenAuthenticationError",
]
