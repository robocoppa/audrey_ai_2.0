"""Explicit operator bootstrap for Audrey's first native administrator."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import asdict
from getpass import getpass
from pathlib import Path
from urllib.parse import quote

import httpx

from audrey.app_state import AccountAdministrationError, ApplicationStore
from audrey.app_state.history_import import (
    HistoryImportError,
    HistoryImportRepository,
    load_archive_export,
)
from audrey.config import get_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audrey-admin",
        description="Administer Audrey's native account authority.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    grant = commands.add_parser(
        "grant-admin",
        help="Activate one Audrey account and grant its admins group.",
    )
    target = grant.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "user_id",
        nargs="?",
        help="Canonical usr_... id shown by /api/me",
    )
    target.add_argument(
        "--email",
        help="Exact account email (fails safely when more than one account matches)",
    )
    importer = commands.add_parser(
        "import-owui-users",
        help="Preview or import unmatched Open WebUI users as pending accounts.",
    )
    importer.add_argument(
        "--apply",
        action="store_true",
        help="Create pending accounts after reviewing the default preview.",
    )
    history = commands.add_parser(
        "import-chat-export",
        help="Preview or import one Audrey chat-export JSON file into native history.",
    )
    history.add_argument("--file", type=Path, required=True)
    history.add_argument("--user-id", required=True, help="Exact target usr_... id")
    history.add_argument("--email", required=True, help="Target account email safety check")
    history.add_argument(
        "--apply",
        action="store_true",
        help="Import after an application-database backup and preview.",
    )
    history.add_argument(
        "--allow-unbound",
        action="store_true",
        help="Permit an older export without Audrey owner metadata.",
    )
    history.add_argument(
        "--backup-to",
        type=Path,
        help="New online SQLite backup path, required with --apply.",
    )
    return parser


async def _grant_admin(user_id: str | None = None, *, email: str | None = None) -> int:
    cfg = get_config()
    application = cfg.raw.get("application", {}) or {}
    store = ApplicationStore(
        application.get("sqlite_path", "/data/audrey_app.sqlite")
    )
    try:
        if email is not None:
            record = await store.bootstrap_admin_by_email(email=email)
        elif user_id is not None:
            record = await store.bootstrap_admin(user_id=user_id)
        else:  # argparse enforces one target; retain a guard for direct callers.
            raise AccountAdministrationError("an account id or email is required")
    except AccountAdministrationError as exc:
        print(json.dumps({"status": "failed", "detail": str(exc)}))
        return 1
    finally:
        store.close()
    print(
        json.dumps(
            {
                "status": "ok",
                "user_id": record.user_id,
                "email": record.email,
                "account_status": record.status,
                "groups": list(record.groups),
            },
            sort_keys=True,
        )
    )
    return 0


async def _import_owui_users(*, apply: bool = False) -> int:
    cfg = get_config()
    token = getpass("Open WebUI administrator API token: ").strip()
    if not token:
        print(json.dumps({"status": "failed", "detail": "an admin token is required"}))
        return 1
    url = cfg.env.owui_url.rstrip("/") + "/api/v1/users/all"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("users"), list):
            raise ValueError("OWUI users response has an unexpected shape")
        users = payload["users"]
        if len(users) > 100_000:
            raise ValueError("OWUI users response is too large")
        seen_emails: set[str] = set()
        seen_subjects: set[str] = set()
        validated: list[tuple[str, str, str]] = []
        for user in users:
            if not isinstance(user, dict):
                raise ValueError("OWUI users response contains an invalid row")
            subject = str(user.get("id") or "").strip()
            email = str(user.get("email") or "").strip()
            name = str(user.get("name") or "").strip()
            if not subject or not email or "@" not in email:
                raise ValueError("OWUI users response contains an incomplete account")
            if subject in seen_subjects or email.casefold() in seen_emails:
                raise ValueError("OWUI users response contains duplicate ids or emails")
            seen_subjects.add(subject)
            seen_emails.add(email.casefold())
            validated.append((subject, email, name))
    except (httpx.HTTPError, ValueError) as exc:
        print(json.dumps({"status": "failed", "detail": str(exc)}))
        return 1

    application = cfg.raw.get("application", {}) or {}
    store = ApplicationStore(application.get("sqlite_path", "/data/audrey_app.sqlite"))
    counts = {"pending": 0, "existing": 0, "ambiguous": 0, "conflict": 0}
    try:
        for subject, email, name in validated:
            outcome = await store.import_pending_owui_user(
                subject=subject,
                email=email,
                display_name=name,
                apply=apply,
            )
            counts[outcome] += 1
    finally:
        store.close()
    print(json.dumps({
        "status": "partial" if counts["ambiguous"] or counts["conflict"] else (
            "applied" if apply else "preview"
        ),
        "source_users": len(validated),
        "would_create" if not apply else "created_pending": counts["pending"],
        "already_in_audrey": counts["existing"],
        "ambiguous_audrey_email": counts["ambiguous"],
        "subject_conflicts": counts["conflict"],
    }, sort_keys=True))
    return 1 if counts["ambiguous"] or counts["conflict"] else 0


def _online_application_backup(source: Path, destination: Path) -> None:
    """Create a non-overwriting, mode-600 SQLite online backup."""
    if destination.resolve() == source.resolve():
        raise HistoryImportError("backup path must differ from the application database")
    if not destination.parent.is_dir():
        raise HistoryImportError("backup parent directory does not exist")
    created = False
    try:
        try:
            descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError as exc:
            raise HistoryImportError("backup destination already exists") from exc
        os.close(descriptor)
        created = True
        uri = f"file:{quote(str(source.resolve()), safe='/')}?mode=ro"
        with (
            closing(sqlite3.connect(uri, uri=True)) as source_db,
            closing(sqlite3.connect(destination)) as backup_db,
        ):
            source_db.backup(backup_db)
            integrity = backup_db.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise HistoryImportError("online SQLite backup failed integrity_check")
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise

def _import_chat_export(
    *,
    file: Path,
    user_id: str,
    email: str,
    apply: bool = False,
    allow_unbound: bool = False,
    backup_to: Path | None = None,
) -> int:
    """Preview without database writes; apply after an online backup."""
    try:
        if not user_id.startswith("usr_") or not email or "@" not in email:
            raise HistoryImportError("exact Audrey user id and account email are required")
        export = load_archive_export(file)
        bound = bool(export.audrey_user_id and export.account_email)
        if bool(export.audrey_user_id) != bool(export.account_email):
            raise HistoryImportError("export owner metadata is incomplete")
        if apply and not bound and not allow_unbound:
            raise HistoryImportError(
                "older export has no owner metadata; preview it first, then use "
                "--allow-unbound only if its owner was independently verified"
            )
        cfg = get_config()
        application = cfg.raw.get("application", {}) or {}
        path = Path(application.get("sqlite_path", "/data/audrey_app.sqlite"))
        if not path.is_file():
            raise HistoryImportError("Audrey application database does not exist")
        uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            summary = HistoryImportRepository(
                connection, threading.RLock()
            ).preview(user_id=user_id, email=email, export=export)
        if apply:
            if backup_to is None:
                raise HistoryImportError("--backup-to is required with --apply")
            _online_application_backup(path, backup_to)
            store = ApplicationStore(path)
            try:
                summary = store.history_imports.apply(
                    user_id=user_id, email=email, export=export
                )
            finally:
                store.close()
    except (HistoryImportError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"status": "failed", "detail": str(exc)}, sort_keys=True))
        return 1

    print(json.dumps({
        "status": "applied" if apply else "preview",
        "user_id": user_id,
        "owner_bound": bound,
        "backup_path": str(backup_to) if apply else None,
        **asdict(summary),
    }, sort_keys=True))
    return 0


def main() -> None:
    args = _parser().parse_args()
    if args.command == "grant-admin":
        raise SystemExit(asyncio.run(_grant_admin(args.user_id, email=args.email)))
    if args.command == "import-owui-users":
        raise SystemExit(asyncio.run(_import_owui_users(apply=args.apply)))
    if args.command == "import-chat-export":
        raise SystemExit(_import_chat_export(
            file=args.file,
            user_id=args.user_id,
            email=args.email,
            apply=args.apply,
            allow_unbound=args.allow_unbound,
            backup_to=args.backup_to,
        ))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
