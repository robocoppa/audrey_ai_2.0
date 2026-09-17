"""Explicit operator bootstrap for Audrey's first native administrator."""

from __future__ import annotations

import argparse
import asyncio
import json
from getpass import getpass

import httpx

from audrey.app_state import AccountAdministrationError, ApplicationStore
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


def main() -> None:
    args = _parser().parse_args()
    if args.command == "grant-admin":
        raise SystemExit(asyncio.run(_grant_admin(args.user_id, email=args.email)))
    if args.command == "import-owui-users":
        raise SystemExit(asyncio.run(_import_owui_users(apply=args.apply)))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
