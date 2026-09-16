"""Explicit operator bootstrap for Audrey's first native administrator."""

from __future__ import annotations

import argparse
import asyncio
import json

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


def main() -> None:
    args = _parser().parse_args()
    if args.command == "grant-admin":
        raise SystemExit(asyncio.run(_grant_admin(args.user_id, email=args.email)))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
