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
        help="Activate one exact Audrey user id and grant its admins group.",
    )
    grant.add_argument("user_id", help="Canonical usr_... id shown by /api/me")
    return parser


async def _grant_admin(user_id: str) -> int:
    cfg = get_config()
    application = cfg.raw.get("application", {}) or {}
    store = ApplicationStore(
        application.get("sqlite_path", "/data/audrey_app.sqlite")
    )
    try:
        record = await store.bootstrap_admin(user_id=user_id)
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
        raise SystemExit(asyncio.run(_grant_admin(args.user_id)))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
