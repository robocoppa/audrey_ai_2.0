"""Process-local privacy gate for durable account-purge handoff.

The uploads database owns the durable receipt.  This small mirror closes the
interval between committing that receipt and custom-tools acknowledging its
own memory/chat cutoff: model tools and public exports must not reach the
sidecar during that interval.  The coordinator repopulates the mirror from
SQLite before the application becomes ready, so a restart does not reopen it.
"""

from __future__ import annotations

_REMOTE_READ_BLOCKS: dict[str, set[str]] = {}


def block_remote_personal_reads(*, user: str, purge_id: str) -> None:
    if user and purge_id:
        _REMOTE_READ_BLOCKS.setdefault(user, set()).add(purge_id)


def unblock_remote_personal_reads(*, user: str, purge_id: str) -> None:
    purges = _REMOTE_READ_BLOCKS.get(user)
    if purges is None:
        return
    purges.discard(purge_id)
    if not purges:
        _REMOTE_READ_BLOCKS.pop(user, None)


def remote_personal_reads_blocked(user: str) -> bool:
    return bool(user and _REMOTE_READ_BLOCKS.get(user))


__all__ = [
    "block_remote_personal_reads",
    "remote_personal_reads_blocked",
    "unblock_remote_personal_reads",
]
