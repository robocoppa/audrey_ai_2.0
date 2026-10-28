"""Write boundaries for callers holding the application store's shared lock."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Commit one write or roll it back without leaving the connection busy."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise
