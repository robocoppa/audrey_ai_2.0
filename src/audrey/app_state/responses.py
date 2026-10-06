"""Bounded, owner-scoped retention of completed Responses and replay history."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

RESPONSE_TTL_SECONDS = 30 * 24 * 60 * 60
RESPONSE_MAX_RECORD_BYTES = 2 * 1024 * 1024
RESPONSE_MAX_OWNER_RECORDS = 100
RESPONSE_MAX_OWNER_BYTES = 30 * 1024 * 1024
RESPONSE_MAX_CHAIN_DEPTH = 32
_SQLITE_MAX_INTEGER = (1 << 63) - 1


class InvalidResponseStateError(ValueError):
    """A response record is malformed or cannot be retained safely."""


class ResponseStorageLimitError(RuntimeError):
    """Retaining the response would exceed a record, owner, or chain limit."""


class ResponseParentNotFoundError(LookupError):
    """The requested parent is unavailable to this response owner."""


class ResponsesRepository:
    """Retain immutable completed responses behind short SQLite transactions.

    Replay is the route's canonical full history, including this response's
    output items. It deliberately does not contain an inherited instructions
    field: a chained request supplies its own instructions separately.
    """

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = connection
        self._lock = lock

    async def save(
        self,
        owner_id: str,
        response: dict[str, Any],
        replay: list[dict[str, Any]],
        previous_response_id: str | None = None,
        now: int | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._save_sync, owner_id, response, replay, previous_response_id, now,
        )

    def _save_sync(
        self,
        owner_id: str,
        response: dict[str, Any],
        replay: list[dict[str, Any]],
        previous_response_id: str | None,
        now: int | None,
    ) -> None:
        owner_id = _identifier(owner_id, "owner id")
        stamp = _timestamp(now)
        if not isinstance(response, dict) or response.get("status") != "completed":
            raise InvalidResponseStateError("only completed response objects may be retained")
        response_id = _identifier(response.get("id"), "response id")
        if not isinstance(replay, list) or any(not isinstance(item, dict) for item in replay):
            raise InvalidResponseStateError("response replay must be an array of objects")
        if previous_response_id is not None:
            previous_response_id = _identifier(previous_response_id, "previous response id")
        response_json = _encode_json(response)
        replay_json = _encode_json(replay)
        record_bytes = len(response_json.encode("utf-8")) + len(replay_json.encode("utf-8"))
        if record_bytes > RESPONSE_MAX_RECORD_BYTES:
            raise ResponseStorageLimitError("stored response exceeds the 2 MiB record limit")
        with self._transaction(stamp):
            owner = self._conn.execute(
                "SELECT 1 FROM app_users WHERE user_id = ?", (owner_id,),
            ).fetchone()
            if owner is None:
                raise InvalidResponseStateError("response owner does not exist")
            if previous_response_id is not None:
                parent = self._row_locked(owner_id, previous_response_id)
                if parent is None:
                    raise ResponseParentNotFoundError("previous response not found")
                parent_record = _record_from_row(parent)
                if parent_record["response"].get("status") != "completed":
                    raise ResponseParentNotFoundError("previous response not found")
                if self._chain_depth_locked(owner_id, previous_response_id) >= RESPONSE_MAX_CHAIN_DEPTH:
                    raise ResponseStorageLimitError("stored response chain exceeds 32 responses")
            duplicate = self._conn.execute(
                "SELECT 1 FROM app_responses WHERE response_id = ?", (response_id,),
            ).fetchone()
            if duplicate is not None:
                raise InvalidResponseStateError("a response with this id is already retained")
            quota = self._conn.execute(
                "SELECT COUNT(*) AS records, COALESCE(SUM("
                "length(CAST(response_json AS BLOB)) + length(CAST(replay_json AS BLOB))"
                "), 0) AS bytes FROM app_responses WHERE owner_id = ?", (owner_id,),
            ).fetchone()
            assert quota is not None
            if int(quota["records"]) >= RESPONSE_MAX_OWNER_RECORDS:
                raise ResponseStorageLimitError("owner has reached the 100 stored-response limit")
            if int(quota["bytes"]) + record_bytes > RESPONSE_MAX_OWNER_BYTES:
                raise ResponseStorageLimitError("owner has reached the 30 MiB stored-response limit")
            self._conn.execute(
                "INSERT INTO app_responses (response_id, owner_id, previous_response_id, "
                "created_at, expires_at, response_json, replay_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    response_id, owner_id, previous_response_id, stamp,
                    stamp + RESPONSE_TTL_SECONDS, response_json, replay_json,
                ),
            )

    async def get(
        self, owner_id: str, response_id: str, now: int | None = None,
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_sync, owner_id, response_id, now)

    def _get_sync(
        self, owner_id: str, response_id: str, now: int | None,
    ) -> dict[str, Any] | None:
        owner_id = _identifier(owner_id, "owner id")
        response_id = _identifier(response_id, "response id")
        with self._transaction(_timestamp(now)):
            row = self._row_locked(owner_id, response_id)
            if row is None:
                return None
            record = _record_from_row(row)
            record["depth"] = self._chain_depth_locked(owner_id, response_id)
            return record

    async def delete(
        self, owner_id: str, response_id: str, now: int | None = None,
    ) -> bool:
        return await asyncio.to_thread(self._delete_sync, owner_id, response_id, now)

    def _delete_sync(self, owner_id: str, response_id: str, now: int | None) -> bool:
        owner_id = _identifier(owner_id, "owner id")
        response_id = _identifier(response_id, "response id")
        with self._transaction(_timestamp(now)):
            cursor = self._conn.execute(
                "DELETE FROM app_responses WHERE owner_id = ? AND response_id = ?",
                (owner_id, response_id),
            )
            return cursor.rowcount == 1

    async def prune_expired(self, now: int | None = None) -> int:
        return await asyncio.to_thread(self._prune_expired_sync, now)

    def _prune_expired_sync(self, now: int | None = None) -> int:
        stamp = _timestamp(now)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                deleted = self._prune_locked(stamp)
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return deleted

    @contextmanager
    def _transaction(self, now: int) -> Iterator[None]:
        with self._lock:
            # Keep expiry cleanup committed when this operation later fails
            # a parent or quota check. Prune again after acquiring the write
            # transaction to include expired rows from other connections.
            self._prune_expired_sync(now)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._prune_locked(now)
                yield
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    def _prune_locked(self, now: int) -> int:
        before = int(self._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0])
        self._conn.execute("DELETE FROM app_responses WHERE expires_at <= ?", (now,))
        after = int(self._conn.execute("SELECT COUNT(*) FROM app_responses").fetchone()[0])
        return before - after

    def _row_locked(self, owner_id: str, response_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT response_id, previous_response_id, created_at, expires_at, "
            "response_json, replay_json FROM app_responses "
            "WHERE owner_id = ? AND response_id = ?", (owner_id, response_id),
        ).fetchone()

    def _chain_depth_locked(self, owner_id: str, response_id: str) -> int:
        seen: set[str] = set()
        current: str | None = response_id
        while current is not None:
            if current in seen or len(seen) >= RESPONSE_MAX_CHAIN_DEPTH:
                raise InvalidResponseStateError("retained response chain is invalid")
            seen.add(current)
            row = self._row_locked(owner_id, current)
            if row is None:
                raise InvalidResponseStateError("retained response chain has a missing parent")
            current = row["previous_response_id"]
        return len(seen)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 200:
        raise InvalidResponseStateError(f"{label} must be a nonempty identifier of at most 200 characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise InvalidResponseStateError(f"{label} contains invalid Unicode") from exc
    return value


def _timestamp(now: int | None) -> int:
    value = int(time.time()) if now is None else now
    if type(value) is not int or not 0 <= value <= _SQLITE_MAX_INTEGER - RESPONSE_TTL_SECONDS:
        raise InvalidResponseStateError("response timestamp must be a nonnegative integer")
    return value


def _validate_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidResponseStateError("response JSON must not contain non-finite numbers")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidResponseStateError("response JSON object keys must be strings")
            _validate_json(item)
        return
    raise InvalidResponseStateError("response record contains a non-JSON value")


def _encode_json(value: Any) -> str:
    try:
        _validate_json(value)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        encoded.encode("utf-8")
        return encoded
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidResponseStateError("response record is not valid UTF-8 JSON") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResponseStateError("retained response JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InvalidResponseStateError(f"retained response JSON contains {value}")


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        response = json.loads(
            row["response_json"], object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
        replay = json.loads(
            row["replay_json"], object_pairs_hook=_unique_object, parse_constant=_reject_constant,
        )
        if not isinstance(response, dict) or response.get("status") != "completed":
            raise InvalidResponseStateError("retained response object is invalid")
        if response.get("id") != row["response_id"]:
            raise InvalidResponseStateError("retained response identity is invalid")
        if not isinstance(replay, list) or any(not isinstance(item, dict) for item in replay):
            raise InvalidResponseStateError("retained response replay is invalid")
        _encode_json(response)
        _encode_json(replay)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidResponseStateError("retained response record is malformed") from exc
    return {
        "response": response,
        "replay": replay,
        "previous_response_id": row["previous_response_id"],
        "created_at": int(row["created_at"]),
        "expires_at": int(row["expires_at"]),
    }


__all__ = [
    "InvalidResponseStateError",
    "ResponseParentNotFoundError",
    "ResponseStorageLimitError",
    "ResponsesRepository",
    "RESPONSE_TTL_SECONDS",
    "RESPONSE_MAX_RECORD_BYTES",
    "RESPONSE_MAX_OWNER_RECORDS",
    "RESPONSE_MAX_OWNER_BYTES",
    "RESPONSE_MAX_CHAIN_DEPTH",
]
