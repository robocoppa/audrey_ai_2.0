"""Owned public watches, immutable check inputs and fenced local execution."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import ipaddress
import json
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Callable
from typing import Any

from audrey.app_state.transactions import write_transaction
from audrey.net.public_fetch import _public_ip, parse_public_url
from audrey.watch_schedule import next_occurrence, normalize_schedule

DEFAULT_LIMITS = {"max_operations": 8, "max_active_seconds": 300}
_UNSET = object()


class WatchError(ValueError):
    """Watch input or resource allowance is invalid."""


class WatchAccessError(WatchError):
    """The watch is unavailable for an active owner."""


class WatchConflictError(WatchError):
    """The definition, revision or execution authority changed."""


class WatchLimitError(WatchError):
    """A bounded storage or execution allowance has been reached."""


def _text(value: str, maximum: int, name: str, *, empty: bool = False, strip: bool = True) -> str:
    if not isinstance(value, str):
        raise WatchError(f"Invalid watch {name}.")
    value = value.strip() if strip else value
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise WatchError(f"Invalid watch {name}.") from None
    if (not value and not empty) or len(value) > maximum:
        raise WatchError(f"Invalid watch {name}.")
    return value


def _bytes_text(value: str, maximum: int, name: str, *, empty: bool = False) -> str:
    value = _text(value, maximum, name, empty=empty, strip=False)
    if len(value.encode("utf-8")) > maximum:
        raise WatchError(f"Watch {name} exceeds its storage allowance.")
    return value


def _integer(value: int, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise WatchError("Invalid watch limit or revision.")


def _limits(value: dict | None) -> dict:
    limits = DEFAULT_LIMITS.copy()
    if value is not None:
        if not isinstance(value, dict) or value.keys() - limits.keys():
            raise WatchError("Invalid watch execution limits.")
        limits.update(value)
    for key, item in limits.items():
        _integer(item, 1, DEFAULT_LIMITS[key])
    return limits


def normalize_watch_url(value: str) -> str:
    """Reject obvious local addresses without DNS or network access."""
    value = _text(value, 4096, "URL")
    try:
        url = parse_public_url(value)
    except ValueError:
        raise WatchError("Watches require a public HTTP(S) URL with a standard port.") from None
    if "?" in value or "#" in value:
        raise WatchError("Watch URLs cannot contain query parameters or fragments.")
    try:
        ipaddress.ip_address(url.host)
    except ValueError:
        if "." not in url.host.rstrip("."):
            raise WatchError("Watches require a public host.") from None
    else:
        if not _public_ip(url.host):
            raise WatchError("Watches require a public host.")
    return str(url)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stamp(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat()


def _schedule(value: dict) -> dict:
    try:
        return normalize_schedule(value)
    except ValueError as exc:
        raise WatchError(str(exc)) from None


class WatchesRepository:
    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn, self._lock = connection, lock
        self._clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC)

    def _transaction(self, work: Callable[[], Any]) -> Any:
        with self._lock, write_transaction(self._conn):
            return work()

    def _owner(self, user_id: str) -> None:
        if self._conn.execute(
            "SELECT 1 FROM app_users u WHERE u.user_id=? AND u.status='active' AND NOT EXISTS "
            "(SELECT 1 FROM account_deletion_requests d WHERE d.user_id=u.user_id)", (user_id,),
        ).fetchone() is None:
            raise WatchAccessError("Watch is unavailable.")

    def _load(self, watch_id: str, user_id: str | None = None) -> dict:
        row = self._conn.execute(
            "SELECT * FROM app_watches WHERE watch_id=? AND (? IS NULL OR user_id=?)",
            (watch_id, user_id, user_id),
        ).fetchone()
        if row is None:
            raise WatchAccessError("Watch is unavailable.")
        record = dict(row)
        self._owner(record["user_id"])
        record["schedule"] = json.loads(record["schedule_json"])
        record["limits"] = json.loads(record["limits_json"])
        return record

    @staticmethod
    def _public_run(record: dict, *, summary: bool = False) -> dict:
        return {
            "id": record["run_id"],
            "definition": json.loads(record["definition_json"]),
            **{key: record[key] for key in (
                "watch_id", "watch_revision", "status", "kind", "scheduled_at", "created_at",
                "updated_at", "completed_at", "attempts", "outcome", "error",
            )},
            "significant": None if record["significant"] is None else bool(record["significant"]),
            "report": "" if summary else record["report"],
            "diff": "" if summary else record["diff"],
        }

    def _public(self, record: dict, *, summary: bool = False) -> dict:
        latest = self._conn.execute(
            "SELECT * FROM app_watch_runs WHERE watch_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
            (record["watch_id"],),
        ).fetchone()
        active = self._conn.execute(
            "SELECT run_id FROM app_watch_runs WHERE watch_id=? AND status IN ('queued','running')",
            (record["watch_id"],),
        ).fetchone()
        return {
            "id": record["watch_id"],
            **{key: record[key] for key in (
                "title", "url", "criterion", "revision", "created_at", "updated_at", "next_run_at",
            )},
            "schedule": record["schedule"], "enabled": bool(record["enabled"]),
            "active_run_id": active["run_id"] if active else None,
            "latest_run": self._public_run(dict(latest), summary=summary) if latest else None,
        }

    def _run(self, run_id: str) -> dict:
        row = self._conn.execute("SELECT * FROM app_watch_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise WatchAccessError("Watch check is unavailable.")
        record = dict(row)
        self._owner(record["user_id"])
        return record

    def _internal(self, record: dict) -> dict:
        definition = json.loads(record["definition_json"])
        return {
            **self._public_run(record), **definition,
            "id": record["run_id"], "run_id": record["run_id"],
            "baseline": json.loads(record["baseline_json"]) if record["baseline_json"] else None,
            "limits": json.loads(record["limits_json"]),
            **{key: record[key] for key in (
                "user_id", "lease_token", "lease_expires_at", "fence", "operations", "active_seconds",
            )},
        }

    def _lease(self, run_id: str, lease_token: str, fence: int) -> dict:
        record = self._run(run_id)
        watch = self._load(record["watch_id"], record["user_id"])
        if (
            record["status"] != "running" or not lease_token or record["lease_token"] != lease_token
            or record["fence"] != fence or record["watch_revision"] != watch["revision"]
            or dt.datetime.fromisoformat(record["lease_expires_at"]) <= self._clock()
        ):
            raise WatchConflictError("Watch check authority changed or expired.")
        return record

    def _account(self, record: dict) -> None:
        if record["accounted_at"] is None:
            return
        start = dt.datetime.fromisoformat(record["accounted_at"])
        end = max(start, min(self._clock(), dt.datetime.fromisoformat(record["lease_expires_at"])))
        record["active_seconds"] += (end - start).total_seconds()
        record["accounted_at"] = _stamp(end)

    @staticmethod
    def _budget(record: dict) -> None:
        if record["active_seconds"] >= json.loads(record["limits_json"])["max_active_seconds"]:
            raise WatchLimitError("Watch active execution allowance exhausted.")

    @staticmethod
    def _clear_lease(record: dict) -> None:
        record.update(lease_token="", lease_expires_at=None, accounted_at=None)

    def _save_run(self, record: dict) -> None:
        keys = (
            "status", "attempts", "fence", "lease_token", "lease_expires_at", "accounted_at",
            "active_seconds", "operations", "available_at", "source_snapshot_id", "outcome",
            "significant", "report", "diff", "error", "completed_at", "limits_json",
        )
        record["updated_at"] = _stamp(self._clock())
        self._conn.execute(
            "UPDATE app_watch_runs SET status=?,attempts=?,fence=?,lease_token=?,lease_expires_at=?,"
            "accounted_at=?,active_seconds=?,operations=?,available_at=?,source_snapshot_id=?,outcome=?,"
            "significant=?,report=?,diff=?,error=?,completed_at=?,limits_json=?,updated_at=? WHERE run_id=?",
            (*[record[key] for key in (*keys, "updated_at")], record["run_id"]),
        )

    def _baseline(self, watch_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT s.record_json FROM app_watch_baselines b JOIN app_watch_snapshots s "
            "ON s.snapshot_id=b.snapshot_id WHERE b.watch_id=?", (watch_id,),
        ).fetchone()
        return json.loads(row["record_json"]) if row else None

    def _prune(self, watch_id: str) -> None:
        # Keep the open failure episode's exact run reachable until recovery.
        protected = self._conn.execute(
            "SELECT n.run_id FROM app_watch_notices n JOIN app_watches w ON w.failure_key=n.dedup_key "
            "WHERE w.watch_id=?", (watch_id,),
        ).fetchone()
        protected_id = protected["run_id"] if protected else ""
        active = bool(self._conn.execute(
            "SELECT 1 FROM app_watch_runs WHERE watch_id=? AND status IN ('queued','running')", (watch_id,),
        ).fetchone())
        self._conn.execute(
            "DELETE FROM app_watch_runs WHERE run_id IN (SELECT run_id FROM app_watch_runs "
            "WHERE watch_id=? AND run_id!=? AND status NOT IN ('queued','running') "
            "ORDER BY created_at DESC,rowid DESC LIMIT -1 OFFSET ?)",
            (watch_id, protected_id, 50 - int(active) - int(bool(protected_id))),
        )
        self._conn.execute(
            "DELETE FROM app_watch_snapshots WHERE watch_id=? AND snapshot_id NOT IN "
            "(SELECT snapshot_id FROM app_watch_baselines) AND snapshot_id NOT IN "
            "(SELECT source_snapshot_id FROM app_watch_runs WHERE source_snapshot_id IS NOT NULL)",
            (watch_id,),
        )

    def _enqueue(self, watch: dict, *, kind: str, scheduled_at: str) -> str:
        run_id, now = str(uuid.uuid4()), _stamp(self._clock())
        definition = {key: watch[key] for key in ("title", "url", "criterion", "schedule")}
        baseline = self._baseline(watch["watch_id"])
        self._conn.execute(
            "INSERT INTO app_watch_runs (run_id,watch_id,user_id,watch_revision,definition_json,"
            "limits_json,baseline_json,kind,scheduled_at,status,available_at,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,'queued',?,?,?)",
            (run_id, watch["watch_id"], watch["user_id"], watch["revision"], _json(definition),
             _json(watch["limits"]), _json(baseline) if baseline else None, kind, scheduled_at, now, now, now),
        )
        self._prune(watch["watch_id"])
        return run_id

    async def create(
        self, *, user_id: str, title: str, url: str, criterion: str, schedule: dict,
        request_id: str, enabled: bool = True, limits: dict | None = None,
        max_per_user: int = 20, max_global: int = 200,
    ) -> dict:
        title, url = _text(title, 100, "title"), normalize_watch_url(url)
        criterion, request_id = _text(criterion, 2000, "criterion"), _text(request_id, 200, "request identity")
        schedule, limits = _schedule(schedule), _limits(limits)
        if type(enabled) is not bool:
            raise WatchError("Invalid watch enabled state.")
        _integer(max_per_user, 1, 20)
        _integer(max_global, 1, 200)
        digest = hashlib.sha256(_json([title, url, criterion, schedule, enabled]).encode()).hexdigest()

        def work() -> dict:
            self._owner(user_id)
            existing = self._conn.execute(
                "SELECT watch_id,request_digest FROM app_watches WHERE user_id=? AND request_id=?",
                (user_id, request_id),
            ).fetchone()
            if existing:
                if existing["request_digest"] != digest:
                    raise WatchConflictError("Request identity already used for another watch.")
                return self._public(self._load(existing["watch_id"], user_id))
            counts = self._conn.execute(
                "SELECT COUNT(*) AS total,COALESCE(SUM(user_id=?),0) AS owned FROM app_watches", (user_id,),
            ).fetchone()
            if counts["owned"] >= max_per_user or counts["total"] >= max_global:
                raise WatchLimitError("Watch capacity reached.")
            watch_id, now = str(uuid.uuid4()), self._clock()
            self._conn.execute(
                "INSERT INTO app_watches (watch_id,user_id,request_id,request_digest,title,url,criterion,"
                "schedule_json,limits_json,enabled,next_run_at,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (watch_id, user_id, request_id, digest, title, url, criterion, _json(schedule), _json(limits),
                 int(enabled), _stamp(next_occurrence(schedule, now)) if enabled else None,
                 _stamp(now), _stamp(now)),
            )
            return self._public(self._load(watch_id, user_id))

        return await asyncio.to_thread(self._transaction, work)

    async def list(self, *, user_id: str, limit: int = 20) -> list[dict]:
        _integer(limit, 1, 20)

        def work() -> list[dict]:
            with self._lock:
                self._owner(user_id)
                return [self._public(self._load(row["watch_id"], user_id), summary=True) for row in
                        self._conn.execute(
                            "SELECT watch_id FROM app_watches WHERE user_id=? ORDER BY created_at DESC,watch_id "
                            "LIMIT ?", (user_id, limit),
                        ).fetchall()]
        return await asyncio.to_thread(work)

    async def get(self, *, user_id: str, watch_id: str) -> dict:
        def work() -> dict:
            with self._lock:
                return self._public(self._load(watch_id, user_id))
        return await asyncio.to_thread(work)

    async def update(
        self, *, user_id: str, watch_id: str, expected_revision: int,
        title: Any = _UNSET, url: Any = _UNSET, criterion: Any = _UNSET,
        schedule: Any = _UNSET, enabled: Any = _UNSET, limits: dict | None = None,
    ) -> dict:
        _integer(expected_revision, 1, 2**63 - 1)
        changes = {}
        for key, value, maximum in (("title", title, 100), ("criterion", criterion, 2000)):
            if value is not _UNSET:
                changes[key] = _text(value, maximum, key)
        if url is not _UNSET:
            changes["url"] = normalize_watch_url(url)
        if schedule is not _UNSET:
            changes["schedule"] = _schedule(schedule)
        if enabled is not _UNSET:
            if type(enabled) is not bool:
                raise WatchError("Invalid watch enabled state.")
            changes["enabled"] = enabled
        if limits is not None:
            changes["limits"] = _limits(limits)

        def work() -> dict:
            record = self._load(watch_id, user_id)
            if record["revision"] != expected_revision:
                raise WatchConflictError("Watch changed. Refresh before trying again.")
            old_url, old_schedule, old_enabled = record["url"], record["schedule"], record["enabled"]
            record.update(changes)
            if record["url"] != old_url:
                self._conn.execute("DELETE FROM app_watch_baselines WHERE watch_id=?", (watch_id,))
                record["failure_key"] = ""
            if not record["enabled"]:
                record["next_run_at"] = None
            elif not old_enabled or old_schedule != record["schedule"]:
                record["next_run_at"] = _stamp(next_occurrence(record["schedule"], self._clock()))
            self._conn.execute(
                "UPDATE app_watch_runs SET status='cancelled',fence=fence+1,lease_token='',"
                "lease_expires_at=NULL,accounted_at=NULL,completed_at=?,updated_at=? "
                "WHERE watch_id=? AND status IN ('queued','running')",
                (_stamp(self._clock()), _stamp(self._clock()), watch_id),
            )
            self._conn.execute(
                "UPDATE app_watches SET title=?,url=?,criterion=?,schedule_json=?,limits_json=?,enabled=?,"
                "revision=revision+1,next_run_at=?,failure_key=?,updated_at=? WHERE watch_id=?",
                (record["title"], record["url"], record["criterion"], _json(record["schedule"]),
                 _json(record["limits"]), int(record["enabled"]), record["next_run_at"], record["failure_key"],
                 _stamp(self._clock()), watch_id),
            )
            return self._public(self._load(watch_id, user_id))
        return await asyncio.to_thread(self._transaction, work)

    async def delete(self, *, user_id: str, watch_id: str, expected_revision: int) -> bool:
        _integer(expected_revision, 1, 2**63 - 1)

        def work() -> bool:
            record = self._load(watch_id, user_id)
            if record["revision"] != expected_revision:
                raise WatchConflictError("Watch changed. Refresh before trying again.")
            self._conn.execute("DELETE FROM app_watches WHERE watch_id=?", (watch_id,))
            return True
        return await asyncio.to_thread(self._transaction, work)

    async def run_now(self, *, user_id: str, watch_id: str, expected_revision: int) -> dict:
        _integer(expected_revision, 1, 2**63 - 1)

        def work() -> dict:
            watch = self._load(watch_id, user_id)
            if watch["revision"] != expected_revision:
                raise WatchConflictError("Watch changed. Refresh before trying again.")
            if not self._conn.execute(
                "SELECT 1 FROM app_watch_runs WHERE watch_id=? AND status IN ('queued','running')", (watch_id,),
            ).fetchone():
                # Microsecond identity also permits deliberate repeated runs with a frozen clock.
                stamp = _stamp(self._clock())
                while self._conn.execute(
                    "SELECT 1 FROM app_watch_runs WHERE watch_id=? AND watch_revision=? "
                    "AND scheduled_at=? AND kind='manual'", (watch_id, watch["revision"], stamp),
                ).fetchone():
                    stamp = _stamp(dt.datetime.fromisoformat(stamp) + dt.timedelta(microseconds=1))
                self._enqueue(watch, kind="manual", scheduled_at=stamp)
            return self._public(watch)
        return await asyncio.to_thread(self._transaction, work)

    def _schedule_due(self) -> None:
        now = self._clock()
        for row in self._conn.execute(
            "SELECT w.watch_id FROM app_watches w JOIN app_users u ON u.user_id=w.user_id "
            "WHERE w.enabled=1 AND w.next_run_at<=? AND u.status='active' AND NOT EXISTS "
            "(SELECT 1 FROM account_deletion_requests d WHERE d.user_id=w.user_id) AND NOT EXISTS "
            "(SELECT 1 FROM app_watch_runs r WHERE r.watch_id=w.watch_id AND r.status IN ('queued','running')) "
            "ORDER BY w.next_run_at,w.watch_id LIMIT 200", (_stamp(now),),
        ).fetchall():
            watch = self._load(row["watch_id"])
            due = dt.datetime.fromisoformat(watch["next_run_at"])
            # Normal polling jitter is scheduled; a missed following occurrence is catch-up.
            following = next_occurrence(watch["schedule"], due, anchor=due)
            self._enqueue(watch, kind="catchup" if following <= now else "scheduled", scheduled_at=_stamp(due))
            self._conn.execute(
                "UPDATE app_watches SET next_run_at=? WHERE watch_id=?",
                (_stamp(next_occurrence(watch["schedule"], now, anchor=due)), watch["watch_id"]),
            )

    def _notice(self, record: dict, *, kind: str, body: str, dedup_key: str) -> None:
        title = json.loads(record["definition_json"])["title"]
        self._conn.execute(
            "INSERT OR IGNORE INTO app_watch_notices (notice_id,user_id,watch_id,run_id,dedup_key,kind,"
            "title,body,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), record["user_id"], record["watch_id"], record["run_id"], dedup_key,
             kind, title, body, _stamp(self._clock())),
        )

    def _final_failure(self, record: dict, error: str) -> None:
        record.update(status="failed", error=error, completed_at=_stamp(self._clock()))
        self._clear_lease(record)
        self._save_run(record)
        watch = self._load(record["watch_id"])
        if not watch["failure_key"]:
            key = "failure:" + record["run_id"]
            self._notice(record, kind="failure", body=error, dedup_key=key)
            self._conn.execute("UPDATE app_watches SET failure_key=? WHERE watch_id=?", (key, watch["watch_id"]))

    async def claim(
        self, *, lease_seconds: int = 60, max_attempts: int = 3, limits: dict | None = None,
    ) -> dict | None:
        _integer(lease_seconds, 1, 3600)
        _integer(max_attempts, 1, 3)
        current_limits = _limits(limits) if limits is not None else None

        def work() -> dict | None:
            self._schedule_due()
            now = self._clock()
            rows = self._conn.execute(
                "SELECT r.* FROM app_watch_runs r JOIN app_users u ON u.user_id=r.user_id "
                "WHERE u.status='active' AND NOT EXISTS (SELECT 1 FROM account_deletion_requests d "
                "WHERE d.user_id=r.user_id) AND ((r.status='queued' AND r.available_at<=?) OR "
                "(r.status='running' AND r.lease_expires_at<=?)) AND NOT EXISTS "
                "(SELECT 1 FROM app_watch_runs other WHERE other.user_id=r.user_id "
                "AND other.status='running' AND other.run_id!=r.run_id) AND NOT EXISTS "
                "(SELECT 1 FROM app_agent_tasks a WHERE a.user_id=r.user_id AND a.status='running') "
                "ORDER BY r.created_at,r.run_id LIMIT 200", (_stamp(now), _stamp(now)),
            ).fetchall()
            for row in rows:
                record = dict(row)
                self._account(record)
                if current_limits:
                    saved = json.loads(record["limits_json"])
                    record["limits_json"] = _json({key: min(saved[key], current_limits[key]) for key in saved})
                effective = json.loads(record["limits_json"])
                if (record["attempts"] >= max_attempts or record["operations"] >= effective["max_operations"]
                        or record["active_seconds"] >= effective["max_active_seconds"]):
                    self._final_failure(record, "Watch execution allowance exhausted.")
                    continue
                record.update(
                    status="running", attempts=record["attempts"] + 1, fence=record["fence"] + 1,
                    lease_token=secrets.token_urlsafe(32),
                    lease_expires_at=_stamp(now + dt.timedelta(seconds=lease_seconds)),
                    accounted_at=_stamp(now), error="",
                )
                self._save_run(record)
                return self._internal(record)
            return None
        return await asyncio.to_thread(self._transaction, work)

    claim_next = claim

    async def validate_lease(self, *, run_id: str, lease_token: str, fence: int) -> bool:
        def work() -> bool:
            with self._lock:
                record = self._lease(run_id, lease_token, fence)
                self._account(record)
                self._budget(record)
                return True
        return await asyncio.to_thread(work)

    authorize = validate_lease

    async def heartbeat(
        self, *, run_id: str, lease_token: str, fence: int, lease_seconds: int = 60,
    ) -> dict:
        _integer(lease_seconds, 1, 3600)

        def work() -> dict:
            record = self._lease(run_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            record["lease_expires_at"] = _stamp(max(
                dt.datetime.fromisoformat(record["lease_expires_at"]),
                self._clock() + dt.timedelta(seconds=lease_seconds),
            ))
            self._save_run(record)
            return self._internal(record)
        return await asyncio.to_thread(self._transaction, work)

    async def reserve_operation(
        self, *, run_id: str, lease_token: str, fence: int, kind: str,
    ) -> str:
        if kind not in {"fetch", "model"}:
            raise WatchError("Invalid watch operation.")

        def work() -> str:
            record = self._lease(run_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            if record["operations"] >= json.loads(record["limits_json"])["max_operations"]:
                raise WatchLimitError("Watch operation allowance exhausted.")
            operation_id = str(uuid.uuid4())
            self._conn.execute(
                "INSERT INTO app_watch_operations (operation_id,run_id,fence,kind,status,created_at) "
                "VALUES (?,?,?,?,'reserved',?)", (operation_id, run_id, fence, kind, _stamp(self._clock())),
            )
            record["operations"] += 1
            self._save_run(record)
            return operation_id
        return await asyncio.to_thread(self._transaction, work)

    async def complete(
        self, *, run_id: str, lease_token: str, fence: int, outcome: str, significant: bool | None,
        report: str, diff: str, text: str, sha256: str, extraction_version: str, final_url: str,
    ) -> dict:
        text = _bytes_text(text, 64000, "source text")
        report, diff = _bytes_text(report, 16000, "report", empty=True), _bytes_text(diff, 16000, "diff", empty=True)
        extraction_version = _text(extraction_version, 100, "extraction version")
        final_url = normalize_watch_url(final_url)
        if sha256 != hashlib.sha256(text.encode()).hexdigest():
            raise WatchError("Watch source digest does not match its text.")
        if outcome not in {"baseline", "unchanged", "changed"} or type(significant) not in {bool, type(None)}:
            raise WatchError("Invalid watch comparison result.")
        if outcome == "changed":
            if type(significant) is not bool or not report.strip() or not diff.strip():
                raise WatchError("Changed checks need a judgment and a comparison report.")
        elif significant is not None or diff:
            raise WatchError("Baseline and unchanged checks have no significance judgment or diff.")

        def work() -> dict:
            record = self._lease(run_id, lease_token, fence)
            self._account(record)
            self._budget(record)
            baseline = json.loads(record["baseline_json"]) if record["baseline_json"] else None
            compatible = baseline is not None and baseline["final_url"] == final_url and baseline[
                "extraction_version"
            ] == extraction_version
            expected = "baseline" if not compatible else "unchanged" if baseline["sha256"] == sha256 else "changed"
            if expected != outcome:
                raise WatchConflictError("Watch comparison does not match its saved baseline.")
            reserved = {row["kind"] for row in self._conn.execute(
                "SELECT kind FROM app_watch_operations WHERE run_id=? AND fence=? AND status='reserved'",
                (run_id, fence),
            ).fetchall()}
            if "fetch" not in reserved or (outcome == "changed" and "model" not in reserved):
                raise WatchConflictError("Reserve watch operations before committing their result.")
            now = _stamp(self._clock())
            snapshot = {"text": text, "sha256": sha256, "final_url": final_url,
                        "extraction_version": extraction_version, "run_id": run_id}
            self._conn.execute(
                "INSERT INTO app_watch_snapshots (snapshot_id,watch_id,record_json,created_at) VALUES (?,?,?,?)",
                (run_id, record["watch_id"], _json(snapshot), now),
            )
            self._conn.execute(
                "INSERT INTO app_watch_baselines (watch_id,snapshot_id) VALUES (?,?) "
                "ON CONFLICT(watch_id) DO UPDATE SET snapshot_id=excluded.snapshot_id", (record["watch_id"], run_id),
            )
            record.update(status="completed", outcome=outcome,
                          significant=None if significant is None else int(significant),
                          report=report, diff=diff, error="", completed_at=now, source_snapshot_id=run_id)
            self._clear_lease(record)
            self._save_run(record)
            self._conn.execute(
                "UPDATE app_watch_operations SET status='committed',committed_at=? WHERE run_id=? AND fence=?",
                (now, run_id, fence),
            )
            self._conn.execute("UPDATE app_watches SET failure_key='' WHERE watch_id=?", (record["watch_id"],))
            if significant:
                self._notice(record, kind="change", body=report, dedup_key="change:" + run_id)
            self._prune(record["watch_id"])
            return self._public_run(record)
        return await asyncio.to_thread(self._transaction, work)

    async def fail(
        self, *, run_id: str, lease_token: str, fence: int, error: str, retry: bool = False,
    ) -> dict:
        error = _text(error, 2000, "error")
        if type(retry) is not bool:
            raise WatchError("Invalid watch retry.")

        def work() -> dict:
            record = self._lease(run_id, lease_token, fence)
            self._account(record)
            limits = json.loads(record["limits_json"])
            if (retry and record["attempts"] < 3 and record["operations"] < limits["max_operations"]
                    and record["active_seconds"] < limits["max_active_seconds"]):
                record.update(status="queued", error=error, available_at=_stamp(
                    self._clock() + dt.timedelta(seconds=min(60 * 2 ** (record["attempts"] - 1), 900)),
                ))
                self._clear_lease(record)
                self._save_run(record)
            else:
                self._final_failure(record, error)
            return self._public_run(record)
        return await asyncio.to_thread(self._transaction, work)

    async def list_runs(self, *, user_id: str, watch_id: str, limit: int = 50) -> list[dict]:
        _integer(limit, 1, 50)

        def work() -> list[dict]:
            with self._lock:
                self._load(watch_id, user_id)
                return [self._public_run(dict(row)) for row in self._conn.execute(
                    "SELECT * FROM app_watch_runs WHERE watch_id=? ORDER BY created_at DESC,rowid DESC LIMIT ?",
                    (watch_id, limit),
                ).fetchall()]
        return await asyncio.to_thread(work)

    async def get_run(self, *, user_id: str, watch_id: str, run_id: str) -> dict:
        def work() -> dict:
            with self._lock:
                self._load(watch_id, user_id)
                record = self._run(run_id)
                if record["watch_id"] != watch_id or record["user_id"] != user_id:
                    raise WatchAccessError("Watch check is unavailable.")
                return self._public_run(record)
        return await asyncio.to_thread(work)

    async def list_notices(self, *, user_id: str, limit: int = 50) -> list[dict]:
        _integer(limit, 1, 50)

        def work() -> list[dict]:
            with self._lock:
                self._owner(user_id)
                return [{"id": row["notice_id"], **{key: row[key] for key in (
                    "watch_id", "run_id", "kind", "title", "body", "created_at", "read_at",
                )}} for row in self._conn.execute(
                    "SELECT * FROM app_watch_notices WHERE user_id=? ORDER BY created_at DESC,notice_id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()]
        return await asyncio.to_thread(work)

    async def notification_feed(self, *, user_id: str, limit: int = 50) -> dict:
        """Read bodies and their count together, without a later hydration await."""
        _integer(limit, 1, 50)

        def work() -> dict:
            with self._lock:
                self._owner(user_id)
                # One SQL statement also supplies a consistent snapshot when a
                # second ApplicationStore connection commits a deletion.
                rows = self._conn.execute(
                    "SELECT n.*, (SELECT COUNT(*) FROM app_watch_notices c WHERE c.user_id=? "
                    "AND c.read_at IS NULL) AS unread_count FROM (SELECT 1) "
                    "LEFT JOIN (SELECT * FROM app_watch_notices WHERE user_id=? "
                    "ORDER BY created_at DESC,notice_id DESC LIMIT ?) n ON 1=1 "
                    "ORDER BY n.created_at DESC,n.notice_id DESC", (user_id, user_id, limit),
                ).fetchall()
                return {
                    "items": [{"id": row["notice_id"], **{key: row[key] for key in (
                        "watch_id", "run_id", "kind", "title", "body", "created_at", "read_at",
                    )}} for row in rows if row["notice_id"] is not None],
                    "unread_count": int(rows[0]["unread_count"]),
                }
        return await asyncio.to_thread(work)

    def fence_response(
        self, *, user_id: str, watches: list[dict] | tuple = (), runs: list[dict] | tuple = (),
        notices: list[dict] | tuple = (),
    ) -> None:
        """Recheck canonical response proofs synchronously after the last await.

        Worker-thread reads can finish before their awaiting route resumes. This
        final short check rejects a body erased or revised during that window.
        It performs no hydration, network work, or asynchronous handoff.
        """
        if len(watches) > 20 or len(runs) > 50 or len(notices) > 50:
            raise WatchError("Watch response exceeds its allowance.")

        def check_run(payload: dict, watch_id: str | None = None) -> None:
            row = self._conn.execute(
                "SELECT r.watch_id,r.updated_at FROM app_watch_runs r JOIN app_watches w "
                "ON w.watch_id=r.watch_id WHERE r.run_id=? AND r.user_id=? AND w.user_id=?",
                (payload["id"], user_id, user_id),
            ).fetchone()
            if row is None or (watch_id is not None and row["watch_id"] != watch_id) or (
                "watch_id" in payload and row["watch_id"] != payload["watch_id"]
            ):
                raise WatchAccessError("Watch check is unavailable.")
            if "updated_at" in payload and row["updated_at"] != payload["updated_at"]:
                raise WatchConflictError("Watch check changed. Refresh before trying again.")

        with self._lock:
            self._owner(user_id)
            for payload in watches:
                row = self._conn.execute(
                    "SELECT revision FROM app_watches WHERE watch_id=? AND user_id=?",
                    (payload["id"], user_id),
                ).fetchone()
                if row is None:
                    raise WatchAccessError("Watch is unavailable.")
                if row["revision"] != payload["revision"]:
                    raise WatchConflictError("Watch changed. Refresh before trying again.")
                if payload.get("latest_run") is not None:
                    check_run(payload["latest_run"], payload["id"])
            for payload in runs:
                check_run(payload)
            for payload in notices:
                row = self._conn.execute(
                    "SELECT n.watch_id,n.run_id,n.title,n.body FROM app_watch_notices n "
                    "JOIN app_watches w ON w.watch_id=n.watch_id JOIN app_watch_runs r ON r.run_id=n.run_id "
                    "WHERE n.notice_id=? AND n.user_id=? AND w.user_id=? AND r.user_id=?",
                    (payload["id"], user_id, user_id, user_id),
                ).fetchone()
                if row is None:
                    raise WatchAccessError("Watch notice is unavailable.")
                if any(row[key] != payload[key] for key in ("watch_id", "run_id", "title", "body")):
                    raise WatchConflictError("Watch notice changed. Refresh before trying again.")

    async def unread_count(self, *, user_id: str) -> int:
        def work() -> int:
            with self._lock:
                self._owner(user_id)
                return self._conn.execute(
                    "SELECT COUNT(*) FROM app_watch_notices WHERE user_id=? AND read_at IS NULL", (user_id,),
                ).fetchone()[0]
        return await asyncio.to_thread(work)

    async def mark_notice_read(self, *, user_id: str, notice_id: str) -> dict:
        def work() -> dict:
            self._owner(user_id)
            row = self._conn.execute(
                "SELECT notice_id FROM app_watch_notices WHERE notice_id=? AND user_id=?", (notice_id, user_id),
            ).fetchone()
            if row is None:
                raise WatchAccessError("Watch notice is unavailable.")
            self._conn.execute(
                "UPDATE app_watch_notices SET read_at=COALESCE(read_at,?) WHERE notice_id=? AND user_id=?",
                (_stamp(self._clock()), notice_id, user_id),
            )
            saved = self._conn.execute(
                "SELECT * FROM app_watch_notices WHERE notice_id=? AND user_id=?", (notice_id, user_id),
            ).fetchone()
            return {"id": saved["notice_id"], **{key: saved[key] for key in (
                "watch_id", "run_id", "kind", "title", "body", "created_at", "read_at",
            )}}
        return await asyncio.to_thread(self._transaction, work)
