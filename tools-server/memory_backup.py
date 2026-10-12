"""Scheduled, private Qdrant memory snapshots and isolated restore verification.

Run inside custom-tools: ``python /app/memory_backup.py backup`` or
``python /app/memory_backup.py verify --latest``. This CLI cannot restore the
production collection. Each bundle also preserves the archive's purge receipts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)
_BUNDLE_NAME = re.compile(r"memory-\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{32}\Z")
_SNAPSHOT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.snapshot\Z")
_VERIFY_PREFIX = "audrey_memory_verify_"


class BackupError(RuntimeError):
    """Sanitized operator failure; never includes payloads or upstream bodies."""


def _private_file(path: Path):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    return os.fdopen(descriptor, "wb")


def _digest(path: Path, deadline: float | None = None) -> dict[str, Any]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise BackupError("backup file verification timed out")
            hasher.update(chunk)
            size += len(chunk)
    return {"filename": path.name, "bytes": size, "sha256": hasher.hexdigest()}


def _sqlite_backup(source: Path, destination: Path, deadline: float) -> None:
    def check_deadline(_status=0, _remaining=0, _total=0):
        if time.monotonic() >= deadline:
            raise BackupError("archive online backup timed out")

    check_deadline()
    with _private_file(destination):
        pass
    uri = f"file:{quote(str(source.resolve()), safe='/')}?mode=ro"
    with (
        contextlib.closing(sqlite3.connect(uri, uri=True, timeout=min(5, max(0.1, deadline - time.monotonic())))) as original,
        contextlib.closing(sqlite3.connect(destination)) as backup,
    ):
        original.backup(backup, pages=128, sleep=0.01, progress=check_deadline)
        backup.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        _check_archive(backup)
        check_deadline()


def _check_archive(connection: sqlite3.Connection) -> int:
    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise BackupError("archive backup failed integrity_check")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise BackupError("archive backup failed foreign_key_check")
    return int(connection.execute("SELECT COUNT(*) FROM archive_user_purges").fetchone()[0])


async def _thread(function, *args):
    # Cancellation must not delete/reuse a path while its worker still writes it.
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await worker
        raise


def _version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", value)
    if match is None:
        raise BackupError("Qdrant version is not a supported version string")
    return tuple(int(part) for part in match.groups())


class MemoryBackup:
    """One collection only, with a shared process/CLI lock and bounded retention."""

    def __init__(
        self, *, qdrant_url: str, collection: str, directory: Path,
        archive_path: Path, retain: int = 7, timeout_s: float = 120,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", collection) or retain < 1 or timeout_s <= 0:
            raise ValueError("collection, positive retention, and timeout are required")
        self.collection = collection
        self.directory = directory
        self.archive_path = archive_path
        self.retain = retain
        self.timeout_s = timeout_s
        self.http = http or httpx.AsyncClient(base_url=qdrant_url, timeout=timeout_s)
        self._owns_http = http is None
        self._collection_path = f"/collections/{quote(collection, safe='')}"

    async def close(self) -> None:
        if self._owns_http:
            await self.http.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = await self.http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise BackupError(f"Qdrant operation failed (HTTP {response.status_code})")
        try:
            body = response.json()
        except ValueError as error:
            raise BackupError("Qdrant returned invalid JSON") from error
        if not isinstance(body, dict):
            raise BackupError("Qdrant returned invalid JSON object")
        return body

    async def _count(self, path: str) -> int:
        result = (await self._request("POST", path + "/points/count", json={"exact": True})).get("result", {})
        count = result.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise BackupError("Qdrant returned invalid point count")
        return count

    def _bundles(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            (path for path in self.directory.iterdir()
             if _BUNDLE_NAME.fullmatch(path.name) and path.is_dir()
             and not path.is_symlink() and (path / "manifest.json").is_file()),
            reverse=True,
        )

    def latest(self) -> Path:
        bundles = self._bundles()
        if not bundles:
            raise BackupError("no completed memory backup exists")
        return bundles[0] / "manifest.json"

    def _lock(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise BackupError("backup directory must not be a symbolic link")
        self.directory.chmod(0o700)
        descriptor = os.open(
            self.directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600,
        )
        lock = os.fdopen(descriptor, "wb")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.close()
            raise BackupError("another memory backup or verification is running") from error
        # A process killed between download and publication can leave scratch
        # bundles. The shared lock proves no live writer can still own these.
        try:
            for pending in self.directory.iterdir():
                if (re.fullmatch(r"\.pending-[0-9a-f]{32}", pending.name)
                        and pending.is_dir() and not pending.is_symlink()):
                    shutil.rmtree(pending)
        except BaseException:
            lock.close()
            raise
        return lock

    async def _acquire_lock(self):
        worker = asyncio.create_task(asyncio.to_thread(self._lock))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Unlike an ordinary thread result this owns a flock. Cancellation
            # before the caller assigns it must still release the returned fd.
            try:
                lock = await worker
            except Exception as error:  # noqa: BLE001 — preserve the cancellation
                log.debug("cancelled backup lock acquisition failed (%s)", type(error).__name__)
            else:
                lock.close()
            raise

    def _finish(self, pending: Path, completed: Path, manifest: dict[str, Any]) -> None:
        with _private_file(pending / "manifest.json") as output:
            output.write((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        pending.rename(completed)
        for old in self._bundles()[self.retain:]:
            shutil.rmtree(old)

    async def backup(self) -> dict[str, Any]:
        lock = await self._acquire_lock()
        pending: Path | None = None
        snapshot_name = ""
        try:
            deadline = time.monotonic() + self.timeout_s
            async with asyncio.timeout(self.timeout_s):
                version = (await self._request("GET", "/")).get("version", "")
                _version(version)
                # An observation only: writes need not stop during snapshot creation.
                observed_count = await self._count(self._collection_path)
                stamp = dt.datetime.now(dt.UTC)
                name = f"memory-{stamp:%Y%m%dT%H%M%S.%fZ}-{uuid.uuid4().hex}"
                pending = self.directory / (".pending-" + uuid.uuid4().hex)
                await _thread(pending.mkdir, 0o700)
                result = (await self._request(
                    "POST", self._collection_path + "/snapshots", params={"wait": "true"},
                )).get("result", {})
                snapshot_name = result.get("name", "")
                if not isinstance(snapshot_name, str) or not _SNAPSHOT_NAME.fullmatch(snapshot_name):
                    snapshot_name = ""
                    raise BackupError("Qdrant returned invalid snapshot name")
                snapshot = pending / "memory.snapshot"
                with _private_file(snapshot) as output:
                    async with self.http.stream(
                        "GET", self._collection_path + "/snapshots/" + snapshot_name,
                    ) as response:
                        if response.status_code != 200:
                            raise BackupError(f"snapshot download failed (HTTP {response.status_code})")
                        async for chunk in response.aiter_bytes():
                            await _thread(output.write, chunk)
                    await _thread(output.flush)
                    await _thread(os.fsync, output.fileno())
                snapshot_info = await _thread(_digest, snapshot, deadline)
                if not snapshot_info["bytes"]:
                    raise BackupError("snapshot download was empty")
                upstream_checksum = result.get("checksum")
                if upstream_checksum and upstream_checksum != snapshot_info["sha256"]:
                    raise BackupError("snapshot download checksum mismatch")
                archive = pending / "chat_archive.db"
                await _thread(_sqlite_backup, self.archive_path, archive, deadline)
                manifest = {
                    "schema": 1, "created_at": stamp.isoformat(),
                    "collection": self.collection, "qdrant_version": version,
                    "live_point_count_observed_before_snapshot": observed_count,
                    "snapshot": snapshot_info, "archive": await _thread(_digest, archive, deadline),
                }
                completed = self.directory / name
                await _thread(self._finish, pending, completed, manifest)
                pending = None
                return {"status": "ok", "manifest": str(completed / "manifest.json"),
                        "snapshot_bytes": snapshot_info["bytes"],
                        "observed_memory_count": observed_count, "purge_receipts_backed_up": True}
        finally:
            if snapshot_name:
                try:
                    await asyncio.shield(self._request(
                        "DELETE", self._collection_path + "/snapshots/" + snapshot_name,
                        timeout=10,
                    ))
                except Exception as error:  # noqa: BLE001 — cleanup never hides primary result
                    log.warning("memory backup source snapshot cleanup failed (%s)", type(error).__name__)
            if pending is not None:
                await _thread(shutil.rmtree, pending)
            lock.close()

    def _load(self, manifest_path: Path, deadline: float) -> tuple[dict[str, Any], Path]:
        bundle = manifest_path.parent
        if (manifest_path.name != "manifest.json" or not _BUNDLE_NAME.fullmatch(bundle.name)
                or bundle.parent.resolve() != self.directory.resolve()
                or bundle.is_symlink() or manifest_path.is_symlink()):
            raise BackupError("manifest must belong to a completed local memory backup")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != 1 or manifest.get("collection") != self.collection:
            raise BackupError("manifest schema or source collection does not match")
        for field, filename in (("snapshot", "memory.snapshot"), ("archive", "chat_archive.db")):
            path = bundle / filename
            if path.is_symlink() or not path.is_file() or _digest(path, deadline) != manifest.get(field):
                raise BackupError("backup checksum or file metadata mismatch")
        uri = f"file:{quote(str((bundle / 'chat_archive.db').resolve()), safe='/')}?mode=ro&immutable=1"
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as archive:
            archive.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            _check_archive(archive)
        if time.monotonic() >= deadline:
            raise BackupError("backup file verification timed out")
        return manifest, bundle / "memory.snapshot"

    @staticmethod
    def _validate_point(point: dict[str, Any]) -> str:
        payload = point.get("payload")
        required = ("user", "key", "value", "tags", "created_at", "updated_at")
        if not isinstance(payload, dict) or any(not isinstance(payload.get(key), str) for key in required):
            raise BackupError("restored memory has invalid payload fields")
        user, key = payload["user"], payload["key"]
        user_tags = [tag[5:] for tag in payload["tags"].replace(",", " ").split() if tag.startswith("user:")]
        if not user or not key or not user_tags or any(tag != user for tag in user_tags):
            raise BackupError("restored memory has invalid owner scope")
        expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user}|{key}"))
        if str(point.get("id")) != expected_id:
            raise BackupError("restored memory id does not match its owner and key")
        return expected_id

    async def verify(self, manifest_path: Path | None = None) -> dict[str, Any]:
        lock = await self._acquire_lock()
        target = _VERIFY_PREFIX + uuid.uuid4().hex
        target_path = "/collections/" + target
        attempted = False
        cleaned = False
        try:
            deadline = time.monotonic() + self.timeout_s
            manifest_path = manifest_path or await _thread(self.latest)
            manifest, snapshot = await _thread(self._load, manifest_path, deadline)
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                source_version = _version(manifest["qdrant_version"])
                target_version = _version((await self._request("GET", "/")).get("version", ""))
                if (source_version[0] != target_version[0]
                        or target_version[1] not in (source_version[1], source_version[1] + 1)
                        or target_version < source_version):
                    raise BackupError("Qdrant snapshot version is incompatible with this server")
                # Unique internal name only; no caller can choose a restore target.
                if target == self.collection or not re.fullmatch(_VERIFY_PREFIX + r"[0-9a-f]{32}", target):
                    raise BackupError("unsafe verification target")
                existing = await self.http.get(target_path)
                if existing.status_code != 404:
                    raise BackupError("verification target is not confirmed absent")
                with snapshot.open("rb") as source:
                    attempted = True
                    await self._request(
                        "POST", target_path + "/snapshots/upload",
                        params={"priority": "snapshot", "wait": "true", "checksum": manifest["snapshot"]["sha256"]},
                        files={"snapshot": ("memory.snapshot", source, "application/octet-stream")},
                    )
                restored_count = await self._count(target_path)
                seen: set[str] = set()
                offsets: set[str] = set()
                offset = None
                while True:
                    body = {"limit": 256, "with_payload": True, "with_vector": False}
                    if offset is not None:
                        body["offset"] = offset
                    page = (await self._request("POST", target_path + "/points/scroll", json=body)).get("result", {})
                    points = page.get("points")
                    if not isinstance(points, list):
                        raise BackupError("restored memory scroll is invalid")
                    for point in points:
                        point_id = self._validate_point(point)
                        if point_id in seen:
                            raise BackupError("restored memory scroll repeated a point")
                        seen.add(point_id)
                    if len(seen) > restored_count:
                        raise BackupError("restored point count does not match payload inventory")
                    offset = page.get("next_page_offset")
                    if offset is None:
                        break
                    if str(offset) in offsets:
                        raise BackupError("restored memory scroll repeated a cursor")
                    offsets.add(str(offset))
                if len(seen) != restored_count:
                    raise BackupError("restored point count does not match payload inventory")
                result = {"status": "passed", "manifest": str(manifest_path),
                          "restored_count": restored_count, "owner_payloads_checked": len(seen),
                          "archive_integrity": "ok", "production_collection_unchanged": True}
        finally:
            try:
                if attempted:
                    response = await asyncio.shield(self.http.delete(target_path, timeout=10))
                    if response.status_code not in (200, 404):
                        raise BackupError("disposable verification collection cleanup failed")
                    cleaned = True
            finally:
                lock.close()
        result["disposable_collection_deleted"] = cleaned
        return result


class MemoryBackupScheduler:
    """Nonblocking startup backup, then periodic backups with short failure retry."""

    def __init__(self, backup: MemoryBackup, *, interval_s: float = 86400,
                 retry_s: float = 300, enabled: bool = True) -> None:
        self.backup = backup
        self.interval_s = interval_s
        self.retry_s = retry_s
        self.enabled = enabled
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._run(), name="memory-backup")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.backup.close()

    async def _run(self) -> None:
        while True:
            try:
                result = await self.backup.backup()
                log.info("memory backup completed (%d bytes)", result["snapshot_bytes"])
                delay = self.interval_s
            except Exception as error:  # noqa: BLE001 — backups must never stop the tool service
                log.warning("memory backup failed (%s); retrying in %.0fs", type(error).__name__, self.retry_s)
                delay = self.retry_s
            await asyncio.sleep(delay)


async def _cli(args) -> dict[str, Any]:
    from settings import settings

    backup = MemoryBackup(
        qdrant_url=settings.qdrant_url, collection=settings.memory_collection,
        directory=settings.memory_backup_dir, archive_path=settings.chat_archive_db_path,
        retain=settings.memory_backup_retain, timeout_s=settings.memory_backup_timeout_s,
    )
    try:
        if args.action == "backup":
            return await backup.backup()
        return await backup.verify(args.manifest)
    finally:
        await backup.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("backup", help="Create a private memory/purge-receipt backup bundle")
    verify = actions.add_parser("verify", help="Restore only a unique disposable collection and delete it")
    source = verify.add_mutually_exclusive_group()
    source.add_argument("--latest", action="store_true", help="Verify the latest completed bundle (default)")
    source.add_argument("--manifest", type=Path, help="Manifest inside the configured backup directory")
    args = parser.parse_args()
    try:
        result = asyncio.run(_cli(args))
    except Exception as error:  # noqa: BLE001 — sanitized, copyable operator output
        detail = str(error) if isinstance(error, BackupError) else type(error).__name__
        print(json.dumps({"schema": 1, "status": "failed", "error": detail}, sort_keys=True))
        raise SystemExit(1) from None
    print(json.dumps({"schema": 1, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
