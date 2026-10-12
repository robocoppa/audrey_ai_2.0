"""Memory snapshots preserve private source data; verification cannot touch production."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools-server"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from memory_backup import (  # noqa: E402
    BackupError,
    MemoryBackup,
    MemoryBackupScheduler,
    _sqlite_backup,
)
from settings import Settings  # noqa: E402


def _point(user="alice@example.com", key="workstation"):
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user}|{key}")),
        "payload": {"user": user, "key": key, "value": "private memory text",
                    "tags": f"user:{user},topic:hardware",
                    "created_at": "2026-10-10T01:00:00+00:00",
                    "updated_at": "2026-10-10T02:00:00+00:00"},
    }


class Qdrant:
    def __init__(self):
        self.version = "1.18.3"
        self.collections = {"kb_memory": [_point(), _point("bob@example.com", "preference")]}
        self.snapshots = {}
        self.requests = []
        self.fail_create = False
        self.fail_download = False
        self.bad_checksum = False
        self.fail_upload = False
        self.fail_delete = False
        self.invalid_owner = False
        self.incorrect_count = False
        self.colliding_target = False
        self.empty_second_page = False

    def __call__(self, request):
        self.requests.append((request.method, request.url.path))
        if request.url.path == "/":
            return httpx.Response(200, json={"version": self.version})
        bits = request.url.path.strip("/").split("/")
        assert bits[0] == "collections"
        collection = bits[1]
        if bits[2:] == ["snapshots"] and request.method == "POST":
            if self.fail_create:
                return httpx.Response(503, json={"error": "private memory text"})
            data = json.dumps(self.collections[collection]).encode()
            name = "memory-" + uuid.uuid4().hex + ".snapshot"
            self.snapshots[name] = data
            checksum = hashlib.sha256(data).hexdigest()
            return httpx.Response(200, json={"result": {"name": name, "checksum": "bad" if self.bad_checksum else checksum}})
        if len(bits) == 4 and bits[2] == "snapshots" and bits[3] != "upload":
            name = bits[3]
            if request.method == "GET":
                if self.fail_download:
                    return httpx.Response(502, text="private memory text")
                return httpx.Response(200, content=self.snapshots[name])
            assert request.method == "DELETE"
            self.snapshots.pop(name)
            return httpx.Response(200, json={"result": True})
        if bits[2:] == ["snapshots", "upload"]:
            assert collection.startswith("audrey_memory_verify_")
            assert collection not in self.collections
            assert request.url.params["priority"] == "snapshot"
            # HTTPX streams a real multipart upload. Read just the file field.
            multipart = request.content
            data = multipart.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
            assert hashlib.sha256(data).hexdigest() == request.url.params["checksum"]
            self.collections[collection] = json.loads(data)
            if self.fail_upload:
                return httpx.Response(500, text="partial restore")
            if self.invalid_owner:
                self.collections[collection][0]["payload"]["user"] = "mallory@example.com"
            return httpx.Response(200, json={"result": True})
        if bits[2:] == ["points", "count"]:
            count = len(self.collections[collection])
            if self.incorrect_count and collection != "kb_memory":
                count += 1
            return httpx.Response(200, json={"result": {"count": count}})
        if bits[2:] == ["points", "scroll"]:
            body = json.loads(request.content)
            assert body["with_vector"] is False
            points = self.collections[collection]
            if "offset" not in body and len(points) > 1:
                result = {"points": points[:1], "next_page_offset": points[1]["id"]}
            else:
                result = {"points": [] if self.empty_second_page else points[1:] if "offset" in body else points,
                          "next_page_offset": None}
            return httpx.Response(200, json={"result": result})
        assert len(bits) == 2
        if request.method == "GET":
            exists = collection in self.collections or self.colliding_target
            return httpx.Response(200 if exists else 404, json={"result": {}})
        assert request.method == "DELETE" and collection != "kb_memory"
        if self.fail_delete:
            return httpx.Response(503, text="failed")
        self.collections.pop(collection, None)
        return httpx.Response(200, json={"result": True})


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "chat_archive.db"
    with sqlite3.connect(source) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE archive_user_purges (user TEXT, cutoff_at TEXT)")
        db.execute("INSERT INTO archive_user_purges VALUES ('alice@example.com', '2026-10-10')")
    qdrant = Qdrant()
    client = httpx.AsyncClient(base_url="http://qdrant:6333", transport=httpx.MockTransport(qdrant))
    backup = MemoryBackup(qdrant_url="http://qdrant:6333", collection="kb_memory",
                          directory=tmp_path / "backups" / "memory", archive_path=source,
                          retain=2, http=client)
    return backup, qdrant, client


async def test_backup_publishes_complete_private_bundle_and_online_purge_receipts(setup):
    backup, qdrant, client = setup
    result = await backup.backup()
    manifest = Path(result["manifest"])
    data = json.loads(await asyncio.to_thread(manifest.read_text))
    assert result["status"] == "ok" and result["observed_memory_count"] == 2
    assert data["collection"] == "kb_memory" and data["qdrant_version"] == "1.18.3"
    assert data["live_point_count_observed_before_snapshot"] == 2
    assert "alice@example.com" not in await asyncio.to_thread(manifest.read_text)
    for path in manifest.parent.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        if path.name != "manifest.json":
            info = data["snapshot" if path.suffix == ".snapshot" else "archive"]
            assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.parent.stat().st_mode & 0o777 == 0o700
    assert backup.directory.stat().st_mode & 0o777 == 0o700
    with sqlite3.connect(manifest.parent / "chat_archive.db") as archive:
        assert archive.execute("SELECT user FROM archive_user_purges").fetchall() == [("alice@example.com",)]
    assert not qdrant.snapshots
    await client.aclose()


@pytest.mark.parametrize("failure", ["fail_create", "fail_download", "bad_checksum"])
async def test_failure_does_not_publish_or_remove_previous_good_bundle(setup, failure):
    backup, qdrant, client = setup
    first = await backup.backup()
    setattr(qdrant, failure, True)
    with pytest.raises(BackupError) as error:
        await backup.backup()
    assert "private memory text" not in str(error.value)
    assert await asyncio.to_thread(Path(first["manifest"]).is_file)
    assert len(backup._bundles()) == 1
    assert not list(backup.directory.glob(".pending-*"))
    assert not qdrant.snapshots
    await client.aclose()


async def test_missing_purge_receipts_rejects_incomplete_bundle(setup):
    backup, qdrant, client = setup
    with sqlite3.connect(backup.archive_path) as db:
        db.execute("DROP TABLE archive_user_purges")
    with pytest.raises(sqlite3.OperationalError):
        await backup.backup()
    assert backup._bundles() == []
    assert not qdrant.snapshots
    await client.aclose()


async def test_retention_removes_only_old_completed_owned_bundles(setup):
    backup, _, client = setup
    backup.directory.mkdir(parents=True)
    unrelated = backup.directory / "keep-me"
    unrelated.mkdir()
    for _ in range(4):
        result = await backup.backup()
    assert len(backup._bundles()) == 2
    assert backup.latest() == Path(result["manifest"])
    assert unrelated.is_dir()
    await client.aclose()


async def test_shared_lock_rejects_overlap_and_cleans_interrupted_scratch(setup):
    backup, _, client = setup
    lock = backup._lock()
    with pytest.raises(BackupError, match="another memory backup"):
        await backup.backup()
    lock.close()
    scratch = backup.directory / (".pending-" + uuid.uuid4().hex)
    scratch.mkdir()
    (scratch / "private").write_text("unfinished")
    await backup.backup()
    assert not scratch.exists()
    await client.aclose()


async def test_verification_restores_all_owner_scoped_points_then_removes_only_disposable_collection(setup):
    backup, qdrant, client = setup
    await backup.backup()
    original = json.dumps(qdrant.collections["kb_memory"])
    result = await backup.verify()
    assert result["status"] == "passed"
    assert result["restored_count"] == result["owner_payloads_checked"] == 2
    assert result["archive_integrity"] == "ok"
    assert result["production_collection_unchanged"] is True
    assert result["disposable_collection_deleted"] is True
    assert json.dumps(qdrant.collections["kb_memory"]) == original
    assert list(qdrant.collections) == ["kb_memory"]
    mutations = [(method, path) for method, path in qdrant.requests if "audrey_memory_verify_" in path]
    assert any(method == "DELETE" for method, _ in mutations)
    assert not any(method == "DELETE" and path == "/collections/kb_memory" for method, path in qdrant.requests)
    await client.aclose()


async def test_empty_snapshot_is_a_valid_verified_inventory(setup):
    backup, qdrant, client = setup
    qdrant.collections["kb_memory"] = []
    await backup.backup()
    result = await backup.verify()
    assert result["status"] == "passed" and result["restored_count"] == 0
    await client.aclose()


@pytest.mark.parametrize("failure", ["fail_upload", "invalid_owner", "incorrect_count", "empty_second_page"])
async def test_failed_or_partial_verification_always_cleans_disposable_collection(setup, failure):
    backup, qdrant, client = setup
    await backup.backup()
    setattr(qdrant, failure, True)
    with pytest.raises(BackupError):
        await backup.verify()
    assert list(qdrant.collections) == ["kb_memory"]
    await client.aclose()


async def test_cleanup_failure_is_reported_as_failure_not_passed(setup):
    backup, qdrant, client = setup
    await backup.backup()
    qdrant.fail_delete = True
    with pytest.raises(BackupError, match="cleanup failed"):
        await backup.verify()
    assert "kb_memory" in qdrant.collections
    await client.aclose()


@pytest.mark.parametrize("mutation", ["snapshot", "archive", "source", "path"])
async def test_untrusted_manifest_or_file_fails_before_restore_network_calls(setup, mutation, tmp_path):
    backup, qdrant, client = setup
    result = await backup.backup()
    manifest = Path(result["manifest"])
    data = json.loads(await asyncio.to_thread(manifest.read_text))
    if mutation in ("snapshot", "archive"):
        (manifest.parent / data[mutation]["filename"]).write_bytes(b"corrupted")
    elif mutation == "source":
        data["collection"] = "another_users_collection"
        await asyncio.to_thread(manifest.write_text, json.dumps(data))
    else:
        manifest = tmp_path / "untrusted.json"
        await asyncio.to_thread(manifest.write_text, json.dumps(data))
    qdrant.requests.clear()
    with pytest.raises(BackupError):
        await backup.verify(manifest)
    assert qdrant.requests == []
    await client.aclose()


@pytest.mark.parametrize("version", ["1.17.4", "1.18.2", "1.20.0", "2.18.3"])
async def test_incompatible_qdrant_version_is_rejected_before_restore(setup, version):
    backup, qdrant, client = setup
    await backup.backup()
    qdrant.requests.clear()
    qdrant.version = version
    with pytest.raises(BackupError, match="incompatible"):
        await backup.verify()
    assert qdrant.requests == [("GET", "/")]
    await client.aclose()


async def test_next_minor_version_can_restore(setup):
    backup, qdrant, client = setup
    await backup.backup()
    qdrant.version = "1.19.0"
    assert (await backup.verify())["status"] == "passed"
    await client.aclose()


async def test_preexisting_random_target_is_never_modified_or_deleted(setup):
    backup, qdrant, client = setup
    await backup.backup()
    qdrant.requests.clear()
    qdrant.colliding_target = True
    with pytest.raises(BackupError, match="not confirmed absent"):
        await backup.verify()
    assert all(method == "GET" for method, _ in qdrant.requests)
    assert list(qdrant.collections) == ["kb_memory"]
    await client.aclose()


async def test_scheduler_start_is_nonblocking_retries_and_cancels_cleanly(caplog):
    entered = asyncio.Event()
    release = asyncio.Event()
    succeeded = asyncio.Event()

    class Backup:
        attempts = 0
        closed = False

        async def backup(self):
            self.attempts += 1
            entered.set()
            await release.wait()
            if self.attempts == 1:
                raise RuntimeError("do not log private memory text")
            succeeded.set()
            return {"snapshot_bytes": 100}

        async def close(self):
            self.closed = True

    backup = Backup()
    scheduler = MemoryBackupScheduler(backup, interval_s=600, retry_s=0.001)
    scheduler.start()
    assert backup.attempts == 0
    await asyncio.wait_for(entered.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(succeeded.wait(), timeout=1)
    await scheduler.stop()
    assert backup.closed is True and scheduler._task is None
    assert "private memory text" not in caplog.text
    assert "retrying" in caplog.text


def test_backup_settings_default_to_private_daily_schedule_and_validate_bounds(tmp_path):
    settings = Settings(_env_file=None, TOOLS_DATA_DIR=tmp_path)
    assert settings.memory_backup_enabled is True
    assert settings.memory_backup_interval_s == 86400
    assert settings.memory_backup_retry_s == 300
    assert settings.memory_backup_retain == 7
    assert settings.memory_backup_dir == tmp_path / "backups" / "memory"
    with pytest.raises(ValueError):
        Settings(_env_file=None, MEMORY_BACKUP_RETAIN=0)


@pytest.mark.parametrize("collection", ["..", ".", "../kb_memory", "kb_memory/elsewhere"])
def test_unsafe_collection_paths_are_rejected(tmp_path, collection):
    with pytest.raises(ValueError):
        MemoryBackup(qdrant_url="http://qdrant:6333", collection=collection,
                     directory=tmp_path, archive_path=tmp_path / "archive")


@pytest.mark.parametrize("name", ["..", ".", "../memory.snapshot", "/memory.snapshot"])
async def test_malformed_snapshot_name_never_becomes_a_collection_delete(setup, name):
    backup, qdrant, client = setup
    original = backup._request

    async def request(method, path, **kwargs):
        if method == "POST" and path.endswith("/snapshots"):
            return {"result": {"name": name}}
        return await original(method, path, **kwargs)

    backup._request = request
    with pytest.raises(BackupError, match="invalid snapshot name"):
        await backup.backup()
    assert not any(method == "DELETE" for method, _ in qdrant.requests)
    await client.aclose()


@pytest.mark.parametrize("action", ["backup", "verify"])
async def test_cancellation_during_lock_acquisition_releases_the_returned_lock(setup, action):
    backup, _, client = setup
    acquired = threading.Event()
    release = threading.Event()
    original = backup._lock

    def delayed_lock():
        lock = original()
        acquired.set()
        assert release.wait(timeout=2)
        return lock

    backup._lock = delayed_lock
    task = asyncio.create_task(getattr(backup, action)())
    assert await asyncio.to_thread(acquired.wait, 2)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # A second descriptor acquires LOCK_EX only if cancellation closed the first.
    with original():
        pass
    await client.aclose()


def test_sqlite_backup_stops_busy_retries_at_its_deadline(tmp_path):
    source = tmp_path / "busy.db"
    destination = tmp_path / "copy.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE archive_user_purges (user TEXT, cutoff_at TEXT)")
        db.execute("BEGIN EXCLUSIVE")
        try:
            with pytest.raises(BackupError, match="timed out"):
                _sqlite_backup(source, destination, time.monotonic() + 0.05)
        finally:
            db.rollback()
