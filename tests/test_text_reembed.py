"""Real local collections prove safe re-embedding and interrupted cutover recovery."""
import asyncio
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from audrey.embedding_profile import QWEN_QUERY_PREFIX, EmbeddingProfile
from audrey.kb import reembed as m


@pytest.fixture
def setup(tmp_path):
    client = QdrantClient(":memory:")
    old = [1.] + [0.] * 767
    ids = [str(uuid.uuid4()) for _ in range(6)]
    for name in ("kb_text", "kb_user_text_owner", "kb_memory", "kb_chat_archive"):
        client.create_collection(name, vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE))
    client.upsert("kb_text", [qm.PointStruct(id=ids[0], vector=old,
        payload={"text": "A retained geology passage", "source": "/source", "kind": "text", "mtime": 7})])
    client.upsert("kb_user_text_owner", [qm.PointStruct(id=ids[i], vector=old,
        payload={"text": "Private retained text", "user": "owner", "file_id": f"file-{i}"}) for i in (1, 2)])
    client.upsert("kb_memory", [qm.PointStruct(id=ids[3], vector=old,
        payload={"key": "desk", "value": "oak", "tags": "user:owner furniture", "user": "owner",
                 "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-02-01T00:00:00+00:00"})])
    client.create_collection("kb_images", vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE))
    client.upsert("kb_images", [qm.PointStruct(id=ids[4], vector=[1.] + [0.] * 511, payload={"caption": "image"})])
    uploads, archive, app = [tmp_path / name for name in ("uploads.sqlite", "archive.sqlite", "app.sqlite")]
    with sqlite3.connect(uploads) as db:
        db.execute("CREATE TABLE file_deletions(user TEXT,file_id TEXT)")
        db.execute("INSERT INTO file_deletions VALUES('owner','file-2')")
    with sqlite3.connect(archive) as db:
        db.executescript('''
        CREATE TABLE archive_chunks(chunk_id TEXT, conversation_id TEXT, user TEXT,
            message_ids_json TEXT, text TEXT, created_at TEXT, indexed_at TEXT);
        CREATE TABLE archive_deletion_outbox(chunk_id TEXT);
        CREATE TABLE archive_user_purges(user TEXT,cutoff_at TEXT);
        CREATE TABLE archive_conversation_deletions(user TEXT,conversation_id TEXT,cutoff_at TEXT);
        INSERT INTO archive_chunks VALUES('c1','conversation','owner','m1,m2','First chat text','2026-02-01T00:00:00+00:00','indexed');
        INSERT INTO archive_chunks VALUES('c2','conversation','owner','m3,m4','Unindexed chat text','2026-02-02T00:00:00+00:00',NULL);
        INSERT INTO archive_chunks VALUES('c3','deleted','owner','m5,m6','Deleted chat text','2026-02-03T00:00:00+00:00','indexed');
        INSERT INTO archive_deletion_outbox VALUES('c3');
        ''')
    with sqlite3.connect(app) as db:
        db.execute("CREATE TABLE accounts(id TEXT)")
    state = tmp_path / "private"
    state.mkdir()
    profile = EmbeddingProfile("qwen3-embedding:4b", "b" * 64, 3, "", QWEN_QUERY_PREFIX)
    embedder = SimpleNamespace(profile=profile, verify_identity=AsyncMock(),
        embed_many=AsyncMock(side_effect=lambda rows: [[1., 0., 0.] for _ in rows]))
    args = dict(state=state, uploads=uploads, archive=archive, app_db=app)
    yield client, embedder, args, ids
    client.close()


async def test_full_rebuild_preserves_owned_payloads_excludes_deletions_and_recovers_unindexed_history(setup):
    client, embedder, args, ids = setup
    report = await m.migrate(client, embedder, {}, **args)
    assert report["status"] == "completed"
    assert [row["points"] for row in report["collections"]] == [1, 1, 1, 2]
    assert client.retrieve("kb_user_text_owner", ids=[ids[2]]) == []
    payload = client.retrieve("kb_text", ids=[ids[0]])[0].payload
    assert payload == {"text": "A retained geology passage", "source": "/source", "kind": "text", "mtime": 7}
    assert client.get_collection("kb_text").config.params.vectors.size == 3
    assert "bm25" in client.get_collection("kb_text").config.params.sparse_vectors
    assert client.get_collection("kb_images").config.params.vectors.size == 512
    assert client.count("kb_images").count == 1
    assert client.get_collection("kb_memory").config.metadata == embedder.profile.metadata
    assert "desk: oak [tags: furniture]" in [text for call in embedder.embed_many.await_args_list for text in call.args[0]]
    assert "Unindexed chat text" in [text for call in embedder.embed_many.await_args_list for text in call.args[0]]
    assert "owner" not in json.dumps(report) and "file-1" not in json.dumps(report)
    journal = json.loads((args["state"] / "manifest.json").read_text())
    assert all(entry["phase"] == "complete" for entry in journal["collections"])
    assert (args["state"] / "uploads.sqlite").stat().st_mode & 0o777 == 0o600
    assert len(list(m.exported(args["state"] / "collection-0.jsonl.gz"))) == 1


async def fail_cutover(client, embedder, args, monkeypatch):
    original = client.upsert
    def failing(name, *a, **kw):
        if name == "kb_text":
            raise RuntimeError("injected target interruption")
        return original(name, *a, **kw)
    monkeypatch.setattr(client, "upsert", failing)
    with pytest.raises(RuntimeError):
        await m.migrate(client, embedder, {}, **args)
    monkeypatch.setattr(client, "upsert", original)


async def test_interrupted_copy_resumes_verified_staging_without_reembedding(setup, monkeypatch):
    client, embedder, args, _ = setup
    await fail_cutover(client, embedder, args, monkeypatch)
    calls = embedder.embed_many.await_count
    assert client.count("kb_text").count == 0
    report = await m.migrate(client, embedder, {}, **args)
    assert report["status"] == "completed"
    assert embedder.embed_many.await_count == calls
    assert client.count("kb_text").count == 1


async def test_changed_tombstones_block_resume_before_any_further_cutover(setup, monkeypatch):
    client, embedder, args, _ = setup
    await fail_cutover(client, embedder, args, monkeypatch)
    with sqlite3.connect(args["uploads"]) as db:
        db.execute("INSERT INTO file_deletions VALUES('owner','file-1')")
    with pytest.raises(m.MigrationError, match="SQLite changed"):
        await m.migrate(client, embedder, {}, **args)


async def test_staging_vector_tampering_blocks_resume_even_when_payloads_match(setup, monkeypatch):
    client, embedder, args, _ = setup
    await fail_cutover(client, embedder, args, monkeypatch)
    journal = json.loads((args["state"] / "manifest.json").read_text())
    stage = journal["collections"][0]["stage"]
    row = next(m.records(client, stage))
    vector = dict(row.vector)
    vector[""] = [0., 1., 0.]
    client.upsert(stage, [qm.PointStruct(id=row.id, payload=row.payload, vector=vector)])
    with pytest.raises(m.MigrationError, match="staging collection changed"):
        await m.migrate(client, embedder, {}, **args)


async def test_invalid_retained_text_never_replaces_originals(setup):
    client, embedder, args, ids = setup
    client.set_payload("kb_text", {"text": ""}, points=[ids[0]])
    with pytest.raises(m.MigrationError, match="recoverable text"):
        await m.migrate(client, embedder, {}, **args)
    assert client.get_collection("kb_text").config.params.vectors.size == 768


async def test_completed_rerun_never_replays_old_private_exports_over_new_data(setup):
    client, embedder, args, _ = setup
    await m.migrate(client, embedder, {}, **args)
    client.upsert("kb_memory", [qm.PointStruct(id=str(uuid.uuid4()), vector=[1., 0., 0.], payload={"key": "new"})])
    with sqlite3.connect(args["app_db"]) as db:
        db.execute("INSERT INTO accounts VALUES('new')")
    calls = embedder.embed_many.await_count
    report = await m.migrate(client, embedder, {}, **args)
    assert report["resumed_or_already_complete"]
    assert embedder.embed_many.await_count == calls
    assert client.count("kb_memory").count == 2


async def test_account_purge_excludes_old_memory_and_history(setup):
    client, embedder, args, _ = setup
    with sqlite3.connect(args["archive"]) as db:
        db.execute("INSERT INTO archive_user_purges VALUES('owner','2026-02-01T00:00:00+00:00')")
    report = await m.migrate(client, embedder, {}, **args)
    roles = {row["role"]: row["points"] for row in report["collections"]}
    assert roles["memory"] == 0 and roles["history"] == 1


async def test_conversation_deletion_cutoff_excludes_old_but_not_new_history(setup):
    client, embedder, args, _ = setup
    with sqlite3.connect(args["archive"]) as db:
        db.execute("INSERT INTO archive_conversation_deletions VALUES('owner','conversation','2026-02-01T00:00:00+00:00')")
    await m.migrate(client, embedder, {}, **args)
    assert client.count("kb_chat_archive").count == 1


def test_cli_without_maintenance_confirmation_does_not_create_report_outside_state(tmp_path):
    report = tmp_path / "outside.json"
    assert m.main(["--config", str(tmp_path/"missing"), "--uploads-db", str(tmp_path/"u"),
        "--archive-db", str(tmp_path/"a"), "--app-db", str(tmp_path/"b"),
        "--state", str(tmp_path/"state"), "--report", str(report)]) != 0
    assert not report.exists()


@pytest.fixture
def cli(setup, monkeypatch):
    client, embedder, args, _ = setup
    config = args["state"].parent / "config.yaml"
    profile = embedder.profile
    settings = {
        "model": profile.model, "digest": profile.digest, "dimensions": profile.dimensions,
        "document_prefix": profile.document_prefix, "query_prefix": profile.query_prefix,
    }
    config.write_text(json.dumps({"text_embedding": settings, "history_embedding": settings}))
    report = args["state"] / "reports" / "result.json"
    ollama = SimpleNamespace(aclose=AsyncMock())
    closed = Mock(wraps=client.close)
    monkeypatch.setattr(client, "close", closed)

    def qdrant_client(**_):
        assert asyncio.get_running_loop().is_running()
        return client

    def ollama_client(_):
        assert asyncio.get_running_loop().is_running()
        return ollama

    monkeypatch.setattr(m, "QdrantClient", qdrant_client)
    monkeypatch.setattr(m, "OllamaClient", ollama_client)
    monkeypatch.setattr(m, "text_embedder", lambda *_: embedder)
    monkeypatch.setattr(m, "history_text_embedder", lambda *_: embedder)
    argv = ["--config", str(config), "--uploads-db", str(args["uploads"]),
        "--archive-db", str(args["archive"]), "--app-db", str(args["app_db"]),
        "--state", str(args["state"]), "--report", str(report), "--maintenance-confirmed"]
    return SimpleNamespace(argv=argv, report=report, state=args["state"], client=client,
        embedder=embedder, ollama=ollama, closed=closed)


def test_main_runs_real_migration_and_closes_both_clients(cli):
    assert m.main(cli.argv) == 0
    report = json.loads(cli.report.read_text())
    assert report["status"] == "completed" and report["exit_code"] == 0
    assert [row["points"] for row in report["collections"]] == [1, 1, 1, 2]
    assert "private_diagnostic" not in report
    cli.ollama.aclose.assert_awaited_once()
    cli.closed.assert_called_once()


@pytest.mark.parametrize(("error_type", "expected_code"), [(RuntimeError, 3), (ValueError, 1)])
def test_main_private_traceback_preserves_primary_failure_and_closes_both_clients(
    cli, capsys, error_type, expected_code,
):
    cli.embedder.verify_identity.side_effect = error_type("PRIVATE_SOURCE_SENTINEL")
    cli.ollama.aclose.side_effect = OSError("PRIVATE_CLOSE_SENTINEL")
    before = os.umask(0o077)
    os.umask(before)
    try:
        assert m.main(cli.argv) == expected_code
        after = os.umask(before)
        assert after == before
    finally:
        os.umask(before)
    report = json.loads(cli.report.read_text())
    assert report["error_kind"] == error_type.__name__ and report["stage"] == "migration"
    assert report["writers_must_remain_stopped"]
    public = cli.report.read_text() + capsys.readouterr().out
    assert "PRIVATE_SOURCE_SENTINEL" not in public and "PRIVATE_CLOSE_SENTINEL" not in public
    diagnostic = cli.state / report["private_diagnostic"]
    assert diagnostic.is_relative_to(cli.state / "diagnostics")
    assert diagnostic.stat().st_mode & 0o777 == 0o600
    private = diagnostic.read_text()
    assert "PRIVATE_SOURCE_SENTINEL" in private and "Traceback" in private
    assert error_type.__name__ in private
    cli.ollama.aclose.assert_awaited_once()
    cli.closed.assert_called_once()


def test_main_reports_cleanup_failure_after_verified_migration(cli, capsys):
    cli.ollama.aclose.side_effect = RuntimeError("PRIVATE_CLOSE_SENTINEL")
    assert m.main(cli.argv) == 3
    report = json.loads(cli.report.read_text())
    assert report["stage"] == "cleanup" and report["error_kind"] == "RuntimeError"
    assert json.loads((cli.state / "manifest.json").read_text())["status"] == "completed"
    assert "PRIVATE_CLOSE_SENTINEL" not in cli.report.read_text() + capsys.readouterr().out
    assert "PRIVATE_CLOSE_SENTINEL" in (cli.state / report["private_diagnostic"]).read_text()
    cli.closed.assert_called_once()


def test_main_refuses_symlinked_private_diagnostic_directory(cli, tmp_path, capsys):
    outside = tmp_path / "outside"
    outside.mkdir()
    (cli.state / "diagnostics").symlink_to(outside, target_is_directory=True)
    cli.embedder.verify_identity.side_effect = RuntimeError("PRIVATE_SOURCE_SENTINEL")
    assert m.main(cli.argv) == 3
    report = json.loads(cli.report.read_text())
    assert report["private_diagnostic_saved"] is False
    assert "private_diagnostic" not in report
    assert list(outside.iterdir()) == []
    assert "PRIVATE_SOURCE_SENTINEL" not in capsys.readouterr().out


def test_main_constructor_failure_closes_already_created_client(cli, monkeypatch):
    def failed_ollama(_):
        raise RuntimeError("PRIVATE_CONSTRUCTOR_SENTINEL")

    monkeypatch.setattr(m, "OllamaClient", failed_ollama)
    assert m.main(cli.argv) == 3
    report = json.loads(cli.report.read_text())
    assert report["stage"] == "ollama_client"
    assert "PRIVATE_CONSTRUCTOR_SENTINEL" not in cli.report.read_text()
    cli.closed.assert_called_once()


def test_private_diagnostic_never_replaces_or_follows_an_existing_file(tmp_path, monkeypatch):
    state = tmp_path / "state"
    diagnostics = state / "diagnostics"
    diagnostics.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("retained")
    target = diagnostics / "failure-collision.traceback.txt"
    target.symlink_to(outside)
    monkeypatch.setattr(m.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    with pytest.raises(FileExistsError):
        m.private_diagnostic(state, RuntimeError("PRIVATE_SOURCE_SENTINEL"))
    assert outside.read_text() == "retained" and target.is_symlink()


def sqlite_source_bytes(path):
    return {suffix: candidate.read_bytes()
            for suffix in ("", "-wal", "-shm", "-journal")
            if (candidate := type(path)(str(path) + suffix)).exists()}


def freeze_wal_source(path, target, statements):
    """Leave a stopped source with committed WAL rows and no rebuilt SHM."""
    target.parent.mkdir()
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        for statement in statements:
            db.execute(statement)
        db.commit()
        shutil.copyfile(path, target)
        shutil.copyfile(str(path) + "-wal", str(target) + "-wal")
    for child in target.parent.iterdir():
        child.chmod(0o400)
    target.parent.chmod(0o500)
    return target


def test_closed_wal_source_on_readonly_directory_fingerprints_and_backs_up(tmp_path):
    directory = tmp_path / "readonly"
    directory.mkdir()
    source = directory / "source.sqlite"
    with closing(sqlite3.connect(source)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE retained(value TEXT)")
        db.execute("INSERT INTO retained VALUES('saved')")
        db.commit()
    assert set(sqlite_source_bytes(source)) == {""}
    source.chmod(0o400)
    directory.chmod(0o500)
    before = sqlite_source_bytes(source)
    state = tmp_path / "private"
    state.mkdir(mode=0o700)
    target = state / "backup.sqlite"
    try:
        fingerprint = m.sql_fingerprint(source, scratch_root=state)
        assert m.backup_sql(source, target) == fingerprint
        assert sqlite_source_bytes(source) == before
        with m.readonly(target) as db:
            assert db.execute("SELECT value FROM retained").fetchall() == [("saved",)]
            assert db.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert target.stat().st_mode & 0o777 == 0o600
        assert list(state.glob(".sqlite-snapshot-*")) == []
    finally:
        directory.chmod(0o700)


def test_wal_only_commits_survive_private_backup_without_source_changes(tmp_path):
    original = tmp_path / "original.sqlite"
    with closing(sqlite3.connect(original)) as db:
        db.execute("CREATE TABLE retained(value TEXT)")
        db.execute("INSERT INTO retained VALUES('checkpointed')")
        db.commit()
    source = freeze_wal_source(original, tmp_path / "readonly" / "source.sqlite",
        ["INSERT INTO retained VALUES('only-in-wal')"])
    state = tmp_path / "private"
    state.mkdir(mode=0o700)
    before = sqlite_source_bytes(source)
    assert set(before) == {"", "-wal"}
    try:
        target = state / "backup.sqlite"
        assert m.backup_sql(source, target) == m.sql_fingerprint(source, scratch_root=state)
        with m.readonly(target) as db:
            assert db.execute("SELECT value FROM retained ORDER BY rowid").fetchall() == [
                ("checkpointed",), ("only-in-wal",)]
        assert sqlite_source_bytes(source) == before
        assert list(state.glob(".sqlite-snapshot-*")) == []
    finally:
        source.parent.chmod(0o700)


def test_nonempty_rollback_journal_refuses_snapshot_without_touching_source(tmp_path):
    source = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(source)) as db:
        db.execute("CREATE TABLE retained(value TEXT)")
    journal = tmp_path / "source.sqlite-journal"
    journal.write_bytes(b"unrecovered rollback state")
    state = tmp_path / "private"
    state.mkdir(mode=0o700)
    before = sqlite_source_bytes(source)
    with pytest.raises(m.MigrationError, match="rollback journal"):
        m.backup_sql(source, state / "backup.sqlite")
    assert sqlite_source_bytes(source) == before
    assert list(state.iterdir()) == []


async def test_wal_migration_reads_verified_backups_for_history_and_deletion_policy(setup, monkeypatch):
    client, embedder, args, _ = setup
    changes = {
        "uploads": ["INSERT INTO file_deletions VALUES('owner','file-1')"],
        "archive": [
            "INSERT INTO archive_chunks VALUES('c4','conversation','owner','m7,m8',"
            "'History committed only in WAL','2026-02-04T00:00:00+00:00',NULL)",
            "INSERT INTO archive_user_purges VALUES('owner','2026-02-01T00:00:00+00:00')",
        ],
        "app_db": ["INSERT INTO accounts VALUES('wal-account')"],
    }
    sources = []
    for key, statements in changes.items():
        original = args[key]
        args[key] = freeze_wal_source(original, original.parent / f"readonly-{key}" / original.name,
                                     statements)
        sources.append((args[key], sqlite_source_bytes(args[key])))
    policies, live_archive = m.policies, m.live_archive

    def read_policies(uploads, archive):
        assert uploads == args["state"] / "uploads.sqlite"
        assert archive == args["state"] / "archive.sqlite"
        return policies(uploads, archive)

    def read_archive(path, cutoff):
        assert path == args["state"] / "archive.sqlite"
        return live_archive(path, cutoff)

    monkeypatch.setattr(m, "policies", read_policies)
    monkeypatch.setattr(m, "live_archive", read_archive)
    try:
        report = await m.migrate(client, embedder, {}, **args)
        assert report["status"] == "completed"
        assert {row["role"]: row["points"] for row in report["collections"]} == {
            "kb": 1, "user_files": 0, "memory": 0, "history": 2}
        embedded = [text for call in embedder.embed_many.await_args_list for text in call.args[0]]
        assert "History committed only in WAL" in embedded
        assert "First chat text" not in embedded
        with m.readonly(args["state"] / "app.sqlite") as db:
            assert db.execute("SELECT id FROM accounts").fetchall() == [("wal-account",)]
        assert all(sqlite_source_bytes(path) == before for path, before in sources)
        assert list(args["state"].glob(".sqlite-snapshot-*")) == []
    finally:
        for path, _ in sources:
            path.parent.chmod(0o700)


async def test_resume_guard_detects_source_change_committed_only_in_wal(setup, monkeypatch):
    client, embedder, args, _ = setup
    await fail_cutover(client, embedder, args, monkeypatch)
    calls = embedder.embed_many.await_count
    with closing(sqlite3.connect(args["uploads"])) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("INSERT INTO file_deletions VALUES('owner','file-1')")
        db.commit()
        with pytest.raises(m.MigrationError, match="Source SQLite changed"):
            await m.migrate(client, embedder, {}, **args)
    assert embedder.embed_many.await_count == calls
    assert client.get_collection("kb_memory").config.params.vectors.size == 768


@pytest.mark.parametrize("remove", [False, True])
async def test_resume_refuses_changed_or_missing_private_sql_backup(setup, monkeypatch, remove):
    client, embedder, args, _ = setup
    await fail_cutover(client, embedder, args, monkeypatch)
    backup = args["state"] / "archive.sqlite"
    if remove:
        backup.unlink()
    else:
        with closing(sqlite3.connect(backup)) as db:
            db.execute("DELETE FROM archive_deletion_outbox")
            db.commit()
    calls = embedder.embed_many.await_count
    with pytest.raises(m.MigrationError, match="Private SQLite backup differs"):
        await m.migrate(client, embedder, {}, **args)
    assert embedder.embed_many.await_count == calls
    assert client.get_collection("kb_memory").config.params.vectors.size == 768


async def test_absent_original_archive_never_replays_stale_sql_backup(setup):
    client, embedder, args, _ = setup
    client.delete_collection("kb_memory")
    shutil.copyfile(args["archive"], args["state"] / "archive.sqlite")
    args["archive"].unlink()
    report = await m.migrate(client, embedder, {}, **args)
    assert {row["role"]: row["points"] for row in report["collections"]}["history"] == 0
    journal = json.loads((args["state"] / "manifest.json").read_text())
    assert journal["sql"]["archive"] is None
    assert m.verified_sql_backups(journal, args["state"])["archive"] is None


def nomic_history_embedder():
    profile = EmbeddingProfile(
        "nomic-embed-text:latest",
        "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
        768, "search_document: ", "search_query: ",
    )
    return SimpleNamespace(
        profile=profile, verify_identity=AsyncMock(),
        embed_many=AsyncMock(side_effect=lambda rows: [[1.] + [0.] * 767 for _ in rows]),
    )


def retained_collection(client, name):
    return {
        "schema": m.schema(client, name),
        "rows": [row.model_dump(mode="json") for row in m.records(client, name)],
    }


async def test_history_scope_replaces_only_history_and_needs_no_upload_or_app_database(setup):
    client, qwen, args, _ = setup
    await m.migrate(client, qwen, {}, **args)
    original_state = args["state"]
    source_before = sqlite_source_bytes(args["archive"])
    untouched = {name: retained_collection(client, name)
                 for name in ("kb_text", "kb_user_text_owner", "kb_memory", "kb_images")}
    args = {**args, "state": original_state.parent / "history-reembed-nomic",
            "uploads": original_state.parent / "missing-uploads.sqlite",
            "app_db": original_state.parent / "missing-app.sqlite"}
    args["state"].mkdir()
    history = nomic_history_embedder()
    previous_calls = qwen.embed_many.await_count
    previous_identity_checks = qwen.verify_identity.await_count

    report = await m.migrate(client, qwen, {}, scope="history", history_embedder=history, **args)

    assert report["status"] == "completed" and report["scope"] == "history"
    assert report["embedding"]["embedding_model"] == "nomic-embed-text:latest"
    assert report["embedding"]["embedding_dimensions"] == 768
    assert [row["role"] for row in report["collections"]] == ["history"]
    assert [row["points"] for row in report["collections"]] == [2]
    assert qwen.embed_many.await_count == previous_calls
    assert qwen.verify_identity.await_count == previous_identity_checks
    assert history.embed_many.await_count == 1
    assert client.get_collection("kb_chat_archive").config.metadata == history.profile.metadata
    assert client.get_collection("kb_chat_archive").config.params.vectors.size == 768
    assert source_before == sqlite_source_bytes(args["archive"])
    for name, before in untouched.items():
        assert retained_collection(client, name) == before
    journal = json.loads((args["state"] / "manifest.json").read_text())
    assert set(journal["sql"]) == {"archive"}
    assert set(journal["profiles"]) == {"history"}
    assert not (args["state"] / "uploads.sqlite").exists()
    assert json.loads((original_state / "manifest.json").read_text())["scope"] == "all"


async def test_all_scope_uses_each_collection_role_profile(setup):
    client, qwen, args, _ = setup
    history = nomic_history_embedder()
    report = await m.migrate(client, qwen, {}, history_embedder=history, **args)
    assert client.get_collection("kb_text").config.params.vectors.size == 3
    assert client.get_collection("kb_memory").config.metadata == qwen.profile.metadata
    assert client.get_collection("kb_chat_archive").config.params.vectors.size == 768
    assert client.get_collection("kb_chat_archive").config.metadata == history.profile.metadata
    assert set(report["embeddings"]) == {"text", "history"}
    assert "First chat text" not in [text for call in qwen.embed_many.await_args_list for text in call.args[0]]
    assert "First chat text" in [text for call in history.embed_many.await_args_list for text in call.args[0]]


async def test_history_scope_resumes_cutover_without_reembedding_or_other_collection_writes(setup, monkeypatch):
    client, qwen, args, _ = setup
    history = nomic_history_embedder()
    original = client.upsert

    def fail_history(name, *positional, **kwargs):
        if name == "kb_chat_archive":
            raise RuntimeError("interrupted history cutover")
        return original(name, *positional, **kwargs)

    monkeypatch.setattr(client, "upsert", fail_history)
    before = retained_collection(client, "kb_text")
    with pytest.raises(RuntimeError, match="interrupted history cutover"):
        await m.migrate(client, qwen, {}, scope="history", history_embedder=history, **args)
    count = history.embed_many.await_count
    monkeypatch.setattr(client, "upsert", original)
    report = await m.migrate(client, qwen, {}, scope="history", history_embedder=history, **args)
    assert report["status"] == "completed"
    assert history.embed_many.await_count == count
    assert retained_collection(client, "kb_text") == before
    qwen.embed_many.assert_not_awaited()
    qwen.verify_identity.assert_not_awaited()


@pytest.mark.parametrize("legacy_journal", [False, True])
async def test_old_unified_migration_state_refuses_split_profile_or_history_scope(setup, legacy_journal):
    client, qwen, args, _ = setup
    await m.migrate(client, qwen, {}, **args)
    manifest = args["state"] / "manifest.json"
    if legacy_journal:
        journal = json.loads(manifest.read_text())
        del journal["scope"]
        del journal["profiles"]
        manifest.write_text(json.dumps(journal))
    before = {name: retained_collection(client, name)
              for name in ("kb_text", "kb_user_text_owner", "kb_memory", "kb_chat_archive", "kb_images")}
    for scope in ("all", "history"):
        with pytest.raises(m.MigrationError, match="scope or embedding profile"):
            await m.migrate(client, qwen, {}, scope=scope, history_embedder=nomic_history_embedder(), **args)
    for name, expected in before.items():
        assert retained_collection(client, name) == expected


async def test_history_scope_preserves_purge_conversation_delete_and_retention_exclusions(setup):
    client, qwen, args, _ = setup
    with sqlite3.connect(args["archive"]) as db:
        db.execute("INSERT INTO archive_user_purges VALUES('owner','2026-02-01T00:00:00+00:00')")
        db.execute("INSERT INTO archive_conversation_deletions VALUES('owner','conversation','2026-02-02T00:00:00+00:00')")
        db.execute("INSERT INTO archive_chunks VALUES('live','current','other','n1,n2','Live current chat',?,NULL)",
                   (m.datetime.now(m.UTC).isoformat(),))
    report = await m.migrate(client, qwen, {}, scope="history", retention_days=5,
                             history_embedder=nomic_history_embedder(), **args)
    assert report["collections"][0]["points"] == 1
    assert next(m.records(client, "kb_chat_archive")).payload["text"] == "Live current chat"


def test_history_cli_does_not_load_or_call_the_general_text_embedder(cli, monkeypatch):
    history = nomic_history_embedder()
    monkeypatch.setattr(m, "history_text_embedder", lambda *_: history)
    text = Mock(side_effect=AssertionError("text embedder must not run for history scope"))
    monkeypatch.setattr(m, "text_embedder", text)
    assert m.main([*cli.argv, "--scope", "history"]) == 0
    assert json.loads(cli.report.read_text())["scope"] == "history"
    text.assert_not_called()
    cli.embedder.embed_many.assert_not_awaited()


def test_main_requires_separate_history_configuration_before_clients(cli):
    config_path = type(cli.report)(cli.argv[cli.argv.index("--config") + 1])
    config = json.loads(config_path.read_text())
    del config["history_embedding"]
    config_path.write_text(json.dumps(config))
    assert m.main(cli.argv) == 2
    assert json.loads(cli.report.read_text())["stage"] == "configuration"
    cli.embedder.embed_many.assert_not_awaited()
    cli.closed.assert_not_called()


@pytest.mark.parametrize("collection", ["kb_text", "kb_memory", "kb_images", "kb_user_text_owner"])
async def test_history_scope_rejects_collection_override_pointing_at_other_data(setup, collection):
    client, qwen, args, _ = setup
    before = retained_collection(client, collection)
    with pytest.raises(m.MigrationError, match="overlaps another data role"):
        await m.migrate(client, qwen, {}, scope="history", history=collection,
                        history_embedder=nomic_history_embedder(), **args)
    assert retained_collection(client, collection) == before


async def test_history_migration_factory_uses_nomic_document_prefix_and_exact_native_width(tmp_path):
    profile = nomic_history_embedder().profile
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"history_embedding": {
        "model": profile.model, "digest": profile.digest, "dimensions": profile.dimensions,
        "document_prefix": profile.document_prefix, "query_prefix": profile.query_prefix,
        "keep_alive": "24h", "timeout_s": 8, "min_score": .4,
    }}))
    ollama = SimpleNamespace(
        tags=AsyncMock(return_value=[{"name": profile.model, "digest": profile.digest}]),
        embed=AsyncMock(return_value=[[1.] + [0.] * 767]),
    )
    embedder = m.history_text_embedder(ollama, path)
    vectors = await embedder.embed_many(["Saved arithmetic exchange"])
    assert len(vectors[0]) == 768
    ollama.embed.assert_awaited_once_with(
        model=profile.model, texts=["search_document: Saved arithmetic exchange"],
        timeout_s=120, keep_alive="24h", truncate=False,
    )


def test_archive_backup_preserves_external_content_fts_without_changing_source(tmp_path):
    source = tmp_path / "archive.sqlite"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE archive_chunks(chunk_id TEXT, text TEXT)")
        db.execute("INSERT INTO archive_chunks VALUES('saved','Calculate 1250 × 1.08')")
        db.execute("CREATE VIRTUAL TABLE archive_chunks_fts USING fts5(text, content='archive_chunks', tokenize='porter unicode61')")
        db.execute("INSERT INTO archive_chunks_fts(archive_chunks_fts) VALUES('rebuild')")
    state = tmp_path / "state"
    state.mkdir()
    before = sqlite_source_bytes(source)
    target = state / "archive.sqlite"
    assert m.backup_sql(source, target) == m.sql_fingerprint(source, scratch_root=state)
    with m.readonly(target) as db:
        assert db.execute("SELECT text FROM archive_chunks_fts WHERE archive_chunks_fts MATCH ?",
                          ('"calculation"',)).fetchall() == [("Calculate 1250 × 1.08",)]
    assert sqlite_source_bytes(source) == before


@pytest.mark.parametrize("completed", [False, True])
async def test_history_resume_refuses_changed_requested_collection_without_mutations(setup, monkeypatch, completed):
    client, qwen, args, _ = setup
    history = nomic_history_embedder()
    upsert = client.upsert

    def interrupted(name, *positional, **kwargs):
        if name == "kb_chat_archive":
            raise RuntimeError("interrupted history cutover")
        return upsert(name, *positional, **kwargs)

    if completed:
        await m.migrate(client, qwen, {}, scope="history", history_embedder=history, **args)
    else:
        monkeypatch.setattr(client, "upsert", interrupted)
        with pytest.raises(RuntimeError, match="interrupted history cutover"):
            await m.migrate(client, qwen, {}, scope="history", history_embedder=history, **args)
        monkeypatch.setattr(client, "upsert", upsert)
    before = {item.name: retained_collection(client, item.name)
              for item in client.get_collections().collections}
    with pytest.raises(m.MigrationError, match="requested collection"):
        await m.migrate(client, qwen, {}, scope="history", history="another-history-collection",
                        history_embedder=history, **args)
    assert not client.collection_exists("another-history-collection")
    assert {item.name: retained_collection(client, item.name)
            for item in client.get_collections().collections} == before


async def test_ordinary_full_migration_refuses_protected_collections_before_any_write(setup):
    client, embedder, args, _ = setup
    client.create_collection("kb_private_text_owner", vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE))
    before = {item.name for item in client.get_collections().collections}
    with pytest.raises(m.MigrationError, match="dedicated local migration"):
        await m.migrate(client, embedder, {}, **args)
    assert {item.name for item in client.get_collections().collections} == before
    assert not list(args["state"].iterdir())
    embedder.verify_identity.assert_not_awaited()
    embedder.embed_many.assert_not_awaited()
    assert client.get_collection("kb_text").config.params.vectors.size == 768


async def test_history_only_migration_leaves_protected_upload_collections_untouched(setup):
    client, embedder, args, _ = setup
    name = "kb_private_text_owner"
    client.create_collection(name, vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE))
    report = await m.migrate(client, embedder, {}, scope="history", **args)
    assert report["status"] == "completed"
    assert client.get_collection(name).config.params.vectors.size == 768
    assert {row["role"] for row in report["collections"]} == {"history"}


async def test_history_only_migration_cannot_target_a_protected_upload_collection(setup):
    client, embedder, args, _ = setup
    name = "kb_private_text_owner"
    client.create_collection(name, vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE))
    with pytest.raises(m.MigrationError, match="overlaps another data role"):
        await m.migrate(client, embedder, {}, scope="history", history=name, **args)
    embedder.verify_identity.assert_not_awaited()
    assert client.get_collection(name).config.params.vectors.size == 768
