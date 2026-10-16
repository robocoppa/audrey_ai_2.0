"""Real local collections prove safe re-embedding and interrupted cutover recovery."""
import json
import sqlite3
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
