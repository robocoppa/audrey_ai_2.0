"""Offline-writer migration from retained text to the configured embedding space.

Private source/vector exports and a durable journal remain on Tower. Only counts,
roles and fingerprints are emitted. Application services must remain stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import sqlite3
import traceback
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from audrey.embedding_profile import load_profile
from audrey.kb import bm25
from audrey.kb.embed import text_embedder
from audrey.models.ollama import OllamaClient

PAGE = 64


class MigrationError(RuntimeError):
    """A migration invariant failed; keep application writers stopped."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        temporary.chmod(0o600)
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def private_diagnostic(state, error):
    """Keep exception details on Tower; never follow a diagnostic symlink."""
    state_fd = os.open(state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.mkdir("diagnostics", mode=0o700, dir_fd=state_fd)
        except FileExistsError:
            pass
        directory_fd = os.open("diagnostics", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=state_fd)
        try:
            os.fchmod(directory_fd, 0o700)
            name = f"failure-{uuid.uuid4().hex}.traceback.txt"
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory_fd)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                traceback.print_exception(error, file=output)
                output.flush()
                os.fsync(output.fileno())
            os.fsync(directory_fd)
            return f"diagnostics/{name}"
        finally:
            os.close(directory_fd)
    finally:
        os.close(state_fd)


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def sql_fingerprint(path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with readonly(path) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise MigrationError("Source SQLite integrity check failed")
        for line in db.iterdump():
            digest.update(line.encode())
            digest.update(b"\n")
    return digest.hexdigest()


def backup_sql(path, target):
    before = sql_fingerprint(path)
    if before is None:
        return None
    with readonly(path) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    target.chmod(0o600)
    if sql_fingerprint(target) != before:
        raise MigrationError("SQLite backup differs from source")
    return before


def records(client, name, *, vectors=True):
    offset = None
    while True:
        page, offset = client.scroll(name, limit=PAGE, offset=offset,
                                     with_payload=True, with_vectors=vectors)
        yield from page
        if offset is None:
            break


def payload_digest(rows):
    # Sort only fixed-size record digests, never the full corpus or vectors.
    hashes = [hashlib.sha256(canonical([str(row.id), row.payload or {}]).encode()).digest()
              for row in rows]
    return len(hashes), hashlib.sha256(b"".join(sorted(hashes))).hexdigest()


def collection_digest(client, name):
    return payload_digest(records(client, name, vectors=False))


def vector_digest(client, name):
    hashes = [hashlib.sha256(canonical([str(row.id),
              row.model_dump(mode="json")["vector"]]).encode()).digest()
              for row in records(client, name)]
    return hashlib.sha256(b"".join(sorted(hashes))).hexdigest()


def verify_copy(client, source, target):
    import math
    left, right = iter(records(client, source)), iter(records(client, target))
    for a, b in __import__("itertools").zip_longest(left, right):
        if a is None or b is None or a.id != b.id:
            raise MigrationError("Copied vector identifiers differ")
        av = a.vector.get("") if isinstance(a.vector, dict) else a.vector
        bv = b.vector.get("") if isinstance(b.vector, dict) else b.vector
        if len(av) != len(bv) or not all(math.isclose(x, y, abs_tol=1e-6, rel_tol=1e-6)
                                        for x, y in zip(av, bv, strict=True)):
            raise MigrationError("Copied dense vectors differ")
        ax = {k: v for k, v in a.vector.items() if k} if isinstance(a.vector, dict) else {}
        bx = {k: v for k, v in b.vector.items() if k} if isinstance(b.vector, dict) else {}
        if ax != bx:
            raise MigrationError("Copied keyword vectors differ")


def export_points(client, name, path):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as output:
        temporary.chmod(0o600)
        for row in records(client, name):
            output.write(row.model_dump_json() + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def exported(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield qm.Record.model_validate_json(line)


def schema(client, name):
    info = client.get_collection(name)
    params = info.config.params
    if not isinstance(params.vectors, qm.VectorParams):
        raise MigrationError("Migration requires unnamed dense text vectors")
    return {"vectors": params.vectors.model_dump(mode="json", exclude_none=True),
            "sparse": {key: value.model_dump(mode="json", exclude_none=True)
                       for key, value in (params.sparse_vectors or {}).items()},
            "metadata": info.config.metadata,
            "indexes": {key: (value.params.model_dump(mode="json", exclude_none=True)
                              if value.params is not None else value.data_type.value)
                        for key, value in info.payload_schema.items()}}


def create(client, name, profile, old_schema, kind):
    sparse = {key: qm.SparseVectorParams.model_validate(value)
              for key, value in old_schema.get("sparse", {}).items()}
    if kind in {"kb", "user_files"}:
        sparse["bm25"] = qm.SparseVectorParams(modifier=qm.Modifier.IDF)
    client.create_collection(name, vectors_config=qm.VectorParams(
        size=profile.dimensions, distance=qm.Distance.COSINE),
        sparse_vectors_config=sparse or None,
        metadata={**(old_schema.get("metadata") or {}), **profile.metadata})
    for key, value in old_schema.get("indexes", {}).items():
        client.create_payload_index(name, key, field_schema=value, wait=True)


def verify_profile(client, name, profile):
    info = client.get_collection(name)
    params = info.config.params.vectors
    if (not isinstance(params, qm.VectorParams) or params.size != profile.dimensions
            or params.distance != qm.Distance.COSINE
            or (info.config.metadata or {}).get("audrey_embedding") != profile.metadata["audrey_embedding"]):
        raise MigrationError("Collection does not match target embedding space")


def live_archive(path, cutoff):
    if not path.exists():
        return []
    with readonly(path) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT c.* FROM archive_chunks c
            WHERE NOT EXISTS (SELECT 1 FROM archive_deletion_outbox d WHERE d.chunk_id=c.chunk_id)
              AND NOT EXISTS (SELECT 1 FROM archive_user_purges p
                              WHERE p.user=c.user AND c.created_at<=p.cutoff_at)
              AND NOT EXISTS (SELECT 1 FROM archive_conversation_deletions d
                              WHERE d.user=c.user AND d.conversation_id=c.conversation_id
                                AND c.created_at<=d.cutoff_at)
            ORDER BY c.chunk_id
        """)
        output = []
        for row in rows:
            if cutoff and timestamp(row["created_at"]) < timestamp(cutoff):
                continue
            point = str(uuid.uuid5(uuid.NAMESPACE_URL, "chat_archive|" + row["chunk_id"]))
            output.append(qm.Record(id=point, payload={
                "user": row["user"], "conversation_id": row["conversation_id"],
                "chunk_id": row["chunk_id"], "message_ids": row["message_ids_json"].split(","),
                "created_at": row["created_at"], "text": row["text"]}))
        return output


def timestamp(value):
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError
        return date.astimezone(UTC)
    except (ValueError, AttributeError, TypeError) as error:
        raise MigrationError("A deletion/retention timestamp is invalid") from error


def policies(uploads, archive):
    deletions, purges = set(), {}
    if uploads.exists():
        with readonly(uploads) as db:
            deletions = set(db.execute("SELECT user,file_id FROM file_deletions"))
    if archive.exists():
        with readonly(archive) as db:
            purges = dict(db.execute("SELECT user,MAX(cutoff_at) FROM archive_user_purges GROUP BY user"))
    return deletions, purges


def source_rows(entry, state, archive, deletions, purges, cutoff):
    if entry["kind"] == "history":
        yield from live_archive(archive, cutoff)
        return
    for row in exported(state / entry["export"]):
        payload = row.payload or {}
        if entry["kind"] == "user_files" and (payload.get("user"), payload.get("file_id")) in deletions:
            continue
        if entry["kind"] == "memory" and payload.get("user") in purges:
            if timestamp(payload.get("updated_at")) <= timestamp(purges[payload["user"]]):
                continue
        yield row


def input_text(kind, payload):
    if kind == "memory":
        if not isinstance(payload.get("key"), str) or not isinstance(payload.get("value"), str):
            raise MigrationError("Memory is missing its retained text")
        tags = ",".join(tag for tag in str(payload.get("tags") or "").replace(",", " ").split()
                        if not tag.startswith("user:"))
        return f"{payload['key']}: {payload['value']}" + (f" [tags: {tags}]" if tags else "")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise MigrationError("A text point has no recoverable text")
    return text


def dump_state(state, journal):
    atomic_json(state / "manifest.json", journal)


def check_sources(journal, paths):
    if {key: sql_fingerprint(path) for key, path in paths.items()} != journal["sql"]:
        raise MigrationError("Source SQLite changed; do not resume an older migration")


async def migrate(client, embedder, config, *, state, uploads, archive, app_db,
                  memory="kb_memory", history="kb_chat_archive", retention_days=0,
                  progress=lambda _: None):
    profile = embedder.profile
    paths = {"uploads": uploads, "archive": archive, "app": app_db}
    manifest = state / "manifest.json"
    await embedder.verify_identity()
    if manifest.exists():
        journal = json.loads(manifest.read_text())
        if journal["profile"] != profile.metadata:
            raise MigrationError("Migration state belongs to another embedding profile")
        if journal.get("status") == "completed":
            for entry in journal["collections"]:
                verify_profile(client, entry["name"], profile)
            report = summary(journal, profile, resumed=True)
            for row, entry in zip(report["collections"], journal["collections"], strict=True):
                row["points"], row["payload_sha256"] = collection_digest(client, entry["name"])
            return report
        check_sources(journal, paths)
    else:
        names = {item.name for item in client.get_collections().collections}
        global_name = config.get("kb", {}).get("text_collection", "kb_text")
        selected = [(name, "user_files" if name.startswith("kb_user_text_") else "kb")
                    for name in sorted(names) if name == global_name or name.startswith("kb_user_text_")]
        selected += [(memory, "memory"), (history, "history")]
        if len({name for name, _ in selected}) != len(selected):
            raise MigrationError("Text collection roles overlap")
        if any(kind == "user_files" and client.count(name, exact=True).count for name, kind in selected) and not uploads.exists():
            raise MigrationError("Upload catalog is required for private-file deletion policy")
        if any(name in names and client.count(name, exact=True).count
               for name in (memory, history)) and not archive.exists():
            raise MigrationError("Archive source is required for memory/history deletion policy")
        journal = {"schema": 1, "status": "preparing", "profile": profile.metadata,
                   "created_at": datetime.now(UTC).isoformat(), "sql": {}, "collections": []}
        for key, path in paths.items():
            journal["sql"][key] = backup_sql(path, state / f"{key}.sqlite")
        journal["retention_cutoff"] = ((datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
                                          if retention_days else "")
        for index, (name, kind) in enumerate(selected):
            exists = name in names
            identity = hashlib.sha256((profile.identity + name).encode()).hexdigest()[:32]
            entry = {"name": name, "kind": kind, "stage": "audrey_embedding_stage_" + identity,
                     "export": f"collection-{index}.jsonl.gz", "phase": "exported",
                     "schema": schema(client, name) if exists else {"sparse": {}, "indexes": {}},
                     "original_exists": exists}
            if exists:
                export_points(client, name, state / entry["export"])
                entry["original_digest"] = list(collection_digest(client, name))
                if list(payload_digest(exported(state / entry["export"]))) != entry["original_digest"]:
                    raise MigrationError("Private vector export did not verify")
            else:
                with gzip.open(state / entry["export"], "wt"):
                    pass
                (state / entry["export"]).chmod(0o600)
                entry["original_digest"] = [0, hashlib.sha256(b"").hexdigest()]
            journal["collections"].append(entry)
        dump_state(state, journal)
    deletions, purges = policies(uploads, archive)
    cutoff = journal["retention_cutoff"]
    for ordinal, entry in enumerate(journal["collections"], 1):
        if entry["phase"] in {"staged", "cutover_started", "complete"}:
            verify_profile(client, entry["stage"], profile)
            if (list(collection_digest(client, entry["stage"])) != entry["expected"]
                    or vector_digest(client, entry["stage"]) != entry["vector_digest"]):
                raise MigrationError("Verified staging collection changed")
            continue
        stage = entry["stage"]
        if client.collection_exists(stage):
            client.delete_collection(stage)
        create(client, stage, profile, entry["schema"], entry["kind"])
        verify_profile(client, stage, profile)
        batch = []
        for row in source_rows(entry, state, archive, deletions, purges, cutoff):
            batch.append(row)
            if len(batch) == 16:
                await write_embedded(client, embedder, stage, entry["kind"], batch)
                batch = []
        if batch:
            await write_embedded(client, embedder, stage, entry["kind"], batch)
        expected = list(payload_digest(source_rows(entry, state, archive, deletions, purges, cutoff)))
        if list(collection_digest(client, stage)) != expected:
            raise MigrationError("Re-embedded payloads did not verify")
        entry.update(phase="staged", expected=expected, vector_digest=vector_digest(client, stage))
        dump_state(state, journal)
        progress({"event": "collection_staged", "role": entry["kind"], "ordinal": ordinal,
                  "points": expected[0]})
    # No production collection changes before every replacement has verified.
    check_sources(journal, paths)
    await embedder.verify_identity()
    for entry in journal["collections"]:
        if entry["phase"] == "complete":
            verify_copy(client, entry["stage"], entry["name"])
            verify_profile(client, entry["name"], profile)
            if list(collection_digest(client, entry["name"])) != entry["expected"]:
                raise MigrationError("Completed target changed while writers were stopped")
            continue
        if entry["phase"] == "staged":
            if entry["original_exists"]:
                if list(collection_digest(client, entry["name"])) != entry["original_digest"]:
                    raise MigrationError("Original text collection changed during migration")
            elif client.collection_exists(entry["name"]):
                raise MigrationError("A new collection appeared during migration")
            entry["phase"] = "cutover_started"
            dump_state(state, journal)
        if client.collection_exists(entry["name"]):
            client.delete_collection(entry["name"])
        create(client, entry["name"], profile, entry["schema"], entry["kind"])
        batch = []
        for row in records(client, entry["stage"]):
            batch.append(qm.PointStruct(id=row.id, vector=row.vector, payload=row.payload))
            if len(batch) == PAGE:
                client.upsert(entry["name"], points=batch, wait=True)
                batch = []
        if batch:
            client.upsert(entry["name"], points=batch, wait=True)
        verify_profile(client, entry["name"], profile)
        if list(collection_digest(client, entry["name"])) != entry["expected"]:
            raise MigrationError("Published text collection failed verification")
        verify_copy(client, entry["stage"], entry["name"])
        entry["phase"] = "complete"
        dump_state(state, journal)
    journal["status"] = "completed"
    dump_state(state, journal)
    return summary(journal, profile)


async def write_embedded(client, embedder, name, kind, rows):
    vectors = await embedder.embed_many([input_text(kind, row.payload or {}) for row in rows])
    points = []
    for row, dense in zip(rows, vectors, strict=True):
        vector = dense
        if kind in {"kb", "user_files"}:
            vector = dict(row.vector) if isinstance(row.vector, dict) else {}
            vector[""] = dense
            indices, values = bm25.document_vector(row.payload["text"])
            vector["bm25"] = qm.SparseVector(indices=indices, values=values)
        points.append(qm.PointStruct(id=row.id, payload=row.payload, vector=vector))
    client.upsert(name, points=points, wait=True)


def summary(journal, profile, *, resumed=False):
    return {"schema": 1, "status": "completed", "resumed_or_already_complete": resumed,
            "embedding": {**profile.provenance, "embedding_identity": profile.identity},
            "collections": [{"role": entry["kind"], "points": entry["expected"][0],
                             "payload_sha256": entry["expected"][1], "verified": True}
                            for entry in journal["collections"]],
            "private_backups_retained": True, "source_sqlite_unchanged": True,
            "images_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "uploads-db", "archive-db", "app-db", "state", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--maintenance-confirmed", action="store_true")
    parser.add_argument("--memory-collection", default=os.getenv("MEMORY_COLLECTION", "kb_memory"))
    parser.add_argument("--archive-collection", default=os.getenv("CHAT_ARCHIVE_COLLECTION", "kb_chat_archive"))
    parser.add_argument("--retention-days", type=int, default=int(os.getenv("CHAT_ARCHIVE_RETENTION_DAYS", "0")))
    args = parser.parse_args(argv)
    previous_umask = os.umask(0o077)
    report, code = {"schema": 1, "status": "setup_error"}, 2
    report_allowed = False
    stage = "setup"
    failure = None
    try:
        if not args.maintenance_confirmed or args.retention_days < 0:
            raise MigrationError("Stopped writers must be confirmed")
        if not args.state.is_absolute() or args.state.is_symlink():
            raise MigrationError("Private migration directory must be absolute and not a symlink")
        args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.state.chmod(0o700)
        if not args.report.resolve().is_relative_to(args.state.resolve()) or args.report.is_symlink():
            raise MigrationError("Report must be inside private migration state")
        report_allowed = True
        stage = "configuration"
        config = yaml.safe_load(args.config.read_text())
        load_profile(args.config)
        with (args.state / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            async def run():
                nonlocal stage, code
                client = ollama = None
                primary_error = None
                try:
                    stage = "qdrant_client"
                    client = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"),
                                          port=int(os.getenv("QDRANT_PORT", "6333")))
                    stage = "ollama_client"
                    ollama = OllamaClient(os.getenv("OLLAMA_HOST", "http://ollama:11434"))
                    stage = "migration"
                    code = 1
                    return await migrate(client, text_embedder(ollama, config), config,
                        state=args.state, uploads=args.uploads_db, archive=args.archive_db,
                        app_db=args.app_db, memory=args.memory_collection,
                        history=args.archive_collection, retention_days=args.retention_days,
                        progress=lambda event: print(json.dumps(event), flush=True))
                except BaseException as error:
                    primary_error = error
                    raise
                finally:
                    cleanup_error = None
                    if ollama is not None:
                        try:
                            await ollama.aclose()
                        except BaseException as error:  # noqa: BLE001 — still close Qdrant
                            cleanup_error = error
                    if client is not None:
                        try:
                            client.close()
                        except BaseException as error:  # noqa: BLE001 — preserve the primary failure
                            if cleanup_error is None:
                                cleanup_error = error
                    if cleanup_error is not None:
                        if primary_error is not None:
                            primary_error.add_note(
                                f"Client cleanup also failed: {type(cleanup_error).__name__}")
                        else:
                            stage = "cleanup"
                            raise cleanup_error
            report = asyncio.run(run())
            code = 0
    except (MigrationError, ValueError, OSError, sqlite3.Error) as error:
        failure = error
        report = {"schema": 1, "status": "failed", "error_kind": type(error).__name__,
                  "writers_must_remain_stopped": True}
        if isinstance(error, MigrationError):
            report["error"] = str(error)
    except Exception as error:  # noqa: BLE001 — private details stay off Telegram
        failure = error
        code = 3
        report = {"schema": 1, "status": "failed", "error_kind": type(error).__name__,
                  "writers_must_remain_stopped": True}
    if failure is not None:
        report["stage"] = stage
        if report_allowed:
            try:
                report["private_diagnostic"] = private_diagnostic(args.state, failure)
            except OSError:
                report["private_diagnostic_saved"] = False
    report["exit_code"] = code
    try:
        if report_allowed:
            atomic_json(args.report, report)
    except OSError:
        os.umask(previous_umask)
        return 3
    print(json.dumps(report, indent=2), flush=True)
    os.umask(previous_umask)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
