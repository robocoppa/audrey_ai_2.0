"""Private file contracts survive upload handovers and exclude legacy workers."""

from __future__ import annotations

import sqlite3

import pytest

from audrey.kb.uploads_db import UploadsDB

USER = "alice@example.com"
OTHER = "bob@example.com"
NOW = "2026-10-19T12:00:00+00:00"
OLD = "2026-10-19T10:00:00+00:00"
MAX_BYTES = 100


@pytest.fixture
def db(tmp_path):
    catalog = UploadsDB(tmp_path / "uploads.sqlite")
    yield catalog
    catalog.close()


async def _reserve(db, kind, identifier, *, private=True, user=USER, size=20):
    kwargs = dict(
        user=user, max_user_bytes=MAX_BYTES, now=NOW,
        expired_before=OLD, private_required=private,
    )
    if kind == "single_shot":
        return await db.reserve_single_upload(
            reservation_id=identifier, bytes_=size, **kwargs,
        )
    return await db.reserve_upload_session(
        upload_id=identifier, filename="evidence.pdf", total_bytes=size,
        part_size=10, parts_total=2, **kwargs,
    )


async def _commit(db, identifier, *, reservation_kind="single_shot", user=USER,
                  status="pending", size=20, file_id=None):
    return await db.commit_reserved_upload(
        reservation_id=identifier, reservation_kind=reservation_kind,
        user=user, file_id=file_id or identifier, filename="evidence.pdf",
        mime="application/pdf", bytes_=size, kind="text", collection="",
        chunks=0, uploaded_at=NOW, status=status, max_user_bytes=MAX_BYTES,
    )


async def _private_upload(db, identifier="protected", *, status="pending", user=USER):
    assert (await _reserve(db, "single_shot", identifier, user=user)).accepted
    assert (await _commit(db, identifier, user=user, status=status)).accepted


async def test_legacy_catalog_upgrade_is_additive_idempotent_and_ordinary(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as old:
        old.executescript("""
            CREATE TABLE uploads (
                file_id TEXT PRIMARY KEY, user TEXT NOT NULL,
                filename TEXT NOT NULL, mime TEXT NOT NULL, bytes INTEGER NOT NULL,
                kind TEXT NOT NULL, collection TEXT NOT NULL, chunks INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL
            );
            CREATE TABLE upload_sessions (
                upload_id TEXT PRIMARY KEY, user TEXT NOT NULL,
                filename TEXT NOT NULL, total_bytes INTEGER NOT NULL,
                part_size INTEGER NOT NULL, parts_total INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE storage_reservations (
                reservation_id TEXT PRIMARY KEY, user TEXT NOT NULL,
                kind TEXT NOT NULL, bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
        """)
        old.execute(
            "INSERT INTO uploads VALUES ('old-file', ?, 'old.txt', 'text/plain', "
            "10, 'text', 'legacy', 2, ?)", (USER, NOW),
        )
        old.execute(
            "INSERT INTO upload_sessions VALUES "
            "('old-session', ?, 'parts.pdf', 20, 10, 2, ?, ?)", (USER, NOW, NOW),
        )
        old.execute(
            "INSERT INTO storage_reservations VALUES "
            "('old-reservation', ?, 'single_shot', 30, ?, ?)", (USER, NOW, NOW),
        )
    for _ in range(2):
        upgraded = UploadsDB(path)
        try:
            row = await upgraded.get_upload("old-file")
            assert (row["private_required"], row["bytes"], row["collection"], row["chunks"]) == (
                0, 10, "legacy", 2,
            )
            assert (await upgraded.get_session("old-session", user=USER))["private_required"] == 0
            assert not await upgraded.reservation_private_required(
                reservation_id="old-reservation", reservation_kind="single_shot", user=USER,
            )
            assert (await upgraded.quota_usage(USER)).total_bytes == 60
            assert await upgraded.private_file_ids(USER) == set()
        finally:
            upgraded.close()


@pytest.mark.parametrize("kind", ["single_shot", "chunked"])
@pytest.mark.parametrize("private", [False, True])
async def test_accepted_contract_survives_restart_and_atomic_commit(tmp_path, kind, private):
    path = tmp_path / "uploads.sqlite"
    first = UploadsDB(path)
    try:
        assert (await _reserve(first, kind, "reservation", private=private)).accepted
        if kind == "chunked":
            await first.record_reserved_part(
                upload_id="reservation", user=USER, part_no=0, bytes_=10, now=NOW,
            )
            await first.record_reserved_part(
                upload_id="reservation", user=USER, part_no=1, bytes_=10, now=NOW,
            )
    finally:
        first.close()
    restarted = UploadsDB(path)
    try:
        assert await restarted.reservation_private_required(
            reservation_id="reservation", reservation_kind=kind, user=USER,
        ) is private
        assert (await _commit(restarted, "reservation", reservation_kind=kind)).accepted
        assert (await restarted.get_upload("reservation"))["private_required"] == int(private)
        # Completed retries cannot overwrite the accepted flag or duplicate quota.
        assert (await _commit(restarted, "reservation", reservation_kind=kind)).accepted
        assert (await restarted.quota_usage(USER)).total_bytes == 20
        assert await restarted.private_file_ids(USER) == ({"reservation"} if private else set())
        with pytest.raises(ValueError, match="missing or not owned"):
            await restarted.reservation_private_required(
                reservation_id="reservation", reservation_kind=kind, user=USER,
            )
        assert (await restarted.list_user(USER))[0]["private_required"] == int(private)
    finally:
        restarted.close()


@pytest.mark.parametrize("kind", ["single_shot", "chunked"])
@pytest.mark.parametrize("private", [False, True])
async def test_retry_cannot_change_accepted_privacy_or_take_another_owner(db, kind, private):
    assert (await _reserve(db, kind, "reservation", private=private)).accepted
    with pytest.raises(ValueError, match="restriction is immutable"):
        await _reserve(db, kind, "reservation", private=not private, size=30)
    with pytest.raises(ValueError, match="another"):
        await _reserve(db, kind, "reservation", private=private, user=OTHER)
    with pytest.raises(ValueError, match="missing or not owned"):
        await db.reservation_private_required(
            reservation_id="reservation", reservation_kind=kind, user=OTHER,
        )
    with pytest.raises(ValueError, match="missing or not owned"):
        await _commit(db, "reservation", reservation_kind=kind, user=OTHER)
    assert await db.reservation_private_required(
        reservation_id="reservation", reservation_kind=kind, user=USER,
    ) is private
    assert (await db.quota_usage(USER)).total_bytes == 20
    assert (await db.quota_usage(OTHER)).total_bytes == 0


@pytest.mark.parametrize("kind", ["single_shot", "chunked"])
@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
async def test_creation_requires_real_boolean_without_accepting_bytes(db, kind, value):
    with pytest.raises(ValueError, match="must be a boolean"):
        await _reserve(db, kind, "bad", private=value)
    assert (await db.quota_usage(USER)).total_bytes == 0


@pytest.mark.parametrize("kind", ["single_shot", "chunked"])
async def test_quota_refusal_retains_accepted_private_reservation(db, kind):
    assert (await _reserve(db, kind, "reservation")).accepted
    # Actual bytes may exceed the original declared amount; rejecting the
    # commit must retain its contract and held quota for retry or cleanup.
    assert not (await _commit(db, "reservation", reservation_kind=kind, size=101)).accepted
    assert await db.reservation_private_required(
        reservation_id="reservation", reservation_kind=kind, user=USER,
    ) is True
    assert await db.get_upload("reservation") is None
    assert (await db.quota_usage(USER)).total_bytes == 20


@pytest.mark.parametrize("table,key", [
    ("uploads", "file_id"), ("upload_sessions", "upload_id"),
    ("storage_reservations", "reservation_id"),
])
@pytest.mark.parametrize("private", [False, True])
async def test_sql_guards_reject_relabelling_and_non_boolean_insert(db, table, key, private):
    if table == "uploads":
        await _reserve(db, "single_shot", "source", private=private)
        await _commit(db, "source")
    else:
        kind = "chunked" if table == "upload_sessions" else "single_shot"
        await _reserve(db, kind, "source", private=private)
    with pytest.raises(sqlite3.IntegrityError, match="restriction is immutable"):
        db._conn.execute(
            f"UPDATE {table} SET private_required = ? WHERE {key} = ?",  # noqa: S608
            (int(not private), "source"),
        )
    row = dict(db._conn.execute(f"SELECT * FROM {table}").fetchone())  # noqa: S608
    row[key] = "invalid"
    row["private_required"] = 2
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        db._conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",  # noqa: S608
            tuple(row.values()),
        )


async def test_legacy_workers_skip_protected_rows_without_starving_ordinary_jobs(db):
    await _private_upload(db)
    # Pending protected rows have an older timestamp but never leave the queue
    # for an unprotected worker. Fetch exclusion also covers persisted rows.
    db._conn.execute("UPDATE uploads SET uploaded_at = ? WHERE file_id = 'protected'", (OLD,))
    await _private_upload(db, "protected-fetch", status="fetch_pending")
    await db.record_upload(
        file_id="ordinary", user=USER, filename="video.mp4", mime="video/mp4",
        bytes_=10, kind="video", collection="", chunks=0, uploaded_at=NOW, status="pending",
    )
    await db.record_url_fetch(
        file_id="ordinary-fetch", user=USER, source_url="https://example.com/video",
        filename="video", uploaded_at=NOW,
    )
    assert (await db.claim_job(lease_id="worker", now=NOW))["file_id"] == "ordinary"
    assert (await db.claim_fetch(lease_id="fetcher", now=NOW))["file_id"] == "ordinary-fetch"
    assert await db.claim_job(lease_id="worker-2", now=NOW) is None
    assert await db.claim_fetch(lease_id="fetcher-2", now=NOW) is None
    for identifier, status in [("protected", "pending"), ("protected-fetch", "fetch_pending")]:
        row = await db.get_upload(identifier)
        assert (row["private_required"], row["status"], row["attempts"], row["lease_id"]) == (
            1, status, 0, "",
        )


async def test_legacy_metadata_handover_and_index_recovery_cannot_rewrite_private_file(db):
    await _private_upload(db, status="ready")
    before = await db.get_upload("protected")
    for owner in [USER, OTHER]:
        with pytest.raises(ValueError, match="cannot overwrite a private file"):
            await db.record_upload(
                file_id="protected", user=owner, filename="replacement.mp4", mime="video/mp4",
                bytes_=90, kind="video", collection="new", chunks=99, uploaded_at=OLD,
            )
    payload = dict(before, mime="video/mp4", kind="video", private_required=False)
    assert not await db.backfill_upload(payload, user=USER, collection="new")
    assert await db.get_upload("protected") == before
    # Missing legacy rows are ordinary; arbitrary index metadata cannot grant
    # an unsupported guarantee or retroactively change accepted processing.
    assert await db.backfill_upload(dict(payload, file_id="legacy"), user=USER, collection="old")
    assert (await db.get_upload("legacy"))["private_required"] == 0


async def test_private_retries_sweeps_and_stale_results_preserve_contract(db):
    await _private_upload(db, status="processing")
    db._conn.execute(
        "UPDATE uploads SET lease_id='old-lease', leased_at=?, attempts=1, "
        "summary='existing summary' WHERE file_id='protected'", (OLD,),
    )
    assert not await db.complete_job(
        file_id="protected", lease_id="old-lease", collection="ordinary-index", chunks=4,
        summary="unprotected result", completed_at=NOW,
    )
    assert (await db.get_upload("protected"))["summary"] == "existing summary"
    assert await db.fail_job(file_id="protected", lease_id="old-lease", reason="retry")
    assert await db.requeue_job("protected", bytes_=21)
    assert await db.claim_job(lease_id="new-lease", now=NOW) is None
    row = await db.get_upload("protected")
    assert (row["private_required"], row["bytes"], row["status"]) == (1, 21, "pending")
    db._conn.execute(
        "UPDATE uploads SET status='processing', lease_id='expired', leased_at=?, attempts=1 "
        "WHERE file_id='protected'", (OLD,),
    )
    await db.sweep_expired_leases(expired_before=NOW, max_attempts=3)
    assert (await db.get_upload("protected"))["private_required"] == 1
    assert await db.claim_job(lease_id="newer-lease", now=NOW) is None


async def test_private_originals_never_enter_legacy_reclamation(db):
    await _private_upload(db, status="ready")
    db._conn.execute(
        "UPDATE uploads SET kind='video', completed_at=? WHERE file_id='protected'", (OLD,),
    )
    assert await db.reclaimable_sources(completed_before=NOW) == []
    assert (await db.get_upload("protected"))["source_freed_at"] == ""


async def test_private_url_mutations_refuse_and_owner_exclusions_include_tombstones(db):
    await _private_upload(db, status="fetching")
    await _private_upload(db, "other-private", user=OTHER)
    db._conn.execute(
        "UPDATE uploads SET source_url='https://example.com/video', lease_id='fetch-lease', "
        "source_freed_at=? WHERE file_id='protected'", (OLD,),
    )
    before = await db.get_upload("protected")
    with pytest.raises(ValueError, match="private URL fetching is unavailable"):
        await db.reserve_url_fetch(
            file_id="protected", user=USER, source_url="https://example.com/video",
            filename="video", ceiling_bytes=10, max_user_bytes=MAX_BYTES,
            uploaded_at=NOW, expired_before=OLD,
        )
    with pytest.raises(ValueError, match="private URL fetching is unavailable"):
        await db.reserve_url_refetch(
            file_id="protected", user=USER, ceiling_bytes=10,
            max_user_bytes=MAX_BYTES, expired_before=OLD,
        )
    with pytest.raises(ValueError, match="private URL fetching is unavailable"):
        await db.record_url_fetch(
            file_id="protected", user=USER, source_url="https://example.com/video",
            filename="video", uploaded_at=NOW,
        )
    with pytest.raises(ValueError, match="private URL fetching is unavailable"):
        await db.complete_fetch(
            file_id="protected", lease_id="fetch-lease", filename="new.mp4",
            mime="video/mp4", bytes_=99, transcript="unexpected remote data",
        )
    assert await db.get_upload("protected") == before
    await db.request_file_deletion("protected", user=USER, requested_at=NOW)
    assert await db.private_file_ids(USER) == {"protected"}
    assert await db.private_file_ids(OTHER) == {"other-private"}
    assert await db.list_user(USER) == []


async def test_reservation_lookup_refuses_unknown_missing_and_wrong_lifecycle(db):
    await _reserve(db, "single_shot", "single")
    await _reserve(db, "chunked", "chunked")
    for identifier, kind in [("single", "chunked"), ("chunked", "single_shot"),
                             ("missing", "single_shot")]:
        with pytest.raises(ValueError, match="missing or not owned"):
            await db.reservation_private_required(
                reservation_id=identifier, reservation_kind=kind, user=USER,
            )
    with pytest.raises(ValueError, match="unknown reservation kind"):
        await db.reservation_private_required(
            reservation_id="single", reservation_kind="url_fetch", user=USER,
        )
    assert (await db.quota_usage(USER)).total_bytes == 40
