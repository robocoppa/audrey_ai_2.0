"""Durable watch ownership, scheduling, atomic baselines and notices."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from audrey.app_state import ApplicationStore
from audrey.app_state import store as store_module
from audrey.app_state.migrations import MIGRATIONS
from audrey.app_state.watches import (
    PROJECT_FILE_TEXT_CRITERION,
    WatchAccessError,
    WatchConflictError,
    WatchError,
    WatchLimitError,
)
from audrey.processing_policy import ProcessingPolicy, processing_scope


class Clock:
    now = dt.datetime(2026, 10, 30, 12, tzinfo=dt.UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += dt.timedelta(seconds=seconds)


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    users = []
    for name in ("alice", "bob"):
        users.append(await store.resolve_external_identity(
            provider="test", subject=name, email=f"{name}@example.com", display_name=name,
            role="user", auth_method="test",
        ))
    clock = Clock()
    store.watches._clock = clock
    store.agents._clock = clock
    yield SimpleNamespace(store=store, owner=users[0], other=users[1], clock=clock)
    store.close()


async def _create(state, **changes):
    values = dict(user_id=state.owner.user_id, title="Release notes", url="https://example.com/releases",
                  criterion="Tell me about new stable releases.", request_id="watch-1",
                  schedule={"kind": "interval", "timezone": "UTC", "interval_minutes": 60})
    values.update(changes)
    return await state.store.watches.create(**values)


def _key(state, watch, **changes):
    values = dict(user_id=state.owner.user_id, watch_id=watch["id"], expected_revision=watch["revision"])
    values.update(changes)
    return values


def _lease(run):
    return {key: run[key] for key in ("run_id", "lease_token", "fence")}


async def _claim(state, watch):
    await state.store.watches.run_now(**_key(state, watch))
    return await state.store.watches.claim()


async def _complete(state, run, text="Version one.", outcome="baseline", significant=None, **changes):
    await state.store.watches.reserve_operation(**_lease(run), kind="fetch")
    if outcome == "changed":
        await state.store.watches.reserve_operation(**_lease(run), kind="model")
    values = dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest(), outcome=outcome,
                  significant=significant, report="Saved comparison.", diff='[{"old":"one","new":"two"}]' if outcome == "changed" else "", final_url=run["url"],
                  extraction_version="public-text-v1")
    values.update(changes)
    return await state.store.watches.complete(**_lease(run), **values)


async def test_create_does_not_dispatch_idempotency_survives_edit_and_reopen(state):
    watch = await _create(state)
    assert watch["next_run_at"] == "2026-10-30T13:00:00+00:00"
    assert watch["latest_run"] is None and watch["active_run_id"] is None
    assert await state.store.watches.claim() is None
    edited = await state.store.watches.update(**_key(state, watch), title="Stable releases")
    assert (await _create(state))["title"] == edited["title"]
    with pytest.raises(WatchConflictError):
        await _create(state, criterion="Other criterion")
    second = ApplicationStore(state.store.path)
    assert (await second.watches.get(user_id=state.owner.user_id, watch_id=watch["id"]))["title"] == "Stable releases"
    second.close()


async def test_project_file_text_retains_only_a_live_attachment_fingerprint(state):
    repository = state.store.watches
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Delivery plans")
    scope = {"user_id": state.owner.user_id, "project_id": project.project_id}
    source = {
        "url": None, "criterion": None, "source_kind": "project_files",
        "source_filename": "Delivery plan.md", "project_id": project.project_id, "enabled": False,
    }
    watch = await _create(state, **source)
    assert watch["source_filename"] == source["source_filename"]
    assert watch["criterion"] == PROJECT_FILE_TEXT_CRITERION
    assert (await _create(state, **source))["id"] == watch["id"]
    with pytest.raises(WatchConflictError):
        await _create(state, **{**source, "source_filename": "Other plan.md"})
    previous_ids = []

    @contextmanager
    def source_guard():
        def check(previous_file_id):
            assert state.store._conn.in_transaction
            previous_ids.append(previous_file_id)
            return previous_file_id is not None

        yield check

    async def finish(run, file_id, content):
        assert run["source_filename"] == source["source_filename"] and run["baseline"] is None
        return await repository.complete_project_file_text(
            **_lease(run), file_id=file_id, digest=hashlib.sha256(content.encode()).hexdigest(),
            source_guard=source_guard,
        )

    first, second = "attachment-first", "attachment-second"
    await state.store.projects.add_file(**scope, file_id=first)
    initial = await _claim(state, watch)
    with pytest.raises(WatchError, match="requested Project file source"):
        await repository.complete_project_files(**_lease(initial))
    baseline = await finish(initial, first, "Tuesday")
    assert baseline["outcome"] == "baseline"
    assert (await finish(await _claim(state, watch), first, "Tuesday"))["outcome"] == "unchanged"
    await state.store.projects.add_file(**scope, file_id=second)
    changed = await finish(await _claim(state, watch), second, "Wednesday")
    assert changed["outcome"] == "changed" and changed["significant"] is True
    notice, = await repository.list_notices(user_id=state.owner.user_id)
    assert notice["run_id"] == changed["id"] and notice["body"] == "The saved file text changed."
    assert previous_ids == [None, first, first]

    assert await state.store.projects.remove_file(**scope, file_id=second)
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_watch_file_text_baselines").fetchone()[0] == 0
    stale = await _claim(state, watch)
    with pytest.raises(WatchAccessError, match="unavailable"):
        await finish(stale, second, "Wednesday")
    assert (await finish(stale, first, "Tuesday"))["outcome"] == "baseline"
    assert previous_ids[-1] is None
    assert len(await repository.list_notices(user_id=state.owner.user_id)) == 1

    for saved in state.store._conn.execute("SELECT record_json FROM app_watch_snapshots"):
        assert json.loads(saved[0]).keys() == {"run_id", "source_kind", "extraction_version"}
    assert state.store._conn.execute(
        "SELECT COUNT(*) FROM app_watch_runs WHERE baseline_json IS NOT NULL",
    ).fetchone()[0] == 0
    with pytest.raises(WatchAccessError):
        await repository.get_snapshot_page(
            user_id=state.owner.user_id, watch_id=watch["id"], run_id=changed["id"], offset=0, limit=100,
        )


async def _private_file_watch(state, **changes):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Protected Project", private_required=True)
    values = {"url": None, "criterion": None, "source_kind": "project_files", "private_required": True,
              "source_filename": "Confidential plan.md", "project_id": project.project_id, "enabled": False}
    values.update(changes)
    return project, await _create(state, **values)


async def _private_attachment(state, project_id, file_id):
    with processing_scope(ProcessingPolicy(local_models={"qwen3.8:latest": "a" * 64}, ollama_base_url="http://localhost:11434")):
        await state.store.projects.add_file(user_id=state.owner.user_id, project_id=project_id,
                                             file_id=file_id, file_is_private=True)


@pytest.mark.parametrize("value", [0, 1, None, "true", []])
async def test_private_watch_mode_is_a_strict_boolean(state, value):
    with pytest.raises(WatchError, match="privacy scope"):
        await _create(state, private_required=value)


@pytest.mark.parametrize("changes", [
    {"source_kind": "public_page"}, {"source_filename": None}, {"project_id": None},
])
async def test_private_watch_only_accepts_explicit_project_filename_scope(state, changes):
    with pytest.raises(WatchError):
        await _private_file_watch(state, **changes)


async def test_private_project_and_file_watch_modes_must_match_canonical_scope(state):
    project, watch = await _private_file_watch(state)
    ordinary = await state.store.projects.create(user_id=state.owner.user_id, name="Ordinary")
    with pytest.raises(WatchAccessError):
        await _create(state, request_id="ordinary-in-private", project_id=project.project_id)
    with pytest.raises(WatchAccessError):
        await _create(state, request_id="private-in-ordinary", url=None, criterion=None,
                      source_kind="project_files", source_filename="plan.md", project_id=ordinary.project_id,
                      private_required=True)
    with pytest.raises(WatchAccessError):
        await state.store.watches.list(user_id=state.other.user_id, project_id=project.project_id)
    assert (await state.store.watches.list(user_id=state.owner.user_id, project_id=project.project_id))[0]["id"] == watch["id"]
    assert (await state.store.watches.list(user_id=state.owner.user_id))[0]["private_required"] is True


async def test_private_watch_pins_mode_and_idempotency_without_widening_ordinary_metadata(state):
    project, watch = await _private_file_watch(state)
    assert watch["private_required"] is True and watch["project"]["private_required"] is True
    assert watch["project"]["name"] == "Protected Project"
    run = await _claim(state, watch)
    assert run["private_required"] is True and run["definition"]["private_required"] is True
    assert run["definition"]["project"]["private_required"] is True
    assert json.loads(state.store._conn.execute("SELECT definition_json FROM app_watch_runs").fetchone()[0])["private_required"] is True
    replay = await _create(state, url=None, criterion=None, source_kind="project_files", private_required=True,
                           source_filename="Confidential plan.md", project_id=project.project_id, enabled=False)
    assert replay["id"] == watch["id"]
    with pytest.raises(WatchConflictError):
        await _create(state, url=None, criterion=None, source_kind="project_files", private_required=False,
                      source_filename="Confidential plan.md", project_id=project.project_id, enabled=False)
    with pytest.raises(sqlite3.IntegrityError, match="privacy scope is immutable"):
        with state.store._lock, state.store._conn:
            state.store._conn.execute("UPDATE app_watches SET private_required=0 WHERE watch_id=?", (watch["id"],))
    with pytest.raises(WatchError, match="original Project"):
        await state.store.watches.update(**_key(state, watch), project_id=None)
    ordinary = await _create(state, request_id="ordinary")
    assert "private_required" not in ordinary


async def test_private_file_baseline_and_activity_retain_no_source_text_or_filename(state):
    project, watch = await _private_file_watch(state, title="Owner supplied secret title")
    await _private_attachment(state, project.project_id, "private-file")

    @contextmanager
    def source_guard():
        yield lambda previous: previous is not None

    async def finish(text):
        run = await _claim(state, watch)
        return await state.store.watches.complete_project_file_text(
            **_lease(run), file_id="private-file", digest=hashlib.sha256(text.encode()).hexdigest(), source_guard=source_guard,
        )

    assert (await finish("Private source material one"))["outcome"] == "baseline"
    assert (await finish("Private source material two"))["outcome"] == "changed"
    changed = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"][0]
    assert changed["title"] == "Private file watch" and changed["body"] == "The saved Private file text changed."
    assert "Confidential" not in repr(changed) and "secret title" not in repr(changed)
    run = await _claim(state, watch)
    await state.store.watches.fail(**_lease(run), error="Parser details for Confidential plan.md")
    notices = (await state.store.notifications.feed(user_id=state.owner.user_id))["items"]
    assert {notice["kind"] for notice in notices} == {"failure", "change"}
    assert all(notice["title"] == "Private file watch" and "Confidential" not in notice["body"] for notice in notices)
    assert state.store._conn.execute("SELECT baseline_json FROM app_watch_runs WHERE run_id=?", (run["id"],)).fetchone()[0] is None
    for row in state.store._conn.execute("SELECT record_json FROM app_watch_snapshots"):
        assert "Private source material" not in row[0] and "Confidential" not in row[0]
    assert await state.store.projects.remove_file(user_id=state.owner.user_id, project_id=project.project_id, file_id="private-file")
    assert state.store._conn.execute("SELECT COUNT(*) FROM app_watch_file_text_baselines").fetchone()[0] == 0


async def test_private_project_deletion_fences_execution_and_pauses_future_work_with_history_retained(state):
    project, watch = await _private_file_watch(state, enabled=True)
    run = await _claim(state, watch)
    await state.store.projects.delete(user_id=state.owner.user_id, project_id=project.project_id)
    for operation in (state.store.watches.validate_lease, state.store.watches.heartbeat):
        with pytest.raises(WatchAccessError):
            await operation(**_lease(run))
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_execution(**_lease(run))
    with pytest.raises(WatchConflictError):
        state.store.watches.fence_response(user_id=state.owner.user_id, watches=[watch])
    with pytest.raises(WatchAccessError):
        await state.store.watches.run_now(**_key(state, watch))
    assert await state.store.watches.claim() is None
    saved = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    assert saved["private_required"] is True and not saved["enabled"] and saved["next_run_at"] is None
    assert saved["project"] == {"id": project.project_id, "name": None, "available": False, "private_required": True}
    assert saved["latest_run"]["status"] == "cancelled"
    await state.store.purge_local_user_data(user_id=state.owner.user_id)
    assert await state.store.watches.list(user_id=state.owner.user_id) == []


async def test_bot_membership_revokes_private_lease_but_failure_cleanup_stays_usable(state):
    _, watch = await _private_file_watch(state)
    run = await _claim(state, watch)
    with state.store._lock, state.store._conn:
        state.store._conn.execute("INSERT INTO user_group_memberships(user_id,group_id,granted_at) VALUES (?,'bots',?)",
                                  (state.owner.user_id, state.clock().isoformat()))
    with pytest.raises(WatchAccessError, match="account access"):
        await state.store.watches.heartbeat(**_lease(run))
    failed = await state.store.watches.fail(**_lease(run), error="Access was revoked.")
    assert failed["status"] == "failed"


async def test_schema48_preserves_ordinary_file_baseline_and_queued_run(state):
    project = await state.store.projects.create(user_id=state.owner.user_id, name="Ordinary")
    await state.store.projects.add_file(user_id=state.owner.user_id, project_id=project.project_id, file_id="ordinary-file")
    watch = await _create(state, url=None, criterion=None, source_kind="project_files", source_filename="plan.md",
                           project_id=project.project_id, enabled=False)
    first = await _claim(state, watch)

    @contextmanager
    def source_guard():
        yield lambda previous: previous is not None

    digest = hashlib.sha256(b"Existing ordinary text.").hexdigest()
    baseline = await state.store.watches.complete_project_file_text(
        **_lease(first), file_id="ordinary-file", digest=digest, source_guard=source_guard,
    )
    queued = await state.store.watches.run_now(**_key(state, watch))
    with state.store._lock, state.store._conn:
        state.store._conn.execute("DROP TRIGGER trg_app_watch_privacy_immutable")
        state.store._conn.execute("ALTER TABLE app_watches DROP COLUMN private_required")
        state.store._conn.execute("DELETE FROM app_schema_migrations WHERE version=48")
    upgraded = ApplicationStore(state.store.path)
    upgraded.watches._clock = state.clock
    try:
        assert upgraded.schema_version == 48
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert upgraded._conn.execute("SELECT digest FROM app_watch_file_text_baselines").fetchone()[0] == digest
        saved = await upgraded.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
        assert saved["active_run_id"] == queued["active_run_id"] and "private_required" not in saved
        assert (await upgraded.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=baseline["id"])) == baseline
        run = await upgraded.watches.claim()
        assert run["run_id"] == queued["active_run_id"] and not run["private_required"]
        completed = await upgraded.watches.complete_project_file_text(
            **_lease(run), file_id="ordinary-file", digest=digest, source_guard=source_guard,
        )
        assert completed["outcome"] == "unchanged"
    finally:
        upgraded.close()


async def test_schema46_preserves_public_watch_history_baseline_and_queued_check(tmp_path, monkeypatch):
    path = tmp_path / "legacy.sqlite"
    clock = Clock()
    stamp = clock().isoformat()
    earlier = (clock() - dt.timedelta(seconds=1)).isoformat()
    definition = {
        "title": "Release notes", "url": "https://example.com/releases", "criterion": "New stable releases",
        "schedule": {"kind": "interval", "timezone": "UTC", "interval_minutes": 60}, "project_id": None,
    }
    limits = json.dumps({"max_operations": 8, "max_active_seconds": 300})
    digest = hashlib.sha256(json.dumps(
        [definition[key] for key in ("title", "url", "criterion", "schedule")] + [False],
        separators=(",", ":"), sort_keys=True,
    ).encode()).hexdigest()
    text = "Version one."
    snapshot = json.dumps({
        "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(), "final_url": definition["url"],
        "extraction_version": "public-text-v1", "run_id": "completed-run",
    })
    with monkeypatch.context() as patch:
        patch.setattr(store_module, "MIGRATIONS", tuple(item for item in MIGRATIONS if item[0] <= 45))
        legacy = ApplicationStore(path)
        try:
            assert legacy.schema_version == 45
            owner = await legacy.resolve_external_identity(
                provider="test", subject="alice", email="alice@example.com",
                display_name="Alice", role="user", auth_method="test",
            )
            with legacy._lock, legacy._conn:
                legacy._conn.execute(
                    "INSERT INTO app_watches (watch_id,user_id,request_id,request_digest,title,url,criterion,"
                    "schedule_json,limits_json,enabled,created_at,updated_at) VALUES "
                    "('public-watch',?,'legacy',?,?,?,?,?,?,0,?,?)",
                    (owner.user_id, digest, definition["title"], definition["url"], definition["criterion"],
                     json.dumps(definition["schedule"]), limits, stamp, stamp),
                )
                legacy._conn.execute(
                    "INSERT INTO app_watch_snapshots VALUES ('completed-run','public-watch',?,?)", (snapshot, stamp),
                )
                legacy._conn.execute("INSERT INTO app_watch_baselines VALUES ('public-watch','completed-run')")
                legacy._conn.execute(
                    "INSERT INTO app_watch_runs (run_id,watch_id,user_id,watch_revision,definition_json,limits_json,"
                    "kind,scheduled_at,status,attempts,fence,operations,available_at,source_snapshot_id,"
                    "outcome,report,created_at,updated_at,completed_at) "
                    "VALUES ('completed-run','public-watch',?,1,?,?,'manual',?,'completed',1,1,1,?,"
                    "'completed-run','baseline','Saved baseline.',?,?,?)",
                    (owner.user_id, json.dumps(definition), limits, earlier, earlier, earlier, stamp, stamp),
                )
                legacy._conn.execute(
                    "INSERT INTO app_watch_operations VALUES ('completed-op','completed-run',1,'fetch','committed',?,?)",
                    (earlier, stamp),
                )
                legacy._conn.execute(
                    "INSERT INTO app_watch_runs (run_id,watch_id,user_id,watch_revision,definition_json,limits_json,"
                    "baseline_json,kind,scheduled_at,status,available_at,created_at,updated_at) VALUES "
                    "('queued-run','public-watch',?,1,?,?,?,'manual',?,'queued',?,?,?)",
                    (owner.user_id, json.dumps(definition), limits, snapshot, stamp, stamp, stamp, stamp),
                )
        finally:
            legacy.close()

    upgraded = ApplicationStore(path)
    try:
        assert upgraded.schema_version == 48
        assert upgraded._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert upgraded._conn.execute(
            "SELECT status FROM app_watch_operations WHERE operation_id='completed-op'",
        ).fetchone()[0] == "committed"
        upgraded.watches._clock = clock
        replayed = await upgraded.watches.create(user_id=owner.user_id, request_id="legacy", enabled=False, **definition)
        assert replayed["id"] == "public-watch" and replayed["active_run_id"] == "queued-run"
        assert replayed["url"] == definition["url"]
        page, _ = await upgraded.watches.get_snapshot_page(
            user_id=owner.user_id, watch_id=replayed["id"], run_id="completed-run", offset=0, limit=100,
        )
        assert page["text"] == text
        run = await upgraded.watches.claim()
        assert run["id"] == "queued-run" and run["baseline"] == json.loads(snapshot)
        finished = await _complete(SimpleNamespace(store=upgraded), run, text=text, outcome="unchanged")
        assert finished["outcome"] == "unchanged"
    finally:
        upgraded.close()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "https://user:pass@example.com", "https://example.com:8443/a",
    "https://example.com/a?key=value", "https://example.com/a?", "https://example.com/#part",
    "https://127.0.0.1", "http://10.0.0.1", "http://[::1]", "http://169.254.169.254/",
    "http://[::ffff:127.0.0.1]", "http://localhost/", "http://example.com\\@localhost/",
])
async def test_unsafe_urls_rejected_without_network(state, url):
    with pytest.raises(WatchError):
        await _create(state, url=url)


@pytest.mark.parametrize("fence", ["disabled", "deletion_requested"])
async def test_ownership_and_active_account_deletion_fence_every_path(state, fence):
    watch = await _create(state)
    for user_id in (state.other.user_id, "missing"):
        with pytest.raises(WatchAccessError):
            await state.store.watches.get(user_id=user_id, watch_id=watch["id"])
        with pytest.raises(WatchAccessError):
            await state.store.watches.run_now(**_key(state, watch, user_id=user_id))
    run = await _claim(state, watch)
    with state.store._lock:
        if fence == "disabled":
            state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        else:
            state.store._conn.execute(
                "INSERT INTO account_deletion_requests(user_id,purge_id,storage_namespace,requested_at) "
                "VALUES (?,?,?,?)", (state.owner.user_id, "purge", state.owner.storage_namespace,
                                      state.clock().isoformat()),
            )
        state.store._conn.commit()
    with pytest.raises(WatchAccessError):
        await state.store.watches.validate_lease(**_lease(run))
    with pytest.raises(WatchAccessError):
        await state.store.watches.list(user_id=state.owner.user_id)
    assert await state.store.watches.claim() is None


async def test_definition_edit_fences_live_work_and_preserves_only_compatible_baseline(state):
    watch = await _create(state)
    baseline = await _claim(state, watch)
    await _complete(state, baseline)
    running = await _claim(state, watch)
    edited = await state.store.watches.update(**_key(state, watch), criterion="Security releases only")
    with pytest.raises(WatchConflictError):
        await _complete(state, running)
    assert (await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=running["id"]))["status"] == "cancelled"
    changed_criterion = await _claim(state, edited)
    assert changed_criterion["baseline"]["run_id"] == baseline["id"]
    edited = await state.store.watches.update(**_key(state, edited), url="https://example.com/other")
    assert (await _claim(state, edited))["baseline"] is None


async def test_downtime_coalesces_one_occurrence_and_duplicate_workers_do_not_overlap(state):
    watch = await _create(state)
    state.clock.advance(5 * 3600 + 120)
    claims = await asyncio.gather(state.store.watches.claim(), state.store.watches.claim())
    run = next(item for item in claims if item)
    assert sum(item is not None for item in claims) == 1
    assert run["kind"] == "catchup" and run["scheduled_at"] == "2026-10-30T13:00:00+00:00"
    assert (await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"]))["next_run_at"] == "2026-10-30T18:00:00+00:00"
    assert len(await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])) == 1
    await state.store.watches.run_now(**_key(state, watch))
    assert len(await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])) == 1


async def test_manual_paused_run_does_not_resume_schedule_and_concurrent_edit_conflicts(state):
    watch = await _create(state, enabled=False)
    assert watch["next_run_at"] is None
    run = await _claim(state, watch)
    assert run["kind"] == "manual"
    await _complete(state, run)
    second = ApplicationStore(state.store.path)
    results = await asyncio.gather(
        state.store.watches.update(**_key(state, watch), title="One"),
        second.watches.update(**_key(state, watch), title="Two"), return_exceptions=True,
    )
    assert sum(isinstance(item, WatchConflictError) for item in results) == 1
    current = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    assert current["enabled"] is False and current["next_run_at"] is None
    second.close()


async def test_reset_conflicts_with_active_check_and_cancel_keeps_the_prior_baseline(state):
    repository = state.store.watches
    watch = await _create(state)
    first = await _claim(state, watch)
    await _complete(state, first)
    ordinary = await _claim(state, watch)
    with pytest.raises(WatchConflictError, match="current watch check"):
        await repository.run_now(**_key(state, watch), rebaseline=True)
    await _complete(state, ordinary, outcome="unchanged")

    queued = await repository.run_now(**_key(state, watch), rebaseline=True)
    assert queued["enabled"] and queued["next_run_at"] == watch["next_run_at"]
    assert queued["revision"] == watch["revision"] + 1
    reset = await repository.claim()
    assert reset["id"] == queued["active_run_id"] and reset["baseline"] is None
    assert reset["rebaseline"] is True
    with pytest.raises(WatchConflictError, match="Watch changed"):
        await repository.run_now(**_key(state, watch), rebaseline=True)

    edited = await repository.update(**_key(state, queued), enabled=False)
    with pytest.raises(WatchConflictError):
        await _complete(state, reset, text="This text must never replace the baseline.")
    saved = await repository.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=reset["id"])
    assert saved["status"] == "cancelled" and saved["rebaseline"] is True
    following = await _claim(state, edited)
    assert following["baseline"]["run_id"] == ordinary["id"]
    assert following["rebaseline"] is False


async def test_research_and_watch_claims_serialize_per_owner_but_allow_other_owner(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    task = await state.store.agents.create(user_id=state.owner.user_id, objective="Research", request_id="agent-1")
    assert await state.store.agents.claim() is None
    other = await state.store.agents.create(user_id=state.other.user_id, objective="Research", request_id="agent-1")
    assert (await state.store.agents.claim())["id"] == other["id"]
    await _complete(state, run)
    claimed = await state.store.agents.claim()
    assert claimed["id"] == task["id"]
    await state.store.watches.run_now(**_key(state, watch))
    assert await state.store.watches.claim() is None


async def test_expired_lease_and_bounded_retries_charge_uncertain_work(state):
    watch = await _create(state)
    first = await _claim(state, watch)
    await state.store.watches.reserve_operation(**_lease(first), kind="fetch")
    state.clock.advance(61)
    recovered = await state.store.watches.claim()
    assert recovered["id"] == first["id"] and recovered["operations"] == 1
    assert recovered["fence"] > first["fence"] and recovered["active_seconds"] == 60
    with pytest.raises(WatchConflictError):
        await state.store.watches.heartbeat(**_lease(first))
    await state.store.watches.fail(**_lease(recovered), error="Temporary network failure", retry=True)
    assert await state.store.watches.claim() is None
    state.clock.advance(120)
    final = await state.store.watches.claim()
    await state.store.watches.fail(**_lease(final), error="Temporary network failure", retry=True)
    assert await state.store.watches.claim() is None
    notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    assert len(notices) == 1 and notices[0]["kind"] == "failure"


async def test_baselines_changed_results_notices_and_failure_recovery_are_atomic(state):
    watch = await _create(state)
    initial = await _claim(state, watch)
    assert (await _complete(state, initial))["outcome"] == "baseline"
    unchanged = await _claim(state, watch)
    await _complete(state, unchanged, outcome="unchanged")
    assert not await state.store.watches.list_notices(user_id=state.owner.user_id)
    changed = await _claim(state, watch)
    await _complete(state, changed, text="Version two.", outcome="changed", significant=True)
    failure = await _claim(state, watch)
    await state.store.watches.fail(**_lease(failure), error="Page unavailable")
    failure_again = await _claim(state, watch)
    assert failure_again["baseline"]["text"] == "Version two."
    await state.store.watches.fail(**_lease(failure_again), error="Still unavailable")
    notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    assert sorted(item["kind"] for item in notices) == ["change", "failure"]
    assert await state.store.watches.unread_count(user_id=state.owner.user_id) == 2
    assert await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=notices[0]["id"])
    assert await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=notices[0]["id"])
    assert await state.store.watches.unread_count(user_id=state.owner.user_id) == 1
    with pytest.raises(WatchAccessError):
        await state.store.watches.mark_notice_read(user_id=state.other.user_id, notice_id=notices[0]["id"])
    recovery = await _claim(state, watch)
    await _complete(state, recovery, text="Version two.", outcome="unchanged")
    third_failure = await _claim(state, watch)
    await state.store.watches.fail(**_lease(third_failure), error="Unavailable again")
    assert len(await state.store.watches.list_notices(user_id=state.owner.user_id)) == 3


async def test_commit_failure_rolls_back_snapshot_baseline_and_notice(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    await _complete(state, run)
    changed = await _claim(state, watch)
    with state.store._lock:
        state.store._conn.execute(
            "CREATE TRIGGER break_watch_notice BEFORE INSERT ON app_watch_notices "
            "BEGIN SELECT RAISE(ABORT,'receipt failed'); END",
        )
    with pytest.raises(sqlite3.IntegrityError, match="receipt failed"):
        await _complete(state, changed, text="Version two.", outcome="changed", significant=True)
    assert not state.store._conn.in_transaction
    assert state.store.watches._baseline(watch["id"])["text"] == "Version one."
    assert (await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=changed["id"]))["status"] == "running"
    assert not await state.store.watches.list_notices(user_id=state.owner.user_id)


async def test_baseline_identity_version_changes_rebaseline_without_notice(state):
    watch = await _create(state)
    await _complete(state, await _claim(state, watch))
    new_version = await _claim(state, watch)
    await _complete(state, new_version, text="New normalized text", extraction_version="public-text-v2")
    new_redirect = await _claim(state, watch)
    await _complete(state, new_redirect, text="Another text", final_url="https://example.com/new", extraction_version="public-text-v2")
    assert not await state.store.watches.list_notices(user_id=state.owner.user_id)


@pytest.mark.parametrize("erase", ["watch", "purge"])
async def test_deletion_cascades_all_new_derivatives_and_fences_dispatch(state, erase):
    watch = await _create(state)
    await _complete(state, await _claim(state, watch))
    failed = await _claim(state, watch)
    await state.store.watches.fail(**_lease(failed), error="Failure")
    live = await _claim(state, watch)
    if erase == "watch":
        await state.store.watches.delete(**_key(state, watch))
    else:
        await state.store.purge_local_user_data(user_id=state.owner.user_id)
    with pytest.raises(WatchAccessError):
        await state.store.watches.validate_lease(**_lease(live))
    for table in ("app_watches", "app_watch_runs", "app_watch_snapshots", "app_watch_baselines",
                  "app_watch_operations", "app_watch_notices"):
        assert state.store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # noqa: S608


async def test_snapshots_are_immutable_and_history_retention_preserves_current_baseline(state):
    watch = await _create(state)
    initial = await _claim(state, watch)
    await _complete(state, initial)
    snapshot_args = dict(user_id=state.owner.user_id, watch_id=watch["id"], run_id=initial["id"])
    page, saved_at = await state.store.watches.get_snapshot_page(**snapshot_args, offset=0, limit=7)
    assert page["text"] == "Version" and page["next_offset"] == 7
    with state.store._lock:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            state.store._conn.execute("UPDATE app_watch_snapshots SET record_json='{}'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            state.store._conn.execute("UPDATE app_watch_runs SET definition_json='{}'")
        state.store._conn.rollback()
    for _ in range(52):
        run = await _claim(state, watch)
        await state.store.watches.fail(**_lease(run), error="Unavailable")
    assert len(await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])) == 50
    assert state.store.watches._baseline(watch["id"])["run_id"] == initial["id"]
    with pytest.raises(WatchAccessError):
        await state.store.watches.get_snapshot_page(**snapshot_args, offset=7, limit=7)
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_response(
            user_id=state.owner.user_id, snapshot=(watch["id"], initial["id"], saved_at),
        )
    notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    assert len(notices) == 1
    assert (await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"],
                                            run_id=notices[0]["run_id"]))["status"] == "failed"
    next_run = await _claim(state, watch)
    assert next_run["baseline"]["text"] == "Version one."


async def test_current_server_limits_clamp_recovered_work_and_create_caps(state):
    watch = await _create(state, max_per_user=1)
    with pytest.raises(WatchLimitError):
        await _create(state, request_id="second", max_per_user=1)
    run = await _claim(state, watch)
    await state.store.watches.reserve_operation(**_lease(run), kind="fetch")
    state.clock.advance(61)
    assert await state.store.watches.claim(limits={"max_operations": 1, "max_active_seconds": 300}) is None
    assert (await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"]))["latest_run"]["status"] == "failed"


async def test_failed_or_unreserved_results_cannot_advance_baseline(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    kwargs = dict(text="Version one.", sha256=hashlib.sha256(b"Version one.").hexdigest(),
                  outcome="baseline", significant=None, report="", diff="", final_url=run["url"],
                  extraction_version="v1")
    with pytest.raises(WatchConflictError):
        await state.store.watches.complete(**_lease(run), **kwargs)
    await state.store.watches.reserve_operation(**_lease(run), kind="fetch")
    with pytest.raises(WatchError):
        await state.store.watches.complete(**_lease(run), **{**kwargs, "sha256": "0" * 64})
    with pytest.raises(WatchError):
        await state.store.watches.complete(**_lease(run), **{**kwargs, "text": "é" * 40000})
    assert state.store.watches._baseline(watch["id"]) is None


async def test_notification_feed_reads_owned_bodies_and_total_in_one_snapshot(state):
    assert await state.store.watches.notification_feed(user_id=state.owner.user_id) == {
        "items": [], "unread_count": 0,
    }
    watch = await _create(state)
    await _complete(state, await _claim(state, watch))
    await _complete(state, await _claim(state, watch), text="Version two.", outcome="changed", significant=True)
    await state.store.watches.fail(**_lease(await _claim(state, watch)), error="Page unavailable")
    feed = await state.store.watches.notification_feed(user_id=state.owner.user_id, limit=1)
    assert len(feed["items"]) == 1 and feed["unread_count"] == 2
    await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=feed["items"][0]["id"])
    assert (await state.store.watches.notification_feed(user_id=state.owner.user_id))["unread_count"] == 1
    assert await state.store.watches.notification_feed(user_id=state.other.user_id) == {
        "items": [], "unread_count": 0,
    }
    await state.store.watches.delete(**_key(state, watch))
    assert await state.store.watches.notification_feed(user_id=state.owner.user_id) == {
        "items": [], "unread_count": 0,
    }


async def test_response_fence_rejects_deleted_or_revised_payloads_after_async_read(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    await _complete(state, run)
    failed = await _claim(state, watch)
    await state.store.watches.fail(**_lease(failed), error="Unavailable")
    saved_watch = await state.store.watches.get(user_id=state.owner.user_id, watch_id=watch["id"])
    saved_runs = await state.store.watches.list_runs(user_id=state.owner.user_id, watch_id=watch["id"])
    saved_notices = await state.store.watches.list_notices(user_id=state.owner.user_id)
    state.store.watches.fence_response(user_id=state.owner.user_id, watches=[saved_watch],
                                      runs=saved_runs, notices=saved_notices)
    # Read-state changes cannot revoke the immutable notice body.
    await state.store.watches.mark_notice_read(user_id=state.owner.user_id, notice_id=saved_notices[0]["id"])
    state.store.watches.fence_response(user_id=state.owner.user_id, notices=saved_notices)
    updated = await state.store.watches.update(**_key(state, watch), title="Edited")
    with pytest.raises(WatchConflictError):
        state.store.watches.fence_response(user_id=state.owner.user_id, watches=[saved_watch])
    # Old run definitions remain legitimate history after a definition edit.
    state.store.watches.fence_response(user_id=state.owner.user_id, runs=saved_runs, notices=saved_notices)
    await state.store.watches.delete(**_key(state, updated))
    for kind, payload in (("watches", [saved_watch]), ("runs", saved_runs), ("notices", saved_notices)):
        with pytest.raises(WatchAccessError):
            state.store.watches.fence_response(user_id=state.owner.user_id, **{kind: payload})


async def test_response_fence_checks_purge_owner_and_live_run_change(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    saved = await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=run["id"])
    state.clock.advance(1)
    await state.store.watches.heartbeat(**_lease(run))
    with pytest.raises(WatchConflictError):
        state.store.watches.fence_response(user_id=state.owner.user_id, runs=[saved])
    await state.store.purge_local_user_data(user_id=state.owner.user_id)
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_response(user_id=state.owner.user_id, watches=[watch])
    with state.store._lock:
        state.store._conn.execute("UPDATE app_users SET status='disabled' WHERE user_id=?", (state.owner.user_id,))
        state.store._conn.commit()
    with pytest.raises(WatchAccessError):
        state.store.watches.fence_response(user_id=state.owner.user_id)


async def test_run_history_retains_the_definition_that_produced_its_report(state):
    watch = await _create(state)
    run = await _claim(state, watch)
    await _complete(state, run)
    await state.store.watches.update(**_key(state, watch), title="Other page", url="https://example.com/other",
                                     criterion="Other criterion")
    saved = await state.store.watches.get_run(user_id=state.owner.user_id, watch_id=watch["id"], run_id=run["id"])
    assert saved["definition"] == {key: watch[key] for key in ("title", "url", "criterion", "schedule", "project")}
    assert "text" not in saved and "lease_token" not in saved
