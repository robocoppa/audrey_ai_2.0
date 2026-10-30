"""Watch lifecycle cancellation and finite attempts without network or models."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.watch_executor import WatchCheckError
from audrey.watch_runtime import WatchRuntime


class Repository:
    def __init__(self):
        self.status = "queued"
        self.attempts = 0
        self.fence = 0
        self.saved = None
        self.error = ""
        self.failures = []
        self.claim_options = []
        self.remaining = 300

    async def claim(self, **options):
        self.claim_options.append(options)
        if self.status != "queued":
            return None
        if self.attempts >= options["max_attempts"]:
            self.status = "failed"
            return None
        self.status = "running"
        self.attempts += 1
        self.fence += 1
        return {
            "run_id": "run", "lease_token": "lease", "fence": self.fence,
            "limits": {"max_active_seconds": self.remaining}, "active_seconds": 0,
        }

    async def validate_lease(self, *, fence, **_lease):
        if self.status != "running" or fence != self.fence:
            raise AgentDispatchStoppedError("The watch changed.")

    async def heartbeat(self, **lease):
        await self.validate_lease(**lease)

    async def complete(self, **result):
        await self.validate_lease(**result)
        self.saved = result
        self.status = "completed"
        self.fence += 1

    async def fail(self, *, error, retry=False, **lease):
        await self.validate_lease(**lease)
        self.failures.append((error, retry))
        self.error = error
        self.status = "queued" if retry else "failed"
        self.fence += 1

    def pause(self):
        self.status = "paused"
        self.fence += 1


def _result():
    return {
        "outcome": "baseline", "significant": None, "report": "Baseline saved.",
        "diff": "", "text": "Public release", "sha256": hashlib.sha256(b"Public release").hexdigest(),
        "extraction_version": "public-text-v1", "final_url": "https://example.org/releases",
    }


def _runtime(repository, executor, **kwargs):
    return WatchRuntime(repository, executor, poll_seconds=0.01, **kwargs)


async def test_one_cycle_claims_and_commits_one_fenced_snapshot_with_current_limits():
    repository = Repository()

    async def execute(run):
        assert run["run_id"] == "run" and run["fence"] == 1
        return _result()

    runtime = _runtime(repository, execute, run_limits={"max_operations": 4, "max_active_seconds": 120})
    assert await runtime.run_once() is True
    assert repository.status == "completed" and repository.saved["text"] == "Public release"
    assert repository.claim_options[0]["limits"] == {"max_operations": 4, "max_active_seconds": 120}
    assert await runtime.run_once() is False


@pytest.mark.parametrize("error", [RuntimeError("credential=secret"), WatchCheckError("The page is incomplete.")])
async def test_failures_retry_only_with_bounded_repository_attempts_and_safe_messages(error):
    repository = Repository()

    async def execute(_run):
        raise error

    runtime = _runtime(repository, execute)
    assert [await runtime.run_once() for _ in range(4)] == [True, True, True, False]
    assert repository.attempts == 3 and repository.saved is None
    assert "credential" not in repository.error and "secret" not in repository.error


async def test_authority_failure_is_terminal_without_retry():
    repository = Repository()

    async def execute(_run):
        raise AgentDispatchStoppedError("Watch access is unavailable.")

    runtime = _runtime(repository, execute)
    assert await runtime.run_once() and not await runtime.run_once()
    assert repository.status == "failed" and repository.attempts == 1


async def test_pause_between_provider_result_and_commit_fences_late_output():
    repository = Repository()

    async def execute(_run):
        repository.pause()
        return _result()

    assert await _runtime(repository, execute).run_once()
    assert repository.status == "paused" and repository.saved is None


async def test_heartbeat_cancels_provider_after_watch_pause():
    repository = Repository()
    started, stopped = asyncio.Event(), asyncio.Event()

    async def execute(_run):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    attempt = asyncio.create_task(_runtime(repository, execute).run_once())
    await asyncio.wait_for(started.wait(), 2)
    repository.pause()
    await asyncio.wait_for(attempt, 2)
    assert stopped.is_set() and repository.saved is None and repository.status == "paused"


async def test_check_timeout_stops_provider_and_never_calls_complete():
    repository = Repository()
    stopped = asyncio.Event()

    async def execute(_run):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    assert await _runtime(repository, execute, check_timeout_seconds=0.02).run_once()
    assert stopped.is_set() and repository.saved is None
    assert repository.status == "queued" and "time limit" in repository.error


async def test_start_stop_are_idempotent_and_interruptions_requeue_saved_occurrence():
    repository = Repository()
    started, stopped = asyncio.Event(), asyncio.Event()

    async def execute(_run):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    runtime = _runtime(repository, execute)
    await runtime.start()
    await runtime.start()
    await asyncio.wait_for(started.wait(), 2)
    await asyncio.wait_for(runtime.stop(), 2)
    await runtime.stop()
    assert stopped.is_set() and repository.status == "queued"
    assert not await runtime.run_once()
    assert repository.attempts == 1 and repository.saved is None


async def test_shutdown_revokes_a_cancellation_resistant_provider_before_bounded_drain():
    repository = Repository()
    started, release, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(_run):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        returned.set()
        return _result()

    runtime = _runtime(repository, execute)
    await runtime.start()
    await asyncio.wait_for(started.wait(), 2)
    await asyncio.wait_for(runtime.stop(), 2)
    assert repository.status == "queued" and not returned.is_set()
    release.set()
    await asyncio.wait_for(returned.wait(), 2)
    await asyncio.sleep(0)
    assert repository.saved is None


async def test_concurrent_wakeups_serialize_execution():
    repository = Repository()
    active, maximum = 0, 0

    async def execute(_run):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1
        return _result()

    runtime = _runtime(repository, execute)
    assert await asyncio.gather(runtime.run_once(), runtime.run_once()) == [True, False]
    assert maximum == 1 and repository.attempts == 1


@pytest.mark.parametrize("result", [None, "unchanged", {}, {**_result(), "extra": True}])
async def test_invalid_executor_result_is_a_failure_not_an_unchanged_snapshot(result):
    repository = Repository()

    async def execute(_run):
        return result

    assert await _runtime(repository, execute).run_once()
    assert repository.status == "queued" and repository.saved is None


@pytest.mark.parametrize("options", [
    {"poll_seconds": 0}, {"lease_seconds": 1}, {"max_attempts": 4},
    {"check_timeout_seconds": float("inf")}, {"run_limits": {"max_operations": 100}},
])
def test_runtime_configuration_is_finite_and_bounded(options):
    with pytest.raises(ValueError):
        WatchRuntime(Repository(), lambda _run: None, **options)
