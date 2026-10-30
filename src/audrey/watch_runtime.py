"""Lifecycle-owned, leased checks of explicit saved public-page watches."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state.watches import WatchLimitError
from audrey.watch_executor import WatchCheckError

logger = logging.getLogger(__name__)
WatchCheckExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class WatchRuntime:
    """One finite check at a time; repository owns schedules and retry backoff."""

    def __init__(
        self, repository: Any, executor: WatchCheckExecutor, *,
        poll_seconds: float = 10.0, lease_seconds: int = 60, max_attempts: int = 3,
        check_timeout_seconds: float = 100.0, run_limits: dict[str, int] | None = None,
    ) -> None:
        if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
            raise ValueError("Invalid watch polling interval.")
        if type(lease_seconds) is not int or not 2 <= lease_seconds <= 300:
            raise ValueError("Invalid watch lease duration.")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("Invalid watch attempt limit.")
        if not math.isfinite(check_timeout_seconds) or not 0 < check_timeout_seconds <= 300:
            raise ValueError("Invalid watch check timeout.")
        if run_limits is not None and (
            not isinstance(run_limits, dict)
            or run_limits.keys() != {"max_operations", "max_active_seconds"}
            or type(run_limits["max_operations"]) is not int
            or not 2 <= run_limits["max_operations"] <= 8
            or type(run_limits["max_active_seconds"]) is not int
            or not 1 <= run_limits["max_active_seconds"] <= 300
        ):
            raise ValueError("Invalid watch run limits.")
        self.repository, self.executor = repository, executor
        self.poll_seconds, self.lease_seconds = poll_seconds, lease_seconds
        self.max_attempts, self.check_timeout_seconds = max_attempts, check_timeout_seconds
        self.run_limits = dict(run_limits) if run_limits is not None else None
        self._wake = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._stopping = False
        self._worker = asyncio.create_task(self._dispatch(), name="audrey-watch-dispatcher")

    async def stop(self) -> None:
        self._stopping = True
        self.wake()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    def wake(self) -> None:
        self._wake.set()

    async def _dispatch(self) -> None:
        while not self._stopping:
            self._wake.clear()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - lifecycle worker remains available
                logger.warning("Watch dispatcher cycle failed (%s)", type(error).__name__)
            try:
                async with asyncio.timeout(self.poll_seconds):
                    await self._wake.wait()
            except TimeoutError:
                pass

    async def run_once(self) -> bool:
        async with self._run_lock:
            if self._stopping:
                return False
            options = {"lease_seconds": self.lease_seconds, "max_attempts": self.max_attempts}
            if self.run_limits is not None:
                options["limits"] = self.run_limits
            run = await self.repository.claim(**options)
            if run is None:
                return False
            await self._run_claim(run)
            return True

    @staticmethod
    def _lease(run: dict[str, Any]) -> dict[str, Any]:
        return {key: run[key] for key in ("run_id", "lease_token", "fence")}

    async def _run_claim(self, run: dict[str, Any]) -> None:
        lease = self._lease(run)
        work = asyncio.create_task(self._execute(run), name="audrey-watch-check")
        watchdog = asyncio.create_task(self._heartbeat(lease), name="audrey-watch-lease")
        try:
            remaining = run["limits"]["max_active_seconds"] - run.get("active_seconds", 0)
            if remaining <= 0:
                raise WatchCheckError("The watch check reached its active-time limit.")
            async with asyncio.timeout(min(remaining, self.check_timeout_seconds)):
                done, _ = await asyncio.wait((work, watchdog), return_when=asyncio.FIRST_COMPLETED)
                if watchdog in done:
                    await watchdog
                await work
        except BaseException as error:
            # Fence before a cancellation-resistant provider can return or dispatch.
            work.cancel()
            watchdog.cancel()
            message, retry = self._failure(error)
            try:
                await self.repository.fail(**lease, error=message, retry=retry)
            except Exception as cleanup_error:  # noqa: BLE001 - revoked/deleted owner may be gone
                logger.debug("Watch cleanup was fenced (%s)", type(cleanup_error).__name__)
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            await self._cancel_tasks(work, watchdog)

    async def _execute(self, run: dict[str, Any]) -> None:
        lease = self._lease(run)
        await self.repository.validate_lease(**lease)
        result = await self.executor(run)
        if not isinstance(result, dict) or result.keys() != {
            "outcome", "significant", "report", "diff", "text", "sha256",
            "extraction_version", "final_url",
        }:
            raise WatchCheckError("The watch check returned an invalid result.")
        await self.repository.validate_lease(**lease)
        await self.repository.complete(**lease, **result)

    async def _heartbeat(self, lease: dict[str, Any]) -> None:
        while True:
            await asyncio.sleep(min(self.poll_seconds, self.lease_seconds / 3))
            await self.repository.heartbeat(**lease, lease_seconds=self.lease_seconds)

    @staticmethod
    def _failure(error: BaseException) -> tuple[str, bool]:
        if isinstance(error, asyncio.CancelledError):
            return "Worker stopped; the saved watch check will retry.", True
        if isinstance(error, AgentDispatchStoppedError):
            return str(error), False
        if isinstance(error, WatchCheckError):
            return str(error), True
        if isinstance(error, WatchLimitError):
            return "The watch check reached its execution allowance.", False
        if isinstance(error, TimeoutError):
            return "The watch check exceeded its time limit.", True
        return "The watch check did not finish successfully.", True

    @staticmethod
    async def _cancel_tasks(*tasks: asyncio.Task) -> None:
        def consume(task: asyncio.Task) -> None:
            with suppress(asyncio.CancelledError, Exception):
                task.exception()

        for task in tasks:
            if not task.done():
                task.cancel()
        done, pending = await asyncio.wait(tasks, timeout=1)
        for task in done:
            consume(task)
        for task in pending:
            task.add_done_callback(consume)
