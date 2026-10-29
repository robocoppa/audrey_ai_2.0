"""Lifecycle-owned execution of bounded, durable native agent steps.

Only committed step results survive a retry. An interrupted model/read operation
may be repeated; repository leases fence every later checkpoint and delivery.
The injected executor owns the model/tool policy for each supported step.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state.agents import AgentTaskError

logger = logging.getLogger(__name__)

AgentExecutor = Callable[[dict[str, Any], str], Awaitable[str]]
_STEP_IDS = ("plan", "research", "review")


class _RejectedStepError(ValueError):
    """A trusted runtime boundary rejected an executor result."""


class AgentRuntime:
    """One queued task at a time, sharing Audrey's existing model admission.

    ``run_once`` is also available without starting the dispatcher, for isolated
    checks. All attempt, operation and active-time accounting is repository-owned
    and persists across process restarts, pause and resume.
    """

    def __init__(
        self,
        repository: Any,
        executor: AgentExecutor,
        *,
        poll_seconds: float = 2,
        lease_seconds: int = 60,
        max_attempts: int = 6,
        step_timeout_seconds: float = 600,
        max_output_chars: int = 64000,
    ) -> None:
        if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
            raise ValueError("Invalid agent polling interval.")
        if type(lease_seconds) is not int or not 2 <= lease_seconds <= 300:
            raise ValueError("Invalid agent lease duration.")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
            raise ValueError("Invalid agent attempt limit.")
        if not math.isfinite(step_timeout_seconds) or not 0 < step_timeout_seconds <= 3600:
            raise ValueError("Invalid agent step timeout.")
        if type(max_output_chars) is not int or not 1 <= max_output_chars <= 64000:
            raise ValueError("Invalid agent output limit.")
        self.repository, self.executor = repository, executor
        self.poll_seconds, self.lease_seconds = poll_seconds, lease_seconds
        self.max_attempts = max_attempts
        self.step_timeout_seconds, self.max_output_chars = step_timeout_seconds, max_output_chars
        self._wake = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._stopping = False
        self._worker = asyncio.create_task(self._dispatch(), name="audrey-agent-dispatcher")

    async def stop(self) -> None:
        """Requeue interrupted work after revoking its lease, then stop polling."""
        self._stopping = True
        self.wake()
        worker, self._worker = self._worker, None
        if worker is None:
            return
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
            except Exception as error:  # noqa: BLE001 - keep the lifecycle worker available
                # Provider/storage errors can contain credentials or source text.
                logger.warning("Agent dispatcher cycle failed (%s)", type(error).__name__)
            try:
                async with asyncio.timeout(self.poll_seconds):
                    await self._wake.wait()
            except TimeoutError:
                pass

    async def run_once(self) -> bool:
        """Claim and execute at most one task; return whether one was claimed."""
        async with self._run_lock:
            if self._stopping:
                return False
            task = await self.repository.claim(
                lease_seconds=self.lease_seconds, max_attempts=self.max_attempts
            )
            if task is None:
                return False
            await self._run_claim(task)
            return True

    @staticmethod
    def _lease(task: dict) -> dict:
        return {
            "task_id": task["id"],
            "lease_token": task["lease_token"],
            "fence": task["fence"],
        }

    async def _run_claim(self, task: dict) -> None:
        lease = self._lease(task)
        work = asyncio.create_task(self._execute_steps(task), name="audrey-agent-steps")
        watchdog = asyncio.create_task(self._heartbeat(lease), name="audrey-agent-lease")
        try:
            remaining = task["limits"]["max_active_seconds"] - task.get("active_seconds", 0)
            if remaining <= 0:
                raise _RejectedStepError("Agent execution time limit reached.")
            async with asyncio.timeout(remaining):
                done, _ = await asyncio.wait((work, watchdog), return_when=asyncio.FIRST_COMPLETED)
                if watchdog in done:
                    await watchdog
                await work
        except BaseException as error:
            # Revoke before waiting for provider cleanup. A callback that ignores
            # cancellation cannot commit, or perform a new guarded dispatch.
            work.cancel()
            watchdog.cancel()
            message, retry = self._failure(error)
            try:
                await self.repository.fail(**lease, error=message, retry=retry)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the original failure
                # A control, purge or newer lease may have already revoked us.
                if not isinstance(cleanup_error, AgentTaskError):
                    logger.warning("Agent lease cleanup failed (%s)", type(cleanup_error).__name__)
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            await self._cancel_tasks(work, watchdog)

    async def _heartbeat(self, lease: dict) -> None:
        interval = min(self.poll_seconds, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            await self.repository.heartbeat(**lease, lease_seconds=self.lease_seconds)

    async def _execute_steps(self, task: dict) -> None:
        if tuple(step.get("id") for step in task["steps"]) != _STEP_IDS:
            raise _RejectedStepError("Saved agent plan is invalid.")
        lease = self._lease(task)
        for step_id in _STEP_IDS:
            step = next(step for step in task["steps"] if step["id"] == step_id)
            if step["status"] == "completed":
                continue
            # Reserve before model/tool dispatch; unknown crash outcomes retain
            # their cumulative charge even when this step must be repeated.
            operation_id = await self.repository.reserve_operation(**lease, step_id=step_id)
            execution_task = {**task, "operation_id": operation_id, "current_step": step_id}
            try:
                async with asyncio.timeout(self.step_timeout_seconds):
                    result = await self.executor(execution_task, step_id)
            except TimeoutError:
                raise _RejectedStepError("Agent step exceeded its time limit.") from None
            if not isinstance(result, str) or not result.strip():
                raise _RejectedStepError("Agent step did not produce a usable result.")
            if len(result) > self.max_output_chars or len(result.encode("utf-8")) > 64000:
                raise _RejectedStepError("Agent step exceeded its output limit.")
            task = await self.repository.checkpoint(**lease, step_id=step_id, result=result)
        final = next(step["result"] for step in task["steps"] if step["id"] == "review")
        await self.repository.complete(**lease, result=final)

    @staticmethod
    def _failure(error: BaseException) -> tuple[str, bool]:
        if isinstance(error, asyncio.CancelledError):
            return "Worker stopped; saved steps will resume.", True
        if isinstance(error, _RejectedStepError):
            return str(error), False
        if isinstance(error, TimeoutError):
            return "Agent execution time limit reached.", False
        if isinstance(error, AgentDispatchStoppedError):
            # This type carries only trusted, server-defined stop descriptions.
            return str(error), False
        if isinstance(error, AgentTaskError):
            return "Agent access, inputs or execution limits changed.", False
        return "Agent step failed.", True

    @staticmethod
    async def _cancel_tasks(*tasks: asyncio.Task) -> None:
        def consume(task: asyncio.Task) -> None:
            with suppress(asyncio.CancelledError, Exception):
                task.exception()

        for task in tasks:
            if not task.done():
                task.cancel()
        # Cancellation-resistant providers must not hold up lifecycle shutdown.
        done, pending = await asyncio.wait(tasks, timeout=1)
        for task in done:
            consume(task)
        for task in pending:
            task.add_done_callback(consume)
