"""One bounded, leased Private step, ready for a native worker adapter.

No startup loop or HTTP scheduler is enabled here. Trusted server code supplies
one step; its work products commit through the existing case and task stores.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal

from audrey.app_state.private_tasks import (
    PrivateTaskConflictError,
    PrivateTaskError,
    PrivateTaskLease,
    PrivateTaskLeaseExpiredError,
    PrivateTasksRepository,
    private_task_scope,
)
from audrey.tools.private_dispatch import PrivateToolHandler, private_tool_scope


class PrivateStepBudgetError(PrivateTaskError):
    """A step reached its remaining server-owned resource limit."""


@dataclass(frozen=True, slots=True)
class StepResult:
    checkpoint: dict
    status: Literal["queued", "waiting_for_user", "paused", "completed", "failed"] = "queued"


class PrivateStep:
    """Explicit progress, budget and tool boundaries for one trusted callback."""

    def __init__(
        self, lease: PrivateTaskLease, task: dict, guard: Callable[[], Awaitable[None]]
    ) -> None:
        self.lease, self._guard = lease, guard
        self._initial_usage, self.limits = task["usage"], task["limits"]
        self.model_tokens = self.tool_calls = 0
        self._reported_tokens = self._reported_tools = 0
        self._progress_lock = asyncio.Lock()

    @property
    def usage(self) -> dict:
        return {
            "model_tokens": self.model_tokens - self._reported_tokens,
            "tool_calls": self.tool_calls - self._reported_tools,
        }

    async def check(self, *, finishing: bool = False) -> dict:
        if not self.lease.work_active and not finishing:
            raise PrivateTaskConflictError("Private task callback has ended.")
        await self._guard()  # Recheck live account, model access and public reference pin.
        return await self.lease.verify()

    async def progress(self) -> None:
        """Call only after an observed token batch, model result or tool result."""
        async with self._progress_lock:
            current = await self.check()
            spent = self.usage
            updated = await self.lease.repository.renew(
                user_id=self.lease.user_id,
                task_id=self.lease.task_id,
                expected_revision=current["revision"],
                lease_token=self.lease.lease_token,
                fence=self.lease.fence,
                pins=self.lease.pins,
                progress=current["progress"] + 1,
                lease_seconds=120,
                usage=spent,
            )
            self._reported_tokens += spent["model_tokens"]
            self._reported_tools += spent["tool_calls"]
            if updated["status"] != "running":
                raise PrivateStepBudgetError("Private task resource budget reached.")

    async def tokens_used(self, count: int) -> None:
        if type(count) is not int or not 0 <= count <= 200000 or self.model_tokens + count > 200000:
            raise PrivateTaskError("Invalid model resource accounting.")
        if count == 0:
            return
        self.model_tokens += count
        await self.progress()

    def tools(self, handlers: dict[str, PrivateToolHandler]) -> dict[str, PrivateToolHandler]:
        def wrap(handler: PrivateToolHandler) -> PrivateToolHandler:
            async def call(arguments: dict) -> Any:
                await self.check()
                if (
                    self._initial_usage["tool_calls"] + self.tool_calls
                    >= self.limits["max_tool_calls"]
                ):
                    raise PrivateStepBudgetError("Private task tool budget reached.")
                self.tool_calls += 1
                await self.progress()  # Charge the actual dispatch before entering its callback.
                try:
                    return await handler(arguments)
                finally:
                    await self.progress()

            return call

        return {name: wrap(handler) for name, handler in handlers.items()}


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Owned verified runtime and exact identities supplied by the tax adapter."""

    app: Any
    pins: dict
    storage_namespace: str
    handlers: dict[str, PrivateToolHandler]
    guard: Callable[[], Awaitable[None]]


Preparation = Callable[[str, str], AbstractAsyncContextManager[PreparedStep]]
StepWork = Callable[[Any, PrivateStep], Awaitable[StepResult]]


class PrivateTaskStepRunner:
    """Execute once; restart recovery and result delivery belong to the adapter."""

    def __init__(
        self, repository: PrivateTasksRepository, prepare: Preparation, *, poll_seconds: float = 1.0
    ):
        if not 0 < poll_seconds <= 5:
            raise ValueError("Private task polling interval is invalid")
        self.repository, self.prepare, self.poll_seconds = repository, prepare, poll_seconds

    async def execute(self, *, user_id: str, task_id: str, work: StepWork) -> dict:
        task = await self.repository.get(user_id=user_id, task_id=task_id)
        if task is None:
            raise PrivateTaskError("Private task not found.")
        # Preparation must enter a verified PrivateProcessingRuntime before yielding.
        async with self.prepare(user_id, task["conversation_id"]) as prepared:
            await prepared.guard()
            claimed = await self.repository.claim(
                user_id=user_id,
                task_id=task_id,
                expected_revision=task["revision"],
                pins=prepared.pins,
                lease_seconds=120,
            )
            with private_task_scope(self.repository, claimed) as lease:
                step = PrivateStep(lease, claimed, prepared.guard)
                remaining = claimed["limits"]["max_wall_seconds"] - claimed["usage"]["wall_seconds"]
                try:
                    with private_tool_scope(
                        user_id=prepared.storage_namespace, handlers=step.tools(prepared.handlers)
                    ):
                        result = await self._run(work, prepared.app, step, remaining)
                    current = await step.check(finishing=True)
                    return await self.repository.checkpoint(
                        user_id=user_id,
                        task_id=task_id,
                        expected_revision=current["revision"],
                        lease_token=lease.lease_token,
                        fence=lease.fence,
                        expected_case_revision=current["case_revision"],
                        pins=prepared.pins,
                        checkpoint_key=f"step-{lease.fence}",
                        checkpoint=result.checkpoint,
                        status=result.status,
                        usage=step.usage,
                    )
                except BaseException as error:
                    reason = (
                        "worker_stopped"
                        if isinstance(error, asyncio.CancelledError)
                        else "step_timeout"
                        if isinstance(error, TimeoutError)
                        else "budget_exhausted"
                        if isinstance(error, PrivateStepBudgetError)
                        else "lease_expired"
                        if isinstance(error, PrivateTaskLeaseExpiredError)
                        else "inputs_changed"
                        if isinstance(error, PrivateTaskConflictError)
                        else "step_failed"
                    )
                    # Do not replace the original failure with a cleanup error.
                    with suppress(PrivateTaskError):
                        await self.repository.abandon(
                            user_id=user_id,
                            task_id=task_id,
                            lease_token=lease.lease_token,
                            fence=lease.fence,
                            reason=reason,
                            usage=step.usage,
                        )
                    raise

    async def _run(
        self, work: StepWork, app: Any, step: PrivateStep, deadline_seconds: float
    ) -> StepResult:
        async def watch():
            while True:
                await asyncio.sleep(self.poll_seconds)
                await step.check()

        attempt = asyncio.create_task(work(app, step))
        watchdog = asyncio.create_task(watch())
        try:
            async with asyncio.timeout(deadline_seconds):
                done, _ = await asyncio.wait(
                    (attempt, watchdog), return_when=asyncio.FIRST_COMPLETED
                )
                if watchdog in done:
                    await watchdog  # A guard failed; stop the callback before accepting work.
                result = await attempt
                if not isinstance(result, StepResult):
                    raise PrivateTaskError("Invalid Private step result.")
                return result
        finally:
            step.lease.work_active = False
            for task in (attempt, watchdog):
                if not task.done():
                    task.cancel()
            # A provider cleanup must not turn the step deadline into an
            # unbounded wait. The owning scopes revoke inherited callbacks on exit.
            done, pending = await asyncio.wait((attempt, watchdog), timeout=1.0)

            def consume(task: asyncio.Task) -> None:
                with suppress(asyncio.CancelledError, Exception):
                    task.exception()

            for task in done:
                consume(task)
            for task in pending:
                task.add_done_callback(consume)


def runtime_task_pins(runtime: Any) -> dict:
    """Fingerprint the verified policy, without inventing locality from model names."""
    policy = runtime.policy
    payload = {
        "models": dict(policy.local_models),
        "ollama": policy.ollama_base_url,
        "tools": sorted(policy.allowed_tools),
        "servers": sorted(policy.tool_servers),
    }
    return {
        "model_digests": dict(policy.local_models),
        "policy_digest": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


@asynccontextmanager
async def prepared_private_step(
    *,
    app: Any,
    runtime: Any,
    principal: Any,
    conversation_id: str,
    pins: dict,
    guard: Callable[[], Awaitable[None]],
) -> AsyncIterator[PreparedStep]:
    """Keep verified model transports and owner-bound tool callbacks scoped together.

    The native adapter supplies freshly verified pins and live input validation;
    it must not reconstruct a policy from persisted digests alone.
    """
    from audrey.private_processing import PrivateProcessingRuntime
    from audrey.tools.private_handlers import make_private_tool_handlers

    if (
        not isinstance(runtime, PrivateProcessingRuntime)
        or not runtime.tax_review
        or pins != runtime_task_pins(runtime)
    ):
        raise PrivateTaskError("A verified Tax review runtime with matching pins is required.")
    await guard()
    async with runtime.scope(app) as execution_app:
        handlers = make_private_tool_handlers(
            execution_app, principal, tax_conversation_id=conversation_id
        )
        yield PreparedStep(execution_app, pins, principal.storage_namespace, dict(handlers), guard)
