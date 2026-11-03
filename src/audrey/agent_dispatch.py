"""Optional live authorization for tools dispatched by a saved agent attempt."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass


class AgentDispatchStoppedError(RuntimeError):
    """The saved attempt no longer authorizes content dispatch."""


@dataclass(frozen=True)
class AgentInputRequest:
    """A planning result that needs a saved user answer before execution."""

    question: str


@dataclass
class _Scope:
    check: Callable[[str], Awaitable[None]]
    start: Callable[[str], Awaitable[int]] | None = None
    finish: Callable[[int, bool, int], Awaitable[None]] | None = None
    active: bool = True


_scope: ContextVar[_Scope | None] = ContextVar("agent_dispatch_scope", default=None)


@contextmanager
def agent_dispatch_scope(
    check: Callable[[str], Awaitable[None]], *,
    start: Callable[[str], Awaitable[int]] | None = None,
    finish: Callable[[int, bool, int], Awaitable[None]] | None = None,
) -> Iterator[None]:
    scope = _Scope(check, start, finish)
    token = _scope.set(scope)
    try:
        yield
    finally:
        scope.active = False
        _scope.reset(token)


async def authorize_agent_tool(name: str) -> None:
    scope = _scope.get()
    if scope is not None:
        if not scope.active:
            raise AgentDispatchStoppedError("The agent attempt has stopped.")
        await scope.check(name)
        if not scope.active:
            raise AgentDispatchStoppedError("The agent attempt has stopped.")


@asynccontextmanager
async def agent_tool_execution(name: str) -> AsyncIterator[Callable[[bool], None]]:
    """Observe an actual allowed invocation, never arguments or tool output.

    Dispatch enters only after its policy/argument/registry guards. Inherited
    callbacks share the scope's active flag and cannot outlive the saved attempt.
    """
    scope = _scope.get()
    failed = False

    def outcome(is_error: bool) -> None:
        nonlocal failed
        failed = bool(is_error)

    if scope is None or scope.start is None or scope.finish is None:
        yield outcome
        return
    if not scope.active:
        raise AgentDispatchStoppedError("The agent attempt has stopped.")
    seq = await scope.start(name)
    started = time.monotonic()
    if not scope.active:
        raise AgentDispatchStoppedError("The agent attempt has stopped.")
    try:
        yield outcome
    except BaseException:
        # Preserve the original failure; controls may already have interrupted
        # this event and revoked the lease while transport was in flight.
        if scope.active:
            with suppress(Exception):
                await scope.finish(seq, True, max(0, int((time.monotonic() - started) * 1000)))
        raise
    else:
        if not scope.active:
            raise AgentDispatchStoppedError("The agent attempt has stopped.")
        await scope.finish(seq, failed, max(0, int((time.monotonic() - started) * 1000)))
        if not scope.active:
            raise AgentDispatchStoppedError("The agent attempt has stopped.")
