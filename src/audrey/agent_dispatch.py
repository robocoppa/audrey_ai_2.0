"""Optional live authorization for tools dispatched by a saved agent attempt."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


class AgentDispatchStoppedError(RuntimeError):
    """The saved attempt no longer authorizes content dispatch."""


@dataclass
class _Scope:
    check: Callable[[str], Awaitable[None]]
    active: bool = True


_scope: ContextVar[_Scope | None] = ContextVar("agent_dispatch_scope", default=None)


@contextmanager
def agent_dispatch_scope(check: Callable[[str], Awaitable[None]]) -> Iterator[None]:
    scope = _Scope(check)
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
