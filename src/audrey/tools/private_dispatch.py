"""Task-owned local tool callbacks for a verified private operation.

A protected tool call never travels through the ordinary sidecar transport.
Handlers are supplied by server code after owner authorization, not by a model
or an HTTP request. Bindings expire on scope exit, including inherited tasks.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from audrey.processing_policy import (
    ProcessingPolicy,
    ProcessingPolicyError,
    current_processing_policy,
)

PrivateToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class PrivateToolArgumentsError(ValueError):
    """Content-free feedback for a callback's declared-schema validation failure."""

    code = "arguments_invalid"
    detail = (
        "The tool arguments do not match its declaration. Use only declared fields "
        "with the stated types and limits."
    )

    def __init__(self) -> None:
        super().__init__(self.detail)


@dataclass(slots=True)
class _ToolBinding:
    policy: ProcessingPolicy
    user_id: str
    handlers: Mapping[str, PrivateToolHandler]
    active: bool = True


_BINDING: ContextVar[_ToolBinding | None] = ContextVar("audrey_private_tools", default=None)


@contextmanager
def private_tool_scope(
    *, user_id: str, handlers: Mapping[str, PrivateToolHandler],
) -> Iterator[None]:
    """Bind server-owned callbacks to the exact active policy and owner."""
    policy = current_processing_policy()
    if (
        policy is None or not isinstance(user_id, str) or not user_id.strip()
        or user_id != user_id.strip() or not isinstance(handlers, Mapping)
    ):
        raise ProcessingPolicyError("private_tool_not_allowed")
    frozen = dict(handlers)
    for name, handler in frozen.items():
        policy.assert_tool_allowed(name)
        if not callable(handler):
            raise ProcessingPolicyError("private_tool_not_allowed")
    previous = _BINDING.get()
    if previous is not None and (
        not previous.active or previous.user_id != user_id
        or any(previous.handlers.get(name) is not handler for name, handler in frozen.items())
    ):
        raise ProcessingPolicyError("processing_policy_downgrade")
    binding = _ToolBinding(policy, user_id, MappingProxyType(frozen))
    token = _BINDING.set(binding)
    try:
        yield
    finally:
        binding.active = False
        _BINDING.reset(token)


def private_tool_handler(name: str, *, user_id: str | None) -> PrivateToolHandler:
    """Refuse missing, expired, differently owned or differently scoped work."""
    binding = _BINDING.get()
    policy = current_processing_policy()
    if (
        policy is None or binding is None or not binding.active
        or binding.policy is not policy or user_id != binding.user_id
        or name not in binding.handlers
    ):
        raise ProcessingPolicyError("private_tool_not_allowed")
    policy.assert_tool_allowed(name)
    return binding.handlers[name]


__all__ = ["PrivateToolArgumentsError", "PrivateToolHandler", "private_tool_scope"]
