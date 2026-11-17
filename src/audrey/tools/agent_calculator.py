"""Pure bounded calculations within one revocable saved Agent Work/Review stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.agent_tool_authority import require_agent_tool_authority
from audrey.identity import Principal
from audrey.processing_policy import current_processing_policy
from audrey.tools.calculator import CalculateRequest, CalculationError, calculate_value
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec, _resolve_refs


@dataclass
class _Binding:
    app: Any
    principal: Principal
    task: dict
    active: bool = True


_BINDING: ContextVar[_Binding | None] = ContextVar("audrey_agent_calculator", default=None)


def agent_calculator_bound() -> bool:
    """Include inherited expired bindings so a callback cannot fall back to HTTP."""
    return _BINDING.get() is not None


def _guard(binding, user_id):
    if (binding is None or not binding.active or _BINDING.get() is not binding
            or current_processing_policy() is not None
            or user_id != binding.principal.storage_namespace):
        raise AgentDispatchStoppedError("The agent calculation has stopped.")
    require_agent_tool_authority(
        binding.app, binding.principal, binding.task, stages={"research", "review"},
        unavailable_message="Agent calculation is unavailable.",
    )


@contextmanager
def agent_calculator_scope(*, app: Any, principal: Principal, task: dict) -> Iterator[None]:
    """Bind task identity; inherited callbacks expire together at scope exit."""
    if (not isinstance(principal, Principal) or principal.status != "active"
            or not principal.storage_namespace or task.get("user_id") != principal.user_id
            or task.get("current_step") not in {"research", "review"}
            or _BINDING.get() is not None):
        raise AgentDispatchStoppedError("Agent calculation is unavailable.")
    binding = _Binding(app, principal, dict(task))
    token = _BINDING.set(binding)
    try:
        _guard(binding, principal.storage_namespace)
        yield
    finally:
        binding.active = False
        _BINDING.reset(token)


def agent_calculator_handler(*, user_id: str | None):
    binding = _BINDING.get()
    if binding is None:
        return None
    _guard(binding, user_id)

    async def calculate(arguments: dict) -> dict:
        _guard(binding, user_id)
        try:
            result = calculate_value(CalculateRequest.model_validate(arguments)).model_dump()
        except (ValidationError, CalculationError):
            _guard(binding, user_id)
            return {"error": "invalid_calculation", "detail": (
                "Use bounded decimal arithmetic, supported unit conversions, or a linear/quadratic equation."
            )}
        _guard(binding, user_id)
        return result

    return calculate


def with_agent_calculator(tools: ToolRegistry) -> ToolRegistry:
    """Replace only this Agent's spec; ordinary chat keeps its existing HTTP tool."""
    schema = CalculateRequest.model_json_schema(ref_template="#/components/schemas/{model}")
    parameters = _resolve_refs(schema, schema.get("$defs", {}))
    parameters.pop("$defs", None)
    declaration = TOOL_DECLARATIONS["calculate"]
    spec = ToolSpec(
        name="calculate",
        description=("Calculate exact bounded decimal arithmetic, percentages, common unit conversions, "
                     "or real roots of linear/quadratic equations in x. Use for nontrivial totals, "
                     "budgets, rates and conversions instead of estimating. Results use decimal "
                     "strings and 40 significant digits. No file, network or code access. "
                     "The result verifies the supplied calculation, not its source facts."),
        parameters=parameters, server_url="http://native.invalid", path="/calculate",
        user_scope=declaration.user_scope, dependencies=declaration.dependencies, in_process_only=True,
    )
    return ToolRegistry({**tools.by_name, "calculate": spec})


__all__ = ["agent_calculator_bound", "agent_calculator_handler", "agent_calculator_scope", "with_agent_calculator"]
