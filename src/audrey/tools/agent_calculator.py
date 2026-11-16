"""Pure bounded calculations within one revocable saved Agent Work/Review stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state.agents import AgentTaskContextError, AgentTaskError
from audrey.identity import Principal
from audrey.model_catalog import _apply_policy, _apply_profile, _model_allows, configured_models
from audrey.processing_policy import current_processing_policy
from audrey.tools.calculator import CalculateRequest, CalculationError, calculate_value
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec, _resolve_refs
from audrey.user_data_visibility import remote_personal_reads_blocked


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
    from audrey.agent_executor import agent_settings

    if (binding is None or not binding.active or _BINDING.get() is not binding
            or current_processing_policy() is not None
            or user_id != binding.principal.storage_namespace):
        raise AgentDispatchStoppedError("The agent calculation has stopped.")
    app, principal, task = binding.app, binding.principal, binding.task
    store = app.state.application_store
    if not agent_settings(app).enabled:
        raise AgentDispatchStoppedError("Agents are unavailable in this processing mode.")
    lease = {key: task[key] for key in ("lease_token", "fence")}
    lease["task_id"] = task["id"]
    with store._lock:
        try:
            store.agents.fence_execution(**lease)
            record = store.agents._lease(**lease)
            current = store._get_principal_sync(principal.user_id)
        except AgentTaskContextError as error:
            # Source-change messages are server-defined and explain recovery.
            # Keep them while using the stop type that aborts native dispatch.
            raise AgentDispatchStoppedError(str(error)) from None
        except AgentTaskError:
            raise AgentDispatchStoppedError("Agent access, inputs or execution limits changed.") from None
        if (current is None or current.status != "active" or "bots" in current.groups
                or current.storage_namespace != principal.storage_namespace
                or record["user_id"] != principal.user_id or task.get("user_id") != principal.user_id
                or task.get("current_step") not in {"research", "review"}
                or store.agents._internal(record)["current_step"] != task["current_step"]
                or remote_personal_reads_blocked(principal.storage_namespace)):
            raise AgentDispatchStoppedError("Agent calculation is unavailable.")
        model = next((item for item in configured_models(app.state.cfg) if item.id == "local"), None)
        policy = next((item for item in store._list_model_access_policies_sync() if item.model_id == "local"), None)
        profile = next((item for item in store._list_model_publication_profiles_sync() if item.model_id == "local"), None)
        if model is None:
            raise AgentDispatchStoppedError("Local model access is unavailable.")
        model = _apply_profile(_apply_policy(model, policy), profile)
        if not model.enabled or not _model_allows(model, current):
            raise AgentDispatchStoppedError("Local model access is unavailable.")


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
