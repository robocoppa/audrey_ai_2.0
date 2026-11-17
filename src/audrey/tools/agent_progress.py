"""One bounded generated working note within an authorized saved Agent Work stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from audrey.agent_dispatch import AgentDispatchStoppedError, agent_dispatch_bound
from audrey.agent_tool_authority import require_agent_tool_authority
from audrey.app_state.agents import AgentTaskContextError, AgentTaskError
from audrey.identity import Principal
from audrey.processing_policy import current_processing_policy
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec


class SaveProgressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    note: str = Field(
        min_length=1, max_length=4000,
        description="A concise user-visible working note, at most 4,000 UTF-8 bytes; no raw reasoning or large source excerpts.",
    )

    @field_validator("note")
    @classmethod
    def bounded_note(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("A nonempty bounded working note is required.")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("A valid UTF-8 working note is required.") from None
        if size > 4000:
            raise ValueError("A bounded working note is required.")
        return value


@dataclass
class _Binding:
    app: Any
    principal: Principal
    task: dict
    active: bool = True


_BINDING: ContextVar[_Binding | None] = ContextVar("audrey_agent_progress", default=None)


def _guard(binding, user_id):
    if (binding is None or not binding.active or _BINDING.get() is not binding
            or current_processing_policy() is not None
            or user_id != binding.principal.storage_namespace):
        raise AgentDispatchStoppedError("The agent progress request has stopped.")
    require_agent_tool_authority(
        binding.app, binding.principal, binding.task, stages={"research"},
        unavailable_message="Saving Agent progress is unavailable.",
    )


@contextmanager
def agent_progress_scope(*, app: Any, principal: Principal, task: dict) -> Iterator[None]:
    if (not isinstance(principal, Principal) or principal.status != "active"
            or not principal.storage_namespace or task.get("user_id") != principal.user_id
            or task.get("current_step") != "research" or _BINDING.get() is not None):
        raise AgentDispatchStoppedError("Saving Agent progress is unavailable.")
    binding = _Binding(app, principal, dict(task))
    token = _BINDING.set(binding)
    try:
        _guard(binding, principal.storage_namespace)
        yield
    finally:
        binding.active = False
        _BINDING.reset(token)


def agent_progress_handler(*, user_id: str | None):
    binding = _BINDING.get()
    if binding is None:
        if agent_dispatch_bound() and current_processing_policy() is not None:
            raise AgentDispatchStoppedError("Agents are unavailable in this processing mode.")
        return None
    _guard(binding, user_id)

    async def save(arguments: dict) -> dict:
        _guard(binding, user_id)
        try:
            request = SaveProgressRequest.model_validate(arguments)
        except ValidationError:
            _guard(binding, user_id)
            return {"error": "invalid_progress", "detail": (
                "Supply only a nonempty working note of at most 4,000 UTF-8 bytes."
            )}
        lease = {key: binding.task[key] for key in ("lease_token", "fence")}
        lease["task_id"] = binding.task["id"]
        try:
            result = await binding.app.state.application_store.agents.save_work_progress(
                **lease, note=request.note, scope_guard=lambda: _guard(binding, user_id),
            )
        except AgentTaskContextError as error:
            raise AgentDispatchStoppedError(str(error)) from None
        except AgentTaskError:
            raise AgentDispatchStoppedError("Agent access, inputs or execution limits changed.") from None
        _guard(binding, user_id)
        return {"saved": True, "saved_at": result["saved_at"], "chars": len(result["note"]),
                "bytes": len(result["note"].encode("utf-8"))}

    return save


def with_agent_progress(tools: ToolRegistry) -> ToolRegistry:
    declaration = TOOL_DECLARATIONS["save_progress"]
    return ToolRegistry({**tools.by_name, "save_progress": ToolSpec(
        name="save_progress",
        description=("Save or replace one concise intermediate draft or working note for this Agent's Work stage. "
                     "Use when useful before a user question or to retain useful intermediate work. "
                     "The latest note survives interruption, but does not complete Work or restore a tool transcript. "
                     "Save user-visible work only, not hidden reasoning, secrets or large source caches. "
                     "The server returns a saved-at timestamp; saved progress is generated, unverified working data."),
        parameters=SaveProgressRequest.model_json_schema(), server_url="http://native.invalid", path="/save_progress",
        user_scope=declaration.user_scope, dependencies=declaration.dependencies, in_process_only=True,
    )})
