"""Canonical authority shared by native saved-Agent tools."""

from __future__ import annotations

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state.agents import AgentTaskContextError, AgentTaskError
from audrey.model_catalog import _apply_policy, _apply_profile, _model_allows, configured_models
from audrey.processing_policy import current_processing_policy
from audrey.user_data_visibility import remote_personal_reads_blocked


def require_agent_tool_authority(app, principal, task, *, stages, unavailable_message):
    """Revalidate identity, fixed inputs, lease, time, stage and local model grants."""
    from audrey.agent_executor import agent_settings

    if current_processing_policy() is not None or not agent_settings(app).enabled:
        raise AgentDispatchStoppedError("Agents are unavailable in this processing mode.")
    store = app.state.application_store
    lease = {key: task[key] for key in ("lease_token", "fence")}
    lease["task_id"] = task["id"]
    with store._lock:
        try:
            store.agents.fence_execution(**lease)
            record = store.agents._lease(**lease)
            current = store._get_principal_sync(principal.user_id)
        except AgentTaskContextError as error:
            # These server-defined source-change messages explain recovery.
            raise AgentDispatchStoppedError(str(error)) from None
        except AgentTaskError:
            raise AgentDispatchStoppedError("Agent access, inputs or execution limits changed.") from None
        if (current is None or current.status != "active" or "bots" in current.groups
                or current.storage_namespace != principal.storage_namespace
                or record["user_id"] != principal.user_id or task.get("user_id") != principal.user_id
                or task.get("current_step") not in stages
                or store.agents._internal(record)["current_step"] != task["current_step"]
                or remote_personal_reads_blocked(principal.storage_namespace)):
            raise AgentDispatchStoppedError(unavailable_message)
        model = next((item for item in configured_models(app.state.cfg) if item.id == "local"), None)
        policy = next((item for item in store._list_model_access_policies_sync() if item.model_id == "local"), None)
        profile = next((item for item in store._list_model_publication_profiles_sync() if item.model_id == "local"), None)
        if model is None:
            raise AgentDispatchStoppedError("Local model access is unavailable.")
        model = _apply_profile(_apply_policy(model, policy), profile)
        if not model.enabled or not _model_allows(model, current):
            raise AgentDispatchStoppedError("Local model access is unavailable.")
        return record
