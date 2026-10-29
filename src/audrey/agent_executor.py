"""Saved research agents use verified local qwen3.8 for every model step."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from audrey.agent_dispatch import AgentDispatchStoppedError, agent_dispatch_scope
from audrey.model_catalog import resolve_model
from audrey.models.ollama import OllamaClient, OllamaError, generation_budget
from audrey.pipeline.prompts import AGENT_PLAN_SYSTEM, AGENT_RESEARCH_SYSTEM, AGENT_REVIEW_SYSTEM
from audrey.processing_policy import (
    ProcessingPolicy,
    ProcessingPolicyError,
    current_processing_policy,
)
from audrey.tools.discovery import ToolRegistry

PUBLIC_RESEARCH_TOOLS = frozenset({"web_search", "web_fetch"})
logger = logging.getLogger(__name__)


def _require_complete_text(
    step_id: str, result: Any, reason: str, *, thinking: bool = False
) -> None:
    label = step_id.capitalize()
    if reason == "length":
        raise AgentDispatchStoppedError(
            f"{label} reached the local model's output limit before finishing."
        )
    if not isinstance(result, str):
        raise AgentDispatchStoppedError(f"{label} returned invalid report text.")
    if not result.strip():
        detail = " after reasoning" if thinking else ""
        raise AgentDispatchStoppedError(f"{label} returned no report text{detail}.")


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = True
    model: Literal["qwen3.8:latest"] = "qwen3.8:latest"
    poll_seconds: float = Field(default=2.0, gt=0, le=60)
    lease_seconds: int = Field(default=60, ge=2, le=300)
    max_attempts: int = Field(default=6, ge=1, le=20)
    step_timeout_seconds: float = Field(default=600.0, gt=0, le=1800)
    max_operations: int = Field(default=96, ge=3, le=1000)
    max_active_seconds: int = Field(default=1800, ge=30, le=7200)
    max_queued_per_user: int = Field(default=5, ge=1, le=20)
    max_queued_global: int = Field(default=50, ge=1, le=1000)
    max_tokens_per_call: int = Field(default=8192, ge=128, le=8192)
    plan_max_tokens_per_call: int = Field(default=2048, ge=128, le=8192)
    max_tool_rounds: int = Field(default=6, ge=1, le=12)

    def tokens_for_step(self, step_id: str) -> int:
        if step_id == "plan":
            return min(self.plan_max_tokens_per_call, self.max_tokens_per_call)
        return self.max_tokens_per_call

    def task_limits(self) -> dict[str, int]:
        return {
            key: getattr(self, key)
            for key in (
                "max_operations",
                "max_active_seconds",
                "max_queued_per_user",
                "max_queued_global",
            )
        }


def agent_settings(app: Any) -> AgentSettings:
    return AgentSettings.model_validate(app.state.cfg.raw.get("agent_tasks", {}) or {})


async def research_authority(app: Any, user_id: str) -> Any:
    store = app.state.application_store
    principal = await store.get_principal(user_id=user_id)
    if principal is None or principal.status != "active" or principal.role == "bot":
        raise AgentDispatchStoppedError("Agent account access is unavailable.")
    if not agent_settings(app).enabled or current_processing_policy() is not None:
        raise AgentDispatchStoppedError("Public research agents are unavailable.")
    if await resolve_model(app.state.cfg, store, principal, "local") is None:
        raise AgentDispatchStoppedError("Local model access is unavailable.")
    tools = getattr(app.state, "tools", ToolRegistry())
    if any(tools.get(name) is None for name in PUBLIC_RESEARCH_TOOLS):
        raise AgentDispatchStoppedError("Public research tools are unavailable.")
    return principal


async def _verify_local(client, model, endpoint, policy=None):
    """Verify installed GGUF weights and pin their digest for this step."""
    try:
        installed = await client.tags()
        if policy is None:
            matches = [
                row for row in installed if row.get("name") == model or row.get("model") == model
            ]
            if len(matches) != 1:
                raise AgentDispatchStoppedError("Local qwen3.8 is not installed.")
            policy = ProcessingPolicy(
                local_models={model: matches[0].get("digest")}, ollama_base_url=endpoint
            )
        policy.assert_local_model_metadata(
            model, installed_models=installed, metadata=await client.show(model)
        )
        return policy
    except (OllamaError, ProcessingPolicyError, TypeError, AttributeError):
        raise AgentDispatchStoppedError(
            "Local qwen3.8 could not be verified. Cloud fallback is disabled."
        ) from None


class _LocalOnlyClient:
    def __init__(self, client, model, *, task_id, step_id):
        self.client, self.model = client, model
        self.task_id, self.step_id = task_id, step_id
        self.calls = 0

    async def chat(self, *, model, **kwargs):
        if model != self.model:
            raise AgentDispatchStoppedError("Agents may only use local qwen3.8.")
        self.calls += 1
        try:
            response = await self.client.chat(model=model, **kwargs)
        except OllamaError as error:
            logger.warning(
                "Agent model request failed task=%s step=%s call=%d error_type=%s",
                self.task_id,
                self.step_id,
                self.calls,
                type(error).__name__,
            )
            raise
        message = response.get("message")
        message = message if isinstance(message, dict) else {}
        content, thinking = message.get("content", ""), message.get("thinking", "")
        calls = message.get("tool_calls")
        reason = response.get("done_reason", "")
        # Provider strings and report/reasoning bodies must never enter logs.
        stop = (
            reason
            if isinstance(reason, str) and reason in {"stop", "length", "load", "unload"}
            else "unknown"
        )
        usage = {
            key: value if type(value := response.get(key)) is int and value >= 0 else None
            for key in ("prompt_eval_count", "eval_count")
        }
        logger.info(
            "Agent model response task=%s step=%s call=%d stop=%s done=%s "
            "input_tokens=%s output_tokens=%s content_chars=%s thinking_chars=%s tool_calls=%d",
            self.task_id,
            self.step_id,
            self.calls,
            stop,
            response.get("done") is True,
            usage["prompt_eval_count"],
            usage["eval_count"],
            len(content) if isinstance(content, str) else None,
            len(thinking) if isinstance(thinking, str) else None,
            len(calls) if isinstance(calls, list) else 0,
        )
        # Incomplete tool arguments must not reach HTTP dispatch either.
        if reason == "length" or not calls:
            _require_complete_text(self.step_id, content, reason, thinking=bool(thinking))
        return response


class ResearchAgentExecutor:
    """One explicit local model; no classifier, model panel or fallback."""

    def __init__(self, app, *, client_factory=None, react_runner=None):
        self.app = app
        self.client_factory = client_factory or OllamaClient
        self.react_runner = react_runner

    async def __call__(self, task, step_id):
        from audrey.pipeline.react import run_react

        settings = agent_settings(self.app)
        repository = self.app.state.application_store.agents
        lease = {"task_id": task["id"], "lease_token": task["lease_token"], "fence": task["fence"]}
        principal = await research_authority(self.app, task["user_id"])
        await repository.validate_lease(**lease)
        prompts = {
            "plan": AGENT_PLAN_SYSTEM,
            "research": AGENT_RESEARCH_SYSTEM,
            "review": AGENT_REVIEW_SYSTEM,
        }
        if step_id not in prompts:
            raise AgentDispatchStoppedError("Unsupported agent step.")
        max_tokens = settings.tokens_for_step(step_id)
        tools = self.app.state.tools.restrict(
            PUBLIC_RESEARCH_TOOLS if step_id == "research" else frozenset()
        )
        endpoints = {spec.name: (spec.server_url, spec.path) for spec in tools.specs()}
        previous = {
            step["id"]: step["result"] for step in task["steps"] if step["status"] == "completed"
        }
        messages = [
            {"role": "system", "content": prompts[step_id]},
            {
                "role": "user",
                "content": json.dumps(
                    {"objective": task["objective"], "saved_work": previous}, ensure_ascii=False
                ),
            },
        ]
        endpoint = self.app.state.cfg.env.ollama_host
        client = self.client_factory(endpoint, trust_env=False)
        try:
            policy = await _verify_local(client, settings.model, endpoint)
            local = _LocalOnlyClient(client, settings.model, task_id=task["id"], step_id=step_id)

            async def authorize():
                await repository.validate_lease(**lease)
                await research_authority(self.app, task["user_id"])

            async def check_generation():
                await _verify_local(client, settings.model, endpoint, policy)
                await authorize()
                await repository.reserve_operation(**lease, step_id=step_id)
                return max_tokens

            async def record_generation(_count):
                await authorize()

            async def check_tool(name):
                await authorize()
                current = self.app.state.tools.get(name)
                if (
                    name not in PUBLIC_RESEARCH_TOOLS
                    or current is None
                    or endpoints.get(name) != (current.server_url, current.path)
                ):
                    raise AgentDispatchStoppedError("This tool is unavailable to the agent.")
                await repository.reserve_operation(**lease, step_id=step_id)

            options = {"num_predict": max_tokens}
            retrieved = []
            with (
                agent_dispatch_scope(check_tool),
                generation_budget(check_generation, record_generation),
            ):
                async with self.app.state.inflight.slot(principal.storage_namespace):
                    if step_id == "research":
                        report = await (self.react_runner or run_react)(
                            local,
                            self.app.state.health,
                            tools,
                            model=settings.model,
                            messages=messages,
                            options=options,
                            timeout_s=settings.step_timeout_seconds,
                            max_rounds=settings.max_tool_rounds,
                            compress_after_round=settings.max_tool_rounds + 1,
                            compress_keep_last=20,
                            max_tool_result_chars=6000,
                            tool_dispatch_timeout_s=30,
                            user_id=principal.storage_namespace,
                            gate=self.app.state.gate,
                            location="local",
                            max_web_searches=4,
                        )
                        result, reason, retrieved = (
                            report.content,
                            report.done_reason,
                            report.retrieved,
                        )
                    else:
                        async with self.app.state.gate.acquire(
                            settings.model, location="local", user_id=principal.storage_namespace
                        ):
                            response = await local.chat(
                                model=settings.model,
                                messages=messages,
                                options=options,
                                timeout_s=settings.step_timeout_seconds,
                            )
                        result = (response.get("message") or {}).get("content") or ""
                        reason = response.get("done_reason", "")
            await authorize()
            _require_complete_text(step_id, result, reason)
            if len(result.encode("utf-8")) > 60000:
                raise AgentDispatchStoppedError("The report exceeded its saved-output limit.")
            if step_id == "research":
                urls, size = [], 0
                for source in retrieved:
                    url = source.get("url", "")
                    if not isinstance(url, str):
                        continue
                    try:
                        parsed = urlsplit(url)
                    except (ValueError, TypeError):
                        continue
                    length = len(url.encode())
                    if (
                        parsed.scheme in {"https", "http"}
                        and parsed.hostname
                        and parsed.username is None
                        and parsed.password is None
                        and length <= 2048
                        and size + length <= 3000
                        and url not in urls
                        and len(urls) < 10
                    ):
                        urls.append(url)
                        size += length
                if not urls:
                    raise AgentDispatchStoppedError(
                        "The research step returned no public source evidence."
                    )
                result += "\n\nSelected public sources:\n" + "\n".join(f"- {url}" for url in urls)
            return result.strip()
        finally:
            await client.aclose()
