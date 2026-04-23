"""Fast path — single-model generation, optionally with ReAct tool use.

Phase 7 update: when the chosen model is in `fast_path.tool_capable_models`
*and* a tool registry is non-empty, we run a ReAct loop (`pipeline/react.py`)
that lets the model call tools before answering. Otherwise it's a one-shot
`ollama.chat`. Streaming still bypasses both (route-layer concern).
"""

from __future__ import annotations

import logging
from typing import Any

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry, ModelSpec, TaskType
from audrey.pipeline.react import ReactResult, run_react
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)


def pick_fast_model(
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
) -> ModelSpec:
    spec = registry.first_healthy(task, health.is_healthy)
    if spec is None:
        raise OllamaError(f"No healthy model available for task={task}")
    return spec


async def run_fast_path(
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
    tools: ToolRegistry | None = None,
    tool_capable_models: set[str] | None = None,
    react_max_rounds: int = 3,
    react_compress_after: int = 2,
    react_max_tool_chars: int = 2000,
    react_dispatch_timeout_s: float = 30.0,
) -> tuple[str, dict[str, Any]]:
    """Return (concrete_model, response_like_dict).

    The returned dict has the shape of an Ollama chat response (`message`,
    `prompt_eval_count`, `eval_count`) so the caller (graph node) doesn't
    care whether ReAct ran or not. When ReAct ran, an extra key
    `_react` carries the loop metadata (rounds, tool calls).
    """
    spec = pick_fast_model(registry, health, task=task)
    use_tools = bool(
        tools and tools.by_name
        and tool_capable_models is not None
        and spec.name in tool_capable_models
    )
    log.info("fast_path task=%s -> %s (tools=%s)", task, spec.name, "on" if use_tools else "off")

    if not use_tools:
        try:
            resp = await ollama.chat(
                model=spec.name, messages=messages,
                options=options or None, timeout_s=timeout_s,
            )
            health.record_success(spec.name)
            return spec.name, resp
        except OllamaError as e:
            health.record_failure(spec.name, str(e))
            raise

    react: ReactResult = await run_react(
        ollama, health, tools,  # type: ignore[arg-type]
        model=spec.name,
        messages=messages,
        options=options,
        timeout_s=timeout_s,
        max_rounds=react_max_rounds,
        compress_after_round=react_compress_after,
        max_tool_result_chars=react_max_tool_chars,
        tool_dispatch_timeout_s=react_dispatch_timeout_s,
    )
    return spec.name, {
        "message": {"role": "assistant", "content": react.content},
        "prompt_eval_count": react.prompt_eval_count,
        "eval_count": react.eval_count,
        "_react": {
            "tool_rounds": react.tool_rounds,
            "tool_calls": [
                {"name": r.name, "elapsed_s": r.elapsed_s, "is_error": r.is_error}
                for r in react.tool_calls
            ],
        },
    }


__all__ = ["run_fast_path", "pick_fast_model"]
