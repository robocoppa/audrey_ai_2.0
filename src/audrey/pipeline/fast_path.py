"""Fast path — single-model generation for non-complex prompts.

Phase 5 scope: classify → pick highest-priority healthy model for that
task type → forward to Ollama (non-streaming). Streaming happens at the
route layer by streaming the same Ollama call.

Tools (ReAct loop, context compression) are layered in Phase 7 — the
hook is here as `use_tools=False` for now.
"""

from __future__ import annotations

import logging
from typing import Any

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry, TaskType

log = logging.getLogger(__name__)


async def run_fast_path(
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    """Return (concrete_model, ollama_response). Raises if no model is healthy."""
    spec = registry.first_healthy(task, health.is_healthy)
    if spec is None:
        raise OllamaError(f"No healthy model available for task={task}")

    log.info("fast_path task=%s -> %s", task, spec.name)
    try:
        resp = await ollama.chat(
            model=spec.name,
            messages=messages,
            options=options or None,
            timeout_s=timeout_s,
        )
        health.record_success(spec.name)
        return spec.name, resp
    except OllamaError as e:
        health.record_failure(spec.name, str(e))
        raise


__all__ = ["run_fast_path"]
