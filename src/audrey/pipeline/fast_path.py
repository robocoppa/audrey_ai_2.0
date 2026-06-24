"""Fast path — single-model generation, optionally with ReAct tool use.

When the chosen model is in `fast_path.tool_capable_models` *and* the tool
registry is non-empty, we run a ReAct loop (`pipeline/react.py`) that lets
the model call tools before answering. Otherwise it's a one-shot
`ollama.chat`. Streaming bypasses both (route-layer concern).

Local calls go through `FairLocalGate`. The non-tools branch holds the
gate around the single `ollama.chat`. The tools branch passes the gate
down into `run_react`, which acquires per-chat so the gate is released
during tool dispatch. Cloud calls bypass the gate entirely
(`gate.acquire` is a no-op when `location != "local"`).
"""

from __future__ import annotations

import logging
from typing import Any

from audrey.metrics import dispatch_total
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry, ModelSpec, TaskType
from audrey.pipeline.fair_gate import FairLocalGate
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


# How many models the non-tools fast path will try before giving up. The
# top healthy model plus one fallback: if the two highest-priority healthy
# models both error mid-request, a third is unlikely to help and the added
# latency is real. Mirrors the deep panel's emergency-fallback cap of 2.
_FAST_FALLBACK_LIMIT = 2


def _healthy_fast_candidates(
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
    limit: int,
) -> list[ModelSpec]:
    """Up to `limit` healthy candidates for `task`, highest priority first.

    Reads health at selection time (same as `pick_fast_model`); a model
    cooled down by an earlier attempt in this same request is skipped on the
    next pass because `_run_one_chat` records the failure before we loop.
    """
    out: list[ModelSpec] = []
    for spec in registry.candidates(task):
        if health.is_healthy(spec.name):
            out.append(spec)
            if len(out) >= limit:
                break
    return out


async def run_fast_path(
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
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
    react_compress_keep_last: int = 1,
    user_id: str | None = None,
    cfg: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Return (concrete_model, response_like_dict).

    The returned dict has the shape of an Ollama chat response (`message`,
    `prompt_eval_count`, `eval_count`) so the caller (graph node) doesn't
    care whether ReAct ran or not. When ReAct ran, an extra key
    `_react` carries the loop metadata (rounds, tool calls).

    **Non-tools branch — bounded fallback.** The top healthy model is tried
    first; if it raises `OllamaError`, it's cooled down and the next healthy
    candidate is tried, up to `_FAST_FALLBACK_LIMIT` models. Only when every
    candidate has failed does the error propagate (the graph node turns it
    into a 502). A transient blip on the highest-priority model no longer
    fails the request when a healthy fallback exists.

    **Tools branch — single-shot.** The ReAct branch is *not* retried across
    models: a mid-loop failure can land after tool side effects (e.g.
    `memory_store`), so a blind model-swap could double-apply them. The
    tool-capable model picked first is the one that answers, and an
    `OllamaError` from it propagates as before. (The failure still cools the
    model down, so the *next* request routes around it.)
    """
    spec = pick_fast_model(registry, health, task=task)
    use_tools = bool(
        tools and tools.by_name
        and tool_capable_models is not None
        and spec.name in tool_capable_models
    )

    if not use_tools:
        # Try the top healthy model, then fall back to the next healthy
        # candidate if it errors. `spec` (already chosen above) is the first
        # entry of this list, so the happy path is identical to before:
        # one model, one chat call, one dispatch increment.
        candidates = _healthy_fast_candidates(
            registry, health, task=task, limit=_FAST_FALLBACK_LIMIT,
        )
        last_err: OllamaError | None = None
        for attempt, cand in enumerate(candidates, start=1):
            log.info(
                "fast_path task=%s -> %s (tools=off, attempt %d/%d)",
                task, cand.name, attempt, len(candidates),
            )
            dispatch_total.labels(
                model=cand.name, task_type=str(task), path="fast",
            ).inc()
            try:
                async with gate.acquire(cand.name, location=cand.location, user_id=user_id):
                    resp = await ollama.chat(
                        model=cand.name, messages=messages,
                        options=options or None, timeout_s=timeout_s,
                    )
                health.record_success(cand.name)
                return cand.name, resp
            except OllamaError as e:
                health.record_failure(cand.name, str(e))
                last_err = e
                log.warning(
                    "fast_path: %s failed (attempt %d/%d): %s",
                    cand.name, attempt, len(candidates), e,
                )
        # Every healthy candidate failed. Re-raise the last error (or a
        # no-healthy-model error if the registry had nothing to try).
        raise last_err or OllamaError(f"No healthy model available for task={task}")

    log.info("fast_path task=%s -> %s (tools=on)", task, spec.name)
    dispatch_total.labels(
        model=spec.name, task_type=str(task), path="fast_react",
    ).inc()
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
        compress_keep_last=react_compress_keep_last,
        user_id=user_id,
        gate=gate,
        location=spec.location,
        cfg=cfg,
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
