"""Deep panel — parallel worker dispatch + pool selection.

Picks the right worker pool from the YAML config based on virtual model:
  - audrey_deep   → deep_panel         (mixed local + cloud)
  - audrey_cloud  → deep_panel_cloud   (cloud-only)
  - audrey_local  → deep_panel_local   (local-only)

Each pool is keyed by task type (`code`, `reasoning`, `general`, `vl`) and
yields a list of worker model names plus a synthesizer (handled in `synthesize`).
Unhealthy or missing models are skipped; the panel keeps going as long as at
least one draft comes back.

Concurrency:
  - Cloud workers run via `asyncio.gather` (Ollama Pro caps at 3, configurable).
  - Local workers are submitted concurrently but serialize through the GPU
    semaphore in `semaphore.py` (default `GPU_CONCURRENCY=1`).

Phase 9: workers are tool-capable. When a `ToolRegistry` is supplied and the
worker model is in `fast_path.tool_capable_models`, the worker runs a ReAct
loop (`pipeline/react.py`) with a tighter per-worker budget from
`agentic.react.deep_worker`. The GPU gate is held for the *entire* loop — not
just a single chat call — so local workers never overlap across tool rounds.
Tool-grounded drafts carry `tool_rounds` > 0 in their `WorkerDraft`.

If `state["subtasks"]` is non-empty, workers are assigned to subtasks
round-robin so each draft answers a different slice. Otherwise every worker
answers the full prompt — the synthesizer reconciles them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from audrey.config import Config
from audrey.metrics import dispatch_total
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.react import ReactResult, run_react
from audrey.pipeline.semaphore import GpuGate
from audrey.pipeline.state import TaskType, WorkerDraft
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)


# Map virtual model → pool key in config.yaml
_POOL_KEYS = {
    "audrey_deep": "deep_panel",
    "audrey_cloud": "deep_panel_cloud",
    "audrey_local": "deep_panel_local",
}


def pool_key_for(virtual_model: str) -> str:
    return _POOL_KEYS.get(virtual_model, "deep_panel")


def _location_of(model: str, registry: ModelRegistry) -> str:
    """Look up the registry-declared location of a model. Default: local."""
    for task in registry.all_task_types():
        for spec in registry.candidates(task):  # type: ignore[arg-type]
            if spec.name == model:
                return spec.location
    return "local"


def select_workers(
    cfg: Config,
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    pool_key: str,
    task: TaskType,
    max_workers_cloud: int,
) -> list[tuple[str, str]]:
    """Return [(model_name, location), ...] for healthy workers in this pool/task.

    Cloud workers are capped at `max_workers_cloud` (Ollama Pro concurrency).
    Local workers are not capped here — the GPU semaphore serializes them.
    """
    pool = cfg.raw.get(pool_key, {}).get(task, {})
    raw_workers: list[str] = list(pool.get("workers", []) or [])

    out: list[tuple[str, str]] = []
    cloud_count = 0
    for name in raw_workers:
        if not health.is_healthy(name):
            log.info("deep_panel: skipping unhealthy worker %s", name)
            continue
        loc = _location_of(name, registry)
        if loc == "cloud":
            if cloud_count >= max_workers_cloud:
                continue
            cloud_count += 1
        out.append((name, loc))
    return out


async def _run_one_worker(
    ollama: OllamaClient,
    health: HealthTracker,
    gate: GpuGate,
    *,
    model: str,
    location: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
    tools: ToolRegistry | None,
    tool_capable: bool,
    react_max_rounds: int,
    react_compress_after: int,
    react_max_tool_chars: int,
    react_dispatch_timeout_s: float,
    user_id: str | None = None,
) -> WorkerDraft:
    """Execute one worker. Always returns a WorkerDraft — never raises.

    If `tool_capable` is True and `tools` has entries, runs a ReAct loop;
    otherwise a single `ollama.chat`. In both cases the GPU gate is held for
    the full duration so local workers strictly serialize, even across
    ReAct rounds (VRAM fits one local model at a time).
    """
    start = time.monotonic()
    use_tools = bool(tool_capable and tools is not None and tools.by_name)
    try:
        async with gate.acquire(model, location=location):
            if use_tools:
                react: ReactResult = await run_react(
                    ollama, health, tools,  # type: ignore[arg-type]
                    model=model,
                    messages=messages,
                    options=options,
                    timeout_s=timeout_s,
                    max_rounds=react_max_rounds,
                    compress_after_round=react_compress_after,
                    max_tool_result_chars=react_max_tool_chars,
                    tool_dispatch_timeout_s=react_dispatch_timeout_s,
                    user_id=user_id,
                )
                elapsed = round(time.monotonic() - start, 2)
                # run_react already records success/failure per chat call.
                return WorkerDraft(
                    model=model,
                    content=react.content,
                    elapsed_s=elapsed,
                    prompt_eval_count=react.prompt_eval_count,
                    eval_count=react.eval_count,
                    tool_rounds=react.tool_rounds,
                    tool_calls=[
                        {"name": r.name, "elapsed_s": r.elapsed_s, "is_error": r.is_error}
                        for r in react.tool_calls
                    ],
                )

            resp = await ollama.chat(
                model=model,
                messages=messages,
                options=options or None,
                timeout_s=timeout_s,
            )
        elapsed = round(time.monotonic() - start, 2)
        msg = resp.get("message", {}) or {}
        content = msg.get("content", "") or ""
        health.record_success(model)
        return WorkerDraft(
            model=model,
            content=content,
            elapsed_s=elapsed,
            prompt_eval_count=int(resp.get("prompt_eval_count", 0) or 0),
            eval_count=int(resp.get("eval_count", 0) or 0),
            tool_rounds=0,
            tool_calls=[],
        )
    except OllamaError as e:
        elapsed = round(time.monotonic() - start, 2)
        health.record_failure(model, str(e))
        log.warning("deep_panel: worker %s failed in %.2fs: %s", model, elapsed, e)
        return WorkerDraft(
            model=model, content="", error=str(e)[:300], elapsed_s=elapsed,
            tool_rounds=0, tool_calls=[],
        )


def _messages_for_subtask(base_messages: list[dict[str, Any]], subtask: str) -> list[dict[str, Any]]:
    """Replace the last user message with the subtask question.

    Keeps any prior system/assistant context intact so the worker still has
    conversation history — only the focal question changes.
    """
    out: list[dict[str, Any]] = []
    replaced = False
    for m in reversed(base_messages):
        if not replaced and m.get("role") == "user":
            out.append({"role": "user", "content": subtask})
            replaced = True
        else:
            out.append(m)
    out.reverse()
    if not replaced:
        out.append({"role": "user", "content": subtask})
    return out


async def run_panel(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: GpuGate,
    *,
    pool_key: str,
    task: TaskType,
    messages: list[dict[str, Any]],
    subtasks: list[str],
    options: dict[str, Any],
    timeout_s: float,
    max_workers_cloud: int,
    tools: ToolRegistry | None = None,
    tool_capable_models: set[str] | None = None,
    react_max_rounds: int = 2,
    react_compress_after: int = 2,
    react_max_tool_chars: int = 2000,
    react_dispatch_timeout_s: float = 30.0,
    user_id: str | None = None,
) -> tuple[list[WorkerDraft], list[str]]:
    """Run the panel and return (drafts, attempted_models).

    `drafts` includes both successes and per-worker errors so callers can
    decide whether enough material exists to synthesize.

    Workers whose model name is in `tool_capable_models` and whose pool has
    a non-empty `ToolRegistry` run ReAct; others run a one-shot chat.
    """
    workers = select_workers(
        cfg, registry, health,
        pool_key=pool_key, task=task, max_workers_cloud=max_workers_cloud,
    )
    # If no workers from the pool are healthy, fall back to the registry's
    # top-N healthy models for this task so we always answer something.
    if not workers:
        log.warning("deep_panel: no healthy pool workers for %s/%s; falling back to registry", pool_key, task)
        for spec in registry.candidates(task):
            if not health.is_healthy(spec.name):
                continue
            workers.append((spec.name, spec.location))
            if len(workers) >= 2:
                break
    if not workers:
        return [], []

    if subtasks:
        per_worker_messages = [
            _messages_for_subtask(messages, subtasks[i % len(subtasks)])
            for i in range(len(workers))
        ]
    else:
        per_worker_messages = [messages] * len(workers)

    capable = tool_capable_models or set()
    for name, _loc in workers:
        dispatch_total.labels(
            model=name,
            task_type=str(task),
            path="deep_react" if name in capable else "deep",
        ).inc()
    coros = [
        _run_one_worker(
            ollama, health, gate,
            model=name, location=loc,
            messages=per_worker_messages[i],
            options=options,
            timeout_s=timeout_s,
            tools=tools,
            tool_capable=(name in capable),
            react_max_rounds=react_max_rounds,
            react_compress_after=react_compress_after,
            react_max_tool_chars=react_max_tool_chars,
            react_dispatch_timeout_s=react_dispatch_timeout_s,
            user_id=user_id,
        )
        for i, (name, loc) in enumerate(workers)
    ]
    drafts = await asyncio.gather(*coros)
    attempted = [name for name, _ in workers]
    return list(drafts), attempted


__all__ = ["pool_key_for", "select_workers", "run_panel"]
