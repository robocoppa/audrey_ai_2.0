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
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.semaphore import GpuGate
from audrey.pipeline.state import TaskType, WorkerDraft

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
) -> WorkerDraft:
    """Execute one worker. Always returns a WorkerDraft — never raises."""
    start = time.monotonic()
    try:
        async with gate.acquire(model, location=location):
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
        )
    except OllamaError as e:
        elapsed = round(time.monotonic() - start, 2)
        health.record_failure(model, str(e))
        log.warning("deep_panel: worker %s failed in %.2fs: %s", model, elapsed, e)
        return WorkerDraft(model=model, content="", error=str(e)[:300], elapsed_s=elapsed)


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
) -> tuple[list[WorkerDraft], list[str]]:
    """Run the panel and return (drafts, attempted_models).

    `drafts` includes both successes and per-worker errors so callers can
    decide whether enough material exists to synthesize.
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
            if len(workers) >= 3:
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

    coros = [
        _run_one_worker(
            ollama, health, gate,
            model=name, location=loc,
            messages=per_worker_messages[i],
            options=options,
            timeout_s=timeout_s,
        )
        for i, (name, loc) in enumerate(workers)
    ]
    drafts = await asyncio.gather(*coros)
    attempted = [name for name, _ in workers]
    return list(drafts), attempted


__all__ = ["pool_key_for", "select_workers", "run_panel"]
