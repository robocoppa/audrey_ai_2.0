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
  - Local workers are submitted concurrently but serialize through
    `FairLocalGate` (default `GPU_CONCURRENCY=1`).

Workers are tool-capable: when a `ToolRegistry` is supplied and the worker
model is in `fast_path.tool_capable_models`, the worker runs a ReAct loop
(`pipeline/react.py`) with a tighter per-worker budget from
`agentic.react.deep_worker`. The GPU gate is held for the *entire* loop —
not just a single chat call — so local workers never overlap across tool
rounds. Tool-grounded drafts carry `tool_rounds` > 0 in their `WorkerDraft`.
Those workers also receive `DEEP_WORKER_SYSTEM` (see `_with_worker_system`),
which tells them they're producing one draft of several — tool-free workers
run on the unmodified conversation, exactly as before.

If `state["subtasks"]` is non-empty, workers are assigned to subtasks
round-robin so each draft answers a different slice. Otherwise every worker
answers the full prompt — the synthesizer reconciles them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from audrey.config import Config
from audrey.metrics import dispatch_total
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.ledger import (
    Claim,
    FactCheckResult,
    ResearchResult,
    Source,
    SourceType,
    hedge_policy,
    inlined_schema,
    parse_factcheck_result,
    parse_research_result,
    usable_url,
)
from audrey.pipeline.messages import last_user_text
from audrey.pipeline.prompts import (
    DEEP_WORKER_SYSTEM,
    FACTCHECK_STRUCTURE_SYSTEM,
    FACTCHECK_SYSTEM,
    RESEARCH_STRUCTURE_SYSTEM,
    RESEARCHER_SYSTEM,
    VERIFIER_SYSTEM,
    WRITER_SYSTEM,
    prompt_from_config,
)
from audrey.pipeline.react import ReactResult, run_react
from audrey.pipeline.state import TaskType, WorkerDraft
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)


# Map virtual model → pool key in config.yaml
#
# The ADAPTIVE models are registered too, not just the forced-deep ones. They
# reach the panel by the token gate rather than by name, which is easy to forget
# when reading this map — and until 2026-08-09 they were absent, so every deep
# turn from `audrey_auto` logged "unknown virtual_model" and fell back to the
# pool it was going to get anyway. A warning that fires on correct behaviour
# stops being read, which is the whole cost: the case it exists to catch (a typo
# here, or a new model shipped without a pool) looks exactly the same.
_POOL_KEYS = {
    "audrey_auto": "deep_panel",
    "audrey_video": "deep_panel",
    "audrey_deep": "deep_panel",
    "audrey_cloud": "deep_panel_cloud",
    "audrey_local": "deep_panel_local",
    "audrey_research": "deep_panel_research",
}


def pool_key_for(virtual_model: str) -> str:
    pool = _POOL_KEYS.get(virtual_model)
    if pool is None:
        # Unknown virtual model — typo in config, or a new model added
        # without a pool registration. Fall back to the default pool so
        # something still answers, but log it so the operator notices.
        log.warning(
            "deep_panel: unknown virtual_model %r, falling back to default pool 'deep_panel'",
            virtual_model,
        )
        return "deep_panel"
    return pool


def pick_panel_timeout(cfg: Config, pool_key: str) -> float:
    """Pick the per-worker timeout for a deep-panel pool.

    `deep_panel_cloud` (cloud-only pool) uses `cfg.timeouts.cloud` because
    cloud workers run in parallel under Ollama Pro's concurrency cap.
    All other pools (`deep_panel`, `deep_panel_local`) use
    `cfg.timeouts.deep_worker` because at least one worker holds the local
    GPU gate and can't overlap with the others. Shared between
    `routes/openai.py` (streaming deep) and `pipeline/graph.py`
    (non-streaming deep) so the two paths can't drift.
    """
    cloud_timeout = float(cfg.timeouts.get("cloud", 120))
    deep_worker_timeout = float(cfg.timeouts.get("deep_worker", 240))
    return cloud_timeout if pool_key == "deep_panel_cloud" else deep_worker_timeout


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
        loc = registry.location_of(name)
        if loc == "cloud":
            if cloud_count >= max_workers_cloud:
                continue
            cloud_count += 1
        out.append((name, loc))
    return out


# A worker's inline thinking block. Ollama normally routes reasoning to a
# separate `thinking` field, but a cloud model has been observed emitting it
# in `content` — with the OPENING tag missing, i.e. `<reasoning>...</think>
# <final draft>` (2026-07-06 eval, glm on bio-euclid: the note carried its
# whole draft twice around a dangling `</think>`). Drafts feed structuring,
# the verifier, and the debug trace, so the leak doubles a worker's context
# contribution, not just its display.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(content: str) -> str:
    """Drop inline thinking from a worker reply.

    Well-formed `<think>…</think>` blocks are removed; a dangling `</think>`
    means everything before it was reasoning, so keep only what follows. If
    stripping would leave nothing (a reply wrapped entirely in think tags),
    return the original — an empty draft reads as a dropped worker, which is
    worse than a tagged one.
    """
    s = _THINK_BLOCK.sub("", content)
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    return s if s.strip() else content


async def _run_one_worker(
    ollama: OllamaClient,
    health: HealthTracker,
    gate: FairLocalGate,
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
    react_compress_keep_last: int = 1,
    react_max_web_searches: int = 0,
    user_id: str | None = None,
    cfg: Any = None,
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
        async with gate.acquire(model, location=location, user_id=user_id):
            if use_tools:
                # Deep panel holds the gate for the *whole* worker (this
                # `async with`), so pass `gate=None` to keep ReAct from
                # double-acquiring. Fast path does the opposite — it passes
                # a real gate so tool-dispatch windows release the GPU.
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
                    compress_keep_last=react_compress_keep_last,
                    max_web_searches=react_max_web_searches,
                    user_id=user_id,
                    gate=None,
                    location=location,
                    cfg=cfg,
                )
                elapsed = round(time.monotonic() - start, 2)
                # run_react already records success/failure per chat call.
                return WorkerDraft(
                    model=model,
                    content=_strip_think(react.content),
                    elapsed_s=elapsed,
                    prompt_eval_count=react.prompt_eval_count,
                    eval_count=react.eval_count,
                    tool_rounds=react.tool_rounds,
                    tool_calls=[
                        {"name": r.name, "elapsed_s": r.elapsed_s, "is_error": r.is_error}
                        for r in react.tool_calls
                    ],
                    web_search_chars=react.web_search_chars,
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
            content=_strip_think(content),
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

    Multi-turn contract: in a multi-turn conversation, "the last user
    message" is this turn's question (the one the planner just decomposed),
    not an arbitrary earlier turn. Earlier user/assistant pairs stay in
    place as history so the worker sees the full thread, but the focal
    question becomes the subtask. If no user message exists at all
    (degenerate input), the subtask is appended as a fresh user turn.
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


def _with_worker_system(
    messages: list[dict[str, Any]], role_prompt: str
) -> list[dict[str, Any]]:
    """Insert the worker role prompt AFTER any leading system messages.

    Deliberately different from `_with_role_system`, which research mode uses to
    prepend. `prompts.compose_system_messages` fixes the canonical order —
    incoming system messages first, so the user's own persona wins on tone, then
    the task role — and this is the `task_role` slot that composer left open
    (see its docstring). Prepending would put the panel's role ahead of the
    user's persona and quietly invert that.

    Returns a new list; `messages` is never mutated, so the caller can keep
    sharing one list across workers that don't get a role.
    """
    idx = 0
    for m in messages:
        if m.get("role") != "system":
            break
        idx += 1
    return [*messages[:idx], {"role": "system", "content": role_prompt}, *messages[idx:]]


def _prepare_panel(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    pool_key: str,
    task: TaskType,
    messages: list[dict[str, Any]],
    subtasks: list[str],
    options: dict[str, Any],
    timeout_s: float,
    max_workers_cloud: int,
    tools: ToolRegistry | None,
    tool_capable_models: set[str] | None,
    react_max_rounds: int,
    react_compress_after: int,
    react_max_tool_chars: int,
    react_dispatch_timeout_s: float,
    react_compress_keep_last: int,
    react_max_web_searches: int,
    user_id: str | None,
) -> tuple[list[tuple[str, str]], list[Any]]:
    """Select workers and build their coroutines — shared by both run_panel variants.

    Returns `(workers, coros)` where `workers` is `[(model, location), ...]`
    and `coros` is the matching list of unawaited `_run_one_worker` coroutines.
    Returns `([], [])` when no worker is available so callers can short-circuit.

    `run_panel` and `run_panel_streaming` differ only in how they await these
    coroutines (`gather` vs `as_completed`); everything up to that point —
    healthy-worker selection, the registry fallback, subtask assignment, the
    `dispatch_total` metric, and coro construction — lives here so the two
    paths can't drift.
    """
    workers = select_workers(
        cfg, registry, health,
        pool_key=pool_key, task=task, max_workers_cloud=max_workers_cloud,
    )
    # If no workers from the pool are healthy, fall back to the registry's
    # top-N healthy models for this task so we always answer something.
    # Cap at 2 to mirror the typical pool size (most pools have 2 workers);
    # this is the emergency path so we don't want to flood the GPU gate or
    # burn cloud quota — two drafts is enough material to synthesize from.
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

    capable = tool_capable_models or set()
    have_tools = bool(tools is not None and tools.by_name)
    worker_role = prompt_from_config(cfg, "deep_worker", DEEP_WORKER_SYSTEM)

    # Only TOOL-CAPABLE workers get the role prompt. A worker running a plain
    # one-shot chat has no evidence to reason about and no search process to
    # narrate, so the prompt would be pure token cost — and leaving those
    # messages untouched keeps the tool-free path byte-identical to before.
    per_worker_messages: list[list[dict[str, Any]]] = []
    for i, (name, _loc) in enumerate(workers):
        msgs = (
            _messages_for_subtask(messages, subtasks[i % len(subtasks)])
            if subtasks else messages
        )
        if have_tools and name in capable:
            msgs = _with_worker_system(msgs, worker_role)
        per_worker_messages.append(msgs)

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
            react_compress_keep_last=react_compress_keep_last,
            react_max_web_searches=react_max_web_searches,
            user_id=user_id,
            cfg=cfg,
        )
        for i, (name, loc) in enumerate(workers)
    ]
    return workers, coros


async def run_panel(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
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
    react_compress_keep_last: int = 1,
    react_max_web_searches: int = 0,
    user_id: str | None = None,
) -> tuple[list[WorkerDraft], list[str]]:
    """Run the panel and return (drafts, attempted_models).

    `drafts` includes both successes and per-worker errors so callers can
    decide whether enough material exists to synthesize.

    Workers whose model name is in `tool_capable_models` and whose pool has
    a non-empty `ToolRegistry` run ReAct; others run a one-shot chat.
    """
    workers, coros = _prepare_panel(
        cfg, ollama, registry, health, gate,
        pool_key=pool_key, task=task, messages=messages, subtasks=subtasks,
        options=options, timeout_s=timeout_s, max_workers_cloud=max_workers_cloud,
        tools=tools, tool_capable_models=tool_capable_models,
        react_max_rounds=react_max_rounds, react_compress_after=react_compress_after,
        react_max_tool_chars=react_max_tool_chars,
        react_dispatch_timeout_s=react_dispatch_timeout_s,
        react_compress_keep_last=react_compress_keep_last,
        react_max_web_searches=react_max_web_searches, user_id=user_id,
    )
    if not coros:
        return [], []
    drafts = await asyncio.gather(*coros)
    attempted = [name for name, _ in workers]
    return list(drafts), attempted


async def run_panel_streaming(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
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
    react_compress_keep_last: int = 1,
    react_max_web_searches: int = 0,
    user_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Streaming variant of `run_panel`.

    Yields one event per worker as it completes (in completion order, not
    submission order), then a final event carrying the full draft list.

    Workers run with the same scheduling as `run_panel` — `asyncio.as_completed`
    only changes *reception* order, not *execution* order. Local workers still
    serialize through the GPU gate; cloud workers still run concurrently up to
    `max_workers_cloud`. Total wall-clock time is identical to `run_panel`.

    Event shapes:
      {"type": "worker_done", "model": str, "ok": bool, "elapsed_s": float}
      {"type": "final", "drafts": list[WorkerDraft], "attempted": list[str]}

    The final event always fires last, even if zero workers were available
    (drafts=[], attempted=[]). Callers can rely on it as the end-of-stream
    sentinel.
    """
    workers, coros = _prepare_panel(
        cfg, ollama, registry, health, gate,
        pool_key=pool_key, task=task, messages=messages, subtasks=subtasks,
        options=options, timeout_s=timeout_s, max_workers_cloud=max_workers_cloud,
        tools=tools, tool_capable_models=tool_capable_models,
        react_max_rounds=react_max_rounds, react_compress_after=react_compress_after,
        react_max_tool_chars=react_max_tool_chars,
        react_dispatch_timeout_s=react_dispatch_timeout_s,
        react_compress_keep_last=react_compress_keep_last,
        react_max_web_searches=react_max_web_searches, user_id=user_id,
    )
    if not coros:
        yield {"type": "final", "drafts": [], "attempted": []}
        return

    drafts: list[WorkerDraft] = []
    for coro in asyncio.as_completed(coros):
        draft = await coro
        ok = bool((draft.get("content") or "").strip())
        drafts.append(draft)
        yield {
            "type": "worker_done",
            "model": draft.get("model", "?"),
            "ok": ok,
            "elapsed_s": float(draft.get("elapsed_s", 0.0) or 0.0),
        }

    attempted = [name for name, _ in workers]
    yield {"type": "final", "drafts": drafts, "attempted": attempted}


# ─── Research mode (audrey_research) — staged pipeline ─────────────────
# A staged pipeline behind the `audrey_research` virtual model:
#   1  RESEARCH    — parallel fan-out of tool-capable researchers that ground
#                    with web_search/kb_search. Cloud researchers run concurrently
#                    (capped at max_research_workers_cloud); a local researcher
#                    serializes through the GPU gate. A second mechanical pass
#                    structures each draft's prose into the claim/source ledger
#                    (opt-in; see _ledger_enabled).
#   2  VERIFY      — one pass auditing the merged findings for unsupported /
#                    overconfident / anachronistic claims.
#   3  FACT-CHECK  — optional tool-using pass that web-confirms high-risk/dated
#                    claims; with a ledger, its prose becomes per-claim verdicts
#                    (drop/hedge/correct) the writer applies.
#   4  WRITE       — one pass turning verified, fact-checked findings into the
#                    answer. The pipeline then deterministically appends the
#                    Sources list and (opt-in) per-claim hedge dispositions.
# The Write stage output IS the final answer (no separate synthesizer). Each
# stage degrades gracefully; the pipeline never raises — like `run_panel`.


def select_researchers(
    cfg: Config,
    registry: ModelRegistry,
    health: HealthTracker,
    *,
    task: TaskType,
    max_researchers_cloud: int,
) -> list[tuple[str, str]]:
    """Return [(model, location), ...] for healthy Stage-1 researchers.

    Mirrors `select_workers` but reads the staged pool's `researchers` list
    and caps cloud researchers at `max_researchers_cloud` (research fans wider
    than normal deep, so it has its own ceiling). Local researchers are not
    capped here — the GPU gate serializes them.
    """
    pool = cfg.raw.get("deep_panel_research", {}).get(task, {})
    names: list[str] = list(pool.get("researchers", []) or [])
    out: list[tuple[str, str]] = []
    cloud_count = 0
    for name in names:
        if not health.is_healthy(name):
            log.info("research: skipping unhealthy researcher %s", name)
            continue
        loc = registry.location_of(name)
        if loc == "cloud":
            if cloud_count >= max_researchers_cloud:
                continue
            cloud_count += 1
        out.append((name, loc))
    return out


def _research_pool(cfg: Config, task: TaskType) -> dict[str, Any]:
    return cfg.raw.get("deep_panel_research", {}).get(task, {}) or {}


def _with_role_system(messages: list[dict[str, Any]], role_prompt: str) -> list[dict[str, Any]]:
    """Prepend a role system message ahead of the conversation."""
    return [{"role": "system", "content": role_prompt}, *messages]


def _format_findings(drafts: list[WorkerDraft]) -> str:
    """Merge successful researcher drafts into one findings block.

    Empty/errored drafts are skipped. Returns "" when nothing usable came
    back — the caller treats that as the no-grounding case.
    """
    parts: list[str] = []
    n = 0
    for d in drafts:
        content = (d.get("content") or "").strip()
        if not content:
            continue
        n += 1
        rounds = int(d.get("tool_rounds", 0) or 0)
        tag = f" (grounded: {rounds} tool rounds)" if rounds > 0 else " (no tools used)"
        parts.append(f"--- researcher {n}{tag} ---\n{content}")
    return "\n\n".join(parts)


def _ledger_enabled(cfg: Config) -> bool:
    """Whether the Phase 26 research ledger (Stage 1+) is on.

    Off by default — the ledger is opt-in via `agentic.research_ledger.enabled`
    so it can be toggled live (no rebuild) while we measure each stage against
    the prose baseline.
    """
    agentic = cfg.raw.get("agentic", {}) or {}
    rl = agentic.get("research_ledger", {}) or {}
    return bool(rl.get("enabled", False))


def _hedge_policy_enabled(cfg: Config) -> bool:
    """Whether Stage 4 deterministic hedging is on.

    Separate from `research_ledger.enabled` so the per-claim disposition guidance
    can be evaluated on/off against a ledger-on baseline (it changes writer
    wording — plain statement becomes explicitly allowed). Off by default; toggled
    live via `agentic.research_ledger.hedge_policy`.
    """
    agentic = cfg.raw.get("agentic", {}) or {}
    rl = agentic.get("research_ledger", {}) or {}
    return bool(rl.get("hedge_policy", False))


def _linkage_lost(n_linked: int, n_sources: int) -> bool:
    """Did the structuring pass DROP linkage the researcher actually supplied?

    Distinguishes the pathology from the honest case, which the first version of
    this marker could not. 2026-08-13, `current-rust-async`: qwen3.6 returned
    `claims=16 linked=1 sources=1` and the marker stayed silent, because it
    tested `not n_linked` and one linked claim defeats a strict-equality cliff —
    the same shape as the over-hedge suppression guard. But widening it to catch
    that shape would have been wrong too: qwen3.6's notes for that case carried
    **no `SOURCES:` block at all**, so fifteen empty `source_ids` were correct.
    The same model on the sibling case, with a SOURCES block, linked 22 of 22.

    So the signal is not "few claims are linked" — it is "the ledger holds
    several sources and almost no claim uses any of them", which is the
    `claims=41 linked=0 sources=5` shape that started this. A ledger with one
    source and one link is a thin researcher, not lost linkage; chase that
    upstream in the researcher prompt, not here.

    Deliberately not a ratio: the linked/claims ratio swings run to run on an
    unchanged config, and a threshold on it would make the marker blink. This is
    a diagnostic and not a behaviour gate, so a looser rule is safe here in a way
    it is not for the hedge-suppression guard."""
    return n_sources >= 2 and n_linked <= 1


async def _structure_one_draft(
    ollama: OllamaClient,
    health: HealthTracker,
    gate: FairLocalGate,
    cfg: Config,
    *,
    model: str,
    location: str,
    prose: str,
    timeout_s: float,
    user_id: str | None,
    worker_idx: int,
) -> ResearchResult | None:
    """Convert one researcher's prose notes into a ResearchResult ledger.

    A second, mechanical pass (no tools) pinned to the ResearchResult JSON
    schema via Ollama `format`. Fail-soft: any tool/parse error returns None,
    and the caller keeps the prose findings unchanged. Claim/source ids are
    prefixed with the worker index so a later merge across workers can't collide.
    """
    if not prose.strip():
        return None
    msgs = [
        {"role": "system", "content": prompt_from_config(cfg, "research_structure", RESEARCH_STRUCTURE_SYSTEM)},
        {"role": "user", "content": f"RESEARCHER NOTES:\n{prose.strip()}\n\nReturn the ledger as JSON."},
    ]
    try:
        async with gate.acquire(model, location=location, user_id=user_id):
            resp = await ollama.chat(
                model=model,
                messages=msgs,
                options={"temperature": 0.0},
                timeout_s=timeout_s,
                format=inlined_schema(ResearchResult),
            )
        health.record_success(model)
    except OllamaError as e:
        health.record_failure(model, str(e))
        log.warning("research: structuring call for %s failed: %s", model, e)
        return None
    raw = (resp.get("message", {}) or {}).get("content", "") or ""
    result = parse_research_result(raw)
    if result is None:
        # Log a sample of what actually came back — empty / truncated / prose /
        # refusal each need a different fix, and we can't tell without seeing it.
        log.info(
            "research: structuring call for %s produced unusable JSON "
            "(len=%d, head=%r, tail=%r)",
            model, len(raw), raw[:200], raw[-120:],
        )
        return None
    if not result.claims:
        # Parsed fine but EMPTY — the model returned a valid but contentless
        # ledger. Show the raw so we can tell a literal {"claims":[]} reply
        # (prompt/schema issue) from anything else.
        log.info(
            "research: structuring call for %s parsed but empty "
            "(claims=0, len=%d, head=%r)",
            model, len(raw), raw[:300],
        )
    # Linkage shape, read AFTER the parser's repair chain so it reflects what
    # downstream actually gets. Read the triple directly — it separates the two
    # very different reasons a claim ends up unsourced.
    n_linked = sum(1 for c in result.claims if c.source_ids)
    log.info(
        "research: structured %s — claims=%d linked=%d sources=%d%s",
        model, len(result.claims), n_linked, len(result.sources),
        "  UNLINKED-LEDGER" if _linkage_lost(n_linked, len(result.sources)) else "",
    )
    # Namespace ids per worker so cross-worker merge can't collide.
    prefix = f"w{worker_idx}_"
    return _prefix_ledger_ids(result, prefix)


def _prefix_ledger_ids(r: ResearchResult, prefix: str) -> ResearchResult:
    """Return a copy of `r` with claim/source ids (and cross-refs) prefixed."""
    claims = [
        Claim(
            id=prefix + c.id,
            text=c.text,
            source_ids=[prefix + s for s in c.source_ids],
            risk=c.risk,
            needs_hedge=c.needs_hedge,
            hedge_reason=c.hedge_reason,
        )
        for c in r.claims
    ]
    sources = [
        Source(
            id=prefix + s.id,
            title=s.title,
            url=s.url,
            source_type=s.source_type,
            supports=[prefix + c for c in s.supports],
        )
        for s in r.sources
    ]
    return ResearchResult(
        summary_notes=r.summary_notes,
        claims=claims,
        sources=sources,
        unresolved_questions=r.unresolved_questions,
    )


def _merge_ledgers(ledgers: list[ResearchResult]) -> ResearchResult:
    """Merge per-worker ledgers into one. Sources deduped by URL; claims
    concatenated (ids already worker-prefixed). Conflicting claims across
    workers are NOT reconciled here — they're surfaced as-is for the
    fact-check stage to flag.

    Dropping a duplicate-URL source must not orphan the claims that cite it
    (2026-07-07 eval, `tech-transformer-attention`: one worker's arXiv sources
    matched another's by URL, and its claims — citing the dropped ids — lost
    all backing, so `hedge_policy` hedged textbook facts). So on a URL hit we
    remap the dropped id to the canonical one, fold the dropped source's
    `supports` into it, and let a typed duplicate upgrade a canonical typed
    `unknown` — the surviving source carries the best of both copies."""
    seen_urls: dict[str, str] = {}  # url → canonical source id
    by_id: dict[str, Source] = {}  # canonical id → kept Source
    remap: dict[str, str] = {}  # dropped duplicate id → canonical id
    merged_sources: list[Source] = []
    for led in ledgers:
        for s in led.sources:
            key = s.url.strip().rstrip("/").lower()
            if key and key in seen_urls:
                canon = by_id[seen_urls[key]]
                remap[s.id] = canon.id
                canon.supports.extend(
                    c for c in s.supports if c not in canon.supports
                )
                if canon.source_type == "unknown" and s.source_type != "unknown":
                    canon.source_type = s.source_type
                continue
            if key:
                seen_urls[key] = s.id
            by_id[s.id] = s
            merged_sources.append(s)
    merged_claims: list[Claim] = []
    summaries: list[str] = []
    unresolved: list[str] = []
    for led in ledgers:
        for c in led.claims:
            if remap:
                c.source_ids = [remap.get(sid, sid) for sid in c.source_ids]
            merged_claims.append(c)
        if led.summary_notes.strip():
            summaries.append(led.summary_notes.strip())
        unresolved.extend(led.unresolved_questions)
    return ResearchResult(
        summary_notes="\n\n".join(summaries),
        claims=merged_claims,
        sources=merged_sources,
        unresolved_questions=unresolved,
    )


async def _single_chat_stage(
    ollama: OllamaClient,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    model: str,
    location: str,
    messages: list[dict[str, Any]],
    timeout_s: float,
    user_id: str | None,
) -> str:
    """Run one non-tool chat call (Verify or Write), gate-held for local.

    Returns the content string, or "" on failure (the pipeline degrades
    rather than raising — same posture as `_run_one_worker`).
    """
    try:
        async with gate.acquire(model, location=location, user_id=user_id):
            resp = await ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.2},
                timeout_s=timeout_s,
            )
        health.record_success(model)
        return (resp.get("message", {}) or {}).get("content", "") or ""
    except OllamaError as e:
        health.record_failure(model, str(e))
        log.warning("research: stage model %s failed: %s", model, e)
        return ""


def _verify_user_block(user_text: str, findings: str) -> str:
    return (
        f"ORIGINAL REQUEST:\n{user_text.strip()}\n\n"
        f"RESEARCHER FINDINGS:\n{findings}\n\n"
        "Audit the findings now. Output your flags."
    )


def _factcheck_user_block(user_text: str, findings: str, critique: str) -> str:
    parts = [
        f"ORIGINAL REQUEST:\n{user_text.strip()}\n",
        f"RESEARCHER FINDINGS:\n{findings}\n",
    ]
    if critique:
        parts.append(f"VERIFIER FLAGS:\n{critique}\n")
    parts.append(
        "Verify the most load-bearing checkable claims with web_search now, "
        "then output your corrections list."
    )
    return "\n".join(parts)


def _render_claims_for_factcheck(claims: list[Claim], sources: list[Source]) -> str:
    """A compact claim list (id, risk, text) for the fact-check structuring
    prompt to reference by claim_id. `claims` is the batch to verdict; `sources`
    is the FULL source set (a batch is judged against all sources, not just the
    ones its claims cite). Sources appended so the checker can judge support
    without re-fetching."""
    lines = ["CLAIMS:"]
    for c in claims:
        src = f" [sources: {', '.join(c.source_ids)}]" if c.source_ids else " [no source]"
        lines.append(f"- {c.id} (risk={c.risk}): {c.text}{src}")
    if sources:
        lines.append("\nSOURCES:")
        for s in sources:
            lines.append(f"- {s.id} ({s.source_type}): {s.title} — {s.url}")
    return "\n".join(lines)


# Claims per structuring call. A single call over a whole 75–95-claim ancient-bio
# ledger reliably collapsed to a near-empty `checks` array (2026-07-08 eval:
# euclid/pythagoras/archimedes returned zero verdicts, so the fact-checker's
# CORRECT/DROP judgments were lost and only the deterministic hedge path
# survived). Batching keeps each call small enough that the model fills every
# verdict, and a collapse in one batch can't zero the others.
_FACTCHECK_STRUCTURE_BATCH = 15


async def _structure_factcheck_batch(
    ollama: OllamaClient,
    health: HealthTracker,
    gate: FairLocalGate,
    cfg: Config,
    *,
    model: str,
    location: str,
    claims: list[Claim],
    sources: list[Source],
    fc_notes: str,
    timeout_s: float,
    user_id: str | None,
) -> FactCheckResult | None:
    """Structure one batch of claims into a FactCheckResult. Second mechanical
    pass (no tools), pinned to the FactCheckResult schema. Fail-soft: any error
    returns None so the chunked caller can keep the batches that succeeded."""
    if not claims:
        return None
    block = (
        f"{_render_claims_for_factcheck(claims, sources)}\n\n"
        f"FACT-CHECKER NOTES:\n{fc_notes.strip()}\n\n"
        "Return the per-claim verdict list as JSON."
    )
    msgs = [
        {"role": "system", "content": prompt_from_config(cfg, "factcheck_structure", FACTCHECK_STRUCTURE_SYSTEM)},
        {"role": "user", "content": block},
    ]
    try:
        async with gate.acquire(model, location=location, user_id=user_id):
            resp = await ollama.chat(
                model=model,
                messages=msgs,
                options={"temperature": 0.0},
                timeout_s=timeout_s,
                format=inlined_schema(FactCheckResult),
            )
        health.record_success(model)
    except OllamaError as e:
        health.record_failure(model, str(e))
        log.warning("research: factcheck structuring for %s failed: %s", model, e)
        return None
    raw = (resp.get("message", {}) or {}).get("content", "") or ""
    result = parse_factcheck_result(raw)
    if result is None:
        log.info(
            "research: factcheck structuring produced unusable JSON "
            "(len=%d, head=%r, tail=%r)",
            len(raw), raw[:200], raw[-120:],
        )
    return result


async def _structure_factcheck(
    ollama: OllamaClient,
    health: HealthTracker,
    gate: FairLocalGate,
    cfg: Config,
    *,
    model: str,
    location: str,
    ledger: ResearchResult,
    fc_notes: str,
    timeout_s: float,
    user_id: str | None,
) -> FactCheckResult | None:
    """Convert the fact-checker's prose notes into a FactCheckResult keyed to
    the claim ledger, structuring in `_FACTCHECK_STRUCTURE_BATCH`-claim chunks
    and merging the results (see the batch-size comment). Every batch sees the
    full source set. Fail-soft: returns None only if EVERY batch failed — a
    partial merge still reaches the writer. Order-preserving; `fatal_errors`
    concatenate across batches. Runs after the fact-checker's ReAct loop."""
    if not ledger.claims:
        return None
    batches = [
        ledger.claims[i:i + _FACTCHECK_STRUCTURE_BATCH]
        for i in range(0, len(ledger.claims), _FACTCHECK_STRUCTURE_BATCH)
    ]
    coros = [
        _structure_factcheck_batch(
            ollama, health, gate, cfg,
            model=model, location=location,
            claims=batch, sources=ledger.sources, fc_notes=fc_notes,
            timeout_s=timeout_s, user_id=user_id,
        )
        for batch in batches
    ]
    results = await asyncio.gather(*coros)
    if all(r is None for r in results):
        return None
    merged = FactCheckResult()
    for r in results:
        if r is None:
            continue
        merged.checks.extend(r.checks)
        merged.fatal_errors.extend(r.fatal_errors)
    return merged


def _factcheck_result_to_corrections(fc: FactCheckResult, ledger: ResearchResult) -> str:
    """Render structured per-claim verdicts into the prose corrections string the
    writer already applies — so the ledger's verdicts reach the writer through the
    same channel the prose fact-checker used (no separate writer path). Lines mirror
    the prose fact-checker's vocabulary so `_has_corrections` and the writer prompt
    work unchanged:
      - unsupported  → DROP (writer omits)
      - conflicting  → UNVERIFIED (writer hedges)
      - needs_hedge  → CORRECT to corrected_text, or UNVERIFIED if none
      - supported    → CONFIRMED (no action; kept for transparency)
    `irrelevant` claims are skipped. Returns _NO_CORRECTIONS when nothing
    actionable came back.
    """
    by_id = {c.id: c.text for c in ledger.claims}
    lines: list[str] = []
    for chk in fc.checks:
        claim_text = by_id.get(chk.claim_id, chk.claim_id)
        if chk.verdict == "unsupported":
            lines.append(f"DROP: {claim_text} — unsupported by sources ({chk.notes})".rstrip())
        elif chk.verdict == "conflicting":
            lines.append(f"UNVERIFIED: {claim_text} — sources conflict; HEDGE it ({chk.notes})".rstrip())
        elif chk.verdict == "needs_hedge":
            if chk.corrected_text:
                lines.append(f"CORRECT: use \"{chk.corrected_text}\" for: {claim_text}")
            else:
                lines.append(f"UNVERIFIED: {claim_text} — HEDGE it ({chk.notes})".rstrip())
        elif chk.verdict == "supported":
            lines.append(f"CONFIRMED: {claim_text}")
        # irrelevant → skip
    if fc.fatal_errors:
        for fe in fc.fatal_errors:
            lines.append(f"UNVERIFIED: contradiction — {fe}; HEDGE the affected claims")
    body = "\n".join(lines).strip()
    # Actionable only if there's a DROP/CORRECT/UNVERIFIED (CONFIRMED-only is no-op).
    if not body or not any(k in body for k in ("DROP:", "CORRECT:", "UNVERIFIED:")):
        return _NO_CORRECTIONS
    return body


# A corrections payload that carries no real corrections — used to detect when
# the fact-check stage found nothing to fix (or could not run) so the writer
# block can omit the block entirely.
_NO_CORRECTIONS = "NO CORRECTIONS"

# Cap the user-facing Sources list — a research answer that drew on a dozen
# pages doesn't need to dump all of them; the most-cited ones are enough.
_MAX_SOURCES_RENDERED = 8


# Source-type authority tiers for ranking the user-facing Sources list. When more
# sources survive than the cap, we keep the most authoritative — so a Wikipedia /
# Stanford / arXiv link isn't crowded out by a facebook-groups or scribd URL that
# SearXNG happened to surface (the +36 eval's "weak domains in Sources" finding).
# Mirrors the hedge_policy authoritative set. Ties keep ledger order (stable sort).
_SOURCE_RANK: dict[str, int] = {
    "official": 3,
    "primary_paper": 3,
    "scholarly": 3,
    "reference": 3,
    "news": 2,
    "company_claim": 1,
    "blog": 1,
    "unknown": 0,
}


def _source_rank(source_type: str) -> int:
    """Authority tier for sorting the Sources list (higher = preferred). Unknown
    or off-enum types sort to the bottom."""
    return _SOURCE_RANK.get(source_type, 0)


# A URL we're willing to show the user: http(s) with a host. Models emit bare
# titles, fragments, "", and the literal string "null" for sources they could
# not link — those get a backfilled id but no usable URL, and we don't list
# them. Shared with `_demote_urlless_authority`, which uses the same predicate
# to decide whether a source can confer authority; see `usable_url`.
_usable_url = usable_url


def _surviving_source_ids(ledger: ResearchResult, fc: FactCheckResult | None) -> set[str]:
    """Source ids that back at least one claim the fact-checker did NOT drop.

    A claim is "dropped" iff the fact-check verdict is `unsupported`. With no
    fact-check result every claim survives. A source backs a claim through either
    direction of the link (`claim.source_ids` or `source.supports`) since models
    populate one or the other inconsistently.

    Only ids that actually exist in `ledger.sources` count: models also emit
    unresolvable refs (titles, fragments — see `_repair_source_links` for the
    repairable shape), and keeping those would make the result non-empty-but-
    useless, silently defeating the caller's empty-linkage fallback."""
    dropped = {
        chk.claim_id for chk in (fc.checks if fc else [])
        if chk.verdict == "unsupported"
    }
    surviving_claims = {c.id for c in ledger.claims if c.id not in dropped}
    real_ids = {s.id for s in ledger.sources}
    keep: set[str] = set()
    for c in ledger.claims:
        if c.id in surviving_claims:
            keep.update(sid for sid in c.source_ids if sid in real_ids)
    for s in ledger.sources:
        if any(cid in surviving_claims for cid in s.supports):
            keep.add(s.id)
    return keep


def _render_sources_block(ledger: ResearchResult | None, fc: FactCheckResult | None) -> str:
    """The user-facing `## Sources` markdown, or "" to render nothing.

    Deterministic — built by us from the ledger, NOT asked of the writer. The
    +21 lesson: making the model emit citations degrades prose; the structure is
    internal, and we render the clean list ourselves after the prose is written.

    Returns "" (omit the block) when there's no ledger, or no surviving source
    has a usable URL — an ungrounded/creative answer must not sprout an empty
    Sources header. Sources are ranked by authority (see `_source_rank`), deduped
    by URL, and capped — so the most authoritative ones win the limited slots.
    """
    if ledger is None or not ledger.sources:
        return ""
    keep = _surviving_source_ids(ledger, fc)
    # Fall back to every source when the linkage yields nothing *renderable* —
    # better to show what was consulted than nothing. Two cases collapse to the
    # same fallback: (1) linkage is empty across the board (models didn't fill
    # source_ids/supports), and (2) linkage is non-empty but NO kept source has a
    # usable URL. Case 2 bit `tech-transformer-attention` (2026-07-15 trace run):
    # a claim linked the one URL-less "Search result" source, so `keep` was
    # non-empty and the fallback DIDN'T fire — even though the ledger also held
    # real arXiv/Wikipedia URLs that simply weren't linked to a surviving claim.
    # Result: `sources:0` on an answer that clearly had citable sources. Gating on
    # "no kept source is renderable" rescues those unlinked-but-usable URLs.
    keep_has_usable_url = any(
        s.id in keep and _usable_url(s.url) for s in ledger.sources
    )
    use_all = not keep or not keep_has_usable_url
    candidates = [s for s in ledger.sources if use_all or s.id in keep]
    # Most-authoritative first so the cap keeps the best sources, not whichever
    # the panel happened to list first. Stable sort → same-tier sources retain
    # ledger order; dedup below then keeps the higher-ranked of a shared URL.
    candidates.sort(key=lambda s: _source_rank(s.source_type), reverse=True)
    seen: set[str] = set()
    rendered: list[str] = []
    for s in candidates:
        if not _usable_url(s.url):
            continue
        key = s.url.strip().rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        title = s.title.strip() or s.url.strip()
        rendered.append(f"- [{title}]({s.url.strip()})")
        if len(rendered) >= _MAX_SOURCES_RENDERED:
            break
    if not rendered:
        return ""
    return "\n\n## Sources\n" + "\n".join(rendered) + "\n"


# How each disposition reads as one line of writer guidance. The writer prompt
# learns to apply these; the mapping is deterministic so the wording is stable.
_DISPOSITION_LABEL: dict[str, str] = {
    "state_plainly": "STATE PLAINLY",
    "attribute_to_company": "ATTRIBUTE TO SOURCE",
    "hedge": "HEDGE",
    "hedge_or_cite_strongly": "HEDGE (unless a strong source backs it)",
}


def _source_types_for_claim(claim: Claim, ledger: ResearchResult) -> set[SourceType]:
    """The source types backing `claim`, via either link direction.

    Models populate `claim.source_ids` or `source.supports` inconsistently, so we
    gather both — same dual-direction logic as `_surviving_source_ids`."""
    backing = {sid for sid in claim.source_ids}
    types: set[SourceType] = set()
    for s in ledger.sources:
        if s.id in backing or claim.id in s.supports:
            types.add(s.source_type)
    return types


def _render_dispositions_block(
    ledger: ResearchResult | None, fc: FactCheckResult | None
) -> str:
    """Per-claim hedging guidance for the writer, or "" to render nothing.

    Renders ONLY the action-bearing dispositions — `attribute_to_company`,
    `hedge`, `hedge_or_cite_strongly` — plus a single framing line that says
    everything else may be stated plainly. `state_plainly` claims are NOT
    enumerated: a dense 3-worker ledger produces dozens of claims, and listing a
    "STATE PLAINLY: <full sentence>" line for each one buried the few that
    actually constrain the writer and nudged it to restate the whole ledger
    verbatim (the bloated, near-duplicate Pythagoras answer). The one
    "state everything else plainly" line preserves the anti-over-hedge intent —
    plain IS the default — without the wall of redundant lines.

    Returns "" (omit the block) when there's no ledger, no surviving claim, no
    ACTION-bearing line (everything is `state_plainly` → the writer's default is
    already plain, nothing to instruct), OR when EVERY surviving claim is `hedge`
    (pure blanket caution — the over-hedge trap that turned a plain "explain
    recursion" into mush). A MIX — some plain, a few hedges/attributions — DOES
    render: that selective handling against a plain-by-default backdrop is exactly
    the point of the stage.

    When NO ledger source has a usable URL, "cite strongly" is an empty option,
    so `hedge_or_cite_strongly` degenerates to plain hedging and counts toward
    the all-hedge suppression — otherwise a search-starved run renders a wall
    of blanket-caution lines through that dodge (observed 2026-07-06: 44 HEDGE
    lines on `current-rust-async`, soaking the answer in "reportedly").

    Rendered lines are deduped by normalized claim text: three workers each
    contribute a near-identical claim ("Euclid flourished c. 300 BCE"), and a
    repeated instruction is pure wall-padding. First occurrence wins (its
    disposition included). Dedup affects the RENDERED lines only — the
    all-plain/all-hedge suppression counters still see every claim, so
    suppression semantics don't shift with worker overlap."""
    if ledger is None or not ledger.claims:
        return ""
    dropped = {
        chk.claim_id for chk in (fc.checks if fc else [])
        if chk.verdict == "unsupported"
    }
    citable = any(_usable_url(s.url) for s in ledger.sources)
    lines: list[str] = []
    seen_texts: set[str] = set()
    n_surviving = 0
    n_hedge = 0
    for c in ledger.claims:
        if c.id in dropped or not (c.text or "").strip():
            continue
        n_surviving += 1
        disposition = hedge_policy(c, _source_types_for_claim(c, ledger))
        if disposition == "state_plainly":
            continue  # plain is the default; the framing line covers these
        if disposition == "hedge" or (
            disposition == "hedge_or_cite_strongly" and not citable
        ):
            n_hedge += 1
        norm = re.sub(r"[\W_]+", " ", c.text).casefold().strip()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        lines.append(f"- {_DISPOSITION_LABEL[disposition]}: {c.text.strip()}")
    # No action lines → all-plain, nothing to instruct. Every claim a plain `hedge`
    # → blanket caution (the over-hedge trap). Both: the writer's defaults cover it.
    if not lines or n_hedge == n_surviving:
        return ""
    return (
        "State every claim plainly EXCEPT these, which need specific handling:\n"
        + "\n".join(lines)
    )


def _has_corrections(corrections: str) -> bool:
    """True when the fact-checker returned actionable corrections."""
    c = (corrections or "").strip()
    if not c:
        return False
    # The prompt's explicit "nothing to fix" sentinel, plus the case where every
    # line merely CONFIRMED a claim (no actionable verdict → nothing to apply).
    # DROP: comes from the Stage-2 claim-ledger checker (unsupported claim).
    if c == _NO_CORRECTIONS:
        return False
    return ("CORRECT:" in c) or ("UNVERIFIED:" in c) or ("DROP:" in c)


def _write_user_block(
    user_text: str, findings: str, critique: str, corrections: str = "",
    dispositions: str = "",
) -> str:
    parts = [f"ORIGINAL REQUEST:\n{user_text.strip()}\n"]
    if findings:
        parts.append(f"VERIFIED FINDINGS:\n{findings}\n")
    else:
        parts.append(
            "VERIFIED FINDINGS:\n[No grounding could be retrieved — the "
            "research stage produced no usable findings.]\n"
        )
    if critique:
        parts.append(f"VERIFIER FLAGS (apply these):\n{critique}\n")
    if _has_corrections(corrections):
        parts.append(f"FACT-CHECK CORRECTIONS (apply these):\n{corrections}\n")
    if dispositions.strip():
        parts.append(f"CLAIM DISPOSITIONS (apply these):\n{dispositions}\n")
    parts.append("Write the final answer for the user now.")
    return "\n".join(parts)


def _react_budget(cfg: Config, key: str, *, default_rounds: int) -> dict[str, int]:
    """Read `agentic.react.<key>`, falling back to the base `react` defaults."""
    react = (cfg.raw.get("agentic", {}) or {}).get("react", {}) or {}
    rw = react.get(key, {}) or {}
    return {
        "max_rounds": int(rw.get("max_rounds", react.get("max_rounds", default_rounds))),
        "compress_after": int(rw.get("compress_after_round", react.get("compress_after_round", 3))),
        "max_tool_chars": int(rw.get("max_tool_result_chars", react.get("max_tool_result_chars", 6000))),
        "compress_keep_last": int(rw.get("compress_keep_last", react.get("compress_keep_last", 1))),
        "dispatch_timeout_s": int(rw.get("dispatch_timeout_s", react.get("dispatch_timeout_s", 30))),
        # Per-worker cap on dispatched web_search calls (0 = unlimited).
        # Real provider quota (Brave) is spent per call and multiplied by
        # panel width, so the budget is enforced per worker in run_react.
        "max_web_searches": int(rw.get("max_web_searches", react.get("max_web_searches", 0))),
    }


def _research_react_budget(cfg: Config) -> dict[str, int]:
    """Stage-1 researcher ReAct budget (`agentic.react.research_worker`)."""
    return _react_budget(cfg, "research_worker", default_rounds=5)


def _factcheck_react_budget(cfg: Config) -> dict[str, int]:
    """Stage-3 fact-checker ReAct budget (`agentic.react.factcheck_worker`)."""
    return _react_budget(cfg, "factcheck_worker", default_rounds=3)


async def run_research_pipeline_streaming(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    task: TaskType,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
    max_researchers_cloud: int,
    tools: ToolRegistry | None = None,
    tool_capable_models: set[str] | None = None,
    user_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the staged research pipeline, yielding stage events.

    Event shapes (in order):
      {"type": "researcher_done", "model": str, "ok": bool, "elapsed_s": float}
          One per researcher as Stage 1 completes (completion order).
      {"type": "findings_ready", "grounded": bool}
          Stage 1 merged; `grounded` False means no usable findings.
      {"type": "verify_done", "ok": bool}
          Stage 2 finished (skipped → not emitted when no grounding).
      {"type": "factcheck_done", "ok": bool}
          Stage 3 finished; `ok` True means actionable corrections were found.
          Emitted only when grounded (skipped → not emitted), and the stage
          itself only runs when a `factchecker` is configured + tool-capable.
      {"type": "write_delta", "text": str}
          A chunk of the Writer's answer (streamed live).
      {"type": "done", "content": str, "writer_model": str, "drafts": list,
       "findings": str, "critique": str, "corrections": str,
       "ledger": dict | None, "factcheck": dict | None, "dispositions": str,
       "error": str}
          Always emitted last. `error` is "" on success, or "no_writer" /
          "write_failed" when the write stage could not produce an answer.
          `ledger`/`factcheck` are the structured intermediates as plain
          dicts (`.model_dump()`), None when the stage didn't run/parse;
          `dispositions` is the hedge-guidance block exactly as handed to
          the writer ("" when the flag is off or the block was suppressed).
          Carried so the route can render the opt-in research trace.

    Never raises — each stage degrades. The Write stage always runs (even
    with no findings) so the user gets a flagged answer rather than nothing.
    """
    pool = _research_pool(cfg, task)
    budget = _research_react_budget(cfg)
    capable = tool_capable_models or set()
    user_text = last_user_text(messages)

    # ── Stage 1: research fan-out ──────────────────────────────────────
    researchers = select_researchers(
        cfg, registry, health, task=task, max_researchers_cloud=max_researchers_cloud,
    )
    drafts: list[WorkerDraft] = []
    if researchers:
        researcher_msgs = _with_role_system(messages, prompt_from_config(cfg, "researcher", RESEARCHER_SYSTEM))
        for name, _loc in researchers:
            dispatch_total.labels(model=name, task_type=str(task), path="research").inc()
        coros = [
            _run_one_worker(
                ollama, health, gate,
                model=name, location=loc,
                messages=researcher_msgs,
                options=options,
                timeout_s=timeout_s,
                tools=tools,
                tool_capable=(name in capable),
                react_max_rounds=budget["max_rounds"],
                react_compress_after=budget["compress_after"],
                react_max_tool_chars=budget["max_tool_chars"],
                react_dispatch_timeout_s=float(budget["dispatch_timeout_s"]),
                react_compress_keep_last=budget["compress_keep_last"],
                react_max_web_searches=budget["max_web_searches"],
                user_id=user_id,
                cfg=cfg,
            )
            for name, loc in researchers
        ]
        for coro in asyncio.as_completed(coros):
            draft = await coro
            ok = bool((draft.get("content") or "").strip())
            drafts.append(draft)
            yield {
                "type": "researcher_done",
                "model": draft.get("model", "?"),
                "ok": ok,
                "elapsed_s": float(draft.get("elapsed_s", 0.0) or 0.0),
            }
    else:
        log.warning("research: no healthy researchers for task %s", task)

    # Surface worker drops. A researcher that returns empty content (an
    # OllamaError, a timeout, or a genuinely empty reply) contributes nothing
    # to grounding and is silently filtered out of the ledger downstream. When
    # that thins a 3-worker fan-out to 1, the answer loses sources without any
    # error — the bio-archimedes case (2026-06-30) lost 2/3 workers this way and
    # produced a sourceless answer. Log it loudly so the next drop is a one-line
    # find, not a forensic reconstruction.
    n_ok = sum(1 for d in drafts if (d.get("content") or "").strip())
    if researchers and n_ok < len(drafts):
        dropped = [
            f"{d.get('model', '?')} (err={d.get('error') or 'empty'}, {d.get('elapsed_s', 0)}s)"
            for d in drafts if not (d.get("content") or "").strip()
        ]
        log.warning(
            "research: %d/%d researchers produced content; dropped: %s",
            n_ok, len(drafts), ", ".join(dropped),
        )

    findings = _format_findings(drafts)
    grounded = bool(findings)

    # ── Stage 1 (cont.): structure the findings into a claim/source ledger ──
    # Opt-in (agentic.research_ledger.enabled). A second mechanical pass per
    # worker converts prose → ResearchResult; fail-soft, and the prose `findings`
    # is unchanged either way (the ledger is carried alongside it). The ledger is
    # then consumed downstream: the fact-check stage turns it into per-claim
    # verdicts, and the write stage uses it to render the Sources list and the
    # hedge dispositions. If structuring fails, those stages fall back to the
    # prose path.
    ledger: ResearchResult | None = None
    if grounded and _ledger_enabled(cfg):
        structure_coros = [
            _structure_one_draft(
                ollama, health, gate, cfg,
                model=d.get("model", "?"),
                location=registry.location_of(d.get("model", "")),
                prose=(d.get("content") or ""),
                timeout_s=timeout_s,
                user_id=user_id,
                worker_idx=i,
            )
            for i, d in enumerate(drafts)
            if (d.get("content") or "").strip()
        ]
        try:
            structured = await asyncio.gather(*structure_coros)
            usable = [s for s in structured if s is not None]
            if usable:
                ledger = _merge_ledgers(usable)
                log.info(
                    "research: ledger built — %d claims, %d sources from %d/%d workers",
                    len(ledger.claims), len(ledger.sources), len(usable), len(structure_coros),
                )
        except Exception:
            # Structuring is additive and optional — never break the answer.
            log.exception("research: ledger structuring errored; continuing prose-only")
            ledger = None

    yield {"type": "findings_ready", "grounded": grounded}

    # ── Stage 2: verify (skipped when no grounding) ────────────────────
    critique = ""
    if grounded:
        verifier = pool.get("verifier")
        if verifier and health.is_healthy(verifier):
            dispatch_total.labels(model=verifier, task_type=str(task), path="research_verify").inc()
            v_msgs = [
                {"role": "system", "content": prompt_from_config(cfg, "verifier", VERIFIER_SYSTEM)},
                {"role": "user", "content": _verify_user_block(user_text, findings)},
            ]
            critique = await _single_chat_stage(
                ollama, health, gate,
                model=verifier, location=registry.location_of(verifier),
                messages=v_msgs, timeout_s=timeout_s, user_id=user_id,
            )
        yield {"type": "verify_done", "ok": bool(critique)}

    # ── Stage 3: fact-check (optional; skipped when no grounding, no
    # factchecker configured, or no tools) ─────────────────────────────
    # The fact-checker is a tool-using ReAct loop: it web_search-confirms the
    # high-risk/dated claims and returns a corrections block the writer applies.
    # Bounded + fail-soft like every stage — any problem returns "" and the
    # pipeline proceeds exactly as the verify→write flow did before.
    corrections = ""
    fc_result: FactCheckResult | None = None  # structured verdicts, for Stage-3 Sources
    factchecker = pool.get("factchecker")
    fc_can_run = bool(
        grounded and factchecker and health.is_healthy(factchecker)
        and tools is not None and tools.by_name and factchecker in capable
    )
    # Log the decision either way — when the stage silently skips, this is the
    # only way to see WHICH precondition failed (config vs health vs tools).
    if not fc_can_run:
        log.info(
            "research: fact-check SKIPPED — grounded=%s factchecker=%r healthy=%s "
            "tools=%s tool_capable=%s",
            grounded, factchecker,
            (health.is_healthy(factchecker) if factchecker else None),
            (bool(tools and tools.by_name)),
            (factchecker in capable if factchecker else None),
        )
    else:
        log.info("research: fact-check stage running with %s", factchecker)
    if fc_can_run:
        fc_budget = _factcheck_react_budget(cfg)
        dispatch_total.labels(model=factchecker, task_type=str(task), path="research_factcheck").inc()
        fc_msgs = [
            {"role": "system", "content": prompt_from_config(cfg, "factchecker", FACTCHECK_SYSTEM)},
            {"role": "user", "content": _factcheck_user_block(user_text, findings, critique)},
        ]
        try:
            async with gate.acquire(factchecker, location=registry.location_of(factchecker), user_id=user_id):
                fc: ReactResult = await run_react(
                    ollama, health, tools,  # type: ignore[arg-type]
                    model=factchecker,
                    messages=fc_msgs,
                    options=options,
                    timeout_s=timeout_s,
                    max_rounds=fc_budget["max_rounds"],
                    compress_after_round=fc_budget["compress_after"],
                    max_tool_result_chars=fc_budget["max_tool_chars"],
                    tool_dispatch_timeout_s=float(fc_budget["dispatch_timeout_s"]),
                    compress_keep_last=fc_budget["compress_keep_last"],
                    max_web_searches=fc_budget["max_web_searches"],
                    user_id=user_id,
                    gate=None,  # gate held for the whole stage (as deep panel does)
                    location=registry.location_of(factchecker),
                    cfg=cfg,
                )
            corrections = (fc.content or "").strip()
            # Structure the fact-check against the claim ledger.
            # When a ledger exists, convert the prose notes into per-claim
            # verdicts and render them back into the corrections string the
            # writer already applies. Catches the "unsupported claim" class
            # (a source that doesn't actually back the claim). Fail-soft: on any
            # problem, keep the prose corrections unchanged.
            if ledger is not None and ledger.claims:
                fc_struct = await _structure_factcheck(
                    ollama, health, gate, cfg,
                    model=factchecker, location=registry.location_of(factchecker),
                    ledger=ledger, fc_notes=corrections,
                    timeout_s=timeout_s, user_id=user_id,
                )
                if fc_struct is not None:
                    fc_result = fc_struct  # carried to the Sources renderer below
                    rendered = _factcheck_result_to_corrections(fc_struct, ledger)
                    n_drop = sum(1 for c in fc_struct.checks if c.verdict == "unsupported")
                    n_hedge = sum(1 for c in fc_struct.checks if c.verdict in ("needs_hedge", "conflicting"))
                    log.info(
                        "research: factcheck ledger — %d checks (%d drop, %d hedge), %d fatal",
                        len(fc_struct.checks), n_drop, n_hedge, len(fc_struct.fatal_errors),
                    )
                    corrections = rendered
        except OllamaError as e:
            health.record_failure(factchecker, str(e))
            log.warning("research: fact-check stage failed: %s", e)
            corrections = ""
        except Exception:
            # The fact-check stage is optional — any unexpected error must
            # degrade to "no corrections", never break the answer.
            log.exception("research: fact-check stage errored; continuing without corrections")
            corrections = ""
    # Only signal the stage when it actually ran — a research request with no
    # factchecker configured (or no tools) skips it silently, so the route
    # doesn't render an empty Fact-checking banner.
    if fc_can_run:
        yield {"type": "factcheck_done", "ok": _has_corrections(corrections)}

    # ── Write stage (always runs) ──────────────────────────────────────
    writer = pool.get("writer")
    fallback = pool.get("fallback_synth")
    candidates = [m for m in (writer, fallback) if m]
    accumulated = ""
    writer_model = ""
    write_error = "no_writer" if not candidates else "write_failed"

    # Stage 4: deterministic per-claim hedging guidance (opt-in, separate flag).
    # Computed from the ledger's surviving claims + their source types — empty
    # (and so omitted from the writer block) when the ledger/flag is off, or when
    # the block would be all-hedge (see _render_dispositions_block).
    dispositions = ""
    if _hedge_policy_enabled(cfg):
        dispositions = _render_dispositions_block(ledger, fc_result)

    w_msgs = [
        {"role": "system", "content": prompt_from_config(cfg, "writer", WRITER_SYSTEM)},
        {
            "role": "user",
            "content": _write_user_block(
                user_text, findings, critique, corrections, dispositions
            ),
        },
    ]
    for model in candidates:
        if not health.is_healthy(model):
            continue
        loc = registry.location_of(model)
        dispatch_total.labels(model=model, task_type=str(task), path="research_write").inc()
        started = False
        try:
            async with gate.acquire(model, location=loc, user_id=user_id):
                async for chunk in ollama.chat_stream(
                    model=model, messages=w_msgs,
                    options={"temperature": 0.3}, timeout_s=timeout_s,
                ):
                    text = (chunk.get("message", {}) or {}).get("content", "") or ""
                    if text:
                        started = True
                        accumulated += text
                        yield {"type": "write_delta", "text": text}
                    if chunk.get("done"):
                        break
            health.record_success(model)
            if accumulated.strip():
                writer_model = model
                write_error = ""
                break
        except OllamaError as e:
            health.record_failure(model, str(e))
            log.warning("research: writer %s failed: %s", model, e)
            if started:
                # Partial answer already on the wire — can't fall back.
                write_error = "write_truncated"
                writer_model = model
                break

    # ── Stage 4 (cont.): append the deterministic Sources list ─────────
    # Built by us from the ledger (NOT asked of the writer — that pressure
    # degrades prose). Only on a clean answer, and only when a
    # grounded ledger yields surviving sources with usable URLs; an
    # ungrounded/creative answer renders nothing.
    if write_error == "" and accumulated.strip():
        sources_block = _render_sources_block(ledger, fc_result)
        if sources_block:
            accumulated += sources_block
            yield {"type": "write_delta", "text": sources_block}
            log.info(
                "research: Sources block appended (%d entries)",
                sources_block.count("\n- "),
            )

    yield {
        "type": "done",
        "content": accumulated,
        "writer_model": writer_model or "none",
        "drafts": drafts,
        "findings": findings,
        "critique": critique,
        "corrections": corrections,
        "ledger": ledger.model_dump() if ledger is not None else None,
        "factcheck": fc_result.model_dump() if fc_result is not None else None,
        "dispositions": dispositions,
        "error": write_error,
    }


async def run_research_pipeline(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    *,
    task: TaskType,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
    max_researchers_cloud: int,
    tools: ToolRegistry | None = None,
    tool_capable_models: set[str] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Non-streaming staged research pipeline. Returns a dict to merge into state.

    Keys: content, writer_model, drafts, research_findings, research_critique,
    research_factcheck, research_ledger, research_factcheck_ledger,
    research_dispositions, error. Drains the streaming variant so the two
    paths share one implementation and can't drift.
    """
    final: dict[str, Any] = {}
    async for evt in run_research_pipeline_streaming(
        cfg, ollama, registry, health, gate,
        task=task, messages=messages, options=options,
        timeout_s=timeout_s, max_researchers_cloud=max_researchers_cloud,
        tools=tools, tool_capable_models=tool_capable_models, user_id=user_id,
    ):
        if evt.get("type") == "done":
            final = evt
    return {
        "content": final.get("content", "") or "",
        "writer_model": final.get("writer_model", "none"),
        "drafts": final.get("drafts", []),
        "research_findings": final.get("findings", ""),
        "research_critique": final.get("critique", ""),
        "research_factcheck": final.get("corrections", ""),
        "research_ledger": final.get("ledger"),
        "research_factcheck_ledger": final.get("factcheck"),
        "research_dispositions": final.get("dispositions", ""),
        "error": final.get("error", ""),
    }


__all__ = [
    "pick_panel_timeout",
    "pool_key_for",
    "run_panel",
    "run_panel_streaming",
    "run_research_pipeline",
    "run_research_pipeline_streaming",
    "select_researchers",
    "select_workers",
]
