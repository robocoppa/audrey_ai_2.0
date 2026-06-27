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

If `state["subtasks"]` is non-empty, workers are assigned to subtasks
round-robin so each draft answers a different slice. Otherwise every worker
answers the full prompt — the synthesizer reconciles them.
"""

from __future__ import annotations

import asyncio
import logging
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
    inlined_schema,
    parse_factcheck_result,
    parse_research_result,
)
from audrey.pipeline.messages import last_user_text
from audrey.pipeline.prompts import (
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
_POOL_KEYS = {
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
                    user_id=user_id,
                    gate=None,
                    location=location,
                    cfg=cfg,
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
            react_compress_keep_last=react_compress_keep_last,
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
        react_compress_keep_last=react_compress_keep_last, user_id=user_id,
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
        react_compress_keep_last=react_compress_keep_last, user_id=user_id,
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
# A three-stage pipeline behind the `audrey_research` virtual model:
#   Stage 1  RESEARCH  — parallel fan-out of tool-capable researchers that
#                        ground with web_search/kb_search. Cloud researchers
#                        run concurrently (capped at max_research_workers_cloud);
#                        a local researcher serializes through the GPU gate.
#   Stage 2  VERIFY    — one pass auditing the merged findings for unsupported
#                        / overconfident / anachronistic claims.
#   Stage 3  WRITE     — one pass turning verified findings into the answer.
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
    fact-checker to flag (Stage 2)."""
    seen_urls: dict[str, str] = {}  # url → canonical source id
    merged_sources: list[Source] = []
    for led in ledgers:
        for s in led.sources:
            key = s.url.strip().rstrip("/").lower()
            if key and key in seen_urls:
                continue
            if key:
                seen_urls[key] = s.id
            merged_sources.append(s)
    merged_claims: list[Claim] = []
    summaries: list[str] = []
    unresolved: list[str] = []
    for led in ledgers:
        merged_claims.extend(led.claims)
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


def _render_claims_for_factcheck(ledger: ResearchResult) -> str:
    """A compact claim list (id, risk, text) for the fact-check structuring
    prompt to reference by claim_id. Sources appended so the checker can judge
    support without re-fetching."""
    lines = ["CLAIMS:"]
    for c in ledger.claims:
        src = f" [sources: {', '.join(c.source_ids)}]" if c.source_ids else " [no source]"
        lines.append(f"- {c.id} (risk={c.risk}): {c.text}{src}")
    if ledger.sources:
        lines.append("\nSOURCES:")
        for s in ledger.sources:
            lines.append(f"- {s.id} ({s.source_type}): {s.title} — {s.url}")
    return "\n".join(lines)


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
    the claim ledger. Second mechanical pass (no tools), pinned to the
    FactCheckResult schema. Fail-soft: any error returns None and the caller
    keeps the prose corrections path."""
    if not ledger.claims:
        return None
    block = (
        f"{_render_claims_for_factcheck(ledger)}\n\n"
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


def _factcheck_result_to_corrections(fc: FactCheckResult, ledger: ResearchResult) -> str:
    """Render structured per-claim verdicts into the prose corrections string the
    writer already applies — so Stage 2's verdicts reach the writer through the
    existing channel (no writer change until Stage 3). Lines mirror the prose
    fact-checker's vocabulary so `_has_corrections` and the writer prompt work
    unchanged:
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
       "findings": str, "critique": str, "corrections": str, "error": str}
          Always emitted last. `error` is "" on success, or "no_writer" /
          "write_failed" when the write stage could not produce an answer.

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

    findings = _format_findings(drafts)
    grounded = bool(findings)

    # ── Stage 1b: structure the findings into a claim/source ledger ────
    # Opt-in (agentic.research_ledger.enabled). A second mechanical pass per
    # worker converts prose → ResearchResult; fail-soft, and the prose
    # `findings` is unchanged either way (the ledger is carried alongside it for
    # the later stages to consume). This stage's only measurable effect on the
    # ANSWER is none yet — it just produces the ledger; we eval that the prose
    # path is undisturbed before wiring the ledger into verify/write/hedge.
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
                    user_id=user_id,
                    gate=None,  # gate held for the whole stage (as deep panel does)
                    location=registry.location_of(factchecker),
                    cfg=cfg,
                )
            corrections = (fc.content or "").strip()
            # ── Stage 2: structure the fact-check against the claim ledger ──
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

    # ── Stage 4: write (always runs) ───────────────────────────────────
    writer = pool.get("writer")
    fallback = pool.get("fallback_synth")
    candidates = [m for m in (writer, fallback) if m]
    accumulated = ""
    writer_model = ""
    write_error = "no_writer" if not candidates else "write_failed"

    w_msgs = [
        {"role": "system", "content": prompt_from_config(cfg, "writer", WRITER_SYSTEM)},
        {"role": "user", "content": _write_user_block(user_text, findings, critique, corrections)},
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

    yield {
        "type": "done",
        "content": accumulated,
        "writer_model": writer_model or "none",
        "drafts": drafts,
        "findings": findings,
        "critique": critique,
        "corrections": corrections,
        "ledger": ledger.model_dump() if ledger is not None else None,
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
    research_factcheck, error. Drains the streaming variant so the two paths
    share one implementation and can't drift.
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
