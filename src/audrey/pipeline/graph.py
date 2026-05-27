"""LangGraph pipeline assembly.

Topology:

    datetime ─► memory_recall ─► classify ─► complexity ─┬─► fast_path ─┬─► END
                                                         │              └─► escalate ─┐
                                                         │                            ▼
                                                         └────────────────────► planner ─► deep_panel ─► synthesize ─► reflect ─┬─► END
                                                                                                                                └─► deep_panel (retry)

Node responsibilities (one paragraph each):

`datetime` prepends a system message with the current ISO-8601 server time
so every downstream node and the model itself reasons from the same "today"
rather than from training cutoff. See `pipeline/context.py`.

`memory_recall` runs next: when `state["user_id"]` is set and the
`memory_search` tool is registered, it semantic-searches the user's stored
memories and prepends any hits as a system message. No-op otherwise.

`classify` decides the task family (code / reasoning / general / vl) via
keyword pre-filter then a router model. `complexity` decides fast vs deep
based on prompt token count and the virtual-model preference.

`escalate?` is the adaptive hook from `fast_path`: if the fast answer is
too short or low-confidence, the graph re-enters in deep mode. `retry?` is
the reflection loop — at most one extra deep-panel pass.

Virtual-model routing (decided by `node_complexity`):
  audrey_deep / audrey_cloud / audrey_local — always deep
  audrey_fast — always fast, escalation suppressed in `route_after_fast_path`
  audrey_auto — adaptive: deep when prompt ≥ token_threshold, fast otherwise

Both `fast_path` and `deep_panel` workers run ReAct when the chosen model is
in `fast_path.tool_capable_models` and the tool registry is non-empty. Deep
workers use `agentic.react.deep_worker.*` (tighter per-worker budget). The
fast-path → deep escalation guard skips when `tool_rounds > 0` —
re-running through deep workers rarely improves an already-grounded answer.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from audrey.config import Config
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.classify import classify_with_registry
from audrey.pipeline.complexity import (
    count_last_user_tokens,
    count_tokens_by_role,
    is_complex,
    is_owui_task_request,
)
from audrey.pipeline.context import datetime_system_message
from audrey.pipeline.deep_panel import pool_key_for, run_panel
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.fast_path import run_fast_path
from audrey.pipeline.memory import (
    MEMORY_STORE_TOOL,
    memory_system_message,
    recall_for_request,
)
from audrey.pipeline.messages import last_user_text
from audrey.pipeline.planner import plan as planner_plan
from audrey.pipeline.prompts import compose_system_messages
from audrey.pipeline.reflect import reflect as reflect_fn
from audrey.pipeline.state import PipelineState
from audrey.pipeline.synthesize import synthesize as synthesize_fn
from audrey.tools.discovery import ToolRegistry

log = logging.getLogger(__name__)


def build_graph(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
    gate: FairLocalGate,
    tools: ToolRegistry,
):
    """Compile the LangGraph StateGraph for this process."""
    router_cfg = cfg.router
    complexity_cfg = cfg.raw.get("complexity", {}) or {}
    complexity_threshold = int(complexity_cfg.get("token_threshold", 500))
    complexity_log_breakdown = bool(complexity_cfg.get("log_breakdown", False))
    fast_timeout = float(cfg.timeouts.get("fast_path", 180))
    deep_worker_timeout = float(cfg.timeouts.get("deep_worker", 240))
    cloud_timeout = float(cfg.timeouts.get("cloud", 120))
    router_timeout = float(router_cfg.get("timeout_s", 20))

    fast_path_cfg = cfg.raw.get("fast_path", {}) or {}
    tool_capable_models = set(fast_path_cfg.get("tool_capable_models", []) or [])
    react_cfg = cfg.raw.get("agentic", {}).get("react", {}) or {}
    react_max_rounds = int(react_cfg.get("max_rounds", 3))
    react_compress_after = int(react_cfg.get("compress_after_round", 2))
    react_max_tool_chars = int(react_cfg.get("max_tool_result_chars", 2000))
    react_dispatch_timeout = float(react_cfg.get("dispatch_timeout_s", 30))
    react_compress_keep_last = int(react_cfg.get("compress_keep_last", 1))

    # Deep-panel workers get a separate ReAct budget — tighter by default
    # because N workers × M rounds multiplies, and local workers hold the
    # GPU gate for the whole loop.
    deep_react_cfg = react_cfg.get("deep_worker", {}) or {}
    deep_react_max_rounds = int(deep_react_cfg.get("max_rounds", 2))
    deep_react_compress_after = int(deep_react_cfg.get("compress_after_round", react_compress_after))
    deep_react_max_tool_chars = int(deep_react_cfg.get("max_tool_result_chars", react_max_tool_chars))
    deep_react_dispatch_timeout = float(deep_react_cfg.get("dispatch_timeout_s", react_dispatch_timeout))
    deep_react_compress_keep_last = int(deep_react_cfg.get("compress_keep_last", react_compress_keep_last))

    agentic = cfg.raw.get("agentic", {})
    planning_cfg = agentic.get("planning", {}) or {}
    planning_enabled = bool(planning_cfg.get("enabled", True))
    planning_min_tokens = int(planning_cfg.get("min_prompt_tokens", 40))
    planning_max_subtasks = int(planning_cfg.get("max_subtasks", 3))
    reflection_cfg = agentic.get("reflection", {}) or {}
    reflection_enabled = bool(reflection_cfg.get("enabled", True))
    reflection_max_retries = int(reflection_cfg.get("max_retries", 1))
    reflection_min_chars = int(reflection_cfg.get("min_answer_chars", 80))
    escalation_cfg = agentic.get("escalation", {}) or {}
    escalation_enabled = bool(escalation_cfg.get("enabled", True))
    escalation_min_chars = int(escalation_cfg.get("min_chars", 100))
    escalation_conf_ceiling = float(escalation_cfg.get("confidence_ceiling", 0.95))

    memory_cfg = agentic.get("memory", {}) or {}
    memory_enabled = bool(memory_cfg.get("enabled", True))
    memory_top_k = int(memory_cfg.get("top_k", 3))
    memory_timeout_s = float(memory_cfg.get("timeout_s", 5))

    max_workers_cloud = int(agentic.get("max_deep_workers_cloud", 3))

    # ── Nodes ─────────────────────────────────────────────────────────

    async def node_datetime(state: PipelineState) -> dict[str, Any]:
        """Prepend a system message with the current date/time.

        Runs first so every downstream node — memory_recall, classify,
        fast_path, deep_panel workers, synth — sees the same timestamp.
        Cheap, no network calls; just a string + a list prepend.
        """
        sys_msg = datetime_system_message()
        return {"messages": [sys_msg, *state["messages"]]}

    async def node_memory_recall(state: PipelineState) -> dict[str, Any]:
        """Keyword-search the user's memories, inject hits + store-hint as system message.

        No-op when memory is disabled or the user isn't identified. When the
        user *is* identified and `memory_store` is available as a tool, we
        always prepend the "write durable facts via memory_store" hint so
        tool-capable models learn to save things — even on requests that
        returned zero recall hits.
        """
        if not memory_enabled:
            return {}
        user_id = (state.get("user_id") or "").strip()
        if not user_id:
            return {}
        hits = await recall_for_request(
            tools, user_id=user_id, messages=state["messages"],
            top_k=memory_top_k, timeout_s=memory_timeout_s,
        )
        include_store_hint = tools is not None and MEMORY_STORE_TOOL in tools.by_name
        sys_msg = memory_system_message(
            hits, user_id=user_id, include_store_hint=include_store_hint, cfg=cfg,
        )
        # Chat-history-search guidance lands only when the live registry
        # has the tool — telling a model how to use a tool it can't
        # dispatch is wasted tokens. Composer enforces the canonical
        # order: memory message first, chat-history guidance after.
        chat_history_available = tools is not None and "chat_history_search" in tools.by_name
        composed = compose_system_messages(
            memory_hint=sys_msg,
            chat_history_guidance=chat_history_available,
        )
        if not composed:
            log.info("memory: no hits / no store hint for user=%s", user_id)
            return {"memory_hits": hits}
        new_messages = [*composed, *state["messages"]]
        log.info(
            "memory: user=%s hits=%d keys=%s store_hint=%s chat_history_hint=%s",
            user_id, len(hits), [h.get("key", "?") for h in hits],
            "on" if include_store_hint else "off",
            "on" if chat_history_available else "off",
        )
        return {"memory_hits": hits, "messages": new_messages}

    async def node_classify(state: PipelineState) -> dict[str, Any]:
        # The shared helper passes current tool names into `classify(...)`
        # so prompts that explicitly name a tool (e.g. "use kb_image_search
        # …") route through the tool-capable fast path instead of getting
        # trapped by `vl_strong`/code keywords. Same call shape as the
        # streaming path in routes/openai.py.
        task, reason, conf = await classify_with_registry(
            ollama,
            user_text=last_user_text(state["messages"]),
            router_cfg=router_cfg,
            cfg=cfg,
            registry=tools,
        )
        log.info("classify: %s (%s, conf=%.2f)", task, reason, conf)
        return {"task_type": task, "classify_reason": reason, "classify_confidence": conf}

    async def node_complexity(state: PipelineState) -> dict[str, Any]:
        complex_, n = is_complex(state["messages"], threshold=complexity_threshold)
        vm = state.get("virtual_model")
        forced_deep = vm in ("audrey_deep", "audrey_cloud", "audrey_local")
        forced_fast = vm == "audrey_fast"
        owui_task = is_owui_task_request(state["messages"])
        if owui_task:
            mode = "fast"
            reason = "owui_task"
        elif forced_deep:
            mode = "deep"
            reason = "forced_by_virtual_model"
        elif forced_fast:
            mode = "fast"
            reason = "forced_by_virtual_model"
        elif complex_:
            mode = "deep"
            reason = f"tokens>={complexity_threshold}"
        else:
            mode = "fast"
            reason = f"tokens<{complexity_threshold}"
        log.info("complexity: %d tokens -> %s (%s)", n, mode, reason)
        if complexity_log_breakdown:
            by_role = count_tokens_by_role(state["messages"])
            last_user = count_last_user_tokens(state["messages"])
            parts = " ".join(f"{r}={by_role[r]}" for r in sorted(by_role))
            log.info("complexity.breakdown: %s last_user=%d", parts, last_user)
        return {"prompt_tokens": n, "complex": complex_, "mode": mode}

    async def node_fast_path(state: PipelineState) -> dict[str, Any]:
        options = _options_from_state(state)
        concrete, resp = await run_fast_path(
            ollama, registry, health, gate,
            task=state["task_type"],
            messages=state["messages"],
            options=options,
            timeout_s=fast_timeout,
            tools=tools,
            tool_capable_models=tool_capable_models,
            react_max_rounds=react_max_rounds,
            react_compress_after=react_compress_after,
            react_max_tool_chars=react_max_tool_chars,
            react_dispatch_timeout_s=react_dispatch_timeout,
            react_compress_keep_last=react_compress_keep_last,
            user_id=(state.get("user_id") or None),
            cfg=cfg,
        )
        msg = resp.get("message", {}) or {}
        react_meta = resp.get("_react") or {}
        return {
            "concrete_model": concrete,
            "content": msg.get("content", "") or "",
            "prompt_eval_count": int(resp.get("prompt_eval_count", 0) or 0),
            "eval_count": int(resp.get("eval_count", 0) or 0),
            "tool_rounds": int(react_meta.get("tool_rounds", 0)),
            "tool_calls_log": list(react_meta.get("tool_calls", []) or []),
        }

    async def node_planner(state: PipelineState) -> dict[str, Any]:
        if not planning_enabled or state.get("prompt_tokens", 0) < planning_min_tokens:
            return {"subtasks": []}
        user_text = last_user_text(state["messages"])
        subs = await planner_plan(
            ollama,
            planner_model=router_cfg.get("model", "qwen3:4b"),
            user_text=user_text,
            timeout_s=router_timeout,
            max_subtasks=planning_max_subtasks,
            cfg=cfg,
        )
        if subs:
            log.info("planner: %d subtasks: %s", len(subs), [s[:60] for s in subs])
        else:
            log.info("planner: no decomposition (atomic prompt)")
        return {"subtasks": subs}

    async def node_deep_panel(state: PipelineState) -> dict[str, Any]:
        pool_key = pool_key_for(state["virtual_model"])
        # Cloud-only pool uses cloud timeout; otherwise the deep-worker timeout.
        timeout_s = cloud_timeout if pool_key == "deep_panel_cloud" else deep_worker_timeout
        options = _options_from_state(state)
        drafts, attempted = await run_panel(
            cfg, ollama, registry, health, gate,
            pool_key=pool_key,
            task=state["task_type"],
            messages=state["messages"],
            subtasks=list(state.get("subtasks") or []),
            options=options,
            timeout_s=timeout_s,
            max_workers_cloud=max_workers_cloud,
            tools=tools,
            tool_capable_models=tool_capable_models,
            react_max_rounds=deep_react_max_rounds,
            react_compress_after=deep_react_compress_after,
            react_max_tool_chars=deep_react_max_tool_chars,
            react_dispatch_timeout_s=deep_react_dispatch_timeout,
            react_compress_keep_last=deep_react_compress_keep_last,
            user_id=(state.get("user_id") or None),
        )
        ok = sum(1 for d in drafts if (d.get("content") or "").strip())
        grounded = sum(1 for d in drafts if int(d.get("tool_rounds", 0) or 0) > 0)
        log.info("deep_panel: pool=%s task=%s workers=%d ok=%d tool_grounded=%d attempted=%s",
                 pool_key, state["task_type"], len(drafts), ok, grounded, attempted)
        return {
            "panel_pool": pool_key,
            "workers_attempted": attempted,
            "drafts": drafts,
        }

    async def node_synthesize(state: PipelineState) -> dict[str, Any]:
        pool_key = state.get("panel_pool") or pool_key_for(state["virtual_model"])
        result = await synthesize_fn(
            cfg, ollama, registry, health, gate,
            pool_key=pool_key,
            task=state["task_type"],
            messages=state["messages"],
            drafts=list(state.get("drafts") or []),
            subtasks=list(state.get("subtasks") or []),
            timeout_s=deep_worker_timeout,
            user_id=(state.get("user_id") or None),
        )
        # Concrete_model exposed to the caller is the synthesizer (or the
        # fallback tag, e.g. "fallback:longest_draft").
        return {
            **result,
            "concrete_model": result.get("synthesizer_model", "deep_panel"),
        }

    async def node_reflect(state: PipelineState) -> dict[str, Any]:
        if not reflection_enabled:
            return {"reflect_attempts": state.get("reflect_attempts", 0),
                    "reflect_passed": True, "reflect_reason": "disabled"}
        attempts = int(state.get("reflect_attempts", 0))
        result = reflect_fn(
            content=state.get("content", "") or "",
            synth_error=state.get("synth_error", "") or "",
            min_chars=reflection_min_chars,
            user_text=last_user_text(state["messages"]),
        )
        log.info("reflect: attempt=%d passed=%s reason=%s", attempts + 1, result.passed, result.reason)
        return {
            "reflect_attempts": attempts + 1,
            "reflect_passed": result.passed,
            "reflect_reason": result.reason,
        }

    # ── Routers ───────────────────────────────────────────────────────

    def route_after_complexity(state: PipelineState) -> str:
        return "fast" if state.get("mode") == "fast" else "deep"

    def route_after_fast_path(state: PipelineState) -> str:
        if not escalation_enabled:
            return "end"
        if state.get("virtual_model") == "audrey_fast":
            # audrey_fast is the always-fast virtual model — never escalates,
            # even on long prompts. Users who want adaptive routing should
            # use audrey_auto; users who want forced deep should use
            # audrey_deep / audrey_cloud / audrey_local.
            return "end"
        if state.get("escalated_from_fast"):
            return "end"  # already came from an escalation; don't loop
        if int(state.get("tool_rounds", 0)) > 0:
            # Fast path used tools — the answer is grounded in real data.
            # Re-running through tool-blind deep workers can only degrade it.
            return "end"
        if state.get("memory_hits"):
            # Same reasoning as tool_rounds: the answer is grounded in recalled
            # memory. Deep workers wash out short-but-correct memory answers
            # ("You're running a Threadripper 7970X" → 93 chars → would escalate
            # on length alone → deep workers don't know the user and give
            # generic "I can't see your computer" responses).
            return "end"
        content = (state.get("content") or "").strip()
        conf = float(state.get("classify_confidence", 0.0))
        too_short = len(content) < escalation_min_chars
        low_confidence = conf < escalation_conf_ceiling and conf > 0
        if too_short or low_confidence:
            log.info("escalate: fast→deep (chars=%d, conf=%.2f, reason=%s)",
                     len(content), conf, "too_short" if too_short else "low_conf")
            return "escalate"
        return "end"

    def route_after_reflect(state: PipelineState) -> str:
        if state.get("reflect_passed"):
            return "end"
        if int(state.get("reflect_attempts", 0)) > reflection_max_retries:
            log.warning("reflect: out of retries (%d), shipping degraded answer",
                        state.get("reflect_attempts", 0))
            return "end"
        log.info("reflect: retrying deep panel (attempt %d/%d)",
                 state.get("reflect_attempts", 0), reflection_max_retries + 1)
        return "retry"

    async def node_mark_escalated(state: PipelineState) -> dict[str, Any]:
        # Bridge between fast_path and the deep branch — flag that we're in
        # an escalation so the deep branch knows not to loop back to fast.
        return {"escalated_from_fast": True, "mode": "deep"}

    # ── Graph wiring ──────────────────────────────────────────────────

    g: StateGraph = StateGraph(PipelineState)
    g.add_node("datetime", node_datetime)
    g.add_node("memory_recall", node_memory_recall)
    g.add_node("classify", node_classify)
    g.add_node("complexity", node_complexity)
    g.add_node("fast_path", node_fast_path)
    g.add_node("escalate_bridge", node_mark_escalated)
    g.add_node("planner", node_planner)
    g.add_node("deep_panel", node_deep_panel)
    g.add_node("synthesize", node_synthesize)
    g.add_node("reflect", node_reflect)

    g.set_entry_point("datetime")
    g.add_edge("datetime", "memory_recall")
    g.add_edge("memory_recall", "classify")
    g.add_edge("classify", "complexity")
    g.add_conditional_edges(
        "complexity", route_after_complexity,
        {"fast": "fast_path", "deep": "planner"},
    )
    g.add_conditional_edges(
        "fast_path", route_after_fast_path,
        {"end": END, "escalate": "escalate_bridge"},
    )
    g.add_edge("escalate_bridge", "planner")
    g.add_edge("planner", "deep_panel")
    g.add_edge("deep_panel", "synthesize")
    g.add_edge("synthesize", "reflect")
    g.add_conditional_edges(
        "reflect", route_after_reflect,
        {"end": END, "retry": "deep_panel"},
    )
    return g.compile()


# ─── Helpers ──────────────────────────────────────────────────────────

def _options_from_state(state: PipelineState) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if (t := state.get("temperature")) is not None:
        options["temperature"] = t
    if (p := state.get("top_p")) is not None:
        options["top_p"] = p
    if (m := state.get("max_tokens")) is not None:
        options["num_predict"] = m
    return options


__all__ = ["build_graph"]
