"""LangGraph pipeline assembly.

Phase 5 build:
    classify → complexity → (fast_path | deep_stub) → end

Deep-panel logic lands in Phase 6; until then `deep_stub` returns a clear
"not yet implemented" message so misrouting is loud, not silent.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from audrey.config import Config
from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient
from audrey.models.registry import ModelRegistry
from audrey.pipeline.classify import classify as classify_fn
from audrey.pipeline.complexity import is_complex
from audrey.pipeline.fast_path import run_fast_path
from audrey.pipeline.state import PipelineState

log = logging.getLogger(__name__)


def build_graph(
    cfg: Config,
    ollama: OllamaClient,
    registry: ModelRegistry,
    health: HealthTracker,
):
    """Compile the LangGraph StateGraph for this process.

    Nodes read from `app.state.*` via closures (bound at build time).
    """
    router_cfg = cfg.router
    complexity_threshold = int(cfg.raw.get("complexity", {}).get("token_threshold", 500))
    fast_timeout = float(cfg.timeouts.get("fast_path", 180))

    async def node_classify(state: PipelineState) -> dict[str, Any]:
        user_text = _last_user_text(state["messages"])
        task, reason, conf = await classify_fn(
            ollama,
            router_model=router_cfg.get("model", "qwen3:4b"),
            router_timeout_s=float(router_cfg.get("timeout_s", 20)),
            max_router_strikes=int(router_cfg.get("max_failures_before_fallback", 2)),
            user_text=user_text,
        )
        log.info("classify: %s (%s, conf=%.2f)", task, reason, conf)
        return {"task_type": task, "classify_reason": reason, "classify_confidence": conf}

    async def node_complexity(state: PipelineState) -> dict[str, Any]:
        complex_, n = is_complex(state["messages"], threshold=complexity_threshold)
        log.info("complexity: %d tokens (threshold=%d) -> %s", n, complexity_threshold,
                 "deep" if complex_ else "fast")
        mode = "deep" if complex_ else "fast"
        return {"prompt_tokens": n, "complex": complex_, "mode": mode}

    async def node_fast_path(state: PipelineState) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if (t := state.get("temperature")) is not None:
            options["temperature"] = t
        if (p := state.get("top_p")) is not None:
            options["top_p"] = p
        if (m := state.get("max_tokens")) is not None:
            options["num_predict"] = m

        concrete, resp = await run_fast_path(
            ollama, registry, health,
            task=state["task_type"],
            messages=state["messages"],
            options=options,
            timeout_s=fast_timeout,
        )
        msg = resp.get("message", {}) or {}
        return {
            "concrete_model": concrete,
            "content": msg.get("content", "") or "",
            "prompt_eval_count": int(resp.get("prompt_eval_count", 0) or 0),
            "eval_count": int(resp.get("eval_count", 0) or 0),
        }

    async def node_deep_stub(state: PipelineState) -> dict[str, Any]:
        # Phase 6 replaces this with real parallel-worker dispatch + synthesis.
        msg = (
            "[deep-panel not yet implemented — Phase 6]. "
            f"Classified as {state.get('task_type')} with "
            f"{state.get('prompt_tokens')} tokens."
        )
        log.warning("deep_stub hit: %s", msg)
        return {"content": msg, "concrete_model": "deep_stub"}

    def route_after_complexity(state: PipelineState) -> str:
        return state.get("mode", "fast")

    g: StateGraph = StateGraph(PipelineState)
    g.add_node("classify", node_classify)
    g.add_node("complexity", node_complexity)
    g.add_node("fast_path", node_fast_path)
    g.add_node("deep_stub", node_deep_stub)

    g.set_entry_point("classify")
    g.add_edge("classify", "complexity")
    g.add_conditional_edges(
        "complexity",
        route_after_complexity,
        {"fast": "fast_path", "deep": "deep_stub"},
    )
    g.add_edge("fast_path", END)
    g.add_edge("deep_stub", END)
    return g.compile()


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                # Multimodal: concatenate text parts
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


__all__ = ["build_graph"]
