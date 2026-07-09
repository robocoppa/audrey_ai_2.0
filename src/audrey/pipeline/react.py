"""ReAct loop wrapping the fast path.

Flow per request (when the chosen model is tool-capable and tools exist):
    round 0: chat(messages, tools)            ─► tool_calls? ─► dispatch in parallel
       │                                              │
       │ no tool_calls                                ▼
       └─► return content                       append tool messages
                                                      │
    round 1+: chat(messages, tools)             … until max_rounds
       │
       │ at max_rounds: chat(messages) without tools ─► force a final answer
       └─► return content

Context compression: after `compress_after_round` rounds, prior `role=tool`
messages get collapsed into a short summary line so the prompt doesn't bloat
into the context window. The model still sees the most recent tool round
verbatim — only older rounds are summarized.

Tool dispatch is fully concurrent within a round (Ollama may emit multiple
tool_calls in one assistant turn). Errors come back as tool messages so the
model can decide how to recover.

When a `FairLocalGate` is supplied, each `ollama.chat` is wrapped with
`gate.acquire(...)`. Tool dispatch sits *outside* the gate hold, so a slow
tool call doesn't head-of-line-block other users' GPU work. The gate is
re-acquired for the next round; if another user slipped in between, the
model may be reloaded — that's the trade we accept to keep tool-wait time
non-blocking. Deep panel passes `gate=None` because it holds the gate at
the whole-worker level (one acquire across all rounds, by design — see
`deep_panel._run_one_worker`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.prompts import REACT_FINAL_ANSWER_USER, prompt_from_config
from audrey.tools.discovery import ToolRegistry
from audrey.tools.dispatch import ToolResult, dispatch_one, to_tool_message

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ReactResult:
    content: str
    tool_rounds: int                          # how many rounds invoked tools
    tool_calls: list[ToolResult] = field(default_factory=list)
    prompt_eval_count: int = 0
    eval_count: int = 0


def _debug_research_trace(cfg: Any) -> bool:
    """True when `agentic.debug_research_trace` is on. Defensive against a
    missing/None cfg so the diagnostic can never break the researcher loop."""
    try:
        return bool(cfg.raw.get("agentic", {}).get("debug_research_trace", False))
    except AttributeError:
        return False


def _summarize_tool_message(msg: dict[str, Any]) -> str:
    # Replaces an older tool result during history compaction (see
    # _compress_history). Wording is deliberate: a model that quotes its own
    # compacted scratchpad must NOT make this read as a *failed* or *empty*
    # search. The old text ("earlier tool call: … N chars elided") leaked into
    # research answers as "searches returned sparse or elided results" — the
    # model narrating its own compaction as a grounding failure. This phrasing
    # makes the omission unambiguous and carries no count to misread as "thin".
    name = msg.get("name", "?")
    return f"[history compacted: an earlier `{name}` result is omitted here to save context]"


# Tool message for a web_search call skipped because the per-request budget is
# spent. Same wording discipline as the compaction stub above: nothing here may
# read as a FAILED search ("error"/"failed"/"empty"), or the model narrates a
# grounding failure that never happened. "Limit reached" is truthful and safe
# to narrate. JSON-shaped like a real result body.
_WEB_SEARCH_BUDGET_STUB = (
    '{"note": "web_search limit for this request reached; no further web '
    'searches will run. Answer from the evidence already gathered."}'
)


def _tool_name(tc: dict[str, Any]) -> str:
    return str(((tc.get("function") or {}).get("name")) or "?")


def _without_web_search(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """The offered-tools list minus web_search; None if nothing remains."""
    if not tools:
        return tools
    kept = [t for t in tools if ((t.get("function") or {}).get("name")) != "web_search"]
    return kept or None


def _compress_history(messages: list[dict[str, Any]], *, keep_last_round: int) -> list[dict[str, Any]]:
    """Replace older tool messages with one-line summaries, keep the last N tool messages verbatim."""
    out: list[dict[str, Any]] = []
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_last_round:
        return list(messages)
    keep_threshold = tool_indices[-keep_last_round]
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i < keep_threshold:
            out.append({"role": "system", "content": _summarize_tool_message(m)})
        else:
            out.append(m)
    return out


@asynccontextmanager
async def _noop_gate():
    yield


def _gate_ctx(gate: FairLocalGate | None, model: str, location: str, user_id: str | None):
    """Acquire the gate per-chat-call when supplied; no-op when None or cloud.

    Deep panel passes `gate=None` because it holds the gate across the whole
    worker (rounds + tool dispatch). Fast path passes a real gate so the
    tool-dispatch window between rounds releases the GPU for other users.
    """
    if gate is None or location != "local":
        return _noop_gate()
    return gate.acquire(model, location=location, user_id=user_id)


async def run_react(
    ollama: OllamaClient,
    health: HealthTracker,
    registry: ToolRegistry,
    *,
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    timeout_s: float,
    max_rounds: int,
    compress_after_round: int,
    max_tool_result_chars: int,
    tool_dispatch_timeout_s: float,
    user_id: str | None = None,
    gate: FairLocalGate | None = None,
    location: str = "local",
    cfg: Any = None,
    compress_keep_last: int = 1,
    max_web_searches: int = 0,
) -> ReactResult:
    """Drive the model through up to `max_rounds` of tool use, then return the answer.

    Errors from individual tool calls become tool messages — the model decides
    how to recover. A network/Ollama error in the chat call propagates as
    `OllamaError` to the caller (graph node) which surfaces it as a 502.

    `max_web_searches` caps how many `web_search` calls this loop will
    dispatch (0 = unlimited). Every dispatched web search spends real
    provider quota (Brave), and a research request multiplies the spend by
    the panel width, so the cap is per-worker. Calls over budget are not
    dispatched: the model gets a budget-note tool message instead, and
    web_search stops being offered in later rounds. Other tools (kb_search,
    memory) are free/local and never capped.
    """
    tools = registry.to_ollama_tools() if registry.by_name else None
    convo: list[dict[str, Any]] = list(messages)
    all_results: list[ToolResult] = []
    rounds_used = 0
    last_resp: dict[str, Any] = {}
    web_searches_used = 0

    async with httpx.AsyncClient() as http:
        for round_idx in range(max_rounds):
            if round_idx >= compress_after_round:
                convo = _compress_history(convo, keep_last_round=compress_keep_last)

            # When the research-trace debug flag is on, log the web_search tool
            # content actually in the model's context this round — so a thin
            # ledger can be diagnosed as "grounding never reached the model"
            # vs. "model discarded grounding it received". Read-only over
            # `convo`; ships dark with the flag (agentic.debug_research_trace).
            if _debug_research_trace(cfg):
                for _m in convo:
                    if _m.get("role") == "tool" and "url" in str(_m.get("content", "")).lower():
                        _c = str(_m.get("content", ""))
                        log.info("research-trace: model=%s round=%d web_search in context: %d chars, head=%r",
                                 model, round_idx, len(_c), _c[:200])

            start = time.monotonic()
            try:
                async with _gate_ctx(gate, model, location, user_id):
                    last_resp = await ollama.chat(
                        model=model, messages=convo, options=options or None,
                        tools=tools, timeout_s=timeout_s,
                    )
                health.record_success(model)
            except OllamaError as e:
                health.record_failure(model, str(e))
                raise

            msg = last_resp.get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            log.info("react: round=%d model=%s tool_calls=%d (%.2fs)",
                     round_idx, model, len(tool_calls), time.monotonic() - start)

            if not tool_calls:
                return ReactResult(
                    content=msg.get("content", "") or "",
                    tool_rounds=rounds_used,
                    tool_calls=all_results,
                    prompt_eval_count=int(last_resp.get("prompt_eval_count", 0) or 0),
                    eval_count=int(last_resp.get("eval_count", 0) or 0),
                )

            # The assistant's tool-call turn must be in history before we add tool results.
            convo.append({
                "role": "assistant",
                "content": msg.get("content", "") or "",
                "tool_calls": tool_calls,
            })

            # Spend the web_search budget in call order; stub the overflow.
            # Stubs answer the model's call (every tool_call needs a tool
            # message) but are NOT recorded in all_results — the tools-used
            # footer must count real dispatches only.
            to_dispatch: list[dict[str, Any]] = []
            stubs: list[ToolResult] = []
            for tc in tool_calls:
                name = _tool_name(tc)
                if name == "web_search" and max_web_searches > 0:
                    if web_searches_used >= max_web_searches:
                        stubs.append(ToolResult(
                            name=name, call_id=tc.get("id"),
                            content=_WEB_SEARCH_BUDGET_STUB,
                            elapsed_s=0.0, is_error=False,
                        ))
                        continue
                    web_searches_used += 1
                to_dispatch.append(tc)
            if stubs:
                log.info("react: web_search budget (%d) reached for %s; skipped %d call(s)",
                         max_web_searches, model, len(stubs))

            # Dispatch concurrently.
            results = await asyncio.gather(*[
                dispatch_one(
                    http, registry, tc,
                    max_result_chars=max_tool_result_chars,
                    timeout_s=tool_dispatch_timeout_s,
                    user_id=user_id,
                )
                for tc in to_dispatch
            ])
            for r in results:
                convo.append(to_tool_message(r))
                all_results.append(r)
            for s in stubs:
                convo.append(to_tool_message(s))
            rounds_used += 1

            # Budget spent → stop offering web_search in later rounds so the
            # model pivots to writing (or free tools) instead of re-calling.
            if max_web_searches > 0 and web_searches_used >= max_web_searches:
                tools = _without_web_search(tools)

        # Hit max_rounds and the model is still asking for tools. Force a final
        # pass without tools so it has to commit to an answer.
        #
        # Two things matter here:
        #   1. Compress older tool messages so the convo stays small.
        #   2. Explicitly prompt the model to write prose. Just setting
        #      tools=None is too weak a signal after N rounds of tool-calling:
        #      models can stall (no bytes for 4+ minutes) or try to emit a
        #      pseudo-tool-call in plain text. A direct user turn saying
        #      "write the final answer now" flips the mode cleanly.
        log.warning("react: max_rounds=%d reached for %s; forcing final answer without tools",
                    max_rounds, model)
        convo = _compress_history(convo, keep_last_round=compress_keep_last)
        # Final-answer instruction lives in pipeline/prompts.py; this is
        # the load-bearing flip from tool-using mode to prose mode. Kept
        # as a user turn rather than a system message because it's a
        # mode-change signal, not background context. `agentic.prompts.
        # react_final_answer` overrides the default when supplied.
        final_answer_text = prompt_from_config(cfg, "react_final_answer", REACT_FINAL_ANSWER_USER)
        convo.append({"role": "user", "content": final_answer_text})
        try:
            async with _gate_ctx(gate, model, location, user_id):
                final = await ollama.chat(
                    model=model, messages=convo, options=options or None,
                    tools=None, timeout_s=timeout_s,
                )
            health.record_success(model)
        except OllamaError as e:
            health.record_failure(model, str(e))
            raise
        msg = final.get("message", {}) or {}
        return ReactResult(
            content=msg.get("content", "") or "",
            tool_rounds=rounds_used,
            tool_calls=all_results,
            prompt_eval_count=int(final.get("prompt_eval_count", 0) or 0),
            eval_count=int(final.get("eval_count", 0) or 0),
        )


__all__ = ["run_react", "ReactResult"]
