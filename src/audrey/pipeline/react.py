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
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from audrey.models.health import HealthTracker
from audrey.models.ollama import OllamaClient, OllamaError
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


def _summarize_tool_message(msg: dict[str, Any]) -> str:
    name = msg.get("name", "?")
    content = msg.get("content", "") or ""
    return f"[earlier tool call: {name} -> {len(content)} chars elided]"


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
) -> ReactResult:
    """Drive the model through up to `max_rounds` of tool use, then return the answer.

    Errors from individual tool calls become tool messages — the model decides
    how to recover. A network/Ollama error in the chat call propagates as
    `OllamaError` to the caller (graph node) which surfaces it as a 502.
    """
    tools = registry.to_ollama_tools() if registry.by_name else None
    convo: list[dict[str, Any]] = list(messages)
    all_results: list[ToolResult] = []
    rounds_used = 0
    last_resp: dict[str, Any] = {}

    async with httpx.AsyncClient() as http:
        for round_idx in range(max_rounds):
            if round_idx >= compress_after_round:
                convo = _compress_history(convo, keep_last_round=1)

            start = time.monotonic()
            try:
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

            # Dispatch concurrently.
            results = await asyncio.gather(*[
                dispatch_one(
                    http, registry, tc,
                    max_result_chars=max_tool_result_chars,
                    timeout_s=tool_dispatch_timeout_s,
                    user_id=user_id,
                )
                for tc in tool_calls
            ])
            for r in results:
                convo.append(to_tool_message(r))
                all_results.append(r)
            rounds_used += 1

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
        convo = _compress_history(convo, keep_last_round=1)
        convo.append({
            "role": "user",
            "content": (
                "You have reached the tool-call budget. Do not call any more tools. "
                "Using only the information already gathered above, write the final "
                "answer to the original request now as plain prose. If the gathered "
                "information is insufficient, say so explicitly — do not fabricate."
            ),
        })
        try:
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
