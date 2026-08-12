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
import json
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
    # Total chars of successful web_search tool-result bodies dispatched this
    # loop — i.e. how much retrieved web content actually reached the model's
    # context. The research-trace diagnostic distinguishes "worker never
    # retrieved" (≈0) from "worker retrieved but wrote thin grounding anyway"
    # (large). Successful web_search only: errors and the budget stub carry no
    # grounding, so counting them would inflate the "reached the model" figure.
    web_search_chars: int = 0


def _debug_research_trace(cfg: Any) -> bool:
    """True when `agentic.debug_research_trace` is on. Defensive against a
    missing/None cfg so the diagnostic can never break the researcher loop."""
    try:
        return bool(cfg.raw.get("agentic", {}).get("debug_research_trace", False))
    except AttributeError:
        return False


def _debug_context_trace(cfg: Any) -> bool:
    """True when `agentic.debug_context_trace` is on.

    Separate flag from the research trace, and deliberately so: this one is
    about EVERY tool, on every path, and its whole point is to be turned on
    for one diagnostic run and off again.
    """
    try:
        return bool(cfg.raw.get("agentic", {}).get("debug_context_trace", False))
    except AttributeError:
        return False


def _context_census(messages: list[dict[str, Any]]) -> str:
    """One line describing what the model is about to be shown.

    ⚠️ Written 2026-08-12 because fourteen A-B runs were diagnosed entirely
    from OUTPUT, and the output cannot distinguish the two things that matter:
    a model that invented "I only have a partial excerpt", and a model
    accurately describing a context that `_compress_history` had already
    thinned to one tool result.

    The correlation that prompted it, over every archived video answer:
    failures run 7–17% at 0–2 successful tool calls and **41% at three** —
    which is exactly where `compress_after_round: 2` and `compress_keep_last:
    1` leave a single tool message verbatim. That is a hypothesis this line
    settles in one run, and no amount of answer-reading ever could.

    Reports live tool results, compaction stubs, and the total character count
    of the whole conversation — the last because nothing in this repo sets
    `num_ctx`, so Ollama's default applies and over-long contexts are
    truncated from the FRONT, silently taking the system prompt with them.
    """
    live: list[str] = []
    stubbed = 0
    total = 0
    for m in messages:
        content = str(m.get("content") or "")
        total += len(content)
        if m.get("role") == "tool":
            live.append(f"{m.get('name', '?')}:{len(content)}")
        elif m.get("role") == "system" and content.startswith("[history compacted:"):
            stubbed += 1
    return (f"live_tool_results=[{', '.join(live) or '-'}] "
            f"compacted_out={stubbed} convo_chars={total}")


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


def _is_error_tool_message(msg: dict[str, Any]) -> bool:
    """True when a tool message carries a dispatch failure body.

    `dispatch.py` serialises every failure as `{"error": …}` — timeout,
    network_error, http_4xx/5xx, unknown_tool, malformed arguments. Sniffing the
    content here, rather than threading `ToolResult.is_error` into the message,
    keeps the wire shape untouched: `_to_ollama_messages` forwards every key on a
    message verbatim, so an extra bookkeeping field would be sent to the model.
    """
    content = msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith('{"error"')


def _compress_history(messages: list[dict[str, Any]], *, keep_last_round: int) -> list[dict[str, Any]]:
    """Replace older tool messages with one-line summaries, keeping N verbatim.

    **The unit is tool MESSAGES, not ReAct rounds.** One round can dispatch
    several tools concurrently, so a 3-round worker running 4 searches plus 2
    failed fetches holds 6+ tool messages. Setting this equal to `max_rounds`
    therefore does NOT protect a round's results — a mistake made here on
    2026-07-21, when `deep_worker` ran `keep_last=3` against 7 tool messages and
    a worker lost all four `web_search` results while retaining two failures,
    then reported that its searches "returned empty results".

    Eviction order: **failures go before successes**, then oldest-first. Without
    that tier, two failed `web_fetch` calls landing at the end of a round
    displace the four good `web_search` results the worker needs in order to
    write — the newest messages are not the most valuable ones.

    `keep_last_round <= 0` is a no-op (see the config note in `config.yaml`); to
    wipe tool history aggressively, lower `compress_after_round` instead.
    """
    out: list[dict[str, Any]] = []
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if keep_last_round <= 0 or len(tool_indices) <= keep_last_round:
        return list(messages)
    # Successes (False) sort before errors (True); within a tier, `-i` puts the
    # most recent first. Keep the top `keep_last_round`, stub everything else.
    ranked = sorted(tool_indices, key=lambda i: (_is_error_tool_message(messages[i]), -i))
    keep = set(ranked[:keep_last_round])
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i not in keep:
            out.append({"role": "system", "content": _summarize_tool_message(m)})
        else:
            out.append(m)
    return out


# ─── The catalogue-only guard ─────────────────────────────────────────
#
# `list_my_files` returns filenames and WHICH artifacts exist — never a word of
# what any of them says (`tools-server/app.py:718`, and the `summary` field was
# deleted from that row on 2026-08-06 to keep it that way). So a turn that
# stops with `list_my_files` as its only successful call and then describes
# what a file CONTAINS is describing something it was never shown.
#
# ⚠️ **This is the fourth lever, and it is in code because the first three were
# words.** The tool description says "a catalogue, not contents" in bold;
# `VIDEO_SPECIALIST_SYSTEM` says "never answer about a file's contents from its
# name alone"; the row field was renamed `artifacts` → `unread_artifacts`. All
# three deployed. The third moved fabrication from ~71% of these turns to ~17%
# and the residue is not reachable by a fourth, because on 2026-08-12 two turns
# of the same case, holding **the same 3,762 bytes of context**, produced one
# invented championship result and one correct "I have not read it yet, shall
# I?". Identical input, opposite output: the divergence is sampling. Every text
# lever changes bytes that were already right.
#
# So the guard does not ask the model again — it fetches the file and lets the
# model answer with the content actually in front of it.
#
# **Each clause below earns its place against a real group**, measured over
# 4,358 archived answers (109 catalogue-only, guard fires on 35):
#
#   - unread artifacts must be non-empty → 44 `silent.mp4` turns skipped, where
#     the honest answer IS "no transcript" and there is nothing to fetch.
#   - exactly ONE named file → 18 "I found two London videos, which did you
#     mean?" turns skipped. Those are correct, and fetching would push a model
#     that rightly asked into guessing.
#   - naming is an EXACT match against filenames we just returned, not a phrase
#     family — which is why this can gate where the frozen prose checks cannot.
#
# Of the 35 it does fire on, 34 are the defect case and 31 of those are
# confirmed fabrications. ⚠️ The archive is all video cases, so non-video
# turns are unmeasured — though a turn where `list_my_files` is the ONLY
# success is close to video-specific by construction.
_CATALOGUE_TOOL = "list_my_files"
#: Summary first: the turns that trip this guard are overwhelmingly "what is
#: this file about", and a summary answers that in one page where a transcript
#: needs several. `visual` last — it is on-screen text, a poor substitute for
#: either.
_ARTIFACT_PREFERENCE = ("summary", "transcript", "visual")


def _catalogue_rows(results: list[ToolResult]) -> list[dict[str, Any]]:
    """Rows from the most recent successful `list_my_files`, `[]` if unreadable.

    ⚠️ Returns `[]` rather than raising when the body will not parse. A listing
    long enough to be cut at `max_tool_result_chars` arrives as invalid JSON,
    and a guard that cannot read the catalogue must let the answer stand — the
    alternative is fetching a file chosen from half a row.
    """
    for r in reversed(results):
        if r.name != _CATALOGUE_TOOL or r.is_error:
            continue
        try:
            body = json.loads(r.content)
        except (json.JSONDecodeError, TypeError, ValueError):
            log.info("catalogue-guard: %s result did not parse; standing down",
                     _CATALOGUE_TOOL)
            return []
        rows = body.get("files") if isinstance(body, dict) else body
        return [x for x in rows if isinstance(x, dict)] if isinstance(rows, list) else []
    return []


def _unread_fetch(answer: str, results: list[ToolResult]) -> tuple[str, str] | None:
    """`(filename, artifact)` the answer needed and did not read, else None."""
    succeeded = {r.name for r in results if not r.is_error}
    if succeeded != {_CATALOGUE_TOOL}:
        return None
    low = (answer or "").lower()
    named = [
        row for row in _catalogue_rows(results)
        if isinstance(row.get("filename"), str)
        and row["filename"]
        and row["filename"].lower() in low
    ]
    if len(named) != 1:
        return None
    unread = [a for a in named[0].get("unread_artifacts") or [] if isinstance(a, str)]
    if not unread:
        return None
    best = next((a for a in _ARTIFACT_PREFERENCE if a in unread), unread[0])
    return named[0]["filename"], best


def _guard_enabled(cfg: Any) -> bool:
    try:
        return bool(cfg.raw.get("agentic", {}).get("react", {})
                    .get("catalogue_guard", True))
    except AttributeError:
        return True


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
    #: Forwarded to every chat call in this loop. `None` omits the field, which
    #: is the pre-2026-08-07 behaviour and the only universally safe request.
    #:
    #: ⚠️ **This loop is NOT fast-path-only.** `deep_panel.py` drives its
    #: workers through it (`:210`) and its factchecker too (`:1502`), so this
    #: default is the only thing keeping `think=False` out of two roles where
    #: it would be wrong: panel workers are **reasoning-is-the-product**, and
    #: the factchecker runs with `format=`, where thinking has broken JSON
    #: before (`ledger.py:203-207`). Only `fast_path` passes a value.
    #:
    #: **Do not add one to the deep-panel calls without probing those roles.**
    #: The measurement behind the fast path's setting was `TOOLS=1` on a
    #: retrieval question, and it says nothing about either of them.
    think: bool | None = None,
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
    web_search_chars = 0  # chars of successful web_search bodies reaching context
    guard_used = False    # the catalogue guard fires at most once per turn

    async with httpx.AsyncClient() as http:
        for round_idx in range(max_rounds):
            if round_idx >= compress_after_round:
                convo = _compress_history(convo, keep_last_round=compress_keep_last)

            # What the model is about to see, AFTER compaction — see
            # `_context_census`. Logged every round so an answer that reports
            # missing content can be checked against the context it was given
            # instead of guessed at from the prose.
            if _debug_context_trace(cfg):
                log.info("context-trace: model=%s round=%d %s",
                         model, round_idx, _context_census(convo))

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
                        tools=tools, timeout_s=timeout_s, think=think,
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
                # The model wants to stop. Before letting it, check whether it
                # is about to describe a file it only ever saw the NAME of —
                # see `_unread_fetch`. Fires on ~0.8% of turns.
                fetch = (
                    _unread_fetch(msg.get("content", "") or "", all_results)
                    if not guard_used and _guard_enabled(cfg) else None
                )
                if fetch:
                    guard_used = True   # once per turn: this cannot re-loop
                    filename, artifact = fetch
                    log.info("catalogue-guard: %s answered from the catalogue "
                             "alone about %r; fetching %s", model, filename, artifact)
                    forced = await dispatch_one(
                        http, registry,
                        {"function": {"name": "get_file_text", "arguments": {
                            "filename": filename, "artifact": artifact,
                        }}},
                        max_result_chars=max_tool_result_chars,
                        timeout_s=tool_dispatch_timeout_s,
                        user_id=user_id,
                    )
                    if not forced.is_error:
                        # ⚠️ A `role=tool` message must follow an assistant turn
                        # carrying the matching `tool_calls`, and the model made
                        # no such call — writing one would put words in its
                        # mouth. The content goes in as a user turn instead,
                        # labelled for what it is.
                        convo.append({"role": "user", "content": (
                            f"System note: you have not read {filename} yet — "
                            f"`list_my_files` lists only which artifacts exist. "
                            f"Here is its {artifact}. Answer from this text, and "
                            f"if it does not cover the question, say so.\n\n"
                            f"{forced.content}"
                        )})
                        # Recorded so the tools footer stays TRUE: this content
                        # really was dispatched and really is in context.
                        all_results.append(forced)
                        rounds_used += 1
                        continue
                    log.warning("catalogue-guard: forced get_file_text failed (%s); "
                                "letting the answer stand", forced.content[:200])

                # ⚠️ THE line that matters, and it is here rather than after the
                # loop. This `return` is the NORMAL exit — the model stopped
                # calling tools and wrote its answer — so it is the path almost
                # every turn takes. The `FINAL` census below only fires when
                # `max_rounds` is exhausted with the model still asking for
                # tools, which is rare; grepping for FINAL alone therefore
                # comes back empty on a perfectly instrumented run (2026-08-12).
                #
                # `convo` here is exactly what produced the answer: compaction
                # ran at the top of this iteration and nothing has been appended
                # since, so this census is the context the prose was written
                # from.
                if _debug_context_trace(cfg):
                    log.info("context-trace: model=%s ANSWERED round=%d %s",
                             model, round_idx, _context_census(convo))
                return ReactResult(
                    content=msg.get("content", "") or "",
                    tool_rounds=rounds_used,
                    tool_calls=all_results,
                    prompt_eval_count=int(last_resp.get("prompt_eval_count", 0) or 0),
                    eval_count=int(last_resp.get("eval_count", 0) or 0),
                    web_search_chars=web_search_chars,
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
                if r.name == "web_search" and not r.is_error:
                    web_search_chars += len(r.content or "")
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
        # ⚠️ THE line that matters. This is the context the final answer is
        # actually written from — after a SECOND compaction pass — and it is
        # the one the paging cases have been failing out of. A turn that made
        # three `get_file_text` calls arrives here holding one of them.
        if _debug_context_trace(cfg):
            log.info("context-trace: model=%s FINAL %s",
                     model, _context_census(convo))
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
                    tools=None, timeout_s=timeout_s, think=think,
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
            web_search_chars=web_search_chars,
        )


__all__ = ["run_react", "ReactResult"]
