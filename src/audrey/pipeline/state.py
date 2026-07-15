"""Pipeline state — shared across all LangGraph nodes.

One `PipelineState` per request. Nodes add keys as they run; later nodes
read what earlier ones wrote. Keeping it flat (no nested dicts) so
LangGraph's default reducer (replace-on-write) is fine.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

TaskType = Literal["code", "reasoning", "general", "vl"]
PipelineMode = Literal["fast", "deep"]


class WorkerDraft(TypedDict, total=False):
    model: str
    content: str
    error: str
    elapsed_s: float
    prompt_eval_count: int
    eval_count: int
    tool_rounds: int                 # ReAct rounds that invoked tools (0 if tool-free)
    tool_calls: list[dict]           # per-call summary: name, elapsed_s, is_error
    web_search_chars: int            # chars of successful web_search bodies that reached the model (research-trace diagnostic)


class PipelineState(TypedDict, total=False):
    # Input — set at request time
    virtual_model: str               # audrey_deep | audrey_cloud | audrey_local | audrey_auto | audrey_fast
    messages: list[dict]             # OpenAI-shaped chat messages
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    user_id: str                     # OpenAI-spec `user` field; "" if unset. Required to enable memory.

    # Memory (per-user, keyword-match via custom-tools /memory_search)
    memory_hits: list[dict]          # recalled entries shown to the model as a system hint

    # Classification
    task_type: TaskType              # chosen task family
    classify_reason: str             # "keyword:code_strong", "router:general", "fallback:general", ...
    classify_confidence: float       # 0.0–1.0 where known

    # Complexity gate
    prompt_tokens: int
    complex: bool                    # True → force deep panel

    # Routing decision
    mode: PipelineMode               # "fast" or "deep"
    concrete_model: str              # the Ollama model that was actually hit (or synthesizer for deep)

    # Planner (deep path)
    subtasks: list[str]              # may be empty; full prompt is run as-is when so

    # Deep-panel results
    panel_pool: str                  # "deep_panel" | "deep_panel_cloud" | "deep_panel_local" | "deep_panel_research"
    workers_attempted: list[str]     # model names dispatched
    drafts: list[WorkerDraft]        # one per worker (success or error)
    synthesizer_model: str           # which synth was used (or the writer, for research mode)
    synth_error: str                 # non-empty if synthesis failed (then fallback synth tried)

    # Research mode (audrey_research) — staged pipeline intermediate outputs.
    # `drafts` above holds the Stage-1 researcher drafts (so reflect/archive
    # see the grounding); the rest carry the later stage products. The last
    # three exist for the opt-in research trace (agentic.debug_research_trace).
    research_findings: str           # merged researcher notes fed to verify+write
    research_critique: str           # verifier's flags ("" if verify skipped/empty)
    research_factcheck: str          # fact-checker's corrections ("" if stage skipped)
    research_ledger: dict | None     # merged ResearchResult dump (None if no ledger)
    research_factcheck_ledger: dict | None  # FactCheckResult dump (None if unstructured)
    research_dispositions: str       # hedge-guidance block as handed to the writer

    # Reflection
    reflect_attempts: int            # 0 = not attempted, 1 = ran once, etc.
    reflect_passed: bool             # final answer cleared the quality gate
    reflect_reason: str              # "ok" | "too_short" | "low_confidence" | "no_drafts"

    # Escalation flag — fast_path → deep when fast answer was inadequate
    escalated_from_fast: bool

    # ReAct (tool use on the fast path)
    tool_rounds: int                 # number of rounds that invoked a tool
    tool_calls_log: list[dict]       # per-call summary: name, elapsed_s, is_error

    # Output
    content: str                     # final assistant text
    prompt_eval_count: int
    eval_count: int
    error: str                       # non-empty means a failure; main route turns it into 5xx

    # Free-form metadata for log/debug — not used by reducers
    meta: dict[str, Any]
