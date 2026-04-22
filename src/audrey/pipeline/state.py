"""Pipeline state — shared across all LangGraph nodes.

One `PipelineState` per request. Nodes add keys as they run; later nodes
read what earlier ones wrote. Keeping it flat (no nested dicts) so
LangGraph's default reducer (replace-on-write) is fine.
"""

from __future__ import annotations

from typing import Literal, TypedDict

TaskType = Literal["code", "reasoning", "general", "vl"]
PipelineMode = Literal["fast", "deep"]


class PipelineState(TypedDict, total=False):
    # Input — set at request time
    virtual_model: str               # audrey_deep | audrey_cloud | audrey_local
    messages: list[dict]             # OpenAI-shaped chat messages
    temperature: float | None
    top_p: float | None
    max_tokens: int | None

    # Classification
    task_type: TaskType              # chosen task family
    classify_reason: str             # "keyword:code_strong", "router:general", "fallback:general", ...
    classify_confidence: float       # 0.0–1.0 where known

    # Complexity gate
    prompt_tokens: int
    complex: bool                    # True → force deep panel (once deep panel exists)

    # Routing decision
    mode: PipelineMode               # "fast" or "deep"
    concrete_model: str              # the Ollama model that was actually hit

    # Output
    content: str                     # final assistant text
    prompt_eval_count: int
    eval_count: int
    error: str                       # non-empty means a failure; main route turns it into 5xx
