"""Reflection — quality gate on the synthesized answer.

Checks:
  - non-empty (synth didn't bail with `no_drafts`)
  - meets `agentic.reflection.min_answer_chars` (default 80)

If the gate fails, the graph runs the deep panel + synth one more time
(max_retries=1). On the retry we add a system note nudging the synthesizer
to be more substantive. If the second pass still fails, we ship what we
have rather than 502 — the answer is tagged with `reflect_passed=False`
in state for log inspection.

Phase 25 dropped the "must contain all three of `## Approach` / `## Answer`
/ `## Caveats`" requirement because the synth prompt no longer asks for
that fixed structure. Length-and-presence is the sole structural check now.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ReflectionResult:
    passed: bool
    reason: str  # "ok" | "too_short" | "no_drafts"


def reflect(
    *,
    content: str,
    synth_error: str,
    min_chars: int,
) -> ReflectionResult:
    """Cheap, deterministic quality check. No LLM calls."""
    if synth_error == "no_drafts":
        return ReflectionResult(False, "no_drafts")

    text = (content or "").strip()
    if len(text) < min_chars:
        return ReflectionResult(False, "too_short")

    return ReflectionResult(True, "ok")


__all__ = ["reflect", "ReflectionResult"]
