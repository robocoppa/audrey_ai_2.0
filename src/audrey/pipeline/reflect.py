"""Reflection — quality gate on the synthesized answer.

Checks:
  - non-empty
  - meets `agentic.reflection.min_answer_chars` (default 80)
  - contains all three section headers (Approach / Answer / Caveats)
    — only enforced when synth succeeded; falls back to length when we
    degraded to a longest-draft answer.

If the gate fails, the graph runs the deep panel + synth one more time
(max_retries=1). On the retry we add a system note nudging the synthesizer
to be more substantive. If the second pass still fails, we ship what we
have rather than 502 — the answer is tagged with `reflect_passed=False`
in state for log inspection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_REQUIRED_HEADERS = ("## Approach", "## Answer", "## Caveats")


@dataclass(slots=True, frozen=True)
class ReflectionResult:
    passed: bool
    reason: str  # "ok" | "too_short" | "missing_sections" | "no_drafts"


def reflect(
    *,
    content: str,
    synth_error: str,
    min_chars: int,
    require_sections: bool,
) -> ReflectionResult:
    """Cheap, deterministic quality check. No LLM calls."""
    if synth_error == "no_drafts":
        return ReflectionResult(False, "no_drafts")

    text = (content or "").strip()
    if len(text) < min_chars:
        return ReflectionResult(False, "too_short")

    if require_sections and synth_error == "":
        # Only enforce structure when synthesis itself succeeded. The
        # longest-draft fallback won't have headers and isn't worth
        # re-running for that reason.
        if not all(h in text for h in _REQUIRED_HEADERS):
            return ReflectionResult(False, "missing_sections")

    return ReflectionResult(True, "ok")


__all__ = ["reflect", "ReflectionResult"]
