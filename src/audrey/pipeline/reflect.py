"""Reflection — quality gate on the synthesized answer.

Checks:
  - non-empty (synth didn't bail with `no_drafts`)
  - meets `agentic.reflection.min_answer_chars` (default 40), UNLESS the
    user explicitly asked for a brief answer ("one sentence", "tldr",
    etc.) — see `_BREVITY_CUES` below.

If the gate fails, the graph runs the deep panel + synth one more time
(max_retries=1). On retry we add a system note nudging the synthesizer
to be more substantive. If the second pass still fails, we ship what we
have rather than 502 — the answer is tagged with `reflect_passed=False`
in state for log inspection.

Length-and-brevity is the sole structural check; we don't require any
fixed section structure in the answer.

The brevity-cue skip exists because the synth prompt is permissive about
length — without the cue check, a correct one-line answer to "what year
is it? answer in one sentence" would trigger a wasteful retry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


# Phrases in the user's prompt that signal "I want a short answer."
# Matched case-insensitively against the most recent user turn. Kept
# narrow on purpose — false positives just bypass a length check, but
# catching every phrasing isn't necessary; the goal is the obvious cases.
_BREVITY_CUES: tuple[str, ...] = (
    "in one sentence",
    "in a sentence",
    "in one word",
    "in a word",
    "one-sentence",
    "one-word",
    "tldr",
    "tl;dr",
    "tl dr",
    "briefly",
    "be brief",
    "short answer",
    "in short",
    "in brief",
    "in a few words",
    "concisely",
)


def _user_wants_brevity(user_text: str) -> bool:
    """True if the user's prompt contains an explicit short-answer cue."""
    text = (user_text or "").lower()
    if not text:
        return False
    return any(cue in text for cue in _BREVITY_CUES)


@dataclass(slots=True, frozen=True)
class ReflectionResult:
    passed: bool
    reason: str  # "ok" | "too_short" | "no_drafts" | "ok_brevity_requested"


def reflect(
    *,
    content: str,
    synth_error: str,
    min_chars: int,
    user_text: str = "",
) -> ReflectionResult:
    """Cheap, deterministic quality check. No LLM calls."""
    if synth_error == "no_drafts":
        return ReflectionResult(False, "no_drafts")

    text = (content or "").strip()
    if len(text) < min_chars:
        # Length floor failed — but if the user explicitly asked for a
        # brief answer, treat it as success and let the answer ship.
        if _user_wants_brevity(user_text):
            return ReflectionResult(True, "ok_brevity_requested")
        return ReflectionResult(False, "too_short")

    return ReflectionResult(True, "ok")


__all__ = ["reflect", "ReflectionResult"]
