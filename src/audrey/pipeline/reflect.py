"""Reflection — quality gate on the synthesized answer.

Checks:
  - non-empty (synth didn't bail with `no_drafts`)
  - meets `agentic.reflection.min_answer_chars` (default 40), UNLESS the
    user explicitly asked for a brief answer ("one sentence", "tldr",
    etc.) — see `_BREVITY_CUES` below.

If the gate fails, the graph runs the deep panel + synth one more time
(max_retries=1). On the retry we add a system note nudging the synthesizer
to be more substantive. If the second pass still fails, we ship what we
have rather than 502 — the answer is tagged with `reflect_passed=False`
in state for log inspection.

Phase 25 dropped the "must contain all three of `## Approach` / `## Answer`
/ `## Caveats`" requirement because the synth prompt no longer asks for
that fixed structure. Length-and-presence is the sole structural check now.

Phase 25 also added the brevity-cue skip after observing that the new
synth prompt (which is permissive about length) plus an
`min_answer_chars=80` floor caused false retries on questions like
"what year is it? answer in one sentence." The user explicitly asked
for a short answer; reflect was punishing the synth for complying.
"""

from __future__ import annotations

import logging
import re
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
