"""`escalation_decision` — when a completed fast turn gets re-run through the panel.

Escalation exists to rescue an answer that came out THIN: too short, or from a
classification nobody was confident in. Every suppression rule is a case where
short-or-unconfident is the CORRECT answer, and getting one wrong is expensive
in a way that never shows up as an error — the user gets a slower, worse answer
and nothing logs a complaint.

That is how the `owui_task` gap survived: OWUI's chat-title turns were being
answered correctly in 32 characters and then escalated to a planner, three deep
workers and a synthesis pass, purely because 32 < the length floor. Observed on
the box 2026-08-10; the panel's own drafts came back at 38, 29 and 39 chars,
because there was nothing there to think about.

The policy was a closure inside `build_graph` with no tests at all. It is now a
pure function so each rule can be stated once, here.
"""

from __future__ import annotations

import pytest

from audrey.pipeline.graph import escalation_decision

_MIN_CHARS = 100
_CONF_CEILING = 0.95


def _decide(**state) -> str:
    return escalation_decision(
        state, enabled=state.pop("enabled", True),
        min_chars=_MIN_CHARS, conf_ceiling=_CONF_CEILING,
    )


# ─── The triggers ─────────────────────────────────────────────────────


def test_a_short_answer_escalates():
    assert _decide(content="Yes.", classify_confidence=1.0) == "escalate"


def test_a_low_confidence_classification_escalates():
    long_enough = "x" * (_MIN_CHARS + 1)
    assert _decide(content=long_enough, classify_confidence=0.25) == "escalate"


def test_a_long_confident_answer_is_left_alone():
    long_enough = "x" * (_MIN_CHARS + 1)
    assert _decide(content=long_enough, classify_confidence=1.0) == "end"


def test_confidence_of_zero_is_unknown_not_low():
    """0.0 means nothing recorded a confidence, not that it was terrible. The
    `conf > 0` guard is what keeps an unmeasured turn from escalating on a
    number that was never set."""
    long_enough = "x" * (_MIN_CHARS + 1)
    assert _decide(content=long_enough, classify_confidence=0.0) == "end"


# ─── The suppressions ─────────────────────────────────────────────────


def test_owui_utility_turns_never_escalate():
    """⚠️ The regression this file exists for. A chat title is SHORT — that is
    the whole job — so the length trigger fires on a correct answer."""
    assert _decide(content="Chess Openings Chat", classify_confidence=1.0,
                   owui_task=True) == "end"


def test_owui_utility_turns_survive_the_other_trigger_too():
    """These turns carry OWUI's template rather than a question, so the router
    has nothing to classify and reports a low confidence. Suppressing only the
    length trigger would have left this path open — and the box log shows both
    firing at once (`chars=32, conf=0.25`)."""
    long_enough = "x" * (_MIN_CHARS + 1)
    assert _decide(content=long_enough, classify_confidence=0.25,
                   owui_task=True) == "end"


def test_audrey_fast_never_escalates():
    assert _decide(content="Yes.", classify_confidence=1.0,
                   virtual_model="audrey_fast") == "end"


def test_an_already_escalated_turn_does_not_loop():
    assert _decide(content="Yes.", classify_confidence=1.0,
                   escalated_from_fast=True) == "end"


def test_a_tool_grounded_answer_is_not_second_guessed():
    """Deep workers here are tool-blind relative to what the fast path just
    did, so re-running a grounded answer can only lose the grounding."""
    assert _decide(content="Yes.", classify_confidence=1.0,
                   tool_rounds=2) == "end"


def test_a_memory_grounded_answer_is_not_second_guessed():
    """"You're running a Threadripper 7970X" is 93 chars and correct. Deep
    workers do not know the user and answer "I can't see your computer"."""
    assert _decide(content="Yes.", classify_confidence=1.0,
                   memory_hits=[{"key": "hardware", "value": "…"}]) == "end"


def test_the_feature_can_be_switched_off_entirely():
    assert _decide(content="Yes.", classify_confidence=0.1, enabled=False) == "end"


# ─── Interaction ──────────────────────────────────────────────────────


@pytest.mark.parametrize("suppressor", [
    {"owui_task": True},
    {"virtual_model": "audrey_fast"},
    {"escalated_from_fast": True},
    {"tool_rounds": 1},
    {"memory_hits": [{"key": "k"}]},
])
def test_every_suppressor_beats_both_triggers_at_once(suppressor):
    """The worst case is both triggers firing together, which is exactly what
    the box logged. A suppressor that only beat one of them would look correct
    in isolation and fail in production."""
    assert _decide(content="hi", classify_confidence=0.25, **suppressor) == "end"


def test_an_empty_state_does_not_crash():
    # The router reads everything with `.get`; a turn that never reached the
    # fast path should end, not raise.
    assert escalation_decision(
        {}, enabled=True, min_chars=_MIN_CHARS, conf_ceiling=_CONF_CEILING,
    ) == "escalate"  # no content == 0 chars == thin, which is the honest read
