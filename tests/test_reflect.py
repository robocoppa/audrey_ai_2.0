"""Tests for reflect() — Phase 25 brevity-cue logic.

The brevity-cue skip exists because the post-Phase-25 synth prompt is
permissive about length, which broke `min_answer_chars=80`-style
floors on prompts like "in one sentence." Reflect was retrying the
whole panel (~10s wasted) for legitimately short answers.

Pure functions over strings — no fixtures.
"""

import pytest

from audrey.pipeline.reflect import _BREVITY_CUES, _user_wants_brevity, reflect

# ─── _user_wants_brevity ───────────────────────────────────────────────

@pytest.mark.parametrize("cue", _BREVITY_CUES)
def test_each_brevity_cue_matches(cue: str):
    # Every entry in the canonical _BREVITY_CUES tuple must trigger.
    # If anyone shortens the tuple by accident, the corresponding row
    # disappears here — making the change visible in test diffs.
    assert _user_wants_brevity(f"What year is it? {cue}.") is True


def test_brevity_cue_match_is_case_insensitive():
    # Mixed case in the user's prompt should still match the lowercase
    # cue. Reflect lowercases internally before comparing.
    assert _user_wants_brevity("Answer In One Sentence please") is True
    assert _user_wants_brevity("TL;DR what's the year") is True


def test_substantive_prompt_does_not_match():
    # A normal question with no brevity cue must NOT trigger — otherwise
    # short-but-broken synth answers would silently pass.
    assert _user_wants_brevity("Explain BTRFS copy-on-write semantics.") is False


def test_empty_or_whitespace_prompt_does_not_match():
    assert _user_wants_brevity("") is False
    assert _user_wants_brevity("   ") is False


def test_unrelated_short_prompt_does_not_match():
    # "What year is it?" alone — no brevity cue — should not trigger.
    # The whole point is that the user has to *say* they want brevity.
    assert _user_wants_brevity("What year is it?") is False


# ─── reflect() ─────────────────────────────────────────────────────────

def test_reflect_passes_long_substantive_answer():
    out = reflect(
        content="A" * 200,
        synth_error="",
        min_chars=40,
        user_text="Explain something at length.",
    )
    assert out.passed is True
    assert out.reason == "ok"


def test_reflect_fails_too_short_when_no_brevity_cue():
    out = reflect(
        content="2026.",
        synth_error="",
        min_chars=40,
        user_text="What year is it?",
    )
    assert out.passed is False
    assert out.reason == "too_short"


def test_reflect_passes_too_short_when_user_requested_brevity():
    # Phase 25 headline case. Without this skip, the panel would retry
    # ~10s of cloud time for a perfectly correct one-word answer.
    out = reflect(
        content="2026.",
        synth_error="",
        min_chars=40,
        user_text="What year is it? Answer in one sentence.",
    )
    assert out.passed is True
    assert out.reason == "ok_brevity_requested"


def test_reflect_distinguishes_ok_from_ok_brevity_requested():
    # Long answer + brevity cue: still "ok" (not "ok_brevity_requested").
    # The reason field is load-bearing for log filtering.
    out = reflect(
        content="A" * 200,
        synth_error="",
        min_chars=40,
        user_text="Be brief: explain BTRFS.",
    )
    assert out.reason == "ok"


def test_reflect_no_drafts_short_circuits_before_length_check():
    # `synth_error="no_drafts"` returns early — even a long content
    # string (impossible in practice, but defensive) should fail.
    out = reflect(
        content="A" * 200,
        synth_error="no_drafts",
        min_chars=40,
        user_text="anything",
    )
    assert out.passed is False
    assert out.reason == "no_drafts"


def test_reflect_strips_whitespace_before_length_check():
    # Synth output is sometimes padded with leading/trailing newlines;
    # those shouldn't count toward `min_chars`.
    out = reflect(
        content="   short   \n\n",
        synth_error="",
        min_chars=40,
        user_text="explain in detail",
    )
    assert out.passed is False
    assert out.reason == "too_short"


def test_reflect_handles_empty_user_text_safely():
    # Older callers (and the non-streaming path before the user_text
    # threading fix) pass "" — must not raise.
    out = reflect(
        content="A" * 200,
        synth_error="",
        min_chars=40,
        user_text="",
    )
    assert out.passed is True
    assert out.reason == "ok"
