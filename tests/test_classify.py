"""Tests for _tool_mention_signal + keyword_classify ordering.

Pre-Phase-8 trap: a prompt like "use kb_image_search to find a rock"
matched `_VL_STRONG` (the word "image") and routed to a tool-blind VL
model. The fix added `_tool_mention_signal` as the first check inside
`keyword_classify`. These tests pin that ordering down so a future
refactor doesn't accidentally demote it below the keyword regexes.

Pure functions over strings.
"""

from audrey.pipeline.classify import (
    _tool_mention_signal,
    keyword_classify,
)

_REGISTERED_TOOLS = {
    "kb_search",
    "kb_image_search",
    "web_search",
    "memory_store",
    "memory_recall",
    "memory_search",
}


# ─── _tool_mention_signal ──────────────────────────────────────────────

def test_tool_mention_matches_each_registered_tool():
    for tool in _REGISTERED_TOOLS:
        sig = _tool_mention_signal(f"please use {tool} for this", _REGISTERED_TOOLS)
        assert sig is not None, f"{tool} should match"
        assert sig.task == "general"
        assert sig.strength == "strong"
        assert sig.reason == f"tool_mention:{tool}"


def test_tool_mention_is_case_insensitive():
    sig = _tool_mention_signal("Use KB_SEARCH to find docs", _REGISTERED_TOOLS)
    assert sig is not None
    assert sig.reason == "tool_mention:kb_search"


def test_tool_mention_requires_word_boundary():
    # The word "memory" alone should NOT match `memory_store` — boundary
    # guards prevent partial substring matches that would over-trigger.
    assert _tool_mention_signal("I have a great memory for faces", _REGISTERED_TOOLS) is None


def test_tool_mention_no_registered_tools_returns_none():
    # Empty registry — no tools to mention, so no signal regardless of text.
    assert _tool_mention_signal("use kb_search to find something", set()) is None


def test_tool_mention_unrelated_prompt_returns_none():
    assert _tool_mention_signal("Tell me about Iceland geology", _REGISTERED_TOOLS) is None


# ─── keyword_classify ordering ─────────────────────────────────────────

def test_tool_mention_overrides_vl_strong():
    # The Phase 8 regression. Without `_tool_mention_signal` running first,
    # the word "image" inside the prompt would match `_VL_STRONG` and
    # route to a tool-blind VL model.
    sig = keyword_classify(
        "use kb_image_search to find a rock image",
        tool_names=_REGISTERED_TOOLS,
    )
    assert sig is not None
    assert sig.task == "general"
    assert sig.reason == "tool_mention:kb_image_search"


def test_tool_mention_overrides_code_strong():
    # A fenced code block ordinarily routes to `code`. If the prompt
    # also names a tool, tool-dispatch wins.
    prompt = "```python\nprint('hi')\n```\nplease use web_search to find docs"
    sig = keyword_classify(prompt, tool_names=_REGISTERED_TOOLS)
    assert sig is not None
    assert sig.task == "general"
    assert sig.reason.startswith("tool_mention:")


def test_keyword_classify_falls_through_to_vl_when_no_tool_named():
    # No tool name in prompt → `_VL_STRONG` should still fire as before.
    # Regression guard: making sure `_tool_mention_signal` isn't
    # accidentally short-circuiting ALL prompts when tool_names is set.
    sig = keyword_classify(
        "What kind of rock is in this image?",
        tool_names=_REGISTERED_TOOLS,
    )
    assert sig is not None
    assert sig.task == "vl"
    assert sig.reason == "vl_strong"


def test_keyword_classify_with_no_tools_arg_works():
    # Caller doesn't pass tool_names at all — old default behavior.
    sig = keyword_classify("```js\nconst x = 1;\n```")
    assert sig is not None
    assert sig.task == "code"
    assert sig.reason == "code_strong"


def test_keyword_classify_review_override_beats_code():
    # "review this code" is reasoning, not code — even though "code" is
    # in the prompt and there might be a snippet attached.
    sig = keyword_classify(
        "review this code for bugs:\n\ndef foo(): pass",
        tool_names=_REGISTERED_TOOLS,
    )
    assert sig is not None
    assert sig.task == "reasoning"
    assert sig.reason == "review_override"


def test_keyword_classify_returns_none_for_plain_prose():
    # No regex hit, no tool mention → router decides.
    assert keyword_classify(
        "tell me about Iceland",
        tool_names=_REGISTERED_TOOLS,
    ) is None
