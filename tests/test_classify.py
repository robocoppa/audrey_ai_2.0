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


# ─── _CODE_WEAK signal coverage ────────────────────────────────────────


def test_code_weak_matches_each_language_term():
    """Every language and build-tool keyword in `_CODE_WEAK` should
    classify a plain mention as `code` (weak strength). Guards against
    a future regex edit accidentally dropping one."""
    expected_terms = [
        "python", "javascript", "typescript", "rust", "golang", "kotlin",
        "dockerfile", "npm", "pip install", "uv add",
        "cargo", "gradle", "make", "cmake",
        "pytest", "unittest", "mypy", "ruff", "eslint",
    ]
    for term in expected_terms:
        sig = keyword_classify(f"can you help me with {term}?")
        assert sig is not None, f"{term!r} should match _CODE_WEAK"
        assert sig.task == "code", f"{term!r} matched but task is {sig.task!r}"
        assert sig.strength == "weak", f"{term!r} matched but strength is {sig.strength!r}"


def test_code_weak_alternation_has_no_duplicates():
    """The `_CODE_WEAK` regex string must not list the same keyword
    twice. Duplicates are harmless at runtime but cost regex compile
    time and pollute the pattern."""
    from audrey.pipeline.classify import _CODE_WEAK

    # Extract bare alternation terms (drop `\b` boundaries and the
    # outer non-capturing group). The pattern is fixed-shape so a
    # simple split is fine.
    pattern = _CODE_WEAK.pattern
    inner = pattern.removeprefix(r"\b(").removesuffix(r")\b")
    terms = [t.strip() for t in inner.split("|") if t.strip()]
    seen: dict[str, int] = {}
    for t in terms:
        seen[t] = seen.get(t, 0) + 1
    dupes = {term: count for term, count in seen.items() if count > 1}
    assert not dupes, f"duplicate _CODE_WEAK terms: {dupes}"


# ─── classify_with_registry — the streaming/graph shared call site ────


class _FakeRegistry:
    """Minimal stand-in for ToolRegistry — only `.names()` is used."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def names(self) -> list[str]:
        return list(self._names)


class _FakeOllama:
    """Records router invocations so a test can assert when one happens."""

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []

    async def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        # Default router response. Tests that exercise the router can
        # override this by passing their own ollama stub.
        return {"message": {"content": '{"task": "general", "confidence": 0.7}'}}


_ROUTER_CFG = {
    "model": "qwen3:4b",
    "timeout_s": 5,
    "max_failures_before_fallback": 1,
}


async def test_classify_with_registry_passes_tool_names_to_keyword_classify():
    """A prompt naming a registered tool must short-circuit to `general`
    via `_tool_mention_signal`, no router call. This is the regression
    test for the streaming-path bug: the streaming `classify_fn` call
    forgot `tool_names`, so the word "image" inside "use kb_image_search"
    would trip `_VL_STRONG` and route to a tool-blind VL model. The
    shared helper now extracts `tool_names` from the registry for both
    call sites."""
    from audrey.pipeline.classify import classify_with_registry

    ollama = _FakeOllama()
    registry = _FakeRegistry(["kb_search", "kb_image_search", "web_search"])
    task, reason, conf = await classify_with_registry(
        ollama,
        user_text="use kb_image_search to find a rock with banding",
        router_cfg=_ROUTER_CFG,
        cfg=None,
        registry=registry,
    )
    assert task == "general"
    assert reason == "keyword:tool_mention:kb_image_search"
    assert conf == 0.95
    # Router must not have been called — keyword short-circuited.
    assert ollama.chat_calls == []


async def test_classify_with_registry_no_registry_means_no_tool_override():
    """When the registry is None, `tool_names` is empty and the classifier
    behaves exactly as if there were no registered tools. A bare "image"
    prompt then trips `_VL_STRONG` and routes to vl — the load-bearing
    scenario the override exists to prevent, demonstrated by removing the
    registry."""
    from audrey.pipeline.classify import classify_with_registry

    ollama = _FakeOllama()
    task, reason, conf = await classify_with_registry(
        ollama,
        user_text="identify the type of image in this attachment",
        router_cfg=_ROUTER_CFG,
        cfg=None,
        registry=None,
    )
    # No tool_names → `_VL_STRONG` matches "image" first.
    assert task == "vl"
    assert reason == "keyword:vl_strong"
    assert conf == 0.95
    assert ollama.chat_calls == []


async def test_classify_with_registry_falls_through_to_router_on_plain_prose():
    """Prompts that don't trip any keyword regex still reach the router.
    This is the "did I break the non-keyword path?" guard."""
    from audrey.pipeline.classify import classify_with_registry

    ollama = _FakeOllama()
    task, reason, _conf = await classify_with_registry(
        ollama,
        user_text="could you help me think through whether this plan is sensible?",
        router_cfg=_ROUTER_CFG,
        cfg=None,
        registry=_FakeRegistry(["web_search"]),
    )
    # Router stub returns task="general" — assert we got there.
    assert task == "general"
    assert reason == "router:general"
    assert len(ollama.chat_calls) == 1


# ─── skip_llm_under_tokens (Phase 16 banner-latency fix) ──────────────


async def test_short_no_keyword_prompt_skips_router_llm():
    """A short, keyword-free prompt under the token gate routes `general`
    WITHOUT a router LLM call. This is the latency fix: the fast-path
    Thinking banner no longer waits on the router model for chit-chat."""
    from audrey.pipeline.classify import classify

    ollama = _FakeOllama()
    task, reason, conf = await classify(
        ollama,
        router_model="qwen3:4b",
        router_timeout_s=5,
        max_router_strikes=1,
        user_text="hey there",
        skip_llm_under_tokens=8,
    )
    assert task == "general"
    assert reason == "short_skip:general"
    assert conf == 0.5
    assert ollama.chat_calls == []  # router never called


async def test_short_prompt_with_weak_keyword_skips_llm_but_keeps_signal():
    """A short prompt that tripped only a *weak* keyword (e.g. 'pip install')
    still skips the router, but the weak signal wins over the bare `general`
    default."""
    from audrey.pipeline.classify import classify

    ollama = _FakeOllama()
    task, reason, conf = await classify(
        ollama,
        router_model="qwen3:4b",
        router_timeout_s=5,
        max_router_strikes=1,
        user_text="pip install ruff",  # _CODE_WEAK, short
        skip_llm_under_tokens=8,
    )
    assert task == "code"
    assert reason.startswith("short_skip_keyword:")
    assert conf == 0.6
    assert ollama.chat_calls == []


async def test_long_prompt_still_calls_router_even_when_skip_enabled():
    """The skip only applies under the token gate. A longer keyword-free
    prompt still reaches the router — we didn't disable classification."""
    from audrey.pipeline.classify import classify

    ollama = _FakeOllama()
    task, reason, _conf = await classify(
        ollama,
        router_model="qwen3:4b",
        router_timeout_s=5,
        max_router_strikes=1,
        user_text=(
            "could you help me think carefully through whether this rather "
            "involved migration plan is actually sensible before I commit to it"
        ),
        skip_llm_under_tokens=8,
    )
    assert task == "general"
    assert reason == "router:general"
    assert len(ollama.chat_calls) == 1


async def test_skip_disabled_by_default_zero_threshold():
    """skip_llm_under_tokens=0 (the default) preserves the old behavior:
    even a one-word prompt reaches the router."""
    from audrey.pipeline.classify import classify

    ollama = _FakeOllama()
    _task, reason, _conf = await classify(
        ollama,
        router_model="qwen3:4b",
        router_timeout_s=5,
        max_router_strikes=1,
        user_text="hi",
        skip_llm_under_tokens=0,
    )
    assert reason == "router:general"
    assert len(ollama.chat_calls) == 1


async def test_strong_keyword_still_wins_over_skip():
    """A short prompt with a STRONG keyword keeps its keyword verdict — the
    skip gate only governs the weak/none case."""
    from audrey.pipeline.classify import classify

    ollama = _FakeOllama()
    task, reason, conf = await classify(
        ollama,
        router_model="qwen3:4b",
        router_timeout_s=5,
        max_router_strikes=1,
        user_text="review this code",  # _REVIEW_OVERRIDE → reasoning, strong
        skip_llm_under_tokens=8,
    )
    assert task == "reasoning"
    assert reason == "keyword:review_override"
    assert conf == 0.95
    assert ollama.chat_calls == []
