"""Tests for the per-worker tools-used footer (Phase 28).

Covers `audrey.pipeline.banners.tool_summary_block` and its inner
`_format_calls` helper. Both are pure functions over plain dicts —
no I/O, no async, no fixtures needed.

The footer is reachable across the deep streaming path
(`drafts[*].tool_calls`) and the tool-capable fast streaming path
(`final["tool_calls_log"]`); both call sites pass the same
`(model, calls)` tuple shape, so the formatter has one definition
to defend.

Also pins the banner header constants the streaming routes depend on
(Phase 7 added a Thinking banner on the fast path that reuses
`BANNER_THINKING` + `BANNER_SEPARATOR`). The constants are part of
the SSE protocol contract — a silent rename would not break any
test, but would change what users see in the chat UI.
"""

from audrey.pipeline.banners import (
    BANNER_DISPATCHING,
    BANNER_PLANNING,
    BANNER_SEPARATOR,
    BANNER_SYNTHESIZING,
    BANNER_THINKING,
    PhaseTicker,
    _format_calls,
    tool_summary_block,
    worker_ok,
)

# ─── _format_calls ─────────────────────────────────────────────────────

def test_format_calls_empty_returns_empty_string():
    assert _format_calls([]) == ""


def test_format_calls_single_call():
    # ✅ always carries the success count, even for one call.
    assert _format_calls([{"name": "kb_search", "is_error": False}]) == "`kb_search` ✅1"


def test_format_calls_collapses_repeats():
    calls = [
        {"name": "kb_search", "is_error": False},
        {"name": "kb_search", "is_error": False},
    ]
    assert _format_calls(calls) == "`kb_search` ✅2"


def test_format_calls_partial_failure_shows_both_counts():
    # One success and one error → `✅1 ❌1`. Each count is plain — no inferring
    # meaning from the presence or absence of a number.
    calls = [
        {"name": "kb_search", "is_error": False},
        {"name": "kb_search", "is_error": True},
    ]
    assert _format_calls(calls) == "`kb_search` ✅1 ❌1"


def test_format_calls_total_failure_shows_zero_successes():
    # Every call errored → `✅0 ❌3`. Total failure reads as plainly as partial;
    # there is no special bare-mark case to misread.
    calls = [
        {"name": "web_search", "is_error": True},
        {"name": "web_search", "is_error": True},
        {"name": "web_search", "is_error": True},
    ]
    assert _format_calls(calls) == "`web_search` ✅0 ❌3"


def test_format_calls_single_call_failure():
    # 1 failed call → `✅0 ❌1`, consistent with every other row.
    assert _format_calls([{"name": "kb_search", "is_error": True}]) == "`kb_search` ✅0 ❌1"


def test_format_calls_partial_failure_among_many():
    # The case that motivated this: many calls, a few errored. `✅10 ❌3`
    # surfaces "10 ok, 3 failed" — unambiguous where the old sticky boolean
    # was not.
    calls = [{"name": "web_search", "is_error": i < 3} for i in range(13)]
    assert _format_calls(calls) == "`web_search` ✅10 ❌3"


def test_format_calls_preserves_first_seen_order():
    # web_search appears first, even though kb_search is alphabetically before.
    calls = [
        {"name": "web_search", "is_error": False},
        {"name": "kb_search", "is_error": False},
        {"name": "web_search", "is_error": False},
    ]
    assert _format_calls(calls) == "`web_search` ✅2, `kb_search` ✅1"


def test_format_calls_handles_missing_name_field():
    # Defensive: ReAct dispatch records always have `name`, but if a future
    # bug drops it we render `?` instead of crashing. Regression guard.
    assert _format_calls([{"is_error": False}]) == "`?` ✅1"


def test_format_calls_handles_missing_is_error_field():
    # Same defensive shape — missing key is treated as "not an error".
    assert _format_calls([{"name": "kb_search"}]) == "`kb_search` ✅1"


# ─── tool_summary_block ────────────────────────────────────────────────

def test_tool_summary_block_empty_input_returns_empty_string():
    assert tool_summary_block([]) == ""


def test_tool_summary_block_all_workers_tool_free_returns_empty_string():
    # Workers ran but none called a tool → footer suppressed entirely.
    # This is the common case for general-knowledge prompts.
    per_worker = [
        ("qwen3.6:35b", []),
        ("llama4", []),
    ]
    assert tool_summary_block(per_worker) == ""


def test_tool_summary_block_single_worker_single_call():
    out = tool_summary_block([
        ("qwen3.6:35b", [{"name": "kb_search", "is_error": False}]),
    ])
    expected = (
        "\n"
        "\n"
        "---\n"
        "> _Tools used:_\n"
        "> - **qwen3.6:35b** — `kb_search` ✅1\n"
    )
    assert out == expected


def test_tool_summary_block_multi_worker_drops_empty_rows():
    # Worker 2 ran but called zero tools — must not get a row.
    out = tool_summary_block([
        ("qwen3.6:35b", [
            {"name": "kb_search", "is_error": False},
            {"name": "kb_search", "is_error": False},
        ]),
        ("llama4", []),
        ("deepseek-v4-pro:cloud", [
            {"name": "web_search", "is_error": False},
            {"name": "kb_search", "is_error": True},
        ]),
    ])
    # A failing row is present, so the header carries the decode legend.
    expected = (
        "\n"
        "\n"
        "---\n"
        "> _Tools used:_"
        "  _(✅ = calls succeeded, ❌ = calls failed)_\n"
        "> - **qwen3.6:35b** — `kb_search` ✅2\n"
        "> - **deepseek-v4-pro:cloud** — `web_search` ✅1, `kb_search` ✅0 ❌1\n"
    )
    assert out == expected


def test_tool_summary_block_no_legend_when_all_calls_succeed():
    # The all-green common case stays uncluttered: no failure mark anywhere
    # → no legend on the header.
    out = tool_summary_block([
        ("qwen3.6:35b", [{"name": "kb_search", "is_error": False}]),
    ])
    assert "> _Tools used:_\n" in out
    assert "❌" not in out
    assert "calls failed" not in out


def test_tool_summary_block_legend_appears_only_with_a_failure():
    # The legend is what makes the ✅/❌ counts decodable by a reader who has
    # never seen the convention — it must appear exactly when a failure does.
    out = tool_summary_block([
        ("m", [{"name": "web_search", "is_error": i < 3} for i in range(13)]),
    ])
    assert "_(✅ = calls succeeded, ❌ = calls failed)_" in out
    assert "`web_search` ✅10 ❌3" in out


def test_tool_summary_block_preserves_worker_order():
    # Workers render in the order given (matches dispatch banner completion
    # order). Don't sort by model name.
    out = tool_summary_block([
        ("zeta-model", [{"name": "kb_search", "is_error": False}]),
        ("alpha-model", [{"name": "kb_search", "is_error": False}]),
    ])
    # zeta-model row appears before alpha-model row.
    zeta_idx = out.index("**zeta-model**")
    alpha_idx = out.index("**alpha-model**")
    assert zeta_idx < alpha_idx


def test_tool_summary_block_starts_with_horizontal_rule_break():
    # OWUI's markdown renderer needs the leading blank lines + `---` to
    # treat the footer as a separate section below the answer body. If
    # this regresses, the footer renders as part of the prose.
    out = tool_summary_block([
        ("m", [{"name": "t", "is_error": False}]),
    ])
    assert out.startswith("\n\n---\n")


def test_tool_summary_block_ends_with_newline():
    # Trailing newline keeps SSE delta concatenation clean — without it,
    # `[DONE]` would land on the same rendered line as the last bullet.
    out = tool_summary_block([
        ("m", [{"name": "t", "is_error": False}]),
    ])
    assert out.endswith("\n")


# ─── Banner header constants ──────────────────────────────────────────
#
# Pin the user-visible strings so a silent rename can't ship to prod.
# The streaming route emits these verbatim into SSE frames; the chat
# UI renders them as markdown blockquotes. Any change here is a UX
# change and must be deliberate.

def test_banner_thinking_constant_shape():
    # Blockquote + italic + bare word. The italic underscores tell the
    # markdown renderer to style the line as a "system aside" rather
    # than plain text. Used by the fast streaming branch only.
    assert BANNER_THINKING == "> _Thinking_"


def test_banner_planning_constant_shape():
    # Deep streaming uses Planning (memory recall + planner) instead
    # of Thinking so users can tell at a glance which branch ran:
    # Thinking → fast, Planning → deep.
    assert BANNER_PLANNING == "> _Planning_"


def test_banner_dispatching_constant_shape():
    assert BANNER_DISPATCHING == "> _Dispatching panel_"


def test_banner_synthesizing_constant_shape():
    assert BANNER_SYNTHESIZING == "> _Synthesizing_"


def test_banner_separator_is_horizontal_rule_with_padding():
    # Two newlines on each side so the markdown renderer treats the
    # `---` as a horizontal rule, not a continuation of the blockquote
    # above. Phase 7 fast-path banner relies on this exact shape.
    assert BANNER_SEPARATOR == "\n\n---\n\n"


def test_fast_path_thinking_line_shows_model():
    # The plain fast path closes the Thinking banner with the concrete
    # model name, same `  ✅ <model>` fragment the deep panel uses per
    # worker — so a fast turn tells the user which model answered, not
    # just a bare checkmark. `> _Thinking_` + this fragment renders as
    # `> _Thinking_  ✅ qwen3-vl:32b`.
    assert BANNER_THINKING + worker_ok("qwen3-vl:32b") == "> _Thinking_  ✅ qwen3-vl:32b"


# ─── PhaseTicker emit_header (Phase 16 banner-latency fix) ────────────


async def test_phase_ticker_emits_header_by_default():
    """Default behavior unchanged: __aenter__ puts the header on the wire."""
    emitted: list[str] = []

    async def emit(text: str) -> None:
        emitted.append(text)

    async with PhaseTicker(BANNER_THINKING, emit, tick_interval_s=999):
        pass

    assert emitted[0] == BANNER_THINKING
    assert emitted[-1] == " ✅\n"


async def test_phase_ticker_skips_header_when_already_on_wire():
    """emit_header=False suppresses the opening header — the fast path
    already emitted `> _Thinking_` before classifying, so the ticker just
    dots the open line and closes it. The closing ✅ still fires."""
    emitted: list[str] = []

    async def emit(text: str) -> None:
        emitted.append(text)

    async with PhaseTicker(BANNER_THINKING, emit, tick_interval_s=999, emit_header=False):
        pass

    assert BANNER_THINKING not in emitted   # header NOT re-emitted
    assert emitted == [" ✅\n"]             # only the closing mark
