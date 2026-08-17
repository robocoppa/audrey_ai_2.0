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
    _failed_calls,
    _format_calls,
    panel_drafts_block,
    research_trace_block,
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
    # A failing row is present, so the block carries the plain-English
    # disclosure AND the header gains the decode legend.
    expected = (
        "\n"
        "\n"
        "---\n"
        "> ⚠️ **1 tool call failed** (`kb_search`) — this answer was written "
        "without what they would have returned, and may be incomplete.\n"
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
    assert "⚠️" not in out


# ─── The failure disclosure ────────────────────────────────────────────
#
# ⚠️ The ❌ notation was NOT enough, and the box proved it 2026-08-10:
# `kb_search` returned 500 for a whole turn while Qdrant restarted, and the
# model answered from `list_my_files` + four `get_file_text` pages, writing a
# confident section about a video it had never read. The only signal anywhere
# on screen was `❌1` in a footer row. This is the mechanical fix — the
# renderer already knows, so nothing depends on the model choosing to say it.


def test_a_failed_call_is_disclosed_in_words():
    out = tool_summary_block([
        ("qwen3.6:35b", [
            {"name": "list_my_files", "is_error": False},
            {"name": "kb_search", "is_error": True},
        ]),
    ])

    assert "**1 tool call failed** (`kb_search`)" in out
    assert "may be incomplete" in out
    # Above the counts, so it is read before the notation it explains.
    assert out.index("tool call failed") < out.index("_Tools used:_")


def test_the_disclosure_names_every_failed_tool_once():
    """Distinct names, first-seen order — a tool that failed three times is
    one name, not three, and the count carries the multiplicity."""
    out = tool_summary_block([
        ("w1", [{"name": "web_search", "is_error": True},
                {"name": "web_search", "is_error": True},
                {"name": "kb_search", "is_error": True}]),
    ])

    assert "**3 tool calls failed** (`web_search`, `kb_search`)" in out


def test_failures_are_counted_across_workers():
    """The deep panel renders one row per worker; a reader should not have to
    add up ❌ marks spread over three rows to learn something went wrong."""
    out = tool_summary_block([
        ("w1", [{"name": "kb_search", "is_error": True}]),
        ("w2", [{"name": "kb_search", "is_error": False}]),
        ("w3", [{"name": "kb_search", "is_error": True}]),
    ])

    assert "**2 tool calls failed**" in out


def test_an_all_green_turn_gets_no_disclosure():
    """The line must be invisible in the common case or it stops being read."""
    out = tool_summary_block([
        ("qwen3.6:35b", [{"name": "kb_search", "is_error": False},
                         {"name": "get_file_text", "is_error": False}]),
    ])

    assert "tool call" not in out
    assert "⚠️" not in out


def test_the_disclosure_reads_the_calls_not_the_rendered_marks():
    """⚠️ Counted off the raw dicts, never by scanning "❌" out of the rendered
    rows. A view and the record that disagree is how a footer ends up claiming
    a failure that did not happen, or missing one that did."""
    n, names = _failed_calls([("w1", [
        {"name": "kb_search", "is_error": True},
        {"name": "kb_search"},           # missing key == not an error
        {"name": "web_search", "is_error": False},
    ])])

    assert (n, names) == (1, ["kb_search"])


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


# ─── panel_drafts_block (debug/eval draft-vs-synth comparison) ─────────

def test_panel_drafts_block_empty_returns_empty_string():
    assert panel_drafts_block([]) == ""


def test_panel_drafts_block_renders_model_meta_and_content():
    out = panel_drafts_block([
        {"model": "qwen3.6:35b", "content": "The draft body.",
         "elapsed_s": 42.34, "tool_rounds": 2, "web_search_chars": 4210},
    ])
    assert "## Panel drafts (debug)" in out
    assert "### qwen3.6:35b — 42.3s · 2 tool rounds · web_search→ctx: 4210 chars" in out
    assert "The draft body." in out


def test_panel_drafts_block_web_search_chars_defaults_zero_for_tool_worker():
    # A tool-using worker that retrieved nothing (native-fetch failure / empty
    # web_search) renders `0 chars` — that's the research-grounding diagnostic
    # signal ("retrieved nothing" vs a large count), so it must show, not omit.
    out = panel_drafts_block([
        {"model": "m", "content": "x", "elapsed_s": 5.0, "tool_rounds": 1},
    ])
    assert "web_search→ctx: 0 chars" in out


def test_panel_drafts_block_omits_web_search_chars_for_tool_free_worker():
    # A tool-free worker never searched — no round count, and no web_search
    # line either (gated on tool_rounds so it can't read as "searched, got 0").
    out = panel_drafts_block([
        {"model": "m", "content": "x", "elapsed_s": 10.0, "tool_rounds": 0,
         "web_search_chars": 0},
    ])
    assert "web_search→ctx" not in out


# ─── Draft-shape diagnostics in the artifact ───────────────────────────
# A malformed draft used to reach the artifact with nothing to explain it: the
# synthesizer repairs it, every check passes, and the only trace is a draft
# that looks slightly odd. These fields are what makes the artifact answer
# "why", so they must appear when they have something to say — and stay out of
# the way when they do not, because a field printed on every draft is a field
# nobody reads.

def test_a_truncated_draft_says_so_in_its_heading():
    out = panel_drafts_block([
        {"model": "m", "content": "half an ans", "elapsed_s": 9.0,
         "done_reason": "length"},
    ])
    assert "done:length" in out


def test_an_ordinary_draft_heading_is_unchanged():
    # `done_reason == "stop"` is the overwhelmingly common case and carries no
    # information. Rendering it would bury the one that does.
    out = panel_drafts_block([
        {"model": "m", "content": "a complete answer", "elapsed_s": 9.0,
         "done_reason": "stop", "raw_content_len": 17},
    ])
    assert "done:" not in out
    assert "raw:" not in out


def test_a_heavily_stripped_draft_shows_what_it_lost():
    """A draft stripped to death and a genuinely terse one read identically.

    The body cannot distinguish them — only the gap between raw and content
    can, which is exactly the ambiguity that made an unfenced draft
    undiagnosable from the artifact.
    """
    out = panel_drafts_block([
        {"model": "m", "content": "tiny", "elapsed_s": 9.0,
         "raw_content_len": 4000},
    ])
    assert "raw:4000→4" in out


def test_trailing_whitespace_alone_does_not_trigger_the_raw_marker():
    out = panel_drafts_block([
        {"model": "m", "content": "body", "elapsed_s": 9.0,
         "raw_content_len": 4 + 8},
    ])
    assert "raw:" not in out


def test_a_split_panel_records_what_each_worker_was_actually_asked():
    """⚠️ `_messages_for_subtask` REPLACES the last user message.

    So "why did this worker answer that?" is unanswerable from the draft
    alone — the question in the artifact is the user's, and the question the
    worker saw is the planner's.
    """
    out = panel_drafts_block([
        {"model": "m", "content": "x", "elapsed_s": 1.0,
         "subtask": "Implement  the\n  eviction  policy"},
    ])
    assert "_asked: Implement the eviction policy_" in out


def test_an_unsplit_panel_adds_no_asked_line():
    out = panel_drafts_block([{"model": "m", "content": "x", "elapsed_s": 1.0}])
    assert "_asked:" not in out


def test_panel_drafts_block_omits_zero_tool_rounds():
    # A tool-free worker's heading carries latency only — "0 tool rounds"
    # would be noise on every non-agentic draft.
    out = panel_drafts_block([
        {"model": "m", "content": "x", "elapsed_s": 10.0, "tool_rounds": 0},
    ])
    assert "tool round" not in out
    assert "### m — 10.0s" in out


def test_panel_drafts_block_failed_worker_named_without_error_markers():
    # A failed worker still gets a subsection (naming who dropped is the
    # point), but its error text must never collide with the eval harness's
    # error markers — brackets are stripped.
    out = panel_drafts_block([
        {"model": "glm-5.2:cloud", "content": "", "error": "[ollama error: boom]"},
    ])
    assert "### glm-5.2:cloud" in out
    assert "_no usable draft — ollama error: boom_" in out
    assert "[ollama error" not in out
    assert "[internal error]" not in out


def test_panel_drafts_block_neutralizes_hr_lines_in_drafts():
    # The eval splits the answer body on the LAST "\n\n---\n\n" — a draft
    # carrying its own hr would truncate the saved answer. Neutralized.
    out = panel_drafts_block([
        {"model": "m", "content": "before\n\n---\n\nafter"},
    ])
    assert BANNER_SEPARATOR not in out
    assert "before" in out and "after" in out


def test_panel_drafts_block_never_contains_banner_separator():
    # Structural invariant: the block contains NO standalone `---` line at all
    # (heading-only opener + hr neutralization), so it can never form the
    # banner/answer separator regardless of how a consumer splits.
    out = panel_drafts_block([
        {"model": "a", "content": "one"},
        {"model": "b", "content": "", "error": "timeout"},
        {"model": "c", "content": "----\nindented hr\n\t---\t\nkept"},
    ])
    assert BANNER_SEPARATOR not in out
    assert out.startswith("\n\n## Panel drafts (debug)")
    # No bare horizontal-rule line anywhere in the rendered block.
    assert not any(ln.strip("- \t") == "" and set(ln.strip()) == {"-"}
                   for ln in out.splitlines())
    assert out.endswith("\n")


def test_panel_drafts_block_preserves_table_rows():
    # Markdown table delimiter rows (|---|---|) are not hr lines — a draft
    # with a table must keep it intact.
    out = panel_drafts_block([
        {"model": "m", "content": "| a | b |\n|---|---|\n| 1 | 2 |"},
    ])
    assert "|---|---|" in out


# ─── research_trace_block (debug/eval staged-pipeline trace) ───────────

def _trace_ledger() -> dict:
    """A merged-ledger dump in the shape the pipeline's done event carries."""
    return {
        "summary_notes": "",
        "claims": [
            {"id": "w0_c1", "text": "Euclid wrote the Elements.",
             "source_ids": ["w0_s1"], "risk": "low", "needs_hedge": False,
             "hedge_reason": None},
            {"id": "w1_c1", "text": "The bath anecdote\nis a late addition.",
             "source_ids": [], "risk": "high", "needs_hedge": True,
             "hedge_reason": "anecdote"},
        ],
        "sources": [
            {"id": "w0_s1", "title": "Euclid — Britannica",
             "url": "https://britannica.com/euclid",
             "source_type": "reference", "supports": ["w0_c1"]},
        ],
        "unresolved_questions": ["Exact dates are unknown"],
    }


def test_research_trace_block_empty_returns_empty_string():
    assert research_trace_block(drafts=[]) == ""


def test_research_trace_block_renders_all_stages_in_order():
    out = research_trace_block(
        drafts=[{"model": "r1", "content": "notes A",
                 "elapsed_s": 12.34, "tool_rounds": 4}],
        ledger=_trace_ledger(),
        factcheck={"checks": [
            {"claim_id": "w0_c1", "verdict": "supported",
             "corrected_text": None, "notes": ""},
            {"claim_id": "w1_c1", "verdict": "unsupported",
             "corrected_text": None, "notes": "no source confirms it"},
        ], "fatal_errors": []},
        critique="The anecdote needs a caveat.",
        corrections="DROP: the bath anecdote",
        dispositions="STATE PLAINLY: w0_c1",
    )
    assert out.startswith("\n\n## Research trace (debug)")
    assert "### Researcher notes" in out
    assert "#### r1 — 12.3s · 4 tool rounds" in out
    assert "notes A" in out
    assert "### Ledger — 2 claims, 1 sources" in out
    assert ("- **w0_c1** (risk: low) — Euclid wrote the Elements. "
            "_(sources: w0_s1)_") in out
    assert "needs hedge — anecdote" in out
    assert "https://britannica.com/euclid" in out
    assert "**Unresolved questions:**" in out
    assert "### Verifier critique" in out
    assert "The anecdote needs a caveat." in out
    assert "### Fact-check verdicts — 2 checks (1 drop, 0 hedge)" in out
    assert "- **w1_c1** — unsupported — no source confirms it" in out
    assert "### Corrections handed to the writer" in out
    assert "### Hedge dispositions handed to the writer" in out
    # Stage order is the pipeline order.
    assert (out.index("Researcher notes") < out.index("### Ledger")
            < out.index("Verifier critique") < out.index("Fact-check verdicts")
            < out.index("Corrections handed") < out.index("Hedge dispositions"))
    assert out.endswith("\n")


def test_research_trace_block_one_lines_claim_text():
    # Claim text is model output — a newline inside it would break the
    # markdown list item in two. Collapsed to one line.
    out = research_trace_block(drafts=[], ledger=_trace_ledger())
    assert "The bath anecdote is a late addition." in out


def test_research_trace_block_omits_empty_stages():
    # Only researcher notes available (ledger off / stages skipped) — the
    # other sections must not render as empty headings.
    out = research_trace_block(drafts=[{"model": "r1", "content": "notes"}])
    assert "### Researcher notes" in out
    assert "Ledger" not in out
    assert "Verifier critique" not in out
    assert "Fact-check" not in out
    assert "handed to the writer" not in out


def test_research_trace_block_never_contains_banner_separator():
    # Same structural invariant as the drafts block: heading-only opener and
    # no standalone hr line anywhere, whatever the model prose contained.
    out = research_trace_block(
        drafts=[{"model": "a", "content": "one\n\n---\n\ntwo"},
                {"model": "b", "content": "", "error": "[ollama error: boom]"}],
        ledger=_trace_ledger(),
        critique="before\n\n----\n\nafter",
        corrections="c\n\t---\t\nd",
        dispositions="e\n---\nf",
    )
    assert BANNER_SEPARATOR not in out
    assert not any(set(ln.strip()) == {"-"} for ln in out.splitlines() if ln.strip())
    # Failed-researcher error text can't fake the eval's error markers.
    assert "[ollama error" not in out
    assert "_no usable draft — ollama error: boom_" in out


def test_research_trace_block_contains_no_sources_heading():
    # The eval locates the real `## Sources` section by the substring
    # "## sources" — a heading like "### Sources" would contain it and hijack
    # the source-quality read. Source lists render as a bold label instead.
    out = research_trace_block(drafts=[{"model": "m", "content": "x"}],
                               ledger=_trace_ledger())
    assert "## sources" not in out.lower()
    assert "**Sources:**" in out


def test_research_trace_block_renders_factcheck_fatal_errors():
    out = research_trace_block(
        drafts=[],
        factcheck={"checks": [], "fatal_errors": ["[schema] parse failed"]},
    )
    assert "**Fatal errors:**" in out
    assert "schema parse failed" in out
    assert "[schema]" not in out


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


def test_a_malformed_draft_says_so_in_its_heading():
    """The gap the first cut of this diagnostic left open.

    The anomaly went to the LOG only, so an artifact could show four fenced
    drafts and one bare one with nothing marking which was which — and the
    artifact is what actually gets read.
    """
    out = panel_drafts_block([
        {"model": "m", "content": "from collections import OrderedDict\n",
         "elapsed_s": 74.6, "shape_anomaly": "unfenced_code"},
    ])
    assert "⚠ unfenced_code" in out


def test_a_clean_draft_carries_no_anomaly_marker():
    out = panel_drafts_block([
        {"model": "m", "content": "```python\nx = 1\n```", "elapsed_s": 1.0,
         "shape_anomaly": ""},
    ])
    assert "⚠" not in out
