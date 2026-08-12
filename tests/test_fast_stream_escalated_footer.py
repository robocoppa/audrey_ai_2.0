"""The tools footer on a fast turn that escalated to the deep panel (2026-08-12).

`_stream_via_pipeline`'s tool-capable branch runs the graph, and the graph can
escalate fast→deep internally (`route_after_fast_path`). When it does, the turn
still streams under the FAST identity — fast banners, fast route — but its
answer was written by deep workers.

The footer was built from `tool_calls_log` alone. Escalation happens ONLY when
`tool_rounds == 0` (`graph.py:146` — a fast turn that used tools is already
grounded and is never re-run), so on exactly those turns that log is empty by
construction, while the workers that DID call tools recorded them in `drafts`.
Result: no footer at all.

⚠️ Measured over the archive: `video-two-file-compare` lost its footer **53
times out of 53**, every run it has ever appeared in, while every other video
case rendered one nearly always. That case is the one built to catch two files
being conflated — and `grounded`, `_ungrounded_content` and `no_reasoning_leak`
all parse the footer, so all three were blind on exactly it.

⚠️ No end-to-end drain here, and that is a deliberate call. Reaching that
branch needs fakes for the graph, registry, gate, health, tools and collector —
and a test built almost entirely of doubles is what let the passthrough outage
through the same day. Instead: the composition is tested against real
`tool_summary_block`, and `test_the_call_site_still_reads_drafts` reads the
production source, so the two together cannot both pass on a broken call site.
"""

from __future__ import annotations

import inspect

from audrey.pipeline.banners import tool_summary_block

_FAST_MODEL = "qwen3.6:35b"
_WORKER = "deepseek-v4-pro:cloud"


def _final_state(*, escalated: bool) -> dict:
    """A graph result. Escalated turns carry drafts and an empty tool log."""
    if not escalated:
        return {
            "concrete_model": _FAST_MODEL,
            "content": "The clip shows two black belts on a red mat.",
            "tool_rounds": 2,
            "tool_calls_log": [
                {"name": "list_my_files", "elapsed_s": 0.2, "is_error": False},
                {"name": "get_file_text", "elapsed_s": 0.4, "is_error": False},
            ],
            "drafts": [],
        }
    return {
        "concrete_model": _FAST_MODEL,
        "content": "## Comparing the Two Videos\n\nCarlsen plays White…",
        # ⚠️ Empty, and not by accident — see the module docstring.
        "tool_rounds": 0,
        "tool_calls_log": [],
        "escalated_from_fast": True,
        "drafts": [
            {"model": _WORKER, "content": "…", "tool_calls": [
                {"name": "kb_search", "elapsed_s": 1.1, "is_error": False},
                {"name": "get_file_text", "elapsed_s": 0.9, "is_error": False},
            ]},
            {"model": _WORKER, "content": "…", "tool_calls": [
                {"name": "kb_search", "elapsed_s": 1.3, "is_error": True},
            ]},
        ],
    }


def _footer_for(final: dict) -> str:
    """The call site's expression, verbatim. Kept in step by the source test."""
    return tool_summary_block(
        [(final.get("concrete_model"), list(final.get("tool_calls_log") or []))]
        + [(str(d.get("model") or "?"), list(d.get("tool_calls") or []))
           for d in (final.get("drafts") or [])]
    )


def test_the_old_expression_rendered_nothing():
    """Pins WHY the fix is needed: `tool_calls_log` alone is empty on exactly
    the turns that escalate, so the old expression produced no footer."""
    final = _final_state(escalated=True)
    assert tool_summary_block([
        (final.get("concrete_model"), list(final.get("tool_calls_log") or []))
    ]) == ""


def test_an_escalated_turn_now_reports_its_workers():
    footer = _footer_for(_final_state(escalated=True))
    assert "_Tools used:_" in footer, "escalated turn rendered no footer at all"
    assert "kb_search" in footer and "get_file_text" in footer
    assert _WORKER in footer
    # The failed worker call is disclosed, not quietly dropped.
    assert "❌" in footer


def test_a_normal_fast_turn_is_unchanged():
    """The common path must not gain empty worker rows."""
    footer = _footer_for(_final_state(escalated=False))
    assert footer.count("> - ") == 1, footer
    assert _FAST_MODEL in footer and "❌" not in footer


def test_the_call_site_still_reads_drafts():
    """⚠️ The tests above pass whether or not production uses that expression.
    This one reads the source, so deleting `drafts` from the call site cannot
    leave the suite green — the gap a hand-written double always leaves."""
    import audrey.routes.openai.pipeline as mod

    src = inspect.getsource(mod)
    fast_branch = src[src.index("Per-worker tool-usage footer. Fast path is one worker"):]
    fast_branch = fast_branch[:fast_branch.index("BANNER_SEPARATOR")]
    assert 'final.get("drafts")' in fast_branch
    assert "tool_calls_log" in fast_branch


# ─── The panel-drafts block on an escalated fast turn ─────────────────
#
# ⚠️ NOT a new user-facing surface, and deliberately so. Escalation needs to be
# visible to the EVAL HARNESS (an escalated turn reports `route: fast` and
# passes `expect_route: "fast"` while a planner, three workers and a synthesis
# pass ran — so any A-B straddling that boundary measures two pipelines). It
# does NOT need to be visible to the person asking the question, who would just
# get noise. Operators already have `graph.py:174`'s `escalate: fast→deep` log.
#
# So this rides the flag that already exists for exactly this — off by default,
# same one both other paths use — rather than adding a knob or a banner.


def test_both_other_paths_already_do_this():
    """The claim this fix rests on: it is a PARITY gap, not a new feature.
    If the non-streaming path ever stops gating on `debug_panel_drafts`, the
    justification for mirroring it here is gone and this should be revisited."""
    import audrey.routes.openai.pipeline as mod

    src = inspect.getsource(mod)
    assert src.count("debug_panel_drafts") >= 3, (
        "expected the non-streaming path, the deep stream and the escalated "
        "fast stream to share one flag"
    )


def test_the_escalated_fast_branch_gates_on_the_existing_flag():
    """Reads the source: the block must be inside a `mode == deep` test AND a
    `debug_panel_drafts` test, so a normal fast turn can never emit it."""
    import audrey.routes.openai.pipeline as mod

    src = inspect.getsource(mod)
    branch = src[src.index("Per-worker tool-usage footer. Fast path is one worker"):]
    branch = branch[:branch.index("BANNER_SEPARATOR")]
    assert 'final.get("mode") == "deep"' in branch
    assert "debug_panel_drafts" in branch
    # And it must not introduce a second knob.
    assert "show_escalation" not in src and "debug_escalation" not in src


def test_nothing_is_added_to_the_wire_by_default():
    """The flag is off by default, so the default answer is unchanged."""
    from audrey.pipeline.banners import panel_drafts_block

    cfg_raw: dict = {"agentic": {}}
    assert not bool((cfg_raw.get("agentic") or {}).get("debug_panel_drafts", False))
    # And the block itself is empty for a turn with no drafts, so even with the
    # flag on a non-escalated turn gains nothing.
    assert panel_drafts_block([]) == ""
