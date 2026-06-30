"""Tests for the ReAct helper bits (history compression).

The `run_react` loop itself is integration-shaped (network, model,
gate) and would need heavy mocking. The helpers it uses are pure
functions — that's what we pin here.

What this file covers:
  - `_compress_history(keep_last_round=N)` keeps N most-recent tool
    messages verbatim and stubs older ones. Pins behavior across
    multiple values of N so a future tuning change to the
    `agentic.react.compress_keep_last` knob is testable in code.
  - The stub names the tool and reads as a *compaction*, not a failed or
    empty search (a model that quotes it must not narrate a grounding failure).
  - Non-tool messages (user, assistant, system) are preserved as-is.
"""

from __future__ import annotations

from audrey.pipeline.react import _compress_history, _summarize_tool_message


def _tool_msg(name: str, content: str) -> dict:
    return {"role": "tool", "name": name, "content": content}


# ─── _compress_history ────────────────────────────────────────────────


def test_compress_history_keeps_only_last_n_tool_messages():
    convo = [
        {"role": "user", "content": "hi"},
        _tool_msg("web_search", "old1"),
        _tool_msg("kb_search", "old2"),
        _tool_msg("memory_search", "newest"),
        {"role": "assistant", "content": "thinking..."},
    ]
    out = _compress_history(convo, keep_last_round=1)
    # The two older tool messages become summary stubs (role=system).
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "system" and "web_search" in out[1]["content"]
    assert out[2]["role"] == "system" and "kb_search" in out[2]["content"]
    # The newest tool message is preserved verbatim.
    assert out[3] == {"role": "tool", "name": "memory_search", "content": "newest"}
    # Non-tool tail stays as-is.
    assert out[4] == {"role": "assistant", "content": "thinking..."}


def test_compress_history_keep_last_round_two():
    """When keep_last_round=2, the two most recent tool messages stay
    verbatim. Pins the config knob `compress_keep_last` to actually
    change behavior."""
    convo = [
        _tool_msg("web_search", "old"),
        _tool_msg("kb_search", "mid"),
        _tool_msg("memory_search", "newest"),
    ]
    out = _compress_history(convo, keep_last_round=2)
    # Old → stub; mid + newest → verbatim.
    assert out[0]["role"] == "system" and "web_search" in out[0]["content"]
    assert out[1] == {"role": "tool", "name": "kb_search", "content": "mid"}
    assert out[2] == {"role": "tool", "name": "memory_search", "content": "newest"}


def test_compress_history_keep_zero_is_a_no_op_today():
    """Regression-only test: `keep_last_round=0` is not a recommended
    value. With the current helper it accidentally no-ops because
    `tool_indices[-0]` evaluates to `tool_indices[0]` (the first
    index), so the "older than this" branch never triggers. Sensible
    values are ≥1; to wipe tool history aggressively, lower
    `compress_after_round` instead.

    This test exists as a tripwire — if someone refactors the helper
    later (e.g. switches to `tool_indices[len(...) - keep_last_round]`),
    the new semantics will show up as a failed assertion here, and the
    config doc note in `config.yaml` should be revisited."""
    convo = [
        _tool_msg("web_search", "a"),
        _tool_msg("kb_search", "b"),
    ]
    out = _compress_history(convo, keep_last_round=0)
    assert out == convo  # current behavior: no compression at 0


def test_compress_history_does_nothing_when_count_below_threshold():
    """When the number of tool messages is at or below `keep_last_round`,
    no compression happens — the convo is returned unchanged."""
    convo = [
        {"role": "user", "content": "hi"},
        _tool_msg("web_search", "a"),
    ]
    out = _compress_history(convo, keep_last_round=1)
    assert out == convo


def test_compress_history_preserves_message_order():
    convo = [
        {"role": "system", "content": "context"},
        _tool_msg("web_search", "old"),
        {"role": "assistant", "content": "after old"},
        _tool_msg("kb_search", "newest"),
    ]
    out = _compress_history(convo, keep_last_round=1)
    # Order preserved; only the old tool message is stubbed.
    assert [m["role"] for m in out] == ["system", "system", "assistant", "tool"]
    assert out[1]["content"].startswith("[history compacted:")


# ─── _summarize_tool_message ──────────────────────────────────────────


def test_summarize_tool_message_names_the_tool():
    msg = _tool_msg("kb_search", "x" * 7500)
    out = _summarize_tool_message(msg)
    assert "kb_search" in out


def test_summarize_tool_message_reads_as_compaction_not_failure():
    """The stub must not look like a failed/empty/elided search. A research
    worker that quotes its own compacted history narrated the old stub as
    "searches returned elided results" (the current-2025-recent regression).
    Pin the wording so that can't recur: it says "compacted", carries no
    misleading char count, and uses none of the failure-flavored words."""
    out = _summarize_tool_message(_tool_msg("web_search", "x" * 7500)).lower()
    assert "compact" in out
    assert "7500" not in out  # no count to misread as "thin results"
    for bad in ("elided", "error", "failed", "empty", "sparse"):
        assert bad not in out


def test_summarize_tool_message_handles_missing_fields():
    """Defensive against malformed history. Should not raise."""
    out = _summarize_tool_message({"role": "tool"})
    assert "?" in out  # falls back to "?" for missing name
