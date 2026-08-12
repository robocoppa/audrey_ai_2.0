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

import json
import logging
from types import SimpleNamespace

import pytest

from audrey.pipeline.react import (
    _WEB_SEARCH_BUDGET_STUB,
    _compress_history,
    _context_census,
    _summarize_tool_message,
    _without_web_search,
    run_react,
)
from audrey.tools.discovery import ToolRegistry, ToolSpec
from audrey.tools.dispatch import ToolResult


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
    """Regression-only test: `keep_last_round=0` is not a recommended value.
    It no-ops via an explicit `<= 0` guard in the helper. (Before the
    2026-07-21 rewrite this fell out *accidentally* from `tool_indices[-0]`
    evaluating to `tool_indices[0]`; the guard makes it intentional.)
    Sensible values are ≥1; to wipe tool history aggressively, lower
    `compress_after_round` instead.

    This test exists as a tripwire — if someone drops the guard while
    refactoring, 0 would start stubbing *every* tool message, and the config
    doc note in `config.yaml` should be revisited."""
    convo = [
        _tool_msg("web_search", "a"),
        _tool_msg("kb_search", "b"),
    ]
    out = _compress_history(convo, keep_last_round=0)
    assert out == convo  # current behavior: no compression at 0


def test_compress_history_evicts_failures_before_successful_searches():
    """A failed tool call must never displace a successful `web_search`.

    The failures here are the two MOST RECENT messages, so a purely
    recency-based policy would keep them and stub both searches — exactly the
    2026-07-21 production failure, where a worker kept two failed `web_fetch`
    calls, lost all four of its searches, and then reported that its searches
    "returned empty results"."""
    err = json.dumps({"error": "http_502", "detail": "bad gateway"})
    convo = [
        _tool_msg("web_search", "grounding one"),
        _tool_msg("web_search", "grounding two"),
        _tool_msg("web_fetch", err),
        _tool_msg("web_fetch", err),
    ]
    out = _compress_history(convo, keep_last_round=2)
    # Both successful searches survive verbatim despite being oldest.
    assert out[0] == convo[0]
    assert out[1] == convo[1]
    # Both failures are stubbed despite being newest.
    assert out[2]["role"] == "system" and "web_fetch" in out[2]["content"]
    assert out[3]["role"] == "system" and "web_fetch" in out[3]["content"]


def test_compress_history_recency_still_decides_among_successes():
    """The error tier must not disturb ordinary recency behavior when every
    tool message succeeded — pins that the new tier is additive."""
    convo = [
        _tool_msg("web_search", "oldest"),
        _tool_msg("kb_search", "middle"),
        _tool_msg("memory_search", "newest"),
    ]
    out = _compress_history(convo, keep_last_round=2)
    assert out[0]["role"] == "system"          # oldest stubbed
    assert out[1] == convo[1]
    assert out[2] == convo[2]


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


# ─── _context_census (the diagnostic, 2026-08-12) ──────────────────────
#
# ⚠️ Written because fourteen A-B runs were diagnosed entirely from ANSWERS,
# and an answer cannot distinguish a model inventing "I only have a partial
# excerpt" from a model correctly describing a context compaction had already
# thinned. This line settles that in one run.


def test_context_census_counts_live_results_and_stubs():
    """The shape a paging turn actually arrives in: three `get_file_text`
    calls, one surviving. That is what the model writes its answer from."""
    convo = [
        {"role": "user", "content": "give me the transcript"},
        _tool_msg("get_file_text", "page one"),
        _tool_msg("get_file_text", "page two"),
        _tool_msg("get_file_text", "page three"),
    ]
    census = _context_census(_compress_history(convo, keep_last_round=1))
    assert "compacted_out=2" in census
    assert census.count("get_file_text:") == 1     # one live result left
    assert "get_file_text:10" in census            # len("page three")


def test_context_census_reports_total_conversation_size():
    """⚠️ Nothing in this repo sets `num_ctx`, so Ollama's per-model default
    applies and an over-long conversation is truncated FROM THE FRONT, taking
    the system prompt with it. Raising `compress_keep_last` trades one failure
    for that one, so the census has to make the size visible."""
    convo = [{"role": "system", "content": "x" * 100},
             _tool_msg("kb_search", "y" * 250)]
    assert "convo_chars=350" in _context_census(convo)


def test_context_census_survives_an_empty_conversation():
    census = _context_census([])
    assert "live_tool_results=[-]" in census and "compacted_out=0" in census


def test_the_stub_is_what_the_census_counts():
    """Coupled on purpose: the census identifies a compacted message by the
    stub's opening text, so a reworded stub must not silently zero the count."""
    stub = _summarize_tool_message(_tool_msg("get_file_text", "x"))
    assert stub.startswith("[history compacted:")
    assert "compacted_out=1" in _context_census(
        [{"role": "system", "content": stub}])


async def test_the_trace_fires_on_the_path_almost_every_turn_takes(
        fake_dispatch, caplog):
    """⚠️ The placement bug, caught live on 2026-08-12: the first census went
    only after the loop, on the `max_rounds`-exhausted path. That path is RARE
    — the normal exit is the model stopping tool calls and returning from
    INSIDE the loop — so a correctly instrumented run produced no `FINAL` line
    at all and looked broken.

    The `ANSWERED` census sits on that normal return, and its `convo` is
    exactly what produced the prose: compaction ran at the top of the
    iteration and nothing was appended after it.
    """
    ollama = _FakeOllama([
        {"message": {"tool_calls": [_call("kb_search", "a")]}},
        {"message": {"content": "done"}, "prompt_eval_count": 1, "eval_count": 1},
    ])
    with caplog.at_level(logging.INFO, logger="audrey.pipeline.react"):
        out = await run_react(
            ollama, _FakeHealth(), _registry_ws_kb(),
            model="m", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5, max_rounds=3, compress_after_round=99,
            max_tool_result_chars=1000, tool_dispatch_timeout_s=5,
            location="cloud", max_web_searches=3,
            cfg=SimpleNamespace(raw={"agentic": {"debug_context_trace": True}}),
        )
    assert out.content == "done"
    answered = [r.message for r in caplog.records if "ANSWERED" in r.message]
    assert len(answered) == 1, [r.message for r in caplog.records]
    # It reports the tool result that was live when the answer was written.
    assert "kb_search:" in answered[0]
    assert "compacted_out=0" in answered[0]
    # And no FINAL line, because max_rounds was never exhausted — the exact
    # shape that made the first version look like it had failed.
    assert not [r for r in caplog.records if "FINAL" in r.message]


async def test_the_trace_stays_silent_when_the_flag_is_off(fake_dispatch, caplog):
    """It ships dark. A diagnostic that logs unconditionally becomes noise
    nobody greps."""
    ollama = _FakeOllama([
        {"message": {"content": "done"}, "prompt_eval_count": 1, "eval_count": 1},
    ])
    with caplog.at_level(logging.INFO, logger="audrey.pipeline.react"):
        await run_react(
            ollama, _FakeHealth(), _registry_ws_kb(),
            model="m", messages=[{"role": "user", "content": "q"}],
            options={}, timeout_s=5, max_rounds=3, compress_after_round=99,
            max_tool_result_chars=1000, tool_dispatch_timeout_s=5,
            location="cloud", max_web_searches=3,
        )
    assert not [r for r in caplog.records if "context-trace" in r.message]


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


# ─── web_search budget (max_web_searches) ─────────────────────────────
#
# Every dispatched web_search spends real provider quota (Brave), so the
# loop enforces a per-worker cap: over-budget calls get a budget-note tool
# message instead of a dispatch, and web_search stops being offered in
# later rounds. These tests drive run_react with a fake Ollama and a fake
# dispatcher — no network.


def _registry_ws_kb() -> ToolRegistry:
    mk = lambda n: ToolSpec(name=n, description=n,  # noqa: E731
                            parameters={"type": "object", "properties": {}},
                            server_url="http://unused", path=f"/{n}")
    return ToolRegistry(by_name={"web_search": mk("web_search"), "kb_search": mk("kb_search")})


def _call(name: str, cid: str) -> dict:
    return {"id": cid, "function": {"name": name, "arguments": {}}}


class _FakeHealth:
    def record_success(self, model): pass
    def record_failure(self, model, err): pass


class _FakeOllama:
    """Returns queued chat responses; records the kwargs of every call."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, *, model, messages, options=None, tools=None, timeout_s=0, think=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)


@pytest.fixture
def fake_dispatch(monkeypatch):
    """Replace the real HTTP dispatcher; records dispatched tool names."""
    dispatched: list[str] = []

    async def _fake(http, registry, tc, *, max_result_chars, timeout_s, user_id=None):
        name = ((tc.get("function") or {}).get("name")) or "?"
        dispatched.append(name)
        return ToolResult(name=name, call_id=tc.get("id"),
                          content='{"ok": true}', elapsed_s=0.01, is_error=False)

    monkeypatch.setattr("audrey.pipeline.react.dispatch_one", _fake)
    return dispatched


async def test_web_search_cap_stubs_overflow_and_stops_offering(fake_dispatch):
    # Round 1: two web searches (budget 3 → both dispatch).
    # Round 2: two more web searches + one kb_search → one web dispatches
    #          (3rd), one is stubbed; kb_search always dispatches.
    # Round 3: model answers. Its chat call must no longer offer web_search.
    ollama = _FakeOllama([
        {"message": {"tool_calls": [_call("web_search", "a"), _call("web_search", "b")]}},
        {"message": {"tool_calls": [_call("web_search", "c"), _call("web_search", "d"),
                                    _call("kb_search", "e")]}},
        {"message": {"content": "final answer"}, "prompt_eval_count": 1, "eval_count": 1},
    ])
    out = await run_react(
        ollama, _FakeHealth(), _registry_ws_kb(),
        model="m", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5, max_rounds=3, compress_after_round=99,
        max_tool_result_chars=1000, tool_dispatch_timeout_s=5,
        location="cloud", max_web_searches=3,
    )
    assert out.content == "final answer"
    assert fake_dispatch == ["web_search", "web_search", "web_search", "kb_search"]
    # Footer data counts real dispatches only — the stub is not in tool_calls.
    assert len(out.tool_calls) == 4
    # web_search_chars sums ONLY successful web_search bodies (11 chars each,
    # '{"ok": true}'), not kb_search and not the budget-stubbed 4th web call:
    # 3 dispatched web_search × 11 = 33.
    assert out.web_search_chars == 3 * len('{"ok": true}')
    # The stubbed call got a budget-note tool message in the convo.
    round3_msgs = ollama.calls[2]["messages"]
    stubs = [m for m in round3_msgs if m.get("role") == "tool"
             and m.get("content") == _WEB_SEARCH_BUDGET_STUB]
    assert len(stubs) == 1
    # And round 3 no longer offers web_search (kb_search still offered).
    round3_tools = ollama.calls[2]["tools"]
    names = {((t.get("function") or {}).get("name")) for t in round3_tools}
    assert names == {"kb_search"}


async def test_web_search_cap_zero_means_unlimited(fake_dispatch):
    ollama = _FakeOllama([
        {"message": {"tool_calls": [_call("web_search", str(i)) for i in range(5)]}},
        {"message": {"content": "done"}},
    ])
    out = await run_react(
        ollama, _FakeHealth(), _registry_ws_kb(),
        model="m", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5, max_rounds=2, compress_after_round=99,
        max_tool_result_chars=1000, tool_dispatch_timeout_s=5,
        location="cloud", max_web_searches=0,
    )
    assert out.content == "done"
    assert fake_dispatch.count("web_search") == 5
    # Both rounds still offered the full toolset.
    for call in ollama.calls:
        names = {((t.get("function") or {}).get("name")) for t in call["tools"]}
        assert "web_search" in names


async def test_web_search_chars_excludes_failed_searches(monkeypatch):
    # A web_search that errors carried NO grounding to the model, so it must
    # not inflate web_search_chars — otherwise "retrieved but thin" and
    # "search failed" read the same in the trace. Here every dispatch errors.
    async def _fail(http, registry, tc, *, max_result_chars, timeout_s, user_id=None):
        name = ((tc.get("function") or {}).get("name")) or "?"
        return ToolResult(name=name, call_id=tc.get("id"),
                          content='{"error": "boom"}', elapsed_s=0.01, is_error=True)

    monkeypatch.setattr("audrey.pipeline.react.dispatch_one", _fail)
    ollama = _FakeOllama([
        {"message": {"tool_calls": [_call("web_search", "a"), _call("web_search", "b")]}},
        {"message": {"content": "done"}},
    ])
    out = await run_react(
        ollama, _FakeHealth(), _registry_ws_kb(),
        model="m", messages=[{"role": "user", "content": "q"}],
        options={}, timeout_s=5, max_rounds=2, compress_after_round=99,
        max_tool_result_chars=1000, tool_dispatch_timeout_s=5,
        location="cloud", max_web_searches=0,
    )
    assert out.web_search_chars == 0


def test_web_search_budget_stub_reads_as_limit_not_failure():
    """Same wording discipline as the compaction stub: a model quoting this
    must not narrate a FAILED search."""
    low = _WEB_SEARCH_BUDGET_STUB.lower()
    assert "limit" in low
    for bad in ("elided", "error", "failed", "empty", "sparse", "unavailable"):
        assert bad not in low


def test_without_web_search_filters_and_none_when_empty():
    ws = {"function": {"name": "web_search"}}
    kb = {"function": {"name": "kb_search"}}
    assert _without_web_search([ws, kb]) == [kb]
    assert _without_web_search([ws]) is None   # nothing left → None, not []
    assert _without_web_search(None) is None
