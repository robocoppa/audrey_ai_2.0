"""Hermetic tests for the centralized prompt module.

Covers:
  - Byte-for-byte regression: each prompt constant matches the
    pre-centralization text verbatim. A reflow in a refactor lands as
    a focused diff against this test rather than silently changing
    model behavior.
  - `prompt_from_config` resolution: missing key, null, empty string,
    non-string, valid override, oversize override (warns).
  - `compose_system_messages` ordering: incoming → task-role → memory
    → chat-history guidance. Each section is optional; composer never
    returns a system message when nothing was supplied for that slot.
  - Chat-history guidance gating: only included when caller flags the
    tool as available.

Pure functions over strings/dicts. No model calls, no I/O.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from audrey.pipeline import prompts
from audrey.pipeline.prompts import (
    CHAT_HISTORY_SEARCH_SYSTEM,
    CLASSIFIER_SYSTEM,
    MEMORY_STORE_HINT,
    PLANNER_SYSTEM,
    REACT_FINAL_ANSWER_USER,
    SYNTH_SYSTEM,
    compose_system_messages,
    prompt_from_config,
)

# ─── Byte-for-byte regression ─────────────────────────────────────────


def test_classifier_system_unchanged():
    expected = (
        "You are a task classifier. Read the user's message and output a JSON object "
        "with exactly these keys:\n"
        '  {"task": "code|reasoning|general|vl", "confidence": 0.0-1.0}\n'
        "Rules:\n"
        "- 'code' = user wants code written, debugged, refactored, explained line-by-line.\n"
        "- 'reasoning' = analysis, comparison, review, multi-step logic, math proofs, explanations.\n"
        "- 'vl' = anything referencing an image, photo, screenshot, or visual identification.\n"
        "- 'general' = chitchat, facts, summaries, everything else.\n"
        "Output ONLY the JSON object. No prose, no markdown."
    )
    assert CLASSIFIER_SYSTEM == expected


def test_planner_system_unchanged():
    expected = (
        "You decompose a user request into 2 or 3 focused sub-questions that, if "
        "answered separately, would together cover the original request. Output a "
        "JSON object with exactly this shape:\n"
        '  {"subtasks": ["...", "...", "..."]}\n'
        "Rules:\n"
        "- 2 to 3 entries, each a complete question or instruction (≤ 200 chars).\n"
        "- Sub-questions must be independent — no 'first do X then Y' chaining.\n"
        "- If the request is already atomic (one clear ask), return {\"subtasks\": []}.\n"
        "Output ONLY the JSON. No prose, no markdown."
    )
    assert PLANNER_SYSTEM == expected


def test_synth_system_unchanged():
    expected = (
        "You are the panel synthesizer. You receive the original user request "
        "plus several draft answers produced in parallel by different worker models. "
        "Some drafts may be tagged `[tool-grounded: N rounds]` — those workers ran "
        "tool calls (knowledge-base search, web search, etc.) before answering, so "
        "their factual claims are backed by retrieved evidence.\n\n"
        "Your job is to produce ONE coherent final answer for the user:\n"
        "- Speak directly to the user. Do NOT explain your synthesis process, "
        "do NOT reference 'drafts' or 'workers,' do NOT include an 'Approach' "
        "preamble. Just write the answer.\n"
        "- Pull the strongest passages from the drafts; don't average them.\n"
        "- Preserve code blocks verbatim from the strongest draft.\n"
        "- Do NOT mention worker model names.\n"
        "- When a tool-grounded draft and a tool-free draft disagree on a "
        "factual point, prefer the tool-grounded one.\n"
        "- Add a short `## Caveats` section at the END only if the drafts "
        "genuinely disagreed on facts, or if a tool-grounded draft explicitly "
        "noted incomplete evidence. Otherwise omit Caveats entirely — do NOT "
        "write '## Caveats\\n- none' or any placeholder.\n"
    )
    assert SYNTH_SYSTEM == expected


def test_react_final_answer_unchanged():
    expected = (
        "You have reached the tool-call budget. Do not call any more tools. "
        "Using only the information already gathered above, write the final "
        "answer to the original request now as plain prose. If the gathered "
        "information is insufficient, say so explicitly — do not fabricate."
    )
    assert REACT_FINAL_ANSWER_USER == expected


def test_memory_store_hint_unchanged():
    expected = (
        "If the user states a durable fact about themselves (preferences, goals, "
        "projects, constraints) or explicitly asks you to remember something, "
        "call the `memory_store` tool with: a short descriptive `key`, the fact "
        "as `value`, and `tags=\"user:{user_id}\"` (use exactly that user tag). "
        "Do this silently — do not narrate the tool call in your reply."
    )
    assert MEMORY_STORE_HINT == expected


def test_chat_history_search_system_is_non_empty():
    # No pre-move text — this constant is new in 2a. Pin shape, not text:
    # the only contract is "non-empty steering string".
    assert isinstance(CHAT_HISTORY_SEARCH_SYSTEM, str)
    assert CHAT_HISTORY_SEARCH_SYSTEM.strip()
    assert "chat_history_search" in CHAT_HISTORY_SEARCH_SYSTEM


# ─── Overrides reach the model (loader wired into call sites) ────────


def test_planner_uses_override(monkeypatch):
    """`agentic.prompts.planner` override flows into the messages
    `plan()` builds. Catches the regression where the loader exists
    but the call site still references the bare constant."""
    import asyncio

    from audrey.pipeline import planner as planner_mod

    captured: dict = {}

    class _FakeOllama:
        async def chat(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return {"message": {"content": '{"subtasks": []}'}}

    cfg = _cfg_with({"planner": "OVERRIDE_PLANNER_SYSTEM"})
    asyncio.run(planner_mod.plan(
        _FakeOllama(),
        planner_model="x", user_text="hello", timeout_s=1.0, max_subtasks=3,
        cfg=cfg,
    ))
    sys_msgs = [m for m in captured["messages"] if m["role"] == "system"]
    assert sys_msgs[0]["content"] == "OVERRIDE_PLANNER_SYSTEM"


def test_planner_falls_back_to_default_without_override():
    import asyncio

    from audrey.pipeline import planner as planner_mod

    captured: dict = {}

    class _FakeOllama:
        async def chat(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return {"message": {"content": '{"subtasks": []}'}}

    asyncio.run(planner_mod.plan(
        _FakeOllama(),
        planner_model="x", user_text="hello", timeout_s=1.0, max_subtasks=3,
        cfg=None,
    ))
    sys_msgs = [m for m in captured["messages"] if m["role"] == "system"]
    assert sys_msgs[0]["content"] == PLANNER_SYSTEM


def test_router_uses_override():
    import asyncio

    from audrey.pipeline import classify as classify_mod

    captured: dict = {}

    class _FakeOllama:
        async def chat(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return {"message": {"content": '{"task": "general", "confidence": 0.9}'}}

    cfg = _cfg_with({"classifier": "OVERRIDE_CLASSIFIER"})
    asyncio.run(classify_mod.router_classify(
        _FakeOllama(),
        router_model="x", user_text="hi", timeout_s=1.0, cfg=cfg,
    ))
    sys_msgs = [m for m in captured["messages"] if m["role"] == "system"]
    assert sys_msgs[0]["content"] == "OVERRIDE_CLASSIFIER"


def test_synth_uses_override():
    from audrey.pipeline.synthesize import _build_synth_messages

    cfg = _cfg_with({"synthesizer": "OVERRIDE_SYNTH"})
    msgs = _build_synth_messages([], "drafts go here", cfg=cfg)
    synth_sys = [m for m in msgs if m["role"] == "system"]
    # _build_synth_messages forwards prior system msgs then appends the
    # synth-system. With empty prior, the synth-system is index 0.
    assert synth_sys[0]["content"] == "OVERRIDE_SYNTH"


def test_memory_store_hint_uses_override():
    from audrey.pipeline.memory import memory_system_message

    cfg = _cfg_with({"memory_store_hint": "OVERRIDE_HINT for {user_id}"})
    msg = memory_system_message(
        hits=[], user_id="alice@example.com", include_store_hint=True, cfg=cfg,
    )
    assert msg is not None
    assert "OVERRIDE_HINT for alice@example.com" in msg["content"]


def test_react_final_answer_uses_override():
    """The override loader is called at the same call site that appends
    the final-answer user turn — we test the loader directly here since
    invoking the full ReAct loop requires a heavy mock."""
    cfg = _cfg_with({"react_final_answer": "WRAP UP NOW"})
    out = prompt_from_config(cfg, "react_final_answer", REACT_FINAL_ANSWER_USER)
    assert out == "WRAP UP NOW"


# ─── Call-site aliases match the module constants ─────────────────────


def test_call_site_aliases_match_central_constants():
    """The pipeline modules keep local underscore aliases for the old
    names; this test pins those aliases against the central constants so
    a stray edit to one side gets caught."""
    from audrey.pipeline.classify import _ROUTER_SYSTEM
    from audrey.pipeline.memory import _MEMORY_STORE_HINT
    from audrey.pipeline.planner import _PLANNER_SYSTEM
    from audrey.pipeline.synthesize import _SYNTH_SYSTEM

    assert _ROUTER_SYSTEM is CLASSIFIER_SYSTEM
    assert _PLANNER_SYSTEM is PLANNER_SYSTEM
    assert _SYNTH_SYSTEM is SYNTH_SYSTEM
    assert _MEMORY_STORE_HINT is MEMORY_STORE_HINT


# ─── prompt_from_config ───────────────────────────────────────────────


def _cfg_with(prompts_dict: dict | None) -> SimpleNamespace:
    """Build a minimal cfg-shaped object: only `raw["agentic"]["prompts"]` matters."""
    raw: dict = {}
    if prompts_dict is not None:
        raw = {"agentic": {"prompts": prompts_dict}}
    return SimpleNamespace(raw=raw)


def test_prompt_from_config_returns_default_when_cfg_is_none():
    assert prompt_from_config(None, "classifier", "DEFAULT") == "DEFAULT"


def test_prompt_from_config_returns_default_when_agentic_missing():
    cfg = SimpleNamespace(raw={})
    assert prompt_from_config(cfg, "classifier", "DEFAULT") == "DEFAULT"


def test_prompt_from_config_returns_default_when_prompts_missing():
    cfg = SimpleNamespace(raw={"agentic": {}})
    assert prompt_from_config(cfg, "classifier", "DEFAULT") == "DEFAULT"


def test_prompt_from_config_returns_default_for_null_override():
    cfg = _cfg_with({"classifier": None})
    assert prompt_from_config(cfg, "classifier", "DEFAULT") == "DEFAULT"


def test_prompt_from_config_returns_default_for_empty_string():
    cfg = _cfg_with({"classifier": ""})
    assert prompt_from_config(cfg, "classifier", "DEFAULT") == "DEFAULT"


def test_prompt_from_config_returns_default_for_whitespace_string():
    cfg = _cfg_with({"classifier": "   \n\t  "})
    assert prompt_from_config(cfg, "classifier", "DEFAULT") == "DEFAULT"


def test_prompt_from_config_returns_default_for_non_string_value(caplog):
    cfg = _cfg_with({"classifier": 42})
    with caplog.at_level(logging.WARNING, logger="audrey.pipeline.prompts"):
        out = prompt_from_config(cfg, "classifier", "DEFAULT")
    assert out == "DEFAULT"
    assert any("not str" in rec.message for rec in caplog.records)


def test_prompt_from_config_uses_valid_override():
    cfg = _cfg_with({"classifier": "OVERRIDE"})
    assert prompt_from_config(cfg, "classifier", "DEFAULT") == "OVERRIDE"


def test_prompt_from_config_warns_on_oversize(caplog):
    # Reset the per-process warning set so the test is self-contained.
    prompts._WARNED_OVERRIDES.discard("classifier")
    big = "x" * (prompts._OVERRIDE_SOFT_CAP_CHARS + 1)
    cfg = _cfg_with({"classifier": big})
    with caplog.at_level(logging.WARNING, logger="audrey.pipeline.prompts"):
        out = prompt_from_config(cfg, "classifier", "DEFAULT")
    # Override still applies — warning is a soft signal, not a kill switch.
    assert out == big
    assert any("soft cap" in rec.message for rec in caplog.records)


def test_prompt_from_config_warns_only_once_per_key(caplog):
    prompts._WARNED_OVERRIDES.discard("planner")
    big = "y" * (prompts._OVERRIDE_SOFT_CAP_CHARS + 1)
    cfg = _cfg_with({"planner": big})
    with caplog.at_level(logging.WARNING, logger="audrey.pipeline.prompts"):
        prompt_from_config(cfg, "planner", "DEFAULT")
        prompt_from_config(cfg, "planner", "DEFAULT")
        prompt_from_config(cfg, "planner", "DEFAULT")
    soft_cap_warns = [r for r in caplog.records if "soft cap" in r.message]
    assert len(soft_cap_warns) == 1


def test_prompt_from_config_rejects_unknown_key(caplog):
    cfg = _cfg_with({"not_a_real_key": "anything"})
    with caplog.at_level(logging.WARNING, logger="audrey.pipeline.prompts"):
        out = prompt_from_config(cfg, "not_a_real_key", "DEFAULT")
    assert out == "DEFAULT"
    assert any("unknown override key" in rec.message for rec in caplog.records)


# ─── compose_system_messages ──────────────────────────────────────────


def test_compose_empty_returns_empty():
    assert compose_system_messages() == []


def test_compose_passes_through_incoming_system_messages_only():
    incoming = [
        {"role": "system", "content": "you are X"},
        {"role": "user", "content": "ignored"},
        {"role": "system", "content": "also X"},
    ]
    out = compose_system_messages(incoming=incoming)
    assert out == [
        {"role": "system", "content": "you are X"},
        {"role": "system", "content": "also X"},
    ]


def test_compose_adds_task_role_after_incoming():
    incoming = [{"role": "system", "content": "owui"}]
    out = compose_system_messages(incoming=incoming, task_role="ROLE")
    assert out == [
        {"role": "system", "content": "owui"},
        {"role": "system", "content": "ROLE"},
    ]


def test_compose_adds_memory_hint_after_task_role():
    memory = {"role": "system", "content": "MEMORY"}
    out = compose_system_messages(task_role="ROLE", memory_hint=memory)
    assert out == [
        {"role": "system", "content": "ROLE"},
        memory,
    ]


def test_compose_canonical_order_full_stack():
    """All four slots populated → order is incoming, task_role, memory, chat-history."""
    incoming = [{"role": "system", "content": "owui"}]
    memory = {"role": "system", "content": "MEM"}
    out = compose_system_messages(
        incoming=incoming,
        task_role="ROLE",
        memory_hint=memory,
        chat_history_guidance=True,
    )
    assert [m["content"] for m in out] == ["owui", "ROLE", "MEM", CHAT_HISTORY_SEARCH_SYSTEM]
    assert all(m["role"] == "system" for m in out)


def test_compose_omits_chat_history_when_flag_off():
    out = compose_system_messages(
        memory_hint={"role": "system", "content": "MEM"},
        chat_history_guidance=False,
    )
    assert out == [{"role": "system", "content": "MEM"}]


def test_compose_uses_custom_chat_history_text_when_provided():
    out = compose_system_messages(
        chat_history_guidance=True,
        chat_history_text="CUSTOM",
    )
    assert out == [{"role": "system", "content": "CUSTOM"}]


def test_compose_skips_chat_history_when_text_is_empty():
    """A custom empty chat-history-text means the caller explicitly disabled
    it; composer must not synthesize a blank system message."""
    out = compose_system_messages(
        chat_history_guidance=True,
        chat_history_text="   ",
    )
    assert out == []


def test_compose_returns_fresh_list_each_call():
    a = compose_system_messages(task_role="X")
    b = compose_system_messages(task_role="X")
    assert a == b
    assert a is not b
    a.append({"role": "system", "content": "mutated"})
    assert len(b) == 1  # b unaffected


@pytest.mark.parametrize("flag", [True, False])
def test_compose_skips_blank_default_chat_history_text(flag, monkeypatch):
    """If the default chat-history text were ever blanked out (e.g. via
    override), composer must not emit a hollow system message."""
    monkeypatch.setattr(prompts, "CHAT_HISTORY_SEARCH_SYSTEM", "  ")
    out = compose_system_messages(chat_history_guidance=flag)
    assert out == []
