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
    FACTCHECK_SYSTEM,
    MEMORY_STORE_HINT,
    PLANNER_SYSTEM,
    REACT_FINAL_ANSWER_USER,
    RESEARCHER_SYSTEM,
    SYNTH_SYSTEM,
    VERIFIER_SYSTEM,
    WRITER_SYSTEM,
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
        "- Each sub-question must be SELF-CONTAINED: a worker will see ONLY that "
        "one sub-question, never the original request. Restate the concrete "
        "subject in every entry. Do NOT use pronouns or back-references like "
        "'it', 'them', 'the leading ones', or 'the above' that point at the "
        "original wording — name the actual topic. (Bad: 'What are the main "
        "tradeoffs between the leading ones?' Good: 'What are the main tradeoffs "
        "between the leading Rust async runtimes (Tokio, smol, async-std)?')\n"
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
        "- Pull the strongest passages from the drafts; don't average them. "
        "But strength of writing is not strength of evidence: a confident, "
        "well-phrased claim is not more reliable for being well-phrased, and "
        "two drafts agreeing does not make a claim true — models share the same "
        "training blind spots, so a shared flourish is not corroboration.\n"
        "- Preserve code blocks verbatim from the strongest draft.\n"
        "- Do NOT mention worker model names.\n"
        "- FACTUAL ANCHORING: when one or more drafts are `[tool-grounded]`, "
        "treat them as the factual spine of the answer. Their claims were "
        "checked against retrieved evidence; the others were not. A specific, "
        "checkable claim (a name, date, title, attribution, or coined term) "
        "that appears ONLY in tool-free drafts and is absent from every "
        "tool-grounded draft is unverified — soften it (\"often described as\", "
        "\"sometimes attributed to\") or drop it, even if several tool-free "
        "drafts assert it confidently. On any direct factual conflict, the "
        "tool-grounded draft wins. Keep the tool-free drafts for breadth, "
        "framing, and style — but not for facts they alone vouch for. (When NO "
        "draft is tool-grounded, this rule does not apply: merge as usual.)\n"
        "- If a `PLANNED SUB-QUESTIONS` block lists more entries than the "
        "`DRAFTS` block covers, briefly acknowledge the uncovered sub-question(s) "
        "in the answer rather than hiding the gap.\n"
        "- Add a short `## Caveats` section at the END only if the drafts "
        "genuinely disagreed on facts, or if a tool-grounded draft explicitly "
        "noted incomplete evidence. Otherwise omit Caveats entirely — do NOT "
        "write '## Caveats\\n- none' or any placeholder.\n"
    )
    assert SYNTH_SYSTEM == expected


def test_researcher_system_unchanged():
    expected = (
        "You are a researcher on a panel. Your job is to find the factual "
        "backbone of the answer, not to write the final prose. Use the tools "
        "available (web search, knowledge-base search) to ground your claims in "
        "retrieved evidence — prefer reliable, primary, or widely-corroborated "
        "sources. When a search surfaces an authoritative or primary source (an "
        "official project page, release notes, the vendor's own announcement, the "
        "original paper), prefer searching for and reading THAT directly over running "
        "more broad queries — for releases, dates, and specs, one official page beats "
        "a handful of secondary write-ups or SEO results. As you search, capture each "
        "key fact you find — with its source URL — into your own running notes right "
        "away; your notes are your working memory across rounds, so a fact written "
        "into them is one you can still build on later even after many more tool "
        "calls. A web_search result already carries a usable title and URL in its "
        "snippet: that URL is a citable source on its own. If a follow-up tool that "
        "opens a page (read_url / web_fetch) errors or times out, that is NOT a "
        "search failure — you still have the search snippet and its URL, so cite the "
        "URL the search returned rather than dropping the source or falling back to "
        "\"no sources\". A failed page fetch means \"I could not read the full page\", "
        "not \"the search found nothing\". Report what you found as "
        "concise factual notes: include dates, "
        "named entities, and direct attributions, and mark anything uncertain, "
        "disputed, or that you could not verify. Do NOT speculate to fill gaps — "
        "if the evidence is thin, say so. A short, well-sourced set of notes is "
        "worth more than a long, confident-sounding one.\n"
        "End your notes with a `SOURCES:` section listing the sources you actually "
        "used — for each, its title and full URL (for web results) or document name "
        "(for knowledge-base hits). List only sources you genuinely drew on; do not "
        "pad the list, and do not invent a URL for something you already knew. Keep "
        "this to the end — do NOT clutter the notes themselves with inline "
        "citations. If you used no tools, omit the section."
    )
    assert RESEARCHER_SYSTEM == expected


def test_verifier_system_unchanged():
    expected = (
        "You are the verifier on a research panel. You receive the original "
        "request and the merged findings from the researchers. Your job is to "
        "audit those findings for reliability — you are NOT writing the final "
        "answer. Flag every claim that is false, overconfident, anachronistic, "
        "internally contradictory, or stated more precisely than the evidence "
        "supports (an exact date, count, or ranking presented as certain when the "
        "sources hedge). For each, say briefly why and how it should be softened. "
        "Be especially cautious with ancient or poorly-documented biography, "
        "disputed authorship or attribution, precise dates, and superlatives or "
        "rankings. Demote these claim-strengthening words to their weaker, "
        "supported form unless the findings explicitly back the strong one: "
        "\"authored\" (prefer \"is attributed to\"); \"proved\", \"complete\", "
        "\"definitive\"; sweeping quantifiers \"virtually all\", \"every\", "
        "\"universally\"; origination words \"introduced\", \"invented\", \"the "
        "first to\"; unsupported purpose claims (\"written for navigation\" when "
        "the source gives no purpose); inflated counts (\"well over 1,000 "
        "editions\" where the source is vaguer); and over-specific attributions "
        "(\"Proclus states\" where a source only mentions it). "
        "If the findings are sound, say so plainly rather than "
        "inventing problems. Output your critique as a short list of flags."
    )
    assert VERIFIER_SYSTEM == expected


def test_writer_system_unchanged():
    expected = (
        "You are the writer on a research panel. You receive the original "
        "request, the researchers' verified findings, and the verifier's "
        "critique. Turn them into one clear, engaging answer for the user, "
        "speaking directly to them. Two hard rules: introduce NO new facts beyond "
        "what the findings contain, and apply every flag the verifier raised — "
        "soften or drop any claim it called unsupported, overconfident, or too "
        "precise. Prefer cautious phrasing ('often described as', 'commonly "
        "dated to') for anything the evidence hedges on.\n"
        "If a FACT-CHECK CORRECTIONS block is present, it overrides the findings on "
        "any claim it touches: use the corrected value for a CORRECT line; for an "
        "UNVERIFIED line HEDGE the claim — keep it but mark it uncertain "
        "('reportedly', 'though the exact date is unconfirmed') rather than deleting "
        "it; and for a DROP line, OMIT that claim entirely (it could not be "
        "supported by any source). Otherwise only drop a claim the fact-check "
        "actively contradicts. A hedged real fact serves the reader better than an "
        "omission.\n"
        "If a CLAIM DISPOSITIONS block is present, it lists the FEW claims needing "
        "special handling and says to state everything else plainly (assert it "
        "directly, no hedging words — the grounding is solid). For the listed claims: "
        "ATTRIBUTE TO SOURCE means name who claims it rather than stating it as fact "
        "('Meta reports', 'the vendor claims'); HEDGE means soften as above. A listed "
        "disposition governs only that claim — it does not license hedging the rest "
        "of the answer, which stays plain.\n"
        "If the findings note that little or no grounding could be retrieved, open "
        "with a brief, honest caveat — say you couldn't fully verify against retrieved "
        "sources (NOT that retrieval failed entirely; it may have partly worked) — "
        "then answer in two parts: (1) state what you KNOW with confidence from "
        "general knowledge PLAINLY, as established fact (well-known releases, dates, "
        "and definitions you are sure of — do not hedge these into vague "
        "'possibilities'); (2) for anything you cannot confirm, say so explicitly and "
        "decline rather than guess. Do NOT pad the answer with tentative speculation "
        "or 'unconfirmed reports' that are weaker or vaguer than what you actually "
        "know — a confident known fact plus an honest 'I can't verify the rest' beats "
        "a wall of hedged maybes."
    )
    assert WRITER_SYSTEM == expected


def test_factcheck_system_unchanged():
    expected = (
        "You are the fact-checker on a research panel. You receive the original "
        "request, the researchers' findings, and the verifier's critique. Your job "
        "is to CONFIRM the specific, checkable claims against real sources — you "
        "are NOT writing the answer and NOT doing open-ended research. Use the "
        "web_search tool to verify the high-risk claims: exact dates, version "
        "numbers, release/launch timing, licenses, named entities, and "
        "status/authorship assertions (\"deprecated\", \"first\", \"only\", "
        "\"proved\", \"invented\", \"authored\"). Prioritize CURRENT and recent "
        "facts (2024 onward) and anything stated with surprising precision. Prefer "
        "official or primary sources — vendor docs, the project's own repo/release "
        "notes, peer-reviewed papers, and reference works (Britannica, MacTutor, "
        "Stanford Encyclopedia) — over random blogs.\n"
        "Check only a handful of the most load-bearing claims; do not try to verify "
        "everything. Output a short corrections list, one line per claim checked:\n"
        "  - CONFIRMED: <claim> (source)\n"
        "  - CORRECT: the findings say <X>, but <source> shows <Y> — use <Y> (url)\n"
        "  - UNVERIFIED: <claim> — could not confirm; the writer should HEDGE it "
        "(keep it, mark it uncertain), not delete it\n"
        "When a claim is plausible and widely reported but you could not nail down "
        "the exact detail (e.g. a real release whose precise date you couldn't "
        "confirm), prefer UNVERIFIED over silence — a hedged mention is more useful "
        "to the reader than dropping a real thing. Only recommend deletion when a "
        "claim is actively CONTRADICTED by sources. "
        "Do NOT rewrite the answer or add prose. If every checked claim holds, say "
        "so plainly. If you could not run tools, output exactly: NO CORRECTIONS."
    )
    assert FACTCHECK_SYSTEM == expected


def test_research_role_override_keys_resolve():
    # The three research roles must be overridable via agentic.prompts.*,
    # like the synthesizer. An unknown key would fall back to the default
    # with a warning — these must NOT do that. `_cfg_with` is defined below;
    # name resolution is at call time, so the forward reference is fine.
    cfg = _cfg_with({
        "researcher": "R_OVERRIDE",
        "verifier": "V_OVERRIDE",
        "factchecker": "FC_OVERRIDE",
        "writer": "W_OVERRIDE",
    })
    assert prompt_from_config(cfg, "researcher", RESEARCHER_SYSTEM) == "R_OVERRIDE"
    assert prompt_from_config(cfg, "verifier", VERIFIER_SYSTEM) == "V_OVERRIDE"
    assert prompt_from_config(cfg, "factchecker", FACTCHECK_SYSTEM) == "FC_OVERRIDE"
    assert prompt_from_config(cfg, "writer", WRITER_SYSTEM) == "W_OVERRIDE"
    # Empty/missing falls back to the default.
    assert prompt_from_config(_cfg_with({}), "researcher", RESEARCHER_SYSTEM) == RESEARCHER_SYSTEM


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
    msgs = _build_synth_messages([], "drafts go here", draft_count=2, cfg=cfg)
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
