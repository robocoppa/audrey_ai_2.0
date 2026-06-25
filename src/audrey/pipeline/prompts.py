"""Central home for Audrey's pipeline prompts.

Every prompt the orchestrator hands to a model lives here. Pipeline
modules import these constants instead of carrying their own — keeps
prompt tuning a single-file change, keeps token accounting honest,
and gives the override loader one place to read from.

Three things this module owns:

  - Default prompt constants. Moved byte-for-byte from their old
    homes (classify / planner / synthesize / react / memory). A
    regression test pins each one against the pre-move source so
    accidental reflows show up in a separate diff.

  - `prompt_from_config(cfg, key, default)`. Reads
    `cfg.raw["agentic"]["prompts"][key]`; `null`, missing, or
    empty-string fall back to the code default. Overrides longer
    than a soft cap emit a one-line warning at load time so a
    runaway persona is visible without being silently truncated.

  - `compose_system_messages(...)`. Single helper every pipeline
    node calls when it wants to add system context. Pins the canonical
    order: incoming → task-role → memory → chat-history. Without this
    helper the ordering drifts every time a new system message is
    added; with it, the rule lives in one function.

`CHAT_HISTORY_SEARCH_SYSTEM` is new in 2a — it reinforces the
chat_history_search tool description on the system-message side so the
model doesn't over-call the archive lookup. The composer adds it only
when `chat_history_search` is actually in the registry; no point
telling the model how to use a tool it can't dispatch.

This module never makes a model call. It only assembles strings and
dicts.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# ─── Default prompts ──────────────────────────────────────────────────
# Moved byte-for-byte from the old call sites. A test pins each one
# against the historical text; changing a prompt should land in a
# focused diff, not as a side-effect of a refactor.

CLASSIFIER_SYSTEM = (
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

PLANNER_SYSTEM = (
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

SYNTH_SYSTEM = (
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

# ─── audrey_research role prompts ─────────────────────────────────────
# The staged research pipeline (Stage 1 research fan-out → Verify → Write)
# gives each stage a distinct role prompt. Researchers ground with tools;
# the verifier audits the findings; the writer turns verified findings into
# the answer without inventing new facts. Each is overridable via
# `agentic.prompts.{researcher,verifier,writer}`.

RESEARCHER_SYSTEM = (
    "You are a researcher on a panel. Your job is to find the factual "
    "backbone of the answer, not to write the final prose. Use the tools "
    "available (web search, knowledge-base search) to ground your claims in "
    "retrieved evidence — prefer reliable, primary, or widely-corroborated "
    "sources. Report what you found as concise factual notes: include dates, "
    "named entities, and direct attributions, and mark anything uncertain, "
    "disputed, or that you could not verify. Do NOT speculate to fill gaps — "
    "if the evidence is thin, say so. A short, well-sourced set of notes is "
    "worth more than a long, confident-sounding one.\n"
    "ALWAYS record where each fact came from. For every note, append the "
    "source it rests on — the page title and full URL for web results, or the "
    "document/source name for knowledge-base hits. The writer can only cite "
    "sources you carry forward, so a fact with no source attached cannot be "
    "cited. Keep a clear `Sources:` list at the end of your notes pairing each "
    "fact to its title + URL."
)

VERIFIER_SYSTEM = (
    "You are the verifier on a research panel. You receive the original "
    "request and the merged findings from the researchers. Your job is to "
    "audit those findings for reliability — you are NOT writing the final "
    "answer. Flag every claim that is false, overconfident, anachronistic, "
    "internally contradictory, or stated more precisely than the evidence "
    "supports (an exact date, count, or ranking presented as certain when the "
    "sources hedge). Also flag any substantive factual claim that carries NO "
    "source — it cannot be cited and should be softened or dropped. For each "
    "flag, say briefly why and how it should be handled. "
    "Be especially cautious with ancient or poorly-documented biography, "
    "disputed authorship or attribution, precise dates, and superlatives or "
    "rankings. If the findings are sound, say so plainly rather than "
    "inventing problems. Output your critique as a short list of flags."
)

WRITER_SYSTEM = (
    "You are the writer on a research panel. You receive the original "
    "request, the researchers' verified findings, and the verifier's "
    "critique. Turn them into one clear, engaging answer for the user, "
    "speaking directly to them. Two hard rules: introduce NO new facts beyond "
    "what the findings contain, and apply every flag the verifier raised — "
    "soften or drop any claim it called unsupported, overconfident, or too "
    "precise. Prefer cautious phrasing ('often described as', 'commonly "
    "dated to') for anything the evidence hedges on.\n"
    "CITE YOUR SOURCES. The findings carry sources (titles + URLs). Add an "
    "inline marker — [1], [2] — to each factual claim, pointing at the source "
    "it rests on, and end the answer with a `## Sources` section listing each "
    "number with its title and full URL as a markdown link. Only cite sources "
    "that actually appear in the findings — never invent a citation or a URL. "
    "If two claims share a source, reuse the same number.\n"
    "If the findings note that little or no grounding could be retrieved, "
    "write from general knowledge BUT open with a brief, honest caveat that "
    "the answer could not be verified against sources and may be incomplete, "
    "keep specific claims (exact dates, attributions, coined terms) "
    "deliberately tentative, and OMIT the `## Sources` section entirely "
    "rather than printing an empty one."
)

REACT_FINAL_ANSWER_USER = (
    "You have reached the tool-call budget. Do not call any more tools. "
    "Using only the information already gathered above, write the final "
    "answer to the original request now as plain prose. If the gathered "
    "information is insufficient, say so explicitly — do not fabricate."
)

MEMORY_STORE_HINT = (
    "If the user states a durable fact about themselves (preferences, goals, "
    "projects, constraints) or explicitly asks you to remember something, "
    "call the `memory_store` tool with: a short descriptive `key`, the fact "
    "as `value`, and `tags=\"user:{user_id}\"` (use exactly that user tag). "
    "Do this silently — do not narrate the tool call in your reply."
)

# New in 2a: steers tool-capable models toward deliberate archive lookup
# without bloating ordinary prompts. Only injected when chat_history_search
# is in the registry — see `compose_system_messages`.
CHAT_HISTORY_SEARCH_SYSTEM = (
    "Use `chat_history_search` only when the user references something they "
    "previously discussed with you, or when answering requires a specific "
    "prior decision. Do not call it for ordinary personalization or to "
    "repeat back recent context — it returns short snippets per call and "
    "burns context every time."
)


# ─── Override loader ──────────────────────────────────────────────────

_OVERRIDE_SOFT_CAP_CHARS = 4000
_WARNED_OVERRIDES: set[str] = set()  # one warning per key per process

_PROMPT_KEYS = frozenset({
    "classifier",
    "planner",
    "synthesizer",
    "researcher",
    "verifier",
    "writer",
    "react_final_answer",
    "memory_store_hint",
    "chat_history_search",
})


def prompt_from_config(cfg: Any, key: str, default: str) -> str:
    """Return the override from `agentic.prompts.<key>` or `default`.

    Resolution rules:
      - Missing key, `None`, or empty / whitespace string → default.
      - Non-string value → default, with a warning (config bug).
      - Override longer than the soft cap → keep the override, log a
        one-line warning. Soft, not hard: a runaway persona is visible
        in logs without us silently truncating user config.

    The warning is emitted at most once per (key, process) so a hot
    config-read path doesn't spam logs.
    """
    if key not in _PROMPT_KEYS:
        # Programmer error — calling code asked for a key we don't know
        # about. Caller gets the default; we log loudly so the typo
        # surfaces in tests / startup.
        log.warning("prompts: unknown override key %r — using default", key)
        return default

    raw = None
    if cfg is not None:
        agentic = getattr(cfg, "raw", {}).get("agentic", {}) or {}
        prompts = agentic.get("prompts", {}) or {}
        raw = prompts.get(key)

    if raw is None:
        return default
    if not isinstance(raw, str):
        log.warning(
            "prompts: override %r is %s, not str — using default",
            key, type(raw).__name__,
        )
        return default
    stripped = raw.strip()
    if not stripped:
        return default
    if len(raw) > _OVERRIDE_SOFT_CAP_CHARS and key not in _WARNED_OVERRIDES:
        log.warning(
            "prompts: override %r is %d chars (> %d soft cap) — token cost will rise",
            key, len(raw), _OVERRIDE_SOFT_CAP_CHARS,
        )
        _WARNED_OVERRIDES.add(key)
    return raw


# ─── System-message composer ──────────────────────────────────────────


def compose_system_messages(
    *,
    incoming: list[dict[str, Any]] | None = None,
    task_role: str | None = None,
    memory_hint: dict[str, Any] | None = None,
    chat_history_guidance: bool = False,
    chat_history_text: str | None = None,
) -> list[dict[str, Any]]:
    """Return a list of system messages in canonical order.

    Order, fixed:
      1. Incoming system messages — preserved as given. Anything the
         user / OWUI sent at `role=system` lands first so the user's
         persona wins on tone.
      2. Task-role prompt — Phase 2a always passes `None` here. The
         slot exists so Phase 2b can drop in a fast-answerer or deep-
         worker role prompt without further wiring.
      3. Memory recall + memory_store hint — passed as a pre-built
         system message because memory.py already formats the body
         (hits + optional store hint). The composer treats it as
         opaque content.
      4. Chat-history search guidance — included only when
         `chat_history_guidance=True`, signaling that the live
         registry has `chat_history_search`. `chat_history_text`
         lets callers override the default text without going through
         the config loader (rare; mostly used in tests).

    Returns a fresh list. The caller is responsible for placing it
    relative to the user/assistant turns. Existing call sites prepend
    these messages before the original user turns.
    """
    out: list[dict[str, Any]] = []
    if incoming:
        for m in incoming:
            if m.get("role") == "system":
                out.append(m)
    if task_role:
        out.append({"role": "system", "content": task_role})
    if memory_hint is not None:
        out.append(memory_hint)
    if chat_history_guidance:
        body = chat_history_text if chat_history_text is not None else CHAT_HISTORY_SEARCH_SYSTEM
        if body.strip():
            out.append({"role": "system", "content": body})
    return out


__all__ = [
    "CLASSIFIER_SYSTEM",
    "PLANNER_SYSTEM",
    "SYNTH_SYSTEM",
    "RESEARCHER_SYSTEM",
    "VERIFIER_SYSTEM",
    "WRITER_SYSTEM",
    "REACT_FINAL_ANSWER_USER",
    "MEMORY_STORE_HINT",
    "CHAT_HISTORY_SEARCH_SYSTEM",
    "prompt_from_config",
    "compose_system_messages",
]
