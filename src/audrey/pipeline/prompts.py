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
    "- 'vl' = the user ATTACHED an image for you to look at, or asks about one "
    "they attached earlier in this conversation.\n"
    "- 'general' = chitchat, facts, summaries, everything else — including "
    "every question about a file already in the user's knowledge base. A video "
    "or image they uploaded earlier is 'general', NOT 'vl': its transcript and "
    "its on-screen descriptions are stored as text and are fetched with tools. "
    "Naming a .mp4 does not make a request visual.\n"
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

# ─── Deep-panel worker role ───────────────────────────────────────────
# Injected only for TOOL-CAPABLE deep-panel workers (see
# `deep_panel._prepare_panel`). Until 2026-07-21 deep workers ran with no role
# prompt at all — they saw the bare conversation and had no idea they were one
# of several drafts feeding a synthesizer.
#
# That gap had a measurable cost. In the 2026-07-21 eval one cloud worker
# declined to answer on half its grounded runs — writing meta-commentary about
# its own search quality ("the gathered material is insufficient... I cannot
# fabricate it") instead of a draft — while its panel-mates, holding evidence of
# the SAME size on the SAME question, wrote complete answers. On
# `deep-pythagoras` the refusing worker held 6,668 chars of search results and
# the worker that wrote a full draft held 6,719. The evidence was there; the
# disposition was the problem, and there was nowhere to address it.
#
# Overridable via `agentic.prompts.deep_worker`.
DEEP_WORKER_SYSTEM = (
    "You are one of several workers answering the same request in parallel. "
    "Your output is a DRAFT — a later pass merges the drafts into the user's "
    "final answer. Write the draft as a direct answer to the request.\n"
    "- Ground checkable claims with the tools available, then write. Thin "
    "search results do not mean you know nothing: state what you know reliably "
    "as plain fact, and let retrieval sharpen the dates, names, and specifics.\n"
    "- Do NOT withhold the whole answer because one part is unverified. Write "
    "what your evidence and knowledge support, and mark a specific gap in one "
    "line where it falls. Declining to answer contributes nothing to the merge; "
    "a partial draft with an honest gap contributes a lot.\n"
    "- Do NOT write about your own search process. How many results you got, "
    "how useful they were, or what someone would need to do for a better "
    "answer is not part of the answer — and another worker's evidence may "
    "already cover the gap you are describing.\n"
    "- Do NOT invent facts, sources, dates, or URLs to fill a gap. An honest "
    "gap is fine; a fabricated specific is not. This rule outranks the ones "
    "above: write more, but never write what you do not have.\n"
    # Deliberately NOT naming a page-opener tool here. Deep workers only have
    # web_search + kb_search + memory — there is no web_fetch/read_url. Naming
    # one (as an earlier draft did, and as RESEARCHER_SYSTEM still does) invited
    # models to CALL it: the 2026-07-21 143043 eval showed every deepseek deep
    # run firing 2–3 `web_fetch` calls that failed as unknown_tool, where the
    # prior prompt-less run made zero. A reassurance about a nonexistent tool
    # manufactures the very dead calls it reassures about.
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
    "- PLAIN BY DEFAULT: state what you know confidently plainly, as "
    "established fact — do not soften well-known, well-established facts into "
    "vague 'possibilities' just because a search came back thin. Reserve "
    "hedging for the two cases above: a specific checkable claim that only a "
    "tool-free draft vouches for, or a genuine factual conflict between drafts. "
    "If the drafts couldn't verify a specific detail (a result, a score, a "
    "date), say so plainly for THAT detail and decline to guess it — but keep "
    "the rest of the answer confident. A confident known fact plus an honest "
    "'I couldn't verify the specifics' beats a wall of hedged maybes.\n"
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
    "sources. When a search surfaces an authoritative or primary source (an "
    "official project page, release notes, the vendor's own announcement, the "
    "original paper), prefer a targeted search for THAT over running more broad "
    # "page", not "source": a vaguer noun stopped the model discriminating. Across
    # the 2026-07-22 protocol pair, "source" dropped official+academic citations
    # 56% -> 39%, and tech-transformer-attention swapped three direct arXiv
    # abstracts for blog explainers ABOUT those papers. "Official page" resolves to
    # something concrete; "official source" admits a well-written write-up.
    "queries — for releases, dates, and specs, one official page beats a handful "
    "of secondary write-ups or SEO results. As you search, capture each "
    "key fact you find — with its source URL — into your own running notes right "
    "away; your notes are your working memory across rounds, so a fact written "
    "into them is one you can still build on later even after many more tool "
    "calls. A web_search result already carries a usable title and URL in its "
    "snippet: that URL is a citable source on its own — cite the URL the search "
    "returned rather than dropping the source or falling back to \"no sources\". "
    "Report what you found as "
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
    # This used to read "If you used no tools, omit the section." — a trailing
    # conditional at the end of a long prompt, and the one instruction here that
    # licenses omitting the block. 2026-08-13 base-rate sweep over 107
    # researcher-notes blocks in 8 archived runs: deepseek-v4-pro omitted it
    # 0/36, glm-5.2 4/36, and the local qwen3.6:35b **16/35**. Every miss had
    # used tools, so the escape hatch never legitimately applied — but a 35B
    # model reading it at the very end, right after two "do not cite inline"
    # instructions, drops the block roughly half the time. A worker's whole
    # claim set then reaches the ledger unsourced and gets hedged or dropped.
    # An unconditional header with an explicit empty form removes the choice.
    # ⚠️ Re-measure with the archive sweep before changing this again; the noun
    # tuning above was won the same way.
    "citations. ALWAYS end with this section, even when it is empty: if you "
    "genuinely used no tools, write exactly `SOURCES: none`. Never omit the "
    "heading."
)

# Phase 26 Stage 1: a second, mechanical pass that converts a researcher's prose
# notes (and the sources they cited) into the structured ledger. Kept separate
# from RESEARCHER_SYSTEM so the researcher's *reasoning* is unchanged — this only
# re-expresses what they already found as claims+sources. The call is pinned to
# the ResearchResult JSON schema (Ollama `format`), so this prompt only needs to
# steer the content, not the shape. Overridable via
# `agentic.prompts.research_structure`.
RESEARCH_STRUCTURE_SYSTEM = (
    "You convert a researcher's notes into a structured claim/source ledger. "
    "Do NOT add facts, do NOT research — only re-express what the notes already "
    "contain. For each load-bearing factual claim, write a `Claim` with its "
    "`text`, the `source_ids` that back it, and a `risk` rating: mark risk "
    "\"high\" for genuinely contestable specifics — recent events or "
    "current-status claims, contested or single-source attributions, "
    "rankings and superlatives (\"first\"/\"only\"/\"invented\"/\"proved\"), "
    "vendor benchmarks, and specifics the notes themselves question. A "
    "widely-reproduced, uncontroversial textbook fact is \"medium\" or "
    "\"low\" even when it is a date, an edition, or an authorship — reserve "
    "\"high\" for claims a careful editor would actually want re-checked. "
    "Set `needs_hedge` true (with a short `hedge_reason`) ONLY when the notes "
    "flag that specific claim as uncertain, disputed, approximate, or "
    "legendary. A session-level sourcing caveat (\"searches returned "
    "nothing\", \"drawn from training data\", \"could not verify this "
    "session\") is NOT per-claim doubt — do not propagate it onto every "
    "claim; the pipeline already knows the grounding state from the sources "
    "themselves. For "
    # ⚠️ THE ID RULE. Measured over the three protocol runs of 2026-08-14 (34
    # structured drafts): a model would emit its sources numbered `s1`, `s2` and
    # then cite `src-corrode` / `src_microsoft_phi4` — two id schemes inside one
    # JSON object, zero overlap, so EVERY claim in that draft reached the writer
    # unsourced. qwen3.6:35b did it in 5 of 12 drafts, deepseek 1 of 11, glm-5.2
    # 0 of 11; it was all-or-nothing every time, never partial. The user message
    # now leads with a real `s1..sN` catalogue of what the researcher actually
    # retrieved, so the ids exist before the model starts writing and it has
    # nothing left to mint. Re-measure with the archive sweep before touching this.
    "When the request begins with a SOURCES RETRIEVED list, those ids are the "
    "ONLY ids you may use: reuse each `sN` exactly as written, for both the "
    "`Source.id` and every `source_ids` entry that refers to it. Never renumber "
    "them and never invent an id of your own — a claim citing an id that is not "
    "in that list is discarded and its evidence is lost. For "
    "each source the notes cite, write a `Source` with its `title`, `url`, and "
    "`source_type` — use \"company_claim\" for a vendor's own benchmark or "
    "marketing assertion (not independent fact), \"official\" for vendor docs / "
    "release notes, \"reference\" for Britannica/MacTutor/encyclopedias, and so "
    "on. LINK EACH CLAIM to the source(s) that back it: if the notes attribute "
    "a claim inline, use that; and when a claim clearly rests on a source the "
    "notes list (e.g. the notes end with a `SOURCES:` block and the claim "
    "restates that source's content), set its `source_ids` to that source even "
    "if the note didn't repeat the citation next to the sentence — the sources "
    "are already in the notes, so wiring a claim to the one it came from is "
    "re-expressing, not inventing. Only leave `source_ids` empty when the notes "
    "genuinely carry no source for that claim (e.g. a pure-reasoning step, or "
    "the researcher explicitly worked from memory with no `SOURCES:` list). Do "
    "NOT fabricate a URL or attach an unrelated source to make a claim look "
    "grounded. Put any leftover prose in `summary_notes`."
)

VERIFIER_SYSTEM = (
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

FACTCHECK_SYSTEM = (
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

# Phase 26 Stage 2: converts the fact-checker's findings into a structured
# FactCheckResult keyed to the claim ledger. Run after the fact-checker's
# web_search ReAct loop (the loop can't also be schema-pinned), with the claim
# list in the prompt so every verdict references a real claim_id. Pinned to the
# FactCheckResult JSON schema (Ollama `format`). Overridable via
# `agentic.prompts.factcheck_structure`.
FACTCHECK_STRUCTURE_SYSTEM = (
    "You convert a fact-checker's findings into a structured per-claim verdict "
    "list. You are given the CLAIMS (each with an id) and the fact-checker's "
    "notes. For each claim the notes actually address, emit a `ClaimCheck` with "
    "its `claim_id` and a `verdict`:\n"
    "  - \"supported\": the sources back the claim as stated.\n"
    "  - \"unsupported\": no source actually supports it (e.g. a work called "
    "\"surviving\" that is in fact lost) — the writer will OMIT it.\n"
    "  - \"conflicting\": sources or other claims contradict it.\n"
    "  - \"needs_hedge\": plausible but the exact form isn't earned — soften it; "
    "set `corrected_text` to the hedged wording.\n"
    "  - \"irrelevant\": not load-bearing; ignore.\n"
    "When a claim should change wording (a corrected value, a hedge, or "
    "attributing a vendor benchmark to the company), put the new wording in "
    "`corrected_text`. Apply these rules: official release dates/model "
    "names/licenses from official docs are NOT hedged unless sources conflict; "
    "company benchmark/performance claims must be phrased as company claims, not "
    "independent findings; ancient biography is hedged unless directly attested; "
    "disputed authorship uses \"attributed to\", not \"authored\"; claims using "
    "\"first\"/\"only\"/\"proved\"/\"invented\"/\"founded\"/\"definitively\"/"
    "\"worldwide\"/\"complete\"/\"all\" require strong support or get a softer "
    "`corrected_text`. EVERY claim in the list gets its own entry in `checks` — "
    "including a claim that contradicts another one. When two claims conflict, "
    "ALSO add one short SENTENCE to `fatal_errors` naming both ids and the "
    "conflict, e.g. \"w1_c19 dates the release to July 17 but w1_c46 says June\". "
    "`fatal_errors` holds sentences, never bare claim ids, and is never a "
    "substitute for a verdict. Do not invent claim_ids — only reference ids from "
    "the list."
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

# Fires at `max_rounds` (react.py), so it is the LAST thing a budget-exhausted
# worker reads before writing — a strong position. The old closing clause ("If
# the gathered information is insufficient, say so explicitly") was true but
# unscoped, and read as permission to replace the answer with a note about the
# search. Every refusal in the 2026-07-21 eval arrived through this path
# (`tool_rounds == max_rounds`). Scoping the admission to the specific missing
# detail keeps the anti-fabrication intent without licensing a non-answer.
#
# NOTE: "using only the information already gathered" is deliberately retained.
# It is in tension with the fix — a worker with thin results may still read it
# as "you may not answer" — but relaxing it invites the parametric-hallucination
# failure the ledger and hedge work exist to prevent. If refusals survive this
# change, that clause is the next lever, and it needs its own measurement.
REACT_FINAL_ANSWER_USER = (
    "You have reached the tool-call budget. Do not call any more tools. "
    "Using only the information already gathered above, write the final "
    "answer to the original request now as plain prose. Where a specific "
    "detail is missing or unconfirmed, say so in one line for THAT detail and "
    "answer the rest — a note about what you could not verify is not a "
    "substitute for the answer, and do not narrate your search process. Do "
    "not fabricate facts, sources, or dates to fill a gap."
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


# Phase 42: the `audrey_video` task role. Deliberately short.
#
# ⚠️ Most of what a "video specialist" would say already lives in the
# custom-tools descriptions, which every model sees on every call and which
# therefore apply to `audrey_auto` users too: `list_my_files` covers
# pending/processing vs empty and "a catalogue, not contents", and
# `get_file_text` covers `next_offset` paging, `total_chars`, and refusing to
# invent content for a missing artifact. Instructing those again here would
# only re-state them for the minority of users who pick this model.
#
# What is left is the part no single tool description owns: the order to call
# them in, and what to do when a read came back partial or covers more than
# one file. Keep this prompt to that. If a behaviour can be fixed in a tool
# description instead, fix it there — it reaches every path.
VIDEO_SPECIALIST_SYSTEM = (
    "You are answering questions about videos and documents this user has "
    "uploaded. Their own files are the source; reach for them before anything "
    "else, and never answer about a file's contents from its name alone.\n"
    "When the user points at ONE recording without naming it — 'that video', "
    "'the one I uploaded yesterday' — call `list_my_files` first and use an "
    "exact filename from it. Never invent a filename. When several files could "
    "match that one reference, say which ones you found and ask, rather than "
    "picking one and answering confidently from the wrong video.\n"
    "A question about 'my videos' or 'my files' — plural, with no single one "
    "singled out — is NOT that case, and must not be narrowed to one file. "
    "Search across them. A topic question answered from one file, with the "
    "others left unmentioned, reads as complete and is not.\n"
    "`get_file_text` returns one page at a time. If you have read only part of "
    "a document, say which part before you characterise the whole. A summary "
    "built from the first page but presented as a summary of the video is "
    "worse than saying you ran out of room and offering to read on.\n"
    "When the user asks about more than one file, read each one separately "
    "and keep them apart in your answer. Never let what one file said stand "
    "in for another, and if you managed to read only one of them, say so."
)


# ─── Override loader ──────────────────────────────────────────────────

_OVERRIDE_SOFT_CAP_CHARS = 4000
_WARNED_OVERRIDES: set[str] = set()  # one warning per key per process

_PROMPT_KEYS = frozenset({
    "classifier",
    "planner",
    "deep_worker",
    "synthesizer",
    "researcher",
    "research_structure",
    "verifier",
    "factchecker",
    "factcheck_structure",
    "writer",
    "react_final_answer",
    "memory_store_hint",
    "chat_history_search",
    "video_specialist",
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


# ─── Task role by virtual model ───────────────────────────────────────

# Which virtual models carry a task-role system message, and the config key
# that overrides each. `audrey_auto` and friends deliberately have no entry:
# a general model gets no role prompt, which is what makes the specialist a
# measurable difference rather than a rename.
_TASK_ROLE_BY_VIRTUAL_MODEL: dict[str, tuple[str, str]] = {
    "audrey_video": ("video_specialist", VIDEO_SPECIALIST_SYSTEM),
}


def task_role_for(virtual_model: str, cfg: Any = None) -> str | None:
    """Return the task-role prompt for a virtual model, or None.

    None means "no role prompt" and is the normal case — only specialists
    have an entry in `_TASK_ROLE_BY_VIRTUAL_MODEL`.
    """
    entry = _TASK_ROLE_BY_VIRTUAL_MODEL.get(virtual_model)
    if entry is None:
        return None
    key, default = entry
    return prompt_from_config(cfg, key, default)


def without_task_role(
    messages: list[dict[str, Any]], role_prompt: str | None
) -> list[dict[str, Any]]:
    """The inverse of `with_task_role`, for the deep-vs-fast token gate ONLY.

    ⚠️ **Why this exists** (found 2026-08-09, from on-box logs):
    `complexity.count_tokens` sums every message, system ones included — and the
    task role is injected at the route, upstream of both gates. So a specialist
    paid ~330 tokens of its own prompt into a 500-token threshold, and turns that
    would have run fast went to the three-worker deep panel instead. Nothing said
    so: the log prints the post-injection count, so `576 tokens -> deep` reads as
    a legitimately large request rather than a 246-token one wearing a prompt.

    The effect is a silent retune. A fixed injection drops the specialist's real
    threshold to ~170 tokens of user content while `audrey_auto` keeps 500, which
    also means an A-B between them compares pipelines, not prompts — four of
    eight phase-42 cases got a panel and a synthesizer their control never saw.

    Only the routing decision is corrected. `prompt_tokens` and
    `count_tokens_by_role` keep counting the full context, because they report
    what was actually sent and that number is not wrong.

    Matched by exact content rather than a marker key: these dicts go to Ollama,
    and an extra field on a message is a wire-format risk for a gate detail.

    Returns a new list; `messages` is never mutated.
    """
    if not role_prompt:
        return messages
    return [
        m for m in messages
        if not (m.get("role") == "system" and m.get("content") == role_prompt)
    ]


def with_task_role(
    messages: list[dict[str, Any]], role_prompt: str | None
) -> list[dict[str, Any]]:
    """Insert a task-role system message AFTER any leading system messages.

    Same placement rule as `compose_system_messages`' `task_role` slot and as
    `deep_panel._with_worker_system`: whatever the user or OWUI sent at
    `role=system` stays first so their persona wins on tone, and the task role
    follows it.

    ⚠️ **Called from the route, not from a pipeline node, and deliberately so.**
    `compose_system_messages` is reached only from `node_memory_recall`, which
    returns early when memory is disabled and again when the user is not
    identified — filling the slot there would make a specialist's entire
    deliverable vanish in both cases with no error and no log line. The route
    is the one place both the streaming and non-streaming paths pass through,
    so injecting here also avoids the split where a specialist works only on
    requests that happened to stream.

    Consequence worth knowing: the datetime and memory system messages are
    *prepended* downstream, so they land ahead of this one. Order among system
    messages is not otherwise load-bearing.

    Returns a new list; `messages` is never mutated.
    """
    if not role_prompt:
        return messages
    idx = 0
    for m in messages:
        if m.get("role") != "system":
            break
        idx += 1
    return [
        *messages[:idx],
        {"role": "system", "content": role_prompt},
        *messages[idx:],
    ]


__all__ = [
    "CLASSIFIER_SYSTEM",
    "PLANNER_SYSTEM",
    "DEEP_WORKER_SYSTEM",
    "SYNTH_SYSTEM",
    "RESEARCHER_SYSTEM",
    "VERIFIER_SYSTEM",
    "FACTCHECK_SYSTEM",
    "WRITER_SYSTEM",
    "REACT_FINAL_ANSWER_USER",
    "MEMORY_STORE_HINT",
    "CHAT_HISTORY_SEARCH_SYSTEM",
    "VIDEO_SPECIALIST_SYSTEM",
    "prompt_from_config",
    "compose_system_messages",
    "task_role_for",
    "with_task_role",
    "without_task_role",
]
