# Campaign 2 Phase 42 — a virtual model that is good at videos

Phase 40 gave the model two tools: `list_my_files` for exact filenames and
`kb_search(filename=, artifact=)` to search inside one. Using them well takes a
**two-step sequence** — list, then search scoped — and nothing tells a general
model to do that. It has to work it out per conversation.

**Status: BUILT on the laptop 2026-08-08, NOT deployed, NOT evaluated.**
Tests and ruff pass; nothing about behaviour is known yet. The §3b gate is
CLEARED — run 2026-08-06.

⚠️ **A third re-scope happened during the build. Read "What the tool
descriptions already say" below before judging the prompt** — the three
behaviours step 1 called "unmeasured and instructed nowhere" turned out to be
instructed, in the `list_my_files` and `get_file_text` descriptions, by commits
that landed the same day step 1 was written. The prompt that shipped is
therefore much smaller than step 1 anticipated, and the honest expectation for
the A-B-A is that it shows little or nothing.

§3b's finding is that **scoping already works unprompted**, so the prompt this
phase writes must NOT be about scoping. Verified from `kb.query` log lines, not
from reading answers:

- a named file scopes, and picks the right artifact —
  `scope=file='jasonRetirement.mp4'(1 id) artifact=transcript`;
- "what do my videos say about X" stays unscoped and uses `kb_search`;
- a two-file comparison does not scope to one of them.

What §3b actually surfaced were three bugs, all now fixed and deployed: a `vl`
misroute to a tool-blind model, a video demoted to `kind='text'` on every boot,
and a fabricated description of an empty file. **None was a prompt problem**,
which is the strongest argument this phase has for staying small — the general
path is now correct on all three prompts, so a specialist has to earn its keep
against a working baseline rather than against a broken one.

---

## Correcting phase 40

Phase 40 says, flatly, **"No `audrey_video` virtual model."** That entry is
about **ingest**, and as an ingest argument it is right and should stand: a
chat-completions endpoint cannot carry bytes, OWUI's own uploads go to OWUI's
storage, and a chat body still meets the 100 MB edge cap that phase 32 exists
to dodge.

It says almost nothing about a **retrieval** specialist, and it was read as
though it did. As a retrieval path the question is not "can a chat endpoint
receive a video" — it is "does a focused prompt use the phase-40 tools better
than an unfocused one", which is an ordinary, measurable question.

## What already exists

Almost all of it, which is why this is a small phase:

| piece | state |
|---|---|
| `VIRTUAL_MODELS` in `routes/openai/routes.py` | a static tuple; adding a name is mechanical |
| the `task_role` slot in `compose_system_messages` | built in phase 2a for exactly this, **never filled on this path** |
| `docs/plans/specialist-prototype-plan.md` | designs this pattern already — domain prompt plus KB-biased search |
| `list_my_files`, `kb_search(filename=, artifact=)`, the `notice` field | phase 40, deployed |

What is missing is the routing entry and the prompt.

## The prompt is the whole deliverable

Everything else is wiring. The behaviours worth naming, each of which is a
real failure seen or predicted:

- **List before searching.** When the user refers to a video without naming a
  file — "that recording", "my standup" — call `list_my_files` first and use
  an exact filename from it. Never guess one. The `notice` on a miss already
  names the alternatives, so a wrong guess costs a turn rather than an answer,
  but not guessing costs nothing.
- **Scope only when the user pointed at one file.** This is phase 40 §3b. A
  filter applied wrongly is worse than no filter, because scoping to the wrong
  video answers confidently from the wrong source and the answer looks sourced.
- **Never scope a comparison.** "What did the two videos say about X" scoped to
  one file answers about one and calls it both.
- **Use `artifact` for what it means.** "What did they say" → `transcript`.
  "What was on screen", slides, signs, on-screen text → `visual`.
- **Say when something is still processing.** `list_my_files` returns
  `waiting_for_s`; a video mid-ingest is not an empty video, and reporting it
  as one is the worst available answer.

## What this does NOT fix

**"Give me the transcript" — and no prompt can fix it.** Observed 2026-08-05:
asked for the transcript of a named video, the model returned a partial excerpt
and said it could "only provide a partial excerpt" because of "system
limitations".

**It was telling the truth.** `agentic.react.max_tool_result_chars` is 2000 on
the fast path, and a transcript chunk is ~992 chars — so a `kb_search` result
keeps **~1.7 hits regardless of `top_k`**, and the rest becomes a bare
`…[truncated]` the model has to interpret. Raising `top_k` returns no more
text. A specialist prompt aimed at this gap would produce a better-worded
version of the same truncated answer, and telling a specialist to "ask for more
chunks" would make it strictly worse by burning rounds for nothing.

The fix is a budget and a marker that says what was lost, then an artifact-read
route — tracked in `PROJECT_STATE.md`, and **it should land before this phase
is evaluated**, or the eval measures the cap rather than the prompt. Worth
knowing here because "what did they say" is the most obvious thing to ask a
video specialist, and one that answers it from 1.7 chunks looks broken.

**`audrey_auto` users get none of it.** If the general path over-scopes or
fails to chain the two tools, that is a tool-description problem and has to be
fixed there. A specialist is additive — it is not a substitute for the default
path being correct, and shipping one is not a reason to close §3b.

**It is a dropdown entry.** Someone has to pick it. Discovery is the same
problem the phase-40 banner only partly solved.

---

## Steps

1. ~~**Run phase 40 §3b first.**~~ ✅ **DONE 2026-08-06 — and it removed most
   of this phase's planned prompt content.** Scoping works unprompted
   (`scope=file='jasonRetirement.mp4'(1 id)` on a named file, `scope=none` on
   "what do my videos say"), and artifact selection works unprompted
   (`artifact=transcript` for "what did they say"). Three of the five
   behaviours under "The prompt is the whole deliverable" are therefore
   already true without a prompt, and instructing them again is wording
   applied to something the code already gets right.

   Also since: the **`list_my_files` summary removal** (2026-08-07) fixed the
   answer-from-metadata reflex structurally, and the **truncation budget**
   (6000 + `get_file_text` paging) landed, which the "What this does NOT fix"
   section says must precede evaluation. Both are deployed.

   ⚠️ **So re-scope the prompt before writing it.** What is left is what
   nothing has been measured on: reading a long transcript **across pages**
   (`offset`/`next_offset`, new and uninstructed anywhere), **comparisons now
   that summaries must be read rather than listed**, and **not reporting a
   still-processing video as empty** (`waiting_for_s`). Writing the original
   five-bullet prompt now would mostly re-state solved problems and dilute the
   two or three instructions that would actually do something.
   ✅ **DONE 2026-08-08 — and re-scoped again on the way.** See "What the tool
   descriptions already say" above: all three surviving behaviours were
   already instructed too. The shipped prompt is three paragraphs covering
   only cross-tool ordering, partial reads, and multi-file answers.
2. ✅ **DONE.** Add `audrey_video` to `VIRTUAL_MODELS` and route it like `audrey_auto` —
   adaptive, not forced deep. A retrieval question is not a panel question, and
   forcing deep puts an ordinary lookup on paid inference.
3. ✅ **DONE, but NOT by filling the `task_role` slot** — see "How it was
   wired" above. Both traps below are why the injection happens at the route
   instead. `VIDEO_SPECIALIST_SYSTEM` lives in `pipeline/prompts.py`.

   ⚠️ **Two traps, found 2026-08-07 while scoping this. Read before wiring.**

   **(a) `compose_system_messages` is only called from `node_memory_recall`,
   and that node returns early — twice.** `graph.py:186` returns `{}` when
   memory is disabled, and `:189` returns `{}` when the user is not
   identified. Filling `task_role` inside that node means the specialist's
   entire deliverable — the prompt — **silently does not land** in either
   case. Not an error, not a log line: a specialist that behaves exactly like
   `audrey_auto` and looks like the prompt was ineffective. The prompt has to
   be composed on a path that does not depend on memory being on or the user
   being known.

   **(b) There is a second, parallel implementation.** `_phase_thinking` in
   `routes/openai.py` runs the same datetime + memory + planner sequence
   inline for the streaming deep path, and `graph.py:168-172` already warns
   that the two must be changed together. Wiring `task_role` in one gives a
   specialist that works on one path and not the other, split by whether the
   request streamed — which is not a distinction any user can see.

   `state["virtual_model"]` is available in both, so the routing value itself
   is not the problem; where the composition happens is.
4. ▶ **OPEN — needs the box.** Confirm the picked model is in `fast_path.tool_capable_models` — a
   specialist that cannot call tools is a model answering from nothing, which
   is exactly what `audrey_passthrough` already does and why phase 40 warned
   against asking it about a video.

## What the tool descriptions already say (found 2026-08-08, during the build)

Step 1 re-scoped the prompt down to three behaviours on the grounds that they
were "instructed nowhere". Reading `tools-server/app.py` before writing the
prompt showed all three are instructed — in the tool descriptions, which every
model sees on every call:

| step 1 said "uninstructed" | where it is actually instructed |
|---|---|
| paging a long transcript (`offset`/`next_offset`) | `get_file_text` description — `next_offset`, `total_chars`, "say how much you have read and how much remains" |
| not reporting a still-processing video as empty | `list_my_files` description — "'pending' or 'processing' … say so rather than reporting it as empty", plus `waiting_for_s` |
| comparisons now that summaries must be read | `list_my_files` — "how two files compare — requires get_file_text or kb_search"; `get_file_text` — the cross-page warning about details getting "mixed up between files" |

Those came from `ea02933`, `e332d38`, `2ab809b` and `e4ac231` — the same
2026-08-06/07 cluster as the truncation budget. Step 1 was written on 08-07 and
did not account for them.

**This is §3b's finding one level deeper, and it is now a pattern worth
naming:** every time someone sits down to write this specialist prompt, the
thing it was going to say has already been fixed somewhere that applies to all
users instead of only the ones who pick the specialist. A tool description
reaches `audrey_auto` too; a task-role prompt reaches only whoever chose the
dropdown entry. **When a candidate behaviour can be expressed as a tool
description, it belongs there, and this phase should keep shrinking.**

What survived into `VIDEO_SPECIALIST_SYSTEM` is only what no single tool
description owns, because it spans tools or spans a whole answer:

- **the call order** — list before reading when the user did not name a file,
  and ask rather than guess when several files match;
- **partial reads** — say which part you read before characterising the whole;
- **multi-file answers** — keep files apart, and say if you only read one.

Three short paragraphs. If the A-B-A shows nothing, the right conclusion is
that the tool descriptions were the specialist all along, and the name should
come back out of `VIRTUAL_MODELS`.

## How it was wired

Both traps in step 3 are avoided by **not** filling the `task_role` slot in
either composer. The role prompt is injected at the route
(`routes/openai/routes.py`, right after `messages` is built), which is the one
place the streaming and non-streaming paths both pass through and which runs
regardless of whether memory is enabled or the user is identified. Helpers are
`prompts.task_role_for` (virtual model → prompt, config-overridable under
`agentic.prompts.video_specialist`) and `prompts.with_task_role` (insert after
any leading system messages, matching `deep_panel._with_worker_system`).

Routing needed no code: `audrey_video` is absent from the forced-deep and
forced-fast lists in both gates, so it falls through to the token count exactly
like `audrey_auto`. `tests/test_virtual_model_routing.py` pins that, and pins
the two gates against each other — the sync hazard `graph.py` warns about in
prose is now a test.

One consequence to know: system-message order for this model is
`[memory, datetime, incoming system, task role, …]`, because datetime and
memory are *prepended* by later nodes. Order among system messages is not
otherwise load-bearing here.

Not changed, and pre-existing: a deep run of `audrey_auto` or `audrey_video`
logs `deep_panel: unknown virtual_model … falling back to default pool` because
neither is in `_POOL_KEYS`. That warning is noise on both, not new.

## Verification

**Same prompts as §3b, this time against `audrey_video`, and compare.** The
whole claim of this phase is "better than the general path at this job", and
that claim is only meaningful as a difference. A single run of the specialist
in isolation proves nothing.

Per the A-B-A rule, a prompt edit needs before/after/before — the metric's
historical spread on this codebase is wide enough that one run cannot separate
signal from noise.

## Rollback

A virtual model nobody selects is inert. Removing the name from
`VIRTUAL_MODELS` removes the feature.

---

## More information for later

**This is the specialist prototype, applied to a domain that now has content.**
`docs/plans/specialist-prototype-plan.md` has sat unstarted partly for want of
a first domain worth specialising on. Video is that domain: phase 41 will make
the corpus grow without anyone uploading anything, and video Q&A has a
genuinely different tool sequence from ordinary KB search, which is what makes
a specialist more than a system prompt with opinions.

**If it works, the generalisation is the config-driven `specialists:` block**
(Option 4 in that plan) rather than a second hand-coded model. Two hand-coded
specialists is the signal to build the general mechanism.

**Watch `complexity.deep_intent_phrases`.** "Thorough", "in depth",
"comprehensive" and "step by step" force the deep panel on a case-insensitive
substring match, so "give me a thorough summary of that video" routes to the
panel regardless of which virtual model was picked. That is not wrong, but it
means a specialist's fast-path prompt is not always the one that ran, and a
confusing eval result is worth checking against this first.
