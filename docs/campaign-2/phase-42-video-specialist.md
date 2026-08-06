# Campaign 2 Phase 42 — a virtual model that is good at videos

Phase 40 gave the model two tools: `list_my_files` for exact filenames and
`kb_search(filename=, artifact=)` to search inside one. Using them well takes a
**two-step sequence** — list, then search scoped — and nothing tells a general
model to do that. It has to work it out per conversation.

**Status: PLANNED, and gated. Do not start before phase 40 §3b has been run.**
The prompt written here is the fix for whatever §3b finds. Writing it first
means guessing at the problem.

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

**"Give me the transcript" — and that turned out to be a real question people
ask.** Observed 2026-08-05: asked for the transcript of a named video, the
model returned a partial excerpt and invented a mechanism to excuse it ("system
limitations … request it again in a new session"). No prompt fixes this,
because `kb_search` returns at most 20 chunks and **no route serves a whole
file** — the `.transcript.txt` sidecar is written on ingest and never read
back. A specialist prompt aimed at this gap would produce a better-worded
version of the same wrong answer.

That is a separate build — an artifact-read route plus a `get_file_text` tool,
whose hard part is the context budget rather than the plumbing — and it is
tracked in `PROJECT_STATE.md`. Worth knowing here because it is the most
obvious thing to ask a "video specialist", and shipping one that still cannot
answer it would look like the specialist is broken.

**`audrey_auto` users get none of it.** If the general path over-scopes or
fails to chain the two tools, that is a tool-description problem and has to be
fixed there. A specialist is additive — it is not a substitute for the default
path being correct, and shipping one is not a reason to close §3b.

**It is a dropdown entry.** Someone has to pick it. Discovery is the same
problem the phase-40 banner only partly solved.

---

## Steps

1. **Run phase 40 §3b first.** Three prompts, two videos. The result decides
   whether the prompt's emphasis is "scope more" or "scope less", and those are
   opposite instructions.
2. Add `audrey_video` to `VIRTUAL_MODELS` and route it like `audrey_auto` —
   adaptive, not forced deep. A retrieval question is not a panel question, and
   forcing deep puts an ordinary lookup on paid inference.
3. Fill the `task_role` slot with `VIDEO_SPECIALIST_SYSTEM` in
   `pipeline/prompts.py`.
4. Confirm the picked model is in `fast_path.tool_capable_models` — a
   specialist that cannot call tools is a model answering from nothing, which
   is exactly what `audrey_passthrough` already does and why phase 40 warned
   against asking it about a video.

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
