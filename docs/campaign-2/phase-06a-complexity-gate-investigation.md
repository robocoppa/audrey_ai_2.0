# Campaign 2 Phase 6a - Complexity gate investigation

Phase 6a started as a probe ("why do follow-up turns route deep when
the user only typed a short question?") and ended up shipping two
targeted fixes after the diagnostic surfaced unexpected root causes.
This doc is the consolidated record of the investigation and what
changed.

## TL;DR

The original hypothesis — *prior tool results bloat the message list
past the threshold* — turned out to be only partly right. The
diagnostic instrumentation we shipped revealed two dominant causes
of misrouted-to-deep turns that the original framing missed:

1. **OWUI utility tasks** (Title Generation in particular) submit the
   entire chat history as one large user message starting with
   `### Task:`. Audrey's gate correctly saw "this is over 500 tokens"
   and routed to deep panel — but the work was a short-output utility
   summary that should never have gone through deep panel at all.
2. **Audrey's own banner/footer markup** (`> _Thinking_...`,
   `> _Planning_...`, `> _Tools used:_`) is part of the streamed
   response body. OWUI captures it as assistant content and ships it
   back as conversation history on the next turn. The gate counted
   that markup as real conversation context.

Both are now fixed:

- `is_owui_task_request(messages)` detects the `### Task:` prefix and
  forces fast in both streaming and graph gates.
  ([pipeline/complexity.py](../../src/audrey/pipeline/complexity.py),
  [routes/openai.py](../../src/audrey/routes/openai.py),
  [pipeline/graph.py](../../src/audrey/pipeline/graph.py))
- `_strip_audrey_markup()` filters blockquote (`>`) and `---`
  separator lines out of assistant message content before tokens
  are counted. Asymmetric — assistant role only; user pastes that
  happen to start with `>` are unaffected.

## Closure verification — 2026-05-17

After 24 hours of normal mixed-use on Unraid with the diagnostic
knobs on:

- **Six `owui_task` firings.** Token counts 592–1629, all
  correctly routed fast. Zero false positives (no legitimate user
  prompt was misclassified as a utility task).
- **Zero `tokens>=500 -> deep` events.** Every auto-gated turn
  routed fast. The complete population of complexity decisions
  ran 218–333 tokens — comfortably below threshold with margin.
- **`complexity.breakdown:` lines show clean `assistant=N` counts**
  on multi-turn conversations (sample: 55, 82, 105, 470, 567,
  1116, 1403). Short follow-ups after short prior responses count
  small (Fix 2 stripping working); longer prior responses count
  proportionally. No evidence of over-stripping or under-stripping.

Diagnostic knobs flipped back to `false` 2026-05-17.
`complexity.log_breakdown`, `debug.log_incoming_payload`, and
`debug.log_incoming_payload_content` are now off in the deployed
config; they remain in tree for future investigations.

The infrastructure built for the probe (the `log_breakdown` and
`log_incoming_payload*` config knobs and the
`scripts/probe_complexity_gate.py` script) remains in place for
future investigations.

## How the investigation actually went

This section is preserved chronologically because the wrong turns
were instructive.

### Stage 1 — Original framing

Phase 6 testing surfaced UX where a short follow-up like *"do you
think may is a good time to go there"* would routinely route to deep
panel after a previous turn had dispatched tools. The first hypothesis
was: *prior `role: "tool"` results are inflating the token count past
the threshold.*

Three candidate fixes were named:

- **Option A.** Raise the `complexity.token_threshold` config knob.
- **Option B.** Make `count_tokens` skip `role: "tool"` messages.
- **Option C.** Use the classifier's `tool_mention:*` signal to
  override the complexity gate back to fast.

The plan was to probe the chat archive offline to size whichever
option matched the data.

### Stage 2 — Offline probe runs, returns zero flips

`scripts/probe_complexity_gate.py` shipped (commit `2d426cb` of
2026-05-14) and was run against a 25-turn archive snapshot. Result:
**Option B reported 0 flipped turns**, contradicting the live log
evidence.

Cause: the chat archive only stores user and assistant messages
([`tools-server/chat_archive.py:61`](../../tools-server/chat_archive.py#L61)).
Tool result messages (`role: "tool"`) are never persisted. So the
script's Option B "skip tool tokens" calculation was a no-op against
the archive — there were no tool tokens to skip in the data source.

Useful finding nonetheless: 36% of archive turns (9/25) crossed 500
tokens *without any tool contribution at all*, on user + assistant
content alone. That established a "natural conversation growth"
floor — strong signal that whatever was happening, it wasn't only
about tool bloat.

### Stage 3 — Live breakdown instrumentation

To measure what the offline probe couldn't, we added an opt-in
config knob `complexity.log_breakdown` and emitted a
`complexity.breakdown:` log line at each gate decision showing
per-role token sums plus the most recent user message's token
count alone:

```text
complexity: 996 tokens -> deep (tokens>=500)
complexity.breakdown: system=209 user=875 last_user=875
```

What surprised us: `assistant=` and `tool=` keys **never appeared**
in any breakdown line. Every turn looked like `system=... user=...
last_user=...` and *`user` always equaled `last_user`*.

That couldn't be right for a real follow-up turn, so the next round
of instrumentation captured the actual incoming payload shape and
content.

### Stage 4 — The payload diagnostic

Two more config knobs shipped:

- `debug.log_incoming_payload` — emits the role + content-length
  list for each incoming request.
- `debug.log_incoming_payload_content` — emits the first 500
  characters of each message's content. PII-bearing; off by
  default.

The first capture revealed two things at once:

```text
incoming.payload: n=6 roles=[
    ('system', 75),
    ('user', 43),
    ('assistant', 372),
    ('user', 69),
    ('assistant', 0),     ← empty assistant turn
    ('user', 73)          ← duplicate of prior user with markdown bullet prefix
]
```

The empty assistant + duplicate-user pattern was traced to OWUI's
auto-task features (Follow Up suggestions specifically). The
content-head log on the assistant entry showed:

```text
'> _Thinking_...... ✅\n\n\n---\n\nThe current temperature in Istanbul ...
\n\n---\n> _Tools used:_\n> - **qwen3.6:35b** — `web_search`'
```

That's Audrey's own banner markup, emitted by
[`pipeline/banners.py`](../../src/audrey/pipeline/banners.py),
persisted in the conversation history that OWUI sent back. The
gate had been counting it as user-supplied complexity.

The second capture, after Follow Up was disabled in OWUI, gave the
decisive data point:

```text
incoming.payload: n=2 roles=[('system', 75), ('user', 2708)]
content head: "### Task:\nGenerate a concise, 3-5 word title with
an emoji summarizing the chat history..."
```

OWUI's **Title Generation** task issues a separate `/v1/chat/completions`
request to Audrey, with the entire conversation history packed into
a single user message preceded by an instruction template. That
2708-character payload becomes ~700 tokens at the gate, which routes
to deep panel.

Phase 6a's "follow-up turns route deep" symptom was, in significant
part, *not actually follow-up turns* — it was OWUI's title-gen
requests being routed through the same `audrey_auto` model and
correctly tripping the gate.

### Stage 5 — Two targeted fixes shipped

The real causes turned out to be orthogonal to the original Option
A/B/C/D framework.

#### Fix 1 — Detect OWUI utility tasks

OWUI templates all share a `### Task:` prefix on the user message:

- Title Generation: `### Task:\nGenerate a concise, 3-5 word title...`
- Tags Generation: `### Task:\nGenerate 1-3 broad tags...`
- Follow Up: `### Task:\nSuggest 3-5 relevant follow-up...`
- Autocomplete: `### Task:\nYou are an autocompletion system...`

`is_owui_task_request(messages)` in
[`pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py)
checks the most recent user message's prefix and returns `True`
when it matches. The streaming gate in
[`routes/openai.py`](../../src/audrey/routes/openai.py) and the
non-streaming `node_complexity` in
[`pipeline/graph.py`](../../src/audrey/pipeline/graph.py) both
short-circuit to fast when this fires — *before* the virtual-model
forced-deep/forced-fast checks. Even `audrey_deep` would route a
title-gen request fast.

Log signature when it fires:

```text
chat.completions (stream) model=audrey_auto task=...(...) tokens=2708 mode=fast owui_task=1
complexity: 2708 tokens -> fast (owui_task)
```

Test coverage in [`tests/test_complexity.py`](../../tests/test_complexity.py)
includes: title-gen detection, tags detection, leading whitespace,
normal user messages (must not trip), task keyword mid-message
(must not trip), latest-user-only scoping, multimodal content, and
the no-user-message edge case.

#### Fix 2 — Strip Audrey's own markup from assistant history

`_strip_audrey_markup(text)` in
[`pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py)
removes:

- Any line starting with `>` (banner blockquote markup, tools-used
  footer rows).
- Any bare `---` line (banner-to-body and footer-to-body separators).

Plugged into `_count_message_tokens` so every existing caller
(`count_tokens`, `count_tokens_by_role`, `is_complex`,
`count_last_user_tokens`) automatically sees the cleaned view.

**Asymmetry that matters.** Stripping is `role == "assistant"` only.
A user message that happens to contain a `>` blockquote (a markdown
paste, a user quoting an earlier banner) is counted in full. Tests
pin this asymmetry.

The visible response in the user's chat is unchanged — this only
affects what the gate sees when prior assistant messages return as
conversation history on follow-up turns.

Effect on a typical multi-turn:

- Per assistant turn: ~30-90 tokens shaved off (banner header
  10-30, tools-used footer 20-60, separators trivial).
- Across a 5-turn conversation with one tool call per assistant
  turn: ~150-450 tokens of pure cruft no longer counted.

## Evidence — captured transcripts

### Transcript A — title-gen routing to deep (pre-fix)

```text
10:07:33  incoming.payload: n=2 roles=[('system', 75), ('user', 2708)]
          content head (user): "### Task:\nGenerate a concise, 3-5 word title with an emoji
                                summarizing the chat history..."
```

Pre-fix outcome: routed to deep panel. ~30-60s wall-clock for a
3-5 word title summary.

Post-fix outcome:

```text
10:25:54  complexity: 716 tokens -> fast (owui_task)
```

Routed fast. Sub-second for the title generation.

### Transcript B — assistant banner persistence

```text
10:56:08  incoming.payload.content: [..., {
            'role': 'assistant',
            'head': '> _Thinking_....... ✅\n\n\n---\n\nThe next FIFA World Cup will
                     take place from **June 11 to July 19, 2026**, ...'
          }, {'role': 'user', 'head': 'who is the king of englad'}]
```

The user's actual follow-up is `who is the king of englad` (a few
tokens). The prior assistant text — the model's real answer plus
Audrey's banner header — is what was inflating the gate input.
After Fix 2, the gate counts only the model's real answer.

## What did NOT end up being the cause

Worth recording because the original framing pointed here and
investigation ruled them out:

- **Tool result messages bloating the message list.** OWUI does
  not pass `role: "tool"` messages back on follow-up turns at all.
  Audrey only ever sees `system`, `user`, `assistant`. Option B as
  originally framed would have been a no-op.
- **Long history accumulation in normal conversations.** Real
  multi-turn conversations *do* grow, but at a rate that — once
  banner markup is stripped from assistant content — rarely
  crosses 500 tokens in observed traffic. Option A (raise the
  threshold) is therefore not needed as a primary fix.
- **The classifier signal being a good fast-track predictor.**
  Option C wasn't pursued because Transcript 2 in the original
  doc showed `tool_mention:*` firing when the model name-dropped
  a tool rather than when the user requested one. Risky predictor.

## What's closed

- **Diagnostic config knobs.** Flipped back to `false` on 2026-05-17
  after 24 hours of clean data. See the Closure verification section
  at the top of this doc.
- **OWUI auto-tasks audit.** Applied 2026-05-16: Title Generation
  on (now routes fast via the `### Task:` detector), Tags / Follow
  Up / Autocomplete / Retrieval Query / Web Search Query / Image
  Prompt / Tools Function Calling all off. Task Model setting
  unchanged (would require making `audrey_fast` public; the
  detector covers it from Audrey's side anyway).
- **Empty-assistant + duplicate-user pattern.** Not observed in
  the 24h post-fix sample. Disabling Follow Up appears to have
  eliminated it.

## Followups tracked elsewhere

- **`chat_history_search` schema mismatch.** Captured at 22:20:45
  on 2026-05-15: model called the tool with `limit=20` but the
  Pydantic schema caps it at 10 (422 response). Not a Phase 6a
  issue; lives in `PROJECT_STATE.md` followups.

## Followups deferred from this phase

These were named in earlier drafts of this doc but did not ship
because the investigation moved the goalposts:

- **Option A — raise threshold.** Not needed in current data.
  Revisit if breakdown logs show genuine long conversations
  crossing 500 after fixes 1 and 2 are deployed.
- **Phase 6b deploy doc.** No separate phase needed; this doc is
  the deploy record.
- **`scripts/probe_complexity_gate.py`** stays in the tree. Its
  histogram is still useful for sizing a future Option A if it
  ever becomes necessary, and the script doesn't depend on the
  config knobs.

## Files touched during 6a

Code:

- [`src/audrey/pipeline/complexity.py`](../../src/audrey/pipeline/complexity.py)
  — added `count_tokens_by_role`, `count_last_user_tokens`,
  `is_owui_task_request`, `_strip_audrey_markup`. Plugged the
  stripper into `_count_message_tokens` so every counter sees the
  cleaned view automatically.
- [`src/audrey/pipeline/graph.py`](../../src/audrey/pipeline/graph.py)
  — `node_complexity` now consults `is_owui_task_request` first
  and emits a `complexity.breakdown:` line when the log flag is
  on.
- [`src/audrey/routes/openai.py`](../../src/audrey/routes/openai.py)
  — streaming gate mirrors graph behavior; adds incoming-payload
  diagnostic logs gated by the `debug.*` knobs.

Tests:

- [`tests/test_complexity.py`](../../tests/test_complexity.py)
  — 21 tests covering all of the above. Suite total 274 pass.

Config:

- [`config.yaml`](../../config.yaml) — `complexity.log_breakdown`
  and the `debug:` block. All default `false`; were flipped on
  during investigation on the Unraid deploy copy.

Scripts:

- [`scripts/probe_complexity_gate.py`](../../scripts/probe_complexity_gate.py)
  — offline archive replay. Kept in tree.

Docs:

- This file.
- [`docs/PROJECT_STATE.md`](../PROJECT_STATE.md) — phase summary
  updated.

## Operational notes

- The `debug.log_incoming_payload_content` log captures user
  message content. Treat its output as PII. The `incoming.payload`
  shape log (without `content`) is fine to share — it carries only
  role names and lengths.
- The `complexity.breakdown` log emits token counts only. Safe to
  share.
- All three knobs are independent and can be toggled separately.
  `complexity.log_breakdown` stays useful for future tuning even
  when the payload logs are off.
