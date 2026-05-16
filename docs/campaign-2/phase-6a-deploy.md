# Campaign 2 Phase 6a - Complexity gate investigation

Not a code-shipping phase. This is a structured probe to characterize
two related routing bugs that share a root cause: **Audrey's
complexity gate sums tokens across every message in the conversation,
so short, simple questions get routed to deep panel once the
conversation around them has accumulated enough mass.**

There are two distinct failure patterns:

- **Pattern 1 — tool-bloat within a conversation.** A fresh
  conversation, fast on turn 1; turn 1's ReAct loop emits multiple
  tool results; turn 2's short follow-up sees those tool results in
  the message list and tips over the gate. Transcript 1 below.
- **Pattern 2 — history accumulation across a long conversation.**
  A trivially short user message ("what is your system prompt") on
  turn N of a long-running chat routes deep because turns 1..N-1
  carry enough assistant/user content alone to exceed the threshold,
  with no tool involvement at all.

The fix candidates differ by pattern:

| Pattern | Fix candidates |
|---|---|
| 1 (tool bloat) | A (raise threshold), B (skip tool tokens), C (classifier override) |
| 2 (history accumulation) | A (raise threshold), D (weight last user message) |

The picking question is which combination matches actual usage data.
This doc lays out the probe sequence, expected shape of the answer,
and what numbers would push the decision one way or the other.

## What we already know

The complexity gate logic is in
[`pipeline/complexity.py:24`](../../src/audrey/pipeline/complexity.py#L24)
and [`pipeline/graph.py:200`](../../src/audrey/pipeline/graph.py#L200).
Today's `count_tokens` sums every message's content with no per-role
weighting.

Two transcripts captured on 2026-05-15 show the bug end-to-end.

### Transcript 1 — Pattern 1, casual web research (no PII)

User on `audrey_auto`: short question about BJJ guard variants, one
turn invoking web_search + kb_search, immediate follow-up.

```text
08:17:05  chat.completions (stream) model=audrey_auto tokens=48 mode=fast
08:17:18  classify: general (router:general, conf=0.95)
08:17:18  complexity: 230 tokens -> fast (tokens<500)
          [model answers, dispatches web_search x2 and kb_search]
08:18:23  classify: general (keyword:tool_mention:web_search, conf=0.95)
08:18:23  complexity: 1097 tokens -> deep (tokens>=500)
```

Two numbers worth pinning down:

- Route-layer `tokens=48` (raw user message) → graph-layer `230 tokens`
  on the same turn is the system-prompt offset that `compose_system_messages`
  prepends (datetime, memory, tool definitions). Roughly 180 tokens here.
- `230 → 1097` between the two turns is the bloat under investigation.
  User prompt on turn 2 was a short follow-up; the +867 jump is prior
  tool results plus prior assistant synthesis carried in conversation
  history.

### Transcript 2 — Pattern 1, chat_history_search follow-up

Same day, different conversation, `audrey_auto`. The model uses
`memory_search` and `chat_history_search` on one turn, then the next
turn routes deep before the user even sends another tool-heavy message.

```text
08:43:06  chat.completions (stream) model=audrey_auto tokens=773 mode=deep
08:43:08  dispatch: memory_search ok in 2.13s (98 chars)
08:43:43  dispatch: chat_history_search ok in 10.01s (99 chars)
          [next turn]
08:45:29  dispatch: memory_search ok in 0.06s (570 chars)
08:45:29  memory: user=... hits=0 keys=[] store_hint=on chat_history_hint=on
08:45:29  classify: general (keyword:tool_mention:chat_history_search, conf=0.95)
08:45:29  complexity: 996 tokens -> deep (tokens>=500)
08:46:26  deep_panel: pool=deep_panel workers=2 ok=2 ... attempted=['qwen3.6:35b', 'kimi-k2.6:cloud']
          ~57s wall-clock for the deep panel
```

Three things this transcript rules in or out:

- `memory: hits=0` means no memory recall content is inflating the
  count — the 996 tokens are conversation history alone.
- The classify reason is `keyword:tool_mention:chat_history_search`.
  That fires whenever the model name-drops the tool, not just when
  the user explicitly asks for it. Important for the Option C
  discussion: the signal can over-fire on synthesis turns where the
  model references a tool by name.
- Wall-clock cost: ~57s for a deep panel that the equivalent fast
  path with one ReAct round would handle in ~10s.

### Pattern 2 — long-conversation history accumulation

Live example reported 2026-05-15: in an ongoing `audrey_auto`
conversation that had been running for many turns, the user asked
"what is your system prompt." The complexity gate routed the turn
deep. The user message itself is trivially short (~6 tokens) and
clearly doesn't deserve deep panel; the prior conversation history
alone tipped the total over 500. No tool messages were involved on
the most recent turns.

This is the same root cause as Pattern 1 (the gate sums every message
in the list), but with a different mechanism: assistant + user
content from prior turns accumulates to the threshold even without
tool roundtrips. Option B (skip tool tokens) does nothing here
because there are no tool tokens to skip. The candidates that help
are A (raise threshold globally) and D (weight the latest user
message more heavily than prior history).

## The decision we need to make

Four candidate fixes, each with different implications. The probe
should produce enough data to pick one — or a small combination.

| Option | Change | When it wins | When it doesn't |
|---|---|---|---|
| **A. Raise threshold** | `config.yaml`: bump 500 → 1500 (or wherever the data lands). | Both patterns are statistical, not categorical — conversations regularly run 600-1200 tokens but rarely exceed 1500. | Long conversations or tool-heavy turns routinely cross any reasonable threshold (>2k tokens by turn 3). |
| **B. Exclude tool tokens** | `count_tokens` skips `role: "tool"` messages. ~4 line change in `complexity.py` + a test. Targets Pattern 1. | The bulk of Pattern 1's gate-crossing tokens are tool results. Excluding them brings most tool follow-ups back below threshold. | Doesn't help Pattern 2 at all. User prompts + assistant history also routinely cross 500 without tool baggage. |
| **C. Classifier override** | If strong keyword signal (e.g. `tool_mention:*`) AND task is `general`/`factoid`, override complexity gate back to fast. | Tool-mention turns are usually single-shot lookups, not synthesis-worthy. | Some tool-mention turns genuinely benefit from deep ("search the KB for X and Y, then compare them"), AND `tool_mention` can fire when the model — not the user — name-drops a tool (see Transcript 2). The Step 4 eyeball pass needs to check what fraction of `tool_mention` turns are user-initiated. |
| **D. Weight last user message** | Score the latest user message at full weight and prior history at a fractional weight (e.g. `score = last_user + history × 0.25`), or gate purely on `last_user` tokens. Targets Pattern 2. ~10 line change. | A short follow-up question in a long conversation is a fast-path task regardless of what came before — "what's being asked now" is the routing signal. | Users routinely paste long content into follow-ups; D would miss those and route them fast when they deserve deep. Tool-bloat Pattern 1 turns still have a short last user message, so D also helps them — but it's not the right *explanation* for the fix there. |

The probe has three jobs:

1. **Establish the prevalence.** What fraction of tool-using turns
   currently cross the threshold?
2. **Characterize the cause.** Of crossings, how much is tool tokens
   vs. user/assistant tokens?
3. **Check the counterfactual.** Under each option, how many
   currently-deep turns would have stayed fast — and would any of
   them have been wrong to stay fast?

## What the offline probe can and can't tell us

`scripts/probe_complexity_gate.py` replays today's gate against the
`chat_archive.db` SQLite. It is useful but bounded by what the archive
contains. The archive stores **only `role: "user"` and
`role: "assistant"` messages** ([`tools-server/chat_archive.py:61`](../../tools-server/chat_archive.py#L61)) —
tool result messages (`role: "tool"`) are never persisted. So:

- The probe's "current" sum and its "Option B (skip tool tokens)" sum
  are **identical** against this data source. Option B always reports
  zero flipped turns.
- The probe still answers a useful question: how often do user +
  assistant tokens alone cross the threshold? On the 2026-05-15
  archive snapshot, 9 of 25 turns (36%) crossed 500 tokens with no
  tool contribution at all. That's the "all conversations grow" floor
  Option A would need to clear.

For the load-bearing question — *how much do tool result messages
contribute to the gate input at runtime?* — the probe cannot answer.
The runtime gate sees the in-memory message list including tool
results; persistence strips them.

To measure that directly, Phase 6a adds an opt-in `complexity.log_breakdown`
config knob that emits per-role token sums on every gate decision.
Enable it for ~24h, scrape logs, then analyse.

## Probe sequence

Step 1 captures targeted live transcripts as a sanity check.
Step 2 is the offline archive replay (bounded as above).
Step 3 is the new live-instrumentation pass that actually measures
tool-token contribution. Steps 4-5 are the post-data eyeball checks.

### Step 1 — characterize one bad turn end-to-end

Already done for two transcripts (see "What we already know" above).
Both show the pattern: short user prompt, prior turn had tool
dispatches, follow-up complexity log lands in the 1000–1100 range.

To capture additional offenders, the useful log filter is:

```bash
docker compose logs --since 10m audrey-ai \
  | grep -E "chat.completions \(stream\)|memory:|classify:|complexity:|fast_path|deep_panel|dispatch:" \
  | tail -80
```

For each offender, record:

- The route-layer `tokens=N mode=...` line (pre-graph estimate).
- The graph-layer `complexity: N tokens -> deep` line (post system
  message composition).
- The classify reason (`router:*` vs `keyword:tool_mention:*`).
- The prior turn's tool dispatches.
- Whether `memory: hits=N` shows non-zero recall content adding to
  the count, or `hits=0` (history is all conversation).

If complexity shows 1500+ tokens, the prior synthesis is the real
bloat, not the tool result — note it.

### Step 2 — offline archive replay (Option A floor)

`scripts/probe_complexity_gate.py` replays today's gate against the
chat archive. As covered above, the archive contains no `role: "tool"`
rows, so the script's Option B column will always read zero. The
useful output is:

- The **histogram of `current` token counts** — tells us where the
  natural conversation-growth floor sits, with no tool contribution
  at all. This sizes Option A.
- The **per-user breakdown** — flags asymmetry between users.

`scripts/probe_complexity_gate.py` depends on tiktoken, so it cannot
run from the Unraid host shell (no `python3`, no `tiktoken`). Two
ways to run it that do work:

**Run on the laptop.** Copy the DB down with `scp`, then run inside
the audrey checkout's venv. Cleanest path because the venv already
has tiktoken.

```bash
# On the laptop, in the audrey repo:
scp <unraid-host>:/mnt/user/appdata/custom-tools/chat_archive.db /tmp/chat_archive.db
.venv/bin/python scripts/probe_complexity_gate.py /tmp/chat_archive.db
.venv/bin/python scripts/probe_complexity_gate.py /tmp/chat_archive.db --per-user
.venv/bin/python scripts/probe_complexity_gate.py /tmp/chat_archive.db --dump-flipped
```

**Run inside the audrey-ai container.** Useful if SSH-copying the DB
off Unraid is a hassle. Audrey-ai has both Python and tiktoken.

```bash
# On Unraid:
docker cp /mnt/user/appdata/custom-tools/chat_archive.db audrey-ai:/tmp/chat_archive.db
docker cp scripts/probe_complexity_gate.py audrey-ai:/tmp/probe.py
docker exec audrey-ai python3 /tmp/probe.py /tmp/chat_archive.db --per-user
```

The script does not write to the DB and does not network out, but it
holds the connection open for the duration of the scan, so prefer
running off a host copy rather than the live mount if the archive is
actively being written to.

The output has four sections:

1. **Aggregate counts** — turns total, deep today, deep under Option
   B, flipped (deep → fast). Each shown as both absolute count and
   percentage.
2. **Histogram** of `current` token counts in 250-token buckets.
   Buckets at or above the threshold are marked with `*`. This is
   what tells you where to set the threshold under Option A.
3. **Per-user breakdown** (if `--per-user`). One row per OWUI user
   with turn count, deep-today count, and flipped count. If one
   user's pattern dominates the other, note the asymmetry — a
   user-keyed config knob may be the right answer.
4. **Flipped turns list** (if `--dump-flipped`). One line per
   would-flip turn: user, current vs B token counts, conversation
   and message IDs, and the first ~120 chars of the user prompt
   for the Step 4 eyeball pass. **PII-bearing — do not paste this
   output into a shared doc.**

#### What the script does NOT include

Two omissions matter:

1. **Tool result messages are not archived.** The archive only stores
   user and assistant messages, so the script's Option B (skip tool
   tokens) always equals current. Step 3 below fills this gap with
   live instrumentation.
2. **System prompt and tool definitions are not archived.** The
   runtime gate sees what `compose_system_messages` prepends
   (datetime, memory, tool definitions). The script's totals are
   therefore lower than runtime by a roughly constant offset per turn
   (perhaps 200-400 tokens depending on memory recall and tool count).
   Use the histogram for shape, not absolute threshold comparisons.

### Step 3 — live per-role token breakdown

This is the load-bearing step. Phase 6a ships a config-gated log line
that emits per-role token sums on every gate decision. Enable it on
Unraid, exercise the bot for ~24h of normal use, then scrape the logs.

#### Enable the breakdown log

In `config.yaml` (Unraid deploy copy), under the `complexity:` block,
flip `log_breakdown: false` to `true`:

```yaml
complexity:
  token_threshold: 500
  log_breakdown: true   # Phase 6a investigation; flip back to false when done
```

Rebuild and restart audrey-ai for the change to load:

```bash
# On Unraid, /mnt/user/appdata/audrey_ai_2.0:
docker compose up -d --build audrey-ai
```

Verify the log line is firing:

```bash
docker compose logs --since 5m audrey-ai | grep "complexity.breakdown:" | tail -5
```

Expected shape (one line per gate decision, paired with the existing
`complexity:` line):

```text
audrey-ai | ... complexity: 996 tokens -> deep (tokens>=500)
audrey-ai | ... complexity.breakdown: assistant=412 system=180 tool=358 user=46 last_user=12
```

Two distinct signals on the breakdown line:

- **Per-role keys** (`system`, `user`, `assistant`, `tool`, or `other`).
  `user=` sums every user message in the conversation. Drives the
  Option B (skip tool) and Option A (threshold) analyses.
- **`last_user=`** sums only the most recent user message. Drives the
  Option D analysis: how often is the last user message tiny while the
  total is large?

#### Run for ~24h of normal use

Just use Audrey normally — short queries, follow-ups, KB lookups,
chat history searches. The point is to capture the natural mix of
turn types, not to engineer specific examples. Two real users in
OWUI is enough sample.

#### Scrape and analyse

From Unraid:

```bash
docker compose logs --since 24h audrey-ai \
  | grep -E "complexity:|complexity.breakdown:" \
  > /tmp/complexity_log.txt
```

Copy `/tmp/complexity_log.txt` down to the laptop and pair adjacent
lines. The two lines fire in sequence, so a simple `paste` works:

```bash
# On the laptop:
scp <unraid-host>:/tmp/complexity_log.txt /tmp/complexity_log.txt
grep "complexity:" /tmp/complexity_log.txt > /tmp/c_total.txt
grep "complexity.breakdown:" /tmp/complexity_log.txt > /tmp/c_break.txt
paste /tmp/c_total.txt /tmp/c_break.txt | head -20
```

What to compute by hand or with awk:

- **Mean `tool=` contribution across all turns.** If it averages 300+
  tokens per turn, tool tokens are a real bloat source — Option B is
  in the running for Pattern 1.
- **Of currently-deep turns** (`-> deep (tokens>=500)`), what fraction
  drop below 500 with the `tool=` sum subtracted? (Option B
  effectiveness against Pattern 1.)
- **Of the same currently-deep turns**, what's the assistant+user
  contribution alone? If it routinely tops 500 without help from
  `tool=`, Option B can't save them — that's the Pattern 2 signature.
- **Mean `last_user=` on currently-deep turns.** If short last-user
  messages (say, ≤50 tokens) routinely route deep while the total is
  large, Option D is in the running.
- **Of currently-deep turns where `last_user < 50`**, what fraction
  would route fast under D (e.g. `last_user + (total - last_user) × 0.25
  < 500`)? This is the Pattern 2 fix sizing.

#### When you're done

Flip `log_breakdown` back to `false` and redeploy. The log line is
cheap but noisy.

### Step 4 — eyeball the would-flip turns

The breakdown data is anonymous (token counts only, no message
content). For the turns where Option B would flip, cross-reference
the timestamp to OWUI's chat view and ask:

- Is the user prompt obviously a one-shot question? ("look up X")
- Or is it a multi-part question that genuinely deserved deep?
  ("compare X and Y from the KB")

If 90%+ of flipped turns are one-shot, Option B is right. If a
meaningful fraction are multi-part and would have wanted deep, Option
B is over-aggressive and we want Option A (a threshold raise that
catches the worst cases but leaves middle-weight conversations alone).

### Step 5 — sanity-check the deep verdicts

For a handful of conversations that legitimately needed deep (long
pasted documents, multi-source synthesis, etc.), confirm Option B
would still route them to deep. The expectation: user-message tokens
+ assistant-message tokens alone should still cross 500 on these.

If a legitimate "complex" conversation has its complexity carried
entirely by tool results, Option B would break it. Unlikely but worth
checking.

## Decision criteria

After steps 1-5, the call goes. The patterns are independent — you
may pick one fix that addresses both, or stack two fixes (e.g. B + D).

- **Option B wins for Pattern 1** if: ≥70% of currently-deep turns
  with non-trivial `tool=` drop below 500 with `tool=` subtracted,
  AND the flipped turns are mostly one-shot questions, AND step 5
  doesn't surface a legitimate-deep case that Option B would break.
- **Option D wins for Pattern 2** if: a meaningful fraction of
  currently-deep turns have `last_user < 50` (i.e. short follow-ups),
  AND a weighted-history scoring scheme would route them fast, AND
  step 5 doesn't surface a long-paste case that D would break.
- **Option A wins** if: even with `tool=` subtracted AND `last_user`
  weighted, most follow-up turns still cross 500. The signal in the
  data is "all conversations grow," and the right answer is to raise
  the threshold to wherever the 75th percentile sits.
- **Option C wins** if: Option B over-flips (turns legitimately
  benefiting from deep get pushed to fast), AND the tool-mention
  classifier signal turns out to be a good predictor of "one-shot
  question." Transcript 2 is the cautionary case — `tool_mention`
  can fire on synthesis turns where the model name-drops a tool.
- **Combined B + D** wins if: Pattern 1 and Pattern 2 are both
  prevalent (~30%+ of deep turns each) and the fixes are
  independent (B targets the tool sum, D targets short-follow-up
  scoring). They don't conflict mechanically.
- **Do nothing** if: <30% of currently-deep turns are tool-influenced
  AND short-follow-up-in-long-history, AND the wall-clock impact is
  tolerable. Both bugs real but rare.

## Followups to ship if probe confirms

These are the deliverables once the probe data is in. They would be
Phase 6b. (`scripts/probe_complexity_gate.py` and the
`complexity.log_breakdown` config knob already ship as part of 6a —
they are the probe itself, not a followup.)

1. **The chosen fix.** One or two of:
   - `config.yaml` edit (Option A, ~1 line).
   - `complexity.py` patch + test (Option B, ~10 lines + a test).
   - `graph.py` `node_complexity` patch + test (Option C, ~15 lines +
     a test).
   - `complexity.py` weighted-history scoring + test (Option D,
     ~15 lines + a test). Likely also exposes a `history_decay`
     config knob for tuning.
2. **Phase 6b deploy doc** with smoke tests that confirm:
   - The BJJ-shaped query now routes fast on its second turn.
   - A long pasted document still routes deep.
   - End-to-end ReAct still works on the fast path.
3. **Disable the breakdown log.** Flip `complexity.log_breakdown`
   back to `false` in `config.yaml` once Phase 6b ships.
4. **Memory note update.** The "math classifier audit" followup in
   PROJECT_STATE.md should be expanded to "math + tool-bloat
   audits" since both are complexity-gate concerns.

## Out of scope for 6a

- Writing or running the probe analysis. That's the work itself, not
  the test plan.
- Changing the `react_max_tool_chars` knob to make tool results
  smaller. Smaller tool results reduce the bloat but don't fix the
  underlying gate-too-aggressive question.
- Investigating reflect-loop `attempts>1` behavior. Separate
  mechanism; mentioned in Phase 6's logs but unrelated.
- Anything OWUI-side. The bug is entirely in Audrey's routing.

## Operational notes during the probe

- The offline probe is read-only. Nothing changes on Unraid; the only
  side effect is reading the SQLite archive.
- The live breakdown log emits **token counts only, no message
  content**. It is safe to share aggregates.
- Cross-referencing flipped turns to OWUI chat content for Step 4
  involves PII. Keep that local; do not paste chat content into a
  shared doc.
