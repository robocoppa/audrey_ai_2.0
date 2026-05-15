# Campaign 2 Phase 6a - Complexity gate investigation

Not a code-shipping phase. This is a structured probe to characterize a
suspected bug surfaced during Phase 6 smoke testing: **follow-up turns
in a tool-using conversation get routed to deep mode because prior
`role: "tool"` results inflate the message token count past the
`complexity.token_threshold`** (default 500).

The fix candidates are known (Option A: raise threshold, Option B:
exclude tool tokens, Option C: classifier override). The picking
question is which one matches actual usage data — not just the single
"tourniquet" example.

This doc lays out the probe sequence, expected shape of the answer, and
what numbers would push the decision one way or the other.

## What we already know

From 2026-05-13 log inspection:

```text
turn 1: 228 tokens → fast    (fresh conversation)
turn 2: 840 tokens → deep    (+612 after one kb_search round)
turn 3: 905 tokens → deep
turn 4: 790 tokens → deep
```

User prompt was short on every turn ("search my uploaded files and tell
me how to apply a tourniquet"). The ~600-token jump between turn 1 and
turn 2 is bigger than any user message — it comes from the prior tool
result (`react_max_tool_chars=2000` ≈ 500 tokens) plus the prior
assistant synthesis.

Wall-clock: ~60-90s per deep response vs. an expected ~10s on fast path
with one ReAct round.

The complexity gate logic is in
[`pipeline/complexity.py:24`](../../src/audrey/pipeline/complexity.py#L24)
and [`pipeline/graph.py:200`](../../src/audrey/pipeline/graph.py#L200).
Today's `count_tokens` sums every message's content with no per-role
weighting.

## The decision we need to make

Three candidate fixes, each with different implications. The probe
should produce enough data to pick one.

| Option | Change | When it wins | When it doesn't |
|---|---|---|---|
| **A. Raise threshold** | `config.yaml`: bump 500 → 1500 (or wherever the data lands). | Tool-using conversations regularly run 600-1200 tokens but rarely exceed 1500. Bug is statistical, not categorical. | Tool conversations routinely cross any reasonable threshold (>2k tokens by turn 3). |
| **B. Exclude tool tokens** | `count_tokens` skips `role: "tool"` messages. ~4 line change in `complexity.py` + a test. | The bulk of the gate-crossing tokens are tool results. Excluding them brings most follow-ups back below threshold. | User prompts + assistant history also routinely cross 500 even without tool baggage. |
| **C. Classifier override** | If strong keyword signal (e.g. `tool_mention:*`) AND task is `general`/`factoid`, override complexity gate back to fast. | Conversations where the user keeps invoking tools are usually single-shot lookups, not synthesis-worthy. | Some tool-mention turns genuinely benefit from deep (e.g. "search the KB for X and Y, then compare them"). |

The probe has three jobs:

1. **Establish the prevalence.** What fraction of tool-using turns
   currently cross the threshold?
2. **Characterize the cause.** Of crossings, how much is tool tokens
   vs. user/assistant tokens?
3. **Check the counterfactual.** Under each option, how many
   currently-deep turns would have stayed fast — and would any of
   them have been wrong to stay fast?

## Probe sequence

All steps are read-only — no code changes. Run on Unraid where the
chat archive SQLite lives.

### Step 1 — characterize one bad turn end-to-end

Pick a recent turn that went deep with `tool_mention` reason. From
audrey-ai logs:

```bash
docker compose logs --since 2h audrey-ai | grep -E "complexity:|classify:|chat.completions" | tail -40
```

For one offender, capture:

- The user message itself (from OWUI's chat view, or from the chat
  archive — see step 2).
- The `complexity: N tokens -> deep (tokens>=500)` line and N.
- The classify reason (`router:*` vs `keyword:tool_mention:*`).
- Whether the prior turn had a tool dispatch (`audrey.tools.dispatch:
  dispatch: ... ok`).

Expected: short user prompt, prior turn had a `kb_search` or similar
dispatch, complexity log shows 600-1200 tokens. If complexity shows
1500+ tokens, the prior synthesis is the real bloat — note it.

### Step 2 — sample the chat archive

The chat archive lives at `/app/data/chat_archive.db` inside the
custom-tools container, bind-mounted from
`/mnt/user/appdata/custom-tools/chat_archive.db` on the Unraid host
([`tools-server/settings.py:57`](../../tools-server/settings.py#L57),
[`compose.yaml:91`](../../compose.yaml#L91)). Read it directly off
the bind mount — no `docker compose cp` needed:

```bash
sqlite3 /mnt/user/appdata/custom-tools/chat_archive.db \
  "SELECT role, COUNT(*) AS n, AVG(length(content)) AS avg_chars, MAX(length(content)) AS max_chars
   FROM messages GROUP BY role ORDER BY role;"
```

Per-user breakdown:

```bash
sqlite3 /mnt/user/appdata/custom-tools/chat_archive.db \
  "SELECT user, role, COUNT(*) AS n, AVG(length(content)) AS avg_chars, MAX(length(content)) AS max_chars
   FROM messages GROUP BY user, role ORDER BY user, role;"
```

What we want:

- Distribution of `length(content)` by role across the archive. The
  question is how often a `role: "tool"` message is multi-hundred
  chars (≈ multi-hundred tokens) vs. small.
- Whether `role: "assistant"` content is also routinely long (would
  push deep even without tool tokens).
- Per-user breakdown — we have two users; their tool-use cadence
  may differ enough to want different defaults eventually.

### Step 3 — replay the gate against archived turns

This is the load-bearing step. For each user message in the archive,
reconstruct the message list **as it would have looked when classify
ran on that user turn** (the conversation up to and including that
message), and compute the token count both ways:

- `current`: tiktoken sum across all messages (today's behavior).
- `proposed_B`: tiktoken sum skipping `role: "tool"` messages.

Then tabulate: how many turns are `current >= 500 AND proposed_B <
500`? Those are the turns Option B would flip from deep to fast.

`scripts/probe_complexity_gate.py` does this. It depends on tiktoken,
so it cannot run from the Unraid host shell (no `python3`, no
`tiktoken`). Two ways to run it that do work:

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

The runtime gate sees the system prompt and tool definitions prepended
to the message list by `compose_system_messages`. Those are not in the
archive. The script's numbers will therefore be lower than runtime by
a roughly constant offset per turn (perhaps 200-400 tokens depending
on memory recall and how many tools are registered).

The **relative** delta between `current` and `proposed_B` is
unaffected — both sum the same set of archived messages. The
absolute threshold comparison shifts a bit, but the *shape* of the
data (how many turns are in each bucket, where the tool-bloat hump
sits) is what we need for the decision.

If the answer ends up sensitive to that constant offset (the
threshold sits right on a histogram cliff edge), add a tiktoken count
of the current `compose_system_messages` output for a representative
turn and shift the threshold input accordingly.

### Step 4 — eyeball the would-flip turns

The probe lists which turns Option B would route differently. For each:

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

After steps 1-5, the call goes:

- **Option B wins** if: ≥70% of currently-deep tool-mention turns
  would flip to fast under B, the flipped turns are mostly one-shot
  questions, and step 5 doesn't surface a legitimate-deep case that
  Option B would break.
- **Option A wins** if: even with tool tokens excluded, most
  follow-up turns still cross 500. The signal in the data is "all
  conversations grow," not "tool results bloat them." Raise the
  threshold to wherever the 75th percentile of conversation length
  sits, leaving the top quartile (long pastes, deep research) above
  the line.
- **Option C wins** if: Option B over-flips (turns legitimately
  benefiting from deep get pushed to fast), AND the tool-mention
  classifier signal turns out to be a good predictor of "one-shot
  question." We'd then use the classifier signal as the override,
  not the token count.
- **Do nothing** if: <30% of currently-deep turns are tool-mention,
  AND the wall-clock impact is tolerable. The bug is real but rare.

## Followups to ship if probe confirms

These are the deliverables once the probe data is in. They would be
Phase 6b. (`scripts/probe_complexity_gate.py` already ships as part of
6a — it is the probe itself, not a followup.)

1. **The chosen fix.** One of:
   - `config.yaml` edit (Option A, ~1 line).
   - `complexity.py` patch + test (Option B, ~10 lines + a test
     case).
   - `graph.py` `node_complexity` patch + test (Option C, ~15 lines +
     a test case).
2. **Phase 6b deploy doc** with smoke tests that confirm:
   - The `tourniquet`-shaped query now routes fast on its second turn.
   - A long pasted document still routes deep.
   - End-to-end ReAct still works on the fast path.
3. **Memory note update.** The "math classifier audit" followup in
   PROJECT_STATE.md should be expanded to "math + tool-bloat
   audits" since both are complexity-gate concerns.

## Out of scope for 6a

- Writing or running the probe script. That's the work itself, not
  the test plan.
- Changing the `react_max_tool_chars` knob to make tool results
  smaller. Smaller tool results reduce the bloat but don't fix the
  underlying gate-too-aggressive question.
- Investigating the third turn's `reflect=too_short/attempts=2`
  behavior. That's a separate reflect-loop concern; mentioned in
  Phase 6's logs but a different mechanism.
- Anything OWUI-side. The bug is entirely in Audrey's routing.

## Operational notes during the probe

- The probe is read-only. Nothing changes on Unraid; the only side
  effect is reading the SQLite archive.
- The chat archive holds all per-user message history. **Don't share
  the raw query output** — it contains user PII (per
  `feedback_no_real_emails_in_content` from the memory notes,
  apply the same caution to chat content).
- Aggregate statistics (counts, histograms, role-grouped sums) are
  fine to share. Single-message content for one's own troubleshooting
  is fine. Cross-user content snippets in a doc are not.
