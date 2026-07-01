# Campaign 2 — Checkpoint testing

A focused live-prompt pass targeted at the **deferred and in-flight
items** that hermetic pytest can't validate. Distinct from
`phase-4-testing.md` (broad smoke after any deploy) — this is a
periodic checkpoint to drain the deferred queue in
`docs/lessons/AUDIT.md` and the followup queue in
`docs/PROJECT_STATE.md`.

Each section names the open question, the prompt(s) that exercise it,
what to watch for, and what outcome closes or escalates the item.

Read `AUDIT.md` and `PROJECT_STATE.md` first so the queue state is
fresh in your head; both decay between sessions.


## Setup

- One OWUI tab logged in as a normal (non-admin) user.
- A second OWUI tab logged in as a *different* OWUI account (for
  user-scope and conversation-id checks).
- An admin OWUI session in a third tab (for auth-cache eviction
  checks).
- A terminal tailing decision lines:

```bash
docker compose logs -f audrey-ai \
  | grep -E "classify:|fast_path|deep_panel|synth:|chat_archive|memory:|auth:"
```

- A second terminal ready to pull full logs on demand:

```bash
docker compose logs --since 5m audrey-ai > /tmp/audrey.log
```

- Note the start time so you can scope log reads.


## Item 1 — Synth-draft size instrumentation

**Queue location:** `PROJECT_STATE.md` "In flight"; Lesson 8 deferred
`consider` in `AUDIT.md`.

**Question:** are real production deep-mode requests producing draft
bundles small enough that the lack of a per-draft cap is fine, or
should we ship a `max_synth_draft_chars` knob?

**Decision criteria:** total p99 < 16 KB → accept, leave uncapped.
16-24 KB → judgment call. > 24 KB → ship a cap.

**Prompts (deep mode, maximal fan-out):**

```text
Give me a thorough comparative analysis of BTRFS, ZFS, and ext4 across
performance, data integrity, snapshot behavior, RAID semantics, and
recovery from bit rot. Cover edge cases.
```

```text
Walk me through everything that happens from the moment I open OWUI
to the moment a deep-mode response renders, including streaming,
banner emission, archive write, and per-user scoping. Be exhaustive.
```

```text
Compare Postgres, MySQL, SQLite, DuckDB, and ClickHouse on durability,
query performance, replication topology, ecosystem maturity, and
operational complexity. For each pair, name a workload where you'd
pick one over the other.
```

Send each on `audrey_deep` (forces deep) and on `audrey_cloud` (cloud
workers, larger drafts). Repeat at different times of day to capture
warm/cold cache and varying tool-call patterns.

**Process the data:**

```bash
docker compose logs --since 1h audrey-ai > /tmp/audrey.log
uv run python scripts/analyze_draft_sizes.py /tmp/audrey.log
```

The script's `WHAT TO DO WITH THE OUTPUT` section names the cutoffs.

**Outcomes:**

- p99 < 16 KB after 50+ deep requests → mark the finding `accepted`
  in `AUDIT.md`, drop from `PROJECT_STATE` in-flight.
- p99 16-24 KB → propose a cap in `AUDIT.md` with a value derived
  from the data; don't ship without explicit approval.
- p99 > 24 KB → escalate from `consider` to `should-fix`, propose
  a cap in `AUDIT.md`.


## Item 2 — Parallel context-injection drift (streaming vs graph)

**Queue location:** Lesson 13 deferred `consider` in `AUDIT.md`.

**Question:** do the streaming-deep path (`_phase_thinking` in
`routes/openai.py`) and the non-streaming graph path
(`node_memory_recall` in `graph.py`) produce identical composed
system-message stacks? They have to stay in sync; nothing pins them.

**Setup:** seed a memory first so recall has something to do.

Send (any virtual model that's tool-capable):

```text
Remember that my favorite database is Postgres.
```

Wait for the response. Confirm `memory_store` fired in the logs.

**Prompt A — non-streaming:**

In OWUI: model picker → toggle streaming OFF (or use a curl with
`stream: false` on `audrey_deep`). Send:

```text
What's my favorite database, and why might I have picked it?
```

**Prompt B — streaming:**

Toggle streaming ON. Send the exact same prompt.

**Watch for:**

| Expect | Watch for |
|---|---|
| Both answers reference Postgres specifically | Streaming answer doesn't recall; non-streaming does (or vice versa) |
| Both reference "today is" datetime context if relevant | One side has the datetime stamp at the top of `state["messages"]`, the other doesn't |
| Identical `chat_history_search` guidance presence | One path teaches the model about chat-history search and the other doesn't |

For the deepest read, dump the resolved system-message stack at the
classify boundary on both paths. If they diverge, the deferred finding
is real; propose extracting `_build_context_messages(...)` shared by
both call sites in `AUDIT.md`.

**Outcomes:**

- Identical behavior across both paths → mark `accepted`, document
  the parallel structure inline.
- Divergence → escalate `consider` → `bug`, propose the shared
  helper.


## Item 3 — Streaming-cancel cleanup (the real-risk one)

**Queue location:** Lesson 4 deferred `consider` in `AUDIT.md`. The
**only** deferred item with real cost — cloud workers burn paid time
after the user has already disconnected.

**Question:** when a client closes the OWUI tab mid-stream, do
downstream deep-panel cloud workers actually get cancelled, or do
they keep running to completion?

**Prompt (audrey_cloud, long deep work):**

```text
Use audrey_cloud. Give me an exhaustive philosophical analysis of
identity persistence in chatbots, covering Parfit's reductionism,
Locke's memory criterion, and Dennett's heterophenomenology. Include
specific objections from each tradition and how a chatbot designer
might respond. Be thorough — at least 1500 words.
```

**Procedure:**

1. Note the wall-clock time when you hit send.
2. Watch the log tail for the deep dispatch — confirm cloud workers
   started.
3. Close the OWUI tab within ~5 seconds of seeing
   `_Dispatching panel_` appear (before synth starts).
4. Note that wall-clock time.
5. Tail the logs for the next 60-90 seconds.

**Watch for:**

| Expect | Watch for |
|---|---|
| Cancellation propagates: worker tasks log cancelled / aborted | Workers log normal completion *after* the disconnect timestamp |
| Synth never fires (no synth log line after disconnect) | A full synth + reflect cycle runs and emits a completion log line for a user who left |
| Archive write either skips or marks `partial=True` | Full assistant content gets archived as if delivered |

**Outcomes:**

- Cancellation works → upgrade `consider` → `accepted` with a
  one-line note pointing at the proof in the logs.
- Workers keep running → escalate `consider` → `bug`, this is
  burning real money on cloud workers; act before the
  `routes/openai.py` lesson, not after.


## Item 4 — Conversation-id stitching (deterministic-hash fallback)

**Queue location:** Lesson 13, §2.4 step-4 fallback — covered in
prose but never live-validated.

**Question:** does the deterministic prefix-hash fallback in
`resolve_conversation_id` actually stitch requests when OWUI omits
`chat_id`?

**Prompt sequence (same OWUI session, two fresh tabs):**

Tab 1, brand-new chat:

```text
I'm going to mention a specific phrase so I can find this conversation
later: "marmalade-cobblestone-zither". Please acknowledge.
```

Wait for the response. Open Tab 2, brand-new chat in the same
account.

Tab 2, identical opening:

```text
I'm going to mention a specific phrase so I can find this conversation
later: "marmalade-cobblestone-zither". Please acknowledge.
```

Then in Tab 2, ask:

```text
Earlier we discussed a specific phrase. What was it?
```

**Watch for:**

| Outcome | Means |
|---|---|
| Tab 2 recalls the phrase via archive | Step 4 fallback fired — first six messages hashed to the same `conversation_id` |
| Tab 2 cannot recall it | OWUI sent distinct `chat_id` values for the two tabs, step 1-3 won the race, step 4 never fires in practice |
| Tab 2 says "we never discussed that" but the archive contains it | The stitch failed and a fresh conversation_id was minted — step 5 fallback |

**Outcomes:**

- Stitching works → confirms step 4 is load-bearing as documented.
- OWUI always sends `chat_id` → note in `AUDIT.md` that step 4 is
  dead-on-arrival in current OWUI but kept as a future-proof
  fallback; lesson prose stays accurate.
- Stitch fails despite no `chat_id` → bug in `resolve_conversation_id`
  or in `_chunk_id` derivation; trace from the logs.


## Item 5 — Auth cache TTL and admin eviction

**Queue location:** new with Lesson 13's §2.1 — covered in prose,
worth a live check now that the surface is described in the lesson.

**Question (a):** does the 30-second TTL actually propagate OWUI
role/permission changes within ~30s?

**Question (b):** does `POST /v1/admin/auth/clear` force immediate
re-probe?

**Procedure (a) — TTL propagation:**

1. Tab 1 logged in as the test user (role = `user`). Confirm a normal
   chat request works.
2. Admin tab: in OWUI, promote the test user to `admin`.
3. Tab 1: immediately try an admin-only endpoint (e.g. via the upload
   page's admin links, or curl an admin route with the user's token).
   Expect 403 (cached role is still `user`).
4. Wait 35 seconds.
5. Retry the admin-only endpoint. Expect success — the cache entry
   has expired and re-probe pulled the new role.

**Procedure (b) — manual eviction:**

1. With Tab 1's role still cached as `admin`:
2. Admin tab: demote Tab 1's user back to `user` in OWUI.
3. Admin tab: hit `POST /v1/admin/auth/clear` (or the email-scoped
   variant against Tab 1's email).
4. Tab 1: immediately try the admin endpoint. Expect 403 — eviction
   forced a re-probe; new role is `user`.

**Watch for:**

| Expect | Watch for |
|---|---|
| Step (a) 5 succeeds after ~30s wait | Step (a) 5 still 403s — cache TTL is broken |
| Step (b) 4 returns 403 immediately | Step (b) 4 still permits admin access — eviction didn't reach the right cache row |
| `auth.py` log line for re-probe after eviction | No re-probe log — eviction silently no-op'd |

**Outcomes:**

- Both work → no new finding; the lesson prose is validated.
- TTL doesn't propagate → bug in `require_user`'s cache check.
- Eviction doesn't work → bug in `clear_auth_cache_for_email` or its
  route wiring.


## Item 6 — Uncovered-sub-questions synth acknowledgement

**Queue location:** recent commit `3354e8a` ("instruct synthesizer
to acknowledge uncovered sub-questions when planner asked for more
parallelism than the pool delivered") — shipped, awaiting live
validation.

**Question:** when the planner decomposes a request into more
sub-questions than the worker pool can run in parallel, does the
synthesizer actually acknowledge the uncovered ones, or silently gloss?

**Prompt (forces planner to decompose into many subtasks):**

```text
Compare these five databases — Postgres, MySQL, SQLite, DuckDB, and
ClickHouse — on each of these dimensions: durability, query
performance, replication topology, ecosystem maturity, and operational
complexity. For every dimension, name a workload where you'd pick
each one. Be specific.
```

That's 5×5 = 25 dimension/db cells, plus 5 workload picks per
dimension. The planner will decompose into more subtasks than the pool
(typically 3-4 workers) can dispatch.

Use `audrey_deep`.

**Watch for:**

| Expect | Watch for |
|---|---|
| Final answer explicitly notes which sub-questions weren't covered | Answer reads as complete, no mention of skipped scope |
| Planner log line shows N subtasks > worker count | Planner returned ≤ pool size (test didn't actually exercise the path) |
| Synth prompt log line (if `log_synth_prompt` is on) includes the "uncovered" hint | Synth prompt missing the hint |

**Outcomes:**

- Acknowledgement present → close the followup, note in
  `PROJECT_STATE` that the commit's behavior is validated live.
- Silent gloss → the synth prompt change didn't take; debug
  `synthesize.py` prompt assembly.


## What this suite is not

- Not a regression suite — hermetic pytest covers that.
- Not a model-quality grader — answer correctness is eyes-only.
- Not exhaustive — only covers deferred items with **live-only**
  validation paths. Items closeable from code-read alone stay in
  `AUDIT.md` and get drained there.

## After draining

For each item with a clear outcome:

1. Update `AUDIT.md` — move from `Deferred` to `Resolved` or
   `Accepted`, with the date and a one-paragraph note pointing at
   what you observed.
2. If an item escalated (e.g. `consider` → `bug`), file the new
   finding under `Open` with severity bumped.
3. Bump `_Last updated:` on `PROJECT_STATE.md` and note the
   checkpoint pass in the "Current state" summary.
4. If the synth-draft instrumentation produced a decision, drop the
   instrumentation from the in-flight list.
