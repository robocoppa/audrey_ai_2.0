# Campaign 2 Phase 4 - Manual UI smoke testing

Hermetic pytest covers the code paths. This document covers the bits
pytest can't: routing decisions on real prompts, tool dispatch against
live tools-server, panel + synth quality on real models, streaming UX,
user-scope isolation across OWUI accounts. Run after any non-trivial
deploy.

The goal is not to grade model quality. The goal is to catch
**behavior regressions** in the parts of the pipeline that pytest
doesn't see: classifier reasons, mode selection, tool calls, banners,
metrics, log lines.

## Setup

Before each session:

- One OWUI tab logged in as a normal (non-admin) user.
- A second OWUI tab (or incognito window) logged in as a *different*
  OWUI user — required for the user-scope check.
- One terminal tailing Audrey logs filtered to decision lines:

```bash
docker compose logs -f audrey-ai \
  | grep -E "classify:|fast_path|deep_panel|synth:|chat_archive|memory:"
```

- One terminal ready to scrape metrics on demand:

```bash
curl -sS http://localhost:8000/metrics \
  | grep -E "audrey_(pipeline|tool_calls|chat_archive)_total"
```

Each category below has a small prompt sheet, the expected behavior,
and the specific failure mode to watch for. Mark each prompt pass/fail
in a scratch note as you go — comparing against your last run is more
useful than judging in isolation.

---

## Category 1 — Classification routing (5 prompts)

Confirms keyword + router decisions still land on the right task type.
Each prompt runs **twice**: once non-streaming, once streaming. The
classify log line should be identical both times — the streaming
codepath had a real bug here in the past where it dropped `tool_names`
and tool-mention prompts misrouted.

The primary signal is the log line:

```text
classify: <task_type> (<reason>, conf=<n>)
```

`<reason>` starts with `keyword:` (keyword pre-filter won) or `router:`
(router model decided).

### 1.1 Strong code keyword

```text
Write a Python function that flattens a nested list.
```

| Expect | Watch for |
|---|---|
| `task=code` | Routed `general` or `reasoning` |
| `reason=keyword:code_strong` | `reason=router:*` instead (regex missed `def `) |

### 1.2 Reasoning strong keyword

```text
Compare BTRFS and ZFS in five concrete dimensions.
```

| Expect | Watch for |
|---|---|
| `task=reasoning` | Routed `general` |
| `reason=keyword:reasoning_strong` | `reason=router:*` (the word `compare` should fire the regex) |

### 1.3 VL keyword

```text
Identify the type of rock in this photo.
```

| Expect | Watch for |
|---|---|
| `task=vl` | Routed `general` (regex missed `photo`/`rock`) |
| `reason=keyword:vl_strong` | A VL model didn't get picked; check the dispatch log too |

### 1.4 Tool-mention override

```text
Use kb_search to find docs about my Threadripper workstation.
```

| Expect | Watch for |
|---|---|
| `task=general` | Routed `vl` (because of the word "docs") or anything not-general |
| `reason=keyword:tool_mention:kb_search` | Any other reason — means the tool-mention override didn't fire |

The streaming run of this prompt is the most important single test in
the suite. It used to misroute. Re-run it in streaming mode and watch
the log line is identical to the non-streaming run.

### 1.5 Router fallback

```text
What's the capital of Iceland?
```

| Expect | Watch for |
|---|---|
| `task=general` | Other task types |
| `reason=router:general` (router was reached) | `reason=fallback:*` — means the router timed out or strike-failed twice |

---

## Category 2 — Mode selection (3 prompts)

Confirms `forced_*` virtual models and the length threshold still work.

The signal is the request log line:

```text
chat.completions (stream) model=<virt> task=<t>(...) tokens=<n> mode=<fast|deep>
```

For non-streaming, look at the line right before the answer is sent.

### 2.1 Tiny prompt, auto

```text
Hi.
```

| Virtual | Expect mode | Watch for |
|---|---|---|
| `audrey_auto` | `fast` | `deep` (over-eager escalation on a 1-token prompt) |

### 2.2 Big prompt, auto

Paste 500–800 tokens of real text (any long article, your own notes,
etc.) and ask: `Summarize the above in three bullets.`

| Virtual | Expect mode | Watch for |
|---|---|---|
| `audrey_auto` | `deep` | `fast` despite exceeding `complexity.token_threshold` (default 500) |

### 2.3 Forced override

```text
Hi.
```

| Virtual | Expect mode | Watch for |
|---|---|---|
| `audrey_deep` | `deep` (forced) | `fast` — means `forced_deep` check didn't fire in the streaming path |
| `audrey_fast` | `fast` (forced) | `deep` |

---

## Category 3 — Tool dispatch (4 prompts)

Confirms tool-capable models actually call tools and the user-scope
invariant holds. Requires logged-in OWUI session.

The two primary signals:

- Audrey log line per tool call: `dispatch: <tool> ok in <s>` or
  the warning variants.
- Metrics counter: `audrey_tool_calls_total{tool=<name>,outcome=ok}`
  should increment.

### 3.1 Web search ("fresh fact")

```text
What's the latest stable BTRFS release this week?
```

| Expect | Watch for |
|---|---|
| Fast-path ReAct runs; `web_search` dispatched ≥1× | No tool call ("according to my training cutoff...") |
| Answer cites Brave results (specific version numbers, dates) | Tool called but the model ignored the result |
| Tools-used footer renders at the bottom of the reply | Footer missing or named `?` |

### 3.2 Memory store

```text
Remember that I prefer ZFS over BTRFS for home use.
```

| Expect | Watch for |
|---|---|
| `memory_store` dispatched once | No tool call; or `tool_calls_total{tool="memory_store",outcome=error}` |
| Reply is brief and **doesn't** narrate the storage | Reply says "I'll remember that for you..." (narration smell — that's a prompt regression) |
| In the dispatch log, the `tags=` field contains `user:<your-email>` | A different user value (means `_force_user_tag` didn't overwrite) |

### 3.3 Memory recall

```text
What did I just tell you to remember about ZFS?
```

| Expect | Watch for |
|---|---|
| `memory_search` or `memory_recall` dispatched | No tool call; answer hallucinates a different preference |
| Answer reflects the stored ZFS-over-BTRFS preference | Model invents a memory it didn't store |

### 3.4 Chat-history search (and user-scope check)

```text
Search my prior chats for what we discussed about Threadripper.
```

| Expect | Watch for |
|---|---|
| `chat_history_search` dispatched | Tool not called (likely `tools=0` if so) |
| Answer cites at least one prior turn | Answer fabricates a prior chat that never happened |

**User-scope check.** Switch to your second OWUI account, paste the
exact same prompt. Expect: no results (or only that second account's
unrelated history). If the second account's session returns the first
account's Threadripper turn, the user-overwrite invariant is broken —
audit `_USER_SCOPED_TOOLS` in `dispatch.py` and the OWUI session JWT
config.

---

## Category 4 — Synth and reflect (3 prompts)

Forces the deep panel + synth to fire. Tests that synth handles
multi-draft input cleanly.

The signal is the answer body — synth quality is mostly eyes-only, but
there are specific anti-patterns that should never appear.

### 4.1 Multi-part comparison

```text
Compare BTRFS and ZFS on snapshots, send/receive semantics, transparent
compression, online integrity scrubbing, and behaviour when a disk fails
in a redundant array. Be specific about modern Linux distros.
```

Virtual: `audrey_deep`.

| Expect | Watch for |
|---|---|
| `deep_panel: pool=... workers=N ok=N tool_grounded=...` in logs | `ok=0` (all workers failed) |
| One coherent table or sectioned answer | Two parallel sections labeled "Draft 1" and "Draft 2" (synth narrating its process) |
| No `## Caveats` placeholder unless drafts genuinely disagreed | A literal `## Caveats\n- none` block — that's a prompt regression |
| No worker model names in the prose | `"As qwen3.6:35b suggested..."` (synth leaked internals) |

### 4.2 Single-topic essay

```text
Write a short essay on why fish in the abyssal zone glow.
```

Virtual: `audrey_deep`.

| Expect | Watch for |
|---|---|
| Synth returns a single essay | Multiple parallel essays jammed together |
| No "Approach" or "Synthesis" preamble | A preamble paragraph explaining what the synthesizer is doing |

### 4.3 Local-only deep panel

Same multi-part question as 4.1, virtual `audrey_local`.

| Expect | Watch for |
|---|---|
| `pool=deep_panel_local` in logs; no cloud worker in the worker list | A cloud model showed up anyway (pool config regression) |
| Synth still produces an answer; quality may be slightly thinner | Synth fell over because the local pool only had one healthy model |

---

## Category 5 — Streaming UX (3 prompts)

Confirms the streaming machinery doesn't regress separately from model
output.

### 5.1 Deep streaming with banners

Paste the multi-part BTRFS/ZFS prompt from 4.1 in the OWUI deep
profile (or use a `curl -N` SSE call for a clean view).

| Expect | Watch for |
|---|---|
| `Thinking ▢` banner appears first | Banner missing or stuck |
| `Dispatching ▢` banner with per-worker checkmarks (`✓ qwen3...`, `✓ kimi...`) | Banners arrive but no checkmarks |
| `Synthesizing ▢` banner, then a horizontal-rule separator | Separator missing; banner doesn't close |
| Synth tokens stream live after the separator | All tokens arrive in one block (synth ran non-streaming) |
| Tools-used footer at the bottom if any worker called tools | Footer missing despite a worker showing `tool_grounded` in the metrics |

### 5.2 Fast streaming with tool use

```text
What's the latest stable BTRFS release this week?
```

Virtual: `audrey_fast`. Browser tab in OWUI is fine.

| Expect | Watch for |
|---|---|
| Stream arrives as **one chunk** after the ReAct loop completes | Tokens stream mid-loop (would be a real regression — the fast tool-using path is documented as one-chunk-after) |
| Per-worker footer renders the `web_search` call | Footer missing |

### 5.3 Mid-stream cancel

Issue a long deep request and **cancel mid-stream** — close the OWUI
tab, or `curl -N -m 3 ...` to force a 3-second timeout.

| Expect | Watch for |
|---|---|
| Audrey log says `stream deep done ... outcome=cancelled` | `outcome=ok` or `outcome=error` (cancel handling drifted) |
| `audrey_chat_archive_writes_total{result="partial"}` increments | Counter doesn't move (cancel path didn't archive) |
| The archive contains a `partial=1` row for what was streamed | No row at all |

The cancel test is the most likely to regress silently because the
answer is gone before you can see it. The metric is your only signal.

---

## Category 6 — End-to-end smoke after any change (3 prompts)

The minimum suite. Run this after every code/config push, even small
ones. ~3 minutes total.

### 6.1 Trivial fast

```text
What is 2+2?
```

Virtual: `audrey_fast`.

| Pass criteria |
|---|
| HTTP 200 |
| Answer contains `4` |
| Round-trip under 5 seconds on warm models |
| Audrey log: `mode=fast`, no tool calls |

### 6.2 Trivial deep

```text
Compare BTRFS and ZFS briefly.
```

Virtual: `audrey_deep`.

| Pass criteria |
|---|
| HTTP 200 |
| Audrey log: `deep_panel: ... workers=2 ok=2` |
| Audrey log: `synth: <model> ok in <s>s` |
| Answer is a real comparison, not "I'd be happy to help compare..." |

### 6.3 Tool dispatch under deep

```text
Search my prior chats for BTRFS and quote one back.
```

Virtual: `audrey_deep`.

| Pass criteria |
|---|
| HTTP 200 |
| Audrey log: at least one `dispatch: chat_history_search ok` |
| Answer cites a real prior turn |
| Metric: `audrey_tool_calls_total{tool="chat_history_search",outcome="ok"}` increments |

If all three pass, the broad pipeline is healthy. Move on.

---

## What to record

After each session, add a one-line note per category to a personal
running log:

```text
2026-05-12  cat 1: pass (3.4 streamed, classify line identical)
            cat 2: pass
            cat 3: pass (3.4 user-scope clean)
            cat 4: pass; synth was a bit verbose on 4.2 (not regression)
            cat 5: 5.3 cancel did not archive partial — see followup
            cat 6: pass
```

Compare against the last run when something fails. "It used to work"
is more useful than "I think this looks wrong."

## What this suite is not

- **Not a quality benchmark.** Model output quality drifts with weights,
  not with code. Use vibes for "is this answer good"; use this suite
  for "is the *machinery* still doing the right thing."
- **Not a unit-test replacement.** Pytest catches code bugs; this
  catches integration and routing bugs that no test fixture can fake.
- **Not exhaustive.** Twenty prompts can't cover every code path. They
  cover the ones most likely to regress when something changes — that's
  the design trade.

## Followups

- A tiny `scripts/smoke-test-ui.sh` could automate Category 6 (the
  three end-to-end prompts) via `curl` against `/v1/chat/completions`,
  parse responses, and exit non-zero if any criterion fails. Run pre-
  deploy. Worth doing once you've used Category 6 manually for a few
  weeks and have a feel for which criteria matter.
- A Grafana panel that surfaces per-tool dispatch counters alongside
  `chat_archive_writes_total` would make Categories 3 and 5 much
  faster to eyeball. Open question in
  [`docs/PROJECT_STATE.md`](../PROJECT_STATE.md).
- Synth-anti-pattern regression: if you ever catch a synth output with
  "Draft 1 / Draft 2" structure or "## Caveats - none" placeholder,
  add a sentence to `SYNTH_SYSTEM` in
  [`src/audrey/pipeline/prompts.py`](../../src/audrey/pipeline/prompts.py)
  forbidding it and re-run Category 4.
