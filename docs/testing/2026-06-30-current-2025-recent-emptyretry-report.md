# Eval report — 2026-06-30 `current-2025-recent`, empty-result retry

Paired with
[`2026-06-30-current-2025-recent-emptyretry-answers.md`](2026-06-30-current-2025-recent-emptyretry-answers.md).
Single-case re-run of the protocol's most search-dependent prompt, after
deploying the SearXNG empty-result retry. Diffed against the same case in the
two prior runs the same day.

## Headline

**The empty-result retry recovered grounding on the exact case that was
failing.** A useless degraded answer ("couldn't verify, no usable primary
documentation," no Sources) became a confident, dated, sourced answer. Validated
live on the box. The fix is a mitigation, not a cure — SearXNG is still
throttled underneath (latency rose), and the durable fix remains renewing the
Brave key.

## The change under test

`tools-server/searxng.py`: a SearXNG `200` with `results: []` is usually a
transient upstream-engine throttle, not "nothing exists." It's a valid HTTP
response, so it passed through as `is_error=False` and silently starved
grounding. Now the client:
- retries an empty result **exactly once** after a 1.5s wait (longer than the
  transport backoff so the throttled engine can recover; one retry only —
  re-hammering worsens it);
- **never caches an empty** (a cached transient-empty would starve every
  identical query in the run for the 900s TTL).

## Before / after — same case, same day

| | run 1 / run 2 (no retry) | run 3 (empty-retry deployed) |
|---|---|---|
| opening | "I couldn't fully verify… no usable primary documentation… internally contradictory" | "here is a chronological overview of the key releases and shifts" |
| grounding | fell back to training knowledge, hedged everything | confident, dated, specific |
| Sources block | **none** (nothing survived to cite) | **3 real primary sources** (deepseek.com, ai.meta.com/blog/llama-4, mistral.ai) |
| detail | 5 vague model mentions | DeepSeek-R1 (Jan 20), V3 (Dec 26), Gemma 3 (Mar 12), Llama 4 (Apr 5), Qwen3 (Apr 29), R1-0528 (May 28), V3.1 (Sept) — params, licenses, context windows; even the Llama 4 LMArena benchmark controversy |
| latency (total) | 207s / 252s | **411s** |

The run-3 answer contains specific, sourced 2025 facts that were **not
retrievable in either prior run**. That is the empty-retry filling transient
throttle-empties before the worker concluded "no sources."

## Footer (new ✅/❌ format)

```
> _Tools used:_  _(✅ = calls succeeded, ❌ = calls failed)_
> - **deepseek-v4-pro:cloud** — web_search ✅13
> - **qwen3.6:35b** — web_search ✅11 ❌3
> - **glm-5.2:cloud** — web_search ✅15
```

39 searches, 3 errored. **The ❌ count is NOT the signal here** — the signal is
that these ✅ searches actually returned *content* this run, where the prior
runs' ✅ searches came back empty (✅ counts `is_error=False`, and an empty result
is not an error). The footer can't show empties directly; the recovered answer
is the proof the retry worked.

## Trade-off

Latency rose **252s → 411s (+159s)**: the 1.5s empty-retry waits stack across
many empties, plus the workers do more real grounding work on the content they
now get back. For this case it's clearly worth it — a degraded answer became a
good sourced one. But it confirms SearXNG is **still throttled** (many empties
still need retrying). Renewing the Brave key would both restore grounding *and*
drop latency, since searches would hit on the first try and SearXNG would idle
as the rare fallback.

## Disposition

- Empty-result retry: **shipped, deployed, validated.** Recovers grounding on
  the failing case.
- ✅/❌ footer hardening: validated live (legend appears with failures, counts
  render).
- Root cause unchanged: Brave 402 → all load on one throttled SearXNG. Durable
  fix is operational (Brave key). Tracked in memory
  `project_searxng_upstream_throttle`.
- Noted follow-up: surface empty-but-OK result counts in the footer (needs a new
  field on the dispatch call record) — would have made today's diagnosis
  instant.
