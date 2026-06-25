# Campaign 2 — Phase 23: audit-finding drain

> **Number reuse note.** "Phase 23" was first used for the `web_search`
> grounding nudge (shipped `e0702cd`, **reverted** `bd95641` — see the
> PROJECT_STATE log). That work is **deprecated/abandoned**; its
> `phase-23-deploy.md` was deleted by the revert. The number is reused here for
> the audit drain (user decision 2026-06-24). The deprecated nudge has no
> remaining artifacts in the tree.

Plan for draining the Open findings filed in the 2026-06-24 full-corpus audit
(both courses). Findings live in `docs/ai-course/AUDIT.md` and
`docs/python-course/AUDIT.md` (gitignored queues). Each was re-verified against
current source on 2026-06-24 before this plan — all still valid, nothing has
gone stale.

The work splits into **one code change** (runtime behavior) and **lesson-doc
drift** (teaching material; no runtime impact). The code item is the only thing
that touches the deployed app, so it gets the deploy doc and the smoke-test
gate; the lesson fixes are local doc edits.

Read `AGENTS.md` "How to work" first (verify-before-claiming, edit-before-
write, no git writes, **no code changes from a lesson audit without explicit
approval**). Run `.venv/bin/pytest tests/ -q` and `.venv/bin/ruff check .`
before reporting the code item done. After the source edit, run
`scripts/check-lesson-links.py` on the changed files (`graph.py`,
`routes/openai/pipeline.py`) — the maintainer course cites them by line.

| Item  | What                                          | Kind        | Gate                          |
|-------|-----------------------------------------------|-------------|-------------------------------|
| 23a   | Synth timeout → `pick_panel_timeout`          | code        | approval → tests → smoke test |
| 23b   | Maintainer-course refresh (L1,4,6,7,8,9,13,15)| lesson docs | per-lesson drain w/ user      |
| 23c   | Beginner-course refresh (L3, L5; L1 nit)      | lesson docs | per-finding drain w/ user     |
| 23d   | Cite-only cleanup band (maintainer)           | lesson docs | mechanical, after 23b         |

Items are independent. **23a can ship without touching the courses**; the
lesson items can be drained on the user's own schedule (audit posture). Suggested
order below.

---

## Item 23a — Deep synthesis timeout bypasses `pick_panel_timeout` (code)

**Source finding.** `docs/ai-course/AUDIT.md`, `consider` — "CODEBASE: deep
synthesis timeout bypasses `pick_panel_timeout`."

**Problem (verified).** Both deep paths compute a *pool-aware* per-worker
timeout for the panel via `pick_panel_timeout(cfg, pool_key)`, then hand the
synthesizer the *raw* `deep_worker_timeout` instead:

| Path        | Panel timeout                                  | Synth timeout                                   |
|-------------|------------------------------------------------|-------------------------------------------------|
| non-stream  | `pick_panel_timeout` ([graph.py:314](../../src/audrey/pipeline/graph.py#L314)) | `deep_worker_timeout` ([graph.py:353](../../src/audrey/pipeline/graph.py#L353)) |
| streaming   | `pick_panel_timeout` ([pipeline.py:574](../../src/audrey/routes/openai/pipeline.py#L574)) | `deep_worker_timeout` ([pipeline.py:625](../../src/audrey/routes/openai/pipeline.py#L625)) |

`pick_panel_timeout` ([deep_panel.py:73-86](../../src/audrey/pipeline/deep_panel.py#L73))
returns `cfg.timeouts.cloud` (240s) for `deep_panel_cloud`, else
`cfg.timeouts.deep_worker` (360s). Config today: `cloud: 240`, `deep_worker: 360`
([config.yaml:219](../../config.yaml#L219), [:225](../../config.yaml#L225)). So a
`deep_panel_cloud` (cloud-only) request runs its **workers** on a 240s budget
but its **synthesizer** on 360s.

**Severity.** Not a crash, not a wrong answer — it's harmless latency headroom
on cloud-only synthesis. The real defect is *drift*: `pick_panel_timeout`'s
docstring says it exists so "the two paths can't drift," and synthesis is
silently drifting from it. A future timeout change to the cloud pool wouldn't
reach the synthesizer.

**Files.**
- `src/audrey/pipeline/graph.py` — `node_synthesize` (the synth call at :353)
- `src/audrey/routes/openai/pipeline.py` — `_run_synth_stream` (synth call at :625)

**Fix.** Both synth call sites already have `pool_key` in scope, so this is a
two-line change plus a small cleanup:

1. **graph.py** — `node_synthesize` already derives
   `pool_key = state.get("panel_pool") or pool_key_for(...)` at :345. Change the
   synth call's `timeout_s=deep_worker_timeout` → `timeout_s=pick_panel_timeout(cfg, pool_key)`.
   (`deep_worker_timeout` is still used by the panel call at :353-... region —
   check it's not orphaned; the panel still uses it via `pick_panel_timeout`, so
   the local `deep_worker_timeout` at :95 may now be unused in this node — grep
   before deleting; it's also read elsewhere in the graph builder.)
2. **pipeline.py** — `_run_synth_stream` change `timeout_s=deep_worker_timeout`
   → `timeout_s=pick_panel_timeout(cfg, pool_key)` (`pool_key` is set at :572,
   `cfg` is in scope). The local `deep_worker_timeout` computed at :573 was only
   there to feed this call — after the change it's dead; remove it (it does not
   feed the panel call, which uses the `timeout_s` at :574). **Verify** with a
   grep that nothing else in the function reads it before deleting.

Net effect: cloud-only deep requests synthesize under the same 240s the panel
used; non-cloud pools are byte-identical (both branches already returned
`deep_worker` for them).

**Behavior invariant.** For `deep_panel` and `deep_panel_local` (the local-
holding pools), `pick_panel_timeout` returns `deep_worker` = exactly today's
value — zero change. Only `deep_panel_cloud` synthesis changes (360 → 240),
aligning it with its own panel.

**Tests** (`tests/test_deep_panel.py` — that's where `pick_panel_timeout` and
the panel helpers are exercised; grep confirmed no synth-timeout test exists):
- `pick_panel_timeout(cfg, "deep_panel_cloud")` returns `cloud` and
  `pick_panel_timeout(cfg, "deep_panel"/"deep_panel_local")` returns
  `deep_worker` — pin the helper directly (cheap, no async).
- A test asserting the **synthesizer** receives the pool-aware value: drive
  `node_synthesize` (or a thin wrapper) with a cloud `pool_key` and assert the
  `synthesize_fn` mock was called with `timeout_s == cfg.timeouts.cloud`, and
  with a local `pool_key` → `timeout_s == cfg.timeouts.deep_worker`. This is the
  regression guard the finding asks for.
- Existing deep-panel/graph tests pass unchanged (proves the local-pool
  invariant).

**Done when.** New tests pass, full suite green, ruff clean, both synth call
sites use `pick_panel_timeout`, any orphaned `deep_worker_timeout` local
removed, `check-lesson-links.py` reports no new confident DRIFT on the two
changed files (line shifts are tiny — re-anchor any lesson cite that moves).
Deploy doc `phase-23-deploy.md`; **smoke test on the box** = one `audrey_cloud`
(or `audrey_deep` routed cloud-only) deep request completes normally.

---

## Item 23b — Maintainer course refresh (`docs/ai-course/`)

**Source finding.** `docs/ai-course/AUDIT.md`, "Full-course accuracy sweep
2026-06-24." Eight lessons carry substantive (not just line-number) staleness,
all from source that landed *after* the lessons shipped: the `routes/openai.py`
package split, inline-image support, deep-intent routing, and the expanded
`OllamaClient`. **No runtime impact** — this is teaching accuracy.

**Posture.** These are audit findings. Per `AGENTS.md`, walk each lesson's
finding with the user and resolve (fix / accept / defer) before editing — and
follow the lesson-writing style when rewording. Drain order below is by how
*wrong* the lesson currently is (actively-false claims first, pure cite drift
last).

**Tier 1 — actively false statements (a reader is misled):**
- **L15** ([lesson-15-openai-routes.md:113-133](../lessons/lesson-15-openai-routes.md))
  — schema snippet still shows `content: str`; current
  `ChatMessage.content` is `str | list[dict[str, Any]]`
  ([schemas.py:25](../../src/audrey/routes/openai/schemas.py#L25)). Also: the
  virtual-model table misses deep-intent + image-turn forcing; the non-streaming
  flow diagram says `_generate_via_pipeline` classifies/counts complexity, but it
  now builds state and awaits the graph; several Q&A route cites moved with the
  split. Needs a focused post-split/inline-image re-anchor + reword pass.
- **L6** ([lesson-06-the-model-layer.md:734](../lessons/lesson-06-the-model-layer.md))
  — `chat()` payload shows `"messages": messages`; current source sends
  `"messages": _to_ollama_messages(messages)`
  ([ollama.py:143](../../src/audrey/models/ollama.py#L143)). That helper
  ([ollama.py:46](../../src/audrey/models/ollama.py#L46)) flattens OpenAI
  `content` parts into Ollama `content` + `images` and is load-bearing for image
  turns — add it to the mental model + re-anchor the whole `OllamaClient`
  subsection.
- **L1** ([lesson-01-foundations.md:587](../lessons/lesson-01-foundations.md))
  — lists `WorkerDraft` under "Dataclasses," but it's
  `class WorkerDraft(TypedDict, total=False)`
  ([state.py:16](../../src/audrey/pipeline/state.py#L16)); the same lesson
  correctly calls it a TypedDict elsewhere. One-line fix: drop the bullet.

**Tier 2 — stale because the model changed (prose + cites):**
- **L7** ([lesson-07-classification-and-routing.md:443-477](../lessons/lesson-07-classification-and-routing.md))
  — route-order table predates `image_turn` + `has_deep_intent`. Current order
  ([graph.py:224-250](../../src/audrey/pipeline/graph.py#L224)): image → owui_task
  → forced-deep → forced-fast → token count → deep_intent. Threshold block moved
  to `config.yaml:296` + `deep_intent_phrases` at `:306`.
- **L8** ([lesson-08-deep-mode.md:700-701](../lessons/lesson-08-deep-mode.md))
  — sample log `complexity: tokens=287 mode=deep reason=token_count` uses old log
  shape (now `complexity: %d tokens -> %s (%s)`) *and* an impossible example
  (287 tokens < 500 threshold wouldn't be deep for token_count). Use a current
  shape, e.g. `complexity: 640 tokens -> deep (tokens>=500)`, or make it
  `deep_intent`.
- **L9** ([lesson-09-tool-use-and-react.md:386-387](../lessons/lesson-09-tool-use-and-react.md))
  — `config.yaml:145` cite for `max_tool_result_chars` now lands in
  `deep_panel_local`; the `agentic.react` block is at `config.yaml:166`,
  `max_tool_result_chars` at `:175`.
- **L13** ([lesson-13-memory-and-context-injection.md:440](../lessons/lesson-13-memory-and-context-injection.md))
  — streaming-deep archive-write and `_phase_thinking` cites point at pre-split
  `routes/openai.py` lines; re-anchor to `routes/openai/pipeline.py` (archive call
  ~:780, `_phase_thinking` ~:831).
- **L4** ([lesson-04-request-lifecycle.md:47](../lessons/lesson-04-request-lifecycle.md))
  — tells the reader to open `routes/openai.py` (now the `routes/openai/`
  package) and points `_stream_openai` at a deleted line. Retarget to
  `routes/openai/routes.py` + `routes/openai/pipeline.py`. (`lesson-14:446` has
  the same stale filename but only as a snippet label.)

**Done when.** Each lesson's finding drained with the user, the agreed edits
applied, `check-lesson-conventions.py` clean on touched lessons, and
`check-lesson-links.py` re-run. Update the AUDIT.md entries to `resolved` with
the date as each lands. Lessons 0, 2, 3, 5, 10, 11, 12, 16 surfaced nothing new.

---

## Item 23c — Beginner course refresh (`docs/python-course/`)

**Source finding.** `docs/python-course/AUDIT.md`, "Full-course accuracy sweep
2026-06-24." Two real, one style call. No runtime impact. Same drain posture as
23b — the beginner course is gated identically.

- **L3** ([lesson-03-one-field-two-shapes.md:72-78](../python-course/lesson-03-one-field-two-shapes.md))
  — `should-fix`, **actively false now.** Says the request schema still pins
  `content` to a string and the list-of-parts branch isn't live. Current source
  accepts `str | list[dict[str, Any]]`
  ([schemas.py:25](../../src/audrey/routes/openai/schemas.py#L25)) and
  `OllamaClient` converts list content for image turns. Rewrite the note as a
  current-state update: the branch began as defensive/standard-shaped code and is
  now live for OWUI image turns. (This is the beginner-course mirror of L15/L6 in
  23b — same source change, three lessons describe it.)
- **L5** ([lesson-05-reading-a-signature.md:75-86](../python-course/lesson-05-reading-a-signature.md))
  — `should-fix`. `__all__` snippet and cite drifted after `has_deep_intent` was
  added. Re-anchor to `complexity.py:154`, add `"has_deep_intent"` to the
  snippet, and fix "these five are the module's public surface" → "these six" (or
  drop the count per the no-specific-counts rule).
- **L1** ([lesson-01-setup.md:151](../python-course/lesson-01-setup.md)) —
  `consider`, style call. Standalone "Your turn" section predates the later
  no-standalone-"Your turn" rule. Works for setup; judgment call whether to fold
  the three tasks into the flow. The setup commands themselves re-checked clean
  on 2026-06-24 (no change).

**Done when.** L3 and L5 drained + edited (or accepted), convention checker
clean, AUDIT.md entries marked `resolved`/`accepted` with date. L1 is the user's
call.

---

## Item 23d — Cite-only cleanup band (maintainer, `nit`)

**Source finding.** `docs/ai-course/AUDIT.md`, `nit` — "remaining citation
cleanup band." After 23b's substantive edits, automated results still show ~18
confident + ~88 advisory cite drifts, mostly expression-level cites the checker
can't auto-anchor. Mechanical, low value, **do last** (after 23b moves lines
around — doing it before would just re-drift).

**Approach.** Run `check-lesson-links.py` over the maintainer corpus, apply the
`fix:` line for each confident DRIFT on identifier-labelled cites, eyeball the
advisory `DRIFT?` band (those are mostly false positives on expression lines —
don't chase them unless one is genuinely wrong). Leave `AUDIT.md`'s own historical
cites as-is (rewriting a dated log falsifies the record).

**Done when.** `check-lesson-links.py` reports zero *confident* DRIFT across the
maintainer corpus; advisory band documented as residual false positives.

---

## Suggested order & commits

23a is the only deployable change and is independent of the doc work — do it
first so the smoke test can run while the doc drain proceeds.

1. **23a** (synth timeout) — code, gated on approval + smoke test.
   `fix(deep-panel): use pool-aware pick_panel_timeout for synthesis`
2. **23b / 23c** (lesson refreshes) — drained per-lesson with the user, edited as
   approved. One `docs(...)` commit per course (or per lesson) as they land:
   `docs(ai-course): refresh L1/4/6/7/8/9/13/15 for post-split + inline-image`
   `docs(python-course): correct L3 live-schema note + L5 __all__ snippet`
3. **23d** (cite cleanup) — last, after 23b's edits settle.
   `docs(ai-course): re-anchor residual cite drift`

23b/23c findings are audit-queue items — the user drains them on their own
schedule; only 23a carries deploy/smoke-test weight.
