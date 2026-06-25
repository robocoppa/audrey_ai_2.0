# Lesson Audit Findings

_Internal working doc. **Gitignored** — does not ship to GitHub._

This file collects code-review findings that surface during lesson
writing. It is the queue Bart drains on his schedule. It lives
*outside* the lesson docs themselves: published lessons stay focused
on teaching, while audit material — which is opinion-laden, code-
specific, and quickly stale — stays here.

Sister to `docs/lessons/CONTINUITY.md` (the lesson-plan tracker) and
`docs/campaign-1/HISTORY.md` (the build-campaign history).

---

## Posture

The course has a dual purpose: teach Bart, AND give the codebase a
fresh-eyes pass. Both halves are first-class — when writing a lesson,
do an audit pass on the files in scope, but capture the findings *here*
rather than inlining them in the lesson.

**No code changes without explicit approval**, ever — even for
obvious-looking nit fixes. This file is the queue; Bart drains it on
his schedule.

### What to look for, every time

(Not exhaustive — use judgment.)

- **Bugs.** Logic errors, race conditions, incorrect error handling,
  wrong assumptions about library behavior, off-by-ones, unchecked
  edge cases.
- **Optimization opportunities.** Synchronous calls in async functions,
  unnecessary work in hot paths, missing caches, O(n²) loops over
  data that's typically large, redundant network round-trips.
- **Smells.** Dead code, duplicated logic, complex code that wants a
  helper, comments that lie, defensive guards for impossible cases,
  features added but never wired in, missing test coverage on
  load-bearing functions.
- **Inconsistencies.** Two places doing the same thing differently
  (e.g. `_options_from_request` vs `_options_from_state`), drift
  between the docstring and the implementation, naming conventions
  that aren't applied uniformly.
- **Documentation.** Stale comments, lies in module docstrings,
  missing context where it would help.

### Severity tags

- `nit` — cosmetic; mention but don't push.
- `consider` — worth thinking about; tradeoff to discuss.
- `should-fix` — real problem, low urgency.
- `bug` — real problem, fix soon.
- `optimization` — measurable inefficiency.
- `smell` — looks suspicious, may or may not be a real problem.

### Status values

- `open` — just flagged, no action drafted.
- `proposed` — change has been drafted, awaiting yes/no.
- `resolved` — change shipped (with date).
- `accepted` — declined intentionally (with one-sentence reason).
- `deferred` — action is real but blocked on a prerequisite (measurement,
  a future lesson's audit pass, etc.). Every `deferred` entry must name
  the prerequisite explicitly so a later session knows what unlocks it.

### Process when Bart wants to act on a finding

1. Propose the change first.
2. Get explicit approval.
3. Edit code.
4. Mark the finding `resolved` here with the date.

---

## Findings log

Format per finding: severity tag, file:line citation, finding,
status. Group by status; within each status, group by lesson.

### Open

#### Full-course accuracy sweep 2026-06-24 (Lessons 0-16)

Initial sweep: convention checker clean; link checker found 18 confident
`DRIFT`, 88 advisory `DRIFT?`, and 0 broken links. The substantive lesson
findings from that sweep have been resolved in lesson prose (Lessons 1, 4, 6,
7, 8, 9, 13, 14, and 15). Recheck after edits: `DOCS_GLOB='docs/ai-course/lesson-*.md'`
found 332 cites checked, 252 ok, **0 confident drift**, 80 advisory `DRIFT?`,
0 broken; convention checker found 0 issues across the edited lesson files.
The remaining open item from this sweep is the code review followup below.

- **`consider` - CODEBASE: deep synthesis timeout bypasses
  `pick_panel_timeout` -> `resolved` 2026-06-24 (Phase 23a).** Both deep paths
  calculated pool-aware panel timeouts but passed the raw `deep_worker` timeout
  to the synthesizer: `graph.py` used `pick_panel_timeout(...)` for `run_panel`,
  then `synthesize(..., timeout_s=deep_worker_timeout)`; the streaming path did
  the same. Config has `cloud: 240` and `deep_worker: 360`, so cloud-only
  (`deep_panel_cloud`) synthesis got 360s while its panel ran on 240s — harmless
  latency headroom but a drift from the helper whose whole job is keeping the two
  paths aligned. **Fix:** both synth call sites now pass
  `pick_panel_timeout(cfg, pool_key)` (the `pool_key` was already in scope at
  each); the orphaned `deep_worker_timeout` local was removed from both
  `build_graph` and the streaming `_stream_via_pipeline`. Cloud-only synthesis
  now uses 240s (matching its panel); `deep_panel`/`deep_panel_local` unchanged
  (the helper returns `deep_worker` for them). Regression test added in
  `tests/test_deep_panel.py` (drives the compiled `synthesize` node with
  `synthesize_fn` stubbed, asserts the forwarded `timeout_s` equals
  `pick_panel_timeout` per virtual model). 503 pytests pass, ruff clean on
  touched files, no new confident cite drift (the 1-line deletions added 8
  advisory `DRIFT?` flags below them — fold into the 23d cite sweep).

#### Lesson 16 (custom-tools sidecar, `tools-server/`) — pre-write audit 2026-06-03

> **✅ DRAINED 2026-06-03** before the lesson write. #1, #2, #4, #5 fixed;
> #3 and #6 accepted (to be teaching callouts, not code changes). Scope
> correction recorded: the backlog's "top-level `chat_archive.py`" doesn't
> exist — the two files are `src/audrey/pipeline/chat_archive.py` (Audrey
> side, Lesson 13) and `tools-server/chat_archive.py` (sidecar side, partly
> Lesson 13). `tools-server/` is now **ruff-clean** (was 3 errors); 449
> pytests pass; the new Settings validator verified (raises on
> overlap≥max_chars, default config loads). Detail under Resolved/Accepted.

#### Lesson 17 (admin routes + auth tail, `routes/admin.py`) — pre-write audit 2026-06-03

> **✅ DRAINED 2026-06-03** before the lesson write. #1, #2, #3 fixed; #4
> settled as a scope decision (cover the full `require_admin` surface,
> spanning `routes/admin.py` + the rediscover route in `main.py`); #5
> accepted. **#3 was a behavior change** (200+error-dict → HTTP 503) so it
> got new test coverage — the chat_archive routes previously had **none**.
> 449 pytests pass (up from 445: +4 chat_archive failure-path tests), ruff
> clean, cite-check 0 drift. Detail under Resolved / Accepted.

#### Lesson 16 (streaming banners, `banners.py`) — pre-write audit 2026-06-03

> **✅ DRAINED 2026-06-03** before the lesson write (per the workflow rule
> that a lesson's own findings are drained before writing it). 4 of 5
> findings fixed (#1, #2, #3, #5), #4 accepted. 445 pytests pass, ruff clean
> on `banners.py`, cite-check 0 drift. Detail under Resolved / Accepted
> below.

#### Full-corpus accuracy audit 2026-06-02 — summary

> **✅ DRAINED 2026-06-03.** All findings below actioned. Final state:
> 445 pytests pass (4 new validator tests), ruff clean, full-corpus
> cite-check **0 confident DRIFT / 0 broken** (down from the band
> documented here), convention checker **0 findings across all 16
> lessons**. What shipped: L2 factual fix (temperature framing); cite
> re-anchoring across L4-L15; convention fixes (L1 forward-refs, L11/L12
> COUNT softening, L15 Phase-13 mentions, L2/L7 test-cite removals); the
> §2.4 mis-attribution + §2.9 stale-snippet + `_ROUTER_SYSTEM`-alias +
> L7 §2.2 prose/link fixes; and the CODEBASE config fix — the dangling
> `deep_panel_local` `fallback_synth` tags repointed to `qwen3.6:35b`
> (the pool's own local primary, keeping the local-only contract) plus
> `_validate_deep_panel_pools` extended to reject any worker/synth/
> fallback name absent from `model_registry` at boot. The advisory
> `DRIFT?` count remains (~67) — those are cite-checker false positives
> on expression-level lines, not real drift. The blocks below are kept
> for the record.

Audited all 16 lessons (0-15) against current source. Method: full-corpus
cite check (278 cites; 0 confident `DRIFT`, 0 broken, 95 advisory
`DRIFT?`), then per-lesson read + targeted source verification +
`check-lesson-conventions.py`.

**Headline: the teaching is sound; the line numbers rotted.** Across the
corpus there is exactly **one factual error** (L2's phantom `temperature`
bound — a Pydantic guard the lesson claims exists but the code lacks).
Everything else is (a) line-cite drift from source files growing after
lessons shipped, or (b) convention nits. No conceptual/mental-model
errors found in any lesson.

**Drift severity by lesson** (worst → clean):
- **L4** — pervasive, every nav line wrong (~+107 openai.py, ~+37
  graph.py); predates a renumber. Needs a full re-anchor pass.
- **L7** — widest after L4 (~29 cites, offsets up to +73), plus an
  internal prose/link inconsistency and a `_ROUTER_SYSTEM` alias nit.
- **L15** — the cancellation-trace cluster (~+9) already filed in the
  earlier L15 block below.
- **L6, L8, L13, L14** — moderate, mostly ±10; concentrated in whichever
  file grew (openai.py for L13/L14).
- **L5, L9, L12** — light (≤5 line offsets).
- **L11** — near-perfect; flagged cites all false positives.
- **L10** — zero drift (cleanest code-walk).
- **L0, L1, L2, L3** — foundations; few/no line cites.

**Cross-cutting findings (worth fixing once, corpus-wide):**
1. **Convention checker has two blind spots.** It does **not** flag
   `tests/` file cites (so the no-test-walkthroughs rule is unenforced —
   L3 §4, L7 lines 114/322 slipped through) and it **over-triggers**
   `COUNT` on illustrative example numbers it can't distinguish from
   codebase metrics (L11 "12 chunks", L12 "47 chunks"/"~5 chunks"). A
   grep sweep for `tests/` across lessons would catch the former.
2. **Cite checker can't confidently re-anchor expression-level cites**
   (only def/class/decorator targets), which is why a corpus-wide
   `DRIFT?`-only result masked real drift. "Green cite-check" ≠ "line
   numbers current."
3. **Possible stale model tags in config.yaml** (filed under L9, a
   CODEBASE not lesson finding): cloud/local/`vl` deep-panel pools still
   reference `qwen3.5:35b-a3b`, `glm-4.7-flash:q8_0`, `gemma4:31b`,
   `nemotron3:33b`, `qwen3-vl:32b` while code/reasoning/general pools use
   the current `glm-5.1:cloud`. Confirm intentional vs leftover.
4. **Forward-ref-by-number violations:** L1 (lines 477, 694), L4 (line
   274, also points at the wrong lesson now). Caught by the convention
   checker.

Per-lesson detail follows.

#### Lesson 0 (introduction) — accuracy audit 2026-06-02 (+2)

Pure orientation prose; no line-cites, no falsifiable code claims.
Architectural assertions all check out (two consumer GPUs, fast-path /
deep-panel / reflection-rerun shape, fair scheduling, per-user
memory/uploads, KB watcher, OWUI session auth, Prometheus+Grafana).
Convention checker: 0 findings.

- **`consider` — Lesson 0 foregrounds the "discrete phases" build
  construct (lesson lines 40-51).** It describes the project as "built
  in discrete phases, each landing one self-contained feature… Every
  phase had a deploy doc" and points readers at `docs/campaign-1/`. No
  literal "Phase N" token, so the convention checker passes — but the
  spirit of the no-Phase-refs rule is that a cold reader landing on the
  published lessons has no idea what the build phases were. Here it's
  arguably fine (Lesson 0 is *about* orientation, and it frames phases
  as historical build context, not as a thing the reader must track).
  Flagging as a judgment call for the author, not a clear violation. If
  acted on: reframe "phases" as "incremental feature-by-feature
  development" without leaning on the phase-doc construct.

#### Lesson 14 (fair scheduling) — accuracy audit 2026-06-02 (+2)

**Exceptionally accurate** — its core-file cites (inflight.py,
fair_gate.py) are nearly all dead-on, because those are the files it
teaches and they haven't grown. Verified: inflight.py _reserve:58,
breach-warn:85, _drop_reservation:100, await-acquire:117, _safe_bucket:148;
fair_gate.py acquire:91, _release:141, _last_granted skip:172,
release-call:139. All concepts verified precisely (Future-as-one-shot-
signal, `_last_granted` round-robin, done-future-sweep race window,
reservation-vs-holder eviction fix, two-layer composition). Convention
checker: 0 findings.

- **`should-fix` (cite drift — the two openai.py cites).** §2.1
  read-along cites `inflight.slot` acquisition at `routes/openai.py:246`
  (non-streaming) and `:341` (streaming) → actually
  [`521`](../../src/audrey/routes/openai.py#L521) and
  [`616`](../../src/audrey/routes/openai.py#L616). ~+275 drift (grown
  file). Lesson lines 92-93. fix: 246 → 521; 341 → 616.
- **`nit` (cite drift, ≤8 lines, ignorable).** fair_gate.py _waiters
  cited `:80`→79; done-future sweep cited `:150`→[`158`](../../src/audrey/pipeline/fair_gate.py#L158);
  inflight.py `except BaseException` cited `:116`→118; main.py
  construction `:60-66`→61-63. All within normal tolerance.

#### Lesson 13 (memory + context injection) — accuracy audit 2026-06-02 (+2)

**Very accurate on its own-subsystem cites; drift is concentrated in the
`openai.py` cites** (the grown file — same root cause as L15). All
auth/memory/chat_archive cites dead-on: auth.py AuthedUser:64,
role-reject:113, require_user:126, require_admin:157, clear_for_email:182;
memory.py recall_for_request:59, memory_system_message:105; chat_archive.py
resolve_conversation_id:57, StreamCollector:116, archive_turn:216;
tools-server build_chunks:160, archive_turn:356. Concepts all verified
(CSRF/header-only, fail-closed role check, 30s cache TTL, conversation-id
ladder, idempotency-by-id-derivation, two-call-site streaming bypass).
Convention checker: 0 findings.

- **`should-fix` (cite drift — all in openai.py).**
  - `payload.user` guard cited `:254-261` → [`257`](../../src/audrey/routes/openai.py#L257).
  - `resolve_conversation_id` call cited `:183` → [`284`](../../src/audrey/routes/openai.py#L284).
  - `_stream_deep_with_banners` cited `:884` (lines 369, 440) →
    [`898`](../../src/audrey/routes/openai.py#L898).
  - direct datetime/recall calls cited `:937` → [`1284-1286`](../../src/audrey/routes/openai.py#L1284).
  - prompts.py tags hint cited `:105` → 109; chat-history guidance `:116`
    → 117; compose_system_messages `:189` ✓.
  fix: re-anchor the openai.py cites (they'll be fixed in the same pass as
  L15's openai.py drift — overlapping targets like `_stream_deep_with_banners`
  → 898 appear in both lessons).

#### Lesson 12 (KB lifecycle) — accuracy audit 2026-06-02 (+2)

Grades very well. No factual/conceptual errors — thread-to-async bridge
(`call_soon_threadsafe`), debouncing (stamp-on-event/dispatch-on-silence),
delete-before-ingest ordering, scroll pagination, two-stores-agree
contract, delete-flow ordering (sqlite-first), startup
`reconcile_with_qdrant` all verified. Most cites correct: watcher.py
KBWatcher:106, _run:156, _delete_vectors:215; reconcile.py
_scroll_sources:88, exists:123, docstring:15-19, KBReconciler:171;
uploads_db.py contract:8-13; files.py delete_file:311-312. The 6
cite-checker flags are false positives (docstrings / non-landmark lines).

- **`nit` (cite drift, small).** watcher.py `_enqueue` cited `:91` →
  [`88`](../../src/audrey/kb/watcher.py#L88); uploads_db.py
  `reconcile_with_qdrant` cited `:171` → [`176`](../../src/audrey/kb/uploads_db.py#L176);
  files.py upload POST cited `:142` → `@router.post` at 144 /
  `upload_file` at [`145`](../../src/audrey/routes/files.py#L145). All ≤5.
- **`consider` (convention checker over-triggers, same as L11) — "47
  chunks" / "~5 chunks".** Lines 435 and 783 trip the `COUNT` rule, but
  both are illustrative (a sample log string; a worked 5 MB→~5 chunks
  trace), not codebase statistics. Same judgment as the L11 COUNT
  finding — the checker can't tell an example number from a baked-in
  metric. Leave or soften per author taste.

#### Lesson 11 (KB ingest + search) — accuracy audit 2026-06-02 (+2)

**Most accurate code-walk lesson with real cites** — nearly every line
number is dead-on. Verified correct: qdrant.py docstring 1-24, TEXT_DIM
:41, point_id :56; ingest.py ingest_text_file :103, delete_by_source
:122; chunk.py load_text :50, chunk_text :98; embed.py _validate_image_url
:86, TextEmbedder :107, _normalize :163; routes/kb.py kb_query :99,
_search_text_merged :119, RRF docstring :128-130, kb_query_image :164,
encoder-pick :176; tools-server/app.py kb_search :270. All concepts
verified (UUIDv5 idempotency, delete-before-upsert, stride/overlap,
shared CLIP space, RRF-when-embedders-differ, SSRF guards, the
winner-take-most merge). The 5 cite-checker flags are all false positives
(correct cites landing on docstrings / call lines / the redirect block).

- **`nit` (cite drift, ~1 line) — redirect block.** Lesson line 663 cites
  `embed.py:197-206` for redirect handling; the block actually runs
  [`198-206`](../../src/audrey/kb/embed.py#L198) (the `follow_redirects=
  False` comment is at 190). Trivial.
- **`consider` (convention checker over-triggers) — "12 chunks → 3".**
  `check-lesson-conventions.py` flags lesson lines 270 and 682 as `COUNT`
  violations ("12 chunks"). But these are a *hypothetical worked example*
  of delete-before-upsert ("imagine a file that produced 12 chunks, now
  3"), not a baked-in codebase metric like "~16k chunks." The no-counts
  rule targets transient real statistics; an illustrative scenario number
  is arguably fine and teaches the idempotency point concretely.
  Judgment call: either leave as-is (the example is clearer with numbers)
  or soften to "a dozen chunks → a few." **Meta: the checker can't
  distinguish illustrative counts from codebase counts — worth knowing
  when triaging its COUNT findings corpus-wide.**

#### Lesson 10 (how function calling works) — accuracy audit 2026-06-02 (+2)

**Cleanest code-walk lesson — zero cite drift.** Both Audrey cites land
dead-on: `to_ollama_tool` at discovery.py:46, `to_tool_message` at
dispatch.py:200, `_strip_unsupported_keywords` at discovery.py:99 (these
sit near the tops of their files, which haven't shifted). The protocol /
dialect content is accurate to current provider behavior: OpenAI
`arguments`-as-string, Anthropic `input_schema` + typed content blocks +
`tool_result` in a user-role message, Ollama mirroring OpenAI with
unconstrained tool decoding, the `tool_choice` value set. Convention
checker: 0 findings.

- **`nit` (will-age, not wrong) — illustrative `claude-opus-4-7` ID.**
  Lesson line 368 uses `claude-opus-4-7` in the Anthropic dialect
  example. It's a generic dialect illustration, not an Audrey claim, and
  it'll age as model IDs advance (current latest is Opus 4.8). No fix
  needed; if ever touched, prefer a version-neutral placeholder like
  `claude-opus-...`.

#### Lesson 9 (tool use + ReAct) — accuracy audit 2026-06-02 (+2)

Mostly conceptual; concepts all accurate (discover-once, dispatcher-
never-raises, the user-overwrite security invariant, truncation budget,
force-final-answer mode change, fast-vs-deep gate policy). Convention
checker: 0 findings. Light cite drift, plus a **codebase** finding that
surfaced here.

- **`should-fix` (cite drift).** discovery.py `ToolSpec` `:38`→39;
  read-along header cite `discovery.py:74` → `_resolve_refs` is at
  [`76`](../../src/audrey/tools/discovery.py#L76). dispatch.py read-along
  header `dispatch.py:68` → `dispatch_one` is at
  [`79`](../../src/audrey/tools/dispatch.py#L79) (`:68` lands in
  `_truncate`); `_force_user_tag` "line 61" → [`72`](../../src/audrey/tools/dispatch.py#L72).
  react.py health `:138`→144; gather `:170`→171.
- **`should-fix` (cite drift) — `config.yaml:138`/`:142` point into the
  wrong section.** Lesson line 119 cites `config.yaml:138` for
  "`agentic.react.*` knobs" and line 387 cites `:142` for
  `max_tool_result_chars`. But line 138 is `fallback_synth:` inside a
  deep-panel pool; the `agentic.react` block starts ~141 and
  `max_tool_result_chars` is ~147. fix: re-anchor to the real
  `agentic.react` lines.
- **`should-fix` (CODEBASE finding, not lesson) — two dangling
  `fallback_synth` tags in `deep_panel_local`.** CONFIRMED 2026-06-03 by
  programmatic cross-check of every pool model against `model_registry`
  (22 registry names). **Only two pool references are absent from the
  registry**, both `fallback_synth` slots in `deep_panel_local`:
  - `qwen3.5:35b-a3b` → `deep_panel_local/{code,reasoning,vl}/fallback_synth`
  - `gemma4:31b` → `deep_panel_local/general/fallback_synth`
  (My initial flag was over-broad — `glm-4.7-flash:q8_0`,
  `qwen3.5:397b-cloud`, `nemotron3:33b`, `qwen3-vl:32b` ARE all in the
  registry. Withdrawn.)
  **Runtime trace when hit:** only fires on `audrey_local` requests where
  the primary synth (`qwen3.6:35b`) is unhealthy/fails. Then the dangling
  fallback (a) passes `health.is_healthy` (unknown model → True,
  health.py:44-48), (b) `registry.location_of` defaults it to **local**
  (registry.py:69, model not found → `"local"`), so (c) it acquires the
  local GPU gate and calls `ollama.chat` for a model Ollama doesn't have
  → `OllamaError` → `record_failure` → degrade-to-longest-draft
  (synthesize.py:201-248). **Soft bug:** no crash, user still gets the
  longest worker draft, but the "fallback synth" is guaranteed dead
  weight — wastes one gate acquisition + a doomed round-trip instead of
  being a working second option. Nothing catches it:
  `_validate_deep_panel_pools` (config.py:136) only checks that
  `synthesizer` is *present*, not that any name exists in the registry.
  **Suggested drain:** repoint the two fallbacks at a registry-present
  model (e.g. `glm-5.1:cloud`, matching the other pools) and optionally
  extend `_validate_deep_panel_pools` to validate worker/synth/fallback
  names against `model_registry` so dangling tags fail at boot. Outside
  lesson scope. (Also confirms **Lesson 9's prose `qwen3.6:35b` at line
  543 is current/correct.**)

#### Lesson 8 (deep mode) — accuracy audit 2026-06-02 (+2)

Large, rich lesson; grades well. **No factual/conceptual errors** — the
four-stage model, asyncio.gather concurrency, gate-held-through-ReAct
(`gate=None`), round-robin subtask math, three-tier synth failure, and
brevity escape hatch all verified. **Config snippets are current** — the
`deep_panel` block at config.yaml:79 matches the lesson's worker/synth
names exactly (`qwen3-coder-next:latest`, `kimi-k2.6:cloud`,
`qwen3.6:35b`, `glm-5.1:cloud`). Convention checker: 0 findings. The
forward-pointer to the memory lesson (line 570) is phrased by name, not
number — compliant. Moderate cite drift (mostly ±10):

- **`should-fix` (cite drift).**
  - graph.py `node_planner` `:265`→[`271`](../../src/audrey/pipeline/graph.py#L271);
    planner log `:278`→284.
  - planner.py chat call `:54`→[`60`](../../src/audrey/pipeline/planner.py#L60).
  - deep_panel.py `select_workers` `:108`→[`89`](../../src/audrey/pipeline/deep_panel.py#L89);
    `_messages_for_subtask` `:232`→217; registry fallback `:284`→285 (ok);
    round-robin `:295` ✓; gate.acquire `:151` ✓.
  - synthesize.py `pick_synthesizer` `:92`→[`82`](../../src/audrey/pipeline/synthesize.py#L82);
    `_build_synth_messages` `:104`→94; no_drafts `:186`→192; candidates
    loop `:211`→201.
  fix: re-anchor; offsets small and consistent within each file.

#### Lesson 7 (classification + routing) — accuracy audit 2026-06-02 (+2)

Heaviest-cite lesson (29 flags). Concepts all accurate (task_type vs
mode split, keyword-before-router cheap pass, tool-mention → general,
review-override → reasoning, escalation guards, separate streaming
driver). But it has the **widest cite drift after Lesson 4**, two
convention issues, one internal inconsistency, and a small factual
imprecision.

- **`should-fix` (cite drift) — many offsets, several large.**
  - messages.py `last_user_text` cited `:20` → [`17`](../../src/audrey/pipeline/messages.py#L17).
  - complexity.py `count_tokens` cited `:24` → [`60`](../../src/audrey/pipeline/complexity.py#L60)
    (+36); `is_complex` cited `:41` → [`114`](../../src/audrey/pipeline/complexity.py#L114)
    (+73); list-content cited `:114` → 106-107.
  - classify.py: code-sig `:32`→33, reasoning `:46`→48, vl `:52`→54,
    `_REVIEW_OVERRIDE` `:59`→61, `_tool_mention_signal` `:87`→75,
    `keyword_classify` `:91`→93, `router_classify` `:141`→127 (-14, and
    the `user_text[:2000]` cap is at 143 not 141), `classify` `:184`→186,
    `classify_with_registry` `:233`→234, `tool_names` `:253`→254.
  - graph.py: `node_complexity` `:200`→213; `route_after_fast_path`
    `:362`→363; **audrey_fast guard `:343`→366** (+23); `too_short`
    `:364`→387; **`route_after_complexity` cited `:419` for the function
    → it's at [`360`](../../src/audrey/pipeline/graph.py#L360)** (the 419
    region is the `add_conditional_edges` wiring, ~429-432; the lesson
    conflates the two).
  - config.yaml complexity threshold `:258`→261.
  fix: re-anchor via grep; offsets are non-uniform across files.

- **`should-fix` (internal inconsistency) — §2.2 prose/link disagree.**
  Lesson line 181: prose says "Open `graph.py:390`" but the link target
  is `#L404`, and the actual `g.add_node(...)` block is ~414. Three
  different numbers for one location. fix: pick the real line (the node
  block) and make prose + link match.

- **`should-fix` (convention) — test-file cites in prose.** Lesson lines
  114 and 322 cite `tests/test_classify.py:61`. Violates the
  no-test-walkthroughs rule. **Meta-finding: the convention checker does
  NOT catch test cites** (it reported 0 findings for this lesson) — so
  this rule is unenforced by tooling across the whole corpus; worth a
  grep sweep for `tests/` in all lessons during the drain. fix: drop the
  two cites; the tool-mention-ordering point is already made in prose.

- **`nit` (factual imprecision) — `_ROUTER_SYSTEM` is an alias.** Lesson
  lines 347-349 say "the router prompt is `_ROUTER_SYSTEM` at
  classify.py:122" and quote JSON-return instructions. The code at 122
  is `_ROUTER_SYSTEM = CLASSIFIER_SYSTEM` — an alias; the actual prompt
  text lives in [`prompts.py:50`](../../src/audrey/pipeline/prompts.py#L50)
  (`CLASSIFIER_SYSTEM`). Reader who opens classify.py:122 sees a one-line
  alias, not the quoted prompt. fix: note it aliases `CLASSIFIER_SYSTEM`
  and point the prompt-text quote at prompts.py.

#### Lesson 6 (the model layer) — accuracy audit 2026-06-02 (+2)

Despite the most cite flags going in (15), Lesson 6 is the **most
accurate code-walk lesson** — no factual or conceptual errors, every
mechanism verified (exponential backoff, monotonic clock, passing
`health.is_healthy` as a predicate, error-draft-not-raise in deep
workers, synth primary/fallback/degrade-to-longest-draft). Most cites
land dead-on: registry.py 16/22/36/53/56/59/69/84 all ✓; health.py
18/44/50/53 ✓; ollama.py 29/44/56/71/87/114 ✓; fast_path.py 31/37 ✓;
deep_panel.py 52/103/109/207 ✓. Convention checker: 0 findings. Drift is
light (mostly ±10) but real:

- **`should-fix` (cite drift) — a cluster of small offsets.**
  - synthesize.py §2.8 anchor cited `:98` → the section's real targets
    are [`pick_synthesizer:82`](../../src/audrey/pipeline/synthesize.py#L82)
    and `synthesize:169`; `:201` (candidates) is correct; `:250`
    (best = max draft) → [`240`](../../src/audrey/pipeline/synthesize.py#L240).
  - ollama.py `_json_object` cited `:201` → [`208`](../../src/audrey/models/ollama.py#L208);
    `embed` cited `:154` → [`161`](../../src/audrey/models/ollama.py#L161).
  - fast_path.py `use_tools` cited `:69` → 71; `pick_fast_model` call
    cited `:70` → `run_fast_path` is at 43 / `pick_fast_model` at 31;
    `gate.acquire` cited `:83` → 85; `except` cited `:90` → 92 (uniform
    ~+2).
  - deep_panel.py `_run_one_worker` cited `:141` → [`121`](../../src/audrey/pipeline/deep_panel.py#L121)
    (-20); `select_workers` is at 89 (the §2.7 "open at :52" lands on
    `_POOL_KEYS`, fine as a section start); cloud-cap cited `:113` → 114;
    the two "fallback starts at" cites `:248` / `:342` land on function
    signatures, not the fallback branch (the registry-fallback comment is
    at [`379`](../../src/audrey/pipeline/deep_panel.py#L379)).
  - graph.py forced-fast cited `:204` → [`217`](../../src/audrey/pipeline/graph.py#L217);
    `run_fast_path` call cited `:222` → [`244`](../../src/audrey/pipeline/graph.py#L244).
  fix: re-anchor the above; recommend a grep pass since offsets vary by
  file. None of the *snippets* are wrong — only the navigation numbers.

#### Lesson 5 (configuration + startup) — accuracy audit 2026-06-02 (+2)

Most cite-disciplined code-walk lesson — uses `#L` anchors and
explicitly groups boot steps "by role rather than by exact line." All
prose/concepts verified accurate (graceful-degradation split, closure /
in-place `reg.by_name` mutation, `@lru_cache(maxsize=1)`, app.state
lifetime reasoning). Most cites correct: config.py EnvOverrides (#L23),
SettingsConfigDict (#L26), ollama_host (#L31, the cite-checker false
positive), _apply_env_overrides (#L73), _load_yaml (#L121) all land
right. Convention checker: 0 findings. Three cites drifted:

- **`should-fix` (cite drift) — three stale line numbers.**
  - `config.py#L136` for `get_config()` (lesson line 148) → actually
    [`163`](../../src/audrey/config.py#L163) (`@lru_cache` at 162).
  - `main.py#L47` for `lifespan` (lesson line 237) → actually
    [`51`](../../src/audrey/main.py#L51) (`@asynccontextmanager` at 50).
  - `main.py#L229` for `rediscover_tools` (lesson line 376) → actually
    [`311`](../../src/audrey/main.py#L311). (The `reg.by_name.clear()` /
    `.update()` snippet shown is accurate, lines 324-325.)
  fix: 136 → 163; 47 → 51; 229 → 311.

#### Lesson 4 (request lifecycle) — accuracy audit 2026-06-02 (+2)

**Heaviest-drift lesson in the corpus.** Lesson 4 predates the file
growth *and* was renumbered (2026-05-05); essentially every "scroll to
line N" navigation pointer is now wrong. The good news: the *code
snippets shown* still match current source verbatim (StateGraph block,
edge list, `node_datetime`, the `OllamaError` catch), the mental model
is correct, and the model names (`qwen3.6:35b`) are current. Only the
line numbers rotted — but there are ~15 of them, so this needs a
dedicated re-anchoring pass.

- **`should-fix` (pervasive nav-line drift).** Correction table
  (`openai.py` cites are off by ~+107, `graph.py` by ~+37):
  - L48/59/226: "line 124" / "lines 124-158" (`@router.post`) → **231**.
  - L64: "`/v1` prefix on line 63" → `APIRouter(prefix="/v1")` at **84**.
  - L67: "`ChatCompletionRequest` defined at line 84" → **164**.
  - L76: "`if payload.stream:` (lines 152-156)" → **289**.
  - L227: "VIRTUAL_MODELS (line 67)" → **88**.
  - L234: "`_stream_via_pipeline()` (line 240)" → **580**.
  - L256: "`_stream_openai` (line 809)" → **1348**.
  - L265: "`pipeline_total` increments (line 181)" → re-locate (the
    metric inc lives in the finally blocks ~1203 / in
    `_run_graph_with_metrics` ~482-505).
  - L284 (table): "`routes/openai.py:124`" → **231**.
  - L288 (table): "`routes/openai.py:401` (`_stream_deep_with_banners`)"
    → both wrong: 401 is a `_passthrough_stream_sse` param;
    `_stream_deep_with_banners` is at **898**. (This is the one the cite
    checker flagged.)
  - L402: "`routes/openai.py:280-284`" for the `OllamaError` catch →
    **524** (`except OllamaError` in `_generate_via_pipeline`).
  - graph.py: L90 "line 376" (StateGraph block) → **413**; L108 node-def
    line list "128, 138, 171, 188, 208, 235, 252, 284, 303, 369" → all
    shifted (defs now run from ~128 onward but the block starts later;
    re-grep `^async def node_`); L110 "lines 388-407" (edges) →
    **425-442**; L176/L285 "graph.py:139" / "graph.py:376" → node_datetime
    and StateGraph block both shifted (verify each).
  fix: re-anchor every numeric cite above. Recommend doing this with a
  fresh grep pass rather than arithmetic, since two files drifted by
  different amounts.

- **`should-fix` (convention) — forward/by-number lesson ref.** Lesson
  line 274: "Lesson 14 walks the exact frame format." Caught by
  `check-lesson-conventions.py`. (Lesson 14 is *earlier*-numbered work
  but the rule is no by-number refs; also note Lesson 14's actual
  successor for the frame format is now Lesson 15, so this points at the
  wrong lesson anyway.) fix: "a later lesson covers the exact frame
  format" — or, more accurately, "Lesson 15 covers" reworded to "when we
  open the OpenAI route in detail."

#### Lesson 3 (foundations: satellite libs) — accuracy audit 2026-06-02 (+2)

Cleanest lesson audited so far. Every falsifiable claim verified:
embedding dims (768-d nomic / 512-d CLIP, config.yaml:209-210 +
embed.py); the `_probe_owui` httpx example's 502/401 detail strings
match auth.py exactly ("Auth backend unreachable.", "Token rejected by
OWUI.", "Auth probe failed (…).") — the "slightly simplified" omissions
(non-JSON guard, missing-email guard, role allowlist) are fair for an
httpx primer; `asyncio_mode = "auto"` at pyproject.toml:113; **"four
alert rules" is exactly right** (AudreyPipelineErrorRate,
ToolCallErrorRate, ToolCallLatencyP95, CloudModelErrorRate in
monitoring/prometheus-rules/audrey.yml). Convention checker: 0 findings.

- **`consider` (convention tension) — §4 names and walks four specific
  tests.** Lesson lines 363/370/378/391 reference
  `test_two_user_round_robin_skips_last_granted`,
  `test_authed_user_has_no_id_field`, `test_kb_embed_ssrf.py`, and the
  `_BREVITY_CUES` test in `test_reflect.py`, with scenario walkthroughs.
  All exist. The no-test-walkthroughs convention (no `tests/` cites, no
  scenario walkthroughs) is written for lessons 4+; here it collides
  with the fact that **this is the pytest foundations primer** — teaching
  the test framework with zero reference to the real suite would be
  artificial. Judgment call for the author: probably keep the
  regression-guard *concept* + one paraphrased scenario, drop the bare
  filenames, so it teaches pytest without becoming a test-walkthrough.
  Not a clear violation given the section's subject. No factual error.

#### Lesson 2 (foundations: orchestration libs) — accuracy audit 2026-06-02 (+2)

- **`should-fix` (factual error) — Lesson 2 teaches a `temperature`
  bounds-check that doesn't exist in the code.** Lesson lines 331 and
  390 present `temperature: float | None = Field(default=None, ge=0.0,
  le=2.0)` as "drawn from the real `ChatCompletionRequest` schema" and
  call it "the line preventing a malicious client from passing
  `temperature=999`." The actual schema has
  [`temperature: float | None = None`](../../src/audrey/routes/openai.py#L168)
  — **no `ge`/`le` bounds**. So the lesson's flagship Pydantic example
  describes a validation guard Audrey does not have; a reader who greps
  for that bound won't find it, and the "stops temperature=999" claim is
  false. This is the most serious Lesson 2 finding — not drift, a
  description of behavior the code lacks. fix: either (a) change the
  lesson example to a `Field(...)` constraint that *does* exist in the
  schema (e.g. `messages = Field(min_length=1)`, which is real and
  already used as the other example), or (b) keep the temperature
  example but relabel it as illustrative-of-Pydantic-capability, not
  "from the real schema" — and decide separately whether the code
  *should* bound temperature (arguably yes; Ollama clamps, but a 422 is
  cleaner than relying on the upstream).

- **`should-fix` (cite drift) — schema cite off by ~+80.** Lesson line
  405 cites `routes/openai.py:78-100` for `ChatMessage` /
  `ChatCompletionRequest` → actually
  [`158-191`](../../src/audrey/routes/openai.py#L158). (Same root cause
  as the Lesson 15 drift — the file grew.) fix: 78-100 → 158-191.

- **`should-fix` (convention) — names three `tests/` files in prose.**
  Lesson line 468 cites `tests/test_classify.py`, `tests/test_reflect.py`,
  `tests/test_fair_gate.py` as evidence nodes are individually testable.
  All three exist, but the no-test-walkthroughs convention says no
  `tests/` cites in lesson prose. fix: keep the point ("each node takes
  state in / returns state out, so it's testable in isolation") and drop
  the three filenames.

- **`nit` (convention) — specific + now-stale count.** Lesson line 556
  calls `models/ollama.py` a "~180-line client"; it's
  [218 lines](../../src/audrey/models/ollama.py). Specific counts are
  discouraged in lessons and this one has drifted. fix: "a small
  purpose-built client" / "a few hundred lines" or drop the number.

- **`nit` (convention) — by-number forward ref in intro.** Lesson line 9
  "Lesson 3 covers the satellite libraries." Lesson 3 is N+1 from Lesson
  2, which the rule permits as "the next lesson," so the convention
  checker doesn't flag it — but it's still a by-number reference. Low
  priority; fix only if sweeping. (The footer link at line 581 to Lesson
  3 is fine — it's the immediate-next-lesson handoff.)

#### Lesson 1 (foundations: language) — accuracy audit 2026-06-02 (+2)

Primer lesson; few line-cites, so the audit is fact-checking the
Audrey-specific claims rather than line numbers. Almost everything
verified clean (see below). One internal contradiction.

- **`should-fix` — §3 "Where you'll see it" miscategorizes
  `WorkerDraft` as a dataclass.** Lesson line 587 lists `WorkerDraft`
  under the **Dataclasses:** heading in `state.py`, but it is a
  `TypedDict` ([`pipeline/state.py:16`](../../src/audrey/pipeline/state.py#L16),
  `class WorkerDraft(TypedDict, total=False)`). The same lesson
  correctly calls it a TypedDict twice elsewhere (lesson lines 558 and
  660, the latter explicitly: "yes, `WorkerDraft` is TypedDict, not
  dataclass"). So this is a self-contradiction, and §3's list is the
  wrong one. fix: drop the `WorkerDraft` bullet from the Dataclasses
  list at lesson line 587 (the other three — `AuthedUser`,
  `ReactResult`, `KBHit` — are correctly dataclasses).

- **`nit` — verified-clean claims, logged so a re-audit can skip
  them.** All of these were checked against source and match:
  `ReactResult` `@dataclass(slots=True)` (react.py:54-55); `KBHit`
  `@dataclass(slots=True)` (qdrant.py:46-47); `AuthedUser`
  `@dataclass(slots=True)` with `email`/`role`/`owui_id`
  (auth.py:63-73); `KeywordSignal` `@dataclass(slots=True, frozen=True)`
  (classify.py:68-69, lesson says "frozen" ✓); `PhaseTicker` exists in
  banners.py:139 (a class with `__aenter__`/`__aexit__`, used as
  `async with` — lesson doesn't over-claim the mechanism); fair-gate
  `acquire` is `@asynccontextmanager`, takes `user_id`, cloud
  (`location != "local"`) short-circuits to a bare `yield`, real shape
  is `try/yield/finally(_release)` (fair_gate.py:90-139) — the lesson's
  simplified snippet is explicitly labeled "simplified," so the minor
  structural difference (enqueue/await placement) is fair; ASYNC240
  `Path.exists()` cases confirmed in ingest.py:68 and reconcile.py:123.
  No action.

- **`should-fix` — two forward-references to later lessons by number
  (convention violation).** Lesson line 477: "We'll meet this properly
  in **Lesson 5** when we open `main.py`." Lesson line 694: "**Lesson
  3** then covers the satellite libraries." The convention rule (and
  AGENTS.md hard rules) say never forward-reference later lessons by
  number — use "the next lesson" / "a later lesson." Caught by
  `check-lesson-conventions.py`. fix: 477 "Lesson 5" → "a later lesson
  (when we open `main.py`)"; 694 "Lesson 3" → "a later lesson." (Note:
  line 690's link to Lesson 2 is fine — Lesson 2 is the immediate next
  lesson and the rule's fix text permits "the next lesson"; the
  in-prose "Lesson 3" two lines later is the violation.)

#### Lesson 15 (routes/openai.py) — accuracy audit 2026-06-02 (+1)

Source-cite drift. `routes/openai.py` grew after Lesson 15 shipped (the
recent minimax-m3 / passthrough streaming-accumulation commits added
lines), so a band of cites now point a few lines off. The lesson *prose*
is still accurate — only the line numbers rotted. The cite checker
reports all of these as `DRIFT?` (advisory), not `DRIFT` (confident),
because most cited lines aren't `def`/`class` landmarks, so it won't
auto-propose a fix; they need manual re-anchoring. **`should-fix`**
(stale cites mislead a reader following along).

- **`should-fix` — cancellation trace cites drifted ~+9 lines (§2.7,
  Q3, Q5).** The whole cancellation section's line numbers are wrong:
  - §2.7 step 2 / Q5 cite the `except asyncio.CancelledError` catch at
    `:1172` → actually [`:1181`](../../src/audrey/routes/openai.py#L1181)
    (`:1172` is now `yield "data: [DONE]\n\n"`).
  - §2.7 step 3 / Q5 cite the inner synth-cancel `try/finally` at
    `:1164-1170` → actually the `finally` is
    [`:1173-1179`](../../src/audrey/routes/openai.py#L1173) (`:1164` is
    now the unrelated `footer = tool_summary_block(...)`).
  - Q3 cites the exception handler yielding `[ollama error]`/`[internal
    error]` at `:1179-1190` → actually `OllamaError`
    [`:1188-1193`](../../src/audrey/routes/openai.py#L1188) and the bare
    `Exception` [`:1194-1199`](../../src/audrey/routes/openai.py#L1194).
  - Q3 cites the archive write at `:1199-1213` → actually the `finally`
    runs `:1200-1222`, `archive_turn` at
    [`:1213`](../../src/audrey/routes/openai.py#L1213).
  fix: re-cite the four ranges to 1181 / 1173-1179 / 1188-1199 / 1213.

- **`should-fix` — `_stream_deep_with_banners` and its SSE helpers
  drifted +7 (§2.5, §2.3).** §2.5 says "Open `routes/openai.py:891`" →
  the function now starts at
  [`:898`](../../src/audrey/routes/openai.py#L898). §2.3 cites the
  `_delta_frame`/`_stop_frame` helpers at `:927-941` → actually
  [`:934-948`](../../src/audrey/routes/openai.py#L934).
  fix: 891 → 898; 927-941 → 934-948.

- **`should-fix` — helper cites drifted (§2.8, Q4, §2.9, Q5).**
  - `_ollama_to_openai_tool_calls` cited at `:854` (§2.8 + Q4, two
    places) → actually [`:861`](../../src/audrey/routes/openai.py#L861).
  - `_phase_dispatch` background `panel_task` cited at `:1018` (Q5) →
    `panel_task = asyncio.create_task(_phase_dispatch(...))` is at
    [`:1024`](../../src/audrey/routes/openai.py#L1024) (`:1018` is a
    config read).
  - `_options_from_state` cited at `pipeline/graph.py:408` (§2.9) →
    actually [`graph.py:450`](../../src/audrey/pipeline/graph.py#L450)
    (+42, the largest single drift).
  - `payload.user` drift-log cited at `:257` (§2.1) → the `log.debug`
    is at [`:258`](../../src/audrey/routes/openai.py#L258); `:257` is
    the guarding `if`. Minor; arguably fine as "around 257."
  fix: 854 → 861 (×2); 1018 → 1024; graph.py:408 → 450; optionally
  257 → 258.

- **`should-fix` — §2.9 code snippet is stale (predates the docstring
  shipped during Lesson 15's own audit).** The lesson shows
  `_options_from_request` as a bare body with no docstring, but the
  function now carries a 7-line docstring (the "Sibling:" cross-ref
  added 2026-06-02 to resolve the §2.9 finding itself). The lesson's
  surrounding prose already explains the sibling relationship, so the
  snippet should either show the docstring or be trimmed to "the body
  maps three knobs" — as-is it contradicts the file it cites.
  fix: regenerate the snippet from
  [`:808`](../../src/audrey/routes/openai.py#L808) or drop the fenced
  block in favor of the prose that follows.

- **`consider` — §2.4 attributes the complexity gate to the
  non-streaming path, but that path has none.** §2.4 is titled "The
  non-streaming path" and its flow diagram shows the `forced fast /
  forced deep / audrey_auto` decision happening there; the OWUI wrinkle
  paragraph then cites
  [`:633`](../../src/audrey/routes/openai.py#L633) — which lives in
  `_stream_via_pipeline` (the *streaming* function). `_generate_via_
  pipeline` (506-578) contains no gate logic at all; it delegates the
  whole decision to the LangGraph complexity node. So a reader who
  opens `:633` while reading "the non-streaming path" lands in the
  wrong function. The diagram is conceptually true (the graph does make
  that choice) but mis-attributes it to route code. Two clean options:
  (a) move the OWUI wrinkle + gate diagram to §2.5 (streaming), where
  the code it cites actually lives, and have §2.4 say "the gate runs
  inside the graph — see `pipeline/complexity.py`"; or (b) keep it in
  §2.4 but reframe as "this decision is made by the complexity node the
  graph runs; the streaming path at `:633` mirrors it inline."

- **`nit` — cite checker doesn't audit this lesson's cites when run on
  the source file (no confident-drift signal).** All drift above shows
  as `DRIFT?` not `DRIFT`, so the "zero confident drift" line in
  PROJECT_STATE is technically true but misleading — the checker simply
  can't re-anchor cites that land on non-landmark lines (it only
  auto-fixes `def`/`class`/decorator targets). Worth knowing that a
  green-ish cite-check run does **not** mean a lesson's line numbers are
  current when the cites point at expression-level lines. No code fix;
  documents a tooling blind spot.

#### Lesson 15 (routes/openai.py) — fresh-eyes pass findings

- **`optimization` — the synth event-loop polls with 50ms timeouts
  instead of using `wait` with FIRST_COMPLETED.**
  [`routes/openai.py:1080`](../../src/audrey/routes/openai.py#L1080)
  loops on `await asyncio.wait_for(events_q.get(), timeout=0.05)`
  while draining the banner queue between attempts. Same pattern
  in [`_drain_q_until_task` at routes/openai.py:1240](../../src/audrey/routes/openai.py#L1240).
  Functionally correct — the 50ms timeout is small enough that
  banner emissions don't visibly lag — but it's a spin-poll with
  ~20 wakeups/sec per active deep stream. The structurally cleaner
  shape is `asyncio.wait({task_get, banner_q_get}, return_when=
  FIRST_COMPLETED)`: zero polling, wakes on either event. Cost is
  near-zero (a few hundred microseconds of CPU per stream), so
  the optimization is principle-driven, not performance-driven.
  **Deferral trigger** (2026-06-02): revisit when Grafana's asyncio
  task-wakeup count climbs under streaming load in a way that
  correlates with concurrent deep streams. Twenty wakeups/sec/stream
  is invisible at single-digit concurrency; the rewrite has real
  complexity risk (interleaving two queues with explicit drain
  semantics, easy to break a streaming corner case), so it should
  be paid only when measurement justifies it.

### Deferred

*(none)*

### Resolved

#### Full-course accuracy sweep 2026-06-24 — lesson-doc fixes

- **`should-fix` — Lesson 1, 4, 6, 7, 8, 9, 13, 14, and 15 documentation drift
  -> `resolved` 2026-06-24.** Updated the maintainer course for the current
  OpenAI route package split, multimodal `ChatMessage.content`, Ollama message
  conversion via `_to_ollama_messages`, deep-intent/image/OWUI routing order,
  deep-mode log shape, `agentic.react` config anchors, streaming archive
  anchors, and passthrough/tool-call response helpers. Verification: lesson
  conventions clean for the edited files; maintainer-course citation checker
  reports 332 checked, 252 ok, 0 confident drift, 80 advisory `DRIFT?`, 0
  broken.

#### Lesson 16 (custom-tools sidecar, `tools-server/`) — pre-write drain 2026-06-03

- **`should-fix` — real emails baked into source/openapi.** Resolved
  2026-06-03. `app.py` `memory_store` description rewritten to drop
  `user:bart@proton.me` (and to note the user scope is auto-filled — see
  #5); `db.py:40` comment's substring-collision example changed to
  `user:al`/`user:alice`. The `memory_store` description ships in
  `/openapi.json` and is sent to the model, so this removed a real address
  from the model-facing surface, not just a doc.

- **`bug` (latent) — `_split_long` crash on `overlap >= max_chars`.**
  Resolved 2026-06-03 (decision: validate at Settings load). Added a
  `@model_validator(mode="after")` on `Settings`
  (`tools-server/settings.py`) that raises at boot if
  `CHAT_ARCHIVE_CHUNK_OVERLAP_CHARS >= CHAT_ARCHIVE_CHUNK_MAX_CHARS`, naming
  both env knobs. Fails fast at sidecar startup instead of crashing on a
  future archive write. Verified: bad pairing raises `ValidationError`,
  default config (100/2500) loads.

- **`smell` — 3 pre-existing ruff errors in `tools-server/`.** Resolved
  2026-06-03. Auto-fixed I001 (import sort) + UP037 (quoted annotations);
  the intentional broad `except Exception` at `db.py:209` got a scoped
  `# noqa: BLE001` with the best-effort reason. `ruff check tools-server/`
  now clean. (Repo-wide ruff still shows the accepted `kb/ingest.py`
  ASYNC240 hints — unrelated.)

- **`consider` — `user` field required in schema but model-shouldn't-fill /
  auto-overridden.** Resolved 2026-06-03 (decision: document). The `user`
  field descriptions on `memory_recall`/`memory_search`/`chat_history_search`
  and the `memory_store` tool description now say the user scope is filled in
  automatically by Audrey. Reduces the chance the model asks for or
  hallucinates a user id. Pairs with the `_USER_SCOPED_TOOLS` override the
  lesson will teach.

#### Lesson 17 (admin routes + auth tail, `routes/admin.py`) — pre-write drain 2026-06-03

- **`should-fix` — module docstring listed 4 of 6 endpoints.** Resolved
  2026-06-03. Added `/chat_archive/prune` and `/chat_archive/stats` to the
  docstring's Endpoints block, plus a note that `/v1/tools/rediscover` is
  admin-gated but lives in `main.py` (it closes over `app.state`).
  [`routes/admin.py`](../../src/audrey/routes/admin.py).

- **`smell` — inline `import httpx as _httpx` in two handlers.** Resolved
  2026-06-03. Hoisted a single top-level `import httpx`; the two inline
  imports and the `_httpx` alias removed.

- **`consider` — chat_archive routes returned HTTP 200 with `{"error":...}`
  on unavailable/unregistered.** Resolved 2026-06-03 (decision: raise).
  Both `chat_archive_prune` and `chat_archive_stats` now
  `raise HTTPException(503, detail=...)` for the `archive_client is None`
  and `host_url() is None` cases, so the failure rides the HTTP status code
  (a monitoring script no longer reads a downed archive as success). The
  downstream `prune_failed`/`stats_failed` return-dicts (which carry the
  upstream status) are unchanged. **Behavior change → new coverage:**
  `test_admin_routes.py` had zero tests on these routes; added 4
  parametrized failure-path tests (client-unavailable + tool-not-registered
  × both handlers). 449 pytests pass, ruff clean.

#### Lesson 16 (streaming banners, `banners.py`) — pre-write drain 2026-06-03

- **`consider` — redundant `return None` + misleading comment in
  `__aexit__`.** Resolved 2026-06-03. Dropped the explicit `return None`
  (the function falls off the end ⇒ None anyway) and reworded the comment
  to "Falling off the end returns None (falsy), so `__aexit__` does not
  suppress — any in-flight exception propagates to the caller."
  [`pipeline/banners.py`](../../src/audrey/pipeline/banners.py). Teaching-
  adjacent: the lesson explains this mechanism, so the comment now states
  it correctly instead of implying the `return None` is a deliberate guard.

- **`smell` — no-op `try/except asyncio.CancelledError: raise` in
  `_tick_loop`.** Resolved 2026-06-03. Removed the wrapper; a bare loop
  propagates the cancellation identically. Added a comment noting the task
  is cancelled by `__aexit__` and the resulting `CancelledError` ends the
  loop with no handler needed.

- **`consider` — backpressure phrasing in the module docstring not obvious
  from the code (tick `await`s the emitter).** Resolved 2026-06-03. Added a
  comment at the `await self._emit(".")` in `_tick_loop` clarifying it
  blocks the tick task only, never the caller doing model work — which is
  why the model side never stalls on banner emission. The lesson's trace
  (§5) leans on this; the comment makes the code self-explain it.

- **`nit` — docstring checkmark drawn inside the header italics, code emits
  it outside.** Resolved 2026-06-03. Fixed the module-docstring example
  (`> _Thinking_ ✅`, mark outside italics) and the `__aexit__` lifecycle
  note to match what the code actually produces.

Verification: 445 pytests pass, ruff clean on `banners.py`, cite-check 0
drift. #4 (worker_ok/fail vs tool_summary_block formatting asymmetry)
accepted — see Accepted section.

#### Lesson 15 (routes/openai.py) — fresh-eyes findings actioned 2026-06-02

- **`nit` — defensive guard for a documented impossibility (item 3b).**
  Resolved 2026-06-02. Tightened `synthesize_stream`'s contract
  rather than removing the route's defensive branch: the docstring
  at [`pipeline/synthesize.py:272`](../../src/audrey/pipeline/synthesize.py#L272)
  now formally states *"`first_token` precedes every `delta`"* as
  an ordering contract callers may rely on, and the implementation
  enforces it structurally (every `delta` yield is guarded by a flag
  set only after emitting `first_token`). The route's defensive
  branch at [`routes/openai.py:1101`](../../src/audrey/routes/openai.py#L1101)
  is kept as a now-provably-unreachable safety net with a comment
  explaining that's what it is and why: if a future refactor
  violates the contract, the visible symptom is a missing banner ✅,
  which points back to the broken invariant rather than silently
  dropping content.

- **`should-fix` — `deep_worker` timeout selection duplicated between
  streaming and non-streaming deep paths (item 2).** Resolved
  2026-06-02. Extracted `pick_panel_timeout(cfg, pool_key) -> float`
  in [`pipeline/deep_panel.py:73`](../../src/audrey/pipeline/deep_panel.py#L73),
  co-located with the existing `pool_key_for`. The duplicated
  expression in [`pipeline/graph.py:291`](../../src/audrey/pipeline/graph.py#L291)
  and [`routes/openai.py:1007`](../../src/audrey/routes/openai.py#L1007)
  both collapsed to a single `pick_panel_timeout(cfg, pool_key)`
  call. Four hermetic tests added to
  [`tests/test_config_validation.py`](../../tests/test_config_validation.py)
  pinning the helper's behavior: cloud pool uses cloud timeout,
  mixed/local pools use deep_worker timeout, defaults of 120/240
  when keys missing. The dead `cloud_timeout` local in `graph.py`
  was also removed (ruff caught it). 441 tests pass (up from 437).

- **`consider` — docstring cross-references for parallel helpers
  (items from `_options_from_request`/`_options_from_state` and
  `_phase_thinking`/`node_datetime` drive-by queues).** Resolved
  2026-06-02. Added cross-reference docstrings to all four sites:
  - [`routes/openai.py:_options_from_request`](../../src/audrey/routes/openai.py#L808)
    points to `pipeline.graph._options_from_state`.
  - [`pipeline/graph.py:_options_from_state`](../../src/audrey/pipeline/graph.py#L446)
    points back.
  - [`routes/openai.py:_phase_thinking`](../../src/audrey/routes/openai.py#L1264)
    documents the parallel structure with the graph nodes.
  - [`pipeline/graph.py:node_datetime`](../../src/audrey/pipeline/graph.py#L138)
    points to `_phase_thinking` so a future contributor changing
    the ordering hits the cross-reference from either side.

#### Lesson 15 (routes/openai.py) — deferred items drained

- **`consider` — streaming-path cancellation propagation.**
  Accepted 2026-06-02 as **not a gap**. Fresh-eyes trace through
  `_stream_deep_with_banners`: the route generator catches
  `asyncio.CancelledError` at
  [`routes/openai.py:1181`](../../src/audrey/routes/openai.py#L1181),
  the inner `try/finally` cancels the synth producer task at
  [`routes/openai.py:1174-1179`](../../src/audrey/routes/openai.py#L1174)
  (`synth_task.cancel()` + `await synth_task` to wait for the cancel
  to land), and the outer `async with inflight.slot(user_id)` /
  `gate.acquire()` release paths fire via context-manager exit (the
  fair-gate cancellation handling is already covered in Lesson 14
  at `fair_gate.py:120-134`). The producer cancellation propagates
  through the `synthesize_stream` async generator into the
  `httpx.AsyncClient` stream, which httpx documents as propagating
  CancelledError into the underlying connection close. The original
  concern — "cloud workers burning paid time after the user left" —
  is real in *theory* (the cancel doesn't reach the upstream provider
  and tell it to stop billing), but Audrey can't help that: once a
  cloud provider has accepted a request, the meter runs until they
  decide to stop. The local cleanup is solid; the remote-billing
  side is out of Audrey's control. No code change needed.

- **`nit` — `_options_from_request` is a near-duplicate of
  `_options_from_state`.** Accepted 2026-06-02 as **document the
  divergence**. Fresh-read confirms the two helpers do the same
  conceptual mapping (`temperature`, `top_p`, `max_tokens` →
  `options` dict for the Ollama client), but the input shapes are
  genuinely different: `_options_from_request` reads a Pydantic
  `ChatCompletionRequest` (typed access via `req.temperature`),
  while `_options_from_state` reads the LangGraph state dict
  (`state.get("temperature")`). Unifying would require either
  (a) introducing a third "view" type that both call sites convert
  to before the helper, or (b) accepting `Any` and using `getattr`
  fallbacks — both worse than the current "two small functions, one
  per shape" arrangement. The right resolution is a docstring on
  each side naming its sibling so a future maintainer doesn't change
  one without checking the other. Drive-by fix queued for next time
  this file is touched for substantive work.

- **`consider` — `VIRTUAL_MODELS` validation lives in the route, not
  the schema.** Accepted 2026-06-02 as **keep the route check**.
  The current implementation at
  [`routes/openai.py:245`](../../src/audrey/routes/openai.py#L245)
  emits a descriptive error (`f"Unknown model {payload.model!r}. Supported virtual models: {list(VIRTUAL_MODELS)}."`)
  which a Pydantic `Literal[...]` schema check would not. Passthrough
  also complicates the schema-validation story: `audrey_passthrough/<x>`
  is a prefix-match, not a literal, so the route branch
  ([`routes/openai.py:242`](../../src/audrey/routes/openai.py#L242))
  has to run *before* any model-validity check anyway. With
  passthrough in the picture, pushing literal validation into the
  schema would create two error paths (Pydantic 422 for non-prefix
  unknowns, route 400 for malformed passthrough) instead of the
  current single 400 path. Status quo is the right answer.

- **`consider` — memory recall + datetime injection have two
  entry points that must stay in sync** (cross-lesson deferral from
  Lesson 13). Accepted 2026-06-02 as **document the parallel
  structure**. Fresh-read confirms
  [`routes/openai.py:1264-1313`](../../src/audrey/routes/openai.py#L1264)
  (`_phase_thinking`) and
  [`pipeline/graph.py:139`](../../src/audrey/pipeline/graph.py#L139)
  (`node_datetime` + `node_memory_recall` + `node_plan`) call the
  same underlying helpers (`datetime_system_message`,
  `recall_for_request`, `memory_system_message`,
  `compose_system_messages`) in the same order. The duplication is
  in the *orchestration*, not the *building blocks* — those were
  already factored out in Lesson 13's audit pass. The orchestration
  shapes are different enough that a shared helper would have to
  thread enough config + callbacks to be its own complication: graph
  nodes take state in/out, the streaming path returns a tuple. The
  classify-drift fix's precedent doesn't carry here because
  `classify_with_registry` is a pure function (inputs in → values
  out, no state shape); `_phase_thinking`'s shape is specific to its
  caller. Recommend a docstring on `_phase_thinking` cross-referencing
  the graph nodes (and vice versa) so a future contributor knows to
  check the sibling. Drive-by queued.

#### Lesson 13 (memory recall + context injection)

- **`smell` — `_last_user_text` duplicated across five files, all
  byte-identical.** Resolved 2026-05-27. Extracted to a new module
  [`pipeline/messages.py:last_user_text`](../../src/audrey/pipeline/messages.py).
  Removed the local defs from
  [`pipeline/memory.py`](../../src/audrey/pipeline/memory.py),
  [`pipeline/graph.py`](../../src/audrey/pipeline/graph.py),
  [`pipeline/synthesize.py`](../../src/audrey/pipeline/synthesize.py),
  [`pipeline/chat_archive.py`](../../src/audrey/pipeline/chat_archive.py),
  and [`routes/openai.py`](../../src/audrey/routes/openai.py); all nine
  callers now import the shared helper. The
  `pipeline.chat_archive.last_user_text` public name was dropped from
  `__all__` since its only external importer was the now-updated
  routes module. Pinned by six tests in
  [`tests/test_messages.py`](../../tests/test_messages.py)
  covering string content, multi-modal list content, most-recent
  selection, no-user-turn empty return, empty-list input, and the
  non-dict-parts-skipped path.

- **`consider` — `pipeline/memory.py` and `pipeline/context.py` had no
  direct test files.** Resolved 2026-05-27. Added
  [`tests/test_context.py`](../../tests/test_context.py) (four tests
  pinning `iso_now`'s shape — string, seconds precision, timezone
  offset, parses-as-aware — plus the `datetime_system_message`
  shape and the "treat as present" phrasing) and
  [`tests/test_memory.py`](../../tests/test_memory.py) (fourteen
  tests covering `recall_for_request`'s four skip paths, the
  `MAX_QUERY_CHARS` clamp, the error/non-JSON/non-list degradation
  paths, plus `memory_system_message`'s six branches: empty,
  hits-only, the 400-char value truncation, the `{user_id}`
  substitution, store-hint-without-user dropped, hits+hint
  composition order).

- **`nit` — `config.yaml:178` memory comment still said SQL LIKE.**
  Resolved 2026-05-27. [`config.yaml:178`](../../config.yaml#L178)
  comment changed from `# short — the LIKE query against SQLite is
  cheap` to `# short — semantic search through custom-tools is fast
  and shouldn't block the request hot path`. Pure comment fix;
  matches the embedding-backed reality the [`pipeline/memory.py:34`](../../src/audrey/pipeline/memory.py#L34)
  comment already described after its 2026-05-12 fix.

#### Lesson 8 (deep mode — planner, panel, synth, reflect)

- **`smell` — `_location_of` imported from `deep_panel` by `synthesize`.**
  Resolved 2026-05-24. Moved the function onto `ModelRegistry` as a new
  public method [`registry.location_of`](../../src/audrey/models/registry.py).
  Both callers (`select_workers` in
  [`deep_panel.py`](../../src/audrey/pipeline/deep_panel.py) and the synth
  attempt loop in [`synthesize.py`](../../src/audrey/pipeline/synthesize.py))
  now call through the facade. The cross-module
  `from audrey.pipeline.deep_panel import _location_of` import is gone.
  Lesson 6's old reference to `_location_of` was refreshed to point at
  the new method. Pinned by three new tests in
  [`tests/test_models.py`](../../tests/test_models.py) covering the happy
  path, the unknown-model `local` default, and the cross-task-type lookup.

- **`consider` — `pick_synthesizer` raises `KeyError` on missing config.**
  Resolved 2026-05-24. Implemented as startup validation rather than
  runtime degradation — same posture as the existing config fast-fail
  pattern. New helper
  [`_validate_deep_panel_pools`](../../src/audrey/config.py) walks every
  `deep_panel*` key in `cfg.raw` and raises `ValueError` listing every
  pool/task missing or empty in `synthesizer`; called from `get_config()`
  after the env merge, so a typo in `config.yaml` crashes the process at
  boot rather than 500ing the first deep request. Pinned by eight new
  tests in [`tests/test_config_validation.py`](../../tests/test_config_validation.py)
  covering missing/empty/multi-error/non-dict/empty-config cases.

- **`consider` — `select_workers` fallback path caps at 2 with no
  comment.** Resolved 2026-05-24 (comment-only). Added a code comment
  above the fallback loop in
  [`pipeline/deep_panel.py:run_panel`](../../src/audrey/pipeline/deep_panel.py)
  explaining the "2" mirrors the typical pool size and is the emergency
  path — bounded so we don't flood the GPU gate or burn cloud quota.
  The streaming twin in `run_panel_streaming` got a back-reference to
  the comment in `run_panel`. No behavior change.

- **`nit` — `pool_key_for` defaults silently to `"deep_panel"`.**
  Resolved 2026-05-24.
  [`pipeline/deep_panel.py:pool_key_for`](../../src/audrey/pipeline/deep_panel.py)
  now `log.warning`s when an unknown virtual model falls back to the
  default pool, naming the unknown model so the operator can find the
  typo. Pinned by two new tests in
  [`tests/test_config_validation.py`](../../tests/test_config_validation.py):
  known virtual models stay silent, unknown ones produce exactly one
  warning and still return a usable pool so the request answers.

- **`nit` — `drafts_block.count('--- draft ')` is brittle.**
  Resolved 2026-05-24.
  [`pipeline/synthesize.py:_build_synth_messages`](../../src/audrey/pipeline/synthesize.py)
  now takes an explicit `draft_count: int` kwarg threaded from
  `len(drafts)` at both call sites (`_try_synth` for the non-streaming
  path, `synthesize_stream` for streaming). A future change to the draft
  separator string can no longer silently break the count the
  synthesizer is told to expect. Existing
  [`test_synth_uses_override`](../../tests/test_prompts.py) updated for
  the new signature.

- **`nit` — `planner.py` truncates `user_text` at 4000 chars silently.**
  Resolved 2026-05-24. [`pipeline/planner.py:plan`](../../src/audrey/pipeline/planner.py)
  emits `log.debug("planner: user_text truncated from %d to 4000 chars
  for planning", len(user_text))` when the input exceeds the cap so the
  "why didn't the planner see the second half?" cases are diagnosable.
  Behavior unchanged.

- **`nit` — `planner.py` silently treats "only 1 subtask" as "no
  decomposition wanted."** Resolved 2026-05-24.
  [`pipeline/planner.py:plan`](../../src/audrey/pipeline/planner.py) now
  emits `log.debug("planner: only 1 subtask returned, treating as no
  decomposition")` when the parsed output has exactly one subtask before
  returning `[]`. The empty-decomposition outcome is unchanged; the
  planner's reasoning is no longer invisible.

- **`nit` — `_messages_for_subtask` docstring doesn't explain the
  multi-turn behavior.** Resolved 2026-05-24.
  [`pipeline/deep_panel.py:_messages_for_subtask`](../../src/audrey/pipeline/deep_panel.py)
  docstring expanded to spell out the multi-turn contract: "the last
  user message" is this turn's question (the one the planner just
  decomposed), earlier turns stay in place as history, and the
  no-user-message degenerate input appends a fresh user turn. No
  behavior change.

#### Lesson 12 (KB lifecycle — watcher, reconcile, uploads)

- **`consider` — `_stream_to_disk` checks the cap after extending.**
  Resolved 2026-05-25.
  [`routes/files.py:_stream_to_disk`](../../src/audrey/routes/files.py)
  now checks `written + len(chunk) > limit_bytes` *before* extending,
  so a single oversized chunk can't push `written` past the cap.
  Same defense-in-depth pattern we already applied to
  `kb/embed.py:_fetch_image`. Pinned by three new tests in
  [`tests/test_files.py`](../../tests/test_files.py):
  single-oversized-chunk pre-write reject, under-cap success path,
  cumulative-overflow second-chunk pre-write reject.

- **`consider` — `KBWatcher._delete_vectors` deletes from both
  collections regardless of suffix.** Resolved 2026-05-25.
  [`kb/watcher.py:_delete_vectors`](../../src/audrey/kb/watcher.py)
  now branches on `path.suffix.lower()` against `IMAGE_SUFFIXES` and
  skips the wrong-collection call. Cuts watcher-driven Qdrant delete
  load roughly in half on bulk operations. Suffix is reliable here
  because the same allowlist gated the file into the queue at
  `_QueueHandler._enqueue` first. The two pre-existing
  `_delete_vectors` tests were updated and two new ones added in
  [`tests/test_kb_watcher.py`](../../tests/test_kb_watcher.py):
  text-suffix hits text-only, image-suffix hits image-only,
  image-suffix is a no-op when no image embedder is configured,
  text-suffix still works without an image embedder.

- **`consider` — first reconcile sweep waits one full interval.**
  Resolved 2026-05-25.
  [`kb/reconcile.py:KBReconciler._run`](../../src/audrey/kb/reconcile.py)
  now runs one sweep immediately at startup, *then* settles into the
  periodic cadence. Closes the 30-minute stale-state window that used
  to follow a `KB_WATCHER_ENABLED=0` stretch. Pinned by
  `test_reconciler_runs_one_sweep_immediately_at_startup` in
  [`tests/test_kb_reconcile.py`](../../tests/test_kb_reconcile.py)
  using a long `interval_s` and an `asyncio.Event` so the test fails
  fast (timeout) if the startup sweep ever regresses.

- **`consider` — double-failure on upload rollback can leave Qdrant
  inconsistent.** Resolved 2026-05-25.
  [`routes/files.py:upload_file`](../../src/audrey/routes/files.py)
  rollback path now wraps `qdrant.delete_by_file_id` in its own
  `try/except`. A Qdrant outage during rollback is logged at `error`
  level naming the file_id and noting that the next boot's
  `reconcile_with_qdrant` will clean up the orphan, then we re-raise
  the original sqlite-write error. The previous behavior — second
  exception masking the first — is gone. Pinned by two new tests in
  [`tests/test_files.py`](../../tests/test_files.py):
  double-failure-doesn't-mask-original-error and
  successful-rollback-still-re-raises.

- **`nit` — sqlite `PRAGMA journal_mode=WAL` with one connection.**
  Resolved 2026-05-25. The `PRAGMA journal_mode=WAL` line in
  [`kb/uploads_db.py:UploadsDB.__init__`](../../src/audrey/kb/uploads_db.py)
  was removed; one guarded connection doesn't benefit from WAL's
  multi-writer story, and dropping it eliminates the `-wal`/`-shm`
  sidecar files. `PRAGMA synchronous=NORMAL` stayed — rollback
  journal at NORMAL still gives crash safety without per-commit
  fsync.

- **`nit` — original filename stored unbounded.** Resolved
  2026-05-25.
  [`routes/files.py:upload_file`](../../src/audrey/routes/files.py)
  now caps the post-strip filename with `[:255]` (Linux NAME_MAX).
  A 10 MB filename string can't bloat sqlite or the Qdrant payload.
  Pinned by two slice-shape tests in
  [`tests/test_files.py`](../../tests/test_files.py): cap engages
  at 10000 chars, no-op when already short.

- **`nit` — `routes/upload_ui.py` re-reads the HTML on every
  request.** Resolved 2026-05-25.
  [`routes/upload_ui.py:upload_page`](../../src/audrey/routes/upload_ui.py)
  now caches the HTML in a module-level `_HTML_CACHE` on first
  read. The file ships in the container image and never changes at
  runtime, so re-reading per request was wasted I/O. No test —
  trivial cache, low-traffic endpoint.

- **`bug` — `KBWatcher._delete_vectors` used hardcoded collection
  literals instead of QdrantKB-supplied names.** Resolved 2026-05-20.
  [`kb/watcher.py:_delete_vectors`](../../src/audrey/kb/watcher.py)
  now passes `collection=self._qdrant.text_collection` and
  `collection=self._qdrant.image_collection`, matching every other
  KB call site. A deployment that renames collections via
  `kb.text_collection` / `kb.image_collection` in `config.yaml` no
  longer silently skips watcher-driven cleanup. Pinned by
  `test_delete_vectors_uses_qdrant_supplied_collection_names` and
  `test_delete_vectors_skips_image_collection_when_no_image_embedder`
  in [`tests/test_kb_watcher.py`](../../tests/test_kb_watcher.py).

- **`bug` — `delete_file` always returned `deleted: True`.**
  Resolved 2026-05-20. [`routes/files.py:delete_file`](../../src/audrey/routes/files.py)
  now returns `DeleteResponse(file_id=file_id, deleted=deleted_row)`
  so a caller deleting a non-existent file_id sees `deleted: false`
  in the response. The Qdrant + disk unlink steps stay best-effort
  cleanup; the sqlite outcome is the honest signal. Comment on the
  return line explains why.

- **`bug` — quota check ran after the upload was on disk.**
  Resolved 2026-05-20. [`routes/files.py:upload_file`](../../src/audrey/routes/files.py)
  now hoists `already = await db.user_total_bytes(user)` and a
  pre-flight `already >= max_total` check before `_stream_to_disk`,
  short-circuiting at the wire for users already over quota. The
  post-stream `already + written > max_total` check still runs to
  catch the case where this upload itself crosses the line (we only
  know the actual upload size after streaming). Net effect: a user
  already-over-quota wastes zero bytes of disk I/O on every rejected
  upload.

- **`nit` — `_QueueHandler._enqueue` skipped dotfiles but not
  dot-directories.** Resolved 2026-05-20.
  [`kb/watcher.py:_enqueue`](../../src/audrey/kb/watcher.py) now
  skips any path whose components include a dot-prefixed segment,
  matching the rule the offline `_iter_files` crawl applies. Stray
  `.git/HEAD`-class files no longer reach the queue via filesystem
  events. Pinned by `test_enqueue_skips_files_inside_dot_directory`
  in [`tests/test_kb_watcher.py`](../../tests/test_kb_watcher.py).

- **`smell` — three callers reached into `QdrantKB._client`
  directly.** Resolved 2026-05-20. Added two public methods to the
  facade: `QdrantKB.list_collections()` (returns `list[str]`) and
  `QdrantKB.scroll_collection(name, *, page_size=256)` (returns
  `list[tuple[str, dict]]`, i.e. `(point_id, payload)` per point).
  Routed all three callers through them:
  [`kb/reconcile.py:_scroll_sources`](../../src/audrey/kb/reconcile.py),
  [`kb/uploads_db.py:reconcile_with_qdrant`](../../src/audrey/kb/uploads_db.py)
  (uses `list_collections`), and
  [`kb/uploads_db.py:_scroll_user_rows`](../../src/audrey/kb/uploads_db.py)
  (uses `scroll_collection`). The module-level `_list_collection_names`
  helper and the entire `_scroll_user_rows_sync` function were
  deleted — both became unnecessary once the facade owned the
  qdrant-client primitives. The `_FakeQdrantKB` in
  [`tests/test_kb_reconcile.py`](../../tests/test_kb_reconcile.py)
  was simplified to mirror the new facade (no more fake `_client`
  attribute, no more fake `Record` class). Pinned by four new tests
  in [`tests/test_kb_qdrant.py`](../../tests/test_kb_qdrant.py)
  covering the empty-when-missing, multi-page, and missing-payload
  contracts.

#### Lesson 11 (KB ingest and search)

- **`consider` — `chunk_text` emits a tail chunk that's mostly
  duplicate content.** Resolved 2026-05-26 (Phase 12).
  [`kb/chunk.py:chunk_text`](../../src/audrey/kb/chunk.py) now skips
  the final iteration's chunk when `(end - prev_end) <=
  chunk_tokens // 10` and at least one chunk has already been
  emitted. With production defaults (chunk=1000, overlap=100),
  threshold = 100 tokens. Validated by running
  `scripts/measure_chunk_tails.py` against `/datasets`: 13.1 % of
  multi-chunk files produced a wasted tail (225 files; 225 wasted
  chunks; 1.36 % of total chunks). Pinned by 10 tests in
  [`tests/test_kb_chunk.py`](../../tests/test_kb_chunk.py) covering
  keeps/drops/threshold boundary/single-chunk/safety-clamp/idx
  preservation. The original audit's proposed fix (`end - start <=
  overlap_tokens`) was a no-op with defaults — discovered while
  writing the measurement script; corrected fix shape derived from
  the same script's `new_tokens` metric.

- **`nit` — `config.yaml` said `/api/embeddings` for the text embedder.**
  Resolved 2026-05-18. Pure comment fix in
  [`config.yaml:206`](../../config.yaml#L206) — `via Ollama /api/embeddings`
  → `via Ollama /api/embed`. Same nit class as the `qdrant.py` docstring
  fix on 2026-05-12.

- **`consider` — `_normalize` returned the zero vector silently.**
  Resolved 2026-05-18. [`kb/embed.py:163-167`](../../src/audrey/kb/embed.py#L163)
  now emits a `log.warning("kb.embed: zero-norm vector skipped
  normalization; check upstream embedder")` before returning the
  un-normalized vector. Real embedders never emit zero vectors for
  non-empty input, so this is a future-regression canary, not a
  hot-path branch.

- **`optimization` — `_collection_exists_sync` listed every collection
  on each call.** Resolved 2026-05-18.
  [`kb/qdrant.py:_collection_exists_sync`](../../src/audrey/kb/qdrant.py)
  now calls `self._client.collection_exists(name)` directly (O(1) on the
  server, available in qdrant-client ≥1.7; pyproject pins ≥1.12). Same
  swap applied to `_ensure_named_sync` and `_list_user_files_sync`.
  `_ensure_sync` (the bulk global path that checks both collections at
  once) deliberately keeps the listing — one `get_collections()` is
  cheaper than two existence calls there.

- **`nit` — `build_*_point` extras could silently clobber reserved
  payload keys.** Resolved 2026-05-18. Added `_check_extras` in
  [`kb/qdrant.py`](../../src/audrey/kb/qdrant.py) that raises `ValueError`
  if the extras dict intersects the reserved set
  `{"source", "kind", "text", "caption", "chunk_idx", "mtime"}`. Pinned by
  `test_build_text_point_rejects_extras_that_clobber_reserved_keys`,
  `test_build_image_point_rejects_reserved_caption_override`, and a
  happy-path test in [`tests/test_kb_qdrant.py`](../../tests/test_kb_qdrant.py)
  confirming the user-upload extras (`user`, `file_id`, `filename`,
  `mime`, `bytes`, `uploaded_at`) still pass through unchanged.

- **`nit` — `_iter_files` skipped dotfiles but not dot-directories.**
  Resolved 2026-05-18. [`kb/ingest.py:_iter_files`](../../src/audrey/kb/ingest.py)
  now skips any file whose relative path under `root` contains a
  dot-prefixed component. A stray `.git/` or `.cache/` under a topic dir
  no longer has its non-dot children ingested. Pinned by
  [`tests/test_kb_ingest_iter.py`](../../tests/test_kb_ingest_iter.py)
  covering plain files, dotfiles at any depth, dot-directory contents,
  and the single-file-root path.

- **`nit` — `EmptyExtractionError` message hardcoded "scanned PDFs".**
  Resolved 2026-05-18. [`kb/extract.py:extract_text`](../../src/audrey/kb/extract.py)
  now branches on suffix: `.pdf` keeps the scanned-PDF hint, every other
  supported suffix gets a generic "the file is empty or contained no
  extractable text" tail. Pinned by parametrized tests in
  [`tests/test_kb_extract.py`](../../tests/test_kb_extract.py) across
  `.txt`, `.md`, `.html`, `.docx`, `.csv` plus the PDF path and the
  whitespace-only-falsy contract.

- **`smell` — python-magic ImportError silently downgraded to extension
  sniffing.** Resolved 2026-05-18 (fail-closed-at-import). Moved
  `import magic` to module scope in
  [`kb/extract.py`](../../src/audrey/kb/extract.py) with a comment
  explaining the dep contract. A future container build that drops the
  dep now crashes at startup rather than silently degrading to extension-
  only mime sniffing (which would defeat `.png.exe` defense). The
  inner `sniff_mime` runtime `ImportError` fallback was removed; the
  libmagic-runtime-failure fallback (e.g. corrupt sample) is kept as
  `log.warning + _guess_from_suffix`.

- **`nit` — `kb_stats` returned `dict[str, Any]` instead of a typed
  model.** Resolved 2026-05-18. Added `StatsResponse` in
  [`routes/kb.py`](../../src/audrey/routes/kb.py) with a docstring on
  the `collections` field explaining the `-1` "unknown" sentinel. The
  `kb_stats` route now declares `response_model=StatsResponse`, so the
  shape and `-1` semantics surface in the OpenAPI spec.

- **`nit` — `qdrant.py` module docstring named the wrong Ollama endpoint
  (`/api/embeddings`).** Resolved 2026-05-12. Updated
  [`kb/qdrant.py:3-5`](../../src/audrey/kb/qdrant.py#L3) to say
  `/api/embed` so it matches `embed.py` and the live `OllamaClient`.

- **`nit` — `audrey-ingest --purge` scope wasn't documented.**
  Resolved 2026-05-18. Module docstring in
  [`kb/cli.py`](../../src/audrey/kb/cli.py) and the argparse `--purge`
  help text both now spell out that `--purge` hits only the global
  `kb_text` / `kb_images` collections, not per-user
  `kb_user_text_*` / `kb_user_images_*` (which are deleted via the
  `/v1/files` DELETE route).

- **`nit` — `QdrantKB.delete_by_source` docstring promised a count but
  never returned one.** Resolved 2026-05-12. Changed return type to
  `None` in [`kb/qdrant.py:130`](../../src/audrey/kb/qdrant.py#L130);
  dropped the `0`/`-1` sentinels; docstring now states qdrant-client does
  not expose a count. Updated the matching fake in
  [`tests/test_kb_reconcile.py:72`](../../tests/test_kb_reconcile.py#L72).
  All existing callers (ingest, cli, watcher, reconcile) ignored the
  return value already.

- **`nit` — `routes/kb.py` `_search_text_merged` / `_search_images_merged`
  named coroutines `tasks`.** Resolved 2026-05-12. Renamed `tasks` → `coros`
  in both helpers ([`routes/kb.py:115`](../../src/audrey/routes/kb.py#L115)
  and [`routes/kb.py:138`](../../src/audrey/routes/kb.py#L138)).

- **`consider` — same-embedder precondition for cross-collection score
  merge.** Resolved 2026-05-12. Added an inline note in
  [`routes/kb.py:107-118`](../../src/audrey/routes/kb.py#L107) and a
  matching one-liner in `_search_images_merged` documenting that the raw
  score merge assumes both collections use the same embedder and distance
  metric.

- **`nit` — `_IMAGE_FETCH_BYTE_CAP` was enforced post-append.** Resolved
  2026-05-12. [`kb/embed.py:198-209`](../../src/audrey/kb/embed.py#L198)
  now checks `len(buf) + len(chunk) > cap` before extending so a single
  oversized chunk can't overshoot the cap. Pinned by
  `test_fetch_image_rejects_oversized_response_before_appending` in
  [`tests/test_kb_embed_ssrf.py`](../../tests/test_kb_embed_ssrf.py).

- **`smell` — `_ensure_user_indexes_sync` swallowed all exceptions.**
  Resolved 2026-05-12. [`kb/qdrant.py:209-225`](../../src/audrey/kb/qdrant.py#L209)
  now catches only `UnexpectedResponse` and re-raises unless the status
  is a 4xx with "exist" in the body (the idempotent "index already
  exists" path). Non-qdrant exceptions (transport, etc.) propagate
  unchanged. New tests in
  [`tests/test_kb_qdrant.py`](../../tests/test_kb_qdrant.py) cover the
  already-exists swallow, 5xx propagation, unrelated 4xx propagation,
  and a non-qdrant `ConnectionError`.

- **`nit` — redirected image fetches surfaced as opaque HTTP errors.**
  Resolved 2026-05-12. [`kb/embed.py:191-201`](../../src/audrey/kb/embed.py#L191)
  detects 3xx before `raise_for_status()` and raises a `ValueError`
  naming the redirect target so the user can resupply the final URL.
  Pinned by `test_fetch_image_reports_redirect_clearly` in
  [`tests/test_kb_embed_ssrf.py`](../../tests/test_kb_embed_ssrf.py).

- **`accepted` — `ASYNC240` sync `Path.stat()`/`Path.exists()` in
  `kb/ingest.py`.** Re-confirmed 2026-05-12. Ingest runs from the watcher
  or the admin `/v1/kb/ingest` route, never on the chat hot path. The
  chat-path KB call only does embedding + Qdrant search; neither touches
  `path.stat()`. Original accepted reasoning still holds.

#### Lesson 9 (tool use and the ReAct loop)

- **`smell` — `discover_one` skips endpoints whose `operation_id == "health"`
  as a special case.**
  Resolved 2026-05-12. Dropped the `op_id == "health"` branch in
  [`tools/discovery.py`](../../src/audrey/tools/discovery.py); tag-based
  filtering (`"tools" not in tags`) already handles `/health` because
  the tools-server tags it `system`. Added
  [`tests/test_discovery.py`](../../tests/test_discovery.py) covering
  the tag filter as the canonical rule, including a regression test
  that synthesizes a POST `/health` operation and confirms it's
  filtered by tag, not the special case.

- **`nit` — `dispatch_one`'s JSON-string `args` handling is fragile under
  refactor.**
  Resolved 2026-05-12. Captured `raw_args = fn.get("arguments")` before
  any rebinding in
  [`tools/dispatch.py`](../../src/audrey/tools/dispatch.py); the
  `JSONDecodeError` branch now slices `raw_args` so a future edit that
  swaps the order can't silently break the error payload. Pinned by
  `test_dispatch_one_json_string_error_echoes_raw_args` in
  [`tests/test_dispatch.py`](../../tests/test_dispatch.py).

- **`nit` — `_compress_history` keep-window is hardcoded `keep_last_round=1`
  at both call sites.**
  Resolved 2026-05-12. Added `agentic.react.compress_keep_last` to
  `config.yaml` (default 1) plus `deep_worker.compress_keep_last`
  (falls back to outer). Threaded through `run_react` and both call
  paths (`fast_path.py`, `deep_panel.py`, `graph.py`,
  `routes/openai.py` `_phase_dispatch`). Pure helper behavior pinned
  by `_compress_history` tests in
  [`tests/test_react.py`](../../tests/test_react.py) across keep values
  0/1/2 and the at-or-below-threshold no-op case.

- **`nit` — `_USER_SCOPED_TOOLS` is hardcoded in `dispatch.py` with no link
  to the tools-server side that defines new tools.**
  Resolved 2026-05-12 (comment-only fix). Added an explicit
  "ADDING A NEW USER-SCOPED TOOL — two places to edit" note above
  `_USER_SCOPED_TOOLS` in
  [`tools/dispatch.py`](../../src/audrey/tools/dispatch.py) so a
  future edit on the tools-server side surfaces the dispatcher
  dependency. The startup-warning option from the original audit
  finding is intentionally not implemented — false-positive risk on
  tools that accept `user` as optional/informational data.

#### Lesson 7 (classification + routing)

- **`bug` — streaming classification does not pass `tool_names`, so explicit
  tool mentions can lose the tool-routing override.**
  Resolved 2026-05-12. Extracted `classify_with_registry(...)` in
  [`pipeline/classify.py`](../../src/audrey/pipeline/classify.py) and
  routed both [`pipeline/graph.py`](../../src/audrey/pipeline/graph.py)
  and [`routes/openai.py`](../../src/audrey/routes/openai.py) through
  it. The helper extracts `tool_names` from the live registry once;
  both paths now behave identically. Added regression tests in
  [`tests/test_classify.py`](../../tests/test_classify.py) covering the
  tool-mention short-circuit, the no-registry fallback, and the
  router-reach path.

- **`nit` — `classify.py` module docstring overstates what weak keyword
  signals do.**
  Resolved 2026-05-12. Updated
  [`pipeline/classify.py`](../../src/audrey/pipeline/classify.py)
  module docstring to say weak signals are held in reserve as a
  router-failure fallback rather than "bump priority for stage 2."
  Behavior unchanged; the docstring now matches the implementation.

- **`nit` — `PipelineState.virtual_model` comment omits `audrey_auto` and
  `audrey_fast`.**
  Resolved 2026-05-12. Updated
  [`pipeline/state.py`](../../src/audrey/pipeline/state.py) comment
  to list all five virtual models.

- **`nit` — memory recall comment still mentions SQL LIKE after the memory
  backend moved to semantic search.**
  Resolved 2026-05-12. Updated
  [`pipeline/memory.py`](../../src/audrey/pipeline/memory.py) comment
  to describe the embedding-dilution rationale for `MAX_QUERY_CHARS`
  rather than the obsolete SQL LIKE one.

#### Lesson 6 (model layer)

- **`nit` — `OllamaError` docstring is narrower than its actual use.**
  Resolved 2026-05-07. Updated
  [`models/ollama.py`](../../src/audrey/models/ollama.py) to describe
  HTTP, transport, and response parsing failures.

- **`nit` — `OllamaClient` class docstring says no default timeout
  even though the constructor sets one.**
  Resolved 2026-05-07. Updated the docstring to describe the startup
  default timeout plus per-call overrides.

- **`should-fix` — model `location` is typed as `Literal` but not
  validated when loaded from YAML.**
  Resolved 2026-05-07. Added `_parse_location(...)` in
  [`models/registry.py`](../../src/audrey/models/registry.py), which
  validates `local` / `cloud` during registry construction and raises
  `ValueError` on unknown values.

- **`should-fix` — successful HTTP responses with malformed JSON bypass
  the model-layer error contract.**
  Resolved 2026-05-07. Added `_json_object(...)` in
  [`models/ollama.py`](../../src/audrey/models/ollama.py) and routed
  `tags()`, `chat()`, and `embed()` through it so invalid JSON or wrong
  top-level shapes become `OllamaError`. `chat()` now records an error
  metric instead of success if response parsing fails.

- **`consider` — core model-layer behavior has no direct tests yet.**
  Resolved 2026-05-07. Added
  [`tests/test_models.py`](../../tests/test_models.py) covering registry
  sorting/copy behavior, healthy fallback selection, location validation,
  health cooldown/reset/backoff, and OllamaClient transport, HTTP status,
  malformed JSON, unexpected shape, and embedding vector-count failures.

#### Lesson 5 (configuration + startup)

- **`nit` — lazy `ToolRegistry` import inside the lifespan.** Resolved
  2026-05-06. Hoisted `ToolRegistry` into the top-of-file
  `from audrey.tools.discovery import ToolRegistry, discover_all`
  ([`main.py:39`](../../src/audrey/main.py#L39)) and dropped the
  function-local import that was at `main.py:71`.

- **`bug` — env defaults silently overwrite YAML lists for
  `dataset_paths` and `tools.servers`.** Resolved 2026-05-06 (Phase
  32). Changed `tool_servers` and `kb_dataset_paths` on `EnvOverrides`
  to `str | None` with `default=None`, and gated their merge in
  `_apply_env_overrides` on `is not None` like the other tunables
  ([`config.py:37-43`](../../src/audrey/config.py#L37),
  [`config.py:86-93`](../../src/audrey/config.py#L86)). Also dropped
  the `${TOOL_SERVERS:-…}` and `${KB_DATASET_PATHS:-…}` fallbacks from
  `compose.yaml` ([`compose.yaml:46-49`](../../compose.yaml#L46)) so
  the YAML's lists are now load-bearing. Override behavior:
  env-var set → env wins; env-var unset → YAML wins.

- **`smell` — `KB_WATCHER_ENABLED` bypasses `EnvOverrides` entirely.**
  Resolved 2026-05-06 (Phase 32). Added
  `kb_watcher_enabled: bool = Field(default=False,
  alias="KB_WATCHER_ENABLED")` to `EnvOverrides`
  ([`config.py:62`](../../src/audrey/config.py#L62)) and changed
  `main.py` to `if cfg.env.kb_watcher_enabled:` instead of reading
  `os.environ` directly. Pydantic Settings handles the 1/true/yes
  parsing natively. `import os` dropped from `main.py` (no other
  usages remained).

- **`smell` — `EnvOverrides.data_dir` is read by nobody.** Resolved
  2026-05-06 (Phase 32). Field deleted. The `/data/...` paths the
  field would have governed are still hardcoded at their call sites
  (e.g. `uploads_db_path` in `kb` config); if we ever want a single
  knob for "audrey's data root" we can reintroduce it then with
  actual readers.

- **`nit` — `get_env()` constructs a new EnvOverrides each call.**
  Resolved 2026-05-06 (Phase 32). Function deleted, removed from
  `__all__`. No callers in the repo.

- **`consider` — `Config._yaml` mutated by `_apply_env_overrides`.**
  Resolved 2026-05-06 (Phase 32). Renamed `Config._yaml` →
  `Config._merged` to make the post-override state obvious from the
  attribute name. No external readers (private attr); pure rename.

### Accepted

#### Lesson 16 (custom-tools sidecar, `tools-server/`)

- **`consider` — `web_search` count ceilings disagree across three layers
  (schema le=10 / cache clamp 20 / `_fetch` unbounded).** Accepted
  2026-06-03 (teach, don't change). Not a live bug — the schema cap of 10
  wins for every tool call, the only path that exists today; the
  "future caller bypassing the schema" is hypothetical with no current
  signal. The lesson covers it as a teaching point about where validation
  actually binds (the Pydantic schema is the real ceiling; the cache/fetch
  clamps are belt-and-suspenders). Revisit if Audrey's internal search path
  ever passes a user-controlled count.

- **`nit` — `stats()` comment references a `/metrics` gauge.** Accepted
  2026-06-03. Metrics is out of lesson scope (dropped 2026-05-28); the
  comment is harmless and the `chunks_unindexed` value it describes is real
  (used by the admin `/chat_archive/stats` route). Left as-is.

#### Lesson 17 (admin routes + auth tail, `routes/admin.py`)

- **`nit` — `chat_archive_prune`/`_stats` duplicate ~6 lines of
  archive-client resolution + `AsyncClient` setup.** Accepted 2026-06-03.
  Two call sites is the judgment threshold for extracting a helper, not a
  requirement; the duplication is small and the two handlers differ
  (GET/POST, timeout, return handling) enough that a shared helper would
  carry parameters for each difference. Left as-is; the lesson presents
  them as two parallel handlers.

- **`consider` (scope) — `/v1/tools/rediscover` lives in `main.py`, not
  `routes/admin.py`.** Resolved as a scope decision 2026-06-03 (no code
  change): Lesson 17 covers the *full* admin surface — everything behind
  `require_admin` — spanning `routes/admin.py` and the rediscover route in
  `main.py`, and explains why rediscover lives in `main.py` (it mutates the
  live `ToolRegistry` via an `app.state` closure). The admin.py docstring
  now cross-references it.

#### Lesson 16 (streaming banners, `banners.py`)

- **`nit` — `worker_ok`/`worker_fail` hardcode a two-space lead while
  `tool_summary_block` builds its own markdown structure; the two
  tail-fragment producers don't share a formatting convention.** Accepted
  2026-06-03. They render in different contexts (inline tail appended to a
  banner line vs. a standalone footer block), so the divergence is by
  design, not accidental — there's no real convention to share. The lesson
  will present them as independent pure formatters rather than implying a
  common one.

#### Lesson 12 (KB lifecycle — watcher, reconcile, uploads)

- **`consider` — debounce window restarts on every event.** Accepted
  2026-05-25.
  [`kb/watcher.py:_run`](../../src/audrey/kb/watcher.py)
  overwrites `pending[(kind, path)] = time.monotonic()` on every
  re-event, so a file changing more often than `debounce_s` never
  flushes. Accepted: this is the intended "wait until things settle"
  behavior. The KB roots (`/mnt/user/knowledge/`) are curated content,
  not editor workspaces — nothing in the real workload looks like
  continuous autosave. The accept reasoning was added as an
  inline comment in `_run` so a future reader understands the
  choice is deliberate.

- **`consider` — `reconcile_with_qdrant` not safe under concurrent
  uploads.** Accepted 2026-05-25 (documented, not fixed).
  [`kb/uploads_db.py:reconcile_with_qdrant`](../../src/audrey/kb/uploads_db.py)
  has a race window between `db.all_users()` and the per-user qdrant
  reads where a concurrent upload could be incorrectly pruned. The
  race is structurally impossible today — `main.py:lifespan` calls
  reconcile before uvicorn accepts traffic — so the fix is to
  document the precondition rather than add locking. The reconcile
  function's docstring now spells out the contract explicitly, and
  `main.py`'s call site has a comment marking the ordering as
  structural. If a future code path calls reconcile concurrent with
  the upload route, the docstring tells the next reader to add
  per-user locking first.

#### Lesson 8 (deep mode — planner, panel, synth, reflect)

- **`consider` — `_BREVITY_CUES` in `reflect.py` is English-only.**
  Accepted 2026-05-24. [`pipeline/reflect.py:35-52`](../../src/audrey/pipeline/reflect.py#L35)
  matches the user's short-answer cues case-insensitively against
  a hardcoded list of English phrases ("in one sentence", "tldr",
  "briefly", etc.). A non-English user asking for a brief answer
  in their own language would trigger a wasteful retry. Accepted:
  Audrey's scope is English-only by current product decision; no
  multilingual support is planned. Surface in lesson narrative,
  not a code fix.

- **`consider` — `_run_one_worker`'s ReAct branch passes the worker's
  per-round accounting unchanged.** Accepted (surfaced in Lesson 8)
  2026-05-24. Verified the underlying behavior in
  [`pipeline/react.py:run_react`](../../src/audrey/pipeline/react.py):
  `ReactResult.prompt_eval_count` / `eval_count` carry the values from
  only the **final** chat call, not the sum across rounds. The
  deep-panel `WorkerDraft` therefore undercounts tokens on tool-grounded
  workers — a tool-using worker that ran 3 rounds reports only the last
  round's token usage. Not a correctness issue (metrics consumers read
  per-worker totals, not pipeline totals) but worth a lesson note so
  future operators reading logs understand the gap. No code change.

- **`nit` — `_parse_planner_output` accepts garbage around the JSON.**
  Accepted (surfaced in Lesson 8) 2026-05-24.
  [`pipeline/planner.py:_parse_planner_output`](../../src/audrey/pipeline/planner.py)
  uses outermost `find('{')` + `rfind('}')` slicing, which is
  intentionally permissive against noisy model output. The audit
  confirmed this: any malformed slice falls through to `json.loads`
  raising `JSONDecodeError`, which the helper catches and degrades to
  `[]` — the same path as every other planner failure mode. Permissive
  input handling is gated by safe-degradation, so there's no
  correctness issue. Lesson narrative explains the trade-off; no code
  change.

- **`consider` — `_format_drafts_for_synth` truncates nothing.** Accepted
  2026-05-24 (deferred to measurement followup).
  [`pipeline/synthesize.py:_format_drafts_for_synth`](../../src/audrey/pipeline/synthesize.py)
  concatenates every draft into the synth user message with no per-draft
  or total cap, so a 4-worker × 8 KB worst-case can hand ~32 KB of draft
  material to the synthesizer. A defensive `max_synth_draft_chars` knob
  was considered but rejected without data — picking a cap blind would
  either throttle good answers (too low) or fail to address the worst
  case (too high). Tracked in `PROJECT_STATE.md` under the followup
  queue alongside the existing chunk-tail measurement.

#### Lesson 11 (KB ingest and search)

- **`smell` — `ingest_text_file` and `ingest_user_text_file` duplicate
  ~90% of their bodies.** Accepted 2026-05-18.
  [`kb/ingest.py:ingest_text_file`](../../src/audrey/kb/ingest.py) and
  [`kb/ingest.py:ingest_user_text_file`](../../src/audrey/kb/ingest.py)
  share the load/extract → chunk → embed → delete → upsert shape with
  three diffs (loader, delete-by-source vs delete-by-file_id, extras
  payload). Both functions are short and individually readable; a shared
  helper threading those three as parameters would be harder to follow
  than the duplication. If a third near-clone is ever added the call
  for extraction gets stronger; until then, leave as-is.

#### Lesson 10 (how function calling works)

- **Lesson 10 audit-pass skip.** Accepted 2026-05-13. Lesson 10 is the
  conceptual function-calling lesson; its file surface (`tools/discovery.py`,
  `tools/dispatch.py`, `pipeline/react.py`, and the `chat(..., tools=...)`
  call sites in `pipeline/graph.py` / `pipeline/deep_panel.py`) was already
  audited fresh-eyes for Lesson 9 in the same campaign. A second pass with
  a different lens is unlikely to surface new findings, and the lesson is
  protocol-focused — it does not introduce new code paths. Skipping the
  audit pass for this lesson is the deliberate call. Findings can still
  land under "Lesson 10" if something surfaces while drafting.

#### Lesson 5 (configuration + startup)

- **`consider` — config-load fast-fails while qdrant boot degrades.**
  Accepted 2026-05-06 (Phase 32). The asymmetry is deliberate: a
  malformed/missing config has no sensible default, so `_load_yaml`
  letting `FileNotFoundError`/`ValueError` bubble out of lifespan
  (and crash the process) is the right behavior. Qdrant's boot path
  degrades because Audrey can still serve chat/tools without the KB.
  Annotated with a one-line comment at
  [`config.py:121-124`](../../src/audrey/config.py#L121) so future
  readers see the choice was intentional.

---

## Already-known issues to revisit during their lessons

These were noted during the build campaign and live in HISTORY.md or
the followup memory notes. The lesson covering each file should
revisit and either confirm "still fine" or upgrade to a real audit
finding here.

- **`ASYNC240` warnings in `kb/ingest.py` + `kb/reconcile.py`.**
  Synchronous `Path.exists()` / `Path.stat()` inside async
  functions. Accepted as a known tradeoff in HISTORY.md (local SSD,
  not on the request hot path). Lesson 11 (KB ingest) and
  Lesson 12 (KB lifecycle) should re-evaluate: is the accepted
  reasoning still right, or has the KB grown to a size where this
  matters? (Both surfaced and accepted again during the Lesson 11/12
  audit passes.)
- **Tools-server `pyproject.toml` vs Dockerfile dual maintenance.**
  Tools-server has a flat-script layout that hatchling can't
  wheel-build; the Dockerfile keeps a manual deps list. Adding a tool
  dep means editing both. The eventual tools-server-internals lesson
  should call this out and ask whether the conversion is worth the
  effort. (Phase 5 deploy partially addressed this; revisit needed.)
