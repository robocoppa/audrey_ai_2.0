# Lesson Audit Findings

_Internal working doc. **Gitignored** — does not ship to GitHub._

This file collects code-review findings that surface during lesson
writing. It is the queue Bart drains on his schedule. It lives
*outside* the lesson docs themselves: published lessons stay focused
on teaching, while audit material — which is opinion-laden, code-
specific, and quickly stale — stays here.

Sister to `docs/PROJECT_STATE.md` (current priority + lesson catalog/tracker)
and `docs/campaign-1/HISTORY.md` (the build-campaign history).

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
#### CODEBASE findings — validated against current source 2026-06-25

All re-verified valid against current source on 2026-06-25. The 3 quick wins were
fixed this session (now under Resolved); the 4 below stay open — each carries a
**validated still-live 2026-06-25** stamp so a future session needn't re-verify
from scratch. (The cite checker only flags lesson cites, not these source-only
findings, so they're hand-verified.)

- **`should-fix` - CODEBASE: chat archive exposes unindexed chunks but has no repair path -> `open` (validated still-live 2026-06-25).** `tools-server/chat_archive.py:618` reports `chunks_unindexed`, and comments at `:31`/`:374` say "a future reconcile can pick it up" — but no such reindex/reconcile worker exists for rows left with `indexed_at IS NULL` after an embed/upsert failure. Suggested drain: add an operator repair endpoint/job, or document the manual repair path where operators actually look.

- **`should-fix` - CODEBASE: streaming planning/panel tasks lack explicit cancel-and-await cleanup -> `open` (validated still-live 2026-06-25).** The deep streaming route creates `think_task` (`routes/openai/pipeline.py:554`) and `panel_task` (`:590`); only `synth_task` (`:634`, finally at `:742`) has a dedicated cancel-and-await `finally`. `think_task`/`panel_task` are only awaited via `_drain_q_until_task` / `.result()`, so a client disconnect *during* planning or panel dispatch can leave background work running longer than intended. Suggested drain: add explicit cancel-and-await around those phase tasks + a cancellation test.

- **`consider` - CODEBASE: planner local Ollama call bypasses the fair gate -> `open` (validated still-live 2026-06-25).** `planner.py:60` calls `ollama.chat(...)` with no `gate.acquire()` wrapper anywhere in the function. If the planner model is local, that call contends with gated local fast/deep/synth/ReAct calls ungoverned. Suggested drain: decide whether planner calls should accept/use `FairLocalGate`, or pin planner config to cloud/small local models intentionally.

#### Surfaced during run-7 research-eval work — 2026-07-08 (not lesson-driven)

Two pre-existing problems hit while fixing S4 (fact-check parser). Neither was
introduced this session — both confirmed failing on clean `main` (HEAD
`e954975`). Logged here so they aren't lost; not yet drained.

- **`should-fix` - CODEBASE: `test_research_stream_no_trace_block_by_default` fails on clean `main` — TWO layers -> `open` (2026-07-08, updated).** [`tests/test_research_stream.py`](../../tests/test_research_stream.py). **Layer 1 (signature):** the line-35 `_FakeOllama.chat()` double raises `TypeError: got an unexpected keyword argument 'format'`; production now passes `format=` (the Ollama structured-output schema) but that double's signature was never updated (the line-210 `_StructuringFakeOllama` one already has `format=`). **Layer 2 (behind it):** adding `format=None` to the line-35 double does NOT make the test pass — it then fails at line 254 `assert "## Research trace (debug)" not in joined`. The test asserts the debug trace block ships dark (flag off by default), but with the structuring call no longer erroring, the trace renders even with `debug_research_trace` unset. So there is a real gating bug (or a stale test) behind the signature: the opt-in trace is not actually gated off in this fake-app path. Confirmed pre-existing by stashing all S4 edits and re-running: identical `TypeError` at HEAD `e954975`. Suggested drain: fix the line-35 signature AND then investigate why the trace block renders flag-off — either the `debug_research_trace` gate regressed or the test's `_fake_app` default now trips the render path; do NOT just patch the assertion.

- **`should-fix` - LESSON 17 (+ some Part-1 lessons): stale `file:line` cites (drift/drift?) into `prompts.py` / `deep_panel.py` / `config.yaml` -> `open` (2026-07-08, worsened).** `scripts/check-lesson-links.py` now reports ~13 `DRIFT` + ~102 `DRIFT?` (was 5/99), 0 broken. Most point at `prompts.py` / `deep_panel.py` lines shifted by the research-trace plumbing commits AND by the S4 chunked-structuring insertion this session (~55 lines added around `deep_panel.py:818–940`, which pushed lesson-17's `deep_panel.py` anchors down). A few non-lesson-17 ones (e.g. `lesson-07:429 → config.yaml#L431`) trace to the earlier `config.yaml` budget/flag edits, not to any src change this session. The S4 code changes themselves are drift-only (no broken cites), and the `ledger.py` cites did not move. Suggested drain: re-anchor lesson-17's (and the affected Part-1 lessons') `file:line` cites against current source in one pass, per the standing "re-anchor by hand after source edits" rule.

#### Lesson 15 — synth event-loop spin-poll (optimization, deferred-by-trigger)

- **`optimization` — the synth event-loop polls with 50ms timeouts instead of `wait` with FIRST_COMPLETED -> `open` (validated still-live 2026-06-25).** [`routes/openai/pipeline.py:649`](../../src/audrey/routes/openai/pipeline.py#L649) loops on `await asyncio.wait_for(events_q.get(), timeout=0.05)` while draining the banner queue; same pattern in [`_drain_q_until_task` at routes/openai/pipeline.py:794](../../src/audrey/routes/openai/pipeline.py#L794). Functionally correct (50ms is small enough that banners don't visibly lag), but it's a spin-poll at ~20 wakeups/sec per active deep stream; the cleaner shape is `asyncio.wait({task_get, banner_q_get}, return_when=FIRST_COMPLETED)`. Cost is near-zero, so this is principle-driven, not performance-driven. **Deferral trigger:** revisit when Grafana's asyncio task-wakeup count climbs under streaming load correlated with concurrent deep streams — the rewrite has real complexity risk (interleaving two queues with explicit drain semantics), so pay it only when measurement justifies it.

#### Lesson (research mode / `audrey_research`) — audit 2026-06-30

Audit of the in-scope files for the planned research-mode lesson:
`run_research_pipeline_streaming` + helpers in `pipeline/deep_panel.py`,
`pipeline/ledger.py`, the research role prompts in `prompts.py`, the research pool
in `config.yaml`. Much of this code was written/edited this same session, so these
are deliberately fresh-eyes-on-my-own-work — weighted toward readability traps a
lesson reader would hit, not just bugs.

**DRAINED 2026-06-30:** #1 and #2 fixed (comment-only — unified deep_panel.py to one
4-stage scheme [research / verify / fact-check / write], rewrote the stale ledger
comment to describe current behavior, dropped a bare phase-number ref); 628 pytests
pass, ruff clean. #3/#4/#5 accepted as lesson teaching-points (timeout-is-per-stage
sidebar; why claims aren't deduped but sources are; the message-build inconsistency
noted in passing). Lesson write is now gated only on outline approval.

- **`resolved` — TWO conflicting "Stage N" numbering schemes in deep_panel.py.** Fixed 2026-06-30.
  The module-level pipeline overview comment numbers it research / **Stage 2=VERIFY**
  / **Stage 3=WRITE** ([`deep_panel.py:483-485`](../../src/audrey/pipeline/deep_panel.py#L483)),
  a 3-stage view. But the inline section comments in `run_research_pipeline_streaming`
  use the 4-stage deploy-doc view — **Stage 2=verify** ([`:1224`](../../src/audrey/pipeline/deep_panel.py#L1224)),
  **Stage 3=fact-check** ([`:1241`](../../src/audrey/pipeline/deep_panel.py#L1241)) —
  AND a comment *inside* the fact-check block says "**Stage 2:** structure the
  fact-check" ([`:1293`](../../src/audrey/pipeline/deep_panel.py#L1293)). So "Stage 2"
  means three different things across one function (verify / write / fact-check-
  structuring), and "Stage 1b" ([`:1186`](../../src/audrey/pipeline/deep_panel.py#L1186))
  is a fourth scheme. This is a real comment-that-lies trap — the lesson must NOT
  inherit phase/stage numbers (course rule), but the code itself is internally
  contradictory. Suggested drain: pick ONE scheme (the deploy-doc 4-stage one is the
  user-facing truth) and make every comment in the function match it; renumber/strip
  the stray "Stage 1b" and the in-block "Stage 2."

- **`resolved` — stale "no effect on the ANSWER yet" comment lies now.** Fixed 2026-06-30.
  [`deep_panel.py:1190-1192`](../../src/audrey/pipeline/deep_panel.py#L1190) still
  says the ledger-build stage's "only measurable effect on the ANSWER is none yet —
  it just produces the ledger; we eval that the prose path is undisturbed before
  wiring the ledger into verify/write/hedge." That was true when Stage 1 shipped dark,
  but the ledger now drives fact-check verdicts, the Sources list, AND the hedge
  dispositions — it has a large effect on the answer. The comment actively
  misdescribes current behavior. Suggested drain: rewrite to "the ledger is built
  here and consumed by the fact-check, Sources, and hedging stages below."

- **`consider` — `timeout_s` is reused unscaled across all four stages -> `open`.**
  Every stage call (`_run_one_worker`, `_structure_one_draft`, verify, fact-check
  ReAct, `_structure_factcheck`, the writer stream) is passed the same `timeout_s`.
  A research request can therefore run research(≤timeout) + structure(≤timeout) +
  verify(≤timeout) + factcheck-ReAct(≤timeout × rounds) + write(≤timeout) serially —
  the wall-clock blowups seen this session (one case 1436s) are partly this:
  no single stage exceeds its timeout, but the *sum* is unbounded. Not a bug (each
  stage is correctly bounded), but worth a lesson sidebar and possibly a per-stage
  budget. Suggested drain: discuss whether the pipeline wants an overall deadline,
  or document that `timeout_s` is per-stage by design.

- **`nit` — `_merge_ledgers` concatenates claims with no dedup or cap -> `open`.**
  [`deep_panel.py` `_merge_ledgers`](../../src/audrey/pipeline/deep_panel.py) keeps
  every claim from every worker (sources are URL-deduped; claims are not). On a dense
  topic 3 workers × ~15 claims = ~45 near-duplicate claims downstream — this was the
  root of the Stage-4 disposition verbosity already addressed at the render layer
  (only action-bearing lines rendered). The merge itself is intentional (conflicting
  claims surface for the fact-checker) so this is a `nit`/teaching-point, not a fix:
  worth explaining in the lesson WHY claims aren't deduped (the fact-checker wants to
  see disagreement) while sources are.

- **`nit` — `_with_role_system` vs inline `{"role":"system",...}` inconsistency -> `open`.**
  Stage 1 builds researcher messages via `_with_role_system(...)`
  ([`:1148`](../../src/audrey/pipeline/deep_panel.py#L1148)), but the verify, fact-check,
  and write stages build their system message inline as a literal dict. Two ways of
  doing the same thing in one function. Minor, but the lesson walks all four stages
  back-to-back so the reader will see both forms. Suggested drain: either route all
  four through the helper or note why Stage 1 differs (it prepends to the full
  message history; the others build a fresh 2-message list).

#### Pointers to completed work (kept brief; detail under Resolved / Accepted)

- **3 quick-win CODEBASE fixes shipped 2026-06-25** (admin reconcile collection
  names; reflect `no_drafts` → end-without-retry; Brave non-429 normalization) —
  see Resolved.
- The 2026-06-24 `+13` prose pass, `+14` cite-fix pass, the 2026-06-02 full-corpus
  audit, and the three 2026-06-03 pre-write drains are all complete — detail under
  Resolved / Accepted. (Their `✅ DRAINED` banners and per-lesson drift tables were
  pruned from "Open" 2026-06-24/25 once the narrative was captured below.)

### Deferred

*(none)*

### Resolved

#### CODEBASE quick-win fixes 2026-06-25

- **`should-fix` - admin KB reconcile passes configured collection names -> `resolved` 2026-06-25.** `routes/admin.py` `kb_reconcile` now calls `reconcile_once(qdrant, text_collection=qdrant.text_collection, image_collection=qdrant.image_collection)` instead of letting it fall back to the `"kb_text"`/`"kb_images"` defaults — so an ad-hoc admin sweep after a config collection-rename hits the *configured* collections (matching the periodic loop in `main.py`). `QdrantKB` already exposed those names as attrs. `tests/test_admin_routes.py` extended: the existing test now asserts the kwargs are forwarded with a non-default stub (`renamed_text`/`renamed_images`); the two sibling reconcile tests got a shared `_fake_qdrant()` stub. 508 pytests pass, ruff clean.

- **`should-fix` - reflect ends on deterministic `no_drafts` instead of retrying -> `resolved` 2026-06-25.** `route_after_reflect` (`graph.py`) now returns `"end"` when `reflect_reason == "no_drafts"`, before the retry-budget check. Previously a `no_drafts` failure (synthesis found zero usable drafts — deterministic) routed to `"retry"` and re-ran the whole deep panel against the same dead workers, producing `no_drafts` again and wasting a panel pass. `node_reflect` already writes `reflect_reason` to state, so the guard reads it directly. +3 `route_after_reflect` tests in `tests/test_deep_panel.py` (no_drafts→end; too_short-under-budget→retry still works; passed→end), reaching the closure via the compiled graph's branch registry. 508 pytests pass.

- **`should-fix` - Brave non-429 HTTP failures normalized -> `resolved` 2026-06-25.** Under `reraise=True`, an exhausted non-429 `httpx.HTTPStatusError` escaped `brave.py`'s `search()` raw, and the `web_search` handler (which caught only `BraveRateLimitError`/`ValueError`) let it fall through to FastAPI as a generic 500. Added `BraveUpstreamError`; the retry block now normalizes a surviving `HTTPStatusError` to it, and the handler maps it to a controlled 503 (matching the 429 path). The dead `except RetryError` (unreachable under `reraise=True`) was removed along with its now-unused import. New `tests/test_brave.py` (MockTransport + stubbed `asyncio.sleep`): persistent 500 → `BraveUpstreamError`; persistent 429 → `BraveRateLimitError`. 508 pytests pass, `tools-server/` ruff clean.

#### Manual line-by-line accuracy pass 2026-06-24 (+14) — both courses

- **Cite-drift + prose fixes across both courses -> `resolved` 2026-06-24 (+14).**
  Fresh full read of every published lesson against current source. `graph.py`
  and `routes/openai/{routes,pipeline,schemas}.py` had shifted a few lines since
  several lessons were last anchored, leaving cites off-by-1 (landing on a blank
  line or docstring) or off-by-several — the kind of drift the mechanical checker
  can't catch (it only checks the cited line *looks* like a landmark). All fixed
  in place, docs-only, no source changes. AI L4 (graph.py node-line list +
  `pipeline_total`/`_stream_openai` lines), L6 (main.py/config.yaml/graph.py),
  L7 (classify.py 201/249 + 8 graph.py cites), L8 (node_planner 293/306), L13
  (graph.py 139, routes.py 109, pipeline.py 853), L15 (~14 cites across
  routes/pipeline/schemas/passthrough/graph; `_delta_frame`/`_stop_frame` cite
  moved from responses.py to pipeline.py:501 where they actually live), Python L4
  (graph.py 475→484). Two prose bugs fixed: L10 §2.2 truncated sentence
  ("...bias, not a" → completed) + stale `claude-opus-4-7`→`claude-opus-4-8` in
  the illustrative Anthropic JSON; L16 §2.5 duplicated "authenticated." word.
  L2 "~180-line client" → "small client" (drop stale count, satisfies the
  no-specific-counts convention). **Post-fix verification:** AI course cite
  checker 332 checked, 253 ok, **0 confident DRIFT, 0 broken**, 79 advisory
  `DRIFT?`; Python course 8/8 ok, 0/0/0; convention checker clean on both.

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
  2026-05-12. [`kb/embed.py:198-209`](../../src/audrey/kb/embed.py#L209)
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
  Resolved 2026-05-12. [`kb/embed.py:191-201`](../../src/audrey/kb/embed.py#L202)
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
  [`config.py:121-124`](../../src/audrey/config.py#L137) so future
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
