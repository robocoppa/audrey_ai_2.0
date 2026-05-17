# Campaign 2 Phase 8 — Cleanup deploy

Two unrelated cleanup items bundled into one phase because each is too
small to warrant its own deploy doc:

1. **`chat_history_search` limit hardening.** Schema cap stays
   sensible; tool description is sharpened so the model stops
   requesting `limit=20` and getting 422'd.
2. **Phase 6 smoke-test remainder.** Tests 2.4-2.7 from the Phase 6
   deploy doc were never run on Unraid. Phase 6 has been live for
   weeks without surfacing regressions, so this is verification
   debt to clear, not active risk.

Both ship in one session.

---

## Part A — `chat_history_search` limit hardening

### What we know

[`tools-server/app.py:401-406`](../../tools-server/app.py#L401)
defines the request schema:

```python
class ChatHistorySearchRequest(BaseModel):
    user: ...
    query: ...
    limit: Annotated[int, Field(ge=1, le=10)] = 5
    ...
```

`limit` is capped at 10. The default is 5. The description field is
empty — the model only sees the JSON-schema constraint at tool
discovery time, not a human-readable explanation of the bound.

Captured failure 2026-05-15 22:20:45 (during Phase 6a investigation):

```text
dispatch: chat_history_search -> 422 in 0.00s:
{"detail":[{"type":"less_than_equal","loc":["body","limit"],
"msg":"Input should be less than or equal to 10","input":20,"ctx":{"le":10}}]}
```

Model called the tool with `limit=20`. Audrey recovered on the next
call (issued with no `limit` argument, which defaulted to 5), but
the wasted dispatch is real, and any pipeline that retried more
aggressively could have hit a longer failure chain.

### What changes

Two changes, both in
[`tools-server/app.py`](../../tools-server/app.py):

1. **Raise the cap to 20.** The original cap of 10 was conservative.
   With the existing 2000-char tool-result truncation in audrey's
   ReAct loop ([`config.yaml`](../../config.yaml) → `agentic.react.max_tool_chars`),
   even 20 hits at typical 100-char snippets fit comfortably. 20
   matches the cap on `kb_search` / `kb_image_search` / `memory_*`
   (all `le=20` already), removing a per-tool asymmetry that has no
   user-visible justification.

2. **Sharpen the description.** Tell the model in plain English what
   `limit` does and what its bounds are. The new description:

   ```python
   limit: Annotated[int, Field(
       ge=1, le=20, default=5,
       description=(
           "Max results to return. Must be 1-20. "
           "Default 5 is right for most lookups; "
           "use a larger limit only when you need broad recall."
       ),
   )] = 5
   ```

This is the standard belt-and-suspenders pattern: tighter
specification at the schema, plus a description the model sees in
its tool registry so it can self-correct before making the call.

### Tests

`tests/test_chat_archive.py` already covers `search()` end-to-end.
Add two new tests targeting the schema directly so the regression
risk is pinned:

- `limit=20` validates (accepts the new upper bound).
- `limit=21` rejects (Pydantic 422).

These tests live alongside the existing chat-archive tests; no new
test file needed.

### Why not just keep `le=10`?

Three reasons not to leave it at 10:

- **Inconsistent with sibling tools.** `kb_search` and friends all
  cap at 20; the model's mental model of "how big can `limit` be"
  is contaminated by the broader tool family.
- **The model already wants more than 10.** The 22:20:45 dispatch
  shows the model reaching for `limit=20` on its own. If the
  request is *reasonable* (a "broad recall" query, a search for
  scattered context), making it succeed at 20 is better than
  forcing a retry at 10.
- **The truncation knob downstream handles the size.** Whether the
  archive returns 5 or 20 hits, the ReAct loop truncates the result
  at 2000 chars before the model sees it. The blast radius is
  already bounded.

### Out of scope

- Changing `kb_search` / `kb_image_search` / `memory_*` caps. They
  already match; no work needed.
- Tightening descriptions on other tools. This is a targeted fix
  for one observed failure, not a sweep.
- Reconsidering the 2000-char ReAct truncation. Separate concern;
  unrelated to schema validation.

---

## Part B — Phase 6 smoke-test remainder

Run all four (2.4-2.7) in one session. The original Phase 6 deploy
doc carries the canonical instructions; this section is a runbook
ordering and a place to record results.

### Prereqs

- Audrey, custom-tools, Qdrant, Ollama all healthy on Unraid.
- One test user with at least one uploaded file (Test 2.5 needs
  a per-user collection).
- One additional small file ready to upload during Test 2.6 (any
  small text file works; Test 2.6 specifically wants an upload
  that *creates* a new per-user collection or re-uses an existing
  one idempotently).
- The `httpbin.org` host reachable from inside the audrey-ai
  container (Test 2.4 happy-path).

### Run order

Order matters: 2.4 → 2.5 → 2.6 → 2.7. Each later test benefits from
the earlier one passing.

#### Test 2.4 — KB image search via URL

Per [phase-6-deploy.md §2.4](phase-6-deploy.md). Two arms:

**Happy path (the one not yet verified):**

```bash
curl -sS -X POST http://localhost:8000/v1/kb/query/image \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://httpbin.org/image/png", "top_k": 3}' | jq .
```

Expected: top hits returned (`results` array non-empty), HTTP 200.
Pin: the byte-cap re-ordering from Phase 6 didn't break the happy
path.

**Redirect arm (already verified during Phase 6a session):**
Re-run anyway for completeness:

```bash
curl -sS -X POST http://localhost:8000/v1/kb/query/image \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://picsum.photos/200", "top_k": 3}' | jq .
```

Expected: 422 with `detail` matching `image embed failed: image_url
returned redirect (302) to '<final URL>'; supply the final URL
directly`.

#### Test 2.5 — Per-user collection merge

Per [phase-6-deploy.md §2.5](phase-6-deploy.md).

In OWUI (logged in as the test user who has uploads), send a query
that should hit content known to be in their uploads. Then:

```bash
docker compose logs --since 30s audrey-ai | grep "kb_search"
```

Expected:
- `had_user_collection=true` appears in the `kb_search_seconds`
  metric label.
- At least one returned hit's `source` references the user-uploaded
  file (visible in the model's response, not just the log).

If the user collection isn't being queried, check the Phase 6 fix to
`routes/kb.py` (the `coros` rename + score-merge precondition note)
landed cleanly.

#### Test 2.6 — Upload flow exercises `_ensure_user_indexes_sync`

Per [phase-6-deploy.md §2.6](phase-6-deploy.md).

Upload one small text file via OWUI's uploads UI (the test user
from 2.5 works fine — the re-upload path is the more important arm
since it exercises the idempotent "index already exists" branch).
Then:

```bash
docker compose logs --since 1m audrey-ai | grep -iE "qdrant|payload index"
```

Expected:
- No warning about index creation failing.
- The narrowed `UnexpectedResponse` handler doesn't get tripped
  (would log a real error if it did).

If you see an `UnexpectedResponse` propagating, **stop and
investigate Qdrant before rolling forward** — that's a real
Qdrant-side problem the prior broad handler was hiding.

#### Test 2.7 — Phase 4 Category 4 sweep

Per [phase-4-testing.md](phase-4-testing.md) Category 4 plus
[phase-6-deploy.md §2.7](phase-6-deploy.md).

In OWUI, send a prompt that triggers `kb_search` end-to-end, e.g.
"look up BTRFS in the KB" (or any topic you know is in the curated
global KB). Confirm:
- The model dispatches `kb_search` (visible in `> _Tools used:_`
  footer).
- Synthesizes an answer from the hits.
- No 4xx/5xx errors in the audrey-ai log during the dispatch.

### Recording results

When you run the tests, update Phase 6's deploy doc's "Followups"
section *or* append a brief verification block to it noting:

- Date verified.
- Which arms passed.
- Anything unexpected (note in this doc too).

Phase 6's verification status in `PROJECT_STATE.md` should then
flip from "partially verified" to "verified".

### If a test fails

Each test corresponds to a specific Phase 6 code change. The
mapping:

| Test | Code change exercised |
|---|---|
| 2.4 happy | [`kb/embed.py`](../../src/audrey/kb/embed.py) pre-append byte-cap check |
| 2.4 redirect | [`kb/embed.py`](../../src/audrey/kb/embed.py) clearer redirect error |
| 2.5 | [`routes/kb.py`](../../src/audrey/routes/kb.py) `coros` rename + score merge |
| 2.6 | [`kb/qdrant.py`](../../src/audrey/kb/qdrant.py) `_ensure_user_indexes_sync` narrowed handler |
| 2.7 | end-to-end smoke; touches all of the above plus dispatch |

Failures are diagnostic — fix the specific module, re-run only that
test. No need to redo earlier passing tests.

---

## Deploy steps

Order: Part A code change first (small, ships through git), then
verify Part A in a quick dispatch test, then run Part B.

### 1. Ship Part A

On the laptop:

```bash
# Edit tools-server/app.py — see Part A "What changes" section.
# Edit tests/test_chat_archive.py — see Part A "Tests" section.
.venv/bin/pytest tests/ -q
.venv/bin/ruff check tools-server/app.py tests/test_chat_archive.py
```

Both must pass. Then commit and push.

### 2. Deploy on Unraid

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build custom-tools
```

`audrey-ai` does not need a rebuild — the schema change is in
custom-tools only. Audrey's tool registry rediscovers from
`/openapi.json` at startup or on demand, but a custom-tools rebuild
gives it a clean reset either way.

### 3. Verify Part A

Force a `chat_history_search` dispatch with `limit=20` (this is the
exact shape that failed pre-fix). In OWUI:

```text
Search my chat history for everything related to <topic>, with at
least 15 results.
```

The model should now request `limit=20` and the dispatch should
return 200 OK. Then:

```bash
docker compose logs --since 1m audrey-ai | grep "chat_history_search"
```

Expected: a `dispatch: chat_history_search ok in X.XXs (NNN chars)`
line, no `422` entries.

### 4. Run Part B smoke tests

Per the runbook above. Allow ~30-60 minutes for all four if
nothing surprises.

### 5. Update PROJECT_STATE.md

When Part B passes, in `docs/PROJECT_STATE.md`:

- Remove the "Phase 6 verification remainder" item from open
  followups.
- Remove the `chat_history_search` schema-mismatch item.
- Bump the current state to "Phase 6 fully verified".

---

## Rollback

**Part A** is plain git revert. No state, no schema migration:

```bash
git revert <part-a-commit>
docker compose up -d --build custom-tools
```

The model will go back to occasionally hitting the 422 wall and
self-correcting; not great but not broken.

**Part B** has no rollback because it's read-only smoke testing.
If a test fails, fix the specific module per the mapping table
above; revert only if the underlying Phase 6 fix turns out wrong
(unlikely after weeks live).

---

## Followups

- **None expected.** Both items are explicit cleanup; they don't
  open new threads.
- If Part B surfaces something unexpected, log it as a new
  followup in `PROJECT_STATE.md` rather than expanding this doc.

---

## Out of scope for Phase 8

- Lesson 10 (KB ingest and search). Sits in the queue separately;
  unrelated to this cleanup.
- Specialist virtual model prototype. Planned at
  [specialist-prototype-plan.md](specialist-prototype-plan.md);
  unrelated.
- Grafana per-tool dispatch panel. Separate followup.
- Chunk-tail measurement script. Separate followup.

## Operational notes

- Part A's verification (step 3 above) involves issuing a real
  chat with the test user. Don't run it in the user's normal
  conversation — start a fresh chat so the bullet-prefix retry
  noise (if any reappears) doesn't pollute archive history.
- Part B's tests are mostly read-only. Test 2.6 writes to the
  per-user Qdrant collection; that's the intended exercise.
- No diagnostic config knobs need flipping. Phase 6a's
  instrumentation is already off as of 2026-05-17.
