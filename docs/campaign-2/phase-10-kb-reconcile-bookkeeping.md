# Campaign 2 Phase 10 — KB global reconcile: bookkeeping + loose-end fixes

A short bookkeeping phase to formally close the long-standing
"Phase 29 KB reconcile pass" followup that's been on the cross-campaign
queue since Phase 19. The mechanism it described **already exists** —
it shipped piecemeal during the Lesson 11/12 audit passes and during
the Lesson 12 audit drain on 2026-05-25, without a consolidating
deploy doc. Phase 10 closes the loop: it writes the deploy doc that
should have accompanied that work, fixes two small loose ends, and
moves the followup off the queue.

## What was already in place before Phase 10

The Phase 29 followup as originally written asked for:

> Periodic background task (default 30 min) that scrolls every point
> in `kb_text` + `kb_images`, checks `payload.source` against
> `Path.exists()`, deletes orphans via `delete_by_source`. … New
> `src/audrey/kb/reconcile.py`, lifecycle wiring in `main.py`, config
> `kb.reconcile.{enabled, interval_s}`. Optional admin endpoint
> `POST /v1/admin/kb/reconcile`. Excludes per-user collections.

A pre-Phase-10 inventory of the codebase shows that **all of this
shipped**:

| Phase 29 spec | Where it lives now | Notes |
| --- | --- | --- |
| Reconcile module | [`src/audrey/kb/reconcile.py`](../../src/audrey/kb/reconcile.py) | `reconcile_collection`, `reconcile_once`, `KBReconciler` |
| Lifecycle wiring | [`src/audrey/main.py:139-148`](../../src/audrey/main.py#L139) | `KBReconciler(...).start()` if `kb.reconcile.enabled` |
| Config knobs | [`config.yaml:230-232`](../../config.yaml#L230) | `enabled: true`, `interval_s: 1800` |
| Admin endpoint | [`src/audrey/routes/admin.py:153`](../../src/audrey/routes/admin.py#L153) | `POST /v1/admin/kb/reconcile`, returns the structured summary |
| Per-user collection exclusion | `reconcile_once` only sweeps the two global collections | Explicit in the docstring |
| Startup sweep | [`src/audrey/kb/reconcile.py:_run`](../../src/audrey/kb/reconcile.py) | Added during the Lesson 12 audit drain on 2026-05-25 |
| Test coverage | [`tests/test_kb_reconcile.py`](../../tests/test_kb_reconcile.py) | 10 tests covering sweep behavior, lifecycle, startup-sweep |

So the question Phase 10 actually has to answer is: **what's left?**

## What's left — three small things

### 1. `config.yaml` reconcile block misattributed to "Phase 30"

The comment above the `kb.reconcile:` block at
[`config.yaml:226`](../../config.yaml#L226) used to say
`# Phase 30: periodic reconcile pass for the global kb_text + kb_images
collections.` That misattribution had two issues: Phase 30 actually
covers image-digest pinning + README cleanup (different work), and the
broader codebase has been moving away from Phase-N labels in published
material since the lessons started.

Replaced with a substance-first comment that explains what the block
does, *when* it runs (immediate startup sweep + periodic), and where
the admin trigger lives. No Phase references — anyone reading config
in a year doesn't need to know which build phase added what.

### 2. No regression test for the admin endpoint

The reconcile *function* was thoroughly tested (10 tests in
`test_kb_reconcile.py`); the *route* that exposes it to operators was
not. If a refactor broke the wiring — say, renaming
`app.state.qdrant` or changing `reconcile_once`'s signature without
updating the handler — only a manual `curl` against the live deploy
would catch it.

Added [`tests/test_admin_routes.py`](../../tests/test_admin_routes.py)
with 3 tests pinning the route-level contract: the handler calls
`reconcile_once` with the right qdrant, returns `result.to_dict()` to
the client, and logs the trigger naming the admin's email (the only
per-call audit trail for who ran the sweep). Pattern follows
`test_auth.py` — fake `Request` + monkeypatched `reconcile_once`
rather than spinning up a full FastAPI `TestClient`.

This is the first test file under `tests/test_admin_routes.py`; future
admin endpoints (auth eviction, archive stats, etc.) have a place to
land their own wiring tests now.

### 3. The cross-campaign followup memory still lists Phase 29 as open

`project_phase19_followups.md` had Phase 29 in its "Still open" section
because nothing flipped it when the implementation work landed mid-
lesson. Moved to "Shipped during Campaign 2" with the Phase 10
verification date; remaining items renumbered.

## Closure verification — *pending*

This phase ships entirely in the repo — no Unraid-side compose changes,
no service restart needed. Verification is:

- `pytest tests/test_kb_reconcile.py tests/test_admin_routes.py -q`
  passes (covered by the full-suite run below).
- Full test suite stays green: `pytest tests/ -q`.
- Ruff clean on touched files.
- A quick smoke check that the admin endpoint is alive on the live
  deploy — see §2 below.

## 1. Deploy (repo-only)

No container rebuild. `git pull` on Unraid is sufficient. Audrey's
running process doesn't load `config.yaml` comments, and the test/doc
changes don't affect runtime. The actual reconcile mechanism has been
running on the deploy since whenever its underlying changes landed.

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
```

If you've been tracking `kb_reconcile=on` in the boot logs already,
nothing changes after the pull — Phase 10 is bookkeeping, not behavior.

## 2. Smoke tests

### 2.1 Reconcile loop is actually running

```bash
docker logs audrey-ai 2>&1 | grep -E 'kb_reconcile|kb.reconcile'
```

Expect a startup line `kb.reconcile: periodic sweep every 1800s` and
periodic `kb.reconcile: pass complete; orphans_deleted=N elapsed=...`
lines (one per `interval_s`, plus one at startup from the immediate
sweep added during the Lesson 12 audit drain).

If you've recently `git pull`ed: the immediate-startup sweep means
you should see one `pass complete` line in the most recent boot's
logs, not just after 30 minutes of waiting.

### 2.2 Admin endpoint responds

```bash
TOKEN="<your OWUI bearer>"
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  http://192.168.1.11:8000/v1/admin/kb/reconcile | jq
```

Expect a JSON response shaped like:

```json
{
  "by_collection": {
    "kb_text":   { "checked": ..., "orphans_deleted": ..., "points_in_orphans": ..., "elapsed_s": ..., "error": "" },
    "kb_images": { "checked": ..., "orphans_deleted": ..., "points_in_orphans": ..., "elapsed_s": ..., "error": "" }
  },
  "total_orphans_deleted": ...,
  "total_elapsed_s": ...
}
```

Confirms the endpoint is wired, auth works, and the response shape
matches what the test pins.

### 2.3 No orphans on a clean KB

On a steady-state KB where the watcher has been running, the admin
endpoint should return `total_orphans_deleted: 0`. A non-zero result is
the signal that the watcher missed something — most likely a delete
during a `KB_WATCHER_ENABLED=0` stretch, a container restart, or
an outside-of-audrey `rm`. Worth checking what got cleaned up if the
number is non-trivial.

## 3. Rollback

The repo-side changes are: the `config.yaml` comment, the new
`tests/test_admin_routes.py` file, this deploy doc, and the
PROJECT_STATE / followup-memory updates. None of them affect runtime.
Rollback is `git revert <commit>` if any of them turn out to be
mis-shapen; nothing on the Unraid side needs touching.

## 4. Operational notes

- **The reconcile is conservative.** It only deletes points whose
  `payload.source` is missing from `Path.exists()`. Per-user uploads
  have no on-disk source (they live in Qdrant only), so even an
  accidental run against a per-user collection would no-op — but we
  scope it to the two global collections anyway as belt-and-suspenders.
- **The startup sweep can be slow on a large KB.** First-pass timing
  on a ~16k-point KB is on the order of a few seconds; nothing alarming,
  but worth knowing. If you ever see boot logs paused at "starting kb
  reconcile" for tens of seconds, the KB has grown past the pagination
  size and is doing more round-trips than intended.
- **Ad-hoc sweeps don't reset the periodic timer.** If you fire the
  admin endpoint at minute 5, the next periodic sweep still runs at
  minute 30, not minute 35.

## 5. Followups

- **Phase 30 — image digest pinning + README refresh.** Still on the
  cross-campaign queue. Independent of Phase 10.
- **A test for the reconcile + watcher race.** Today the reconciler
  and the watcher can both target the same `delete_by_source` call
  for the same orphan path. Qdrant treats the second as a no-op, but
  there's no test pinning that contract. Low priority — accepted
  behavior, just unverified.
