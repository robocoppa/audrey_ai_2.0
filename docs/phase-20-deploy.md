# Phase 20 — fair per-user scheduling

**Goal:** when two users send queries at the same time, each gets their
*first* request served before either gets their *second*. No more
"user A queues up 10 deep requests and user B waits behind all of
them." Loose rate limiting, no errors thrown — extra requests wait
until a slot frees.

Two layers, both keyed by the OWUI user id Audrey already plumbs
through:

- **Layer 1 — `FairLocalGate`.** Replaces the global-FIFO `GpuGate`
  at the local-GPU acquire point (deep panel workers + synth on a
  local model). Same `concurrency=1` invariant for VRAM/PSU budget,
  but instead of strict FIFO, an `OrderedDict[user, deque[Future]]`
  round-robins across users on each release. Cloud calls bypass.
  Anonymous/no-user-id requests share a synthetic `__anon__` bucket.
- **Layer 2 — per-user in-flight cap.** Wraps both
  `_generate_via_pipeline` and `_stream_via_pipeline` so the cap
  holds for the full pipeline (classify → fast/deep → synth). Default
  3 concurrent requests per user. Soft cap: extra requests *wait* via
  `await sem.acquire()`, no 429.

What changed:

- **`src/audrey/pipeline/fair_gate.py` (new)** — `FairLocalGate`.
  Exact same `acquire(model, *, location, user_id=None)` shape as
  the old `GpuGate` plus the `user_id` kwarg. Cancellation-safe: a
  parked future that gets cancelled (client disconnect) is skipped
  on the next release rather than granting the slot to a dead
  waiter.
- **`src/audrey/routes/inflight.py` (new)** — `UserInflightRegistry`.
  Lazy LRU dict of `asyncio.Semaphore(N)` per user. `slot(user_id)`
  context manager. `_safe_bucket()` truncates email-shaped ids in
  log lines.
- **`src/audrey/pipeline/semaphore.py` (deleted)** — `GpuGate` is
  gone. Don't add a back-import.
- **`src/audrey/pipeline/{deep_panel,graph,synthesize}.py`** — type
  hints changed from `GpuGate` to `FairLocalGate`; every
  `gate.acquire` call now passes `user_id=...` (threaded from
  `PipelineState.user_id` for the graph path, from `payload.user`
  for the streaming path).
- **`src/audrey/routes/openai.py`** — both pipeline entry points
  wrapped in `inflight.slot(payload.user)`. Streaming synth call
  passes `user_id` so synth's gate acquire is fair too.
- **`src/audrey/main.py`** — instantiate `FairLocalGate` and
  `UserInflightRegistry` in the lifespan, attach to `app.state`.
  Boot log line gained `max_inflight_per_user=N`.
- **`src/audrey/metrics.py`** — new
  `audrey_user_inflight_blocked_seconds` histogram. Bucketed
  0/0.05/0.5/2/10/30/120s.
- **`config.yaml`** — new `fairness` section with
  `max_inflight_per_user` (default 3) and `max_tracked_users`
  (default 1024). `MAX_INFLIGHT_PER_USER` env override.

What stays the same:

- `GPU_CONCURRENCY=1` invariant. Same PSU budget, same single local
  worker at a time.
- Cloud path — fully parallel, bounded only by Ollama Pro's 3-cloud
  cap server-side.
- Per-user data isolation (memory, uploads) — already correct,
  untouched.
- All existing metrics and log shapes. New metric is additive; no
  label changes.
- Single uvicorn process, single event loop. No new dependencies.

Out of scope (deliberately):

- **Tunable per-user weights.** Every user gets equal round-robin
  rotation. No "admin gets 2 slots."
- **RPM rate limit.** The in-flight cap is what the user asked
  for — loose, no errors. Cost-protection RPM is a separate axis.
- **Persistent scheduler state.** Container restart drops queues;
  in-flight requests die anyway.
- **Fast-path gate plumbing.** `pipeline/fast_path.py` still calls
  `ollama.chat` directly without going through the gate — was true
  pre-Phase 20 and remains true. The in-flight cap covers the
  fairness story for fast-path; a future tightening would wrap the
  fast-path call in `gate.acquire` too.

**Prereqs:** Phase 19 verified. No data migrations.

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 5 audrey-ai | grep ready
```

Expected ready line gains a new field:

```
ready: ollama=...; task types=[...]; gpu_concurrency=1; max_inflight_per_user=3; tools=...
```

If `max_inflight_per_user=3` doesn't appear, the rebuild didn't pick
up `main.py` — try `--no-cache`.

---

## 2. Smoke tests

All from the **laptop** with `$ADMIN_TOKEN` exported. Each test is
self-contained.

### 2.1 Single-user, single query — no regression

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"one sentence on rsync"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq -r '.choices[0].message.content'
```

Expected: an answer. If anything 500s, Phase 20 broke a happy path —
roll back and inspect.

### 2.2 Two users, one query each — fairness sanity

Run in two terminals concurrently:

```bash
# Terminal A
curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":true,"user":"alice","messages":[{"role":"user","content":"one sentence on rsync"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions

# Terminal B (fire ~1s later)
curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":true,"user":"bob","messages":[{"role":"user","content":"one sentence on tar"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions
```

Both should start banners immediately. Local gate serializes the
*per-worker* dispatch but each user holds the gate for one worker at
a time, alternating. No starvation.

Audrey log should show interleaved `deep_panel:` / worker activity
between the two requests, not all of `alice`'s workers before any of
`bob`'s.

### 2.3 One user, 10 concurrent — in-flight cap holds

```bash
for i in $(seq 1 10); do
  (curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_fast\",\"stream\":false,\"user\":\"spammer\",\"messages\":[{\"role\":\"user\",\"content\":\"req $i\"}]}" \
    https://chat.builtryte.xyz/v1/chat/completions > /tmp/req_$i.json &)
done
wait
ls -la /tmp/req_*.json | wc -l   # expect 10
```

Audrey log should show a few "inflight: user=spamm… waited Xs for
slot" lines for the 4th–10th requests (only when wait > 1s).

Check the metric:

```bash
curl -s http://<unraid-ip>:8000/metrics | grep audrey_user_inflight_blocked_seconds
```

Expected: samples in non-zero buckets, especially the `2.0` and
`10.0` buckets.

None of the 10 requests should error.

### 2.4 Fairness vs. spam — the headline test

Fire 5 concurrent requests as `alice`, then 1 from `bob` ~500ms
later. `bob`'s request should *not* sit behind all 5 of `alice`'s
local-GPU acquires:

```bash
# alice fires 5
for i in 1 2 3 4 5; do
  (curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_local\",\"stream\":true,\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"alice $i\"}]}" \
    https://chat.builtryte.xyz/v1/chat/completions > /tmp/alice_$i.log &)
done

# bob fires 1, slightly later
sleep 0.5
curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":true,"user":"bob","messages":[{"role":"user","content":"bob 1"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions > /tmp/bob_1.log

wait
```

Note that alice will have 3 in-flight (cap=3) and 2 queued at layer
2. Bob's single request will be in-flight immediately because he has
his own semaphore. At the local gate (layer 1), alice's 3 in-flight
requests will rotate with bob's 1 — bob's local-GPU acquire should
land before alice's 2nd local-GPU acquire (round-robin).

Tail the log; you should see `bob` in deep_panel activity *between*
alice's workers, not after all of them:

```bash
docker compose logs --since 2m audrey-ai | grep -E 'deep_panel:|inflight:'
```

### 2.5 Mid-wait disconnect — scheduler stays clean

Start a long deep request, kill curl mid-wait, then verify a new
request from a different user still works:

```bash
# Start 5 alice requests
for i in 1 2 3; do
  (curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_local\",\"stream\":true,\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"alice $i\"}]}" \
    https://chat.builtryte.xyz/v1/chat/completions > /dev/null &)
done

# Cancel one mid-wait (Ctrl-C the parent shell, or kill by pid)
# ...wait a few seconds, then:

# bob should still get served fairly
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"user":"bob","messages":[{"role":"user","content":"hello"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq -r '.choices[0].message.content'
```

bob should answer normally. No "stuck waiter" symptom — if bob never
returns, `_release()` in `fair_gate.py` has a bug; the cancellation
path's `while dq and dq[0].done(): dq.popleft()` is what should
clean alice's dead waiters.

### 2.6 Anonymous bucket — programmatic clients share one queue

Three concurrent curl calls *without* a `user` field should all
funnel into `__anon__`:

```bash
for i in 1 2 3 4 5; do
  (curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_fast\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"anon $i\"}]}" \
    https://chat.builtryte.xyz/v1/chat/completions > /tmp/anon_$i.json &)
done
wait
```

Requests 4 and 5 should wait at the in-flight cap (default 3) just
like real users. The anonymous bucket isn't bypassed.

If you want to exercise smoke tests *without* contention, give each
script its own user id (`"user": "smoke-1"`, `"smoke-2"`, …).

---

## 3. Tuning

If the cap is too tight (multiple OWUI tabs from one user
deadlocking on themselves), bump in `compose.yaml`:

```yaml
environment:
  MAX_INFLIGHT_PER_USER: 5
```

Or in `config.yaml`:

```yaml
fairness:
  max_inflight_per_user: 5
```

Restart audrey-ai. Cap is per-process, not persistent — no
migration needed.

If tracking memory grows large (many distinct users over a long
uptime), bump `max_tracked_users`. Default 1024 should be fine for
home/small-team scale.

---

## 4. Rollback

Phase 20 touches several files but is one logical change:

```bash
git checkout <previous-sha> -- \
  src/audrey/pipeline/fair_gate.py \
  src/audrey/pipeline/semaphore.py \
  src/audrey/pipeline/synthesize.py \
  src/audrey/pipeline/deep_panel.py \
  src/audrey/pipeline/graph.py \
  src/audrey/routes/inflight.py \
  src/audrey/routes/openai.py \
  src/audrey/main.py \
  src/audrey/metrics.py \
  src/audrey/config.py \
  config.yaml
docker compose up -d --build audrey-ai
```

(`semaphore.py` will be re-created by the checkout; `fair_gate.py`
and `inflight.py` will need `git rm` if you want them gone, but
leaving the new files unreferenced is harmless.)

---

## 5. Follow-ups (not Phase 20)

- **Wrap fast-path calls in the gate too.** `pipeline/fast_path.py`
  still calls `ollama.chat` directly without `gate.acquire`. This
  was the same under `GpuGate`, but a single user's 3 concurrent
  fast-path local-model requests bypass the layer-1 fairness
  entirely. The in-flight cap protects against spam, but a real
  fix would be `async with gate.acquire(spec.name, location=...,
  user_id=user_id)` around the chat call.
- **Per-user-bucket gauge of current queue depth.** Useful for
  diagnosing pile-ups but expensive on cardinality. Could add a
  scalar Gauge `audrey_local_gate_queue_total` (sum across all
  users) and `audrey_user_inflight_active` (sum of `_inuse`
  values).
- **Operator command to reset queue state.** Today the only way to
  wipe state is restarting audrey-ai. A `POST /v1/admin/fairness/clear`
  could drain dead waiters and reset all in-flight counters.
  Probably overkill until/unless we hit a stuck-state bug in prod.
