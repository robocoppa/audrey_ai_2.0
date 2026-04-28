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

Run from the **laptop** with `$ADMIN_TOKEN` exported. Replace
`192.168.1.11` with your Unraid LAN IP if different.

### Two important gotchas (learned during phase-20 verification)

1. **Test against `http://<unraid-ip>:8000` directly, not via
   Cloudflare.** Cloudflare's burst connection-rate limit drops 1–2 of
   any 5+ concurrent TLS handshakes from the same source IP, even with
   a small stagger. Failed requests come back as `curl: (56) Failure
   when receiving data from the peer` and never reach audrey, so
   they're invisible in the audrey logs and they corrupt the test.
   The `/v1/chat/completions` endpoint accepts the OWUI Bearer token
   directly on port 8000 — same auth, no tunnel in the path.
2. **Use plain `&` not `(... &)` subshells.** Subshells detach the
   curl from your shell, so `wait` doesn't see them and
   `kill <pid>` won't reach them. Plain `&` keeps each curl as a job
   of the current shell, with a real `$!` PID.
3. **Always capture stderr separately** (`2>/tmp/foo_$i.err`).
   Without separation, parallel curl error messages tangle byte-for-byte
   into unreadable garbage.
4. **Stagger spawns by 200–300ms** even for LAN-direct, just to keep
   port allocation and connection setup tidy. The 5 spawns spread over
   1s is still well within the "concurrent for fairness purposes"
   window.

The tests below incorporate all four.

### 2.1 Single-user, single query — no regression

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"one sentence on rsync"}]}' \
  http://192.168.1.11:8000/v1/chat/completions | jq -r '.choices[0].message.content'
```

Expected: an answer. If anything 500s, Phase 20 broke a happy path —
roll back and inspect.

### 2.2 Two users, one query each — fairness sanity

```bash
rm -f /tmp/alice_*.log /tmp/bob_1.log /tmp/alice_*.err /tmp/bob_1.err

# alice
curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":true,"user":"alice","messages":[{"role":"user","content":"one sentence on rsync"}]}' \
  http://192.168.1.11:8000/v1/chat/completions > /tmp/alice_1.log 2>/tmp/alice_1.err &

sleep 1

# bob
curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":true,"user":"bob","messages":[{"role":"user","content":"one sentence on tar"}]}' \
  http://192.168.1.11:8000/v1/chat/completions > /tmp/bob_1.log 2>/tmp/bob_1.err &

wait
```

What "pass" looks like:

- Both `chat.completions (stream)` log lines appear ~1s apart.
- Both `stream deep done` log lines have similar `elapsed=` values
  (within ~5s of each other). Verified phase-20 example: alice
  elapsed=138.43s, bob elapsed=138.68s — they shadowed each other
  through the whole pipeline.
- The gate metric shows ~6 acquires (3 workers × 2 users) waiting in
  total ~250s — that's expected with `concurrency=1`.

What "fail" looks like (FIFO winning): bob's `elapsed` would be
roughly `2 × alice's elapsed` because bob waited behind all of
alice's panel before getting any gate slot.

Verification commands:

```bash
# from Unraid host
docker logs --since 4m audrey-ai 2>&1 | grep -E 'chat\.completions|stream deep done|inflight'
curl -s http://192.168.1.11:8000/metrics | grep -E 'audrey_gpu_gate_wait_seconds_(count|sum)'
```

### 2.3 One user, 10 concurrent — in-flight cap holds

```bash
rm -f /tmp/req_*.json /tmp/req_*.err
for i in $(seq 1 10); do
  curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_fast\",\"stream\":false,\"user\":\"spammer\",\"messages\":[{\"role\":\"user\",\"content\":\"req $i\"}]}" \
    http://192.168.1.11:8000/v1/chat/completions > /tmp/req_$i.json 2>/tmp/req_$i.err &
  sleep 0.2
done
wait
echo "log sizes:"
wc -c /tmp/req_*.json
echo "errors (should all be empty):"
wc -c /tmp/req_*.err
```

Audrey log (run from Unraid):

```bash
docker logs --since 5m audrey-ai 2>&1 | grep -i 'inflight:'
curl -s http://192.168.1.11:8000/metrics | grep audrey_user_inflight_blocked_seconds
```

What "pass" looks like:

- All 10 `.json` files non-empty, all 10 `.err` files empty.
- A few `inflight: user=spamm… waited Xs for slot` lines for the
  4th–10th requests (only fires when wait > 1s).
- `audrey_user_inflight_blocked_seconds` shows samples in non-zero
  buckets, especially the `2.0` / `10.0` / `30.0` buckets.

None of the 10 should error or 429.

### 2.4 Fairness vs. spam — the headline test

Fire 5 concurrent requests as `alice`, then 1 from `bob` ~500ms
later. `bob`'s should finish in roughly *single-request time + a
modest fairness tax*, not `5 × alice_elapsed`.

```bash
rm -f /tmp/alice_*.log /tmp/bob_1.log /tmp/alice_*.err /tmp/bob_1.err

for i in 1 2 3 4 5; do
  curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_local\",\"stream\":true,\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"alice $i\"}]}" \
    http://192.168.1.11:8000/v1/chat/completions > /tmp/alice_$i.log 2>/tmp/alice_$i.err &
  sleep 0.3
done

sleep 0.5
time curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":true,"user":"bob","messages":[{"role":"user","content":"bob 1"}]}' \
  http://192.168.1.11:8000/v1/chat/completions > /tmp/bob_1.log 2>/tmp/bob_1.err

wait
echo "alice log sizes:"
wc -c /tmp/alice_*.log
echo "errors:"
cat /tmp/alice_*.err /tmp/bob_1.err 2>/dev/null
```

Note that alice has cap=3, so requests 4 and 5 sit at layer 2 until
slots free. The 3 in-flight alice requests share the local gate with
bob's 1 — round-robin.

What "pass" looks like:

- bob's `time real` is ~`1.2× to 2× single-request time`. Verified
  phase-20 LAN-direct run: solo `audrey_local` ~140–180s, 5-vs-1
  contended bob landed at **3m13.36s = 193.36s**. That's ~15–55s of
  fairness tax. Notably bob's panel finished *first* — ~20s before
  alice's first completed — because bob has 1 panel × 3 workers
  while each alice panel is competing with 2 other alice panels for
  gate time, so bob's worker round-robin lands sooner.
- All 5 alice `.log` files non-empty, all 6 `.err` files empty.
- Verified inflight waits for alice's 4th and 5th: 276.95s and
  350.68s respectively. Layer-2 cap of 3 strictly enforced.

What "fail" looks like (FIFO winning): bob's `time real` would be
~`5 × alice_elapsed`, often 10+ minutes.

You can also see the layer-2 cap in action — at least one of
alice_4 or alice_5 should log `inflight: user=alice waited Xs for
slot` with X in the tens of seconds:

```bash
docker logs --since 10m audrey-ai 2>&1 | grep 'inflight:'
```

### 2.5 Mid-wait disconnect — scheduler stays clean

Fire alice requests, kill one mid-flight, verify bob still gets
served and no audrey traceback.

```bash
rm -f /tmp/alice_*.log /tmp/alice_*.err

# Capture pids so we can kill one
declare -a alice_pids
for i in 1 2 3; do
  curl -sS --no-buffer -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_local\",\"stream\":true,\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"alice $i\"}]}" \
    http://192.168.1.11:8000/v1/chat/completions > /tmp/alice_$i.log 2>/tmp/alice_$i.err &
  alice_pids+=($!)
  sleep 0.3
done

echo "alice pids: ${alice_pids[@]}"
sleep 2  # let them get into the pipeline
kill "${alice_pids[1]}"
echo "killed pid ${alice_pids[1]} (alice_2)"

sleep 2

# bob — should answer normally despite alice's queued chaos
echo "--- bob's request ---"
time curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"user":"bob","messages":[{"role":"user","content":"hello"}]}' \
  http://192.168.1.11:8000/v1/chat/completions
echo ""

wait
echo "alice log sizes (alice_2 should be small/empty):"
wc -c /tmp/alice_*.log
```

What "pass" looks like:

- bob returns a clean JSON response with content (not a 500, not a
  hang).
- bob's `time real` is reasonable for `audrey_fast` (~5–60s
  depending on model load).
- alice_1 and alice_3 finish normally — they shouldn't be stuck
  behind alice_2's dead waiter at the gate.
- Audrey logs show no `Traceback` referencing `fair_gate.py` or
  `inflight.py`.

Verification:

```bash
docker logs --since 5m audrey-ai 2>&1 | grep -iE 'traceback|stream deep done|inflight'
```

You should see `inflight: user=alice waited Xs for slot` lines
(layer 2 caught alice's 4th/5th attempts if they got that far) and
clean `outcome=ok` for the survivors. The killed alice_2 may not
appear in the log at all if its cancellation hit before
`_stream_deep_with_banners` started — the `inflight.slot()` cleanup
is silent on the success path.

### 2.6 Anonymous bucket — programmatic clients share one queue

5 concurrent curls *without* a `user` field should all funnel into
`__anon__`:

```bash
rm -f /tmp/anon_*.log /tmp/anon_*.err
for i in 1 2 3 4 5; do
  curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_fast\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"anon $i\"}]}" \
    http://192.168.1.11:8000/v1/chat/completions > /tmp/anon_$i.log 2>/tmp/anon_$i.err &
  sleep 0.2
done
wait
echo "log sizes:"
wc -c /tmp/anon_*.log
```

Verification (Unraid):

```bash
docker logs --since 5m audrey-ai 2>&1 | grep 'inflight:'
```

What "pass" looks like: at least one `inflight: user=__anon__
waited Xs for slot` line for the 4th or 5th request.

If you want to exercise smoke tests *without* contention, give each
curl its own user id (`"user": "smoke-1"`, `"smoke-2"`, …) — each
gets a fresh per-user semaphore.

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
