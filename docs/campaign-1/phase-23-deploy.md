# Phase 23 — Fast-path GPU gating (gate-per-call inside ReAct)

**Goal:** close the long-standing fast-path gating gap. Today
`run_fast_path` and the streaming fast path call `ollama.chat` /
`ollama.chat_stream` directly without acquiring `FairLocalGate`. Phase
20's per-user in-flight cap mitigates at the request level (max 3 deep
requests per user), but a single user's 3 concurrent fast-path local
calls — or 2+ users' concurrent local fast calls — still hit the GPU
simultaneously, violating the `concurrency=1` PSU/VRAM invariant.

Fix: thread the gate through `run_fast_path`, the streaming
`_stream_openai`, and `run_react`. Gate is acquired **per `ollama.chat`
call**, not held across the whole request — so during a slow tool
dispatch (e.g. 30s `web_search`), the GPU is free for other users'
work. This is the "Option B" from the Phase 23 plan; the only cost is
a possible model-reload tax between rounds if another user slips in,
which is bounded and visible in `audrey_gpu_gate_wait_seconds`.

Deep panel is **unchanged**: `_run_one_worker` keeps holding the gate
for the whole worker (rounds + tool dispatch). It passes `gate=None`
into `run_react` so ReAct doesn't try to double-acquire. The "hold for
whole worker" pattern is correct for deep panel because all N workers
will run regardless of gate granularity — releasing between rounds
would only shuffle order, not save GPU time.

**Fairness-bias fix (bundled into Phase 23):** while validating Phase
23 we caught a pre-existing bug in `FairLocalGate._release` from
Phase 20 — a user with multiple queued waiters stayed at the head of
the round-robin OrderedDict because `move_to_end()` is a no-op when
they're the only entry. New buckets joining the queue got appended
*behind* the head, so they only got served after the head's queue
fully drained. That's FIFO-by-bucket, not round-robin.

The fix tracks `_last_granted` explicitly. On release, the picker
walks the dict and skips the last-granted bucket whenever another
bucket has waiters, falling back to re-granting only when the
last-granted is the *only* bucket queued. Verified locally with a
multi-user test: alice fires 3 staggered requests, bob fires a 4th
mid-flight, charlie fires a 5th later → grant order is now
`a1, b1, a2, c1, a3, a4, a5` (alternation), not `a1, a2, a3, b1, c1`
(starvation). Phase 20's deep-panel smoke test 2.4 happened to mask
this in deep mode because each request makes ~10 acquires (one per
worker × multiple rounds), amortizing the bias. Fast path with 1-3
acquires per request makes it stark.

What stays the same:

- All existing metrics — no new metric in Phase 23. The existing
  `audrey_gpu_gate_wait_seconds` histogram now picks up fast-path
  contention automatically.
- Deep-panel ordering and worker scheduling.
- Cloud calls bypass the gate (`location != "local"` makes
  `gate.acquire` a no-op).
- Per-user in-flight cap (Phase 20 layer 2) — unchanged.

What changed:

- `src/audrey/pipeline/react.py` — `run_react` accepts optional
  `gate: FairLocalGate | None` and `location: str`. Each `ollama.chat`
  (rounds 0..N-1 plus the forced-final call) is wrapped with
  `gate.acquire(model, location, user_id)`. Tool dispatch
  (`asyncio.gather(dispatch_one(...))`) sits *outside* the gate hold.
- `src/audrey/pipeline/fast_path.py` — `run_fast_path` now requires
  `gate: FairLocalGate`. Non-tools branch wraps the single
  `ollama.chat` in `gate.acquire`. Tools branch passes the gate down
  into `run_react`.
- `src/audrey/routes/openai.py` — streaming non-tools fast path
  (`_stream_openai`) accepts `gate`, `location`, `user_id` kwargs and
  wraps the `chat_stream` loop with `gate.acquire`. The streaming
  *tool-capable* fast path already routes through the graph
  (`run_fast_path` via `node_fast_path`), so it inherits the gate fix
  automatically.
- `src/audrey/pipeline/deep_panel.py` — `_run_one_worker` passes
  `gate=None, location=location` into `run_react`. Comment added
  explaining why (worker-level gate hold is correct for deep panel).
- `src/audrey/pipeline/graph.py` — `node_fast_path` passes `gate` into
  `run_fast_path`.
- `src/audrey/pipeline/fair_gate.py` — round-robin `_release` fix:
  added `_last_granted` field; picker skips the last-granted bucket
  when another bucket has waiters. Pre-existing bias from Phase 20
  surfaced by Phase 23's fast-path traffic.

Out of scope (deliberately):

- **Reentrant gate.** Could let deep panel pass `gate=gate` into ReAct
  and have the gate ignore re-acquires from the same task. Adds
  complexity for no win — the explicit `gate=None` opt-out is clearer.
- **Tool-dispatch metrics under the gate.** Phase 22's
  `audrey_tool_call_seconds` already covers per-tool latency. Whether
  the gate was held during dispatch isn't useful by itself — what
  matters is `audrey_gpu_gate_wait_seconds`, which already exists.
- **Per-round gate-wait observation.** Each `gate.acquire` already
  feeds `audrey_gpu_gate_wait_seconds`, so a fast-path ReAct request
  with 3 rounds will produce 3 observations. That's correct — each
  acquire is a real wait. The histogram count will grow faster than
  request count under tool-using fast-path traffic.

**Prereqs:** Phase 20 verified. No env vars, no migrations.

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 5 audrey-ai | grep ready
```

Custom-tools is not affected. No prometheus / grafana changes.

---

## 2. Smoke tests

> Reminder: from the Unraid host, `curl http://audrey-ai:8000` does
> *not* resolve — that's docker-DNS only inside `ollama-net`. Use
> `http://localhost:8000` from the host, or `http://192.168.1.11:8000`
> from another LAN box, or `docker exec audrey-ai curl ...`.

### 2.1 Container starts cleanly

```bash
docker compose logs --tail 30 audrey-ai | grep -E 'ready|ERROR|Traceback'
```

Expect: one `ready: ...` line, no errors. If you see
`TypeError: run_fast_path() missing 1 required positional argument:
'gate'`, the graph.py change didn't land — rebuild with `--no-cache`.

### 2.2 Single-user fast-path contention (the headline)

Goal: prove the gate is now acquired on local fast-path requests.

```bash
# Reset the gate-wait histogram baseline by reading current count.
BEFORE=$(curl -s http://localhost:8000/metrics \
  | grep '^audrey_gpu_gate_wait_seconds_count ' \
  | awk '{print $2}')

# Fire 5 concurrent audrey_fast requests. Pick a prompt long/complex
# enough to actually run on a local model (audrey_fast picks the
# first healthy model for the classified task; for a short general
# prompt that's typically a local 35b).
for i in 1 2 3 4 5; do
  curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_fast\",\"stream\":false,\"user\":\"smoke-23-2\",\"messages\":[{\"role\":\"user\",\"content\":\"give me a one-paragraph history of the screwdriver request $i\"}]}" \
    http://localhost:8000/v1/chat/completions \
    -o /tmp/resp_$i.json &
done
wait

AFTER=$(curl -s http://localhost:8000/metrics \
  | grep '^audrey_gpu_gate_wait_seconds_count ' \
  | awk '{print $2}')

DELTA=$(awk "BEGIN {print $AFTER - $BEFORE}")
echo "gate_wait observations: before=$BEFORE after=$AFTER (delta=$DELTA)"
```

Note: prometheus exposes histogram `_count` as a float (`46.0`), so
we use `awk` for the subtraction — bash's `$(())` rejects decimals.

Expect: delta ≥ 5 (one observation per `ollama.chat` call;
non-tool-capable models do exactly 1). If delta is 0, the gate isn't
being acquired — fast path didn't get the new code.

If the chosen fast model is tool-capable and tools fire, delta will
be larger (one observation per ReAct round + one for the forced-final
call). That's expected.

### 2.3 Two-user fairness across fast-path local

Goal: prove round-robin works for fast path the same way it does for
deep.

**Important:** background alice's three requests so they're truly
concurrent. The naïve `for ...; do curl; done` (no `&`) runs them
*serially*, which means bob never competes for the gate and the test
proves nothing.

Terminal A — fire all three of alice's requests in the background:
```bash
for i in 1 2 3; do
  ( time curl -sS -X POST -H "Authorization: Bearer $ALICE_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"audrey_fast\",\"stream\":false,\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"alice request $i — write 200 words on the history of typewriters\"}]}" \
      http://localhost:8000/v1/chat/completions \
      -o /tmp/alice_$i.json ) 2> /tmp/alice_${i}.time &
done
```

Terminal B — fire bob ~5s later, while alice's three are all still
in flight:
```bash
sleep 5
time curl -sS -X POST -H "Authorization: Bearer $BOB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"user":"bob","messages":[{"role":"user","content":"bob request — write 200 words on the history of bicycles"}]}' \
  http://localhost:8000/v1/chat/completions \
  -o /tmp/bob.json
```

Back in terminal A:
```bash
wait
cat /tmp/alice_1.time /tmp/alice_2.time /tmp/alice_3.time
```

**Expect:** bob's request finishes *second* — after alice_1 (which
held the gate when bob arrived) but *before* alice_2 and alice_3
(which were queued behind alice_1). The fairness round-robin in
`FairLocalGate` skips back-to-back grants to the same user when
another bucket has waiters. So the grant order should be:

```
alice_1 (was running) → bob → alice_2 → alice_3
```

To verify the order from logs:
```bash
docker compose logs --since 5m audrey-ai | grep -E 'react: round=|fast_path task='
```

The `react: round=0` lines carry elapsed seconds — match each one
back to the request that started that long ago. You should see bob's
chat call complete *before* alice_2 and alice_3's first chat calls.

Common mistakes:

- **Loop without `&`** runs serially → bob never competes → test
  proves nothing.
- **Same user-id for all four requests** (e.g. forgetting to change
  `"user":"alice"` to `"user":"bob"`) → all four go into one bucket,
  no round-robin to test.
- **Anonymous bob** (omitting `"user"` field) → bob falls into the
  `__anon__` bucket, which is its own bucket and *should* still
  round-robin against alice; just be aware.

### 2.4 Tool-dispatch releases the GPU (the headline benefit)

Goal: prove that during a slow tool dispatch, another user can run
local GPU work.

This requires a tool-capable local model in the fast path. Default
is `qwen3.6:35b` per `config.yaml fast_path.tool_capable_models`.

Terminal A — fire a fast-path request that will trigger `web_search`
(the slowest realistic tool, ~5-30s wall clock):
```bash
time curl -sS -X POST -H "Authorization: Bearer $ALICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"user":"alice","messages":[{"role":"user","content":"search the web for the current Bitcoin price and tell me what you found"}]}' \
  http://localhost:8000/v1/chat/completions \
  -o /tmp/alice_web.json
```

Terminal B — 2-3 seconds after A starts (when alice is mid-tool-dispatch):
```bash
sleep 3
time curl -sS -X POST -H "Authorization: Bearer $BOB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"user":"bob","messages":[{"role":"user","content":"in one sentence, what is rsync?"}]}' \
  http://localhost:8000/v1/chat/completions \
  -o /tmp/bob_short.json
```

Expect: bob's request finishes well before alice's. Bob's `time real`
should be ~5-15s (one local model run). Pre-fix, bob would have had
to wait for alice's GPU slot to free, which doesn't happen until
alice's tool dispatch returns — making bob's total latency dominated
by alice's web_search wait.

Verify via metrics:
```bash
curl -s http://localhost:8000/metrics \
  | grep -E '^audrey_gpu_gate_wait_seconds_(bucket|count|sum)' \
  | head -20
```

Expect: at least one observation in the `0.5`/`2.0` range from bob
(short wait while alice's first chat call is still running) — *not*
a 30s observation (which would mean bob waited the whole tool
dispatch).

### 2.5 Cloud is unchanged

Goal: prove `audrey_cloud` and any cloud worker doesn't hit the gate.

```bash
BEFORE=$(curl -s http://localhost:8000/metrics \
  | grep '^audrey_gpu_gate_wait_seconds_count ' \
  | awk '{print $2}')

curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":false,"messages":[{"role":"user","content":"in two sentences, what is rsync?"}]}' \
  http://localhost:8000/v1/chat/completions \
  -o /tmp/cloud.json

AFTER=$(curl -s http://localhost:8000/metrics \
  | grep '^audrey_gpu_gate_wait_seconds_count ' \
  | awk '{print $2}')

echo "cloud delta: $(awk "BEGIN {print $AFTER - $BEFORE}")"
```

Expect: delta = 0. Cloud workers don't hit the local gate. (If the
cloud synth picks a local model for some reason — it shouldn't —
delta could be 1; investigate the synth fallback in that case.)

### 2.6 Deep panel hasn't regressed

Goal: prove deep panel still works and total wall-clock hasn't
materially changed.

```bash
time curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_local","stream":false,"messages":[{"role":"user","content":"give me a 200-word comparison of ZFS and BTRFS"}]}' \
  http://localhost:8000/v1/chat/completions \
  -o /tmp/local_deep.json
```

Expect: similar wall-clock to pre-Phase-23 `audrey_local` baseline
(typically 90-180s). Look for `deep_panel: pool=deep_panel_local
... ok=N` in the logs — workers should still complete in serial as
they always have. If wall-clock has grown materially, the
`gate=None` opt-out in `_run_one_worker` didn't land.

### 2.7 Streaming fast path is gated

```bash
BEFORE=$(curl -s http://localhost:8000/metrics \
  | grep '^audrey_gpu_gate_wait_seconds_count ' \
  | awk '{print $2}')

curl -sS -N -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":true,"user":"smoke-23-7","messages":[{"role":"user","content":"in 200 words, what is the history of the lightbulb?"}]}' \
  http://localhost:8000/v1/chat/completions \
  > /tmp/stream.txt

# Wait a sec for metric to flush.
sleep 1
AFTER=$(curl -s http://localhost:8000/metrics \
  | grep '^audrey_gpu_gate_wait_seconds_count ' \
  | awk '{print $2}')

echo "stream delta: $(awk "BEGIN {print $AFTER - $BEFORE}")"
head -c 400 /tmp/stream.txt
```

Expect: delta = 1 (one acquire for the whole token stream — that's
the right granularity since tokens are GPU-bound start to finish, no
tool dispatch to release for in this branch). The output should show
SSE frames with `delta: {"content": ...}`.

### 2.8 Inflight cap still works

```bash
# Same user, 5 simultaneous deep requests. Only 3 should enter the
# pipeline at once; the 4th and 5th wait at the inflight semaphore.
for i in 1 2 3 4 5; do
  curl -sS -X POST -H "Authorization: Bearer $ALICE_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"audrey_local\",\"stream\":false,\"user\":\"alice\",\"messages\":[{\"role\":\"user\",\"content\":\"deep request $i — 100 words on rsync\"}]}" \
    http://localhost:8000/v1/chat/completions \
    -o /tmp/inflight_$i.json &
done
wait

docker compose logs --since 8m audrey-ai | grep "inflight: user=alice"
```

Expect: 2 lines of `inflight: user=alice waited <s>s for slot`
(requests 4 and 5 waited). Phase 23 didn't touch this layer — this
test confirms it didn't regress.

---

## 3. Rollback

```bash
git checkout <previous-sha> -- \
  src/audrey/pipeline/react.py \
  src/audrey/pipeline/fast_path.py \
  src/audrey/pipeline/deep_panel.py \
  src/audrey/pipeline/graph.py \
  src/audrey/routes/openai.py
docker compose up -d --build audrey-ai
```

Five-file revert. No config / no env / no schema changes.

---

## 4. Operational notes for future work

- **Reload tax visibility.** If the per-call gate acquire pattern
  causes noticeable model reloads in practice (you'll see it in
  Ollama logs as repeated `loading model` lines for the same model),
  the mitigation is to drop more local models from
  `fast_path.tool_capable_models` so fewer fast-path requests trigger
  the multi-round ReAct loop. Today only `qwen3.6:35b` is local
  in that set; the others are cloud, which doesn't hit this codepath.
- **Histogram interpretation.** `audrey_gpu_gate_wait_seconds_count`
  no longer corresponds 1:1 with requests. A tool-using fast-path
  request can produce 3+ observations (one per ReAct round plus the
  forced-final). Aggregate `_sum` divided by `_count` still gives a
  meaningful average wait time *per chat call*; if you want average
  *per request*, you'll need to combine with `audrey_pipeline_total`.
- **The streaming fast path holds the gate longer than ideal.** It
  currently wraps the entire `async for chunk in chat_stream(...)`
  loop. If a token stream takes 60s but only generates content for
  10s, the gate is held during the silent reasoning portion too.
  That's correct — Ollama is computing on the GPU the whole time —
  but it means streaming fast-path is the most contention-prone
  branch. The fix if this becomes painful is non-trivial (chunk-level
  release would force per-token reload), so leave it.
