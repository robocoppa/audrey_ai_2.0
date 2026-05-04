# Phase 22 — observability bundle

Three loosely related Phase-17 follow-ups bundled into one deploy:

- **A.** Per-tool dispatch metrics inside the ReAct loop —
  `audrey_tool_calls_total{tool, outcome}` Counter and
  `audrey_tool_call_seconds{tool}` Histogram. Closes the gap where we
  could see model dispatches but not which *tool* fired (or whether it
  succeeded, errored, or timed out).
- **B.** Prometheus alert rules — error-rate alerts on the pipeline,
  per-tool, per-cloud-model. Catches problems without you needing to
  manually grep metrics.
- **C.** Targeted auth-cache eviction —
  `POST /v1/admin/auth/clear/{email}`. Replaces the originally-planned
  OWUI user-deletion webhook (OWUI v0.9.x doesn't support outbound
  webhooks for user-deletion — verified against v0.9.2 source). Manual
  drill instead: admin deletes a user in OWUI, immediately curls this
  endpoint, target user's tokens evicted in <1s without disturbing other
  cached sessions.

What stays the same:

- All existing metrics, log shapes, route prefixes, auth model.
- 30s auth cache TTL — unchanged. The new endpoint is an *additional*
  manual lever, not a replacement for the TTL.
- Cloudflared rules — `^/v1/admin` already covers the new
  `/v1/admin/auth/clear/{email}` path.

Out of scope (deliberately):

- **Auto-eviction on upstream 401.** Caching by definition skips the
  upstream call, so a stale entry can't observe a fresh 401. Only
  options are TTL shortening or manual eviction; we already have the
  former, and Phase 22C adds the latter.
- **Per-token eviction.** A user can have multiple sessions
  (multi-device); evicting one token doesn't help when the operator's
  intent is "this user is gone." Email-keyed eviction sweeps all of a
  user's tokens at once.
- **OWUI-side webhook implementation.** Researched, not feasible on
  v0.9.x. Would require a fork, which is out of scope for our infra.

**Prereqs:** Phase 19 + 20 + 21 verified. No env vars, no migrations.

---

## 1. Deploy

### Audrey

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 5 audrey-ai | grep ready
```

### Prometheus rules (Phase 22B)

> **Phase 24 update:** the Prometheus stack now lives in the audrey
> repo at `monitoring/`, and the rules dir is bind-mounted directly
> from `monitoring/prometheus-rules/` — **no `cp` step required**.
> The instructions below describe the pre-Phase-24 layout where
> rules had to be copied into `/mnt/user/appdata/prometheus/rules/`.
> Post-Phase-24, just `git pull` on Unraid and reload Prometheus.
> See `docs/phase-24-deploy.md`.

The canonical rule file lives at
`monitoring/prometheus-rules/audrey.yml` in the audrey repo. Copy it
into the prometheus container's rules directory and reload:

```bash
mkdir -p /mnt/user/appdata/prometheus/rules
cp /mnt/user/appdata/audrey_ai_2.0/monitoring/prometheus-rules/audrey.yml \
   /mnt/user/appdata/prometheus/rules/audrey.yml

# Reload prometheus without restarting
curl -sX POST http://192.168.1.11:9090/-/reload
```

For `--web.enable-lifecycle` to allow the reload, the prometheus
compose at `/mnt/user/appdata/prometheus/compose.yaml` must have it
in the `command:` block. If reload returns a 405, restart instead:

```bash
docker restart prometheus
```

The prometheus config also needs to know about the rules directory.
Add to `/mnt/user/appdata/prometheus/prometheus.yml` if not already
present:

```yaml
rule_files:
  - "rules/*.yml"
```

And the corresponding mount in
`/mnt/user/appdata/prometheus/compose.yaml`:

```yaml
volumes:
  - ./rules:/etc/prometheus/rules:ro
```

---

## 2. Smoke tests

### 2.1 Per-tool metrics fire (Phase 22A)

Trigger a request that exercises tools — a code-task or factual prompt
that the ReAct loop will likely call `web_search` or `kb_search` for:

```bash
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"user":"smoke","messages":[{"role":"user","content":"what is the latest version of python? use web search to confirm"}]}' \
  http://192.168.1.11:8000/v1/chat/completions | jq -r '.choices[0].message.content' | head -3
```

Then check the new metrics:

```bash
curl -s http://192.168.1.11:8000/metrics | grep -E 'audrey_tool_calls_total|audrey_tool_call_seconds_count'
```

What "pass" looks like:
- At least one `audrey_tool_calls_total{tool="web_search",outcome="ok"}`
  series with count >= 1.
- Corresponding `audrey_tool_call_seconds_count{tool="web_search"}`
  series with the same count.
- No mystery series (only `tool ∈ {kb_search, kb_image_search,
  memory_recall, memory_search, memory_store, web_search}`).

### 2.2 Failure outcome distinguishable from timeout (Phase 22A)

Force a tool failure by stopping `custom-tools` briefly:

```bash
docker stop custom-tools
sleep 2

# Fire a request that should call memory_search
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"user":"smoke","messages":[{"role":"user","content":"recall my hardware setup"}]}' \
  http://192.168.1.11:8000/v1/chat/completions > /dev/null

docker start custom-tools

# Check that error/timeout outcome was recorded
curl -s http://192.168.1.11:8000/metrics | grep -E 'audrey_tool_calls_total.*outcome="(error|timeout)"'
```

Expected: at least one count > 0 for `outcome="error"` or `outcome="timeout"`.

### 2.3 Prometheus rules loaded (Phase 22B)

```bash
curl -s http://192.168.1.11:9090/api/v1/rules | jq '.data.groups[] | select(.name=="audrey") | .rules[].name'
```

Expected output:

```
"AudreyPipelineErrorRate"
"AudreyToolCallErrorRate"
"AudreyToolCallLatencyP95"
"AudreyCloudModelErrorRate"
```

If empty, the rules file isn't loaded — check the prometheus config
for `rule_files` and the mount.

### 2.4 Targeted cache eviction (Phase 22C)

```bash
# 1. Populate the cache for a test user (any cached lookup works)
curl -sS -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://192.168.1.11:8000/v1/admin/auth/status | jq

# Note the cached_entries count.

# 2. Make any audrey call to ensure the admin user's entry is cached
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"hello"}]}' \
  http://192.168.1.11:8000/v1/chat/completions > /dev/null

# 3. Check cached_entries went up by 1.

# 4. Evict yourself (or replace robocoppa@proton.me with a real user email)
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://192.168.1.11:8000/v1/admin/auth/clear/robocoppa@proton.me | jq
```

Expected response:

```json
{
  "cleared": 1,
  "email": "robocoppa@proton.me",
  "by": "robocoppa@proton.me"
}
```

Idempotency: calling the same endpoint twice returns `cleared: 0` the
second time.

Smoke test passes when:
- A real email evicts >= 1 entry on first call, 0 on second.
- A nonexistent email returns `cleared: 0`, no error.
- Other users' cached entries are untouched (use
  `/v1/admin/auth/status` to verify cached_entries went down by exactly
  the expected amount).

---

## 3. Operational drill — using 22C after deleting a user in OWUI

```bash
# 1. (in OWUI admin UI) delete the user user@example.com
# 2. Immediately:
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://chat.builtryte.xyz/v1/admin/auth/clear/user@example.com
# 3. Done. Their cached tokens (if any) are gone in <1s; their next
#    audrey request gets 401 from OWUI's probe.
```

If you forget step 2, the worst case is the deleted user retains
access for up to 30s (the cache TTL) before the next probe rejects
them. Not catastrophic, but tighter is better.

---

## 4. Rollback

Phase 22 is additive — no existing routes / metrics / behavior
changed. Reverting is one commit:

```bash
git checkout <previous-sha> -- \
  src/audrey/auth.py \
  src/audrey/routes/admin.py \
  src/audrey/tools/dispatch.py \
  src/audrey/metrics.py \
  monitoring/prometheus-rules/audrey.yml \
  docs/phase-22-deploy.md
docker compose up -d --build audrey-ai

# Optional — remove the prometheus rule
rm /mnt/user/appdata/prometheus/rules/audrey.yml
curl -sX POST http://192.168.1.11:9090/-/reload
```

The new metrics' label dimensions add bytes to `/metrics` body; nothing
else regressing.

---

## 5. Follow-ups (not Phase 22)

- **Auto-eviction on user-not-found from OWUI probe.** Today a deleted
  user's stale dict entry just sits there until TTL or sweep — not a
  security hole (the entry won't be returned), but cosmetically
  unclean. Could pop the dead entry on the next probe attempt.
- **Per-tool dashboard panel in Grafana.** The new metrics are now
  available; an actual dashboard panel would surface them visually
  alongside the existing pipeline panels.
- **OWUI fork or sidecar.** If user-deletion auto-eviction becomes
  important enough, build a tiny polling daemon that periodically
  reconciles OWUI's user list against audrey's auth cache. Not worth
  it for current scale.
