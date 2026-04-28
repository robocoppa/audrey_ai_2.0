# Phase 18a — date/time context injection

**Goal:** every model call in the pipeline gets the current ISO-8601
date/time as a system message, so answers stop hedging recency
("as of 2024–2025…") and time-sensitive questions get accurate framing.

Why now: caught directly in a phase-18 deep run. The synth produced
"as of 2024–2025" while today is 2026-04-27. Models trained on
older snapshots don't *know* what year it is unless told. Cheap to
inject, large quality lift on time-sensitive prompts.

What changed:

- **`src/audrey/pipeline/context.py` (new)** — `iso_now()` and
  `datetime_system_message()` helpers. Single source of truth for the
  format, so future tweaks are one-file.
- **`src/audrey/pipeline/graph.py`** — new `node_datetime` is the
  graph's entry point. Runs before `memory_recall`; prepends the
  system message to `state["messages"]` so every downstream node
  (classify, fast_path, deep_panel workers, synth) sees it.
- **`src/audrey/routes/openai.py`** — streaming deep route bypasses
  the graph for orchestration, so `_phase_thinking()` calls
  `datetime_system_message()` directly. Same effect via a different
  entry point.

What stays the same:

- No new dependencies, no env vars, no config changes.
- All pipeline metrics fire at the same sites with the same labels.
- Memory recall, classify, planner, synth — none of those changed
  semantics, they just see one extra system message at the top.

Out of scope (deliberately):

- **User local timezone.** The server-side injection uses *server*
  time. For a user in a different timezone, "now" the server sees
  isn't quite "now" the user means. The companion fix (OWUI side,
  §2 below) covers this — it's optional but recommended.
- **Caching the timestamp.** Each request resolves `datetime.now()`
  fresh — accurate to the second. Caching would shave nanoseconds at
  the cost of stale timestamps on long-running containers.
- **Non-ISO formats.** ISO-8601 because models parse it more reliably
  than human formats. If a future use case wants "Monday afternoon"
  framing, it can be added without removing the ISO line.

**Prereqs:** Phase 18 verified (banner streaming working).

---

## 1. Deploy

```bash
cd /mnt/user/appdata/audrey_ai_2.0
git pull
docker compose up -d --build audrey-ai
docker compose logs --tail 20 audrey-ai | grep ready
```

No env vars. No image-level changes (stdlib `datetime` only).

---

## 2. OWUI side — user local timezone (recommended)

OWUI supports template variables in system prompts. Set a global
system prompt that injects the *user's browser-local* time alongside
audrey's server-side injection. Both pieces matter — server time
covers programmatic clients, user-local covers timezone-sensitive
questions.

In OWUI: **Admin Panel → Settings → General → Default System Prompt**:

```
User local date and time: {{CURRENT_DATETIME}}
User timezone: {{CURRENT_TIMEZONE}}
```

OWUI substitutes both values client-side (from the browser's clock)
before sending the request to audrey. The model sees:

1. Audrey's `Current server date and time: 2026-04-27T...` (always)
2. OWUI's `User local date and time: 2026-04-27T... User timezone: ...`
   (when the user is on OWUI; programmatic clients won't have this)

Order doesn't matter — both are system messages, the model can
reconcile them.

---

## 3. Smoke tests

### 3.1 Confirm the message is in the request

Run on **laptop** with `$ADMIN_TOKEN` exported. Use a non-streaming
call so the full request → response cycle is easy to inspect:

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"what year is it right now? answer with just the year."}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq -r '.choices[0].message.content'
```

Expected: the current year. If the model hedges ("I don't have access
to the current date") or returns last year, the system message isn't
landing in the worker's request — check the audrey log (next test).

### 3.2 Confirm both code paths inject

The non-streaming graph and the streaming-deep route are separate
injection points. Hit each and look at the model's output for date
awareness:

```bash
# Non-streaming (audrey_fast goes through the graph → node_datetime)
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_fast","stream":false,"messages":[{"role":"user","content":"what is today'"'"'s date?"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions | jq -r '.choices[0].message.content'

# Streaming deep (audrey_cloud goes through _phase_thinking → datetime_system_message)
curl -sS -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"audrey_cloud","stream":true,"messages":[{"role":"user","content":"what is today'"'"'s date?"}]}' \
  https://chat.builtryte.xyz/v1/chat/completions
```

Both should answer with today's date in their respective formats.

### 3.3 Container timezone sanity check (Unraid)

The injected timestamp uses the container's local timezone. If you
see UTC in the system message but you wanted (say) `America/Denver`,
set the `TZ` env var:

```bash
docker exec audrey-ai date            # what the container thinks
docker exec audrey-ai sh -c 'echo $TZ' # current setting (may be empty)
```

If empty / wrong, add to `compose.yaml` under `audrey-ai.environment`:

```yaml
environment:
  TZ: America/Denver
```

Rebuild and recheck.

---

## 4. Rollback

Revert the three files; the helper module is inert if nothing imports
it. If you want to fully remove:

```bash
git checkout <previous-sha> -- \
  src/audrey/pipeline/context.py \
  src/audrey/pipeline/graph.py \
  src/audrey/routes/openai.py
docker compose up -d --build audrey-ai
```

---

## 5. Follow-ups (not phase 18a)

- **Human-readable variant** when the use case actually needs it
  ("Monday afternoon"). Add as a second helper, gate by config flag.
- **Per-user timezone hint** beyond what OWUI sends. Today the OWUI
  side covers this for OWUI users; programmatic clients are stuck
  with server time.
