# Phase 33 - Lesson 6 model-layer hardening

Lesson-driven hardening phase. The Lesson 6 audit pass covered the
model layer: `ModelRegistry`, `HealthTracker`, and `OllamaClient`,
plus their fast-path, deep-panel, and synth call sites. Two behavior
fixes landed, two stale docstrings were cleaned up, and direct
model-layer tests were added. Lesson 6 was written alongside the code
changes so the docs now teach the current behavior.

The headline behavior change: invalid model `location` values now fail
fast at startup, and successful HTTP responses from Ollama with bad
JSON/body shape now become `OllamaError` instead of leaking raw parse
errors past the health/fallback machinery.

What stays the same:
- Model registry contents and model priorities.
- Five virtual models: `audrey_auto`, `audrey_fast`, `audrey_deep`,
  `audrey_cloud`, `audrey_local`.
- Fast/deep/synth routing shape.
- Local/cloud model names still use the same Ollama-compatible client.
- No env vars, compose changes, data migrations, or Qdrant changes.

What changed:
- **`src/audrey/models/registry.py`** -
  - Added runtime validation for `ModelSpec.location`.
  - Accepted locations are `local` and `cloud`.
  - Invalid config now raises `ValueError` while Audrey is building
    the registry during startup. This is intentional: a bad model
    registry is a bad config, not a graceful-degradation case.
- **`src/audrey/models/ollama.py`** -
  - `OllamaError` docstring now matches actual use: HTTP, transport,
    and response parsing failures.
  - `OllamaClient` docstring now describes the startup default timeout
    plus per-call timeout overrides.
  - Added `_json_object(...)` and routed `tags()`, `chat()`, and
    `embed()` through it.
  - Bad JSON or wrong top-level body shape from a 2xx Ollama response
    now raises `OllamaError`.
  - `chat()` records an error metric if response parsing fails,
    instead of counting the malformed response as `outcome="ok"`.
  - Added optional `transport=` injection for clean `httpx.MockTransport`
    tests.
- **`tests/test_models.py`** -
  - New focused coverage for registry sorting/copy behavior, healthy
    fallback selection, invalid location rejection, health cooldown /
    reset / capped backoff, and OllamaClient transport/status/body
    failures.
- **`docs/lessons/lesson-06-the-model-layer.md`** -
  - New public lesson covering the model layer through the
    `ModelRegistry -> HealthTracker -> FairLocalGate -> OllamaClient`
    path.
- **`docs/lessons/lesson-05-configuration-and-startup.md`** -
  - End-of-lesson pointer now links directly to Lesson 6.

Out of scope:
- Reworking model pool choices or priorities.
- Changing cloud/local routing policy.
- Adding live config reload for model registry changes.
- Deep ReAct behavior and tool-call prompting. Lesson 6 mentions ReAct
  only enough to explain where the selected model goes.
- Streaming JSON-line validation in `chat_stream()`. Non-JSON stream
  lines are still logged and skipped as before; the fixed malformed-body
  contract applies to `tags()`, non-streaming `chat()`, and `embed()`.

## 1. Deploy

No config or data migration required.

```bash
# Laptop:
git pull   # after the Phase 33 commit lands

# Unraid (from /mnt/user/appdata/audrey_ai_2.0):
git pull
docker compose up -d --build audrey-ai
docker compose logs --since 2m audrey-ai | grep -E "ready|Invalid model location|pipeline=compiled"
```

Expected:

- No `Invalid model location` line.
- A normal readiness line ending with `pipeline=compiled`.
- Same task types and model registry count as before.

## 2. Smoke tests

### 2.1 Confirm Audrey starts with the current model registry

```bash
docker compose logs --since 2m audrey-ai | grep "ready:"
```

Expected: one readiness line with task types including `code`,
`reasoning`, `general`, and `vl`.

### 2.2 Confirm a fast request still works

Use any OpenAI-compatible client or curl with a valid OWUI bearer token.
If running from the Unraid host, `localhost:8000` is correct. If
running from your laptop, replace it with `http://<unraid-ip>:8000`.

```bash
USER_TOKEN="<valid OWUI bearer token>"
AUDREY_URL="http://localhost:8000"

curl -sS -o /tmp/audrey-health.json -w "HTTP %{http_code}\n" "$AUDREY_URL/health"
cat /tmp/audrey-health.json

curl -sS -o /tmp/audrey-fast.json -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_fast",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Briefly explain what BTRFS is."}
    ]
  }'

cat /tmp/audrey-fast.json | jq .
cat /tmp/audrey-fast.json | jq '.model, .choices[0].message.content'
```

Expected: `HTTP 200` and a normal non-streaming chat response. In logs,
the fast path should dispatch one concrete model for task `general`.
If the HTTP code is not 200, inspect the full JSON body printed by
`jq .` before looking at the filtered `.model` / `.choices` fields.

### 2.3 Confirm a deep request still works

```bash
curl -sS -o /tmp/audrey-deep.json -w "HTTP %{http_code}\n" \
  "$AUDREY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audrey_deep",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Compare BTRFS snapshots and ZFS snapshots in a short table."}
    ]
  }'

cat /tmp/audrey-deep.json | jq .
cat /tmp/audrey-deep.json | jq '.model, .choices[0].message.content'
```

Expected: `HTTP 200`, plus deep-panel logs showing workers and a
synthesizer. A single worker failure should not kill the whole response;
it should become draft error data and synthesis should proceed if
another draft exists.

### 2.4 Optional negative config check

Only do this on a disposable branch or after making a backup of
`config.yaml`.

```bash
cp config.yaml /tmp/config.yaml.phase33.bak

# Edit one model entry temporarily:
# location: clodu

docker compose up -d --build audrey-ai
docker compose logs --since 1m audrey-ai | grep "Invalid model location"
```

Expected: Audrey fails startup with a clear `Invalid model location`
message naming the task/model. Restore your backup immediately and
restart:

```bash
cp /tmp/config.yaml.phase33.bak config.yaml
docker compose up -d --build audrey-ai
```

Do not run this check if there are unrelated local config edits you
need to keep.

### 2.5 Local verification

On the laptop:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src/audrey/models tests/test_models.py
git diff --check
```

Expected from this phase before commit:

- `120 passed`
- Ruff: `All checks passed!`
- `git diff --check`: no output

## 3. Rollback

Plain git rollback. No migrations or persisted state changes.

```bash
git revert <phase-33-commit>
docker compose up -d --build audrey-ai
```

If rollback happens because startup now catches a bad `location`, prefer
fixing the config typo instead of reverting the validation. The new
failure is protecting the GPU scheduling contract.

## 4. Operational notes

- A startup failure mentioning `Invalid model location` points to
  `config.yaml` model registry data, not Ollama availability.
- An `OllamaError` mentioning invalid JSON or expected JSON object now
  means Audrey reached Ollama but the response body was not shaped like
  Audrey's client contract. That model should be cooled down by the
  existing health path.
- The new tests use `httpx.MockTransport`; they do not require Ollama
  or network access.

## 5. Lesson status

Lesson 6 is ready for review:

- `docs/lessons/lesson-06-the-model-layer.md`
- Lesson 5 now links forward to Lesson 6.

The public lesson avoids audit internals and teaches the model layer as
it exists after this phase.
