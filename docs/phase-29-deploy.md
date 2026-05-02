# Phase 29 — Starter test suite

Six earlier phases each shipped at least one bug that a unit test
would have caught:

- **Phase 23** `FairLocalGate._release` round-robin starvation —
  pure asyncio, no Ollama needed.
- **Phase 26** `me.id` instead of `me.email` — instant `AttributeError`
  under a mocked `AuthedUser`.
- **Phase 27** `KB_WATCHER_ENABLED=1` missing from compose default —
  watcher integration test would've shown `_run` never started.
- **Phase 27** SSRF `_validate_image_url` — already verified ad-hoc
  against 13 cases, not committed.
- **Phase 28** `tool_summary_block` formatter — verified at the python
  REPL during deploy, never committed.

Phase 29 commits the test suite. Six test files, ~360 LOC, **101
assertions in 0.5s**.

What stays the same:
- All production code paths. Tests are additive.
- No CI wiring (deferred). Run locally on the laptop with `pytest`.
- No coverage gate. First test suite — the goal is regression guards
  on known-fragile spots, not a percent.

What changed:
- **`tests/test_banners.py`** (14 cases) — `tool_summary_block` +
  `_format_calls` formatter. Locks down per-worker rendering, error
  marker OR, empty-row drop, leading `\n\n---\n` break.
- **`tests/test_fair_gate.py`** (7 cases) — cloud bypass, single-user
  serialization, **two-user round-robin (Phase 23 starvation guard)**,
  cancellation safety, anonymous-bucket funnel, concurrency clamp.
- **`tests/test_reflect.py`** (27 cases — 16 cue parametrizations + 11
  standalone) — every `_BREVITY_CUES` entry, case-insensitivity,
  `ok_brevity_requested` vs `ok` reason distinction, no_drafts short
  circuit, whitespace-strip before length check.
- **`tests/test_classify.py`** (11 cases) — `_tool_mention_signal`
  ordering, the Phase 8 vl_strong override, code-strong override,
  word-boundary guard, fall-through to keyword chain.
- **`tests/test_kb_embed_ssrf.py`** (22 cases — parametrized 9 IPs +
  5 schemes + standalone) — promotes the 13 ad-hoc Phase 27
  verification cases. Stubs `socket.getaddrinfo` so tests run offline.
- **`tests/test_auth.py`** (20 cases) — happy path, 401/502 paths,
  pending-role rejection, header-parsing edges, cache-hit reuse,
  targeted eviction, **and the Phase 26 `me.id` regression guard**.
- **`src/audrey/pipeline/reflect.py`** — drop unused `re` import
  (incidental cleanup found while writing the tests).

Out of scope (deliberately):

- **No integration tests against live Ollama / Qdrant / custom-tools.**
  Those stay in `docs/phase-N-deploy.md` smoke tests. Keeps the suite
  hermetic and fast (0.5s total).
- **No CI wiring.** GitHub Actions / similar is a Phase 30+ topic if
  it earns its place. Right now the suite runs locally during dev.
- **No coverage threshold.** The first test suite shouldn't gate on
  percent — let the suite mature first.
- **No tests for synthesize / planner / deep_panel.run_panel.** They
  orchestrate too many external calls; skip until something breaks.
- **No tests for the streaming SSE plumbing.** The framing logic
  (`_delta_frame` / `_stop_frame`) is a pass-through; the substantive
  test is the smoke run in `phase-28-deploy.md`.

**Prereqs:** all phases through 28 verified. Adds nothing to the
runtime image — `pytest` lives in `[project.optional-dependencies].dev`,
which the production Dockerfile doesn't install.

---

## 1. Deploy

There is no production deploy for Phase 29 — the suite never runs
inside the audrey-ai container at runtime. To run the tests:

### Laptop (primary path)

```bash
# One-time setup if you haven't installed dev extras:
uv pip install -e ".[dev]"

# Run all tests:
.venv/bin/pytest tests/ -v

# Run one file:
.venv/bin/pytest tests/test_fair_gate.py -v

# Run one test:
.venv/bin/pytest tests/test_fair_gate.py::test_two_user_round_robin_skips_last_granted -v
```

Expected: `101 passed in <1s`.

### Unraid (optional sanity check)

The audrey-ai production image doesn't include pytest. To run the
suite against the deployed image you'd need to install dev extras
inside the container — not worth doing routinely, but useful for
post-rebuild verification:

```bash
docker exec -it audrey-ai pip install pytest pytest-asyncio
docker exec -it audrey-ai pytest /app/tests/ -v
```

Skipping this is fine. The suite has zero external dependencies (no
Ollama / Qdrant / OWUI) and its behavior is identical on laptop and
container.

---

## 2. Smoke checks

Phase 29 verifies itself by running. The "smoke" is just `pytest`
returning green.

### 2.1 All tests pass

```bash
.venv/bin/pytest tests/ -q
```

Expected: `101 passed in <1s`. Anything else means a regression.

### 2.2 Phase 23 round-robin starvation guard fires

```bash
.venv/bin/pytest tests/test_fair_gate.py::test_two_user_round_robin_skips_last_granted -v
```

Expected: green. This test is the load-bearing one — if a future
refactor of `FairLocalGate._release` reverts to FIFO-by-bucket
behavior, this test fails first.

### 2.3 Phase 26 `me.id` regression guard fires

```bash
.venv/bin/pytest tests/test_auth.py::test_authed_user_has_no_id_field -v
```

Expected: green. If anyone adds an `.id` field to `AuthedUser` in a
refactor, this test fails immediately, forcing the change to be
deliberate rather than accidental.

### 2.4 Phase 28 footer formatter pinned down

```bash
.venv/bin/pytest tests/test_banners.py -v
```

Expected: 14/14 green. `_format_calls` and `tool_summary_block`
behavior is now committed, not just verified at the REPL.

### 2.5 Phase 27 SSRF guard pinned down

```bash
.venv/bin/pytest tests/test_kb_embed_ssrf.py -v
```

Expected: 22/22 green. The 13 ad-hoc cases from Phase 27 verification
are now committed.

---

## 3. Rollback

Delete the `tests/` directory contents. Production code is unchanged
except for the unused-import cleanup in `reflect.py` — revert that
file too if you want a strict rollback:

```bash
rm tests/test_*.py
git checkout src/audrey/pipeline/reflect.py
```

No rebuild needed; no production behavior was touched.

---

## 4. Operational notes

- **Tests run on the laptop, not Unraid.** The production image
  doesn't carry pytest. Unraid sanity check is optional and rarely
  needed.
- **Test wall-clock is ~0.5s for the whole suite.** If a test starts
  taking measurable time, it's probably reaching out to a real
  service that the stub didn't catch — investigate, don't add a
  long timeout.
- **`test_fair_gate.py` uses `asyncio.Event` for ordering.** Never
  `asyncio.sleep` for synchronization — that's the route to flaky
  tests on slow runners. The `await asyncio.sleep(0)` calls in the
  fair-gate tests are deliberate (yield to the event loop), not
  timing-dependent.
- **`test_auth.py` uses a hand-rolled `_FakeAsyncClient`** instead of
  `respx`. The auth module only ever calls `AsyncClient().get(...)`
  once, so a 30-line fake is smaller than the dep would be. If
  another module ever needs more elaborate httpx mocking, revisit.
- **`test_kb_embed_ssrf.py` stubs `socket.getaddrinfo`.** The real
  function would hit DNS — slow, non-hermetic, and you'd need to
  control private IP responses, which you can't. The stub returns
  whatever IP the test asks for; the SSRF guard's behavior is
  proven against synthetic IPs.
- **What this suite would NOT have caught:** the Phase 27 compose
  hardening (`tools=0` race + `KB_WATCHER_ENABLED=1` env wiring).
  Both are configuration concerns that fail at container startup,
  not at function-call boundaries. A future "deploy-time integration
  smoke" (start the container, hit `/v1/models`, check tool count)
  would catch them — but that's not Phase 29's scope.
- **Ruff status at end of Phase 29:** `.venv/bin/ruff check src/audrey/`
  reports 130 pre-existing warnings. Phase 29 did not add to them
  meaningfully (5 added in tests/, mostly intentional unicode and
  one `noqa`'d S104 false-positive). A category-by-category sweep is
  queued in memory as a future cleanup pass.
