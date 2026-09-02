# Security Remediation — Item 1: Unpublish host ports

_Status: **CLOSED / SUPERSEDED** — the original no-port strategy was not the final design._

Audrey port 8000 was later restored for authenticated LAN clients such as
Hermes after the KB routes gained user/service-token authentication.
custom-tools remains unpublished. The working log below records the discarded
Strategy A and is retained only as history, not as a deploy runbook.

This is the working log for Item 1 of the security review remediation plan
([`security-review-remediation-plan.md`](security-review-remediation-plan.md) —
the master index of all four findings and their sequencing). It captures the
plan, the decisions, the exact changes, and the verification steps. It originally doubled as the deploy runbook for that discarded change.

## Finding being addressed

**Finding #1 (High) — Cross-user data access via unauthenticated LAN-published
routes.**

`compose.yaml` publishes `audrey-ai` on host port `8000` and `custom-tools` on
host port `8001`, both binding `0.0.0.0` on the Unraid host. Neither service's
data routes are behind `require_user`:

- custom-tools (`tools-server/app.py`): `/memory_search`, `/memory_recall`,
  `/memory_store`, `/kb_search`, `/chat_history_search` all take `user` as a
  plain request-body field with no auth binding.
- Audrey KB (`src/audrey/routes/kb.py`): `/v1/kb/query` and `/v1/kb/query/image`
  take `user` the same way and merge that user's private upload collection.

The `_USER_SCOPED_TOOLS` override in `src/audrey/tools/dispatch.py` only forces
`user` to the authenticated identity on the **model → tool** path. A direct HTTP
call from any LAN host bypasses it entirely. Result: anyone on the LAN can read
another user's memories, chat history, and private uploaded documents, or write
into their memory.

The KB query routes deliberately have no `require_user` (documented in
`AGENTS.md`) because internal ReAct dispatches don't carry a browser token. So
the correct fix is a **network boundary**, not route-level auth: stop publishing
these ports to the LAN. Container-to-container traffic uses `ollama-net` DNS and
does not depend on the host publish.

## Goal

Remove (or tightly scope) the host-port publishes for `audrey-ai` and
`custom-tools` so the unauthenticated data routes are no longer reachable from
the LAN, **without** disrupting:

- OWUI → Audrey (the live user path)
- Audrey → custom-tools (tool dispatch)
- Prometheus → Audrey `/metrics` (scrape)

## Pre-change facts established (from review)

All confirmed by grep against the repo during the security review:

| Consumer | How it reaches the service | Depends on host publish? |
|---|---|---|
| OWUI → Audrey | `ollama-net` DNS `http://audrey-ai:8000/v1` (deploy docs, `.env.example:77`) | **No** |
| Audrey → custom-tools | `config.yaml:461` `http://custom-tools:8001` | **No** |
| Prometheus → Audrey | `monitoring/config/prometheus.yml:12` target `audrey-ai:8000` (shared net) | **No** |
| Laptop `eval_research.py` | `--base-url http://192.168.1.11:8000/v1` (direct) OR OWUI `:8080/api` | **Yes** (direct mode) |
| Laptop `kb_score_probe.py` | `KB_PROBE_BASE_URL=http://192.168.1.11:8000` (default `audrey-ai:8000`) | **Yes** |

So the **only** legitimate consumers of the published host ports are the
developer's own laptop scripts. Everything on the live serving path uses
container DNS and is unaffected.

## Baseline (exact current state)

`compose.yaml` before the change:

```yaml
  audrey-ai:
    ...
    ports:
      - "8000:8000"     # lines 45-46
  custom-tools:
    ...
    ports:
      - "8001:8001"     # lines 86-87
```

## Plan

1. [x] Create this working doc.
2. [ ] Re-verify the baseline compose port lines and the consumer table above.
3. [ ] Decide port strategy and record rationale (see options below).
4. [ ] Edit `compose.yaml` to remove/scope the publishes.
5. [ ] Handle laptop eval-script access (redirect to OWUI path or dev profile).
6. [ ] Write the verification checklist (box commands for the user, since the
       laptop can't exec against Unraid).

## Strategy options (decision in the log below)

- **A — Remove `ports:` entirely.** Cleanest. Services reachable only over
  `ollama-net`. Laptop scripts must go through OWUI (`:8080/api`) or an SSH
  tunnel. Matches "OWUI is the only public surface" from AGENTS.
- **B — Loopback bind (`127.0.0.1:8000:8000`).** Keeps on-box localhost access
  (e.g. `docker exec`-free curl from the Unraid shell) but removes LAN exposure.
  Laptop still needs a tunnel for direct access.
- **C — Dev-profile port override.** A `profiles: ["debug"]` sidecar or a
  commented `ports:` block that's opt-in. Most flexible, most moving parts.

## Decision: Strategy A — remove `ports:` entirely

Chosen because re-verification confirmed the eval harness's **documented,
default, repeatable** access path already goes through OWUI, not Audrey's
`:8000`:

- `docs/testing/README.md:74` and `scripts/eval_research.py:34` both document
  `AUDREY_EVAL_BASE_URL=http://192.168.1.11:8080/api` (OWUI) as the setup.
- `run_all_evals.sh:14` assumes `192.168.1.11:8080` (OWUI).
- Direct-to-Audrey (`:8000/v1`) is explicitly the *non-repeatable* fallback in
  the eval_research docstring ("JWTs expire, so the OWUI path is the repeatable
  one").
- `kb_score_probe.py:44` defaults to container DNS `http://audrey-ai:8000`; the
  `192.168.1.11:8000` mode is opt-in via `KB_PROBE_BASE_URL`.

So removing the host publishes costs only two **opt-in debug modes** that hit
`:8000` directly, both of which have a supported alternative (OWUI path / SSH
tunnel). Loopback bind (B) was rejected as unnecessary — nothing on the box
needs localhost HTTP to these services (the healthchecks run *inside* each
container against `127.0.0.1`, which is unaffected by the host publish). Dev
profile (C) was rejected as over-engineered for a single developer who already
has the OWUI path.

## Changes made (working tree — not yet deployed)

1. **`compose.yaml`** — removed both `ports:` blocks (`8000:8000` and
   `8001:8001`), replaced with an explanatory comment on each service saying why
   the port is unpublished, which legitimate consumers use ollama-net DNS
   instead, and the SSH-tunnel command for laptop debugging. YAML re-validated:
   both services parse, keep `ollama-net`, and have no `ports` key.
2. **`scripts/eval_research.py`** — updated the docstring note about direct-
   `:8000` access to say it now requires an SSH tunnel (the OWUI `:8080/api`
   default path is unchanged and still works with no extra steps).
3. **`scripts/kb_score_probe.py`** — updated the docstring/usage to drop the
   `http://192.168.1.11:8000` over-the-LAN form (now dead) and point at
   on-box / tunnelled access. Default `http://audrey-ai:8000` (container DNS)
   is unchanged.

No application code changed. No route added/removed. The change is purely the
deployment network boundary + doc honesty.

## Verification checklist (run on the Unraid box after deploy)

The laptop can't exec against Unraid, so these are for the user to run on the
box. Deploy first:

```bash
cd /mnt/user/appdata/audrey_ai_2.0
docker compose up -d --force-recreate audrey-ai custom-tools
```

Then verify — **expect the first two to now FAIL/refuse, the rest to PASS:**

1. **LAN exposure is gone** (run from the laptop or any OTHER LAN host, not the
   box). Both should now hang/refuse instead of returning JSON:
   ```bash
   curl -m 5 http://192.168.1.11:8000/health          # expect: connection refused / timeout
   curl -m 5 http://192.168.1.11:8001/health          # expect: connection refused / timeout
   ```
   Before the change these returned `{"status":"ok"}`. A refusal here is the fix
   working. (The cross-user data routes rode on the same publish, so if `/health`
   is unreachable, so are they.)

2. **In-container healthchecks still green** (the publish removal must not affect
   `127.0.0.1` inside the container):
   ```bash
   docker inspect --format '{{.State.Health.Status}}' audrey-ai      # expect: healthy
   docker inspect --format '{{.State.Health.Status}}' custom-tools   # expect: healthy
   ```

3. **Audrey ↔ custom-tools DNS path intact** (from inside audrey-ai):
   ```bash
   docker exec audrey-ai python -c "import urllib.request as u; print(u.urlopen('http://custom-tools:8001/health', timeout=3).read())"
   # expect: b'{"status":"ok"}'
   ```

4. **OWUI → Audrey (the live user path) still works.** In the OWUI web UI, send
   one message to `audrey_fast` and one to `audrey_research`; both should stream
   a normal answer. (OWUI reaches Audrey at `http://audrey-ai:8000/v1` over
   ollama-net — unaffected by the host publish.)

5. **Prometheus still scraping.** In Grafana / Prometheus, confirm the
   `audrey-ai` target is still `UP` (scrape target `audrey-ai:8000` is on the
   shared network, not the host port).

6. **Eval harness still runs** (from the laptop, over the LAN/VPN — uses the OWUI
   path, so unaffected):
   ```bash
   .venv/bin/python scripts/eval_research.py --only euclid
   # expect: reaches OWUI :8080/api and passes structural checks
   ```

If step 1 still returns JSON after `--force-recreate`, the recreate didn't take
— confirm with `docker port audrey-ai` / `docker port custom-tools` (both should
print nothing). A plain `restart` does NOT drop a published port; the
`--force-recreate` (or a `down`/`up`) is required.

## Rollback

Reverting is a one-file change: restore the two `ports:` blocks in
`compose.yaml` and `docker compose up -d --force-recreate audrey-ai custom-tools`.
No data migration, no schema change, no state to unwind.

## Residual risk after this change

- LAN exposure of the unauthenticated data routes: **closed** (they're no longer
  host-reachable). This does not *fix* the missing auth on those routes — it
  removes the network path to them. A process already on `ollama-net` (a
  compromised sibling container) could still reach them. That's a much smaller
  surface and is acceptable as the boundary fix; deeper auth is a separate,
  larger item.
- Findings #2 (collection-name collision) and #3 (root containers) are
  untouched — separate work items.

## Progress log

- **2026-07-18** — Doc created. Consumer table and baseline captured from the
  review greps.
- **2026-07-18** — Re-verified baseline against the live `compose.yaml` (ports
  at lines 45-46 and 86-87, unchanged from the review). Traced eval-harness auth
  path: default/documented route is OWUI `:8080/api`, direct-`:8000` is an
  opt-in fallback only. **Decided Strategy A** (remove `ports:` entirely).
- **2026-07-18** — Edited `compose.yaml` (removed both publishes + added
  explanatory comments); re-validated YAML. Updated `eval_research.py` and
  `kb_score_probe.py` docstrings to reflect that direct-`:8000` now needs a
  tunnel. Wrote the box-side verification checklist + rollback. **Working-tree
  changes complete; awaiting deploy + on-box verification by the user.**
