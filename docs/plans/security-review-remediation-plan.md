# Security Review — Remediation Plan (master)

_Created 2026-07-18. Source: whole-project security review, 2026-07-18._

**Current status (2026-09-01): closed as an active plan.** The controls are
complete and Unraid-verified: Audrey port 8000 is deliberately LAN-published
for authenticated clients after its KB routes gained user/service-token auth;
custom-tools remains unpublished; exact-user collection filtering is enforced;
and the writable services run as non-root UID 99:GID 100. The image-fetch DNS
rebinding gap remains an accepted low-risk residual behind authenticated access.
The original sequence below is retained as historical decision context and is
not the current deploy runbook.

This is the master plan for the security review findings. Each work item has
its own detail doc as it's picked up; this file is the index + sequencing +
risk assessment. Per-item docs:

- Item 1 → [`security-item-1-unpublish-ports.md`](security-item-1-unpublish-ports.md)
  (**CLOSED** — final network and route controls are deployed and verified)

## Threat model context

As deployed, the **public internet only sees Open WebUI** through the cloudflared
tunnel. `audrey-ai` and `custom-tools` are not tunnelled. So none of the findings
below are internet-exploitable — the threat model that matters is **anything on
the home LAN**: another device, a guest, or malware that has pivoted onto a LAN
host. Every finding traces to one root: **the app trusts the network for
isolation, but the network boundary is wider than assumed.**

## Findings

### #1 (High) — Cross-user data access via unauthenticated LAN-published routes

`compose.yaml` publishes `audrey-ai` on host `8000` and `custom-tools` on host
`8001`, both binding `0.0.0.0`. Neither service's data routes are behind
`require_user`:

- custom-tools (`tools-server/app.py`): `/memory_search`, `/memory_recall`,
  `/memory_store`, `/kb_search`, `/chat_history_search` take `user` as a plain
  body field.
- Audrey KB (`src/audrey/routes/kb.py`): `/v1/kb/query`, `/v1/kb/query/image`
  take `user` the same way and merge that user's private upload collection.

The `_USER_SCOPED_TOOLS` override in `src/audrey/tools/dispatch.py` only forces
`user` to the authenticated identity on the **model → tool** path. A direct HTTP
call from any LAN host bypasses it. Result: anyone on the LAN can read another
user's memories, chat history, and private uploaded documents, or write into
their memory.

The KB routes deliberately lack `require_user` (documented in `AGENTS.md`) —
internal ReAct dispatches don't carry a browser token. So the fix is a **network
boundary**, not route auth.

### #2 (Medium) — Per-user collection name collision leaks uploaded documents

`sanitize_user` (`src/audrey/kb/user_store.py`) collapses every non-alphanumeric
run (`.`, `-`, `_`, `@`) to a single `_`, so `a.b@c.com`, `a-b@c.com`, and
`a_b@c.com` all map to the same collection. The per-user search path `_search`
(`src/audrey/kb/qdrant.py`) applies **no `user` payload filter** — it trusts the
collection name for isolation. So colliding accounts read each other's document
chunks, **even through the fully authenticated model path**. If OWUI signup is
open, an attacker can register a colliding address deliberately. Deletes are safe
(double-filtered on `user` in `delete_by_file_id`); reads are not.

### #3 (Medium) — Containers run as root

Neither Dockerfile has a `USER` directive, so both processes run as UID 0. Any
RCE or container escape (e.g. via a document parser on an uploaded file) lands as
root, with `/mnt/user/knowledge` and `/data` bind-mounted in. Base images and
`uv` are already digest-pinned — non-root is the remaining gap.

### #4 (Low) — SSRF DNS-rebinding residual + notes

- `src/audrey/kb/embed.py` resolves the image host once to reject private IPs,
  but httpx re-resolves on connect, so a rebinding host could still be hit.
  Documented as out-of-scope; re-assess after #1 shrinks its exposure.
- Auth token cache (`src/audrey/auth.py`) caches validated tokens 30s → a revoked
  token stays valid up to 30s. Documented, admin `clear` endpoints exist.
  Acceptable, no change.
- `debug.log_incoming_payload_content` logs full user message text. Off by
  default; keep it off in prod.

## Work items & sequencing

| Order | Item | Fix approach | Risk of fix | Value | Verify gate |
|---|---|---|---|---|---|
| 1 | #1 LAN exposure | Remove host `ports:` publishes (ollama-net only) | Low | Highest | OWUI chat + Prometheus still work; LAN curl to :8000/:8001 refused |
| 2 | #2 collection collision | Add raw-`user` payload filter to per-user `_search` branch | Low-med | High | Two-account isolation test (unit + live) |
| 3 | #3 root containers | Add non-root `USER` + `--chown` app/data dirs | Med | Med | Clean boot + upload round-trip + ingest on box |
| 4 | #4 SSRF re-assess | Re-evaluate after #1; connect-to-resolved-IP+SNI if still needed | — | Low | Decide after #1 |

Rationale for order: Item 1 is highest-value / lowest-risk and shrinks the blast
radius of #2 and #4, so it goes first. Items 1 and 2 are independent and could
ship together. Item 3 is the one with real deploy risk (bind-mount file
ownership under the new UID) — sequence it last so a rollback there doesn't hold
up the higher-value fixes. Everything except the bind-mount permission work in #3
is testable hermetically on the laptop first.

## Detail on the fixes (items 2–4, for when they're picked up)

**Item 2:** Thread an optional `user` filter through `QdrantKB._search` and apply
it in the `kb_user_*` branch of `_search_text_merged` / `_search_images_merged`
(`src/audrey/routes/kb.py`). Global-collection search stays unfiltered; only the
per-user branch gains the filter. Additive (`must` clause) → worst case is fewer
results, never a new leak. Chosen over changing `sanitize_user`, which would
orphan every existing collection and force a re-ingest. Extend `test_kb_qdrant.py`
/ `test_kb_query_floor.py` with a two-user-same-collection fixture; cover both
`min_score` floor-on and floor-off runs.

**Item 3:** Add non-root `USER` + `--chown`ed app/data dirs to both Dockerfiles.
Main risk is bind-mount ownership: `/data`, `/mnt/user/knowledge`, and the CLIP
cache (currently `/root/.cache/clip` — it moves) must match the new UID or the
app fails to write at boot. Verify clean boot, an upload round-trips, the watcher
ingests, CLIP weights load from the new cache path. Bind-mount permission work is
on-box only (laptop can't touch Unraid).

**Item 4:** After Item 1, the image route is model-reachable only (no longer
LAN-reachable), which may make the rebinding residual acceptable as-is. If not,
the fix is connect-to-resolved-IP with SNI-hostname (noted out-of-scope in the
code today). Keep `debug.log_incoming_payload_content` off. No change proposed for
the 30s token cache.

## What was already solid (no action)

Parameterized SQL throughout; mime-sniff-gated uploads with streaming size +
per-user byte quotas; image-fetch SSRF guard with `follow_redirects=False` and a
25 MB cap; filename directory-stripping; model-supplied `user` overridden by the
authenticated identity on the pipeline path; passthrough gated by config
allowlist + role; OWUI as sole identity source; `.env` gitignored with no
committed secrets; digest-pinned base images.

## Status log

- **2026-07-18** — Plan created from the security review. Item 1 started (see its
  detail doc); items 2–4 queued.
