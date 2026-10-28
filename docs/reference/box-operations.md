# Audrey operations reference

Operational reference for the native application. Campaigns are closed; this
guide describes reusable operations, not a deployment or acceptance queue.
The user operates Tower. Agents prepare commands from the laptop and do not
infer remote results or perform write-side Docker/Unraid operations.

## 1. Checkouts and entry points

- Laptop: /home/bart/Documents/github/audrey/audrey_ai_2.0.
- Tower: /mnt/user/appdata/audrey_ai_2.0 on Unraid, LAN 192.168.1.11.
- Browser: https://ai.builtryte.xyz through Cloudflare Access and audrey-ui.
- Laptop API route: http://192.168.1.11:8000 over LAN/WARP. Tailscale reachability
  has not been established as working; do not substitute its address by assumption.
- Current service wiring is compose.yaml; host-specific overrides are in .env.
  Never hardcode Docker-assigned container IPs or infer free host ports.

## 2. Runtime tools

Tower is Docker-only; do not assume a host Python, uv or repository .venv.
Use the owning application's /opt/venv/bin/python inside its image. Do not assume
every image has curl or the same interpreter path. For exact runner/auth commands,
read [live testing](live-smoke-testing.md) and the [testing protocol](testing-protocol.md).

## 3. Service boundaries

| Service | Entry point / network | Responsibility |
|---|---|---|
| audrey-ui | Tower loopback 8090 to container 8080 by default | Static browser app and same-origin API proxy |
| audrey | Host 8000 to container 8000 | API, orchestration and canonical application state |
| custom-tools | Internal custom-tools:8001 on ollama-net; no host publish | This repo's tools-server/, including search and memory |
| media-worker | Internal media-net with no Internet egress | Claimed local media processing |
| media-fetcher | fetch-net with download egress | URL downloads and job handoff |
| Ollama / Qdrant | Managed outside root Compose | Models / vector collections |
| Prometheus / Grafana | monitoring/compose.yaml | Metrics / dashboards |
| cloudflared | Managed outside root Compose | Public tunnel to the native UI |

Container-name DNS resolves inside its Docker network, not from Tower's host shell.
media-net is internal; fetch-net needs egress. Separate network names alone do not
establish isolation. The UI's upstream is explicit and uses request-time Docker
DNS so backend recreation does not leave a stale address.

## 4. Request and data ownership

Browser → Cloudflare Access/tunnel → audrey-ui → audrey → models, Qdrant and
custom-tools. The backend is API-only; do not point the browser hostname there.
custom-tools is tools-server/ in this repository, not a separate codebase.
Brave/SearXNG behavior belongs to that service.

Canonical application/upload records and originals remain authoritative over
search projections. Preserve existing owner/storage namespaces. Private requests
retain verified local processing and tool boundaries.

Hermes document/sheet work belongs to its separate Bot Tools MCP/Nextcloud/Collabora
stack. Audrey does not proxy or approve those workspace actions.

## 5. Rebuilds and runner selection

Use the changed-files table in [live testing](live-smoke-testing.md#service-rebuilds).
Rebuild only affected services; recreate services after bind-mounted config or
environment changes. Root Compose owns application services; monitoring is separate.
Harness/case edits in the mounted evals/ directory do not by themselves require an
eval-image rebuild. Avoid indiscriminate image pruning: that image is build-only.

## 6. Keep Tower config identical to the repository

config.yaml is canonical. Edit it in the laptop checkout and use the user's normal
commit/push/pull workflow. Do not recommend in-place Tower edits or a separate
production copy. Secrets and host-specific values stay in .env. Config changes
require recreation of affected services. A deployment conflict is diagnosed from
the actual diff; historical one-time restoration commands are not routine steps.

## 7. Evaluations

[evals/README.md](../../evals/README.md) owns protocols, cases, auth and artifacts;
[eval-runbook](../guides/eval-runbook.md) owns measurement interpretation.
Hermetic tests prove plumbing; quality assessment reads actual retained answers.
Long runs follow the shared notification/report protocol. Keep reports accessible
from the laptop; a Tower-only path is not a report handoff.

## 8. Diagnosis

Inspect actual response bodies, tool traces, source availability, task/provider
usage and stop reasons before attributing a failure. Tool HTTP success does not
guarantee search hits. A removed original explains absent playback even when its
summary/transcript survives. Keep credentials and client content out of shared output.

## 9. Recovery boundaries

Use canonical backups and supported migration/restore tools. Search rebuilds do
not restore missing source bytes or saved memories. Preserve current deletion
tombstones and account purge cutoffs across recovery. A restore is an explicit
operator action, not an automatic consequence of a failed probe.

## 10. Memory backups and isolated restore verification

`custom-tools` takes a snapshot of its configured memory collection (normally
`kb_memory`) in a background task at startup, then every 24 hours. Chat and tool
startup do not wait for it. Failures retry after five minutes. Each completed
bundle contains the snapshot, an online SQLite copy of `chat_archive.db` (including
account purge cutoffs), and a manifest with SHA-256 checksums, sizes, collection,
Qdrant version, and creation time. Files are mode 600; bundle directories are 700.

Bundles live at `/app/data/backups/memory` inside `custom-tools`, bound to
`/mnt/user/appdata/custom-tools/backups/memory` on Tower. This is outside Qdrant's
volume. Seven completed bundles are retained by default. Include this private
directory in the normal host backup and off-host copy; these snapshots alone do
not protect against loss of Tower. The application database has its separate
`audrey-admin backup-app-state` command.

The `.env.example` lists `MEMORY_BACKUP_ENABLED`, `MEMORY_BACKUP_INTERVAL_S`,
`MEMORY_BACKUP_RETRY_S`, `MEMORY_BACKUP_RETAIN`, and `MEMORY_BACKUP_TIMEOUT_S`.
The default timeout is two minutes per backup/verification, with a short cleanup
budget afterward. No `.env` change is needed to enable the defaults.

To verify a completed backup, run this short foreground check **on Tower**
from /mnt/user/appdata/audrey_ai_2.0. It requires
Qdrant to be reachable, but no Audrey token, upload, or model generation:

```bash
docker compose exec -T custom-tools /opt/venv/bin/python /app/memory_backup.py verify --latest
```

Success: exit 0 and JSON with `status: passed`,
`production_collection_unchanged: true`, `disposable_collection_deleted: true`,
`archive_integrity: ok`, and equal `restored_count` / `owner_payloads_checked`.
It checks local checksums and archive integrity, uploads the snapshot to a new
random `audrey_memory_verify_...` collection, checks every restored point's
owner/key/id and count, then deletes that collection. It cannot accept a
production restore target. Output contains counts and paths, never memory text
or owner identities. A zero count is valid when no memories have been saved.

If a backup is still running, the shared lock reports that instead of starting
another write; retry when it finishes. If no completed bundle exists, inspect
`custom-tools` logs for `memory backup` before retrying. A manual backup can be
made on Tower with:

```bash
docker compose exec -T custom-tools /opt/venv/bin/python /app/memory_backup.py backup
```

It prints `status: ok` and its manifest path when complete; then the verification
command above checks that bundle. Keep bundle contents private; only the compact
check output is suitable for pasting or sharing.

### Recovering production memory later

The verifier deliberately does not restore production. A real recovery is an
operator operation performed with memory writes stopped and a reviewed backup.
Restoring rolls back additions and corrections after the snapshot, and can bring
back memories deleted since it was taken. Individual memory deletions have no
separate tombstone log; reconcile those deletions before reopening access.

Preserve the **current** archive SQLite database and its purge cutoffs. Do not
replace it with the older bundled archive over a surviving current database.
After restoring memory, reapply every current account-purge cutoff to Qdrant
before serving requests, preserving memories updated after that cutoff. The
ordinary maintainer retries only unfinished purges, so a restore needs this
explicit step even for purges previously marked complete. The bundled archive
copy provides receipts if the current archive has also been lost; it cannot
recover deletion requests made after the bundle.

Qdrant snapshots support the same minor server version (at least the source
patch) or the next minor version. The verifier enforces that bound and uses
`priority=snapshot` when recovering its empty disposable collection. A manifest's
live point count is labelled as an observation before snapshot creation; it is
not an atomic snapshot count because user writes keep running. Verification
compares the restored collection's exact count with its complete payload scroll.
See [Qdrant snapshot operations](https://qdrant.tech/documentation/operations/snapshots/)
and the [snapshot upload API](https://api.qdrant.tech/v-1-18-x/api-reference/snapshots/recover-from-uploaded-snapshot).
