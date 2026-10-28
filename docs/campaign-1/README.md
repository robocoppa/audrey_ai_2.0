# Campaign 1 — closed

Closed by user direction on **2026-10-27**. This campaign established the
orchestrator, knowledge base, memory, file ownership, scheduling and monitoring.
The old deployment and smoke handoffs have been removed; their lasting decisions
are recorded below. Closure adds no new claim about deployment or live acceptance.

The native application and identity system introduced in [Campaign 4](../campaign-4/README.md)
supersede this campaign's external UI, authentication proxy and upload shell.
Use [the current operations references](../reference/live-smoke-testing.md) for
deployment and testing, and [the lessons](../lesson-ai/README.md) for the current code.

## Retained decisions

| Phase | Delivered behavior or decision |
| --- | --- |
| 1 | Ollama uses persistent model/auth storage and joins the application network. Container recreation must preserve those bind mounts. [Ollama guide](../guides/unraid-ollama.md). |
| 2 | `custom-tools` exposes tools through OpenAPI; Audrey discovers their schemas rather than duplicating them. |
| 3 | Qdrant persists semantic indexes separately from the application container. |
| 4 | The FastAPI orchestrator exposes an OpenAI-compatible chat surface over Ollama. |
| 5 | Model registry priorities, health tracking, task classification and complexity routing select the fast path. |
| 6 | Deep mode gathers independent drafts, synthesizes them, and applies a bounded reflection step. |
| 7 | Fast-path ReAct dispatches registered tools within explicit round and result budgets. |
| 8 | Text/image ingest, CLI, watcher and query routes share the KB infrastructure. Dataset paths inside containers use the mounted path. |
| 9 | Deep workers can run ReAct; local model calls contend through the same GPU gate. |
| 11 | Best-effort memory recall precedes classification and supplies small, user-scoped context. |
| 12 | Semantic memory moved to Qdrant. Exact-key replacement is deterministic per user; unscoped writes are refused. |
| 13 | Uploaded content belongs to the authenticated user. Listing, deletion and retrieval must enforce that ownership server-side. |
| 14 | Replaced user-typed identity with authenticated identity. Its external authentication proxy was later retired. |
| 15 | SQLite indexes upload metadata and quota usage; Qdrant holds searchable content. Keep metadata/content failure paths consistent. |
| 16 | Administrative authentication-cache controls were introduced; their original proxy-specific implementation was later superseded. |
| 17 | Prometheus measures pipeline outcomes, model calls, dispatch, gate wait and retrieval with bounded labels, never per-user identifiers. |
| 18 | Streaming progress names the active pipeline stage while work runs. |
| 18a | A request receives fresh ISO-8601 datetime context before memory recall; streaming and graph entry points must both include it. |
| 19 | Synthesizer output streams after its first token, and the synthesizing progress stage ends at that boundary. |
| 20 | Per-user in-flight limits and fair local-model scheduling prevent one user from monopolizing the GPU. |
| 21 | Docker installs declared project dependencies from `pyproject.toml`, avoiding a second dependency list. |
| 22 | Per-tool metrics, alerts and targeted administrative cache controls made dispatch and access failures observable. |
| 23 | Gate local model calls individually within ReAct; release the GPU while tools run. Round-robin scheduling must remain fair at queue boundaries. |
| 24 | Prometheus/Grafana configuration moved into the repository; persistent monitoring data stays outside source. |
| 24a | Runtime storage was renamed to `appdata/runtime`; the one-time host migration is closed. |
| 24b | Separate dependency installation from package-source installation so source edits reuse the dependency cache. |
| 25 | Synthesis treats prior banners as presentation markup and reflection respects explicit brevity cues. |
| 26 | Protected chat, ingest and tool-administration entry points; user-scoped memory reads cannot trust a model-written user. |
| 27 | Remote image fetching rejects unsafe destinations/redirects; watcher delete/move events remove stale vectors before replacement ingest. |
| 28 | Streaming tool footers derive from actual dispatch records, including per-worker outcomes. |
| 29 | Established hermetic tests that require no live model, Qdrant or tool server. Historical test counts are intentionally omitted. |
| 30 | Reconcile periodically removes orphaned global KB points. Per-user upload collections are managed separately. |
| 31 | Pin upstream container images to tested digests; locally built image tags are separate from upstream pins. |
| 32 | Explicit environment values override YAML; unset variables must preserve YAML lists. Configuration is validated at startup. |
| 33 | Model locations validate at startup. Malformed successful Ollama responses become `OllamaError` so health tracking and fallback handle them. |

There was no standalone Phase 10 document. Historical phase numbers remain useful
in commit history; they are not a work queue. Later campaigns refine these contracts.
