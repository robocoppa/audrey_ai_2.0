# Campaign 3 — closed

Closed by user direction on **2026-10-27**. The retained documents are compact
contracts for optimization, research, retrieval and media work. Their historical
deployment queues, command receipts and repeated session logs are removed.
Closure is administrative: it neither implements old proposals nor converts an
unperformed or failed quality check into a pass.

| Phase | Contract |
| --- | --- |
| 16 | [Fast-path fallback and acknowledgment](phase-16-fast-path-model-fallback.md) |
| 17 | [Shared deep-panel setup](phase-17-deep-panel-dedup.md) |
| 18 | [Concurrent tool discovery](phase-18-concurrent-tool-discovery.md) |
| 19 | [OpenAI route package](phase-19-split-openai-routes.md) |
| 20 | [Concurrent cloud drafts](phase-20-concurrent-cloud-workers.md) |
| 21 | [Local worker deadlines](phase-21-local-worker-timeout-fix.md) |
| 22 | [Explicit depth intent](phase-22-depth-intent-routing.md) |
| 23 | [Pool-aware synthesis deadline](phase-23-deep-synthesis-timeout.md) |
| 25 | [Research fact checking](phase-25-research-fact-check-stage.md) |
| 26 | [Research claim/source ledger](phase-26-research-claim-ledger.md) |
| 27 | [On-box evaluation runner](phase-27-eval-on-box.md) |
| 28 | [Research grounding boundaries](phase-28-research-grounding-diagnostic.md) |
| 29 | [Readable web-page fetching](phase-29-web-fetch-page-opener.md) |
| 30 | [Web-fetch connection contract](phase-30-web-fetch-ssrf-hardening.md) |
| 31 | [KB query authentication](phase-31-kb-query-auth.md) |
| 32 | [Chunked media transport](phase-32-video-upload-transport.md) |
| 33 | [Leased media jobs](phase-33-video-job-lifecycle.md) |
| 34 | [Isolated media worker](phase-34-media-worker-container.md) |
| 35 | [Timestamped transcripts](phase-35-video-transcript.md) |
| 36 | [Keyframe descriptions](phase-36-video-visual-assessment.md) |
| 37 | [Media summaries](phase-37-video-summary.md) |
| 38 | [Measured video optimization](phase-38-video-optimise.md) |
| 39 | [Dense and lexical retrieval](phase-39-hybrid-retrieval.md) |
| 40 | [Finding and scoping owned files](phase-40-uploads-in-chat.md) |
| 41 | [URL media acquisition](phase-41-url-video-ingest.md) |
| 42 | [Video retrieval role](phase-42-video-specialist.md) |
| 43 | [Unscoped KB coverage proposal](phase-43-kb-result-coverage.md) |

Phase 43's coverage expansion is **closed without implementation**. The optional
corpus-separation and multi-address-fetch ideas are also closed without a planned
implementation. The video role is not certified superior to the general model;
historically unperformed visual-fairness and summary edge-case checks are closed
without new validation. No item here is an active queue.

This numbering continues Campaign 2. There is no standalone Phase 24 document;
the opt-in research design is retained in Phase 26. Later work lives in
[Campaign 4](../campaign-4/README.md). Current run instructions belong in
[`evals/README.md`](../../evals/README.md) and
[the operations references](../reference/live-smoke-testing.md).
