# Campaign 4 — closed

**Closed by owner October 27, 2026.** All campaign work, browser acceptance
followups and implementation queues are closed. This is a scope decision;
unreported checks are not retroactively recorded as passes. Closure does not
implement retired or deferred proposals. New work starts only with a new user request.

Each phase retains its implemented behavior, security boundaries and material
limitations. Model measurements remain in [MODEL-FACTS](../../evals/MODEL-FACTS.md).
Operational guidance lives in [box operations](../reference/box-operations.md)
and [live smoke testing](../reference/live-smoke-testing.md); this campaign is
no longer a deployment or test handoff.

## Phases

| Phase | Retained contract | Closure |
|---|---|---|
| 01 | [Platform hardening](phase-01-platform-hardening.md) | Closed |
| 02 | [Native application](phase-02-native-application.md) | Closed |
| 03 | [Skills](phase-03-skills.md) | Closed |
| 04 | [File downloads](phase-04-file-downloads.md) | Closed |
| 05 | [Responses foundation](phase-05-responses-api.md) | Closed |
| 06 | [Routing and embeddings](phase-06-system-one-routing.md) | Closed; retain existing model split |
| 07 | [Scanned PDFs](phase-07-scanned-pdf-ocr.md) | Closed |
| 08 | [Audio ingestion](phase-08-audio-ingestion.md) | Closed |
| 09 | [Broader audio](phase-09-broader-audio.md) | Closed |
| 10 | [Answer provenance](phase-10-ordinary-answer-provenance.md) | Closed |
| 11 | [File explorer, creation and previews](phase-11-native-file-explorer.md) | Closed |
| 12 | [Navigation](phase-12-sidebar-navigation.md) | Closed |
| 13 | [Responses extensions](phase-13-responses-multimodal-input.md) | Closed |
| 14 | [Bots and tokens](phase-14-bot-accounts-and-token-lifetimes.md) | Closed |
| 15 | [Composer and Projects](phase-15-composer-and-projects.md) | Closed |
| 16 | [Document tooling boundary](phase-16-native-document-tools.md) | Closed; authoring service retired |
| 17 | [Operations and API](phase-17-operations-and-api.md) | Closed; unbuilt bot work excluded |
| 18 | [Admin and project presentation](phase-18-admin-and-project-presentation.md) | Closed |
| 19 | [Calculation and evidence tools](phase-19-evidence-tools.md) | Closed |
| 20 | [References and Private Tax review](phase-20-reference-libraries-and-private-tax-assistance.md) | Closed with accuracy and coverage limits |

## Standing decisions

- English user input only. Native automatic skill selection uses deterministic
  file intent and the shipped document/video bundles; no model classifier is
  activated by historical selector studies.
- Tax review supports selected-source review and conditional arithmetic. It
  does not determine deduction/credit eligibility, certify a complete return
  or file one. Broad comparison coverage remains incomplete; closure does not
  change the observed model failures or the disclosed reference-library gap.
- Private means verified local server processing with durable reuse restrictions.
  Browser transport, server storage and administrator access retain their stated
  boundaries; Private does not promise an offline browser connection.
- Downloaded video originals are retained and count against file quota. A new
  retention policy cannot restore already-reclaimed source bytes.
- Shared file creation belongs to Audrey. Interactive office editing remains in
  the existing Hermes Bot Tools MCP/Nextcloud/Collabora workspace.
- Hermes keeps its working Chat Completions adapter and owns tool execution and
  approvals. Further bot integration, tool-contract expansion, background jobs,
  webhooks and protocol migration are outside this closed campaign.
- Saved Responses reduce caller payload and history bookkeeping; full retained
  context still reaches the model. No token or billing savings are implied.
- The existing historical-import CLI stays dormant and explicit-request-only.
  Alternate authentication/network routes, broader Private
  OCR/media, state tax adapters, fine-tuning and model experiments remain outside
  implemented scope. Their old proposals are not an active backlog.
