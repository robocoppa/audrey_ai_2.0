# Campaign 4 Phase 16 — retired document authoring

**Status:** Closed by owner October 27, 2026. The proposed document authoring
service is retired and remains unimplemented.

Hermes document/sheet operations belong to the existing Bot Tools MCP API and Nextcloud/Collabora workspace at `cloud.builtryte.xyz`. They are not Audrey browser features.

Audrey supports upload, extraction, summaries, search, previews/downloads,
attachments and grounded answers. Shared local document/spreadsheet generation
is implemented in [Phase 11](phase-11-native-file-explorer.md#generated-files).
The proposed interactive authoring panel, document-job/approval routes, template
worker and conversion flows were removed.

Already-deployed schema 20 remains in migration history. Its inert document tables have no attached repository, public routes, or claiming worker. Backup accounting and privacy purge retain coverage so existing databases remain restorable and removable data is not stranded.

Bot workspace changes are maintained and tested in that service's workflow, separately from Audrey Campaign 4.

Saved Tax review XLSX downloads belong to
[Phase 20](phase-20-reference-libraries-and-private-tax-assistance.md#20f--saved-review-files).
Interactive office editing and the retired job/approval APIs remain outside Audrey.
