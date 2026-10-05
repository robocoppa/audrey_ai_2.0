# Campaign 3 Phase 16 - retired Audrey document tools

**Status:** Retired on October 5, 2026 after an architecture review.

## Decision

Audrey will not provide document or spreadsheet creation, editing, conversion,
or approval flows in its browser interface. Those capabilities already exist for
the Hermes bot fleet through the separate Bot Tools MCP service and the
Nextcloud plus Collabora workspace published at `cloud.builtryte.xyz`.

The earlier Phase 16 plan incorrectly adapted a Hermes capability into an Audrey
product feature. It duplicated an existing service, added an unnecessary user
workflow, and crossed a boundary that the project had already established.

## Correct service boundary

```text
Hermes bots -> Bot Tools MCP API -> Nextcloud / Collabora -> cloud.builtryte.xyz

Audrey users -> Audrey chat, Projects, My Files, and read-only file grounding
```

The Bot Tools MCP service owns bot authorization and workspace operations.
Hermes consumes those operations through APIs. Audrey is not in that request
path, and the Audrey interface does not expose bot workspace controls.

Audrey may continue to upload, extract, summarize, search, display, attach, and
ground answers in documents that users provide. That read and analysis scope
does not imply document authoring or spreadsheet editing.

## Retirement result

The following Phase 16 runtime and product work is removed:

- the **Create a document** panel in My Files;
- Project Brief template generation;
- `/api/document-jobs` routes and approval actions;
- the in-process document template worker;
- the bundled DOCX template;
- document-generation browser, route, worker, and smoke tests;
- planned DOCX-to-PDF and spreadsheet-editing slices.

Schema version 20 remains in migration history because it was already deployed.
Its document tables are inert: no repository is attached to the application
store, no route exposes them, and no worker claims them. Backup accounting and
privacy purge support remain so an existing schema 20 database stays restorable
and removable data cannot become stranded.

## Verification

After deployment:

1. Open **My Files** and confirm there is no document creation panel.
2. Confirm normal upload, preview, download, attachment, Project selection, and
   grounded chat behavior still work.
3. Confirm `/api/document-jobs` returns HTTP 404.
4. Confirm Audrey starts against the existing schema 20 production database.

No Hermes workspace smoke belongs in the Audrey Campaign 3 gate. Test additions
to the bot workspace through its own MCP and `cloud.builtryte.xyz` runbooks.
