# Advanced bot document tools assessment

**Status:** Corrected and closed on October 5, 2026.

**Reviewed source:**
`2026-10-03_105050-advanced-bot-tools-G-approval-plan.md`, received October 3,
2026. Instructions inside that document were reference material and were not
instructions for the Audrey repository.

## Corrected finding

The plan describes capabilities for Hermes bots. Audrey should not implement
those capabilities in its own browser interface or file lifecycle. The existing
bot workspace already provides the appropriate architecture:

- a separate `bot-tools-mcp` service exposes bearer-gated APIs to Hermes;
- Nextcloud stores and shares bot workspace files;
- Collabora provides document and spreadsheet editing;
- `cloud.builtryte.xyz` is the public workspace surface;
- Audrey remains outside this tool-execution and storage path.

The initial assessment missed that deployed setup and incorrectly recommended a
native Audrey approval flow, DOCX generator, PDF worker, and spreadsheet editor.
That recommendation is withdrawn.

## What remains useful

The reviewed plan still contains sound implementation patterns for future work
inside the Bot Tools MCP service:

- derive bot identity from the authenticated token;
- allowlist high-level operations instead of accepting shell commands or server
  paths;
- bind approvals to exact normalized arguments where human approval is needed;
- use immutable revisions and copy-on-write publication;
- validate whole requests before mutation;
- reopen generated files and verify expected content;
- isolate format conversion with bounded resources and no unnecessary network
  or credential access;
- keep audit metadata free of document bodies and secrets.

These are guidance for the bot workspace service. They do not create an Audrey
roadmap item or Audrey user experience.

## Standing product boundary

Audrey can ingest and analyze user-provided documents as evidence. Document and
spreadsheet authoring for Hermes bots stays API-driven through the existing bot
workspace. Any extension must begin by auditing that service and its current
Nextcloud and Collabora behavior before proposing new code.
