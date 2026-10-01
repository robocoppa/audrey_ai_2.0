# Campaign 3 Phase 11 - native file explorer

**Status:** In progress. Slice 11A is implemented and awaits its frontend build
and native browser gate.

## Goal

Make selecting and managing a growing private file library fast enough for
normal use. Replace large repeated file cards with dense explorer rows while
preserving Audrey's owner-bound file actions, processing status, search, and
accessibility contracts.

## Folder model

The current file API stores owner, filename, kind, timestamps, status, and
artifacts; it has no durable user-created directory path. Slice 11A therefore
uses honest virtual folders based on existing metadata:

- All files
- Documents
- Images
- Audio
- Videos

Each folder displays a count and filters the same owner-scoped listing. This
gives the browsing density and familiar navigation of an explorer without
creating folder state that the server cannot persist. Custom named folders,
move operations, and nested paths require a later storage/API slice.

## Slice 11A

### Chat attachment explorer

- The paperclip opens a compact single-column file list rather than large card
  blocks.
- Search and the five type folders narrow ready files in place.
- Each row shows a small type mark, filename, kind, size, and selection state.
- The existing ten-file and per-turn image limits still disable unavailable
  choices.
- Upload remains available in the picker.
- The arrow, Escape, and an outside click still close the picker.

### Files explorer

- A left Library pane provides the five virtual folders with live counts.
- Search, status, and sorting stay in a compact toolbar.
- Files render as dense rows with name, status, type, size, indexed chunk count,
  upload time, and compact Source, Download, Open, and Delete actions.
- Summary text leaves the listing rows and remains available through Open,
  keeping the list scannable.
- Upload and video-fetch controls live in a collapsed Add files disclosure.
- On narrow screens the folder pane becomes a horizontal folder strip and rows
  retain their existing mobile action layout.

### Upload and run-detail interactions

- Single-request uploads now use browser upload byte events. Chat reports
  Preparing, a measured percentage, then Finishing while waiting for Audrey's
  response. Chunked uploads retain their part-based progress.
- Models and Tool calls use controlled expandable popovers. Opening either one
  closes its peer; an outside click or Escape closes the open popover. This
  applies to both live activity and saved answers after refresh.

## Automated contracts

- The single-request upload test drives a measured half-upload event and
  requires progress `0`, `0.5`, `1` plus same-origin credentials and JSON error
  negotiation.
- Existing native attachment upload, selection limit, dismissal, Files CRUD,
  filtering, sorting, artifact viewing, and mobile tests are retained.
- Browser contracts now use the explorer folders, collapsed Add files panel,
  compact row metadata, and Open action.
- Live and saved Models/Tool calls contracts require mutual exclusion and
  outside-click dismissal.

The laptop has the installed frontend dependency tree but no Node runtime, so
the TypeScript, Vitest, build, and Playwright commands must run in the normal
`audrey-ui` build environment.

**Backend result, 2026-10-01:** Passed. The full hermetic backend suite passed
3,098 tests. The diff check is clean. No Python file changed in this slice, so
there is no changed-file Ruff target.

## Native browser gate

1. Open a conversation with several ready files and click the paperclip.
   Confirm files appear as compact rows and substantially more fit on screen.
2. Search for a filename, then switch among Documents, Images, Audio, and
   Videos. Confirm counts and visible rows match. Select and remove one file.
3. Close the picker with its arrow. Reopen it and close it by clicking elsewhere
   in the conversation.
4. Upload a file from chat. Confirm the status moves through Preparing,
   percentage updates when transfer time permits, and Finishing. Confirm the
   ready file attaches exactly once.
5. Open Files. Confirm Add files starts collapsed, the Library folders filter
   the dense rows, and search, status, sorting, Open, Download, and Delete remain
   usable.
6. Complete a tool-backed answer that shows Models and Tool calls. Open Models,
   then Tool calls; Models must close. Click elsewhere; Tool calls must close.
7. Hard refresh and repeat step 6 on the saved answer.

## Completion gate

Slice 11A closes when the frontend build and automated UI suites pass and the
seven browser checks above pass on the deployed native UI.
