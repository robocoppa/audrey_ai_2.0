# Campaign 3 Phase 11 — native file explorer

**Status:** Complete and accepted.

## Library model

Chat attachment selection and My Files use compact explorer rows with search and metadata-based folders: All files, Documents, Images, Audio, and Videos. Counts reflect the same owner-scoped catalog. These are virtual filters; custom folders, nesting, and file moves are not implemented.

Chat rows show filename, kind, size, and selection state. Only Ready files can be selected; existing ten-file and image limits remain enforced. Upload is available, and arrow, Escape, or outside click dismisses the picker.

My Files uses dense rows, a folder pane, search/status/sort controls, processing metadata, and compact Open/Source/Download/Delete actions. Summary stays inside Open instead of enlarging listing rows. Narrow layouts keep folder and action navigation usable.

## Interaction contracts

Single-request uploads show Preparing, measured browser transfer percentage, then Finishing while waiting for the server. Chunked uploads retain part-based progress.

Sources, Models, and Tool calls form one controlled popover group: opening one closes the others; outside click or Escape closes the current popover. Live activity and saved answers follow the same behavior.

Model telemetry is a separate operations phase; it is not another Phase 11 file-explorer slice.
