# Campaign 4 Phase 11 — native file explorer

**Status:** Explorer accepted. Upload presentation followup built; browser acceptance pending.

## Library model

Chat attachment selection and My Files use compact explorer rows with search and metadata-based folders: All files, Documents, Images, Audio, and Videos. Counts reflect the same owner-scoped catalog. These are virtual filters; custom folders, nesting, and file moves are not implemented.

Chat rows show filename, kind, size, and selection state. Only Ready files can be selected; existing ten-file and image limits remain enforced. Starting a valid chat upload immediately closes the picker; progress, processing status and errors stay visible beside the composer. Rejected file selections leave it open. Arrow, Escape, or outside click also dismisses it.

My Files uses dense rows, a folder pane, search/status/sort controls, processing metadata, and compact Open/Source/Download/Delete actions. Summary stays inside Open instead of enlarging listing rows. Narrow layouts keep folder and action navigation usable.

## Interaction contracts

Uploads in My Files, chat and Projects show Preparing, transfer percentage,
then **Upload complete. Processing your file…** with an indeterminate indicator
while waiting for the server. A transfer cannot display 100% before completion.
Queued processing retains its existing status and Ready polling.

My Files' Add files panel uses a wider video-link form on the left and a compact
device-upload form on the right. Link and file fields share height and styling;
a compact plus button opens device selection beside the filename or No file
chosen. Captions sit outside both fields. Both actions share height and a baseline;
progress/results sit below both forms so they cannot misalign the buttons.
Long links stay inside their field. Narrow screens stack link then upload.
File types and size limits are available in a separate compact disclosure.

Deploy after the normal commit/push/pull on Tower in
`/mnt/user/appdata/audrey_ai_2.0`: `docker compose up -d --build audrey-ui`.
No backend rebuild or data/index change. In the browser, open My Files > Add files,
paste a long video link and check the layout; upload a PDF and check that transfer
completion changes to Processing before Ready. The upload is required for that
status check. Existing file ingestion/reuse acceptance remains closed.

Sources, Models, and Tool calls form one controlled popover group: opening one closes the others; outside click or Escape closes the current popover. Live activity and saved answers follow the same behavior.

Model telemetry is a separate operations phase; it is not another Phase 11 file-explorer slice.
