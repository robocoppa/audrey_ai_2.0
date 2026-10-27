# Campaign 4 Phase 11 — native file explorer

**Status:** Explorer/shared file creation accepted. Larger-file browsing and native-style previews built and locally verified October 26; uploaded PDF previews and richer generated-file formatting are approved followups.

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

## Generated files

Native ordinary and Private chat can call `create_file` for TXT, Markdown, CSV,
JSON, DOCX and XLSX. Python standard-library renderers, python-docx and openpyxl
run locally in Audrey. Models supply content and safe filenames, with bounded
rows/sheets and input/output size; they cannot choose server paths or run renderer
code. Spreadsheet strings remain literals. The tool supports eight outputs per
reply. Direct model passthrough and existing Hermes adapters are unchanged.

Generated files use the same owner catalog, protected storage, quota, download
and deletion lifecycle as uploads. My files and Project rows label them Created
by Audrey. A generated file joins the current owned Project when space permits;
if that Project changes or fills, the file stays in My files with explicit tool
feedback. Successful outputs receive server-created chat links that survive
reload; reconciling them preserves the next draft and attachments. Identical
filename/content in the same conversation reuses its file. Changed content
creates a separate immutable file; no prior output or upload is overwritten.
A deleted output is not resurrected by a retry; choose a new filename deliberately.

File origin is immutable in the upload catalog and Project relation (application
schema 25, existing relations default to upload). Generated outputs do not enter
RAG or automatic Project grounding, and do not change an active Tax task's source
pins. Explicit reads/attachments label them working outputs, not independent
source evidence; they cannot verify extracted Tax facts. Private generation and
reads use local code and preserve durable privacy flags. Source/sidecar files live
in the configured upload directory, not the evictable Tax export cache. Tax case
workbooks retain their separate revision-bound export contract.

This provides file generation/download. The existing Hermes/Nextcloud/Collabora
office editor integration keeps its current behavior.

Shared creation/download and owned library/Project visibility are accepted
October 26: all three ordinary DOCX/XLSX and Private Markdown outputs worked.

## Browser previews

Generated chat and Project file links have Preview beside Download. My files >
Open uses the same content viewer for generated files. It displays readable DOCX
paragraphs, formatted Markdown, plain text/JSON, and CSV/XLSX tables with numbered
rows and column headings. Text and numeric cells align left; numbers retain
equal-width digits. Multiple worksheets have keyboard-accessible tabs.
The modal fits desktop/mobile screens, closes with Escape/outside click/Close,
restores trigger focus and preserves the unsent chat draft. Download remains
available. Existing uploaded document/image/media viewers retain their contracts.

The backend returns bounded read-only JSON, using saved generated document text
and locally parsed spreadsheet values. It does not evaluate formulas, load remote
workbook links or run model/index calls. Markdown suppresses raw HTML and remote
images. Previews enforce owner/scope checks, file locks, deletion rechecks and
private no-store headers. The same viewer supports saved Tax workbook revisions,
using the existing owner/case authorization, revision cache and expiry behavior.
Nothing in a preview changes case facts or the file itself.

### Larger previews and native-style appearance — current slice

Documents use a continuous paper reading surface, spreadsheets use a light grid
with row/column headings and worksheet tabs, Markdown is rendered, and plain
text/JSON uses a code-style surface. Numeric spreadsheet values stay left aligned.
This approximates familiar file types; exact Word/Excel layout, embedded images,
advanced formatting and interactive editing are not claimed.

Text and DOCX previews have Previous/Next sections with exact character positions.
Each request is bounded to 100,000 characters and, for DOCX, 1,000 paragraphs;
continuation preserves text at character and paragraph boundaries. Spreadsheet
requests show 200 rows, with Previous/Next controls and independent worksheet
positions. Sheet indices and row numbers remain stable. Total rows appear only
when the reader has observed the end, not from untrusted workbook dimensions.

Limits remain eight sheets and 32 columns. Source input is at most 8 MiB, with
bounded ZIP inflation. Cells/columns that cannot be represented have a separate
omission notice; content on later browsable sections is not described as lost.
Source files remain downloadable and unchanged. Paging makes no model/index calls
and repeats owner/scope/deletion checks for every request, including saved Tax
workbook revisions.

### Approved followups

The main campaign queue owns their order. Uploaded PDF browser preview will add
page navigation for existing owned PDFs without changing ingestion or Private
processing. Richer DOCX/XLSX generation will improve formatting in the actual files
and show supported formatting in previews. Native-style surfaces above do not
mean those formatting changes have already shipped. These remain read-only
previews and generated outputs; office editing stays outside Audrey.

### Deployment and acceptance

After normal commit/push, run on Tower in `/mnt/user/appdata/audrey_ai_2.0`:
pull, then `docker compose up -d --build audrey audrey-ui`. No schema/config,
sidecar, model, dependency or index change.

Backend/component regressions, desktop/mobile browser checks, typecheck, build
and changed-file lint pass locally. Browser acceptance is pending. At
`https://ai.builtryte.xyz`, preview an existing long document or worksheet and
use Next then Previous: text/rows retain their order, worksheet tabs retain their
place, and the new presentation stays usable on desktop/mobile. A short existing
file is enough to inspect appearance; paging needs a file beyond a section limit.
Use a current long file where possible; no upload, regeneration, or broad model
suite is required for already accepted creation or Tax save/download behavior.
Download remains available and closing returns to the unsent chat draft. No
separate smoke document is needed.
