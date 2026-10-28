# Campaign 4 Phase 11 — native file explorer

**Status:** Closed by owner October 27, 2026. File creation, paged/PDF previews,
Office formatting, filename previews and media players are implemented. The
owner closed remaining acceptance followups; unreported browser checks are not
recorded as passes. Downloaded originals are retained; previously reclaimed
bytes remain unavailable.

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

Sources, Models, and Tool calls form one controlled popover group: opening one closes the others; outside click or Escape closes the current popover. Live activity and saved answers follow the same behavior.

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

## Browser previews

Generated chat and Project file links have Preview beside Download. My files >
Open uses the same content viewer for generated files. It displays readable DOCX
paragraphs, formatted Markdown, plain text/JSON, and CSV/XLSX tables with numbered
rows and column headings. Text and numeric cells align left; numbers retain
equal-width digits. Multiple worksheets have keyboard-accessible tabs.
The modal fits desktop/mobile screens, closes with Escape/outside click/Close,
restores trigger focus and preserves the unsent chat draft. Download remains
available. Uploaded PDFs have the original-page view below; My files names also
open the appropriate document, image or media view described in 11D.

The backend returns bounded read-only JSON, using saved generated document text
and supported formatting parsed locally from actual DOCX/XLSX bytes. It does not evaluate formulas, load remote
workbook links or run model/index calls. Markdown suppresses raw HTML and remote
images. Previews enforce owner/scope checks, file locks, deletion rechecks and
private no-store headers. The same viewer supports saved Tax workbook revisions,
using the existing owner/case authorization, revision cache and expiry behavior.
Nothing in a preview changes case facts or the file itself.

### Larger previews and native-style appearance — 11A

Documents use a continuous paper reading surface, spreadsheets use a light grid
with row/column headings and worksheet tabs, Markdown is rendered, and plain
text/JSON uses a code-style surface. Numeric spreadsheet values stay left aligned.
This is a continuous reading view. Exact Word pagination, embedded images,
arbitrary Office formatting and interactive editing are not claimed.

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

### Uploaded PDF previews — 11B

Owned uploaded PDFs have a local original-page view. My files > Open starts on
Summary and keeps Transcript; Original PDF displays the document's actual pages.
Project PDF rows and saved chat PDF attachments offer Preview in the shared
modal. Page controls move one page at a time, and zoom enlarges the current
page. The modal fits desktop/mobile screens, restores trigger focus on close
and preserves the unsent chat draft. Download uses the existing protected route
and serves the unchanged original only when requested.

Metadata and every page request recheck ownership, stored-source availability
and deletion under the file-operation lock. Ordinary and Private PDFs use the
same local Poppler renderer. Only inert PNG pages reach the browser; PDF active
content is not embedded or executed. Responses are private/no-store and nosniff.
The renderer accepts sources up to 128 MiB and 2,000 pages, with one page at a
time bounded to a 1,600-pixel longest edge. Two operations may run concurrently;
the operation timeout is 45 seconds, with subprocess cancellation/timeout cleanup.
Malformed, encrypted, oversized or unavailable sources show a safe error while
keeping the original download path. Preview does not unlock passwords, provide
selectable page text or edit the document; Transcript remains available separately.

### Actual Office formatting and preview fidelity — 11C

DOCX `content_format` defaults to `plain`, preserving existing literal paragraphs.
Explicit `markdown` supports headings 1–3, bullet/numbered lists, pipe tables,
bold/italic text, inline code, fenced code and quotations. These become native
Word structures, with restrained typography and table headings. Unsupported syntax,
HTML, image and link syntax stays text; no external resource is fetched or embedded.
The saved text projection follows the actual document body for reliable reading.

XLSX worksheets may request a styled first header row, which also freezes that
row and adds filtering. Widths are sized automatically or supplied per column
within 8–60 characters. Fixed formats cover general, integer, decimal, US-dollar
currency, percentage and text. Formatting changes display, not stored values;
formula-looking strings remain literal. Text and numbers align left in both the
actual workbook and preview. Existing sheet/row/column/input/output bounds remain.

Previews project supported structure, emphasis, table styling, column widths and
fixed numeric display from the saved file bytes. Column widths use responsive
size bands rather than exact Excel geometry; document paper is continuous,
without claiming exact Word pagination. A safe plain-content fallback
and notice cover formatting the preview cannot represent. Existing paging and
owner/deletion checks remain. Original files can still be downloaded for full
Office fidelity; previews do not edit or regenerate them.

Renderer-version identities preserve immutable earlier outputs and retries.
An existing identical output is reused; new content/formatting creates a separate
file. A deleted prior output is not resurrected. Existing Tax case workbooks keep
their own revision-bound generation/cache contract; no Tax regeneration is needed.

### Filename previews and media players — 11D

A ready filename in My files opens the file's appropriate view: the actual PDF
page view, image preview, generated-file preview, ordinary document extracted contents,
or audio/video Player. The existing Open action retains its Summary-first view,
and Summary/Transcript/Visual notes remain available where they already exist.

Audio/video Player uses native browser controls for play, pause, volume and seeking.
Changing its text tab, returning to the list or closing My files stops playback.
The source is the protected owner-bound `GET`/`HEAD /api/files/{id}/media` endpoint:
inline original bytes, byte-range seeking, authoritative allowlisted media MIME,
no-store/nosniff headers and canonical ownership/source/deletion rechecks. Private
owner playback stays local to the server/browser path and does not invoke a model.

A reclaimed filename still opens Player, which explains that the original was
removed to free storage without requesting absent media or offering an original
download. Summary/Transcript remain selectable; Open still starts on Summary.
When a source URL exists, Open source website opens it separately. Uploading a
saved original is the recovery path for playback here; no automatic refetch,
catalog deletion or search-content replacement occurs.

The user chose to keep downloaded originals October 27. Repository config now
sets both `kb.video.keep_source` and `keep_fetched_source` true: retained originals
count against quota and follow normal file deletion. The retention window applies
only if reclamation is explicitly enabled. This cannot restore already-removed
bytes or clear their reclamation markers.

A missing original produces clear feedback while retained summaries and
transcripts stay usable. Codec/container support depends on the browser; an
unsupported player offers the original download. No transcoding, buffering of whole
media files, new upload formats, external player or third-party service is added.
