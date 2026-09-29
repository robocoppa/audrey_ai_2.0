# Campaign 3 Phase 4 — file and artifact downloads

**Status:** Slice 4A is live-settled on 2026-09-29. Slice 4B's backend
live smoke passed all three derived artifacts on 2026-09-29; its one native
browser download check remains open.

## Goal

Let a file owner recover Audrey's stored originals and useful derived text from
the native Files surface. Downloads must keep the same owner boundary and
storage-lifecycle semantics as file inspection, without buffering large files
through application memory.

## Product rules

- Resolve every download from the authenticated owner's file listing.
- Return the same 404 for an unknown id and another user's id.
- Return 410 when the owner record remains but its original source was reclaimed
  or is missing.
- Send originals as attachments with their stored filename and media type.
- Mark private downloads `no-store` and `nosniff`.
- Preserve HTTP range support for large originals and media clients.
- Do not advertise an original download after the source was reclaimed or
  before a fetched video has landed.
- Keep derived artifacts separate from originals. Their availability, names,
  and formats come from the artifact contract rather than the upload filename.

## Slice 4A — original files

The native backend exposes:

```text
GET /api/files/{file_id}/download
```

The route reuses the existing owner-bound listing and source-path resolution.
Starlette's file response streams the file and handles byte ranges. It sends
`Content-Disposition: attachment`, the original media type, and the private
response headers.

The native Files dialog shows a **Download** action beside an original that can
still exist. The link is same-origin, carries the stored filename, and is
hidden for reclaimed sources and URL fetches that have not produced a local
file. The endpoint remains authoritative if storage changes after the listing:
it returns 410 instead of an empty or unrelated download.

Laptop coverage proves:

- exact original bytes, media type, Unicode filename disposition, and response
  headers;
- byte-range status, bytes, and `Content-Range`;
- the same 404 response for unknown and foreign ids;
- 410 for reclaimed or missing sources;
- authentication enforcement;
- encoded same-origin URLs and Files-dialog visibility;
- the production frontend build and full backend/frontend suites.

## Targeted live gate

**Result:** Passed on 2026-09-29. Exact bytes, byte ranges, attachment headers,
cross-owner 404, and cleanup all passed against the deployed stack.

Deploy both the backend and native UI, then run the download smoke from the
laptop checkout. It creates one small text upload, verifies a full download and
a ranged download, checks that a second Audrey account receives 404, deletes
the upload, and drains cleanup through the existing repair endpoint.

```bash
cd /home/bart/Documents/github/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python scripts/smoke_file_download.py
)
```

Success is exit code zero and JSON ending in `"status": "passed"`. The download
block must report HTTP 200 and 206, the identity block must report cross-owner
HTTP 404, and cleanup must report `repair_status: "ready"`.

In the native browser, open **Files** and download one existing file whose
original is still stored. The browser should save it under its original name,
and its bytes should match the uploaded file. This one manual click covers the
built UI action; the script covers the backend contract and cleanup.

## Slice 4B — derived artifacts

The native backend exposes:

```text
GET /api/files/{file_id}/artifacts/{artifact}/download
```

Here, `artifact` is `transcript`, `visual`, or `summary`. The route reuses
the exact owner and sidecar resolution used by the paged artifact reader. It
returns 404 when the selected sidecar is missing or empty and 422 for a
non-video file. Original-source reclamation does not affect these downloads.

Download names are derived from the original video basename:

| Artifact | Filename |
|---|---|
| Transcript | `<video>.transcript.txt` |
| Visual descriptions | `<video>.visual-notes.txt` |
| Summary | `<video>.summary.txt` |

The Files viewer shows a download action only after the selected artifact page
reports real content. A legacy row summary without a summary sidecar remains
visible as a fallback, but it does not claim a downloadable artifact.

Laptop coverage proves exact UTF-8 bytes, all three filenames, private response
headers, owner scoping, absent and empty sidecars, authentication, and derived
downloads after the original video was reclaimed. The current full suite passes
2,970 Python tests and 29 frontend tests.

## Slice 4B targeted live gate

**Backend result:** Passed on 2026-09-29 against a fetched video. Summary,
transcript, and visual-notes downloads each returned HTTP 200 with the expected
stable filename. The native browser download action remains the final check.

After deploying the backend and native UI, run the read-only smoke from the
laptop checkout:

```bash
cd /home/bart/Documents/github/audrey_ai_2.0
(
  set -a
  source .env.test.local
  set +a
  AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python scripts/smoke_artifact_download.py
)
```

No upload is needed when the smoke account already has a **Ready** video with
text in at least one **View text** tab. Otherwise, upload or fetch a short video
with clear speech or visible scene changes and wait for it to become Ready.

The script scans Ready videos, preferring ones whose originals were reclaimed,
and selects the first with a real summary, transcript, or visual-notes sidecar.
Set `AUDREY_ARTIFACT_SMOKE_FILE_ID` only when a particular video must be
checked. The script pages each artifact through the existing reader, compares
every available download byte-for-byte, validates its filename and private
headers, and expects HTTP 404 for missing artifacts. It creates or deletes
nothing.

Success is exit code zero, `"status": "passed"`, and
`"available_count"` of at least one. Use the filename from the output's
`"file"` block for the browser check. Under **Files → View text**, a non-empty
tab should show its download action and an empty tab should not. The saved name
must use the suffix in the table.

Phase 4 completes after the remaining native browser download check passes.

During this smoke, the downloaded summary exposed a separate generation-quality
bug: the GLM cloud model repeated its writing assignment instead of describing
the video. Rejecting that text worked, but a live re-upload proved a retry
against the same model also failed and left the video without a summary. The
revised generator uses local qwen3.8 as primary and local ornith-1.5:35b as an
independent fallback. Both passed the production prompt against identical
synthetic video material with thinking disabled. The generator keeps up to
three sentences within an 80-word safety cap. This does not alter the artifact
download contract.
