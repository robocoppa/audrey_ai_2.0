# Campaign 3 Phase 4 — file and artifact downloads

**Status:** Slice 4A is laptop-complete on 2026-09-29. Deployment and the
targeted live smoke remain open. Later slices will add downloads for derived
transcripts, visual descriptions, and summaries.

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
original is still stored. The browser should save it under its original name, and its bytes should
match the uploaded file. This one manual click covers the built UI action; the
script covers the backend contract and cleanup.

## Slice 4B — derived artifacts

Add explicit download representations for available transcript, visual, and
summary text. Define stable filenames and plain-text formats, expose only
artifacts that exist for the owned file, and keep original-source reclamation
independent from derived-artifact availability.

The 4B live gate should use one already-processed video and verify only the
artifact types it actually owns. It should not repeat 4A's original-byte,
range, or cross-owner proof unless 4B changes the shared authorization path.
