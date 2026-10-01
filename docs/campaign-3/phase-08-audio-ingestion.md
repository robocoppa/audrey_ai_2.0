# Campaign 3 Phase 8 - audio ingestion

**Status:** Complete. Slice 8A and Slice 8B are live-passed.

## Goal

Let a user upload spoken audio, wait for the existing durable media worker,
read its transcript and summary in Files, search or ask questions about its
contents, and attach it to native chat without Audrey pretending the recording
is a video.

Slice 8A established the path with MP3. Slice 8B adds WAV, M4A, and FLAC after
measuring their real libmagic MIME values and proving each container through
the same ffmpeg extraction boundary.

## Slice 8A - spoken MP3 files

The upload flow is:

1. Audrey applies the existing filename, byte, quota, and libmagic gates.
2. A sniffed audio/mpeg source is stored as kind audio with status pending.
3. The existing media queue leases the file to media-worker.
4. ffprobe finds an audio stream and no video stream. ffmpeg creates the same
   bounded 16 kHz mono WAV used for video speech, Whisper transcribes it, and
   the visual pass is skipped.
5. Audrey writes and indexes the owner-scoped transcript, creates an
   audio-specific two or three sentence summary, and moves the row to ready.
6. The native Files viewer offers Summary and Transcript. It does not offer
   Visual notes for audio.
7. A native chat attachment preserves kind audio. The run receives the
   authenticated filename manifest and uses Audrey's existing file and
   knowledge tools to inspect contents.

This reuses the media queue instead of adding an audio queue. Audio-only input
was already a supported ffmpeg shape inside the worker: has_video is false, so
frame extraction is skipped while speech still follows the normal
transcription path.

## Slice 8B - WAV, M4A, and FLAC

The browser and backend now advertise and accept these measured pairs:

| Suffix | Sniffed MIME |
|---|---|
| `.wav` | `audio/x-wav` |
| `.m4a` | `audio/x-m4a` |
| `.flac` | `audio/flac` |

Real one-second ffmpeg fixtures produced each value through libmagic and then
passed ffprobe plus conversion to the 16 kHz mono WAV consumed by Whisper.
Admission remains fail-closed: a filename extension only controls the browser
hint; the uploaded bytes must sniff as an allowed audio MIME.

No new queue, database migration, Files branch, attachment kind, or summary
prompt is needed. Once admitted, every format is the same first-class `audio`
kind established by Slice 8A.

## Type and persistence boundaries

Audio is a first-class kind across:

- upload classification and native file responses;
- Files filtering, icon, progress text, artifact tabs, and chat presentation;
- canonical message attachment validation and API responses;
- tools-server catalogue and artifact guidance;
- Qdrant-to-SQLite reconciliation.

Application migration 16 rebuilds app_message_attachments with audio in its
kind constraint. Existing attachment rows are copied before the old table is
dropped, its owner index is recreated, and a foreign-key check runs before the
migration commits.

Qdrant artifact points still carry point kind text, as video artifacts do.
Reconciliation now derives the file kind from the artifact MIME: audio/* stays
audio and every other media artifact retains the existing video repair. This
prevents an audio recording from changing to video on the next Audrey restart.

Audio originals are retained. Source reclamation remains
limited to kind video, so processed audio stays downloadable and reprocessable.

## Summary behavior

The existing video prompt remains the default and is unchanged. Audio selects
a separate prompt that describes a recording for a listener, uses transcript
material only, and asks for the same natural two or three sentence library
description the user accepted for video summaries.

Summary generation remains fail-soft. A transcript and its indexed chunks are
useful even when the configured summary model is unavailable, so a missing
summary does not fail the audio row.

## Laptop verification

- MP3 remains admitted as `audio/mpeg`.
- Real WAV, M4A, and FLAC fixtures sniff as `audio/x-wav`, `audio/x-m4a`, and
  `audio/flac` respectively.
- All three new containers pass ffprobe and conversion to Whisper's 16 kHz mono
  WAV input.
- Each suffix enters the durable queue as kind audio, and a claim preserves its
  MIME and source extension.
- Native Files classifies every MIME as Audio and offers Summary and Transcript
  without Visual notes.
- Focused format, queue, decoder, claim, and Files checks pass: 138 tests.
- The full hermetic backend suite passes: 3,061 tests with one existing FastAPI
  deprecation warning.
- Changed-file Ruff, Python compilation, and diff checks pass.
- Frontend code does not change in Slice 8B; its file inputs already consume the
  backend's advertised extension list.

## Live results and Slice 8B manual check

Slice 8A passed in the native browser on 2026-09-30. The user accepted MP3
upload and processing, Summary and Transcript presentation, grounded chat, and
the saved attachment after refresh. The earlier decoder failure remains fixed
by the deployed `av<19` pin and build-time WAV probe.

Slice 8B passed its native-browser gate on 2026-10-01. WAV, M4A,
and FLAC were accepted as Audio and reached Ready. The M4A recording passed the
full Files/chat path: Summary and Transcript presentation, grounded chat, and
attachment persistence after refresh. This closes Phase 8.

## Completion gate

The gate passed: WAV, M4A, and FLAC reached Ready and the M4A Files/chat
flow passed. OGG, Opus, AAC, music analysis, diarization, and speaker labels
remain later work under item 6.
