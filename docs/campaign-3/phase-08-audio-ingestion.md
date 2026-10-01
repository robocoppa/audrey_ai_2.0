# Campaign 3 Phase 8 - MP3 audio ingestion

**Status:** Slice 8A is laptop-complete and awaits deployment plus its manual native-browser check.

## Goal

Let a user upload a spoken MP3, wait for the existing durable media worker,
read its transcript and summary in Files, search or ask questions about its
contents, and attach it to native chat without Audrey pretending the recording
is a video.

Slice 8A accepts audio/mpeg with the .mp3 suffix. Other audio containers remain
outside this slice until their real libmagic and ffmpeg behavior is measured
and pinned.

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
prevents an MP3 from changing to video on the next Audrey restart.

Audio originals are retained in this first slice. Source reclamation remains
limited to kind video, so a processed MP3 stays downloadable and reprocessable.

## Summary behavior

The existing video prompt remains the default and is unchanged. Audio selects
a separate prompt that describes a recording for a listener, uses transcript
material only, and asks for the same natural two or three sentence library
description the user accepted for video summaries.

Summary generation remains fail-soft. A transcript and its indexed chunks are
useful even when the configured summary model is unavailable, so a missing
summary does not fail the audio row.

## Laptop verification

- MP3 is the only new advertised suffix and maps to audio/mpeg.
- A sniffed MP3 enters the durable queue as kind audio and pending.
- A job claim preserves the audio kind, MIME, MP3 path, and empty fetched-video
  caption handoff.
- Transcript and summary readers accept audio; native visual-artifact requests
  reject it.
- Audio summaries use recording and listener language without visual material.
- Reconciliation preserves audio across restarts.
- Canonical fresh and upgraded databases preserve old attachments and accept a
  new audio attachment under schema 16.
- The generated smoke fixture is a real MP3 with audio and no video stream.
- Focused backend checks pass: 259 tests.
- The full hermetic backend suite passes: 3,043 tests with one existing FastAPI
  deprecation warning.
- Changed-file Ruff, Python compilation, and diff checks pass.
- Frontend packages are present, but this shell has no Node or npm executable,
  so the Vitest and TypeScript build commands cannot run here. The deployment
  build is the remaining compile proof for the UI changes.

## Decoder compatibility repair

The first synthetic live attempt reached the worker but failed before Whisper
could read its WAV: faster-whisper 1.1.1 passed `metadata_errors` to `av.open`,
while a fresh dependency resolution had installed PyAV 19 after that argument
was removed. The worker now pins `av<19`. Its image build also decodes a real
16 kHz mono WAV before baking model weights, so the same incompatibility fails
the build instead of the first queued recording.

## Deploy and manual live check

Rebuild `audrey`, `audrey-ui`, `custom-tools`, and `media-worker`. The worker
rebuild is required for the PyAV pin and decoder build gate. It can take longer
than the application build because it bakes the Whisper model.

Use the actual native Files and Chat surfaces for acceptance:

1. Choose a short spoken MP3 with one clear, distinctive fact.
2. Upload it in **Files**. Confirm it is labeled **Audio** and moves from
   **Transcribing** to **Ready**.
3. Filter Type to **Audio** and choose **View text**. Confirm **Summary** opens
   first, **Transcript** contains the spoken words, and there is no **Visual
   notes** tab.
4. Confirm the summary is a natural two or three sentence description of the
   recording.
5. Attach that Ready MP3 to a new chat and ask about the distinctive fact.
   Confirm Audrey answers from the recording.
6. Hard refresh and confirm the saved message still shows the audio attachment.

The synthetic `tests/smoke/smoke_audio_ingest.py` remains available as an
optional protocol diagnostic. It creates and removes its own MP3, which is
useful for separating API, worker, and speech-recognition failures, but it is
not the product acceptance gate.

## Completion gate

Slice 8A completes when the rebuilt worker passes the manual browser flow above.
WAV, M4A, FLAC, music analysis, diarization, and speaker labels remain later
work.
