# Campaign 3 Phase 8 - MP3 audio ingestion

**Status:** Slice 8A is laptop-complete and awaiting its targeted Unraid smoke.

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
- The full hermetic backend suite passes: 3,033 tests with one existing FastAPI
  deprecation warning.
- Changed-file Ruff, Python compilation, and diff checks pass.
- Frontend packages are present, but this shell has no Node or npm executable,
  so the Vitest and TypeScript build commands cannot run here. The deployment
  build is the remaining compile proof for the UI changes.

## Deploy and targeted live smoke

Rebuild audrey, audrey-ui, and custom-tools. The existing media-worker image
already handles audio-only ffmpeg input, and this slice does not change worker
code or packages.

Run the targeted smoke from the laptop checkout over the working LAN/WARP
route. **You do not upload anything first.** The script uses the laptop's local
ffmpeg flite source to create a short spoken MP3, uploads it, waits for the
deployed worker, reads known words from its transcript, confirms audio has no
visual artifact, deletes the upload, and drains cleanup.

    cd /home/bart/Documents/github/audrey/audrey_ai_2.0
    (
      set -a
      source .env.test.local
      set +a
      AUDREY_SMOKE_BASE_URL=http://192.168.1.11:8000 .venv/bin/python tests/smoke/smoke_audio_ingest.py
    )

Success is exit code zero and JSON ending in "status": "passed" with:

- upload HTTP 200, kind audio, and initial_status pending;
- processing final_status ready, a positive chunk count, a positive duration,
  and transcript source whisper;
- transcript HTTP 200 with all four required words: audio, blue, lantern, and
  ready;
- summary reader HTTP 200 with a nonempty library description;
- visual reader HTTP 422;
- cleanup deleted true and repair status ready.

A failure that says ffmpeg cannot generate the fixture is a laptop prerequisite
failure and happens before upload. A row-level failure includes the worker's
bounded reason. A transcript-word failure prints the recognized text so the
fixture or transcription can be corrected without guessing.

## Browser check

After the automated smoke passes, use any short spoken MP3 for the visible UI
check:

1. Open **Files** and upload the MP3. No video is needed.
2. Confirm the row says **audio**, shows the music-note icon, and changes from
   **Transcribing** to **Ready**.
3. Filter Type to **Audio**, open **View text**, and confirm only **Summary** and
   **Transcript** tabs exist. Read the transcript and make sure it matches the
   recording.
4. Attach the Ready MP3 to a new chat message and ask one specific question
   answered in the recording. Confirm Audrey answers from its contents and the
   saved message still shows the attachment after a hard refresh.

## Completion gate

Slice 8A completes when the targeted automated smoke passes on Unraid and the
browser check confirms audio filtering, artifact tabs, and one saved chat
attachment. WAV, M4A, FLAC, music analysis, diarization, and speaker labels
remain later work.

