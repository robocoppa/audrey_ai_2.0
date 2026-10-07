# Campaign 3 Phase 08 — spoken audio ingestion

**Status:** Complete and accepted.

## Supported inputs

| Format | Measured MIME |
|---|---|
| MP3 | `audio/mpeg` |
| WAV | `audio/x-wav` |
| M4A | `audio/x-m4a` |
| FLAC | `audio/flac` |

Admission uses sniffed bytes; filename extensions are browser hints. Audio is a first-class file and attachment kind, preserved through catalog reconciliation and restart.

## Processing and product behavior

The existing media queue claims audio, ffprobe verifies its stream, ffmpeg converts it to bounded 16 kHz mono WAV, and Whisper transcribes it. There is no visual pass. Audrey stores/indexes an owner-scoped transcript and generates a natural two or three sentence audio summary from transcript evidence only.

My Files shows Summary and Transcript, with no Visual notes. Audio can be downloaded, searched, attached to native chat, and used for grounded questions. Originals remain retained; video-source reclamation does not apply.

Summary failure does not fail a useful transcript. The media-worker image retains its `av<19` decoder compatibility pin and build-time WAV probe. No separate queue or audio-only storage authority is introduced.

Broader admitted containers are in [Phase 09](phase-09-broader-audio.md).
