# Campaign 3 Phase 09 — broader spoken audio

**Status:** Complete and accepted.

## Additional supported inputs

| Format | Measured container/MIME |
|---|---|
| OGG | Ogg Vorbis / `audio/ogg` |
| Opus | Ogg Opus / `audio/ogg` |
| AAC | Raw ADTS / `audio/x-hx-aac-adts` |

AAC inside M4A uses the existing M4A path. Admission uses real byte sniffing, retains the original suffix, and reuses Phase 08's queue, bounded WAV decoding, transcript/summary, download, search, and attachment behavior. No new artifact or frontend branch is required.

Speaker diarization, speaker labels, music analysis, and event/scene analysis are parked. Reopen only for a concrete user question that current transcripts or visual notes cannot answer, with a defined output and measured quality gate.
