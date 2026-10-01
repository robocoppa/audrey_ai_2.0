# Campaign 3 Phase 9 - broader spoken-audio support

**Status:** In progress. Slice 9A is laptop-complete and awaits its native
browser gate.

## Goal

Accept the remaining common spoken-audio containers that Audrey's current
worker can decode, while preserving the same owner, queue, transcript,
summary, Files, and chat boundaries established in Phase 8.

Phase 9 does not infer new media-analysis features from container support.
Speaker diarization and broader music or scene analysis require their own
measured slices after concrete user workflows define their output and quality
requirements.

## Slice 9A - OGG, Opus, and AAC admission

Real one-second ffmpeg fixtures were measured through the same libmagic and
ffmpeg installations used by Audrey's upload and worker contracts:

| Suffix | Container/codec fixture | Sniffed MIME |
|---|---|---|
| `.ogg` | Ogg Vorbis | `audio/ogg` |
| `.opus` | Ogg Opus | `audio/ogg` |
| `.aac` | raw ADTS AAC | `audio/x-hx-aac-adts` |

The suffix is a browser hint. Admission remains based on the sniffed bytes.
Both Ogg variants share the MIME reported by libmagic, while their stored
suffix is retained so a claimed job names the original container accurately.
AAC in an M4A container remains the already-supported `.m4a` path; the new
`.aac` entry is specifically the measured raw ADTS container.

No new queue, database migration, frontend branch, artifact type, or prompt is
needed. The backend publishes its derived extension list to both native file
pickers. Once admitted, each file is stored as kind `audio`, claimed with its
original suffix and MIME, decoded to Whisper's bounded 16 kHz mono WAV, and
presented through the existing Summary, Transcript, download, search, and chat
attachment paths.

## Verification

The laptop gate requires:

- real Ogg Vorbis, Ogg Opus, and ADTS AAC fixtures to sniff to the table above;
- ffprobe to find their audio streams and ffmpeg to decode every fixture to
  Whisper's WAV input;
- upload admission to create pending audio jobs;
- job claims to preserve kind, MIME, and source suffix;
- native Files to classify all three formats as Audio and expose Summary plus
  Transcript without Visual notes; and
- the established upload allowlist invariants to remain fail closed.

**Laptop result, 2026-10-01:** Passed. All three real fixtures produced the
measured MIME values and decoded to Whisper-compatible WAV. The focused upload,
allowlist, decoder, claim, and native Files suites passed 156 tests. The full
hermetic backend suite passed 3,092 tests with one existing FastAPI deprecation
warning; changed-file Ruff, compilation, and diff checks are clean.

The live gate is manual because it must exercise the actual product surface:

1. Upload one spoken `.ogg`, `.opus`, and raw ADTS `.aac` file through Files.
2. Confirm each is accepted as Audio and reaches Ready.
3. Open each file and confirm a Summary and Transcript are present, with no
   Visual notes tab.
4. Attach one of them to a new chat, ask one question grounded in its speech,
   and confirm the attachment survives a browser refresh.
5. Delete the three disposable files and confirm they leave Files.

## Later Phase 9 slices

### Speaker-aware transcripts

Diarization remains separate from admission. Before implementation, choose a
real multi-speaker workflow, measure speaker-count and segment-boundary quality,
define how unknown speakers appear in the transcript and citations, and retain
the current plain transcript as a rollback path.

### Broader media analysis

Do not add music, event, or scene-specific analysis from a generic feature
list. Start with a real user question that the current transcript and visual
notes cannot answer, define the artifact needed to answer it, then measure the
smallest model and processing path that produces a useful result.

## Completion gate

Slice 9A closes when all three formats pass the native live gate. Later speaker
and media-analysis slices open only after their user workflow and quality gate
are explicit.
