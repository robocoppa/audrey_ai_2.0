# Campaign 3 Phase 10 - ordinary-answer provenance

**Status:** Complete. Slice 10A passed its native browser gate on
2026-10-01.

## Goal

Show which public pages and uploaded file artifacts reached an ordinary Fast or
Deep answer through Audrey's tools, using deterministic server observations.
Do not ask an answer-writing model to reproduce URLs, filenames, or citation
identifiers from memory.

## Existing foundation

Native runs already observe successful web-search and web-fetch URLs, sanitize
them, save them on the assistant message, and render them after the answer in a
compact expandable Sources disclosure. Models and Tool calls use adjacent
disclosures. All three survive refresh without restoring raw tool cards.

The remaining gap was private file evidence. Knowledge and file-reader results
identify what was read with a filename and artifact rather than a public URL.
The URL-only observation boundary discarded those rows, so an answer grounded
in an uploaded transcript or document could show its Tool calls but no source.

## Slice 10A - private file evidence envelope

Successful content-bearing tools now project these identities:

- `kb_search` and `kb_image_search`: every distinct filename/artifact pair in
  the retained result rows;
- `get_file_text`: the exact filename and resolved artifact page returned to
  the model; and
- existing `web_search` and `web_fetch`: the same sanitized public URLs as
  before.

`list_my_files` remains excluded. It is a catalogue that says a file exists;
it does not read the file, so presenting its names as answer evidence would be
false provenance.

The private envelope contains only a bounded display title such as
`meeting.opus · Transcript`, an internal identity used to deduplicate repeated
hits, and the producing tool name. The identity is hashed before it becomes a
run-event source id. Retrieved text, snippets, storage paths, user identifiers,
and raw tool bodies never enter source events or saved message metadata.

The existing canonical `app_message_sources` relation already accepts a titled
source with no URL, and the native Sources disclosure already renders such a
source as text instead of a link. This slice therefore needs no migration and
no new chat card or frontend branch.

## Laptop verification

- Multiple chunks from one file/artifact collapse to one identity.
- Different artifacts from the same file remain distinguishable.
- Full file reads produce one filename/artifact source.
- File contents and internal paths do not enter events.
- Catalogue-only and unidentified global-KB rows produce no provenance.
- Public web URL behavior remains unchanged.
- The actual observed-dispatch boundary emits the file source.
- Existing source persistence, native run, Fast observation, and model/tool
  activity contracts remain green.

**Laptop result, 2026-10-01:** Passed. The focused evidence tests passed 24
checks, the complete ReAct, observation, canonical persistence, Fast, and
native-run regression surface passed 114 tests, and the full hermetic backend
suite passed 3,098 tests. Changed-file Ruff, compilation, and diff checks are
clean.

## Live native browser gate

1. Use an existing uploaded document, audio file, or video with one fact that
   can be recognized in its document or transcript.
2. In Fast mode, ask Audrey a question that requires that fact. If needed,
   attach the file so the request unambiguously names it.
3. Confirm the answer finishes and the compact Sources disclosure appears on
   the same line as Models and Tool calls.
4. Expand Sources. Confirm it names the exact file and artifact, such as
   `meeting.opus · Transcript`, and does not display retrieved text or a local
   storage path.
5. Expand Tool calls separately. Confirm the existing compact tool summary is
   still present and no raw tool cards appear in chat.
6. Hard refresh the browser and confirm the file source, Models, and Tool calls
   remain attached to the saved assistant answer.
7. Ask which files are uploaded. If that turn only lists the catalogue, confirm
   it does not present those filenames as Sources.

## Completion gate

**Passed, 2026-10-01.** The user confirmed the complete native browser gate,
including grounded private-file Sources, the catalogue-only control, compact
Models and Tool calls presentation, and refresh persistence.

A later claim-level citation slice requires a deterministic claim-to-evidence
link; it must not infer linkage from the prose answer.
