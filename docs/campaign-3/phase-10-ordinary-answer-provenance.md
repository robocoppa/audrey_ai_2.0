# Campaign 3 Phase 10 — ordinary-answer provenance

**Status:** Complete and accepted.

## Evidence contract

The server observes content-bearing tool results for ordinary Fast/Deep answers and saves deterministic evidence identities:

- `web_search` and `web_fetch`: sanitized public URLs.
- `kb_search` and `kb_image_search`: distinct identified filename/artifact pairs from retained results.
- `get_file_text`: the returned filename and resolved artifact.
- `list_my_files`: excluded, because listing a catalog does not read its contents.

Repeated chunks from one file/artifact collapse to one source; different artifacts remain distinct. Unidentified global-KB rows do not manufacture file evidence.

Private sources contain a bounded display title and deduplication identity/tool name. Run-event ids use a hashed identity. Retrieved text, snippets, storage paths, user identifiers, and raw tool bodies do not enter source events or saved source metadata.

## Presentation and limits

Saved Sources, Models, and Tool calls appear as compact adjacent expandable disclosures and survive refresh. A private source without a URL is displayed as text. Raw tool cards do not return.

These are observed sources, not claim-level citations. Any future claim-to-evidence feature requires deterministic linkage rather than inference from answer prose.
