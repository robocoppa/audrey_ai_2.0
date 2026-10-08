# Campaign 3 Phase 10 — ordinary-answer provenance

**Status:** Foundation accepted. Immediate per-answer summaries are built;
browser acceptance is pending.

## Evidence contract

The server observes content-bearing tool results for ordinary Fast/Deep answers and saves deterministic evidence identities:

- `web_search` and `web_fetch`: sanitized public URLs.
- `kb_search` and `kb_image_search`: distinct identified filename/artifact pairs from retained results.
- `get_file_text`: the returned filename and resolved artifact.
- `list_my_files`: excluded, because listing a catalog does not read its contents.

Repeated chunks from one file/artifact collapse to one source; different artifacts remain distinct. Unidentified global-KB rows do not manufacture file evidence.

Private sources contain a bounded display title and deduplication identity/tool name. Run-event ids use a hashed identity. Retrieved text, snippets, storage paths, user identifiers, and raw tool bodies do not enter source events or saved source metadata.

## Presentation and limits

Sources, Models, and Tool calls appear as compact adjacent expandable
disclosures on each assistant answer as soon as its run ends. They remain across
later turns and refresh, including when progress display is off. Active runs keep
the existing optional progress row; completed runs do not duplicate it. Opening
one disclosure closes its peers; outside click and Escape close it.

A private source without a URL is displayed as text. Tool names describe actual
operations (for example `get_file_text`), rather than creating a separate tool for
the selected skill. Repeated calls are grouped with their count and status. Raw
tool cards, arguments, and result bodies do not return.

These are observed sources, not claim-level citations. Any future claim-to-evidence feature requires deterministic linkage rather than inference from answer prose.
