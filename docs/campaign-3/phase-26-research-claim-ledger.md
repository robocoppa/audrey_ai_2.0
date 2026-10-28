# Campaign 3 Phase 26 — Research claim/source ledger

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The ledger is internal scaffolding for the opt-in research pipeline. Researcher
notes are structured into claims and sources; fact-check verdicts become writer
corrections. URL deduplication and worker-prefixed claim IDs preserve provenance.

Parsers fail soft on malformed output and accept fenced JSON or a bare top-level
array where supported. Structure failures retain the usable prose path rather
than failing the whole answer. Unsupported claims can be dropped, conflicting
ones hedged, and ungrounded controls must not receive blanket uncertainty.

A deterministic Sources block uses surviving claim support, validates HTTP(S)
URLs, ranks source types, deduplicates and caps the list. It is produced outside
writer prose. Search can fall back from Brave to SearXNG on quota exhaustion;
provider failures must remain observable.

The five staged ledger changes were historically deployed and evaluated. That
record does not guarantee every live web run has equally rich grounding. Current
flags and prompts are in `config.yaml`; current evaluation guidance is in
[`evals/README.md`](../../evals/README.md).
