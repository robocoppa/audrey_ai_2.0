# Campaign 2 Phase 12 — Chunk-tail suppression

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The text chunker drops a final near-duplicate tail when its new content beyond
the previous chunk is at most 10% of `chunk_tokens`. The previous chunk's
overlap already covers that text. This applies only after a chunk has been
emitted and only to the final iteration; single chunks and healthy tails remain.

The threshold derives from chunk size rather than a separate tuning option.
Implementation: [`kb/chunk.py`](../../src/audrey/kb/chunk.py).
