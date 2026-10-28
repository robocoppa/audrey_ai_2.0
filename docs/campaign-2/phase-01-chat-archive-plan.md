# Campaign 2 Phase 01 — Searchable chat archive

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Chat history is saved automatically and searched deliberately through a user-scoped
tool. It is separate from curated durable memory and is not injected into every
prompt. Archive writes are best-effort: a storage failure must not fail chat.

`custom-tools` owns raw archive records in SQLite and the semantic Qdrant index.
Audrey captures both streaming and non-streaming responses through shared capture
logic. The authenticated pipeline user overrides model-supplied identity. Search
returns compact snippets; raw records permit index rebuilding without model calls.

Conversation identity, deduplication, interrupted-stream handling and retention
belong to the archive implementation. Current code starts at
[`pipeline/chat_archive.py`](../../src/audrey/pipeline/chat_archive.py) and
[`tools-server/chat_archive.py`](../../tools-server/chat_archive.py).
