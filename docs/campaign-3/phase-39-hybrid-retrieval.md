# Campaign 3 Phase 39 — Dense and lexical retrieval

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Dense semantic retrieval and BM25 lexical retrieval are fused so paraphrases
and literal quotes can both find evidence. Hybrid collections retain the original
dense vectors while adding the lexical representation. A low dense cosine must
not discard a strong lexical hit before fusion.

Global and user-scoped retrieval retain their ownership filters. A meaningful
isolation check first proves the query finds the owner's document, then verifies
another user cannot retrieve it. An empty query for everyone proves no isolation.
The historical migration and owner/other-user quote checks passed.
