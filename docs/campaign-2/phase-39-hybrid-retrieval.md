# Campaign 2 Phase 39 — hybrid retrieval (BM25 alongside the vectors)

The KB can find things that *mean* what you asked. It cannot reliably find
things that *say* what you asked. This phase adds a lexical retriever next to
the dense one and merges them, so an exact quote and a vague question both
work.

**Status: PLANNED.**

Not video-specific, despite being found there. Every collection has this
problem; transcripts just made it obvious, because a transcript is the one
artifact people quote from verbatim.

---

## The measurements that prompted this

All taken 2026-08-03 against `nomic-embed-text` on the box, via
[`scripts/embed_prefix_probe.py`](../../scripts/embed_prefix_probe.py). The
query is a six-word phrase that appears **verbatim** in every passage below.

| passage | length | cosine | vs `kb.min_score` 0.53 |
|---|---|---|---|
| the surrounding two sentences | ~240 chars | **0.578** | passes |
| the surrounding paragraph | ~630 chars | **0.460** | **fails** |
| an unrelated passage (decoy) | ~120 chars | 0.431 | correctly rejected |

Two things follow, and the second is the alarming one.

**Dilution with length is steep.** The same exact phrase drops 0.118 by adding
surrounding context that a reader would call relevant.

**The margin collapses before the score does.** At 630 characters the correct
passage scores 0.460 against a decoy's 0.431 — a gap of **0.029**. The
retriever has not merely become strict, it has stopped being able to tell the
right answer from an unrelated one. No floor setting rescues that: lower it and
the decoy comes too, raise it and the answer is lost. This is why the fix
cannot be a tuning change.

### What was ruled out first

- **Chunk size.** Transcript chunks went 1000 → 250 tokens in Phase 35. It
  helped (3 chunks became 9) and did not fix quoting, because a 250-token chunk
  is still ~800 characters — the failing row of the table above.
- **nomic task prefixes.** `search_query:` / `search_document:` raise scores by
  a near-constant amount (+0.033 short, +0.069 long, +0.049 on the *decoy*), so
  they lift everything without widening the margin. A full re-embed of the KB
  would have bought nothing. Measured before proposing, and the theory died on
  the number.
- **Lowering `kb.min_score`.** With a 0.029 margin, the floor is not the
  problem. It is currently the only thing keeping unrelated results out.

## Design decisions

### Dense and sparse, merged — not one or the other

A short quoted phrase is a *lexical* query and dense vectors are *semantic*.
These are different jobs and neither retriever is bad at its own. Replacing the
vectors with BM25 would break the paraphrase queries that already work today
(a 25-word paraphrase scored 0.586 and returned correctly).

The failure to design against is a hybrid that is worse than either half — most
often because the two score scales get averaged as if they were comparable.

### Merge by rank, not by score — and let Qdrant do it

BM25 scores are unbounded and corpus-relative; cosine is bounded and absolute.
There is no correct constant that converts one to the other, and any weighted
sum of the two silently re-tunes itself as the corpus grows.

Reciprocal Rank Fusion takes only the *position* of a document in each list:

```
score(d) = Σ  1 / (k + rank_i(d))
```

It needs no normalisation, no per-corpus calibration, and degrades gracefully
when one retriever returns nothing.

**`qdrant-client` 1.17.1 implements this server-side** (verified 2026-08-03,
against the installed client). `query_points(prefetch=[...],
query=FusionQuery(fusion=Fusion.RRF))` runs both retrievers and fuses them in
one round trip. So this phase does *not* write a merge function, does not pull
two full result sets over the wire, and does not own the `k` constant.
`Fusion.DBSF` — distribution-based score fusion — is also available and worth
measuring against RRF once there is a corpus to measure on.

### `kb.min_score` has to be replaced, not ported

The floor exists for a real reason — [`routes/kb.py`](../../src/audrey/routes/kb.py)
records the 2026-07-15 incident where a query the corpus could not answer
returned its least-irrelevant junk, which then read as a real source. That
requirement survives this phase.

But RRF output is not a similarity, so `>= 0.53` is meaningless against it. The
replacement has to be a rule about *evidence*, not magnitude — a document
appearing in only one retriever's list, at a poor rank, with no lexical overlap
at all, is the shape of junk. This is the most likely thing to get wrong here,
and the incident it protects against was silent.

### Chunk size stops being one compromise

Today chunk size is a single number trading exact-phrase recall against
semantic coherence, and no value is good at both — Phase 35 walked that
trade-off in both directions. With a lexical retriever handling quotes, chunks
can be sized for meaning alone. Expect the transcript chunk size to go *back
up* once this lands.

### Where the lexical index lives

Qdrant, alongside the existing points, rather than a second store.

The alternative — sqlite FTS5, which the repo already has a connection to — was
rejected because per-user isolation, the delete-by-file_id path, and the
reconcile sweep are all solved once in Qdrant and would each need solving again
for a second index. A retrieval store that can disagree with itself about which
documents exist is a worse problem than the one being fixed.

## What's in scope

- **[`kb/qdrant.py`](../../src/audrey/kb/qdrant.py)** — a sparse vector or
  full-text index on the existing collections, and a search that returns both
  lists.
- **[`kb/ingest.py`](../../src/audrey/kb/ingest.py)** — populate the lexical
  index on every write path (`ingest_text_file`, `ingest_user_text_file`,
  `ingest_transcript_segments`).
- **[`routes/kb.py`](../../src/audrey/routes/kb.py)** — the hybrid query path,
  RRF merge, and the junk rule that replaces `min_score`.
- **A backfill** — much cheaper than first assumed, for three reasons verified
  against the installed client on 2026-08-03:

  1. `update_collection(sparse_vectors_config=...)` **adds sparse vectors to an
     existing collection.** No collection is recreated and no dense vector is
     recomputed, so the embedder is never called and the GPU is never touched.
  2. `update_vectors` **writes vectors to existing points without touching
     their payload**, so the backfill cannot corrupt metadata even if it fails
     midway.
  3. A BM25 vector is a tokeniser plus corpus statistics — arithmetic, not a
     model. It is computed from the `payload.text` already stored on every
     point, so **nothing needs re-reading from source**. The earlier worry that
     user uploads could not be re-ingested (their source bytes are gone for
     text files) does not apply.

  What remains is a scroll-and-update pass per collection, resumable, with the
  corpus statistics computed in a first pass before any write.
- **`config.yaml`** — `kb.hybrid.enabled`, `rrf_k`, per-retriever `top_k`.

## What's NOT in scope

- **No reranking.** A cross-encoder over the merged top-N is the next lever
  after this one, and it is worth measuring separately.
- **No query expansion or rewriting.**
- **No embedder change.** Ruled out above, with numbers.
- **No new chunk sizes.** They move *after* this lands, driven by what the
  merged retriever needs, not before.

## The parts that will bite

- **The backfill is still the risky part**, even though it turned out cheap.
  It touches every point in every collection, and it runs against a live
  search. Resumability matters more than speed.
- **Half-indexed is worse than un-indexed.** During backfill, a document present
  in the dense index and absent from the lexical one ranks in one list only and
  loses to fully-indexed documents. The merge has to be provably fair to
  partially-indexed corpora, or the rollout degrades results while it runs.
- **BM25 is tokenizer-dependent.** Stemming, stopwords and casing all change
  what "exact" means. `[00:08:46]` timestamps are already out of transcript
  text (Phase 35) — do not let them back in through the lexical path.
- **Two retrievers, two outages.** If the sparse index is unavailable the query
  must degrade to dense-only and say so in the response, not fail.
- **User isolation applies to both paths.** Every Qdrant read is scoped by
  `user` and `file_id` today. A new retrieval path is a new place to forget
  that, and the failure is one user reading another's uploads.

## Deploy on Unraid

**On the box**, from `/mnt/user/appdata/audrey_ai_2.0`:

```
docker compose up -d --build audrey-ai
```

Qdrant is unchanged as a container; the index lives inside collections it
already hosts.

## Verification (to be written against the built phase)

**0. The exact quote that started this returns its chunk.** `"and watch us play
some baseball"` must come back from the transcript at rank 1. This is the
acceptance test for the whole phase.

**1. Paraphrase queries do not regress.** The 25-word paraphrase that scored
0.586 must still return the same chunk. A hybrid that fixes quoting and breaks
meaning is a net loss.

**2. The decoy margin widens.** Re-run the probe's comparison through the merged
retriever; the gap between the correct passage and an unrelated one must be
much larger than 0.029.

**3. Junk is still refused.** A query the corpus cannot answer returns nothing,
not its least-irrelevant document. This is the 2026-07-15 incident and the
reason `min_score` exists.

**4. Isolation holds on both paths.** A second user cannot retrieve the first
user's transcript by quoting it — the lexical path makes this newly easy to get
wrong, since an exact phrase match needs no similarity at all.

**5. Dense-only degradation.** With the lexical index unavailable, queries still
answer.

### Rollback

`kb.hybrid.enabled: false` restores the dense-only path. The lexical index stays
on disk unused, so re-enabling costs nothing and no backfill is repeated.

## What this unblocks

Quoting. Every artifact in the KB becomes findable by what it literally says as
well as by what it means — which matters most for the things people quote:
transcripts, procedures, error messages, and the exact wording of a document
they half-remember.
