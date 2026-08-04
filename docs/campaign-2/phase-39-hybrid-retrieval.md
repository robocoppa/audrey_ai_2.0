# Campaign 2 Phase 39 — hybrid retrieval (BM25 alongside the vectors)

The KB can find things that *mean* what you asked. It cannot reliably find
things that *say* what you asked. This phase adds a lexical retriever next to
the dense one and merges them, so an exact quote and a vague question both
work.

**Status: BUILT, NOT YET DEPLOYED.** Off by default (`kb.hybrid.enabled:
false`) and inert until [`scripts/migrate_bm25.py`](../../scripts/migrate_bm25.py)
has run.

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

And the end-to-end result through the real retriever, taken 2026-08-03 after
the transcript was re-ingested at 250-token chunks with the timestamps removed
— so this is the *best* the dense path currently does:

| query | result |
|---|---|
| a 10-word paraphrase of a passage | its chunk at **0.796** |
| a 6-word phrase appearing **verbatim** in that transcript | **nothing** |

That pair is the whole argument for this phase. Dense retrieval is not weak
here — it improved sharply with the phase 35 chunking fix, from 0.586 to 0.796
on the same query. It is simply the wrong retriever for a quote, and no amount
of making it better at meaning will make it find a literal string.

Two more things follow from the table above, and the second is the alarming one.

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

### Sparse vectors cannot be added to an existing collection (corrected)

**The original plan said the opposite, and it was wrong.** It claimed
`update_collection(sparse_vectors_config=...)` adds sparse vectors to an
existing collection, so nothing is recreated. That was checked on 2026-08-03
against the installed *client*, which proves the method exists — not that a
server accepts the call. It does not:

```
400 Wrong input: Not existing vector name error: bm25
```

Qdrant 1.18.3, confirmed 2026-08-03 by
[`scripts/bm25_probe.py`](../../scripts/bm25_probe.py). That call only edits
the params of a sparse vector that already exists. qdrant-client's local mode
refuses identically, for the same reason, so no amount of hermetic testing
would have caught it either — this needed a real server.

The same shape of error as the nomic-prefix theory earlier in this campaign:
a plausible mechanism, confirmed at the wrong layer. It was caught this time
because the probe ran before anything was built on it.

**What survives is the part that mattered.** Rebuilding is not re-embedding.
The dense vectors are already in Qdrant and scroll back out with
`with_vectors=True`, so the migration copies them across verbatim and computes
the sparse vector from `payload.text`, which is stored on every point. The
embedder is never called, no GPU is touched, and no source file is re-read —
which is what makes it affordable, and is load-bearing because for text
uploads the source bytes are long gone.

Also verified on the real server, because each of these could have forced a
much larger phase:

| checked | result |
|---|---|
| unnamed dense + named sparse in one collection | works |
| dense search with no `using=` after the change | works |
| a point with no sparse vector yet | still dense-searchable |
| the real `document_vector` output | accepted, IDF live |

The second row is why this phase does not touch a single existing dense call
site. Had Qdrant required the dense vector to be *named* once sparse vectors
exist, every read and write in `kb/qdrant.py` would have had to change, and
the collection swap and the code deploy would have had to be simultaneous.

### Merge by rank, not by score — but not in Qdrant (corrected)

BM25 scores are unbounded and corpus-relative; cosine is bounded and absolute.
There is no correct constant that converts one to the other, and any weighted
sum of the two silently re-tunes itself as the corpus grows.

Reciprocal Rank Fusion takes only the *position* of a document in each list:

```
score(d) = Σ  1 / (k + rank_i(d))
```

It needs no normalisation, no per-corpus calibration, and degrades gracefully
when one retriever returns nothing.

Qdrant does implement this server-side, and it works —
`query_points(prefetch=[...], query=FusionQuery(fusion=Fusion.RRF))`, verified
against 1.18.3. **It is deliberately not used, and that reverses the original
plan.**

A fused score cannot be audited. Qdrant returns the RRF value and discards
what each retriever thought, and the junk rule below has to be stated in terms
of *evidence* rather than magnitude — which needs exactly the numbers fusion
throws away. A hit scoring 0.0163 might be a passage both retrievers ranked
second, or the least-irrelevant document in a corpus that cannot answer the
question at all. Those must be treated differently, and after fusion they are
indistinguishable.

So the merge lives in [`kb/fusion.py`](../../src/audrey/kb/fusion.py), which
is twenty lines and testable without a server. The cost is a second Qdrant
round trip, issued concurrently with the first, on a path that already waits
on an embedding call. `Fusion.DBSF` remains available and worth measuring once
there is a corpus to measure on.

### The evidence rule shipped broken once, and how (2026-08-03)

The first live run returned PowerApps and ServiceNow documents for "how do
mRNA vaccines work" — the 2026-07-15 incident, reproduced by the rule written
to prevent it. Worth recording in full, because the mistake was in reasoning
rather than in code.

`term_overlap` counted **every** query term, function words included. Three of
that query's five terms are `how`, `do` and `work`, which every long document
contains, so junk cleared a 0.5 threshold while `mrna` and `vaccines` appeared
nowhere at all.

The reason it was written that way is the interesting part. `kb/bm25.py`
argues at length that no stopword list is needed, because `Modifier.IDF`
already drives common terms to near-zero weight. That argument is correct —
**for the stored vector**. `term_overlap` was added later, is an unweighted
count, and applies no IDF, so the same words that cost nothing in the vector
were worth full marks as evidence. One inherited argument, two different jobs.

The fix is a stopword list used *only* by `term_overlap`; the vectors still
keep every term. Re-measured against the passages that actually came back:

| passage | overlap |
|---|---|
| ServiceNow README (junk) | 0.25 |
| PowerApps audio control (junk) | 0.33 |
| USFS botany index (junk) | 0.667 |
| a real answer | 0.75 |
| the verbatim transcript | 1.00 |

`min_term_overlap` moved 0.5 → **0.7**, which is the gap. It is a narrow one —
the worst junk case is a single content word below the threshold — so this
number is a measurement against this corpus, not a constant.

A second defect shipped alongside it: the response reported each hit's *raw*
retriever score, so a cosine of 0.47 sat next to a BM25 score of 13.8, in an
order explained by neither. The fused RRF value is now what comes back, since
it is the only number that describes the list it is in.

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
- **[`scripts/migrate_bm25.py`](../../scripts/migrate_bm25.py)** — the
  collection rebuild, since sparse vectors cannot be added in place. Per
  collection: build a scratch collection with sparse config, copy every point
  into it (dense verbatim, sparse computed, payload byte for byte), **verify
  the counts match**, and only then delete the original, recreate it correctly,
  and copy back. Two copies rather than one because Qdrant has no rename, and
  an alias would leave the real collection name pointing somewhere else
  forever.

  Resumable at the one moment it matters. If it dies before the original is
  deleted, nothing is lost. If it dies in the window where the scratch is the
  only copy, a rerun finishes from the scratch instead of rebuilding from an
  original that no longer exists — the bug that window caused was caught by
  `tests/test_migrate_bm25.py` before the script ran anywhere.

  A BM25 vector is arithmetic over the `payload.text` already on every point,
  so **nothing is re-read from source** — which matters because for text
  uploads the source bytes are long gone.

  **Scoped to `kb_text` and `kb_user_text_*`** — exactly what the hybrid query
  path reads. `kb_chat_archive` and `kb_memory` are text collections too, and
  migrating them would look consistent, but `tools-server` owns them: it
  creates them itself and upserts bare-list dense vectors through code this
  phase does not touch. Nothing searches them lexically, so rebuilding them
  buys nothing today and takes on the risk of reshaping another service's
  storage to do it. The rule is worth stating generally — *migrate what a
  query path reads, not everything that shares a shape.*
- **`config.yaml`** — `kb.hybrid.enabled` (off), `rrf_k`, `min_term_overlap`.

## What's NOT in scope

- **No reranking.** A cross-encoder over the merged top-N is the next lever
  after this one, and it is worth measuring separately.
- **No query expansion or rewriting.**
- **No embedder change.** Ruled out above, with numbers.
- **No new chunk sizes.** They move *after* this lands, driven by what the
  merged retriever needs, not before.

## The parts that will bite

- **The migration is the risky part**, even though it turned out cheap. It
  deletes and rebuilds every text collection, and it runs against a live
  search. Resumability matters more than speed, and the count check before the
  delete matters more than either.
- **A collection is missing for a few seconds mid-rebuild**, between the delete
  and the recreate. Reads against it return no hits during that window —
  annoying, not damaging, and the reason to run this when nobody is asking
  questions.
- **Half-indexed is fine, and was verified.** A point with no sparse vector
  stays dense-searchable, so a collection mid-migration behaves exactly as it
  did before the phase. Ingest asks `has_sparse` per collection so it never
  writes a sparse vector into a collection that would reject it.
- **BM25 is tokenizer-dependent.** Stemming, stopwords and casing all change
  what "exact" means. `[00:08:46]` timestamps are already out of transcript
  text (Phase 35) — do not let them back in through the lexical path.
- **Two retrievers, two outages.** If the sparse index is unavailable the query
  must degrade to dense-only and say so in the response, not fail.
- **User isolation applies to both paths.** Every Qdrant read is scoped by
  `user` and `file_id` today. A new retrieval path is a new place to forget
  that, and the failure is one user reading another's uploads.

## Deploy

Three steps, in this order. The code ships inert, so steps 1 and 2 are safe to
separate by as long as you like.

**1. On the box**, from `/mnt/user/appdata/audrey_ai_2.0` — `docker compose up
-d --build audrey-ai`. Qdrant is unchanged as a container; the index lives
inside collections it already hosts. Hybrid is off, so nothing about retrieval
changes yet. New text collections created from here on are born with sparse
config and never need migrating.

**2. On the laptop**, from the repo root — the migration. Dry run first; it
lists what would be rebuilt and stops.

```
.venv/bin/python scripts/migrate_bm25.py --host 192.168.1.11 --dry-run
.venv/bin/python scripts/migrate_bm25.py --host 192.168.1.11
```

**3.** Set `kb.hybrid.enabled: true` in `config.yaml`, push, pull, and
recreate `audrey-ai`.

### Rollback

`kb.hybrid.enabled: false` restores the dense-only path in one setting. The
sparse vectors stay on disk unused, so re-enabling costs nothing and no
migration is repeated. Nothing else in the phase changes existing behaviour:
the dense vectors are byte-identical to what they were, and every dense search
still runs with no `using=`.

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

## What this unblocks

Quoting. Every artifact in the KB becomes findable by what it literally says as
well as by what it means — which matters most for the things people quote:
transcripts, procedures, error messages, and the exact wording of a document
they half-remember.
