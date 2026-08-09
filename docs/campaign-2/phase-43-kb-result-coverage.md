# Campaign 2 Phase 43 — coverage in unscoped KB results

`kb_search` returns hits **pooled across files and ordered by score**. When a
question spans several similar documents, the top-k can be almost entirely one
of them — and `list_my_files` has already told the model the others exist. So it
writes a section per file and fills the empty ones from whatever it has.

**Status: PLANNED. Nothing built.** Stage 0 is a measurement whose result
decides whether Stage 1 gets built at all.

This is the input half of a problem whose output half already shipped
(2026-08-09): the `kb_search` tool description now says results are pooled, that
each result carries its own `filename`, and that knowing a file exists is not
the same as having read it. That change is **deployed but unmeasured**. If it
turns out to be sufficient, this phase closes at Stage 0 having added one log
field — which is a good outcome, not a wasted one.

---

## The evidence

Phase 42's third A-B run, on `What do my videos say about the London System?`
against a corpus holding two London-System videos — a 7-minute narrated blitz
game and a 30-minute instructional lesson.

`audrey_video` produced correct headings with wrong contents: structured theory
(Bf4/e3/c3/Nge2, a "Kashdan Trap") attributed to the **blitz commentary**, which
contains none of it. `audrey_auto` failed differently and worse on the same
prompt — its trace was `kb_search ×1` alone, no listing and no file reads, and
it credited both videos' speakers to **the user** ("*You* comment that it hasn't
been super popular…"). Reproducible across two runs.

Two symptoms, one cause: a pooled result set whose `filename` field goes unread,
or which never carried the second file in the first place.

**What was ruled out.** Truncation is not corrupting the labels.
`_truncate_payload` (`src/audrey/tools/dispatch.py:151`) drops whole `results`
items rather than cutting mid-JSON, so every surviving hit keeps its `filename`
(`src/audrey/routes/kb.py:104`). The labels arrive intact and go unread.

---

## Why not fix the truncator

The obvious move is to make `_truncate_payload` drop fairly across files instead
of from the tail. Rejected, for two reasons.

**It is the wrong layer.** That function is deliberately tool-agnostic — its
docstring says so: *"Chosen by serialized size rather than by name so a tool
added later needs no entry here."* It picks the heaviest list by serialized size
and knows nothing about what is in it. Teaching it about `filename` buys this
one fix and costs a property that every future tool gets for free. Filenames are
the KB route's business.

**It may not be the binding constraint.** `top_k` defaults to 5
(`tools-server/app.py:175`) and the model asked for 10. If those 10 hits are
already 8:2 toward the longer transcript — very likely, since a 30-minute lesson
has far more matching material than a 7-minute commentary — then the model had
almost nothing from the second file *before* anything was dropped, and fair
truncation would change nothing at all.

Balancing at the source fixes both cases and leaves the generic truncator alone:
put one hit from each file in the head, and a tail drop can no longer delete a
file entirely.

---

## Steps

### Stage 0 — measure (~10 lines, ships something durable)

`src/audrey/routes/kb.py:357` already logs every query. Add the per-file
distribution of the returned hits to it:

```
kb.query: user=… scope=none top_k=10 -> 10 hit(s) in 0.08s  files=[How-to-WIN:8, Carlsen:2]
```

Worth keeping whatever happens next. Right now nothing on the box can answer
"did the model have material from both files" after the fact, and that hole is
why this bug survived three A-B runs — every answer *read* as though both files
had been consulted.

Bucket empty filenames (global-KB hits) under a single label rather than
omitting them, so a skew between global and user material is visible too.

**Decision gate.** Re-run the plural case and read the split:

| Observed split | Meaning | Action |
|---|---|---|
| Roughly balanced | Truncation is dropping a file | Build Stage 1 |
| Heavily skewed | Retrieval never surfaced the file | Build Stage 1 — it fixes this too |
| One file absent entirely | Not a ranking problem | **Stop.** Ingest or embedding; wrong phase |

The third row is the one worth taking seriously. A file that never appears in an
unscoped search has an ingest problem, and building a re-ranker on top of it
would hide the fault rather than fix it.

### Stage 1 — guarantee head coverage (~25 lines)

In `src/audrey/routes/kb.py`, after hits are merged and before `QueryResponse`
is built (`:360`): when a search is **unscoped** and its hits span more than one
`filename`, reorder so that every file's best hit precedes any file's second
hit. Score order applies within the head and throughout the tail.

Diversity-then-relevance. It composes with what already exists rather than
working around it — the head now carries one hit per file, so the existing
tail-drop truncation preserves coverage for free.

**Pure reorder.** No change to scoring, to `min_score`, to which hits are
returned, or to the response schema. Only the order of `results`.

Constraints:

- **Global-KB hits have an empty `filename`** (`src/audrey/routes/kb.py:104`).
  Treat `""` as one bucket, never as N singleton files — otherwise a geology
  query with no uploads involved gets reshuffled for nothing.
- **Scoped searches skip this entirely.** `filename=` was passed, there is one
  file, there is nothing to balance. Detect it the same way `_scope_label`
  already does (`:221`).
- **Never promote a hit that failed `min_score`.** Reordering what came back is
  in scope; changing what comes back is not.

**The cost, stated plainly:** a genuinely dominant file loses one slot to a
weaker hit from another file. That is the trade being bought, and it is the
right one when the question named plural files. It is the wrong one for a
single-topic global-KB search, which is why the `""` bucket rule above is
load-bearing rather than an optimisation.

### Stage 2 — verify

Add a bleed case to `scripts/eval_prompts_video.json` that asserts on
**attribution, not coverage**. The run-3 failure produced correct headings with
wrong contents, so a case checking only "mentions both files" passes it
cleanly — which is exactly how it survived. The case has to be readable as
wrong: a claim that appears in one file's transcript and not the other's.

Then one A-B sweep, `audrey_video` vs `audrey_auto`, same protocol.

⚠️ Read the `Standing gotchas` entry on injected prompts before trusting any
A-B on this corpus.

---

## Tests

In a new `tests/test_kb_result_coverage.py` — the existing `test_kb_query_*`
files are scoped to auth and score floors, and this is neither:

- multi-file hits interleave so each file appears before any file's second hit;
- single-file hits pass through unchanged (identity);
- global-KB-only hits pass through unchanged (identity) — the `""` bucket;
- mixed global + upload hits keep global as one bucket;
- a tail drop at realistic sizes still leaves every file represented — the
  property the whole phase exists to establish, tested against
  `_truncate_payload` rather than assumed.

---

## Verification

1. Deploy, re-run the plural case, read `files=[…]` in the `kb.query` log line.
2. Confirm every file in the corpus that should match appears in the head.
3. Read the paired answers for attribution, not coverage.

---

## Rollback

Pure reorder with no schema change and nothing persisted, so `git revert` is
complete. Only `audrey-ai` needs rebuilding — `tools-server` is untouched.

Stage 0 is independently revertible and worth keeping even if Stage 1 is
abandoned.

---

## What this does NOT fix

**The model can still bleed given balanced material.** This phase changes what
arrives, not what the model does with it. The comprehension half is the
`kb_search` description change from 2026-08-09, which is deployed and
unmeasured — so if Stage 0 shows a balanced split, that description change was
the entire answer and Stage 1 should not be built.

**It does not make truncation fair.** A file whose hits all land in the tail
still loses them; the head guarantee means it loses *some* rather than *all*.
Making the cut itself fair remains possible later, and remains the wrong first
move for the reasons above.

**It does not touch scoping.** Phase 40 §3b established that scoping already
works unprompted, and that finding is unaffected: this phase only reorders
results within a search that was correctly left unscoped.
