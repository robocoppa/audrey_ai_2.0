"""Reciprocal rank fusion, and the junk rule that replaces `kb.min_score`.

## Why the merge is here and not in Qdrant

The phase 39 plan chose server-side fusion — `query_points(prefetch=[...],
query=FusionQuery(fusion=Fusion.RRF))` — on the reasoning that it needs no
merge function, no normalisation, and no `k` constant of our own. All of that
is true and it is still the better call for ranking alone.

It was reversed for one reason: **a fused score cannot be audited.** Qdrant
returns the RRF value and discards what each retriever thought, and the same
plan requires a junk rule stated in terms of evidence rather than magnitude —
which needs exactly the numbers fusion throws away. A hit scoring 0.0163 might
be a passage both retrievers ranked second, or the least-irrelevant document
in a corpus that cannot answer the question at all. Those must be treated
differently and after fusion they are indistinguishable.

The cost of merging here is a second Qdrant round trip, issued concurrently
with the first, on a path that already waits on an embedding call. The benefit
is that the 2026-07-15 incident stays defended by the mechanism that was
proven to defend it.

## Why rank and not score

BM25 is unbounded and corpus-relative; cosine is bounded and absolute. No
constant converts one to the other, and any weighted sum of the two silently
re-tunes itself as the corpus grows. RRF reads only a document's *position* in
each list:

    score(d) = sum over retrievers of  1 / (k + rank(d))

No normalisation, no calibration, and a retriever that returns nothing simply
contributes nothing instead of dragging every score toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from audrey.kb.bm25 import term_overlap
from audrey.kb.qdrant import KBHit

# The rank-fusion constant. 60 is the value from the original RRF paper and
# the one Qdrant itself defaults to. It flattens the difference between the
# top few positions — rank 1 scores 1/61 and rank 2 scores 1/62 — so a
# document both retrievers merely liked outranks one that a single retriever
# loved. That is the intended behaviour: agreement is the signal.
RRF_K = 60


@dataclass(slots=True)
class FusedHit:
    """A hit plus the evidence that admitted it.

    The evidence fields are the reason this module exists. They are what a
    fused score alone cannot tell you, and they are carried through so the
    junk rule can be applied *after* ranking rather than being folded into it.
    """

    hit: KBHit
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    overlap: float = 0.0
    retrievers: set[str] = field(default_factory=set)


def _key(hit: KBHit) -> tuple[str, int]:
    """Identity of a chunk across two result lists.

    `(source, chunk_idx)` rather than the Qdrant point id, because the same
    chunk is reachable from the global collection and a user's own — merging
    those into one result is wanted, and it is what the dense-only path
    already did by sorting a concatenated list.
    """
    return (hit.source, hit.chunk_idx)


def reciprocal_rank_fusion(
    dense: list[KBHit],
    lexical: list[KBHit],
    *,
    rrf_k: int = RRF_K,
    query: str = "",
) -> list[FusedHit]:
    """Merge two ranked lists into one, keeping each retriever's evidence.

    Both inputs must already be ordered best-first; their positions are the
    only thing read from them. Either may be empty.
    """
    fused: dict[tuple[str, int], FusedHit] = {}

    for rank, hit in enumerate(dense, start=1):
        entry = fused.setdefault(_key(hit), FusedHit(hit=hit, score=0.0))
        entry.score += 1.0 / (rrf_k + rank)
        # Ranks can repeat across the global and per-user collections; the
        # better position is the honest one to report.
        if entry.dense_rank is None or rank < entry.dense_rank:
            entry.dense_rank = rank
            entry.dense_score = hit.score
        entry.retrievers.add("dense")

    for rank, hit in enumerate(lexical, start=1):
        entry = fused.setdefault(_key(hit), FusedHit(hit=hit, score=0.0))
        entry.score += 1.0 / (rrf_k + rank)
        if entry.lexical_rank is None or rank < entry.lexical_rank:
            entry.lexical_rank = rank
        entry.retrievers.add("lexical")

    for entry in fused.values():
        entry.overlap = term_overlap(query, entry.hit.text) if query else 0.0

    # Ties broken by dense score so the order is deterministic. Two documents
    # at the same position in both lists is otherwise dict-insertion order,
    # which makes tests flaky and results irreproducible.
    return sorted(
        fused.values(),
        key=lambda e: (e.score, e.dense_score or 0.0),
        reverse=True,
    )


def passes_evidence(
    entry: FusedHit, *, min_score: float, min_overlap: float,
) -> bool:
    """Whether a fused hit has earned its place in the results.

    The rule this replaces was `cosine >= kb.min_score`, and it exists because
    of the 2026-07-15 trace run: a vaccine query the corpus could not answer
    returned PowerApps and Forest-Service documents, which then read to the
    researcher as real sources. An empty result is handled gracefully
    downstream; an off-topic one is not.

    A fused score cannot express that rule, so the rule is applied to the
    evidence instead. A hit stays if **either** retriever independently
    vouches for it:

    - **Dense**: its cosine clears `min_score`, exactly as before. Nothing
      about the semantic path's protection changes.
    - **Lexical**: the chunk contains at least `min_overlap` of the query's
      distinct terms. This is what makes an exact quote retrievable even
      though its cosine is nowhere near the floor — the case the whole phase
      exists for — while a document sharing one incidental word with the query
      still gets nothing.

    Appearing in the lexical list is deliberately *not* sufficient on its own.
    BM25 returns its top-N whether or not the match is any good, so a
    one-common-word overlap arrives with a real rank and a real score. That is
    the same failure as the 2026-07-15 incident wearing different clothes.
    """
    if entry.dense_score is not None and entry.dense_score >= min_score:
        return True
    return "lexical" in entry.retrievers and entry.overlap >= min_overlap


__all__ = ["RRF_K", "FusedHit", "passes_evidence", "reciprocal_rank_fusion"]
