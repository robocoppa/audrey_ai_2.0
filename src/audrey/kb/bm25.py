"""BM25 sparse vectors for the lexical half of hybrid retrieval (Phase 39).

The dense retriever finds text that *means* what you asked. This finds text
that *says* it. A transcript is the artifact people quote verbatim, and phase
35 measured the gap: a 10-word paraphrase returns its chunk at 0.796 while a
6-word phrase appearing word-for-word in the same transcript returns nothing.

## Why there is no model here

BM25 is arithmetic over token counts, not a learned function, so this module
is stdlib only and calls nothing over the network. That is what makes the
backfill cheap — every value below is computed from `payload.text`, which is
already stored on every point, so no source file is re-read and no embedder is
ever invoked.

## Why the IDF half is missing on purpose

The full BM25 score is `idf(term) * tf_component(term, doc)`. Only the second
factor is computed here. The first is Qdrant's job:
`SparseVectorParams(modifier=Modifier.IDF)` makes the server compute inverse
document frequency from the collection's own statistics at query time.

That split matters more than it looks. IDF depends on the whole corpus, so
computing it here would mean every existing vector became subtly wrong the
moment a new document was ingested — a retrieval index that silently rots as
it is used. Handing the corpus-dependent half to the thing that owns the
corpus means a vector written today is still correct after a thousand more
uploads.

## Why there is no stopword list

Because `Modifier.IDF` makes one redundant, and a hand-maintained list is
exactly the failure the phase plan warns about: "BM25 is tokenizer-dependent —
stemming, stopwords and casing all change what *exact* means."

A term appearing in every document gets an IDF near zero and contributes
nothing to the score, which is what a stopword list is *for*. Doing it by list
as well would add a second, worse mechanism: someone would have to decide
whether "us" is a stopword, and a query made entirely of common words — "to be
or not to be" — would tokenize to nothing and become permanently unquotable.
Letting IDF do it costs a few more indices per vector and cannot produce that
failure.

## Why there is no stemming

Declined deliberately. Stemming widens recall by conflating "play" with
"playing", which is the *dense* retriever's job and it is already good at it.
On this side it only blurs the thing being asked for: someone quoting a
sentence wants that sentence.

It also has a cost that is easy to miss — the tokenizer is a permanent
contract. Every indexed vector is a function of it, so changing tokenization
later invalidates the entire index and requires a full re-run of the backfill.
Fewer moving parts in this function is worth real money later.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

# Standard BM25 constants. `k1` sets how fast term frequency saturates — a
# term appearing 10 times is worth a little more than 5 times, not twice as
# much. `b` sets how hard long documents are penalised.
K1 = 1.2
B = 0.75

# Length normalisation needs a corpus average, which a single chunk cannot
# know. A fixed value is the standard resolution (fastembed uses the same
# trick) and works because only the *ratio* to this number matters: it decides
# how steeply a longer chunk is discounted, not the absolute score, and every
# document in a collection is measured against the same constant.
#
# 256 sits between the two chunk sizes actually in use — 250 tokens for
# transcripts, 1000 for documents — so transcript chunks are scored at roughly
# neutral length and a full document chunk is discounted for being four times
# the size. That discount is wanted: length dilution is the failure that
# started this phase.
AVG_LEN = 256

# Letters and digits, with apostrophes kept inside a word so "don't" stays one
# token rather than becoming "don" + "t". Everything else splits.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into terms, in order, with duplicates.

    Order is irrelevant to BM25 but duplicates are not — term frequency is
    half the formula.
    """
    return _TOKEN_RE.findall(text.lower())


def term_index(token: str) -> int:
    """Map a term to the uint32 index Qdrant stores it under.

    CRC32 rather than `hash()`, which is randomised per process for strings:
    an index built by one worker would not match a query tokenized by another,
    and the resulting silence would look like a corpus problem rather than a
    hashing one.

    Two terms colliding on the same index is possible and harmless at this
    scale — a collision makes one rare term score as though it were another,
    across a 2^32 space and a vocabulary in the low hundreds of thousands.
    """
    return zlib.crc32(token.encode("utf-8"))


def document_vector(text: str) -> tuple[list[int], list[float]]:
    """The BM25 term-frequency component of `text`, as `(indices, values)`.

    Empty text gives two empty lists, which is a legal sparse vector — a point
    with no terms is simply never returned by a lexical search, which is
    correct for an image caption or an empty transcript.
    """
    tokens = tokenize(text)
    if not tokens:
        return [], []

    # Normalising by the document's own length is what stops a long chunk from
    # outscoring a short one purely by containing more words.
    norm = K1 * (1.0 - B + B * (len(tokens) / AVG_LEN))

    # Keyed by index, not by token, so a hash collision merges into one entry
    # instead of producing a duplicate index — Qdrant rejects those.
    weights: dict[int, float] = {}
    for token, tf in Counter(tokens).items():
        idx = term_index(token)
        weights[idx] = weights.get(idx, 0.0) + tf * (K1 + 1.0) / (tf + norm)
    indices = sorted(weights)
    return indices, [weights[i] for i in indices]


def query_vector(text: str) -> tuple[list[int], list[float]]:
    """The query side of the dot product: every distinct term, weight 1.0.

    Query term frequency is conventionally ignored — asking for "fox fox" does
    not mean you want it twice as much. The IDF weighting that makes rare
    terms matter more than common ones is applied by Qdrant against both
    sides, so it is deliberately absent here too.
    """
    indices = sorted({term_index(t) for t in tokenize(text)})
    return indices, [1.0] * len(indices)


def term_overlap(query: str, text: str) -> float:
    """Fraction of the query's distinct terms that appear in `text`, 0.0-1.0.

    This is the evidence test that replaces a score threshold on the lexical
    side. BM25 scores are unbounded and corpus-relative, so there is no
    constant that means "this match is real" — but "the document contains four
    of the query's five words" means the same thing in every corpus, at every
    size, forever.

    An empty query has no evidence to offer and scores 0.0 rather than
    dividing by zero.
    """
    q = {term_index(t) for t in tokenize(query)}
    if not q:
        return 0.0
    d = {term_index(t) for t in tokenize(text)}
    return len(q & d) / len(q)


__all__ = [
    "AVG_LEN", "B", "K1",
    "document_vector", "query_vector", "term_index", "term_overlap", "tokenize",
]
