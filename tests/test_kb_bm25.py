"""Tests for the BM25 sparse vectoriser (Phase 39).

The tokenizer is a permanent contract — every vector in the index is a
function of it, and changing it invalidates all of them — so the tests pin its
behaviour rather than just its outputs' shape.
"""

from __future__ import annotations

import itertools
import math

from audrey.kb import bm25


class TestTokenize:
    def test_case_is_folded(self):
        assert bm25.tokenize("Fox FOX fox") == ["fox", "fox", "fox"]

    def test_punctuation_splits_and_is_dropped(self):
        assert bm25.tokenize("play, some. baseball!") == ["play", "some", "baseball"]

    def test_apostrophes_stay_inside_a_word(self):
        """Splitting on the apostrophe would make "don't" into "don" + "t",
        and "t" is a term that matches every contraction in the corpus."""
        assert bm25.tokenize("don't") == ["don't"]
        assert bm25.tokenize("it's Jason's") == ["it's", "jason's"]

    def test_digits_are_kept(self):
        """A quote can be a version number or an error code."""
        assert bm25.tokenize("error 502 on v1.18") == ["error", "502", "on", "v1", "18"]

    def test_duplicates_are_preserved(self):
        """Term frequency is half of BM25; deduping here would erase it."""
        assert bm25.tokenize("a a a") == ["a", "a", "a"]

    def test_empty_and_symbol_only_text_give_no_terms(self):
        assert bm25.tokenize("") == []
        assert bm25.tokenize("--- !!! ---") == []


class TestTermIndex:
    def test_the_same_term_always_gets_the_same_index(self):
        """CRC32, not `hash()`. Python randomises string hashing per process,
        so an index written by the worker would not match a query tokenized by
        the API — and the silence would look like a corpus problem."""
        assert bm25.term_index("baseball") == bm25.term_index("baseball")
        assert bm25.term_index("baseball") == 1396649963

    def test_indices_fit_in_uint32(self):
        for word in ("a", "baseball", "supercalifragilistic", "502", "don't"):
            assert 0 <= bm25.term_index(word) < 2**32


class TestDocumentVector:
    def test_indices_are_sorted_and_unique(self):
        """Qdrant rejects a sparse vector with duplicate indices."""
        idx, val = bm25.document_vector("the fox and the hound and the fox")
        assert idx == sorted(idx)
        assert len(idx) == len(set(idx))
        assert len(idx) == len(val)

    def test_empty_text_is_a_legal_empty_vector(self):
        """An image caption or a silent video's transcript. It must be
        representable, not an error — it simply never matches."""
        assert bm25.document_vector("") == ([], [])

    def test_term_frequency_increases_weight(self):
        once = dict(zip(*bm25.document_vector("fox " + "pad " * 20), strict=True))
        twice = dict(zip(*bm25.document_vector("fox fox " + "pad " * 19), strict=True))
        fox = bm25.term_index("fox")
        assert twice[fox] > once[fox]

    def test_term_frequency_saturates(self):
        """The point of `k1`: each further occurrence is worth less than the
        one before it. Without saturation, one keyword-stuffed chunk wins
        every query that mentions it.

        Compared one occurrence at a time — against a multi-step jump the
        later gap is simply the sum of more (smaller) steps and can exceed the
        earlier one while the curve is still saturating perfectly well."""
        fox = bm25.term_index("fox")
        pad = " " + "pad " * 40
        w = [dict(zip(*bm25.document_vector("fox " * n + pad), strict=True))[fox]
             for n in (1, 2, 3, 4)]
        gaps = [b - a for a, b in itertools.pairwise(w)]
        assert gaps == sorted(gaps, reverse=True), gaps

    def test_a_longer_document_scores_its_terms_lower(self):
        """Length normalisation. This is the dilution that started the phase:
        the same phrase must not be worth less simply because a reader would
        call the surrounding paragraph relevant — but it must not be worth
        *more* either, or long chunks win everything."""
        fox = bm25.term_index("fox")
        short = dict(zip(*bm25.document_vector("the quick fox"), strict=True))
        long = dict(zip(*bm25.document_vector("the quick fox " + "filler " * 200),
                        strict=True))
        assert long[fox] < short[fox]

    def test_the_weight_matches_the_bm25_formula(self):
        """Pinned against the arithmetic, not against a previous run — a
        regression here is silent and shifts every ranking in the corpus."""
        text = "fox " * 3 + "pad " * 7  # 10 tokens, tf(fox) = 3
        idx, val = bm25.document_vector(text)
        got = dict(zip(idx, val, strict=True))[bm25.term_index("fox")]
        norm = bm25.K1 * (1 - bm25.B + bm25.B * (10 / bm25.AVG_LEN))
        assert math.isclose(got, 3 * (bm25.K1 + 1) / (3 + norm))


class TestQueryVector:
    def test_every_distinct_term_gets_weight_one(self):
        idx, val = bm25.query_vector("fox fox hound")
        assert val == [1.0, 1.0]
        assert set(idx) == {bm25.term_index("fox"), bm25.term_index("hound")}

    def test_it_matches_the_document_side_indices(self):
        """The two halves must agree on tokenization or nothing ever matches.
        Worth its own test because the failure is total silence, not an
        error."""
        q_idx, _ = bm25.query_vector("Watch us play some Baseball!")
        d_idx, _ = bm25.document_vector("and watch us play some baseball.")
        assert set(q_idx) <= set(d_idx)

    def test_an_empty_query_is_an_empty_vector(self):
        assert bm25.query_vector("!!!") == ([], [])


class TestTermOverlap:
    """The evidence test that replaces a score threshold on the lexical side.
    BM25 scores are corpus-relative; "contains four of the five words asked
    for" means the same thing in every corpus at every size."""

    def test_a_verbatim_quote_scores_one(self):
        assert bm25.term_overlap(
            "and watch us play some baseball",
            "we'd love it if you came out and watch us play some baseball this year",
        ) == 1.0

    def test_an_unrelated_passage_scores_zero(self):
        assert bm25.term_overlap("watch us play baseball", "quarterly revenue figures") == 0.0

    def test_one_incidental_word_scores_low(self):
        """The failure this rule is built to catch. BM25 hands back its top-N
        whether or not the match is good, so this arrives with a real rank."""
        assert bm25.term_overlap(
            "and watch us play some baseball",
            "some notes about the meeting",
        ) == 1 / 6  # 'some' of {and, watch, us, play, some, baseball}

    def test_repeated_query_terms_do_not_inflate_the_fraction(self):
        assert bm25.term_overlap("fox fox fox", "one fox") == 1.0

    def test_an_empty_query_scores_zero_rather_than_dividing_by_zero(self):
        assert bm25.term_overlap("", "anything at all") == 0.0
        assert bm25.term_overlap("!!!", "anything at all") == 0.0
