"""Tests for RRF and the evidence rule that replaces `kb.min_score` (Phase 39).

The rule is the risky part of the phase. The failure it guards against —
2026-07-15, a query the corpus could not answer returning PowerApps and
Forest-Service documents that then read as real sources — was *silent*. It
did not error, it did not look wrong, and it was only caught by reading a
trace. So the tests here spend most of their effort on what must be refused.
"""

from __future__ import annotations

from audrey.kb.fusion import RRF_K, passes_evidence, reciprocal_rank_fusion
from audrey.kb.qdrant import KBHit


def _hit(source: str, text: str = "", *, score: float = 0.0, idx: int = 0) -> KBHit:
    return KBHit(score=score, source=source, kind="text", chunk_idx=idx,
                 text=text, payload={})


class TestReciprocalRankFusion:
    def test_a_document_both_retrievers_rank_beats_one_either_loved(self):
        """The whole reason to fuse. Agreement is the signal — a passage both
        retrievers merely liked is a better answer than one a single retriever
        was certain about."""
        agreed = _hit("agreed.txt")
        dense_only = _hit("dense.txt")
        lexical_only = _hit("lex.txt")

        out = reciprocal_rank_fusion(
            dense=[dense_only, agreed],
            lexical=[lexical_only, agreed],
        )

        assert out[0].hit.source == "agreed.txt"
        assert out[0].retrievers == {"dense", "lexical"}

    def test_the_score_is_the_rrf_formula(self):
        out = reciprocal_rank_fusion(dense=[_hit("a.txt")], lexical=[_hit("a.txt")])
        assert out[0].score == 1.0 / (RRF_K + 1) * 2

    def test_either_list_may_be_empty(self):
        """A retriever that returns nothing must contribute nothing, not drag
        every score toward zero. Half the point of using ranks."""
        assert reciprocal_rank_fusion(dense=[_hit("a.txt")], lexical=[])[0].score == \
            1.0 / (RRF_K + 1)
        assert reciprocal_rank_fusion(dense=[], lexical=[_hit("a.txt")])[0].score == \
            1.0 / (RRF_K + 1)
        assert reciprocal_rank_fusion(dense=[], lexical=[]) == []

    def test_the_same_chunk_from_two_collections_merges_into_one_result(self):
        """A chunk reachable from both the global collection and the user's own
        must not appear twice — the dense-only path already merged these by
        sorting a concatenated list."""
        out = reciprocal_rank_fusion(
            dense=[_hit("shared.txt", idx=3, score=0.9), _hit("shared.txt", idx=3, score=0.8)],
            lexical=[],
        )
        assert len(out) == 1

    def test_different_chunks_of_one_source_stay_separate(self):
        out = reciprocal_rank_fusion(
            dense=[_hit("doc.txt", idx=1), _hit("doc.txt", idx=2)], lexical=[],
        )
        assert len(out) == 2

    def test_the_best_rank_is_the_one_recorded(self):
        out = reciprocal_rank_fusion(
            dense=[_hit("x.txt", idx=9, score=0.4), _hit("x.txt", idx=9, score=0.7)],
            lexical=[],
        )
        assert out[0].dense_rank == 1
        assert out[0].dense_score == 0.4

    def test_ordering_is_deterministic_when_scores_tie(self):
        """Two documents at the same position in both lists would otherwise
        come out in dict-insertion order, which makes results irreproducible
        and tests flaky."""
        a = reciprocal_rank_fusion(
            dense=[_hit("a.txt", score=0.9), _hit("b.txt", score=0.1)], lexical=[],
        )
        assert [h.hit.source for h in a] == ["a.txt", "b.txt"]

    def test_overlap_is_measured_against_the_query(self):
        out = reciprocal_rank_fusion(
            dense=[_hit("a.txt", "we play baseball on sundays")],
            lexical=[],
            query="play baseball",
        )
        assert out[0].overlap == 1.0

    def test_no_query_means_no_overlap_claimed(self):
        """0.0 is "not measured" as well as "no terms shared". The evidence
        rule must therefore never admit on overlap alone without a query,
        which is what `passes_evidence` requires the lexical retriever for."""
        out = reciprocal_rank_fusion(dense=[_hit("a.txt", "anything")], lexical=[])
        assert out[0].overlap == 0.0


class TestEvidenceRule:
    """What must be refused, mostly."""

    def test_a_dense_hit_above_the_floor_is_kept(self):
        """The old rule, unchanged. Nothing about the semantic path's
        protection may regress."""
        out = reciprocal_rank_fusion(dense=[_hit("a.txt", score=0.8)], lexical=[])
        assert passes_evidence(out[0], min_score=0.53, min_overlap=0.5)

    def test_a_dense_hit_below_the_floor_is_refused(self):
        """2026-07-15. The corpus could not answer, so the nearest vectors
        were junk that read as real sources."""
        out = reciprocal_rank_fusion(dense=[_hit("junk.txt", score=0.41)], lexical=[])
        assert not passes_evidence(out[0], min_score=0.53, min_overlap=0.5)

    def test_an_exact_quote_is_kept_despite_a_hopeless_cosine(self):
        """The acceptance case for the entire phase. This chunk scores 0.46
        against a 0.53 floor — the old rule threw it away — but it contains
        every word of the query."""
        quote = "and watch us play some baseball"
        out = reciprocal_rank_fusion(
            dense=[],
            lexical=[_hit("transcript.txt", f"come on out {quote} this year")],
            query=quote,
        )
        assert passes_evidence(out[0], min_score=0.53, min_overlap=0.5)

    def test_one_incidental_shared_word_is_refused(self):
        """The 2026-07-15 failure wearing lexical clothes. BM25 returns its
        top-N whether or not the match is good, so this arrives with a real
        rank and a real score."""
        out = reciprocal_rank_fusion(
            dense=[_hit("offtopic.txt", "some notes about the meeting", score=0.3)],
            lexical=[_hit("offtopic.txt", "some notes about the meeting", score=0.3)],
            query="and watch us play some baseball",
        )
        assert out[0].overlap < 0.5
        assert not passes_evidence(out[0], min_score=0.53, min_overlap=0.5)

    def test_being_in_the_lexical_list_is_not_by_itself_enough(self):
        """Rank is not evidence. A retriever that returns five results returns
        a fifth-best whether or not anything matched."""
        out = reciprocal_rank_fusion(
            dense=[], lexical=[_hit("a.txt", "utterly unrelated prose")],
            query="watch us play baseball",
        )
        assert out[0].lexical_rank == 1
        assert not passes_evidence(out[0], min_score=0.53, min_overlap=0.5)

    def test_a_high_overlap_hit_never_seen_by_the_lexical_retriever_is_refused(self):
        """Overlap alone cannot admit a hit — only the lexical retriever
        vouching for it can. Otherwise a dense hit below the floor that
        happens to share words would slip past the floor it just failed,
        which is the rule quietly disabling itself."""
        out = reciprocal_rank_fusion(
            dense=[_hit("a.txt", "watch us play baseball", score=0.2)],
            lexical=[],
            query="watch us play baseball",
        )
        assert out[0].overlap == 1.0
        assert not passes_evidence(out[0], min_score=0.53, min_overlap=0.5)

    def test_a_zero_floor_keeps_every_dense_hit(self):
        """`kb.min_score` defaults to 0.0 = OFF. A deployment that never tuned
        it must not silently gain a floor from this phase."""
        out = reciprocal_rank_fusion(dense=[_hit("a.txt", score=0.01)], lexical=[])
        assert passes_evidence(out[0], min_score=0.0, min_overlap=0.5)
