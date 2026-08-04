"""Tests for the hybrid query path in `routes/kb.py` (Phase 39).

The route is where the two retrievers, the fusion and the evidence rule meet
real config. The units are tested next door; what these pin is the wiring —
which is where a phase like this actually goes wrong, by turning itself on
when it should not, or by quietly dropping user isolation on the new path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from audrey.kb.qdrant import KBHit
from audrey.routes.kb import _search_text_hybrid


def _hit(source: str, text: str, score: float, idx: int = 0) -> KBHit:
    return KBHit(score=score, source=source, kind="text", chunk_idx=idx,
                 text=text, payload={})


class _FakeQdrant:
    """Records what each retriever was asked for, and from which collection."""

    def __init__(self, *, dense=None, lexical=None, user_dense=None,
                 user_lexical=None, has_user_collection=False):
        self.text_collection = "kb_text"
        self._dense = dense or {}
        self._lexical = lexical or {}
        self._user_dense = user_dense or []
        self._user_lexical = user_lexical or []
        self._has_user = has_user_collection
        self.dense_calls: list[str] = []
        self.lexical_calls: list[tuple[str, str]] = []

    async def collection_exists(self, name: str) -> bool:
        return self._has_user

    async def search_text(self, vec, *, top_k, collection=None):
        target = collection or self.text_collection
        self.dense_calls.append(target)
        return self._user_dense if collection else list(self._dense.values())

    async def search_lexical(self, query, *, top_k, collection=None):
        target = collection or self.text_collection
        self.lexical_calls.append((target, query))
        return self._user_lexical if collection else list(self._lexical.values())


CFG = {"enabled": True, "rrf_k": 60, "min_term_overlap": 0.7}


class TestTheJunkThatShippedOn20260803:
    """Regression tests built from the first live run of the hybrid path.

    All three queries were run against the real corpus with
    `min_term_overlap: 0.5` and no stopword list, and all three returned
    documents that must never have been returned. They are reproduced here
    with the actual sources that came back.
    """

    @pytest.mark.asyncio
    async def test_a_vaccine_query_returns_nothing_from_a_powerapps_corpus(self):
        """The 2026-07-15 incident, reproduced by the rule meant to prevent
        it. `how`, `do` and `work` are three of the query's five terms, so
        every long document cleared a 0.5 threshold while `mrna` and
        `vaccines` appeared nowhere."""
        q = _FakeQdrant(dense={}, lexical={
            1: _hit("/datasets/servicenow/.../README.md",
                    "This will not work unless you do the following steps", 11.39),
            2: _hit("/datasets/powerapps/.../work-with-views.md",
                    "How to work with views and how you do a query", 11.10),
        })

        hits, _ = await _search_text_hybrid(
            q, [0.1], query="how do mRNA vaccines work", top_k=5, user=None,
            min_score=0.53, cfg=CFG)

        assert hits == []

    @pytest.mark.asyncio
    async def test_a_baseball_quote_does_not_drag_in_powerapps_docs(self):
        """`and`, `us`, `some` and `play` are four of six terms, so a document
        about audio controls passed while sharing one real word."""
        quote = "and watch us play some baseball"
        q = _FakeQdrant(
            dense={1: _hit("transcript.txt", f"come out {quote} this year", 0.47)},
            lexical={
                1: _hit("transcript.txt", f"come out {quote} this year", 9.9),
                2: _hit("/datasets/powerapps/.../control-audio-video.md",
                        "Use this control to play some media and let us know", 13.83),
            })

        hits, _ = await _search_text_hybrid(
            q, [0.1], query=quote, top_k=5, user=None, min_score=0.53, cfg=CFG)

        assert [h.source for h in hits] == ["transcript.txt"]

    @pytest.mark.asyncio
    async def test_below_floor_documents_do_not_ride_in_on_function_words(self):
        """The transcript is a real hit at 0.796. The USFS and BJJ documents
        scored 0.43-0.46 — below the 0.53 floor that had refused them for a
        year — and were admitted purely on `he`/`did`/`a`/`when`/`i`."""
        query = "he actually did make a difference when I first started"
        q = _FakeQdrant(
            dense={
                1: _hit("transcript.txt", "he actually did make a difference "
                                          "when I first started here", 0.796),
                2: _hit("/datasets/herbal-medicine/.../index.html",
                        "he did not make a note of it when i first started", 0.463),
            },
            lexical={2: _hit("/datasets/herbal-medicine/.../index.html",
                             "he did not make a note of it when i first started", 8.1)},
        )

        hits, _ = await _search_text_hybrid(
            q, [0.1], query=query, top_k=5, user=None, min_score=0.53, cfg=CFG)

        assert [h.source for h in hits] == ["transcript.txt"]


class TestReportedScore:
    @pytest.mark.asyncio
    async def test_the_score_is_the_fused_rank_not_the_raw_retriever_score(self):
        """Shipped on 2026-08-03 returning a cosine of 0.47 next to a BM25
        score of 13.8 — two scales in one list, ordered by neither, and
        meaningless to anything comparing them."""
        q = _FakeQdrant(
            dense={1: _hit("a.txt", "alpha text", 0.83)},
            lexical={2: _hit("b.txt", "alpha text", 13.83)},
        )

        hits, _ = await _search_text_hybrid(
            q, [0.1], query="alpha text", top_k=5, user=None, min_score=0.53, cfg=CFG)

        assert all(0.0 < h.score < 1.0 for h in hits), [h.score for h in hits]
        assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_an_exact_quote_is_returned_though_its_cosine_fails_the_floor(self):
        """The acceptance case for the phase. Dense finds nothing usable; the
        lexical retriever finds the chunk containing every word of the query."""
        quote = "and watch us play some baseball"
        q = _FakeQdrant(
            dense={1: _hit("other.txt", "unrelated text", 0.41)},
            lexical={1: _hit("transcript.txt", f"come out {quote} this year", 3.2)},
        )

        hits, _ = await _search_text_hybrid(
            q, [0.1], query=quote, top_k=5, user=None, min_score=0.53, cfg=CFG)

        assert [h.source for h in hits] == ["transcript.txt"]

    @pytest.mark.asyncio
    async def test_a_paraphrase_still_works(self):
        """A hybrid that fixes quoting and breaks meaning is a net loss."""
        q = _FakeQdrant(dense={1: _hit("t.txt", "some passage", 0.796)}, lexical={})

        hits, _ = await _search_text_hybrid(
            q, [0.1], query="a paraphrase sharing no words", top_k=5, user=None,
            min_score=0.53, cfg=CFG)

        assert [h.source for h in hits] == ["t.txt"]

    @pytest.mark.asyncio
    async def test_junk_is_still_refused(self):
        """2026-07-15: a query the corpus could not answer returned its
        least-irrelevant documents, which then read as real sources."""
        q = _FakeQdrant(
            dense={1: _hit("junk1.txt", "powerapps servicenow", 0.44),
                   2: _hit("junk2.txt", "forest service memo", 0.42)},
            lexical={1: _hit("junk1.txt", "powerapps servicenow", 0.9)},
        )

        hits, _ = await _search_text_hybrid(
            q, [0.1], query="how do mrna vaccines work", top_k=5, user=None,
            min_score=0.53, cfg=CFG)

        assert hits == []

    @pytest.mark.asyncio
    async def test_a_lexical_hit_sharing_one_word_is_refused(self):
        """BM25 returns a top-N whether or not the match is any good, so this
        arrives with a real rank and a real score."""
        q = _FakeQdrant(
            dense={},
            lexical={1: _hit("notes.txt", "some notes about the meeting", 1.1)},
        )

        hits, _ = await _search_text_hybrid(
            q, [0.1], query="and watch us play some baseball", top_k=5, user=None,
            min_score=0.53, cfg=CFG)

        assert hits == []

    @pytest.mark.asyncio
    async def test_both_retrievers_search_the_users_private_collection(self):
        """The lexical path is a new place to forget user scoping, and the
        failure is one user reading another's uploads."""
        q = _FakeQdrant(has_user_collection=True)

        _, had_user = await _search_text_hybrid(
            q, [0.1], query="anything", top_k=5, user="bart@proton.me",
            min_score=0.53, cfg=CFG)

        assert had_user
        assert q.dense_calls == ["kb_text", "kb_user_text_bart_proton_me"]
        assert [c for c, _ in q.lexical_calls] == [
            "kb_text", "kb_user_text_bart_proton_me"]

    @pytest.mark.asyncio
    async def test_no_user_means_only_the_global_collection(self):
        q = _FakeQdrant(has_user_collection=True)

        _, had_user = await _search_text_hybrid(
            q, [0.1], query="anything", top_k=5, user=None, min_score=0.53, cfg=CFG)

        assert not had_user
        assert q.dense_calls == ["kb_text"]

    @pytest.mark.asyncio
    async def test_an_empty_lexical_index_degrades_to_dense_only(self):
        """A collection that predates the migration returns nothing lexical.
        That must answer, not fail — it is the state every deploy starts in."""
        q = _FakeQdrant(dense={1: _hit("a.txt", "text", 0.8)}, lexical={})

        hits, _ = await _search_text_hybrid(
            q, [0.1], query="whatever", top_k=5, user=None, min_score=0.53, cfg=CFG)

        assert [h.source for h in hits] == ["a.txt"]

    @pytest.mark.asyncio
    async def test_top_k_is_respected_after_filtering(self):
        q = _FakeQdrant(dense={
            i: _hit(f"d{i}.txt", "text", 0.9 - i / 100) for i in range(10)})

        hits, _ = await _search_text_hybrid(
            q, [0.1], query="whatever", top_k=3, user=None, min_score=0.53, cfg=CFG)

        assert len(hits) == 3

    @pytest.mark.asyncio
    async def test_the_query_reaches_the_lexical_retriever_verbatim(self):
        """It is tokenized there, not here. A route that pre-processed the
        query would desynchronise it from how documents were indexed."""
        q = _FakeQdrant()

        await _search_text_hybrid(
            q, [0.1], query="Watch Us Play!", top_k=5, user=None,
            min_score=0.53, cfg=CFG)

        assert q.lexical_calls[0][1] == "Watch Us Play!"


class TestConfigWiring:
    def test_hybrid_is_off_by_default(self):
        """It must stay off until `scripts/migrate_bm25.py` has run — an
        un-migrated collection has no lexical index to search."""
        from audrey.routes.kb import _hybrid_cfg
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            cfg=SimpleNamespace(raw={"kb": {}}))))
        assert not _hybrid_cfg(request).get("enabled")

    def test_a_missing_cfg_does_not_raise(self):
        from audrey.routes.kb import _hybrid_cfg
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        assert _hybrid_cfg(request) == {}

    def test_the_shipped_config_enables_hybrid_with_sane_knobs(self):
        """Pinned in both directions on purpose.

        `enabled` was False until the migration ran on 2026-08-03; it is True
        now because every text collection the query path reads has a bm25
        vector. A flip in either direction is a real decision — off means
        falling back to dense-only and losing exact quoting, on without a
        migration means a lexical retriever with nothing in it — so neither
        should happen without this test objecting."""
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.yaml").read_text())
        hybrid = cfg["kb"]["hybrid"]
        assert hybrid["enabled"] is True
        assert hybrid["rrf_k"] == 60
        assert 0.0 < hybrid["min_term_overlap"] <= 1.0
