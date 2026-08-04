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


CFG = {"enabled": True, "rrf_k": 60, "min_term_overlap": 0.5}


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

    def test_the_shipped_config_has_hybrid_off_with_documented_knobs(self):
        """The deployed default. Turning this on before the migration is
        pointless; turning it on *by accident* would be worse."""
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.yaml").read_text())
        hybrid = cfg["kb"]["hybrid"]
        assert hybrid["enabled"] is False
        assert hybrid["rrf_k"] == 60
        assert 0.0 < hybrid["min_term_overlap"] <= 1.0
