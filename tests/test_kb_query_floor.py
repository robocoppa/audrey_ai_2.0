"""Hermetic tests for the kb_search cosine floor (`_search_text_merged` min_score).

The KB always returns its top_k NEAREST vectors regardless of distance, so an
off-domain query gets the least-irrelevant junk (2026-07-15 trace: a vaccine
query against a geology/botany KB returned PowerApps / Forest-Service docs). The
floor turns "nothing relevant" into an empty result. Default 0.0 = OFF.

Fakes the qdrant surface `_search_text_merged` uses (`search_text`,
`collection_exists`) so tests run offline. No embeddings, no network.
"""

from __future__ import annotations

from audrey.kb.qdrant import KBHit
from audrey.routes.kb import _search_text_merged


def _hit(score: float, source: str) -> KBHit:
    return KBHit(score=score, source=source, kind="text", chunk_idx=0, text=source, payload={})


class _FakeQdrant:
    """Returns a fixed hit list from search_text; no per-user collection."""

    def __init__(self, hits: list[KBHit]):
        self._hits = hits

    async def search_text(self, vec, *, top_k, collection=None):
        return list(self._hits[:top_k])

    async def collection_exists(self, name: str) -> bool:
        return False


async def test_floor_drops_below_threshold_junk():
    # On-domain miss: every hit is far (low cosine). With a floor, all are cut →
    # empty result, which the researcher handles gracefully (no junk injected).
    q = _FakeQdrant([_hit(0.28, "powerapps"), _hit(0.24, "servicenow"), _hit(0.19, "forest-service")])
    hits, _ = await _search_text_merged(q, [0.0], top_k=5, user=None, min_score=0.35)
    assert hits == []


async def test_floor_keeps_above_threshold_hits():
    q = _FakeQdrant([_hit(0.72, "on-topic"), _hit(0.20, "junk")])
    hits, _ = await _search_text_merged(q, [0.0], top_k=5, user=None, min_score=0.35)
    assert [h.source for h in hits] == ["on-topic"]


async def test_default_floor_zero_keeps_everything():
    # Default 0.0 = OFF: no behavior change from before the floor existed.
    q = _FakeQdrant([_hit(0.28, "a"), _hit(0.05, "b")])
    hits, _ = await _search_text_merged(q, [0.0], top_k=5, user=None)
    assert [h.source for h in hits] == ["a", "b"]


async def test_floor_applied_before_top_k_cut():
    # A below-floor hit must not consume a top_k slot. Two below-floor hits are
    # returned by qdrant AHEAD of one above-floor hit. With top_k=1, a naive
    # "cut to top_k then filter" would take the first (below-floor) hit and then
    # drop it → empty. Correct order (filter → sort → cap) keeps "real".
    q = _FakeQdrant([_hit(0.30, "junk-1"), _hit(0.28, "junk-2"), _hit(0.55, "real")])
    hits, _ = await _search_text_merged(q, [0.0], top_k=1, user=None, min_score=0.4)
    assert [h.source for h in hits] == ["real"]
