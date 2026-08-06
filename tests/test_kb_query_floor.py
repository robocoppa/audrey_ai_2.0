"""Hermetic tests for the kb_search cosine floor (`_search_text_merged` min_score).

The KB always returns its top_k NEAREST vectors regardless of distance, so an
off-domain query gets the least-irrelevant junk (2026-07-15 trace: a vaccine
query against a geology/botany KB returned PowerApps / Forest-Service docs). The
floor turns "nothing relevant" into an empty result. Default 0.0 = OFF.

Fakes the qdrant surface `_search_text_merged` uses (`search_text`,
`collection_exists`) so tests run offline. No embeddings, no network.
"""

from __future__ import annotations

from audrey.kb.qdrant import KBHit, SearchScope
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


# ─── the floor stands down when the user named the file (2026-08-06) ────
#
# The floor defends a GLOBAL search from the least-irrelevant thing in the
# corpus. A search scoped to named files has no such corpus: every candidate is
# a chunk of a file the user chose, so the worst hit available is the least
# relevant part of the right document. §3b measured what the floor costs there
# — `top_k=5 -> 2 hits` on a video entirely about the subject asked about.


def _scope(*file_ids: str, artifact: str = "") -> SearchScope:
    return SearchScope(file_ids=list(file_ids), artifact=artifact)


USER = "a@b.c"


class _FakeUserQdrant(_FakeQdrant):
    """A user who HAS a private collection.

    Required for the scoped cases: a scope narrowed to one user's files skips
    the global collection entirely (`_scoped_to_one_users_files`), so a fake
    without a user collection has nothing to search and returns nothing —
    which would make these tests pass for the wrong reason.
    """

    def __init__(self, hits: list[KBHit]):
        super().__init__(hits)
        self.asked: list[int] = []

    async def search_text(self, vec, *, top_k, collection=None, scope=None):
        self.asked.append(top_k)
        return list(self._hits[:top_k])

    async def collection_exists(self, name: str) -> bool:
        return True


async def test_a_scoped_search_keeps_its_low_scorers():
    q = _FakeUserQdrant([_hit(0.72, "the-bit-they-quoted"), _hit(0.20, "later-in-the-same-video")])
    hits, had_user = await _search_text_merged(
        q, [0.0], top_k=5, user=USER, min_score=0.35, scope=_scope("vid-1"),
    )
    assert had_user
    # 0.20 is a weak match for the question and still a chunk of the file the
    # user pointed at. Cutting it answers a question about that video from less
    # of that video.
    assert [h.source for h in hits] == ["the-bit-they-quoted", "later-in-the-same-video"]


async def test_an_unscoped_search_is_untouched():
    q = _FakeQdrant([_hit(0.72, "on-topic"), _hit(0.20, "junk")])
    hits, _ = await _search_text_merged(q, [0.0], top_k=5, user=None, min_score=0.35)
    assert [h.source for h in hits] == ["on-topic"]


async def test_artifact_scoping_alone_does_not_stand_the_floor_down():
    # `artifact=transcript` narrows to a KIND of chunk across every file the
    # user has. That is still a corpus-wide search, and the 2026-07-15 failure
    # is still available to it — a vaccine query would simply return the
    # least-irrelevant *transcript*.
    q = _FakeQdrant([_hit(0.72, "on-topic"), _hit(0.20, "junk")])
    hits, _ = await _search_text_merged(
        q, [0.0], top_k=5, user=None, min_score=0.35,
        scope=SearchScope(artifact="transcript"),
    )
    assert [h.source for h in hits] == ["on-topic"]


async def test_a_negative_cosine_survives_a_scoped_search():
    # Why the relaxed floor is -inf and not 0.0. A cosine can be negative, so
    # 0.0 would still be a filter — a small one, applied for no reason, in the
    # one case where filtering is what we are trying not to do.
    q = _FakeUserQdrant([_hit(0.40, "a"), _hit(-0.05, "b")])
    hits, _ = await _search_text_merged(
        q, [0.0], top_k=5, user=USER, min_score=0.35, scope=_scope("vid-1"),
    )
    assert [h.source for h in hits] == ["a", "b"]


async def test_the_scoped_path_does_not_over_fetch():
    # The over-fetch exists to survive the floor rejecting near neighbours.
    # With no floor there is nothing to survive, and asking Qdrant for 4x the
    # rows is a bigger scan for results that would be discarded at the cap.
    q = _FakeUserQdrant([_hit(0.9, "a")])
    await _search_text_merged(
        q, [0.0], top_k=5, user=USER, min_score=0.35, scope=_scope("vid-1"),
    )
    assert q.asked == [5]

    unscoped = _FakeUserQdrant([_hit(0.9, "a")])
    await _search_text_merged(unscoped, [0.0], top_k=5, user=USER, min_score=0.35)
    # Unchanged where the floor still applies: 4x, capped at 40.
    assert unscoped.asked == [20, 20]
