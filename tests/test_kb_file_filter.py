"""Scoping a KB search to one uploaded file (Phase 40, step 3).

The filter itself is three lines of Qdrant condition. What is worth pinning is
everything around it, because every failure mode here produces a *plausible*
answer rather than an error:

  - **The lexical side.** Phase 39 made this path hybrid. A filter applied to
    the dense retriever only leaves BM25 returning chunks from every other
    file, reciprocal-rank fusion interleaves the two, and the result is a
    confident answer sourced partly from the wrong video. Nothing logs, and a
    dense-only filter passes every other test in this suite — which is exactly
    why `TestBothRetrievers` exists.
  - **A filename that matches nothing.** Must return no hits. The tempting
    implementation — build no filter when resolution fails — silently widens
    a scoped question into a full-corpus one.
  - **Resolution accuracy.** Scoping to the wrong file answers confidently
    from the wrong source, which is worse than not scoping at all. Matching is
    case-insensitive but otherwise exact; near-misses must miss.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.kb import qdrant as qdrant_mod
from audrey.kb.qdrant import KBHit, QdrantKB, SearchScope, _as_filter
from audrey.kb.uploads_db import UploadsDB
from audrey.routes.kb import (
    _exclude_deleted_private_files,
    _search_text_hybrid,
    _search_text_merged,
)
from audrey.routes.kb import router as kb_router

CFG = {"enabled": True, "rrf_k": 60, "min_term_overlap": 0.0}
SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)


def _hit(source: str, text: str, score: float, idx: int = 0, **payload) -> KBHit:
    return KBHit(score=score, source=source, kind="text", chunk_idx=idx,
                 text=text, payload=payload)


class _ScopeRecordingQdrant:
    """Records the scope each retriever received, per collection."""

    def __init__(self, *, user_dense=None, user_lexical=None, has_user=True):
        self.text_collection = "kb_text"
        self._user_dense = user_dense or []
        self._user_lexical = user_lexical or []
        self._has_user = has_user
        self.dense: list[tuple[str, object]] = []
        self.lexical: list[tuple[str, object]] = []

    async def collection_exists(self, name: str) -> bool:
        return self._has_user

    async def search_text(self, vec, *, top_k, collection=None, scope=None):
        self.dense.append((collection or self.text_collection, scope))
        return list(self._user_dense) if collection else []

    async def search_lexical(self, query, *, top_k, collection=None, scope=None):
        self.lexical.append((collection or self.text_collection, scope))
        return list(self._user_lexical) if collection else []

    async def search_hybrid(self, vec, query, *, top_k, collection=None, scope=None):
        return (
            await self.search_text(vec, top_k=top_k, collection=collection, scope=scope),
            await self.search_lexical(query, top_k=top_k, collection=collection, scope=scope),
        )


# ─── The filter object ────────────────────────────────────────────────


class TestSearchScope:
    def test_an_unset_scope_produces_no_filter(self):
        assert _as_filter(None) is None
        assert _as_filter(SearchScope()) is None

    def test_a_scope_matching_no_files_cannot_be_constructed(self):
        """The distinction the whole step turns on, enforced rather than
        documented.

        `None` means "search everything". `[]` would mean "a file was named
        and does not exist" — a state with no safe downstream representation,
        since an empty `MatchAny` can degrade to matching everything. Rather
        than a sentinel every consumer must remember to check, it is refused
        at construction: a caller whose lookup found nothing must not search.
        """
        with pytest.raises(ValueError, match="must not search"):
            SearchScope(file_ids=[])

        assert SearchScope(file_ids=["f1"]).file_ids == ["f1"]
        assert SearchScope().file_ids is None

    def test_a_file_scope_filters_on_file_id(self):
        flt = _as_filter(SearchScope(file_ids=["f1", "f2"]))
        assert flt is not None
        keys = [c.key for c in flt.must]
        assert keys == ["file_id"]
        assert flt.must[0].match.any == ["f1", "f2"]

    def test_an_artifact_scope_adds_a_second_condition(self):
        flt = _as_filter(SearchScope(file_ids=["f1"], artifact="visual"))
        assert [c.key for c in flt.must] == ["file_id", "artifact"]
        assert flt.must[1].match.value == "visual"

    def test_an_artifact_alone_is_a_valid_scope(self):
        """"What was on screen" across every video is a coherent question."""
        flt = _as_filter(SearchScope(artifact="transcript"))
        assert [c.key for c in flt.must] == ["artifact"]

    def test_file_artifact_and_user_filters_compose(self):
        flt = _as_filter(SearchScope(
            file_ids=["f1"], artifact="transcript", user="alice@example.com",
        ))

        assert [c.key for c in flt.must] == ["file_id", "artifact", "user"]
        assert flt.must[2].match.value == "alice@example.com"

    def test_deleted_file_ids_are_excluded_from_private_reads(self):
        flt = _as_filter(SearchScope(
            user="alice@example.com",
            excluded_file_ids=["deleted-1", "deleted-2"],
        ))

        assert [condition.key for condition in flt.must] == ["user"]
        assert [condition.key for condition in flt.must_not] == ["file_id"]
        assert flt.must_not[0].match.any == ["deleted-1", "deleted-2"]


async def test_private_search_adds_authenticated_users_deletion_tombstones():
    class _DeletionIndex:
        async def file_deletion_ids(self, user: str) -> set[str]:
            assert user == "user-1"
            return {"deleted-2", "deleted-1"}

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(uploads_db=_DeletionIndex())),
    )
    original = SearchScope(file_ids=["kept"], artifact="transcript")

    scope = await _exclude_deleted_private_files(
        request, user="user-1", scope=original,
    )

    assert scope == SearchScope(
        file_ids=["kept"],
        artifact="transcript",
        excluded_file_ids=["deleted-1", "deleted-2"],
    )
    assert original.excluded_file_ids is None


# ─── The trap: both retrievers, or neither ────────────────────────────


class TestFanOut:
    """`QdrantKB.search_hybrid` is what makes a dense-only filter impossible
    to express. These test the real class, not a fake, because the guarantee
    lives in it rather than in the route that calls it."""

    @pytest.mark.asyncio
    async def test_one_scope_in_reaches_both_retrievers(self, monkeypatch):
        fake_client = MagicMock()
        monkeypatch.setattr(qdrant_mod, "QdrantClient", lambda **_: fake_client)
        kb = QdrantKB(host="x", port=0)
        seen: dict[str, object] = {}

        async def _dense(vector, *, top_k, collection=None, scope=None):
            seen["dense"] = scope
            return []

        async def _lexical(query, *, top_k, collection=None, scope=None):
            seen["lexical"] = scope
            return []

        monkeypatch.setattr(kb, "search_text", _dense)
        monkeypatch.setattr(kb, "search_lexical", _lexical)
        scope = SearchScope(file_ids=["f1"], artifact="transcript")

        await kb.search_hybrid([0.1], "handover", top_k=5, scope=scope)

        assert seen["dense"] is scope
        assert seen["lexical"] is scope

    @pytest.mark.asyncio
    async def test_it_returns_the_two_lists_unfused(self, monkeypatch):
        """Fusion and the evidence rule are retrieval policy and stay with the
        route that owns their config — the storage wrapper only fans out."""
        fake_client = MagicMock()
        monkeypatch.setattr(qdrant_mod, "QdrantClient", lambda **_: fake_client)
        kb = QdrantKB(host="x", port=0)

        async def _dense(vector, *, top_k, collection=None, scope=None):
            return [_hit("d.txt", "dense", 0.9)]

        async def _lexical(query, *, top_k, collection=None, scope=None):
            return [_hit("l.txt", "lexical", 12.0)]

        monkeypatch.setattr(kb, "search_text", _dense)
        monkeypatch.setattr(kb, "search_lexical", _lexical)

        dense, lexical = await kb.search_hybrid([0.1], "q", top_k=5)

        assert [h.source for h in dense] == ["d.txt"]
        assert [h.source for h in lexical] == ["l.txt"]


class TestBothRetrievers:
    @pytest.mark.asyncio
    async def test_the_scope_reaches_the_lexical_side_too(self):
        """The one that catches a dense-only filter.

        Phase 39's plan named this as the failure mode nobody would notice,
        because the fused result looks almost right.
        """
        q = _ScopeRecordingQdrant()
        scope = SearchScope(file_ids=["f1"])

        await _search_text_hybrid(
            q, [0.1], query="handover", top_k=5, user="alice@example.com",
            min_score=0.0, cfg=CFG, scope=scope,
        )

        expected = SearchScope(file_ids=["f1"], user="alice@example.com")
        assert [s for _, s in q.dense] == [expected]
        assert [s for _, s in q.lexical] == [expected]

    @pytest.mark.asyncio
    async def test_both_sides_get_the_identical_scope_object(self):
        """Not merely 'a filter each' — the same one. Two separately-built
        filters are how the two sides drift apart later."""
        q = _ScopeRecordingQdrant()
        scope = SearchScope(file_ids=["f1"], artifact="transcript")

        await _search_text_hybrid(
            q, [0.1], query="handover", top_k=5, user="alice@example.com",
            min_score=0.0, cfg=CFG, scope=scope,
        )

        private_scope = q.dense[0][1]
        assert private_scope is q.lexical[0][1]
        assert private_scope is not scope
        assert private_scope == SearchScope(
            file_ids=["f1"], artifact="transcript", user="alice@example.com",
        )

    @pytest.mark.asyncio
    async def test_an_unscoped_search_only_scopes_the_private_collection(self):
        """Global reads stay broad; private reads always enforce ownership."""
        q = _ScopeRecordingQdrant()

        await _search_text_hybrid(
            q, [0.1], query="handover", top_k=5, user="alice@example.com",
            min_score=0.0, cfg=CFG,
        )

        assert [c for c, _ in q.dense] == ["kb_text", "kb_user_text_alice_example_com"]
        assert [c for c, _ in q.lexical] == ["kb_text", "kb_user_text_alice_example_com"]
        assert q.dense[0][1] is None
        assert q.lexical[0][1] is None
        assert q.dense[1][1] is q.lexical[1][1]
        assert q.dense[1][1] == SearchScope(user="alice@example.com")

    @pytest.mark.asyncio
    async def test_a_file_scope_skips_the_global_collection(self):
        """An upload's chunks only ever live in the user's collection, so the
        global search under a file filter is a guaranteed-empty round trip."""
        q = _ScopeRecordingQdrant()

        await _search_text_hybrid(
            q, [0.1], query="handover", top_k=5, user="alice@example.com",
            min_score=0.0, cfg=CFG, scope=SearchScope(file_ids=["f1"]),
        )

        assert [c for c, _ in q.dense] == ["kb_user_text_alice_example_com"]
        assert [c for c, _ in q.lexical] == ["kb_user_text_alice_example_com"]
        assert q.dense[0][1] is q.lexical[0][1]
        assert q.dense[0][1] == SearchScope(
            file_ids=["f1"], user="alice@example.com",
        )

    @pytest.mark.asyncio
    async def test_the_non_hybrid_path_is_scoped_as_well(self):
        """`kb.hybrid.enabled` is still false by default on some deployments,
        so the merged path cannot be left behind."""
        q = _ScopeRecordingQdrant()
        scope = SearchScope(file_ids=["f1"])

        await _search_text_merged(
            q, [0.1], top_k=5, user="alice@example.com", min_score=0.0, scope=scope,
        )

        assert [c for c, _ in q.dense] == ["kb_user_text_alice_example_com"]
        assert q.dense[0][1] is not scope
        assert q.dense[0][1] == SearchScope(
            file_ids=["f1"], user="alice@example.com",
        )


# ─── Resolution, end to end through the route ─────────────────────────


@pytest.fixture
def db(tmp_path: Path) -> UploadsDB:
    return UploadsDB(tmp_path / "uploads.sqlite")


async def _add(db: UploadsDB, file_id: str, *, user: str, filename: str) -> None:
    await db.record_upload(
        file_id=file_id, user=user, filename=filename, mime="video/mp4",
        bytes_=1024, kind="video", collection="", chunks=0,
        uploaded_at="2026-08-01T00:00:00+00:00", status="ready",
    )


def _build_app(db: UploadsDB, monkeypatch) -> tuple[FastAPI, dict]:
    """Mount the kb router over a fake retriever; capture the scope built."""
    captured: dict = {}
    from audrey.routes import kb as kb_module

    async def _fake_hybrid(qdrant, vec, *, query, top_k, user, min_score, cfg, scope=None):
        captured["scope"] = scope
        captured["user"] = user
        return ([_hit("/u/f1.txt", "the handover", 0.9,
                      filename="standup.mp4", artifact="transcript")], True)

    monkeypatch.setattr(kb_module, "_search_text_hybrid", _fake_hybrid)

    app = FastAPI()
    app.include_router(kb_router)
    app.state.uploads_db = db
    app.state.qdrant = object()
    app.state.text_embedder = SimpleNamespace(embed_one=_embed_one)
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=SECRET, owui_url="http://owui"),
        raw={"kb": {"hybrid": {"enabled": True}}},
    )
    return app, captured


async def _embed_one(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _query(app: FastAPI, body: dict):
    return TestClient(app).post(
        "/v1/kb/query", json=body, headers={"X-Audrey-Service-Token": SECRET},
    )


class TestFilenameResolution:
    @pytest.mark.asyncio
    async def test_a_known_filename_resolves_to_its_file_id(self, db, monkeypatch):
        await _add(db, "f1", user="alice@example.com", filename="standup.mp4")
        app, captured = _build_app(db, monkeypatch)

        r = _query(app, {"query": "handover", "user": "alice@example.com",
                         "filename": "standup.mp4"})

        assert r.status_code == 200
        assert captured["scope"].file_ids == ["f1"]

    @pytest.mark.asyncio
    async def test_matching_is_case_insensitive(self, db, monkeypatch):
        await _add(db, "f1", user="alice@example.com", filename="StandUp.MP4")
        app, captured = _build_app(db, monkeypatch)

        _query(app, {"query": "handover", "user": "alice@example.com",
                     "filename": "standup.mp4"})

        assert captured["scope"].file_ids == ["f1"]

    @pytest.mark.asyncio
    async def test_a_partial_name_does_not_match(self, db, monkeypatch):
        """A near-miss must miss. Resolving 'standup' to 'standup.mp4' looks
        helpful right up to the call where it picks the wrong one of two."""
        await _add(db, "f1", user="alice@example.com", filename="standup.mp4")
        app, captured = _build_app(db, monkeypatch)

        r = _query(app, {"query": "handover", "user": "alice@example.com",
                         "filename": "standup"})

        assert r.json()["results"] == []
        assert "scope" not in captured, "Qdrant should not have been searched at all"

    @pytest.mark.asyncio
    async def test_duplicate_filenames_resolve_to_every_match(self, db, monkeypatch):
        """Filenames are not unique. Two uploads called the same thing means
        the user's 'in standup.mp4' honestly refers to both."""
        await _add(db, "f1", user="alice@example.com", filename="standup.mp4")
        await _add(db, "f2", user="alice@example.com", filename="standup.mp4")
        app, captured = _build_app(db, monkeypatch)

        _query(app, {"query": "handover", "user": "alice@example.com",
                     "filename": "standup.mp4"})

        assert sorted(captured["scope"].file_ids) == ["f1", "f2"]

    @pytest.mark.asyncio
    async def test_another_users_file_does_not_resolve(self, db, monkeypatch):
        """Resolution is scoped to the effective user, so naming someone
        else's filename finds nothing rather than reaching their chunks."""
        await _add(db, "f1", user="bob@example.com", filename="secret.mp4")
        app, captured = _build_app(db, monkeypatch)

        r = _query(app, {"query": "anything", "user": "alice@example.com",
                         "filename": "secret.mp4"})

        assert r.json()["results"] == []
        assert "scope" not in captured


class TestUnknownFilename:
    @pytest.mark.asyncio
    async def test_it_returns_empty_rather_than_searching_everything(
        self, db, monkeypatch,
    ):
        """The silent-widening failure. A scoped question whose file does not
        exist must not become an unscoped one."""
        app, captured = _build_app(db, monkeypatch)

        r = _query(app, {"query": "handover", "user": "alice@example.com",
                         "filename": "nope.mp4"})

        assert r.status_code == 200
        assert r.json()["results"] == []
        assert "scope" not in captured

    @pytest.mark.asyncio
    async def test_it_names_the_files_that_do_exist(self, db, monkeypatch):
        """Empty results alone are ambiguous between 'that file says nothing
        about this' and 'there is no such file'. A model told the first will
        report it, confidently, about a file the user does not have.

        Naming the alternatives costs nothing — the lookup already read these
        rows in order to fail — and closes the loop in one turn. Without it
        the model's likeliest next move is to guess a second filename.
        """
        await _add(db, "f1", user="alice@example.com", filename="standup.mp4")
        await _add(db, "f2", user="alice@example.com", filename="retro.mp4")
        app, _ = _build_app(db, monkeypatch)

        notice = _query(app, {"query": "handover", "user": "alice@example.com",
                              "filename": "nope.mp4"}).json()["notice"]

        assert "nope.mp4" in notice
        assert "standup.mp4" in notice
        assert "retro.mp4" in notice

    @pytest.mark.asyncio
    async def test_a_user_with_no_uploads_is_told_that_instead(self, db, monkeypatch):
        """"Available files are: " followed by nothing reads as a bug. The
        real reason is worth stating plainly."""
        app, _ = _build_app(db, monkeypatch)

        notice = _query(app, {"query": "handover", "user": "alice@example.com",
                              "filename": "nope.mp4"}).json()["notice"]

        assert "has not uploaded any files" in notice

    @pytest.mark.asyncio
    async def test_the_listing_in_the_notice_is_capped(self, db, monkeypatch):
        """A prolific user's whole file list does not belong in a tool result,
        but the count must stay exact so the reply is never misleading about
        what was left out."""
        for i in range(25):
            await _add(db, f"f{i}", user="alice@example.com", filename=f"v{i:02d}.mp4")
        app, _ = _build_app(db, monkeypatch)

        notice = _query(app, {"query": "handover", "user": "alice@example.com",
                              "filename": "nope.mp4"}).json()["notice"]

        assert "and 5 more" in notice
        assert "v00.mp4" in notice
        assert "v24.mp4" not in notice

    @pytest.mark.asyncio
    async def test_duplicate_filenames_are_flagged_rather_than_merged_silently(
        self, db, monkeypatch,
    ):
        """The caller asked about one thing. An answer stitched from two
        recordings sharing that name is wrong in a way it cannot see."""
        await _add(db, "f1", user="alice@example.com", filename="standup.mp4")
        await _add(db, "f2", user="alice@example.com", filename="standup.mp4")
        app, _ = _build_app(db, monkeypatch)

        body = _query(app, {"query": "handover", "user": "alice@example.com",
                            "filename": "standup.mp4"}).json()

        assert body["results"], "both files should still be searched"
        assert "2 uploaded files are named" in body["notice"]

    @pytest.mark.asyncio
    async def test_an_ordinary_search_carries_no_notice(self, db, monkeypatch):
        app, _ = _build_app(db, monkeypatch)
        assert _query(app, {"query": "handover", "user": "alice@example.com"}).json()[
            "notice"] == ""


class TestHitAttribution:
    @pytest.mark.asyncio
    async def test_hits_name_their_file_and_artifact(self, db, monkeypatch):
        """Without these, a caller cannot say which file an answer came from:
        `source` for an upload is the sidecar path, which is file_id-derived
        and unreadable."""
        app, _ = _build_app(db, monkeypatch)

        hit = _query(app, {"query": "handover", "user": "alice@example.com"}).json()[
            "results"][0]

        assert hit["filename"] == "standup.mp4"
        assert hit["artifact"] == "transcript"


class TestArtifactFilter:
    @pytest.mark.asyncio
    async def test_an_artifact_can_be_requested_without_a_filename(
        self, db, monkeypatch,
    ):
        app, captured = _build_app(db, monkeypatch)

        _query(app, {"query": "on the slide", "user": "alice@example.com",
                     "artifact": "visual"})

        assert captured["scope"].artifact == "visual"
        assert captured["scope"].file_ids is None

    @pytest.mark.asyncio
    async def test_an_unknown_artifact_is_rejected(self, db, monkeypatch):
        """Enumerated at the edge so the model cannot invent a fourth kind and
        get a silently empty result."""
        app, _ = _build_app(db, monkeypatch)

        r = _query(app, {"query": "x", "user": "alice@example.com",
                         "artifact": "subtitles"})

        assert r.status_code == 422


# ─── The scope is logged, so §3b is answerable (2026-08-06) ───────────
#
# The route used to log only the failure path — a filename matching nothing —
# so a successful scoped search and a deliberately unscoped one left identical
# traces. Phase 40 §3b asks "does the model scope only when the user pointed at
# one file", and with a single video in the KB both behaviours produce the same
# answer prose. Without this line the check cannot be run at all.


class TestScopeIsLegibleAfterTheFact:
    def test_an_unscoped_query_says_so(self):
        from audrey.routes.kb import TextQuery, _scope_label

        assert _scope_label(TextQuery(query="what did they say"), None) == "scope=none"

    def test_a_filename_scope_names_the_file_and_the_id_count(self):
        from audrey.kb.qdrant import SearchScope
        from audrey.routes.kb import TextQuery, _scope_label

        label = _scope_label(
            TextQuery(query="x", filename="jasonRetirement.mp4"),
            SearchScope(file_ids=["abc"]),
        )
        assert "jasonRetirement.mp4" in label
        assert "1 id" in label

    def test_a_duplicate_filename_shows_more_than_one_id(self):
        """The ambiguous case the route already warns about in `notice`.
        Two ids behind one name is the difference between a scoped answer and
        a stitched-together one, so the count is the part worth logging."""
        from audrey.kb.qdrant import SearchScope
        from audrey.routes.kb import TextQuery, _scope_label

        label = _scope_label(
            TextQuery(query="x", filename="standup.mp4"),
            SearchScope(file_ids=["a", "b"]),
        )
        assert "2 ids" in label

    def test_an_artifact_only_scope_is_reported(self):
        """`artifact` narrows without a filename, so reading `req.filename`
        alone would call this unscoped."""
        from audrey.kb.qdrant import SearchScope
        from audrey.routes.kb import TextQuery, _scope_label

        label = _scope_label(
            TextQuery(query="x", artifact="visual"), SearchScope(artifact="visual"),
        )
        assert "artifact=visual" in label
        assert label != "scope=none"
