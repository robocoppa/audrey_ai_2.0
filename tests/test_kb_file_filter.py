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

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.kb.qdrant import KBHit, SearchScope, _as_filter
from audrey.kb.uploads_db import UploadsDB
from audrey.routes.kb import _search_text_hybrid, _search_text_merged
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


# ─── The filter object ────────────────────────────────────────────────


class TestSearchScope:
    def test_an_unset_scope_produces_no_filter(self):
        assert _as_filter(None) is None
        assert _as_filter(SearchScope()) is None

    def test_a_scope_that_matched_no_files_is_not_the_same_as_no_scope(self):
        """The distinction the whole step turns on. `None` means "search
        everything"; `[]` means "a file was named and it does not exist", and
        conflating them turns a scoped question into a corpus-wide one."""
        assert SearchScope(file_ids=[]).matches_nothing is True
        assert SearchScope(file_ids=["f1"]).matches_nothing is False
        assert SearchScope().matches_nothing is False

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


# ─── The trap: both retrievers, or neither ────────────────────────────


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

        assert [s for _, s in q.dense] == [scope]
        assert [s for _, s in q.lexical] == [scope]

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

        assert q.dense[0][1] is q.lexical[0][1] is scope

    @pytest.mark.asyncio
    async def test_an_unscoped_search_is_unchanged(self):
        """Omitting the filter must restore the previous behaviour exactly —
        both collections, both retrievers, no filter."""
        q = _ScopeRecordingQdrant()

        await _search_text_hybrid(
            q, [0.1], query="handover", top_k=5, user="alice@example.com",
            min_score=0.0, cfg=CFG,
        )

        assert [c for c, _ in q.dense] == ["kb_text", "kb_user_text_alice_example_com"]
        assert [c for c, _ in q.lexical] == ["kb_text", "kb_user_text_alice_example_com"]
        assert all(s is None for _, s in q.dense + q.lexical)

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
        assert q.dense[0][1] is scope


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
    async def test_it_says_the_file_is_missing(self, db, monkeypatch):
        """Empty results alone are ambiguous between 'that file says nothing
        about this' and 'there is no such file'. A model told the first will
        report it, confidently, about a file the user does not have."""
        app, _ = _build_app(db, monkeypatch)

        notice = _query(app, {"query": "handover", "user": "alice@example.com",
                              "filename": "nope.mp4"}).json()["notice"]

        assert "nope.mp4" in notice
        assert "list_my_files" in notice

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
