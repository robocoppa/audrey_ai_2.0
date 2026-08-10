"""Which files an unscoped KB search actually returned material from.

`kb_search` pools hits across files and orders them by score, so a question
that spans two documents can be answered from a top-k that is almost entirely
one of them. That failure is invisible in the answer: `list_my_files` has
already told the model the other file exists, so it writes a section per file
and fills the thin one from whatever it has. Three A-B runs read as though both
videos had been consulted.

Stage 0 is the measurement — `files=[…]` on the `kb.query` line — and it is
worth keeping whatever happens to the rest of the phase, because nothing on the
box could previously answer "did the model have material from both files"
after the fact.

⚠️ The empty-filename bucket is the load-bearing rule here, not a detail.
Global-KB hits carry no filename (`src/audrey/routes/kb.py:104`); counted
individually they would render a geology question with no uploads involved as a
dozen nameless files and bury the one line that matters.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.kb.qdrant import KBHit
from audrey.routes.kb import _clip, _file_distribution, _search_text_hybrid
from audrey.routes.kb import router as kb_router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)


def _hit(filename: str = "", score: float = 0.9, *,
         source: str = "/u/f1.txt", text: str = "…") -> KBHit:
    return KBHit(score=score, source=source, kind="text", chunk_idx=0,
                 text=text, payload={"filename": filename})


# ─── The distribution itself ──────────────────────────────────────────


def test_the_skew_is_readable_from_the_first_entry():
    """The whole point of the field. Heaviest file first, so the decision this
    phase gates on — balanced, skewed, or absent — is one glance."""
    hits = [_hit("carlsen.mp4")] + [_hit("how-to-win.mp4")] * 8 + [_hit("carlsen.mp4")]

    assert _file_distribution(hits) == "files=[how-to-win.mp4:8, carlsen.mp4:2]"


def test_global_hits_share_one_bucket():
    """⚠️ Not one entry per hit. These have no filename at all, and a search
    that never touched an upload is the common case."""
    assert _file_distribution([_hit(), _hit(), _hit()]) == "files=[<global>:3]"


def test_global_and_upload_hits_are_distinguishable():
    """The split this phase is really about is user material vs everything
    else, so the two must never merge into one unlabelled count."""
    out = _file_distribution([_hit(), _hit("standup.mp4"), _hit()])

    assert out == "files=[<global>:2, standup.mp4:1]"


def test_a_missing_filename_key_counts_as_global():
    """Payloads reach here from two collections and two retrievers. An absent
    key and an empty string mean the same thing — no uploader — and reading one
    of them as a file named "None" would invent a document."""
    bare = KBHit(score=0.9, source="/g/x.txt", kind="text", chunk_idx=0,
                 text="…", payload={})

    assert _file_distribution([bare]) == "files=[<global>:1]"


def test_an_empty_result_set_still_reports_the_field():
    """⚠️ A missing field and a zero read identically in a log, and that exact
    ambiguity cost four rounds of greps on the streaming panel line: the field
    was absent, which looked like "no such turns ran"."""
    assert _file_distribution([]) == "files=[]"


def test_ties_are_ordered_by_name_so_the_line_is_stable():
    """Two files with equal counts must not swap places between runs — a log
    line that reorders itself invites reading a difference into two identical
    searches."""
    hits = [_hit("zebra.mp4"), _hit("apple.mp4")]

    assert _file_distribution(hits) == "files=[apple.mp4:1, zebra.mp4:1]"


def test_long_filenames_are_shortened():
    """Uploads here are long descriptive titles. Three untruncated names push
    the counts off the end of a terminal line, and a field nobody can read at a
    glance does not do the job it was added for."""
    name = "How to WIN with the London System every single time - GothamChess.mp4"
    out = _file_distribution([_hit(name)])

    assert out == "files=[How to WIN with the London System every…:1]"
    assert len(out) < len(name)


def test_clipping_leaves_short_text_exactly_as_it_was():
    """No ellipsis, no padding, no quoting — the common case must survive the
    helper untouched or every short line grows noise."""
    assert _clip("carlsen.mp4", 40) == "carlsen.mp4"
    assert _clip("x" * 40, 40) == "x" * 40


def test_the_label_names_which_set_is_being_counted():
    """⚠️ Two distributions per query, and confusing them inverts the reading.

    `files=` is what survived; `cut=` is what the evidence floor removed. A
    file absent from `files=` and present in `cut=` was FILTERED — no
    reordering can recover it, because the hit is gone. Absent from both means
    it was merely outranked, which a diversity reorder fixes. Same numbers,
    opposite conclusions.
    """
    hits = [_hit("carlsen.mp4")]

    assert _file_distribution(hits) == "files=[carlsen.mp4:1]"
    assert _file_distribution(hits, label="cut") == "cut=[carlsen.mp4:1]"


def test_shortening_keeps_enough_to_tell_two_files_apart():
    """A prefix-only truncation is the risk: two videos from the same series
    can share thirty characters. This pins the budget as generous enough for
    the corpus that motivated the phase, not merely non-zero."""
    a = _file_distribution([_hit("London System - Part 1 - opening ideas and setup.mp4")])
    b = _file_distribution([_hit("London System - Part 1 - middlegame plans.mp4")])

    assert a != b


# ─── Cut by the floor vs merely outranked ─────────────────────────────


class _FakeHybridQdrant:
    """Global collection only — `user=None` keeps the fan-out to one call."""

    text_collection = "kb_text"

    def __init__(self, dense: list[KBHit], lexical: list[KBHit]):
        self._dense = dense
        self._lexical = lexical

    async def collection_exists(self, name: str) -> bool:
        return False

    async def search_hybrid(self, vec, query, *, top_k, collection=None, scope=None):
        return (list(self._dense), list(self._lexical))


_HYBRID_CFG = {"enabled": True, "rrf_k": 60, "min_term_overlap": 0.7}


def _hybrid_line(caplog) -> str:
    return next(m for m in caplog.messages if m.startswith("kb.hybrid:"))


async def test_the_hybrid_line_names_the_files_the_floor_removed(caplog):
    """⚠️ The fork this line exists to resolve.

    A file missing from `files=[…]` is ambiguous between "filtered out" and
    "outranked", and those need opposite fixes — a hit the floor removed is
    gone from the list and no reordering can recover it. On 2026-08-10 an
    unscoped London query returned 20 of 20 hits from one of two ingested
    videos, and `top_k` caps at 20, so nothing could see which case it was.
    """
    caplog.set_level(logging.INFO)
    q = _FakeHybridQdrant(
        dense=[
            _hit("how-to-win.mp4", 0.81, source="/u/a.txt",
                 text="the london system chess opening starts with d4"),
            # Under the 0.53 cosine floor, and in no lexical list, so the
            # overlap branch cannot rescue it either.
            _hit("carlsen.mp4", 0.44, source="/u/b.txt",
                 text="he plays a quick blitz game here"),
        ],
        lexical=[],
    )

    hits, _ = await _search_text_hybrid(
        q, [0.1], query="london system chess opening", top_k=10, user=None,
        min_score=0.53, cfg=_HYBRID_CFG,
    )

    assert [h.payload["filename"] for h in hits] == ["how-to-win.mp4"]
    line = _hybrid_line(caplog)
    assert "fused=2 -> 1 kept" in line
    assert "kept=[how-to-win.mp4:1]" in line
    assert "cut=[carlsen.mp4:1]" in line


async def test_a_file_outranked_out_of_top_k_shows_in_kept_not_cut(caplog):
    """⚠️ The case the whole line exists to distinguish, and the one `cut=`
    alone gets wrong.

    Both hits clear the floor; `top_k=1` drops one from the RESULTS. It is
    still in the pool — which is exactly what makes it recoverable by
    reordering — so it must appear in `kept=` and not in `cut=`. Reading only
    `cut=[]` here would say "nothing was filtered" and leave the real question
    unanswered.
    """
    caplog.set_level(logging.INFO)
    q = _FakeHybridQdrant(
        dense=[
            _hit("how-to-win.mp4", 0.81, source="/u/a.txt", text="london system d4"),
            _hit("carlsen.mp4", 0.79, source="/u/b.txt", text="london system blitz"),
        ],
        lexical=[],
    )

    hits, _ = await _search_text_hybrid(
        q, [0.1], query="london system", top_k=1, user=None,
        min_score=0.53, cfg=_HYBRID_CFG,
    )

    assert [h.payload["filename"] for h in hits] == ["how-to-win.mp4"]
    line = _hybrid_line(caplog)
    assert "fused=2 -> 2 kept" in line
    # Present in the pool despite being absent from the results.
    assert "carlsen.mp4:1" in line[line.index("kept="):line.index("cut=")]
    assert "cut=[]" in line


async def test_one_chunk_cut_does_not_mean_the_file_left_the_pool(caplog):
    """The ambiguity that made `cut=` insufficient on its own (2026-08-10).

    A file with one chunk below the floor and others above appears in BOTH
    lists. Seeing it in `cut=` and concluding "the floor removed that file"
    points the fix at the floors when the pool still holds it.
    """
    caplog.set_level(logging.INFO)
    q = _FakeHybridQdrant(
        dense=[
            _hit("how-to-win.mp4", 0.81, source="/u/a.txt", text="london system d4"),
            _hit("carlsen.mp4", 0.77, source="/u/b.txt", text="london system blitz"),
            _hit("carlsen.mp4", 0.41, source="/u/c.txt", text="unrelated chatter"),
        ],
        lexical=[],
    )

    await _search_text_hybrid(
        q, [0.1], query="london system", top_k=10, user=None,
        min_score=0.53, cfg=_HYBRID_CFG,
    )

    line = _hybrid_line(caplog)
    assert "fused=3 -> 2 kept" in line
    assert "carlsen.mp4:1" in line[line.index("kept="):line.index("cut=")]
    assert "cut=[carlsen.mp4:1]" in line


# ─── Through the route ────────────────────────────────────────────────


async def _embed_one(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    """The route over a fake retriever returning a deliberately skewed set."""
    from audrey.routes import kb as kb_module

    async def _fake_merged(qdrant, vec, *, top_k, user, min_score, scope=None):
        return ([_hit("how-to-win.mp4")] * 3 + [_hit("carlsen.mp4")], True)

    monkeypatch.setattr(kb_module, "_search_text_merged", _fake_merged)

    app = FastAPI()
    app.include_router(kb_router)
    app.state.qdrant = object()
    app.state.text_embedder = SimpleNamespace(embed_one=_embed_one)
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=SECRET, owui_url="http://owui"),
        raw={},
    )
    return app


def _query_line(app, caplog, query: str = "london system") -> str:
    caplog.set_level(logging.INFO)
    r = TestClient(app).post(
        "/v1/kb/query", json={"query": query, "user": "alice@example.com"},
        headers={"X-Audrey-Service-Token": SECRET},
    )
    assert r.status_code == 200
    return next(m for m in caplog.messages if m.startswith("kb.query:"))


def test_the_query_line_carries_the_distribution(app, caplog):
    """Computing it and forgetting to log it would pass every test above."""
    line = _query_line(app, caplog)

    assert "files=[how-to-win.mp4:3, carlsen.mp4:1]" in line
    # The fields it already had must survive the addition.
    assert "scope=none" in line
    assert "4 hit(s)" in line


def test_the_query_line_carries_the_text_that_was_searched(app, caplog):
    """⚠️ The searched text, not the user's question.

    The model writes its own `kb_search` query and nothing else on the box
    records what it sent — `dispatch:` logs the tool name and result size,
    `react:` logs counts. Without this field a skewed `files=[…]` is
    ambiguous between "retrieval missed the file" and "it searched something
    else entirely", and on 2026-08-10 that ambiguity was the whole difference:
    the plural London question ranks the second video 4th when asked verbatim,
    while the model's own turn returned nothing from it.
    """
    line = _query_line(app, caplog, "what do my videos say about the london system")

    assert "q='what do my videos say about the london system'" in line


def test_a_long_query_is_clipped_out_of_the_way(app, caplog):
    """A pasted wall of text as the query must not push `files=[…]` off the end
    of the line — that field is the reason the line is being read. `TextQuery`
    accepts up to 2,000 characters, so this is a reachable input, not a
    hypothetical one."""
    line = _query_line(app, caplog, "london system " * 40)

    field = line[line.index("q="):line.index(" scope=")]
    assert field.endswith("…'")
    assert len(field) < 70
