"""What a model actually receives when a tool result does not fit.

Written after a real failure on 2026-08-05. Asked for a video's transcript, the
model returned a partial excerpt and said "due to system limitations I can only
provide a partial excerpt … request it again in a new session". The limitation
was **real** — `agentic.react.max_tool_result_chars` is 2000 on the fast path —
and the model was right to report it. What it could not know was the size of
the hole or that no retry would help, because the marker said neither.

Two kinds of case here, and the second kind is the point:

  - **Assertions** on the new behaviour: the count is reported, retrying is
    named as useless, JSON stays parseable, nothing is truncated that fits.
  - **Measurements** (`TestMeasured`) that print the real numbers for this
    codebase's own chunk sizes. They assert only the load-bearing fact and
    report the rest, because the useful output is the table, not a green tick.
    Run with `-s` to read them:

        .venv/bin/python -m pytest tests/test_tool_truncation.py -k Measured -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from audrey.tools.dispatch import _truncate, _truncate_payload

_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def _fast_path_cap() -> int:
    """The live cap, read from `config.yaml` rather than copied.

    A constant here was wrong within the hour. It said `2000` and was labelled
    as `agentic.react.max_tool_result_chars`, which stopped being true the
    moment that value was raised to 6000 — so the measurement printed a
    confident table describing a configuration that no longer existed. A test
    whose whole purpose is to report the real numbers must not carry its own
    copy of one.
    """
    cfg = yaml.safe_load(_CONFIG.read_text("utf-8"))
    return int(cfg["agentic"]["react"]["max_tool_result_chars"])


FAST_PATH_CAP = _fast_path_cap()

#: What the fast path was set to when the 2026-08-05 failure was observed.
#: Kept as history so the comparison table below still shows what that turn
#: actually had to work with — it is not read from anywhere.
CAP_AT_FAILURE = 2000


def _hit(chars: int, idx: int = 0) -> dict:
    """One `kb_search` hit, shaped like `routes/kb.py::Hit`."""
    return {
        "score": 0.7231, "source": "/data/uploads/a_b_c/8f3d.transcript.txt",
        "kind": "text", "chunk_idx": idx, "text": "x" * chars,
        "filename": "jason retirement.mp4", "artifact": "transcript",
    }


def _kb_response(n: int, chars: int = 992) -> dict:
    return {"query": "what did they say about retirement",
            "results": [_hit(chars, i) for i in range(n)], "notice": ""}


class TestItSaysWhatWasLost:
    def test_a_result_that_fits_is_untouched(self):
        payload = _kb_response(1, chars=50)
        content = json.dumps(payload, ensure_ascii=False)
        assert _truncate_payload(payload, content, 10_000) == content

    def test_the_count_of_dropped_results_is_reported(self):
        payload = _kb_response(12)
        content = json.dumps(payload, ensure_ascii=False)
        out = json.loads(_truncate_payload(payload, content, FAST_PATH_CAP))
        note = out["_truncated"]
        # The number the model needs to answer honestly: it can now say "I can
        # see 1 of 12 matching passages" instead of guessing at a cause.
        assert f"of {len(payload['results'])} results" in note
        assert str(len(payload["results"]) - len(out["results"])) in note

    def test_the_body_is_still_parseable_json(self):
        # The old character cut severed the JSON mid-string, leaving the model
        # to interpret a half-word and an unbalanced brace.
        payload = _kb_response(12)
        content = json.dumps(payload, ensure_ascii=False)
        out = _truncate_payload(payload, content, FAST_PATH_CAP)
        assert json.loads(out)["results"]  # parses, and kept something

    def test_it_says_retrying_will_not_help(self):
        # The whole reason this exists. A model reading a bare marker
        # reasonably infers "ask for more", which costs a round out of three
        # and returns an identical amount of text.
        payload = _kb_response(12)
        out = _truncate_payload(payload, json.dumps(payload), FAST_PATH_CAP)
        assert "larger top_k" in json.loads(out)["_truncated"]

    def test_it_never_exceeds_the_cap(self):
        for n in (2, 5, 12, 40):
            payload = _kb_response(n)
            content = json.dumps(payload, ensure_ascii=False)
            assert len(_truncate_payload(payload, content, FAST_PATH_CAP)) <= FAST_PATH_CAP

    def test_a_payload_with_no_list_still_reports_its_size(self):
        # http-error bodies and any tool returning a bare object.
        payload = {"error": "http_500", "detail": "x" * 5000}
        out = _truncate(json.dumps(payload), 500)
        assert len(out) <= 500
        assert "of 5,0" in out  # the true size survives in the marker

    def test_a_non_json_body_falls_back_to_a_char_cut(self):
        out = _truncate_payload(None, "y" * 5000, 500)
        assert len(out) <= 500
        assert "truncated" in out

    def test_one_oversized_item_still_degrades_to_a_char_cut(self):
        # A single hit bigger than the whole cap: there is no number of items
        # to keep, so the character path has to catch it rather than looping.
        payload = _kb_response(1, chars=50_000)
        content = json.dumps(payload, ensure_ascii=False)
        out = _truncate_payload(payload, content, FAST_PATH_CAP)
        assert len(out) <= FAST_PATH_CAP
        assert "truncated" in out

    def test_the_longest_list_is_the_one_shrunk(self):
        # Chosen by serialized size, not by name, so a tool added later needs
        # no entry anywhere.
        payload = {"tags": ["a", "b"], "results": [_hit(992, i) for i in range(9)]}
        out = json.loads(_truncate_payload(
            payload, json.dumps(payload), FAST_PATH_CAP))
        assert out["tags"] == ["a", "b"]
        assert len(out["results"]) < 9


class TestMeasured:
    """Numbers, not verdicts. Run with `-s` to read them."""

    def test_how_much_of_a_transcript_search_survives(self, capsys):
        # `kb.video.transcript_chunk_tokens` is 250; measured against real
        # transcript prose that is ~992 chars.
        rows = []
        for top_k in (5, 10, 20):
            payload = _kb_response(top_k)
            content = json.dumps(payload, ensure_ascii=False)
            out = json.loads(_truncate_payload(payload, content, FAST_PATH_CAP))
            rows.append((top_k, len(content), len(out.get("results", []))))

        with capsys.disabled():
            moved = "" if FAST_PATH_CAP == CAP_AT_FAILURE else \
                f"  (was {CAP_AT_FAILURE:,} when the failure was observed)"
            print(f"\n  fast-path cap: {FAST_PATH_CAP:,} chars, live from "
                  f"config.yaml agentic.react.max_tool_result_chars{moved}")
            print(f"  {'top_k':>6} {'full size':>11} {'hits kept':>10}")
            for top_k, full, kept in rows:
                print(f"  {top_k:>6} {full:>10,}c {kept:>10}")

        # THE finding, stated durably rather than for one budget. Among the
        # top_k values whose *full* response already exceeds the cap, the yield
        # is flat — so a model that overflowed and retries with a bigger top_k
        # burns a round for nothing. That is what the marker now says in words.
        #
        # The first version of this asserted an identical yield at EVERY top_k,
        # which was true only while the cap was 2000 and every response
        # overflowed. At 6000 the row for top_k=5 fits whole, and the
        # assertion failed — correctly, because the premise had changed under
        # it. A measurement test that encodes one configuration's arithmetic
        # stops being a measurement.
        overflowing = [kept for _k, full, kept in rows if full > FAST_PATH_CAP]
        assert len(set(overflowing)) <= 1, (
            f"once the response exceeds the {FAST_PATH_CAP:,}-char cap the yield "
            f"should stop moving, got {overflowing}"
        )

        # The counterintuitive corollary, and the reason the marker tells the
        # model to narrow rather than widen: a request small enough to fit
        # whole avoids both the dropped items and the note's own overhead, so
        # asking for LESS can return MORE.
        best = max(rows, key=lambda r: r[2])
        with capsys.disabled():
            if best[0] != max(r[0] for r in rows):
                print(f"  → top_k={best[0]} yields {best[2]} hits, more than "
                      f"top_k={max(r[0] for r in rows)} yields "
                      f"{[r[2] for r in rows][-1]}: a response that fits whole "
                      "escapes both the drop and the note.")

    @pytest.mark.parametrize("cap", [2000, 4000, 6000, 12000])
    def test_what_each_budget_buys(self, cap, capsys):
        payload = _kb_response(20)
        content = json.dumps(payload, ensure_ascii=False)
        out = json.loads(_truncate_payload(payload, content, cap))
        kept = len(out.get("results", []))
        with capsys.disabled():
            # 2000 was the fast path at the time of the failure, 4000 is
            # deep_worker, 6000 is research_worker — the latter two each raised
            # after this same wall was hit in that role. The arrow marks
            # whichever row the fast path is on now.
            here = " ← fast path" if cap == FAST_PATH_CAP else ""
            print(f"    cap {cap:>6,}c -> {kept:>2} of 20 transcript hits "
                  f"({kept * 992:,}c of prose){here}")
        assert kept >= 1
