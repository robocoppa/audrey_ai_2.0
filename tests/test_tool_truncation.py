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

import pytest

from audrey.tools.dispatch import _truncate, _truncate_payload

# The deployed fast-path cap. `audrey_auto` → `fast_path` → `react_max_tool_chars`.
FAST_PATH_CAP = 2000


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
            print(f"\n  fast-path cap: {FAST_PATH_CAP:,} chars"
                  f"   (agentic.react.max_tool_result_chars)")
            print(f"  {'top_k':>6} {'full size':>11} {'hits kept':>10}")
            for top_k, full, kept in rows:
                print(f"  {top_k:>6} {full:>10,}c {kept:>10}")

        kept = {r[2] for r in rows}
        # THE finding: asking for more returns no more. Every top_k lands on
        # the same number of surviving hits, so a model that retries with a
        # bigger top_k burns a round for nothing — which is what the marker
        # now tells it in words.
        assert len(kept) == 1, f"expected an identical yield at every top_k, got {kept}"

    @pytest.mark.parametrize("cap", [2000, 4000, 6000, 12000])
    def test_what_a_bigger_budget_would_buy(self, cap, capsys):
        payload = _kb_response(20)
        content = json.dumps(payload, ensure_ascii=False)
        out = json.loads(_truncate_payload(payload, content, cap))
        kept = len(out.get("results", []))
        with capsys.disabled():
            # 2000 is the fast-path default, 4000 is deep_worker, 6000 is
            # research_worker — each raised after this same failure was seen in
            # that role. This is the table that says what the fast path is
            # currently buying, and what the other budgets would buy it.
            print(f"    cap {cap:>6,}c -> {kept:>2} of 20 transcript hits "
                  f"({kept * 992:,}c of prose)")
        assert kept >= 1
