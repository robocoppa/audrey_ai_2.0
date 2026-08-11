"""The kb_search tool description's file-provenance guidance (2026-08-09).

Source-level, because the description is what the model actually reads and the
failure it prevents is invisible in any unit test: an answer built from pooled
chunks reads perfectly and cites the wrong file.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from app import app  # noqa: E402


def _description(operation_id: str) -> str:
    for route in app.routes:
        if getattr(route, "operation_id", None) == operation_id:
            return route.description or ""
    raise AssertionError(f"no route with operation_id={operation_id!r}")


class TestKbSearchExplainsFileProvenance:
    """Both phase-42 A-B failures were one root cause: results come back pooled
    across files, and a model that never reads `filename` attributes them to
    whatever it already had in mind. `audrey_video` wrote a section about a
    video it had no chunks from; `audrey_auto` credited the videos' speakers to
    the user. Neither is a prompt problem — the guidance belongs here, where it
    reaches every model rather than only whoever picked the dropdown entry.
    """

    def test_it_says_results_are_pooled(self):
        d = _description("kb_search").lower()
        assert "pooled" in d

    def test_it_points_at_the_per_result_filename(self):
        assert "`filename`" in _description("kb_search")

    def test_it_ties_each_claim_to_the_file_its_hit_came_from(self):
        d = _description("kb_search").lower()
        assert "attribute each claim" in d
        assert "never let one file's content answer for another" in d

    def test_it_does_not_discourage_searching(self):
        """⚠️ The regression this exists to prevent, seen on the box 2026-08-09.

        The first version of this guidance said "if you have no hits from a
        file, say so instead of writing a section about it". Intended as an
        accuracy rule for AFTER a search; read by the model as a prohibition on
        writing about unread files, whose cheapest satisfaction is to not search
        at all. Both `audrey_video` and `audrey_auto` stopped answering the
        unscoped plural case entirely — `list_my_files` and then a question back
        to the user. `audrey_auto` moved too, which is what identified the
        description rather than the specialist prompt as the cause.

        A tool description is read as instructions about when to call the tool,
        not only about how to use its output. Wording that makes calling it feel
        risky suppresses the call.
        """
        d = _description("kb_search").lower()
        assert "search first" in d
        # The exact phrasings that produced the punt. Not a general ban on the
        # words — a ban on re-deriving this instruction shape.
        assert "instead of writing a section" not in d
        assert "no hits from a file" not in d

    def test_the_scoping_instruction_survives(self):
        # The pooling guidance was appended to an existing description; this
        # guards the half that was already load-bearing.
        assert "list_my_files" in _description("kb_search")


class TestAThinSearchIsNotEvidenceOfAbsence:
    """⚠️ The two descriptions between them produced a confident non-answer.

    2026-08-11, `video-ambiguous-singular` [audrey_auto]: four `kb_search`
    calls, no `get_file_text`, and the verdict "the specific content regarding
    the initial moves was not successfully extracted". Four cases later the
    SAME model read that exact fact off the SAME file in one `get_file_text`.
    The content was never missing.

    The cause is in the descriptions, not the prompts. `get_file_text` said
    reading a file for a topic question "wastes rounds"; `kb_search` said to
    report what you found. On a thin result the model had been steered off the
    only move that would have settled it, so it reported absence. Neither said
    what to do when a search fails to surface content that exists.

    The trigger is deliberately "you are about to claim it isn't there", not
    "the search was thin" — narrow enough that it cannot swing back into
    reading whole files for ordinary topic questions, which is the failure the
    surrounding wording exists to prevent.
    """

    def test_kb_search_says_a_miss_is_not_proof(self):
        d = _description("kb_search").lower()
        assert "not proof the content is absent" in d
        assert "get_file_text" in d

    def test_get_file_text_carries_the_matching_exception(self):
        """Both must agree. Two tool descriptions contradicting each other is
        worse than either being wrong on its own."""
        d = _description("get_file_text").lower()
        assert "one exception" in d
        assert "search miss is not" in d

    def test_the_steer_away_from_reading_survives(self):
        """The exception must not swallow the rule it is an exception to."""
        d = _description("get_file_text").lower()
        assert "wastes rounds" in d
        assert "use kb_search instead" in d

    def test_it_does_not_re_derive_the_shape_that_backfired(self):
        """⚠️ Same guard as `test_it_does_not_discourage_searching`, from the
        other direction: wording that makes SEARCHING feel unreliable would
        push models to read whole files by default."""
        d = _description("kb_search").lower()
        for banned in ("search is unreliable", "do not trust", "always read the file"):
            assert banned not in d


class TestPagingNeverSendsTheUserElsewhere:
    """⚠️ The paging dead-end, seen twice. 2026-08-10: "use video transcription
    software". 2026-08-11: "download the video file and use a speech-to-text
    service" — in an answer that also quoted the transcript's exact length
    (33,626 characters) and had already read 8,000 of them.

    Audrey transcribed the file. The remaining pages are one more call, and
    nothing in the description had ever said so.
    """

    def test_it_forbids_sending_the_user_to_another_tool(self):
        d = _description("get_file_text").lower()
        assert "never tell the user to obtain the text some other way" in d
        assert "transcription software" in d

    def test_it_denies_the_technical_limit_excuse(self):
        d = _description("get_file_text").lower()
        assert "never call the remaining pages a technical limit" in d

    def test_it_says_the_rest_is_always_reachable(self):
        d = _description("get_file_text").lower()
        assert "one more call" in d
