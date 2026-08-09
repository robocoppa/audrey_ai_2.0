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
