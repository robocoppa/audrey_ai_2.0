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

    def test_it_forbids_writing_about_an_unread_file(self):
        # The specific failure: `list_my_files` proves a file exists, which the
        # model then treats as licence to describe it.
        d = _description("kb_search").lower()
        assert "no hits from a file" in d
        assert "say so" in d

    def test_the_scoping_instruction_survives(self):
        # The pooling guidance was appended to an existing description; this
        # guards the half that was already load-bearing.
        assert "list_my_files" in _description("kb_search")
