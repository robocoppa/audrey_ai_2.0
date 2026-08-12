"""`list_my_files` presents its artifact list as UNREAD (2026-08-12).

The defect this guards is the one that passes every other check. Across the
archived video eval runs, "summarise my most recent upload" was answered with
`list_my_files` as the only successful tool call in 30 of 42 judgeable turns —
and each of those 30 invented a different match from the same filename: a
rear-naked choke, a guillotine, a mounted triangle, an eye-gouging
disqualification, a coach recast as a competitor's father. The prose was
fluent, the structure correct, and none of it came from the file.

Deleting `summary` from the row in 2026-08-06 stopped the listing carrying
contents. It did not stop the listing being answered FROM. `unread_artifacts`
is the follow-up: the field name is the last thing the model reads before
deciding it has enough, so it is where the "you have not read this" goes.

Source-level on purpose, like test_kb_search_description.py — the description
and the field name are what the model sees, and no unit test of the handler
can tell whether they say the right thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from app import MyFileRow, app  # noqa: E402


def _description(operation_id: str) -> str:
    for route in app.routes:
        if getattr(route, "operation_id", None) == operation_id:
            return route.description or ""
    raise AssertionError(f"no route with operation_id={operation_id!r}")


class TestTheWireNameIsUnchanged:
    """⚠️ The rename is presentational and must stay that way. Audrey's
    `ModelFileRow` sends `artifacts`; a tools-server that stopped accepting
    that name would read every row as having no artifacts and report real
    files as empty — a worse failure than the one being fixed, and one that
    only shows up when the two containers deploy out of step.
    """

    def test_it_parses_the_wire_name_artifacts(self):
        row = MyFileRow.model_validate({
            "filename": "a.mp4", "kind": "video", "status": "ready",
            "uploaded_at": "2026-08-08T00:00:00Z",
            "artifacts": ["summary", "visual"],
        })
        assert row.unread_artifacts == ["summary", "visual"]

    def test_a_row_without_the_field_still_parses_as_empty(self):
        row = MyFileRow.model_validate({
            "filename": "silent.mp4", "kind": "video", "status": "ready",
            "uploaded_at": "2026-08-08T00:00:00Z",
        })
        assert row.unread_artifacts == []

    def test_it_reaches_the_model_under_the_unread_name(self):
        row = MyFileRow.model_validate({
            "filename": "a.mp4", "kind": "video", "status": "ready",
            "uploaded_at": "2026-08-08T00:00:00Z",
            "artifacts": ["summary"],
        })
        dumped = row.model_dump(by_alias=True)
        assert dumped["unread_artifacts"] == ["summary"]
        assert "artifacts" not in dumped


class TestTheDescriptionSaysNamesAreNotContents:
    def test_it_names_the_unread_field(self):
        assert "unread_artifacts" in _description("list_my_files")

    def test_it_says_a_listed_artifact_is_not_a_read_one(self):
        d = _description("list_my_files").lower()
        assert "names, not contents" in d
        assert "never what it says" in d

    def test_it_names_the_catalogue_only_state_as_not_having_the_answer(self):
        """The 30 fabrications are exactly this state, so the description
        should describe it rather than leave it to be inferred."""
        d = _description("list_my_files").lower()
        assert "called only list_my_files" in d
        assert "from the filename" in d

    def test_the_empty_case_still_forbids_describing_the_file(self):
        """Regression: the empty-artifacts guidance predates this change and
        was rewritten around the new field name. `video-empty-artifacts`
        depends on it and passes today."""
        d = _description("list_my_files").lower()
        assert "empty unread_artifacts list" in d
        assert "never describe it" in d

    def test_it_still_says_the_listing_is_a_catalogue(self):
        assert "catalogue, not contents" in _description("list_my_files")
