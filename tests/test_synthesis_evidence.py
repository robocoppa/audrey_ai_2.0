"""Observed tool activity must not promote unsupported drafts to verified facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from audrey.pipeline.react import run_react
from audrey.pipeline.synthesize import _build_synth_messages, _format_drafts_for_synth
from audrey.tools.discovery import ToolRegistry, ToolSpec


def test_failed_reference_search_and_catalog_only_draft_reach_synth_honestly():
    draft = {
        "model": "local-worker", "content": "An unsupported, confident change claim.",
        "tool_rounds": 4,
        "tool_calls": [
            {"name": name, "is_error": failed}
            for name, failed in [
                ("reference_search", True), ("reference_lookup", False),
                ("reference_search", True), ("reference_lookup", False),
            ]
        ],
        "retrieved": [],
    }
    block = _format_drafts_for_synth("Compare the two editions", [draft], [])
    # Exercise the actual synth boundary, rather than a summary used only in tests.
    messages = _build_synth_messages([], block, draft_count=1)
    content = messages[-1]["content"]
    assert "Successful calls: reference_lookup × 2" in content
    assert "Failed calls: reference_search × 2" in content
    assert "no retrieved reference passage or page is recorded" in content
    assert "substantive rules, amounts and changes remain unverified" in content
    assert "tool-grounded" not in content
    assert draft["content"] in content  # Evidence limitations do not discard the draft.


def test_successful_reference_page_is_activity_not_whole_draft_verification():
    block = _format_drafts_for_synth("Explain this edition", [{
        "model": "local-worker", "content": "A page-based draft.", "tool_rounds": 1,
        "tool_calls": [{"name": "read_reference", "is_error": False}],
        "retrieved": [{"tool": "read_reference", "title": "Guide · Page 1",
                       "url": "https://www.irs.gov/pub/irs-pdf/p17.pdf#page=1"}],
    }], [])
    assert "Successful calls: read_reference × 1" in block
    assert "Recorded public source identities: 1" in block
    assert "not claim verification" in block
    assert "OWNED-REFERENCE LIMITATION" not in block
    assert "tool-grounded" not in block


def test_private_file_and_calculator_results_are_not_labeled_missing_evidence():
    block = _format_drafts_for_synth("Calculate from the upload", [{
        "model": "local-worker", "content": "The result is 1350.", "tool_rounds": 2,
        "tool_calls": [{"name": "get_file_text", "is_error": False},
                       {"name": "calculate", "is_error": False}],
        "retrieved": [],
    }], [])
    assert "get_file_text × 1, calculate × 1" in block
    assert "does not describe private file or other tool results" in block
    assert "OWNED-REFERENCE LIMITATION" not in block
    assert "The result is 1350." in block


def test_old_round_count_without_outcomes_does_not_manufacture_evidence():
    block = _format_drafts_for_synth("Question", [{
        "model": "legacy-worker", "content": "Legacy draft", "tool_rounds": 3,
    }], [])
    assert "Successful calls: none recorded" in block
    assert "tool-grounded" not in block
    assert "verified facts" not in block


@pytest.mark.parametrize("names, guided, search_available", [
    ({"reference_lookup", "read_reference", "reference_search"}, True, True),
    ({"reference_lookup", "read_reference"}, True, False),
    ({"reference_lookup", "calculate"}, False, False),
    ({"calculate"}, False, False),
])
@pytest.mark.asyncio
async def test_reference_guidance_reaches_react_only_for_available_tools(
    names, guided, search_available,
):
    registry = ToolRegistry(by_name={name: ToolSpec(
        name=name, description=name, parameters={"type": "object", "properties": {}},
        server_url="http://unused", path=f"/{name}",
    ) for name in names})
    captured = []

    async def chat(**kwargs):
        captured.append(kwargs["messages"])
        return {"message": {"content": "Answer without a new tool call."}}

    incoming = [{"role": "system", "content": "Existing instructions"},
                {"role": "user", "content": "Compare the editions"}]
    await run_react(
        SimpleNamespace(chat=chat), SimpleNamespace(record_success=lambda _: None),
        registry, model="local-worker", messages=incoming, options={}, timeout_s=1,
        max_rounds=2, compress_after_round=3, max_tool_result_chars=3500,
        tool_dispatch_timeout_s=1,
    )
    assert len(captured) == 1  # No additional model/tool call is forced.
    assert captured[0][0] == incoming[0]
    assert captured[0][-1] == incoming[-1]
    guidance = [m["content"] for m in captured[0]
                if m["role"] == "system" and "Owned reference evidence" in m["content"]]
    assert bool(guidance) is guided
    if guided:
        assert "read relevant pages with read_reference" in guidance[0]
        assert "reference_lookup returns catalog metadata, not document contents" in guidance[0]
        assert ("reference_search" in guidance[0]) is search_available
    assert len(incoming) == 2  # The caller's saved transcript is unchanged.
