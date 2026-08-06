"""The `get_file_text` tool, and the budget ladder it has to sit inside.

The tool is a thin proxy — the reading, paging and line-snapping all live in
Audrey's `/v1/files/artifact` route, which `tests/test_files_artifact.py` owns.
What is worth pinning here is the part that spans three files and can therefore
drift without anything failing:

    tools-server page size  <  Audrey's max_tool_result_chars

A page larger than the dispatcher's cap is cut mid-word on arrival, which is
exactly the failure the paging was built to prevent — reintroduced one layer
up, and invisible from either side on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import yaml

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import app as tools_app  # noqa: E402
from settings import settings  # noqa: E402

_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

_PAGE = {
    "filename": "jason retirement.mp4", "artifact": "transcript",
    "text": "[00:00:00] hello\n", "offset": 0, "next_offset": 17,
    "total_chars": 51_000, "note": "This is characters 0-17 of 51,000.",
}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://audrey-ai:8000",
    )


class TestTheBudgetLadder:
    def test_a_page_fits_inside_the_dispatcher_cap(self):
        cfg = yaml.safe_load(_CONFIG.read_text("utf-8"))
        cap = int(cfg["agentic"]["react"]["max_tool_result_chars"])
        # The rung that matters. Same species as the KB timeout ladder
        # (24s embed < 27s custom-tools < 30s dispatch): two services each
        # holding a number that only means something relative to the other's.
        assert settings.file_text_page_chars < cap, (
            f"a {settings.file_text_page_chars}-char page cannot survive a "
            f"{cap}-char tool-result cap; the transcript would be cut mid-word "
            "on arrival despite the route ending it on a line boundary"
        )

    def test_there_is_headroom_for_the_json_envelope(self):
        cfg = yaml.safe_load(_CONFIG.read_text("utf-8"))
        cap = int(cfg["agentic"]["react"]["max_tool_result_chars"])
        # `text` travels inside a JSON object carrying filename, note, offsets
        # and totals, and JSON-escaping a transcript's newlines expands it.
        # 20% is a judgement, but zero headroom is definitely wrong.
        assert settings.file_text_page_chars <= cap * 0.8


class TestTheProxy:
    async def test_the_model_cannot_choose_the_page_size(self):
        sent = {}

        def handler(request: httpx.Request) -> httpx.Response:
            sent.update(**__import__("json").loads(request.content))
            return httpx.Response(200, json=_PAGE)

        tools_app.app.state.audrey = _client(handler)
        await tools_app.get_file_text(tools_app.GetFileTextRequest(
            user="a@b.c", filename="jason retirement.mp4",
        ))
        # `limit` is fixed by the sidecar, not exposed as a tool argument: a
        # model-supplied limit could not respect a cap it cannot see.
        assert sent["limit"] == settings.file_text_page_chars
        assert "limit" not in tools_app.GetFileTextRequest.model_fields

    async def test_the_offset_and_artifact_are_passed_through(self):
        sent = {}

        def handler(request: httpx.Request) -> httpx.Response:
            sent.update(**__import__("json").loads(request.content))
            return httpx.Response(200, json=_PAGE)

        tools_app.app.state.audrey = _client(handler)
        await tools_app.get_file_text(tools_app.GetFileTextRequest(
            user="a@b.c", filename="x.mp4", artifact="visual", offset=4000,
        ))
        assert sent["offset"] == 4000
        assert sent["artifact"] == "visual"

    async def test_the_paging_fields_reach_the_model(self):
        tools_app.app.state.audrey = _client(
            lambda _r: httpx.Response(200, json=_PAGE))
        out = await tools_app.get_file_text(tools_app.GetFileTextRequest(
            user="a@b.c", filename="x.mp4",
        ))
        # Without these the model is back to guessing how much it is missing,
        # which is the whole failure this tool exists to end.
        assert out.next_offset == 17
        assert out.total_chars == 51_000
        assert out.note

    async def test_an_audrey_error_is_reported_not_swallowed(self):
        tools_app.app.state.audrey = _client(
            lambda _r: httpx.Response(404, text="no such file"))
        with pytest.raises(Exception) as exc:
            await tools_app.get_file_text(tools_app.GetFileTextRequest(
                user="a@b.c", filename="ghost.mp4",
            ))
        assert "404" in str(exc.value) or "no such file" in str(exc.value)

    async def test_an_unreachable_audrey_is_a_502(self):
        def handler(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        tools_app.app.state.audrey = _client(handler)
        with pytest.raises(Exception) as exc:
            await tools_app.get_file_text(tools_app.GetFileTextRequest(
                user="a@b.c", filename="x.mp4",
            ))
        assert "502" in str(exc.value) or "unreachable" in str(exc.value)


class TestTheDescription:
    """The description is the only part of a tool a model actually reads."""

    def test_it_steers_away_from_kb_search_for_whole_documents(self):
        desc = _tool_description()
        # The observed failure was reaching for kb_search to get a transcript.
        assert "kb_search" in desc
        assert "transcript" in desc

    def test_it_explains_how_to_page(self):
        desc = _tool_description()
        assert "next_offset" in desc
        assert "offset" in desc

    def test_it_tells_the_model_not_to_imply_it_has_everything(self):
        # The precise thing that went wrong: a model presenting one fragment as
        # all it could retrieve, without saying how much was left.
        assert "total_chars" in _tool_description()


def _tool_description() -> str:
    for route in tools_app.app.routes:
        if getattr(route, "path", "") == "/get_file_text":
            return route.description or ""
    raise AssertionError("get_file_text route not found")
