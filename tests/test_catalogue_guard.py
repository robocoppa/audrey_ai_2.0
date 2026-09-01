"""The catalogue-only guard (2026-08-12).

`list_my_files` carries filenames and WHICH artifacts exist, never a word of
what any of them says. A turn that stops with it as its only successful call
and then describes what a file CONTAINS invented that content.

Three text levers had already been aimed at this (tool description, task-role
prompt, and renaming the row field `artifacts` → `unread_artifacts`). The third
cut it from ~71% of these turns to ~17%, and then two turns of the same case
holding **byte-identical context** produced one invented championship result
and one correct "I have not read it yet". Identical input, opposite output —
so the residue is sampling, and no fourth sentence can reach it. The guard
fetches the file instead of asking again.

The skip cases below are not hypotheticals. Each is the biggest real group in a
sweep of 4,358 archived answers, and each would be actively made worse by
firing: `silent.mp4` has nothing to fetch, and a model that correctly asked
"which of your two London videos?" would be pushed into picking one.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from audrey.pipeline.react import _catalogue_rows, _unread_fetch, run_react
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec
from audrey.tools.dispatch import ToolResult

GRACIE = "Roger Gracie VS Rafael Lovato Jr _ World Championship 2009.mp4"
LONDON = "How to WIN with the London System.mp4"
MAGNUS = "Magnus Carlsen Teaches How to Win with the London System.mp4"


def _row(filename: str, unread: list[str]) -> dict:
    return {"filename": filename, "kind": "video", "status": "ready",
            "uploaded_at": "2026-08-01", "duration_s": 290.0,
            "unread_artifacts": unread}


def _listing(*rows: dict, error: bool = False) -> ToolResult:
    return ToolResult(name="list_my_files", call_id="c1",
                      content=json.dumps({"files": list(rows)}),
                      elapsed_s=0.1, is_error=error)


def _ok(name: str) -> ToolResult:
    return ToolResult(name=name, call_id="c2", content="{}",
                      elapsed_s=0.1, is_error=False)


# ─── when it fires ────────────────────────────────────────────────────


def test_it_fires_on_the_defect():
    """The measured case: catalogue call only, one named file, artifacts
    unread. 34 of the 35 archive-wide fires look exactly like this."""
    answer = f"The video {GRACIE} captures a classic match ending in a choke."
    assert _unread_fetch(answer, [_listing(_row(GRACIE, ["summary", "visual"]))]) \
        == (GRACIE, "summary")


def test_summary_is_preferred_over_transcript():
    """These turns ask what a file is ABOUT. A summary answers that in one
    page; a transcript needs several and `max_rounds` will not buy them."""
    got = _unread_fetch(f"{LONDON} is a chess tutorial.",
                        [_listing(_row(LONDON, ["transcript", "summary"]))])
    assert got == (LONDON, "summary")


def test_it_falls_back_to_whatever_is_unread():
    got = _unread_fetch(f"{LONDON} is a chess tutorial.",
                        [_listing(_row(LONDON, ["visual"]))])
    assert got == (LONDON, "visual")


# ─── when it must not ─────────────────────────────────────────────────


def test_nothing_unread_means_nothing_to_fetch():
    """`silent.mp4` — 44 of 109 catalogue-only turns in the archive. The
    honest answer IS "no transcript", and there is no artifact to go get."""
    answer = "silent.mp4 has no transcript — it appears to have no speech."
    assert _unread_fetch(answer, [_listing(_row("silent.mp4", []))]) is None


def test_two_named_files_is_a_disambiguation_not_a_claim():
    """18 archived turns say "I found two London System videos — which?".
    That is correct behaviour. Firing here would push a model that rightly
    asked into guessing one."""
    answer = f"I found two London System videos:\n1. {LONDON}\n2. {MAGNUS}\nWhich did you mean?"
    assert _unread_fetch(answer, [
        _listing(_row(LONDON, ["summary"]), _row(MAGNUS, ["summary"])),
    ]) is None


def test_a_file_that_is_not_in_the_catalogue_is_not_fetched():
    """"What did they say in teamOffsite2025.mp4?" when no such file exists.
    Nothing matches, so nothing is fetched."""
    answer = "I could not find teamOffsite2025.mp4 in your uploads."
    assert _unread_fetch(answer, [_listing(_row(LONDON, ["summary"]))]) is None


def test_it_stands_down_when_any_other_tool_succeeded():
    """The premise is that NO tool returned contents. One successful
    `kb_search` breaks it, and the answer may be perfectly grounded."""
    answer = f"The video {GRACIE} shows a match ending in a choke."
    assert _unread_fetch(answer, [
        _listing(_row(GRACIE, ["summary"])), _ok("kb_search"),
    ]) is None


def test_a_failed_catalogue_call_is_not_a_success():
    answer = f"The video {GRACIE} shows a choke."
    assert _unread_fetch(answer, [_listing(_row(GRACIE, ["summary"]), error=True)]) is None


def test_an_unparseable_listing_stands_down(caplog):
    """A listing cut at `max_tool_result_chars` arrives as invalid JSON. A
    guard that cannot read the catalogue must let the answer stand rather
    than fetch a file chosen from half a row."""
    truncated = ToolResult(name="list_my_files", call_id="c1",
                           content='{"files": [{"filename": "How to W',
                           elapsed_s=0.1, is_error=False)
    with caplog.at_level(logging.INFO, logger="audrey.pipeline.react"):
        assert _unread_fetch(f"{LONDON} is about chess.", [truncated]) is None
    assert any("standing down" in r.message for r in caplog.records)


def test_no_tools_at_all_does_not_fire():
    assert _unread_fetch("Some answer with no tools.", []) is None


@pytest.mark.parametrize("body", ['{"files": "nope"}', "[]", "null", '{"x": 1}'])
def test_catalogue_rows_survives_odd_shapes(body):
    rows = _catalogue_rows([ToolResult(name="list_my_files", call_id="c",
                                       content=body, elapsed_s=0.0, is_error=False)])
    assert rows == []


# ─── the loop, end to end ─────────────────────────────────────────────
#
# ⚠️ These drive the REAL `dispatch_one` over a mock transport rather than a
# stubbed dispatcher, and that is deliberate. The guard builds a `tool_call`
# dict by hand — no `id`, arguments as a dict rather than a JSON string — and a
# stub would accept any shape at all. On 2026-08-12 a hand-edited double in
# this suite accepted a kwarg the real client rejected and took passthrough
# down for 35 minutes while every test stayed green. The only layer that can
# vouch for a synthetic call is the one that dispatches it.


#: ⚠️ Captured BEFORE any patching. `audrey.pipeline.react.httpx` is the httpx
#: module itself, so `setattr(..., "AsyncClient", f)` rebinds it for everyone —
#: including the replacement, which would then call itself forever.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch_client(monkeypatch, handler) -> None:
    monkeypatch.setattr(
        "audrey.pipeline.react.httpx.AsyncClient",
        lambda *a, **k: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler)),
    )


class _FakeHealth:
    def record_success(self, model): pass
    def record_failure(self, model, err): pass


class _FakeOllama:
    """Replays scripted responses; records what it was shown each round."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen: list[list[dict]] = []

    async def chat(self, *, model, messages, options=None, tools=None,
                   timeout_s=0, think=None):
        self.seen.append(list(messages))
        return self._responses.pop(0)


def _registry() -> ToolRegistry:
    def make_spec(name: str) -> ToolSpec:
        declaration = TOOL_DECLARATIONS[name]
        return ToolSpec(
            name=name,
            description=name,
            parameters={"type": "object", "properties": {}},
            server_url="http://tools",
            path=f"/{name}",
            user_scope=declaration.user_scope,
            dependencies=declaration.dependencies,
            purge_gated=declaration.purge_gated,
        )

    names = ("list_my_files", "get_file_text")
    return ToolRegistry(by_name={name: make_spec(name) for name in names})


@pytest.fixture
def mock_tools(monkeypatch):
    """Patch the client `run_react` builds, so real dispatch hits a fake socket."""
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        args = json.loads(request.content)
        calls.append((request.url.path, args))
        if request.url.path == "/list_my_files":
            return httpx.Response(200, json={"files": [_row(GRACIE, ["summary", "visual"])]})
        return httpx.Response(200, json={
            "filename": args["filename"], "artifact": args["artifact"],
            "text": "Two black belts, a red mat. No result is announced.",
            "offset": 0, "next_offset": None, "total_chars": 51,
        })

    _patch_client(monkeypatch, handler)
    return calls


async def _run(ollama, guard: bool = True):
    return await run_react(
        ollama, _FakeHealth(), _registry(),
        model="m", messages=[{"role": "user", "content": "Summarise my latest upload."}],
        options={}, timeout_s=5, max_rounds=3, compress_after_round=99,
        max_tool_result_chars=4000, tool_dispatch_timeout_s=5,
        user_id="alice@example.com", location="cloud",
        cfg=SimpleNamespace(raw={"agentic": {"react": {"catalogue_guard": guard}}}),
    )


def _catalogue_then(answer: str, *rest):
    """Round 0 calls list_my_files; round 1 answers; then `rest`."""
    return [
        {"message": {"content": "", "tool_calls": [
            {"id": "a", "function": {"name": "list_my_files", "arguments": {}}}]}},
        {"message": {"content": answer}, "prompt_eval_count": 1, "eval_count": 1},
        *rest,
    ]


async def test_the_guard_fetches_and_the_model_answers_again(mock_tools, caplog):
    """The whole point: a fabricated answer is not returned. The file is
    fetched and the model writes a second answer with the text in context."""
    ollama = _FakeOllama(_catalogue_then(
        f"{GRACIE} ends with a rear-naked choke in the -98kg final.",
        {"message": {"content": "The clip shows two black belts on a red mat; "
                                "no result is announced."},
         "prompt_eval_count": 1, "eval_count": 1},
    ))
    with caplog.at_level(logging.INFO, logger="audrey.pipeline.react"):
        out = await _run(ollama)

    assert "no result is announced" in out.content
    assert "rear-naked choke" not in out.content
    # The forced call really happened, against the real dispatcher.
    assert ("/get_file_text", {"filename": GRACIE, "artifact": "summary",
                               "user": "alice@example.com"}) in mock_tools
    assert any("catalogue-guard" in r.message for r in caplog.records)


async def test_the_fetched_text_is_in_front_of_the_model(mock_tools):
    """Fetching is worthless if the content never reaches the next prompt."""
    ollama = _FakeOllama(_catalogue_then(
        f"{GRACIE} ends with a choke.",
        {"message": {"content": "second"}, "prompt_eval_count": 1, "eval_count": 1},
    ))
    await _run(ollama)
    final = "".join(m.get("content", "") for m in ollama.seen[-1])
    assert "Two black belts, a red mat" in final
    assert "you have not read" in final


async def test_the_footer_counts_the_forced_call(mock_tools):
    """`tool_calls` drives the user-visible tools footer and the `grounded`
    eval check. The content genuinely reached the model, so recording it is
    the honest thing — and a footer that hid it would leave `grounded`
    flagging an answer that is now correctly sourced."""
    ollama = _FakeOllama(_catalogue_then(
        f"{GRACIE} ends with a choke.",
        {"message": {"content": "second"}, "prompt_eval_count": 1, "eval_count": 1},
    ))
    out = await _run(ollama)
    assert [r.name for r in out.tool_calls] == ["list_my_files", "get_file_text"]


async def test_it_fires_at_most_once(mock_tools):
    """A model that ignores the fetched text and repeats itself must not put
    the loop into a fetch/answer cycle."""
    repeat = f"{GRACIE} ends with a choke."
    ollama = _FakeOllama(_catalogue_then(
        repeat,
        {"message": {"content": repeat}, "prompt_eval_count": 1, "eval_count": 1},
    ))
    out = await _run(ollama)
    assert out.content == repeat
    assert sum(1 for p, _ in mock_tools if p == "/get_file_text") == 1


async def test_the_config_flag_turns_it_off(mock_tools):
    """It ships behind a knob so it can be A-B'd against the same cases."""
    fabricated = f"{GRACIE} ends with a rear-naked choke."
    ollama = _FakeOllama(_catalogue_then(fabricated))
    out = await _run(ollama, guard=False)
    assert out.content == fabricated
    assert not [p for p, _ in mock_tools if p == "/get_file_text"]


async def test_a_failed_fetch_lets_the_answer_stand(monkeypatch, caplog):
    """If the artifact route is down, the turn degrades to today's behaviour
    rather than losing the answer entirely."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/list_my_files":
            return httpx.Response(200, json={"files": [_row(GRACIE, ["summary"])]})
        return httpx.Response(503, text="artifact store unavailable")

    _patch_client(monkeypatch, handler)
    answer = f"{GRACIE} ends with a choke."
    with caplog.at_level(logging.WARNING, logger="audrey.pipeline.react"):
        out = await _run(_FakeOllama(_catalogue_then(answer)))
    assert out.content == answer
    assert any("letting the answer stand" in r.message for r in caplog.records)
