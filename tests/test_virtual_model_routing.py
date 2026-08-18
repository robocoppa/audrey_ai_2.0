"""Guards for the virtual-model → routing-mode contract (phase 42).

The deep-vs-fast gate is implemented **twice**: as `node_complexity` in
`pipeline/graph.py` for non-streaming runs, and inline in
`_stream_via_pipeline` in `routes/openai/pipeline.py` for streaming ones.
`graph.py` already carries a comment saying the two must change together;
these tests make that mechanical, because the failure mode is invisible from
the outside — a virtual model that routes one way when the client asked for a
stream and the other way when it didn't, which no user can see or report
usefully.

Source-level rather than behavioural on purpose: the gate is inline in both
places, not a shared function, so there is nothing to import and call. If it
is ever extracted into one helper, delete these and test the helper instead —
that would be strictly better.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from audrey.pipeline.prompts import task_role_for
from audrey.routes.openai.routes import VIRTUAL_MODELS

_SRC = Path(__file__).resolve().parents[1] / "src" / "audrey"
_GRAPH = _SRC / "pipeline" / "graph.py"
_STREAM = _SRC / "routes" / "openai" / "pipeline.py"

_FORCED_DEEP = re.compile(r"forced_deep\s*=\s*[\w.]+\s+in\s+\(([^)]*)\)")
_FORCED_FAST = re.compile(r'forced_fast\s*=\s*[\w.]+\s*==\s*"([^"]+)"')


def _forced_deep(path: Path) -> set[str]:
    m = _FORCED_DEEP.search(path.read_text())
    assert m, f"no forced_deep tuple found in {path.name} — did the gate move?"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _forced_fast(path: Path) -> str:
    m = _FORCED_FAST.search(path.read_text())
    assert m, f"no forced_fast check found in {path.name} — did the gate move?"
    return m.group(1)


def test_both_gates_force_deep_for_the_same_models():
    assert _forced_deep(_GRAPH) == _forced_deep(_STREAM)


def test_both_gates_force_fast_for_the_same_model():
    assert _forced_fast(_GRAPH) == _forced_fast(_STREAM)


@pytest.mark.parametrize("path", [_GRAPH, _STREAM], ids=["graph", "streaming"])
def test_audrey_video_is_not_forced_into_either_mode(path):
    """`audrey_video` routes adaptively, exactly like `audrey_auto`.

    Forcing it deep would put an ordinary file lookup on paid panel
    inference; forcing it fast would stop a genuinely long request from
    escalating. Both are decided by falling through to the token count.
    """
    assert "audrey_video" not in _forced_deep(path)
    assert _forced_fast(path) != "audrey_video"


@pytest.mark.parametrize("path", [_GRAPH, _STREAM], ids=["graph", "streaming"])
def test_forced_lists_only_name_real_virtual_models(path):
    unknown = _forced_deep(path) - set(VIRTUAL_MODELS)
    assert not unknown, f"{path.name} forces unknown model(s): {unknown}"
    assert _forced_fast(path) in VIRTUAL_MODELS


def test_audrey_video_is_exposed():
    assert "audrey_video" in VIRTUAL_MODELS


def test_exactly_one_virtual_model_carries_a_task_role():
    """Two hand-coded specialists is the documented signal to build the
    config-driven `specialists:` block instead of adding a third by hand."""
    with_role = [vm for vm in VIRTUAL_MODELS if task_role_for(vm) is not None]
    assert with_role == ["audrey_video"]


# ─── The gate must measure the request, not Audrey's own scaffolding ──


_GATE_CALL = re.compile(r"is_complex\(\s*(\w+)\s*,\s*threshold=")


@pytest.mark.parametrize("path", [_GRAPH, _STREAM], ids=["graph", "streaming"])
def test_the_deep_gate_excludes_the_injected_task_role(path):
    """Both gates must feed `is_complex` a task-role-stripped message list.

    The bug this pins (2026-08-09, caught only from on-box logs): the task role
    is injected at the route, `count_tokens` sums system messages too, and so a
    specialist spent ~330 tokens of its own prompt against a 500-token
    threshold. Turns that should have run fast went to the three-worker panel —
    which also means an A-B against `audrey_auto` silently compares pipelines
    rather than prompts.

    Source-level for the same reason as the rest of this file: the gate is
    inline in both places. Asserting on the variable name passed to
    `is_complex` is crude, but it fails loudly if either gate is rewritten to
    pass the raw list again, and that is exactly the regression worth catching.
    """
    src = path.read_text()
    m = _GATE_CALL.search(src)
    assert m, f"no is_complex(...) gate call found in {path.name} — did it move?"
    assert m.group(1) == "gate_messages", (
        f"{path.name} gates on {m.group(1)!r}, not the task-role-stripped list"
    )
    assert "without_task_role(" in src, (
        f"{path.name} never strips the task role before gating"
    )


def test_every_model_that_can_route_deep_has_a_panel_pool():
    """A virtual model reaching the panel must be registered in `_POOL_KEYS`.

    Not about the pool it lands in — the fallback already picks the right one
    for the adaptive models. It is about the warning staying meaningful: while
    `audrey_auto` and `audrey_video` were unregistered, "unknown virtual_model"
    fired on ordinary correct traffic, so the case it exists to catch (a new
    specialist shipped without a pool) was indistinguishable from noise.

    `audrey_fast` is excluded because `forced_fast` means it can never arrive.
    """
    from audrey.pipeline.deep_panel import _POOL_KEYS

    can_route_deep = set(VIRTUAL_MODELS) - {"audrey_fast"}
    missing = can_route_deep - set(_POOL_KEYS)
    assert not missing, f"virtual model(s) can reach the panel unregistered: {missing}"


# ─── The sidecar call: both gates must declare "no single target" ──────


_DESCRIBE_CALL = re.compile(r"describe_for_text_model\((.*?)^\s*\)", re.S | re.M)
_TARGET_KWARG = re.compile(r'target_model\s*=\s*(".*?"|[\w.\[\]"\']+)')


@pytest.mark.parametrize("path", [_GRAPH, _STREAM], ids=["graph", "streaming"])
def test_both_pipelines_call_the_sidecar_with_no_target_model(path):
    """Both deep paths must pass `target_model=""`, and it is not a placeholder.

    `describe_for_text_model` short-circuits on `if target_model and
    is_vision_capable(target_model, cfg)`. Passing an empty string is what makes
    a deep pick transcribe unconditionally — correct, because a panel is
    several workers of mixed capability plus a synthesizer and there is no one
    model whose eyes could be tested.

    ⚠️ The regression this catches looks like a FIX. Threading the picked model
    through reads as obviously better, and it would silently stop transcribing
    for any model named in `vision.also_capable` — on the one image path that
    is known to work, with no error and no log line saying so. The answer would
    just quietly get worse for whichever worker happened to be asked.

    Source-level for the same reason as the rest of this file: the call is
    inline in both places, so there is nothing to import and call.
    """
    m = _DESCRIBE_CALL.search(path.read_text())
    assert m, f"no describe_for_text_model(...) call found in {path.name} — did it move?"
    t = _TARGET_KWARG.search(m.group(1))
    assert t, f"{path.name} calls the sidecar without an explicit `target_model=`"
    assert t.group(1) == '""', (
        f"{path.name} passes target_model={t.group(1)} — a deep pick has no single "
        "target model, and a non-empty one lets `vision.also_capable` suppress "
        "transcription on the deep path. See test_vision_sidecar.py."
    )
