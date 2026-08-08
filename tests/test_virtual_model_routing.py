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
