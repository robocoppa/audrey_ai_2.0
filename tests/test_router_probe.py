"""`router_probe` measures whether a candidate can hold the router slot.

The router is the least forgiving slot in the config: hot path, ungated against
the GPU, and its output has to parse or `classify` falls through to
`("general", "fallback:general", 0.25)` — served, silently misrouted.

Two properties are pinned harder than the arithmetic:

1. **The probe uses production's parser and prompt, not its own copy.** A probe
   with a private reimplementation can pass while production fails, which is the
   one outcome that makes it worse than no probe.
2. **A timid-but-accurate model is reported as a problem.** The escalation
   trigger is `conf < 0.95` strictly, so a router that routes perfectly at 0.8
   sends nearly every turn into a deep panel — three cloud calls each, against a
   hard budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import router_probe as rp  # noqa: E402


class TestItExercisesTheRealClassifier:
    def test_it_calls_productions_router_classify(self, monkeypatch):
        # Not a reimplementation: the probe must go through the same function
        # the pipeline uses, so the prompt and the parser cannot drift apart.
        import audrey.pipeline.classify as classify

        seen: list[str] = []

        async def _fake(ollama, *, router_model, user_text, timeout_s, cfg=None,
                        response_format=None):
            seen.append(user_text)
            return "general", 0.97, '{"task":"general","confidence":0.97}'

        monkeypatch.setattr(classify, "router_classify", _fake)
        import asyncio
        r = asyncio.run(rp.probe_model(None, "m:1b", 1, 20.0, None))
        assert len(seen) == len(rp.CASES)
        assert r["parsed"] == len(rp.CASES)

    def test_the_real_parser_rejects_prose(self):
        # The failure mode a small model actually exhibits: it chats.
        from audrey.pipeline.classify import _parse_router_output
        assert _parse_router_output("Sure! This looks like a coding question.") == (None, 0.0)

    def test_the_real_parser_accepts_a_fenced_block(self):
        from audrey.pipeline.classify import _parse_router_output
        task, conf = _parse_router_output('```json\n{"task":"code","confidence":0.9}\n```')
        assert (task, conf) == ("code", 0.9)

    def test_an_out_of_vocabulary_task_is_a_parse_failure(self):
        from audrey.pipeline.classify import _parse_router_output
        assert _parse_router_output('{"task":"chitchat","confidence":0.99}') == (None, 0.0)


class TestSchemaPinningIsOptOut:
    """Production behaviour must be untouched until someone deliberately opts in."""

    def test_router_classify_defaults_to_no_schema(self):
        import inspect

        from audrey.pipeline.classify import router_classify
        assert inspect.signature(router_classify).parameters["response_format"].default is None

    def test_the_schema_constrains_task_to_the_valid_set(self):
        # The enum is the load-bearing part: an out-of-vocabulary task becomes
        # impossible to emit, instead of being caught and thrown away after.
        from audrey.pipeline.classify import _VALID_TASKS, ROUTER_SCHEMA
        assert set(ROUTER_SCHEMA["properties"]["task"]["enum"]) == _VALID_TASKS
        assert ROUTER_SCHEMA["required"] == ["task", "confidence"]

    def test_the_schema_reaches_ollama_as_format(self):
        import asyncio

        import audrey.pipeline.classify as classify
        from audrey.pipeline.classify import ROUTER_SCHEMA

        seen: dict = {}

        class _Ollama:
            async def chat(self, **kw):
                seen.update(kw)
                return {"message": {"content": '{"task":"code","confidence":0.99}'}}

        asyncio.run(classify.router_classify(
            _Ollama(), router_model="m:1b", user_text="hi", timeout_s=5,
            response_format=ROUTER_SCHEMA,
        ))
        assert seen["format"] is ROUTER_SCHEMA

    def test_omitting_it_sends_no_format(self):
        import asyncio

        import audrey.pipeline.classify as classify

        seen: dict = {}

        class _Ollama:
            async def chat(self, **kw):
                seen.update(kw)
                return {"message": {"content": '{"task":"code","confidence":0.99}'}}

        asyncio.run(classify.router_classify(
            _Ollama(), router_model="m:1b", user_text="hi", timeout_s=5,
        ))
        assert seen["format"] is None

    def test_the_probe_labels_the_pinned_arm(self, monkeypatch):
        import asyncio

        import audrey.pipeline.classify as classify
        from audrey.pipeline.classify import ROUTER_SCHEMA

        async def _ok(ollama, *, router_model, user_text, timeout_s, cfg=None,
                     response_format=None):
            return "general", 0.99, "{}"

        monkeypatch.setattr(classify, "router_classify", _ok)
        pinned = asyncio.run(rp.probe_model(None, "m:1b", 1, 20.0, None, ROUTER_SCHEMA))
        plain = asyncio.run(rp.probe_model(None, "m:1b", 1, 20.0, None, None))
        assert "schema-pinned" in pinned["model"]
        assert "schema-pinned" not in plain["model"]


class TestCases:
    def test_every_case_expects_a_task_the_parser_accepts(self):
        from audrey.pipeline.classify import _VALID_TASKS
        for prompt, expected in rp.CASES:
            assert expected in _VALID_TASKS, prompt

    def test_vl_is_deliberately_absent(self):
        # Image turns are pinned to the vl pool before classify runs, so the
        # router never has to produce `vl` in production. Adding a vl case here
        # would measure something the router is not asked to do.
        assert not any(e == "vl" for _, e in rp.CASES)

    def test_all_three_live_tasks_are_covered(self):
        assert {e for _, e in rp.CASES} == {"code", "reasoning", "general"}


class TestScoring:
    def _result(self, **over):
        base = {
            "model": "m:1b", "total": 10, "parsed": 10, "correct": 9,
            "parse_rate": 1.0, "accuracy": 0.9, "conf_median": 0.96,
            "conf_at_or_above_ceiling": 9, "latency_median": 0.4,
            "latency_max": 0.9, "failures": [],
        }
        base.update(over)
        return base

    def test_accuracy_is_over_calls_that_parsed_not_all_calls(self):
        # Mixing them would flatter a model that failed to answer half the time.
        import asyncio

        import audrey.pipeline.classify as classify

        calls = {"n": 0}

        async def _half(ollama, *, router_model, user_text, timeout_s, cfg=None,
                        response_format=None):
            calls["n"] += 1
            if calls["n"] % 2:
                return None, 0.0, "parse_error:blah"
            return "general", 0.99, "{}"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(classify, "router_classify", _half)
            r = asyncio.run(rp.probe_model(None, "m:1b", 1, 20.0, None))
        assert r["parsed"] == 5
        assert r["parse_rate"] == 0.5
        # 3 of the 5 that parsed happen to be `general` cases.
        assert r["accuracy"] == r["correct"] / 5

    def test_a_low_parse_rate_is_disqualifying_in_the_report(self):
        out = rp.render(self._result(parse_rate=0.4, parsed=4), 0.95)
        assert "DISQUALIFIED" in out
        assert "fallback:general" in out

    def test_a_timid_router_is_flagged_as_a_cost_problem(self):
        # Perfect accuracy, nothing clears the ceiling -> escalates constantly.
        out = rp.render(
            self._result(accuracy=1.0, correct=10, conf_median=0.80,
                         conf_at_or_above_ceiling=0),
            0.95,
        )
        assert "COST" in out
        assert "escalate" in out

    def test_a_good_candidate_gets_no_warnings(self):
        out = rp.render(self._result(), 0.95)
        assert "DISQUALIFIED" not in out
        assert "COST" not in out

    def test_failures_are_truncated_but_counted(self):
        fails = [(f"p{i}", "routed general, wanted code") for i in range(9)]
        out = rp.render(self._result(failures=fails), 0.95)
        assert "and 3 more" in out


class TestEntryPoint:
    def test_no_model_is_a_usage_error(self, monkeypatch, capsys):
        import asyncio
        monkeypatch.delenv("MODEL", raising=False)
        assert asyncio.run(rp.amain()) == 2
        assert "set MODEL" in capsys.readouterr().err

    def test_blank_model_entries_are_ignored(self, monkeypatch, capsys):
        import asyncio
        monkeypatch.setenv("MODEL", " , ,")
        assert asyncio.run(rp.amain()) == 2

    def test_the_parse_floor_is_strict_enough_to_matter(self):
        # A router failing 1 call in 5 would still be routing a fifth of all
        # traffic to `fallback:general`. The floor has to sit above that.
        assert rp._PARSE_FLOOR >= 0.9


class TestGpuGatingIsDocumentedNotAssumed:
    def test_classify_with_registry_still_takes_no_gate(self):
        # The probe's central warning rests on this. If a gate parameter is
        # ever added, the warning becomes wrong and this test says so.
        import inspect

        from audrey.pipeline.classify import classify_with_registry
        params = inspect.signature(classify_with_registry).parameters
        assert "gate" not in params, (
            "classify is now gated — update router_probe's warning about the "
            "router evicting the deep worker"
        )


@pytest.mark.parametrize("raw,want", [
    ('{"task":"code","confidence":1.5}', ("code", 1.0)),      # clamped
    ('{"task":"code","confidence":-3}', ("code", 0.0)),       # clamped
    ('{"task":"code","confidence":"high"}', ("code", 0.5)),   # unparseable -> 0.5
    ('{"task":"code"}', ("code", 0.5)),                       # absent -> 0.5
])
def test_confidence_edge_cases_match_production(raw, want):
    # These land straight in the escalation decision, so the probe's confidence
    # column has to mean exactly what production's does.
    from audrey.pipeline.classify import _parse_router_output
    assert _parse_router_output(raw) == want
