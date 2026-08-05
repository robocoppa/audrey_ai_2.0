"""Tests for vision cost attribution (Phase 38).

Phase 36 shipped one number — 62.3s per keyframe, with a 4x spread it could
not explain — and phase 38 has to spend it. Every lever in the phase-38 plan
is a bet on which *part* of that number is large:

  - queue        → scheduling, or moving description off the local GPU
  - load         → `keep_alive`; batching and downscaling do nothing for it
  - prompt_eval  → smaller frames, or several frames per call
  - eval         → a shorter prompt and a `num_predict` cap

Those fixes are mutually exclusive in practice, so guessing wrong costs a
build and a deploy. `VisionTiming` reads the answer out of the response
Ollama was already sending.

The traps pinned here are the ones that would make the measurement lie rather
than fail: nanosecond arithmetic that silently reports milliseconds, and a
missing `total_duration` being attributed entirely to queueing — which would
manufacture the exact signal the metric exists to detect.
"""

from __future__ import annotations

from typing import Any

import pytest

from audrey.models.health import HealthTracker
from audrey.models.registry import ModelRegistry
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.pipeline.vision import VisionTiming, describe_one_image

# A real Ollama non-streaming response, trimmed to the timing fields. Every
# duration is in NANOSECONDS — this shape is the reason the conversion is
# tested rather than eyeballed.
_REAL_RESPONSE: dict[str, Any] = {
    "model": "qwen3-vl:32b",
    "message": {"role": "assistant", "content": "Two men seated in red chairs."},
    "done": True,
    "total_duration": 62_300_000_000,
    "load_duration": 4_100_000_000,
    "prompt_eval_count": 312,
    "prompt_eval_duration": 2_700_000_000,
    "eval_count": 486,
    "eval_duration": 55_200_000_000,
}


class TestNanosecondArithmetic:
    """The units are the whole risk. Off by 1000x and 62 seconds reads as 62
    milliseconds, which looks like the problem solved itself."""

    def test_durations_are_nanoseconds_and_come_back_as_seconds(self):
        t = VisionTiming.from_response(_REAL_RESPONSE)

        assert t.total_s == pytest.approx(62.3)
        assert t.load_s == pytest.approx(4.1)
        assert t.prompt_eval_s == pytest.approx(2.7)
        assert t.eval_s == pytest.approx(55.2)

    def test_counts_are_counts_not_durations(self):
        t = VisionTiming.from_response(_REAL_RESPONSE)

        assert t.prompt_tokens == 312
        assert t.eval_tokens == 486

    def test_the_stages_roughly_account_for_the_total(self):
        """Not exact — Ollama's total carries a little overhead the named
        stages don't — but a breakdown that summed to a third of the total
        would mean the fields were being read wrong."""
        t = VisionTiming.from_response(_REAL_RESPONSE)

        assert t.load_s + t.prompt_eval_s + t.eval_s == pytest.approx(t.total_s, rel=0.05)


class TestMissingAndMalformedTimings:
    """A timing that cannot be read must never fail a describe that worked.
    The description is the product; this is instrumentation."""

    def test_a_response_with_no_timing_fields_yields_zeros(self):
        t = VisionTiming.from_response({"message": {"content": "hi"}})

        assert (t.total_s, t.load_s, t.prompt_eval_s, t.eval_s) == (0.0, 0.0, 0.0, 0.0)
        assert (t.prompt_tokens, t.eval_tokens) == (0, 0)

    def test_an_empty_response_does_not_raise(self):
        assert VisionTiming.from_response({}) == VisionTiming()

    def test_null_fields_are_treated_as_absent(self):
        t = VisionTiming.from_response({"total_duration": None, "eval_count": None})

        assert t.total_s == 0.0
        assert t.eval_tokens == 0

    def test_a_non_numeric_duration_does_not_raise(self):
        t = VisionTiming.from_response({"total_duration": "ages", "eval_count": "lots"})

        assert t.total_s == 0.0
        assert t.eval_tokens == 0

    def test_a_negative_duration_is_floored_at_zero(self):
        """A negative stage would subtract from the attribution and make the
        breakdown stop summing to the wall clock."""
        t = VisionTiming.from_response({"load_duration": -5_000_000_000})

        assert t.load_s == 0.0


class TestQueueTime:
    """The one stage Ollama does not report, and the one that distinguishes
    "the model is slow" from "the model waited behind chat"."""

    def test_queue_is_the_wall_clock_ollama_cannot_account_for(self):
        t = VisionTiming.from_response(_REAL_RESPONSE)

        # 70s on the caller's stopwatch, 62.3s of it inside Ollama.
        assert t.queue_s(70.0) == pytest.approx(7.7)

    def test_no_queue_when_the_call_started_immediately(self):
        t = VisionTiming.from_response(_REAL_RESPONSE)

        assert t.queue_s(62.3) == pytest.approx(0.0)

    def test_a_wall_clock_under_the_total_does_not_go_negative(self):
        """Two clocks, one of them Ollama's, so they will disagree slightly.
        A negative observation would be rejected by the histogram anyway."""
        t = VisionTiming.from_response(_REAL_RESPONSE)

        assert t.queue_s(62.0) == 0.0

    def test_an_unreported_total_is_not_attributed_to_queueing(self):
        """The trap. `queue = wall - total` with `total == 0` reports the
        entire call as queue time, so a backend that omits its timings would
        show up as permanent gate contention — a fabricated diagnosis, and one
        that points at the most expensive lever in the plan."""
        t = VisionTiming.from_response({"message": {"content": "hi"}})

        assert t.queue_s(62.3) == 0.0


class _Cfg:
    def __init__(self) -> None:
        self.raw: dict[str, Any] = {"vision": {"timeout_s": 120}}
        self.model_registry = {
            "vl": [{"name": "qwen3-vl:32b", "priority": 100, "location": "local"}],
        }


class _Ollama:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    async def chat(self, *, model, messages, options=None, tools=None, timeout_s=None, think=None):
        return self._response


@pytest.mark.asyncio
class TestDescribeOneImageReturnsTiming:
    """The keyframe path. `describe_images` (chat) deliberately discards its
    timing — this is the one caller that has a pipeline to tune."""

    async def _describe(self, response: dict[str, Any]):
        cfg = _Cfg()
        return await describe_one_image(
            "data:image/jpeg;base64,AAAA",
            ollama=_Ollama(response),
            registry=ModelRegistry(cfg),  # type: ignore[arg-type]
            health=HealthTracker(),
            gate=FairLocalGate(concurrency=1),
            cfg=cfg,
            user_id="bart@proton.me",
        )

    async def test_the_timing_comes_back_with_the_description(self):
        description, model, timing = await self._describe(_REAL_RESPONSE)

        assert description == "Two men seated in red chairs."
        assert model == "qwen3-vl:32b"
        assert timing.eval_s == pytest.approx(55.2)
        assert timing.eval_tokens == 486

    async def test_a_backend_with_no_timings_still_describes(self):
        """Instrumentation must not be able to break the product."""
        description, _model, timing = await self._describe(
            {"message": {"content": "a whiteboard"}},
        )

        assert description == "a whiteboard"
        assert timing == VisionTiming()


class TestThinkingIsMeasuredNotInferred:
    """Phase 38's most expensive mistake, made twice: a describe call generated
    9,486 tokens to produce 12,490 characters, and that ratio was reasoned from
    — against an assumed ~4 chars/token — to a confident conclusion about
    thinking tokens. It was wrong, `think: false` changed nothing measurable,
    and the `num_predict` sized from the same bad arithmetic dropped three of
    six keyframes.

    Ollama returns reasoning in `message.thinking`, separate from
    `message.content`. Reading it is a lookup. There is no longer an excuse for
    inferring it.
    """

    def test_thinking_is_read_from_the_message(self):
        t = VisionTiming.from_response({
            "message": {"content": "Two men in chairs.", "thinking": "Let me look..."},
        })

        assert t.thinking_chars == len("Let me look...")

    def test_no_thinking_field_is_zero_not_unknown(self):
        """Zero has to mean 'the model returned none', because that is the
        reading that says thinking is genuinely off."""
        t = VisionTiming.from_response({"message": {"content": "Two men."}})

        assert t.thinking_chars == 0

    def test_a_null_thinking_field_is_zero(self):
        t = VisionTiming.from_response({
            "message": {"content": "Two men.", "thinking": None},
        })

        assert t.thinking_chars == 0

    def test_a_non_string_thinking_field_does_not_raise(self):
        t = VisionTiming.from_response({"message": {"content": "x", "thinking": 42}})

        assert t.thinking_chars == 0

    def test_a_response_with_no_message_does_not_raise(self):
        assert VisionTiming.from_response({"eval_count": 10}).thinking_chars == 0

    def test_a_non_dict_message_does_not_raise(self):
        assert VisionTiming.from_response({"message": "oops"}).thinking_chars == 0

    def test_the_real_response_shape_reports_no_thinking(self):
        """The fixture at the top of this file has no thinking field, which is
        what a call with `think: false` honoured looks like."""
        assert VisionTiming.from_response(_REAL_RESPONSE).thinking_chars == 0


class TestTruncationIsNamed:
    """A frame lost to `num_predict` and a frame lost to a broken vision model
    look identical from the worker: both are a 502 with an empty description.
    They want opposite responses — one is a one-line config change, the other
    is a deployment problem.

    Measured 2026-08-04: three of six probe runs on a cluttered office frame
    spent all 2,048 tokens reasoning and emitted zero characters. The cap has
    now been wrong twice in the same direction because nothing said so.
    """

    def test_done_reason_is_carried(self):
        t = VisionTiming.from_response({
            "message": {"content": ""}, "done_reason": "length", "eval_count": 2048,
        })

        assert t.done_reason == "length"
        assert t.eval_tokens == 2048

    def test_a_normal_stop_is_carried_too(self):
        t = VisionTiming.from_response({
            "message": {"content": "A slide."}, "done_reason": "stop",
        })

        assert t.done_reason == "stop"

    def test_a_missing_done_reason_is_empty_not_none(self):
        """It reaches an f-string in the 502 detail; `None` would render as
        the word 'None' in an error a human is meant to act on."""
        assert VisionTiming.from_response({"message": {"content": "x"}}).done_reason == ""
