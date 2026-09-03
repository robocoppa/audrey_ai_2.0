"""Contracts for Audrey's client-neutral run-event vocabulary."""

from __future__ import annotations

import json

import pytest

from audrey.pipeline.run_events import (
    RUN_EVENT_ADAPTER,
    RunEvent,
    RunEventEmitter,
    RunEventProtocolError,
    dump_run_event,
)
from audrey.routes.openai.streaming import OpenAIStreamSession, StreamOutcome


def _emitter(events: list[RunEvent]) -> RunEventEmitter:
    return RunEventEmitter(
        run_id="run_test",
        conversation_id="con_test",
        assistant_message_id="msg_assistant",
        mode="research",
        virtual_model="audrey_research",
        sink=events.append,
    )


def test_complete_event_vocabulary_is_ordered_and_round_trips() -> None:
    events: list[RunEvent] = []
    emitter = _emitter(events)

    emitter.run_started()
    emitter.message_started()
    emitter.stage_started("researching", label="Researching")
    emitter.stage_progress(".", stage="researching")
    emitter.tool_started("call_1", name="web_search")
    emitter.tool_arguments("call_1", arguments={"query": "Euclid"})
    emitter.tool_finished("call_1", status="succeeded", result={"count": 2})
    emitter.source_observed(
        "source_1",
        title="Euclid",
        url="https://example.com/euclid",
        source_type="reference",
    )
    emitter.stage_finished("researching", status="succeeded")
    emitter.text_delta("answer")
    emitter.usage_reported(prompt_tokens=10, completion_tokens=2)
    emitter.message_finished(status="completed")
    emitter.run_finished(
        status="succeeded",
        finish_reason="stop",
        concrete_model="writer",
    )

    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert len({event.event_id for event in events}) == len(events)
    assert [event.type for event in events] == [
        "run.started",
        "message.started",
        "stage.started",
        "stage.progress",
        "tool.started",
        "tool.arguments",
        "tool.finished",
        "source.observed",
        "stage.finished",
        "text.delta",
        "usage.reported",
        "message.finished",
        "run.finished",
    ]
    wire = [dump_run_event(event) for event in events]
    restored = [RUN_EVENT_ADAPTER.validate_python(item) for item in wire]
    assert restored == events
    assert wire[-1]["status"] == "succeeded"
    assert wire[-1]["virtual_model"] == "audrey_research"
    assert "user_id" not in json.dumps(wire)


def test_emitter_rejects_out_of_order_and_duplicate_transitions() -> None:
    events: list[RunEvent] = []
    emitter = _emitter(events)

    with pytest.raises(RunEventProtocolError, match="run has not started"):
        emitter.message_started()

    emitter.run_started()
    with pytest.raises(RunEventProtocolError, match="already started"):
        emitter.run_started()
    with pytest.raises(RunEventProtocolError, match="message is not active"):
        emitter.text_delta("too early")

    emitter.message_started()
    emitter.tool_started("call_1", name="kb_search")
    with pytest.raises(RunEventProtocolError, match="not active"):
        emitter.tool_finished("call_1", status="failed", error="too early")
    emitter.tool_arguments("call_1", arguments={"query": "x"})
    emitter.message_finished(status="incomplete")
    with pytest.raises(RunEventProtocolError, match="active tool calls"):
        emitter.run_finished(status="failed", error_code="tool_incomplete")

    emitter.tool_finished("call_1", status="failed", error="unavailable")
    emitter.run_finished(status="failed", error_code="tool_failed")
    with pytest.raises(RunEventProtocolError, match="already finished"):
        emitter.text_delta("too late")


def test_openai_session_renders_from_events_without_exposing_status_as_answer() -> None:
    events: list[RunEvent] = []
    session = OpenAIStreamSession(
        virtual_model="audrey_deep",
        fingerprint_model="writer",
        completion_id="chatcmpl-test",
        created=123,
        run_id="run_test",
        conversation_id="con_test",
        assistant_message_id="msg_assistant",
        mode="deep",
        event_sink=events.append,
    )

    role_frame = session.role_frame()
    session.stage_started("planning", label="Planning")
    status_frame = session.status_frame("> _Planning_", stage="planning")
    session.stage_finished("planning")
    answer_frame = session.content_frame("answer")
    session.usage_reported(prompt_tokens=10, completion_tokens=2)
    session.terminal.finish(StreamOutcome.OK, finish_reason="stop")
    terminal_frame = session.terminal_frame()

    assert [event.type for event in events] == [
        "run.started",
        "message.started",
        "stage.started",
        "stage.progress",
        "stage.finished",
        "text.delta",
        "usage.reported",
        "message.finished",
        "run.finished",
    ]
    assert [
        event.delta
        for event in events
        if event.type == "text.delta"
    ] == ["answer"]
    assert json.loads(role_frame[6:])["choices"][0]["delta"] == {
        "role": "assistant"
    }
    assert json.loads(status_frame[6:])["choices"][0]["delta"] == {
        "content": "> _Planning_"
    }
    assert json.loads(answer_frame[6:])["choices"][0]["delta"] == {
        "content": "answer"
    }
    assert json.loads(terminal_frame[6:])["choices"][0]["finish_reason"] == "stop"
    assert session.done_frame() == "data: [DONE]\n\n"


def test_terminal_frame_closes_an_interrupted_stage_as_failed() -> None:
    events: list[RunEvent] = []
    session = OpenAIStreamSession(
        virtual_model="audrey_deep",
        fingerprint_model="writer",
        event_sink=events.append,
    )
    session.role_frame()
    session.stage_started("planning", label="Planning")
    session.terminal.finish(StreamOutcome.ERROR, finish_reason="stop")

    session.terminal_frame()

    terminal_types = [event.type for event in events[-3:]]
    assert terminal_types == ["stage.finished", "message.finished", "run.finished"]
    assert events[-3].status == "failed"
    assert events[-2].status == "incomplete"
    assert events[-1].status == "failed"
    assert events[-1].error_code == "pipeline_error"
