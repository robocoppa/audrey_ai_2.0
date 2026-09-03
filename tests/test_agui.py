"""Contracts for Audrey's isolated AG-UI protocol adapter."""

from __future__ import annotations

import json

import pytest

from audrey.pipeline.agui import (
    AGUI_EVENT_ADAPTER,
    AgUiCursor,
    AgUiCursorError,
    AgUiRunEventAdapter,
    dump_agui_event,
    format_agui_cursor,
    parse_agui_cursor,
)
from audrey.pipeline.run_events import RunEvent, RunEventEmitter


def _events(*, terminal: str = "succeeded") -> list[RunEvent]:
    events: list[RunEvent] = []
    emitter = RunEventEmitter(
        run_id="run_test",
        conversation_id="con_test",
        assistant_message_id="msg_assistant",
        mode="research",
        virtual_model="audrey_research",
        sink=events.append,
    )
    emitter.run_started()
    emitter.message_started()
    emitter.stage_started("researching", label="Researching")
    emitter.stage_progress("Searching", stage="researching")
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
    if terminal == "succeeded":
        emitter.run_finished(
            status="succeeded",
            finish_reason="stop",
            concrete_model="writer",
        )
    else:
        emitter.run_finished(status="failed", error_code="pipeline_error")
    return events


def test_adapter_maps_complete_vocabulary_to_camel_case_agui() -> None:
    adapter = AgUiRunEventAdapter(
        thread_id="con_test",
        run_id="run_test",
        assistant_message_id="msg_assistant",
    )
    wire = [
        dump_agui_event(adapted)
        for event in _events()
        for adapted in adapter.adapt(event)
    ]

    assert [item["type"] for item in wire] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "STEP_STARTED",
        "CUSTOM",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "CUSTOM",
        "STEP_FINISHED",
        "TEXT_MESSAGE_CONTENT",
        "CUSTOM",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    for payload in wire:
        AGUI_EVENT_ADAPTER.validate_python(payload)

    assert wire[0]["threadId"] == "con_test"
    assert wire[0]["runId"] == "run_test"
    assert wire[1] == {
        "type": "TEXT_MESSAGE_START",
        "timestamp": wire[1]["timestamp"],
        "messageId": "msg_assistant",
        "role": "assistant",
    }
    assert wire[4]["toolCallName"] == "web_search"
    assert wire[4]["parentMessageId"] == "msg_assistant"
    assert json.loads(wire[5]["delta"]) == {"query": "Euclid"}
    assert json.loads(wire[7]["content"]) == {"count": 2}
    assert wire[8]["name"] == "audrey.source.observed"
    assert wire[8]["value"]["sourceId"] == "source_1"
    assert wire[11]["name"] == "audrey.usage.reported"
    assert wire[-1]["outcome"] == {"type": "success"}
    assert wire[-1]["usage"] == [
        {
            "model": "writer",
            "inputTokens": 10,
            "outputTokens": 2,
            "totalTokens": 12,
        }
    ]
    assert "thread_id" not in json.dumps(wire)
    assert "user_id" not in json.dumps(wire)


def test_failed_tool_and_terminal_do_not_expose_private_error_text() -> None:
    events: list[RunEvent] = []
    emitter = RunEventEmitter(
        run_id="run_test",
        conversation_id="con_test",
        assistant_message_id="msg_assistant",
        mode="fast",
        virtual_model="audrey_fast",
        sink=events.append,
    )
    emitter.run_started()
    emitter.message_started()
    emitter.tool_started("call_1", name="web_fetch")
    emitter.tool_arguments("call_1", arguments={"url": "https://example.com"})
    emitter.tool_finished(
        "call_1",
        status="failed",
        error="private provider detail",
    )
    emitter.message_finished(status="incomplete")
    emitter.run_finished(
        status="failed",
        finish_reason="private provider detail",
        error_code="pipeline_error",
    )
    adapter = AgUiRunEventAdapter(
        thread_id="con_test",
        run_id="run_test",
        assistant_message_id="msg_assistant",
    )

    wire = [
        dump_agui_event(adapted)
        for event in events
        for adapted in adapter.adapt(event)
    ]

    result = next(item for item in wire if item["type"] == "TOOL_CALL_RESULT")
    terminal = wire[-1]
    assert json.loads(result["content"]) == {
        "error": "tool_failed",
        "status": "failed",
    }
    assert terminal == {
        "type": "RUN_ERROR",
        "timestamp": terminal["timestamp"],
        "message": "Run failed.",
        "code": "pipeline_error",
    }
    assert "private provider detail" not in json.dumps(wire)


def test_cancelled_run_uses_stable_run_error() -> None:
    events: list[RunEvent] = []
    emitter = RunEventEmitter(
        run_id="run_test",
        conversation_id="con_test",
        assistant_message_id="msg_assistant",
        mode="deep",
        virtual_model="audrey_deep",
        sink=events.append,
    )
    emitter.terminate_incomplete(
        status="cancelled",
        finish_reason="cancelled",
        error_code="cancelled_by_user",
    )
    adapter = AgUiRunEventAdapter(
        thread_id="con_test",
        run_id="run_test",
        assistant_message_id="msg_assistant",
    )

    terminal = dump_agui_event(adapter.adapt(events[-1])[0])

    assert terminal["type"] == "RUN_ERROR"
    assert terminal["message"] == "Run was cancelled."
    assert terminal["code"] == "cancelled_by_user"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", AgUiCursor()),
        ("12", AgUiCursor(source_sequence=12)),
        ("12.1", AgUiCursor(source_sequence=12, part=1)),
        ("12.2", AgUiCursor(source_sequence=12, part=2)),
    ],
)
def test_agui_cursor_round_trip(raw: str, expected: AgUiCursor) -> None:
    cursor = parse_agui_cursor(raw)

    assert cursor == expected
    assert cursor.native_after_sequence == (11 if raw.startswith("12.") else int(raw))


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "-1",
        "+1",
        "01",
        "1.",
        ".1",
        "1.0",
        "0.1",
        "1.2.3",
        "x",
        "1" * 65,
    ],
)
def test_agui_cursor_rejects_ambiguous_values(raw: str) -> None:
    with pytest.raises(AgUiCursorError, match="invalid"):
        parse_agui_cursor(raw)


def test_agui_cursor_tracks_fanout_parts() -> None:
    cursor = parse_agui_cursor("7.1")

    assert format_agui_cursor(source_sequence=7, part=2) == "7.2"
    assert cursor.consumed(source_sequence=7, part=1)
    assert not cursor.consumed(source_sequence=7, part=2)
    assert not cursor.consumed(source_sequence=8, part=1)
