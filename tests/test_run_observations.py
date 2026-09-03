"""Safe native-event projection for real tool dispatches and sources."""

from __future__ import annotations

import json

from audrey.pipeline.run_events import RunEvent, RunEventEmitter, dump_run_event
from audrey.pipeline.run_observations import RunEventToolObserver
from audrey.tools.dispatch import ToolResult


def _observer() -> tuple[RunEventToolObserver, list[RunEvent]]:
    events: list[RunEvent] = []
    emitter = RunEventEmitter(
        run_id="run-observation",
        conversation_id="conversation-observation",
        assistant_message_id="message-observation",
        mode="fast",
        virtual_model="audrey_fast",
        sink=events.append,
    )
    emitter.run_started()
    emitter.message_started()
    return RunEventToolObserver(emitter), events


def _result(
    name: str,
    content: str,
    *,
    elapsed_s: float = 0.0126,
    is_error: bool = False,
) -> ToolResult:
    return ToolResult(
        name=name,
        call_id="model-call-id",
        content=content,
        elapsed_s=elapsed_s,
        is_error=is_error,
    )


def test_tool_projection_redacts_private_arguments_and_raw_result_bodies():
    observer, events = _observer()
    event_call_id = observer.started({
        "id": "model-call-id",
        "function": {
            "name": "memory_store",
            "arguments": {
                "key": "preferred-theme",
                "value": "private-memory-value",
                "tags": "user:alice@example.com secret-tag",
                "user": "forged@example.com",
                "future_secret": "new-field-secret",
            },
        },
    })
    observer.finished(
        event_call_id,
        _result(
            "memory_store",
            '{"value":"private-result-body","token":"sk-secret"}',
        ),
    )

    started, arguments, finished = events[-3:]
    assert started.type == "tool.started"
    assert started.tool_call_id.startswith("tool_")
    assert started.tool_call_id != "model-call-id"
    assert arguments.type == "tool.arguments"
    assert arguments.arguments == {
        "key": "preferred-theme",
        "value": "[redacted]",
        "tags": "[redacted]",
        "user": "[redacted]",
        "future_secret": "[redacted]",
    }
    assert finished.type == "tool.finished"
    assert finished.result == {
        "status": "succeeded",
        "elapsedMs": 13,
        "contentBytes": 51,
        "sourceCount": 0,
    }
    wire = json.dumps([dump_run_event(event) for event in events])
    assert "private-memory-value" not in wire
    assert "private-result-body" not in wire
    assert "sk-secret" not in wire
    assert "forged@example.com" not in wire


def test_url_arguments_drop_credentials_query_tokens_and_fragments():
    observer, events = _observer()
    observer.started({
        "function": {
            "name": "web_fetch",
            "arguments": {
                "url": "https://user:pass@example.com/a/page?token=secret#private",
                "max_chars": 1200,
            },
        },
    })

    arguments = events[-1]
    assert arguments.type == "tool.arguments"
    assert arguments.arguments == {
        "url": "https://example.com/a/page",
        "max_chars": 1200,
    }


def test_error_projection_exposes_only_a_stable_code():
    observer, events = _observer()
    call_id = observer.started({
        "function": {"name": "web_search", "arguments": {"query": "coffee"}},
    })
    observer.finished(
        call_id,
        _result(
            "web_search",
            json.dumps({
                "error": "provider_timeout",
                "detail": "private provider response and bearer token",
            }),
            is_error=True,
        ),
    )

    finished = events[-1]
    assert finished.type == "tool.finished"
    assert finished.status == "failed"
    assert finished.error == "provider_timeout"
    assert "private provider response" not in json.dumps(dump_run_event(finished))


def test_untrusted_error_text_collapses_to_generic_code():
    observer, events = _observer()
    call_id = observer.started({
        "function": {"name": "web_search", "arguments": "not-json"},
    })
    observer.finished(
        call_id,
        _result(
            "web_search",
            '{"error":"Authorization failed for secret@example.com"}',
            is_error=True,
        ),
    )

    arguments, finished = events[-2:]
    assert arguments.type == "tool.arguments"
    assert arguments.arguments == {"_status": "arguments_not_json"}
    assert finished.type == "tool.finished"
    assert finished.error == "tool_failed"


def test_public_sources_are_deduplicated_across_tool_calls():
    observer, events = _observer()
    source = {
        "title": "Coffee history",
        "url": "https://user:pass@example.com/coffee?token=secret#private",
        "tool": "web_search",
    }
    first = observer.started({
        "function": {"name": "web_search", "arguments": {"query": "coffee"}},
    })
    observer.finished(
        first,
        _result("web_search", '{"results":[]}'),
        sources=[source],
    )
    second = observer.started({
        "function": {"name": "web_search", "arguments": {"query": "coffee history"}},
    })
    observer.finished(
        second,
        _result("web_search", '{"results":[]}'),
        sources=[source],
    )

    sources = [event for event in events if event.type == "source.observed"]
    assert len(sources) == 1
    assert sources[0].source_id.startswith("src_")
    assert sources[0].title == "Coffee history"
    assert sources[0].url == "https://example.com/coffee"
    assert sources[0].source_type == "web_search"
