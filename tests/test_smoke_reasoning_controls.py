"""The smoke must not mistake unsupported effort or broken SSE for success."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "smoke"))
import smoke_reasoning_controls as smoke


def sender(*, broken=None):
    sent = []

    def send(path, payload):
        sent.append((path, payload))
        index = len(sent)
        if index <= 3:
            code = "reasoning_controls_conflict" if index == 1 else "reasoning_unsupported"
            return (200 if broken == "guard" else 400), "application/json", json.dumps({"detail": {"error": code}})
        marker = (payload.get("input") or payload["messages"][0]["content"]).split("only ")[1].rstrip(".")
        if index == 4:
            if broken == "unsupported":
                return 400, "application/json", json.dumps({"detail": {"error": "reasoning_unsupported"}})
            return 200, "application/json", json.dumps({"id": "chatcmpl-test", "choices": [{
                "finish_reason": "length" if broken == "truncated_chat" else "stop", "message": {"content": marker},
            }]})
        response = {"id": "resp_test", "status": "completed", "reasoning": {"effort": "low"}, "output_text": marker}
        if broken == "echo":
            response.pop("reasoning")
        events = [
            {"type": "response.created", "response": {"id": "resp_test"}},
            {"type": "response.output_text.delta", "delta": marker},
            {"type": "response.failed" if broken == "terminal" else "response.completed", "response": response},
        ]
        if broken == "identity":
            response["id"] = "resp_other"
        raw = "".join(f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n"
                      for index, event in enumerate(events))
        return 200, "text/event-stream", raw
    return send, sent


def test_smoke_checks_admission_and_only_two_generation_requests():
    send, sent = sender()
    result = smoke.run(send)
    assert result["status"] == "passed"
    assert result["generation_requests"] == 2
    assert len(sent) == 5
    assert sent[3][1]["reasoning_effort"] == "none"
    assert sent[4][1]["reasoning"] == {"effort": "low"}
    assert sent[3][1]["max_tokens"] == sent[4][1]["max_output_tokens"] == 4096
    assert not any("store" in payload or "tools" in payload for _, payload in sent)


@pytest.mark.parametrize("failure,count", [("guard", 1), ("unsupported", 4), ("truncated_chat", 4),
                                           ("echo", 5), ("terminal", 5), ("identity", 5)])
def test_smoke_stops_at_failure_without_retrying_or_downgrading(failure, count):
    send, sent = sender(broken=failure)
    result = smoke.run(send)
    assert result["status"] == "failed"
    assert result["error"]
    assert len(sent) == count
