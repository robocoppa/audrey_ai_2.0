"""The remote-input live harness must reject missing evidence and failed streams."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "smoke"))
import smoke_responses_remote_inputs as smoke


def reply(text="Dummy PDF file | PYTHON"):
    return {"id": "resp_test", "status": "completed", "output_text": text,
            "output": [{"content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}}


@pytest.mark.parametrize("text", ["", "PYTHON", "Dummy PDF file", "Dummy PDF file pythonic"])
def test_harness_requires_both_document_text_and_image_label(text):
    with pytest.raises(smoke.http.SmokeError, match="remote PDF and image"):
        smoke._validate(reply(text))


@pytest.mark.parametrize("failure", [None, "terminal", "deltas", "sequence"])
def test_stream_requires_normal_terminal_sequence_and_matching_deltas(failure):
    answer = reply()
    events = [
        {"type": "response.created", "sequence_number": 0},
        {"type": "response.output_text.delta", "sequence_number": 1, "delta": answer["output_text"] if failure != "deltas" else "wrong"},
        {"type": "response.completed" if failure != "terminal" else "response.incomplete", "sequence_number": 2 if failure != "sequence" else 3, "response": answer},
    ]
    data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    if failure:
        with pytest.raises(smoke.http.SmokeError):
            smoke._validate_stream(data, "text/event-stream")
    else:
        assert smoke._validate_stream(data, "text/event-stream")["pdf_text"]


@pytest.mark.parametrize("bad_guard", [False, True])
def test_private_url_guards_run_before_any_generation(monkeypatch, capsys, bad_guard):
    monkeypatch.setattr(smoke.http, "BASE_URL", "http://example.test")
    monkeypatch.setattr(smoke.http, "USER_TOKEN", "user-test")
    calls = []

    def request(path, **kw):
        calls.append(kw)
        return 422, json.dumps({"detail": {"error": "wrong" if bad_guard else "responses_remote_input_blocked"}}).encode(), "application/json"

    monkeypatch.setattr(smoke.http, "_request", request)
    monkeypatch.setattr(smoke.http, "_json", lambda *a, **kw: (_ for _ in ()).throw(smoke.http.SmokeError("stop before generation")))
    assert smoke.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["uploads_created"] == 0
    assert len(calls) == (1 if bad_guard else 2)
    assert "completed" not in result


def _stream_body(answer):
    events = [
        {"type": "response.created", "sequence_number": 0},
        {"type": "response.output_text.delta", "sequence_number": 1, "delta": answer["output_text"]},
        {"type": "response.completed", "sequence_number": 2, "response": answer},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_streamed_only_runs_one_model_call_with_explicit_budget(monkeypatch, capsys):
    monkeypatch.setattr(smoke.http, "BASE_URL", "http://example.test")
    monkeypatch.setattr(smoke.http, "USER_TOKEN", "user-test")
    payloads = []

    def request(path, **kwargs):
        assert path == "/v1/responses"
        payloads.append(json.loads(kwargs["body"]))
        return 200, _stream_body(reply()), "text/event-stream"

    monkeypatch.setattr(smoke.http, "_request", request)
    monkeypatch.setattr(smoke.http, "_json", lambda *a, **kw: pytest.fail("completed call must not repeat"))

    assert smoke.main(["--case", "streamed", "--max-output-tokens", "8192"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["case"] == "streamed"
    assert result["status"] == "passed"
    assert result["streamed"]["progress_hidden"] is True
    assert result["uploads_created"] == 0
    assert "completed" not in result
    assert "guards" not in result
    assert len(payloads) == 1
    assert payloads[0]["stream"] is True
    assert payloads[0]["max_output_tokens"] == 8192


def test_stream_smoke_rejects_progress_as_answer_text():
    answer = reply("> _Thinking_ ✅ model\n\n---\nDummy PDF file | PYTHON")
    with pytest.raises(smoke.http.SmokeError, match="progress banner"):
        smoke._validate_stream(_stream_body(answer), "text/event-stream")


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_invalid_smoke_budget_is_rejected_before_network(monkeypatch, budget):
    monkeypatch.setattr(smoke.http, "_request", lambda *a, **kw: pytest.fail("network called"))
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--max-output-tokens", budget])
    assert exc.value.code == 2
