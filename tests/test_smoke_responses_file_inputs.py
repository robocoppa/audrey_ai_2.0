"""Prove the live harness rejects false passes and cleans up on failures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SMOKE_DIR = Path(__file__).parent / "smoke"
sys.path.insert(0, str(SMOKE_DIR))
import smoke_responses_file_inputs as smoke  # noqa: E402


def _response(text="FILE_REF_TEST RED"):
    return {
        "id": "resp_test", "status": "completed", "output_text": text,
        "output": [{"content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 15, "output_tokens": 4, "total_tokens": 19},
    }


def _stream(text="FILE_REF_TEST RED", *, terminal="response.completed", delta=None):
    events = [
        {"type": "response.created", "sequence_number": 0},
        {"type": "response.output_text.delta", "sequence_number": 1, "delta": text if delta is None else delta},
        {"type": terminal, "sequence_number": 2, "response": _response(text)},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_stream_requires_matching_deltas_and_success_terminal():
    assert smoke._validate_stream(_stream(), "text/event-stream", "FILE_REF_TEST")["delta_count"] == 1
    with pytest.raises(smoke.SmokeError, match="match the completed"):
        smoke._validate_stream(_stream(delta="different"), "text/event-stream", "FILE_REF_TEST")
    with pytest.raises(smoke.SmokeError, match="complete normally"):
        smoke._validate_stream(_stream(terminal="response.failed"), "text/event-stream", "FILE_REF_TEST")


def test_color_check_requires_a_word_not_incidental_substring():
    with pytest.raises(smoke.SmokeError, match="both file inputs"):
        smoke._validate_response(_response("FILE_REF_TEST considered BLUE"), "FILE_REF_TEST")


@pytest.mark.parametrize("failure", [None, "generation", "second_upload", "cleanup"])
def test_smoke_always_attempts_cleanup_of_created_uploads(monkeypatch, capsys, failure):
    monkeypatch.setattr(smoke, "BASE_URL", "http://example.test")
    monkeypatch.setattr(smoke, "USER_TOKEN", "user-test")
    monkeypatch.setattr(smoke, "ADMIN_TOKEN", "admin-test")
    uploads = []
    deleted = []
    probes = []
    sentinel = ""

    def upload(filename, mime, content):
        nonlocal sentinel
        if failure == "second_upload" and uploads:
            raise smoke.SmokeError("upload interrupted")
        if mime == "text/plain":
            sentinel = content.decode().split(" is ")[1].split(".")[0]
        file_id = f"file_{len(uploads)}"
        uploads.append(file_id)
        return {"id": file_id}

    def request_json(path, *, token, method="GET", payload=None, expected=frozenset({200})):
        if path == "/api/me":
            return 200, {"id": token}
        if method == "DELETE":
            deleted.append(path.rsplit("/", 1)[1])
            if failure == "cleanup" and len(deleted) == 1:
                raise smoke.SmokeError("delete interrupted")
            return 200, {"deleted": True}
        if expected == frozenset({404}):
            probes.append(payload)
            return 404, {"detail": "File not found."}
        if expected == frozenset({422}):
            return 422, {"detail": "wrong kind"}
        if failure == "generation":
            raise smoke.SmokeError("model failed")
        return 200, _response(sentinel + " RED")

    monkeypatch.setattr(smoke, "_upload", upload)
    monkeypatch.setattr(smoke, "_wait_ready", lambda _ids: None)
    monkeypatch.setattr(smoke, "_json", request_json)
    monkeypatch.setattr(smoke, "_request", lambda *_args, **_kwargs: (
        200, _stream(sentinel + " RED"), "text/event-stream",
    ))
    exit_code = smoke.main()
    result = json.loads(capsys.readouterr().out)
    assert deleted == uploads
    assert exit_code == (0 if failure is None else 1)
    assert result["status"] == ("passed" if failure is None else "failed")
    if failure is None:
        assert result["cleanup"]["uploads_deleted"] == 2
        assert result["guards"]["deleted_http"] == 404
        assert len(probes) == 5
