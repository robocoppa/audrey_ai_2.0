"""The storage smoke proves exact retrieval, private restart state, and cleanup."""
from __future__ import annotations

import copy
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "smoke"))
import smoke_responses_storage as smoke

ROOT_ID = "resp_" + "a" * 32
CHILD_ID = "resp_" + "b" * 32
UNSTORED_ID = "resp_" + "c" * 32
MARKER = "STORAGE_TEST"
USER_ASSERTION = "test-user-assertion"
OTHER_ASSERTION = "test-other-assertion"


def _response(response_id=ROOT_ID, answer=smoke.ROOT_ANSWER, *, store=True, parent_id=None):
    return {
        "id": response_id, "object": "response", "status": "completed", "model": smoke.MODEL,
        "store": store, "previous_response_id": parent_id, "output_text": answer,
        "output": [{"type": "message", "id": "msg_" + response_id[5:], "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": answer, "annotations": [], "logprobs": []}]}],
        "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}, "incomplete_details": None,
    }


def _events(response):
    item = response["output"][0]
    text = response["output_text"]
    return [
        {"type": "response.created", "response": {"id": response["id"], "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "status": "in_progress", "content": []}},
        {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text[:8]},
        {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text[8:]},
        {"type": "response.output_text.done", "item_id": item["id"], "output_index": 0, "content_index": 0, "text": text},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]


def _sse(events):
    for index, event in enumerate(events):
        event.setdefault("sequence_number", index)
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.parametrize("failure", [None, "store", "parent", "status", "wrong_answer", "usage", "output_text", "progress"])
def test_response_validation_accepts_only_correct_stored_answer(failure):
    response = _response(CHILD_ID, MARKER, parent_id=ROOT_ID)
    if failure == "store":
        response["store"] = 1
    elif failure == "parent":
        response["previous_response_id"] = "resp_other"
    elif failure == "status":
        response["status"] = "incomplete"
    elif failure == "wrong_answer":
        response = _response(CHILD_ID, smoke.ROOT_ANSWER, parent_id=ROOT_ID)
    elif failure == "usage":
        response["usage"]["total_tokens"] = 41
    elif failure == "output_text":
        response["output"][0]["content"][0]["text"] = "different"
    elif failure == "progress":
        response = _response(CHILD_ID, "> _Thinking_ model", parent_id=ROOT_ID)
    if failure:
        with pytest.raises(smoke.http.SmokeError):
            smoke._validate_response(response, MARKER, store=True, parent_id=ROOT_ID)
    else:
        assert smoke._validate_response(response, MARKER, store=True, parent_id=ROOT_ID)["sentinel"] is True


@pytest.mark.parametrize("failure", [None, "content_type", "sequence", "terminal", "native_event", "response_id", "item_id", "done_item", "delta", "text_done", "added_missing", "text_done_missing", "function_event", "added_type"])
def test_stored_stream_requires_stable_identity_matching_deltas_and_completion(failure):
    events = _events(_response(CHILD_ID, MARKER, parent_id=ROOT_ID))
    content_type = "text/event-stream"
    if failure == "content_type":
        content_type = "application/json"
    elif failure == "sequence":
        events[2]["sequence_number"] = 99
    elif failure == "terminal":
        events[-1]["type"] = "response.incomplete"
    elif failure == "native_event":
        events.insert(2, {"type": "TOOL_CALL_START"})
    elif failure == "response_id":
        events[0]["response"]["id"] = ROOT_ID
    elif failure == "item_id":
        events[2]["item_id"] = "msg_wrong"
    elif failure == "done_item":
        events[-2]["item"] = {**events[-2]["item"], "id": "msg_wrong"}
    elif failure == "delta":
        events[2]["delta"] = "wrong"
    elif failure == "text_done":
        events[4]["text"] = "wrong"
    elif failure == "added_missing":
        del events[1]
    elif failure == "text_done_missing":
        del events[4]
    elif failure == "function_event":
        events.insert(2, {"type": "response.function_call_arguments.delta"})
    elif failure == "added_type":
        events[1]["item"]["type"] = "function_call"
    if failure:
        with pytest.raises(smoke.http.SmokeError):
            smoke._validate_stream(_sse(events), content_type, MARKER, ROOT_ID)
    else:
        response, result = smoke._validate_stream(_sse(events), content_type, MARKER, ROOT_ID)
        assert response["id"] == CHILD_ID
        assert result["terminal"] == "response.completed"
        assert result["delta_count"] == 2


@pytest.mark.parametrize("bad", ["wrong_error", "sse", "invalid_json"])
def test_reference_guard_never_accepts_sse_or_a_different_not_found(monkeypatch, bad):
    data = {"detail": "not found"} if bad == "wrong_error" else smoke._NOT_FOUND
    body = b"not-json" if bad == "invalid_json" else json.dumps(data).encode()
    content_type = "text/event-stream" if bad == "sse" else "application/json"
    monkeypatch.setattr(smoke.http, "_request", lambda *_a, **_k: (404, body, content_type))
    with pytest.raises((smoke.http.SmokeError, ValueError)):
        smoke._not_found("/v1/responses/resp_missing", token=USER_ASSERTION)


class FakeAPI:
    def __init__(self, monkeypatch, *, failure=None):
        monkeypatch.setattr(smoke.http, "BASE_URL", "http://example.test")
        monkeypatch.setattr(smoke.http, "USER_TOKEN", USER_ASSERTION)
        monkeypatch.setattr(smoke.http, "ADMIN_TOKEN", OTHER_ASSERTION)
        monkeypatch.setattr(smoke.http, "_json", self.json)
        monkeypatch.setattr(smoke.http, "_request", self.request)
        self.saved = {}
        self.marker = ""
        self.payloads = []
        self.denials = []
        self.deletions = []
        self.failure = failure

    def json(self, path, *, token, method="GET", payload=None, expected=frozenset({200})):
        if path == "/api/me":
            return 200, {"id": "owner" if token == USER_ASSERTION else ("owner" if self.failure == "same_owner" else "other")}
        assert token == USER_ASSERTION
        if path == "/v1/responses":
            assert method == "POST"
            self.payloads.append(copy.deepcopy(payload))
            if len(self.payloads) == 1:
                assert payload["store"] is True
                assert "previous_response_id" not in payload
                self.marker = payload["input"].split("marker: ")[1].split(".")[0]
                response = _response()
                self.saved[ROOT_ID] = copy.deepcopy(response)
                return 200, response
            assert payload["store"] is False
            assert payload["previous_response_id"] == CHILD_ID
            assert self.marker not in payload["input"]
            assert all(key not in payload for key in ("instructions", "tools", "tool_choice"))
            return 200, _response(UNSTORED_ID, self.marker, store=False, parent_id=CHILD_ID)
        response_id = path.rsplit("/", 1)[1]
        if method == "DELETE":
            self.deletions.append(response_id)
            if response_id not in self.saved:
                assert 404 in expected
                return 404, copy.deepcopy(smoke._NOT_FOUND)
            del self.saved[response_id]
            self.saved.pop(CHILD_ID, None)
            return 200, {"id": response_id, "object": "response.deleted", "deleted": True}
        assert response_id in self.saved
        result = copy.deepcopy(self.saved[response_id])
        if self.failure == "retrieval_changed":
            result["usage"]["input_tokens"] += 1
        return 200, result

    def request(self, path, *, token, method="GET", body=None, content_type="", expected=frozenset({200})):
        payload = json.loads(body) if body is not None else None
        if expected == frozenset({404}):
            self.denials.append((path, token, method, payload))
            if path != "/v1/responses":
                response_id = path.rsplit("/", 1)[1]
                assert token == OTHER_ASSERTION or response_id not in self.saved
            else:
                assert payload["stream"] is True
                assert self.marker not in payload["input"]
                assert token == OTHER_ASSERTION or payload["previous_response_id"] not in self.saved
            return 404, json.dumps(smoke._NOT_FOUND).encode(), "application/json"
        assert path == "/v1/responses" and method == "POST" and token == USER_ASSERTION
        self.payloads.append(copy.deepcopy(payload))
        assert payload["stream"] is True and payload["store"] is True
        assert payload["previous_response_id"] == ROOT_ID
        assert self.marker not in payload["input"]
        assert all(key not in payload for key in ("instructions", "tools", "tool_choice"))
        response = _response(CHILD_ID, self.marker, parent_id=ROOT_ID)
        self.saved[CHILD_ID] = copy.deepcopy(response)
        events = _events(response)
        if self.failure == "stream_incomplete":
            events[-1]["type"] = "response.incomplete"
        return 200, _sse(events), "text/event-stream"


def test_all_runs_exactly_three_generations_and_cleans_parent_and_child(monkeypatch, capsys):
    api = FakeAPI(monkeypatch)
    assert smoke.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["generation_calls"] == 3 == len(api.payloads)
    assert report["uploads_created"] == 0
    assert not api.saved
    assert api.deletions == [ROOT_ID]
    assert len(api.denials) == 9
    assert report["completed"]["retrieval"]["exact_terminal_object"] is True
    assert report["streamed"]["previous_instructions_not_inherited"] is True
    assert report["not_stored"]["get_http"] == 404
    assert report["cleanup"]["descendant_cascade"] is True
    assert all(payload["max_output_tokens"] == 8192 for payload in api.payloads)


def test_capture_then_restart_verify_uses_private_snapshot_and_zero_new_generations(monkeypatch, tmp_path, capsys):
    api = FakeAPI(monkeypatch)
    snapshot = tmp_path / "state" / "snapshot.json"
    assert smoke.main(["capture", "--snapshot", str(snapshot)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "captured"
    assert set(api.saved) == {ROOT_ID, CHILD_ID}
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    saved = json.loads(snapshot.read_text())
    assert saved["root"] == api.saved[ROOT_ID]
    assert saved["child"] == api.saved[CHILD_ID]
    assert USER_ASSERTION not in snapshot.read_text()
    assert OTHER_ASSERTION not in snapshot.read_text()
    assert smoke.main(["verify", "--snapshot", str(snapshot)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["generation_calls"] == 0
    assert len(api.payloads) == 3
    assert report["restart"]["root"]["exact_terminal_object"] is True
    assert report["restart"]["child"]["exact_terminal_object"] is True
    assert not snapshot.exists() and not api.saved


@pytest.mark.parametrize("failure", ["same_owner", "retrieval_changed", "stream_incomplete"])
def test_failed_generation_or_retrieval_cleans_created_records_without_snapshot(monkeypatch, tmp_path, capsys, failure):
    api = FakeAPI(monkeypatch, failure=failure)
    snapshot = tmp_path / "snapshot.json"
    assert smoke.main(["capture", "--snapshot", str(snapshot)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed" and "error" in report
    assert not api.saved and not snapshot.exists()
    assert len(api.payloads) == {"same_owner": 0, "retrieval_changed": 1, "stream_incomplete": 2}[failure]


def test_verify_mismatch_preserves_snapshot_and_records_for_targeted_retry(monkeypatch, tmp_path, capsys):
    api = FakeAPI(monkeypatch)
    snapshot = tmp_path / "snapshot.json"
    assert smoke.main(["capture", "--snapshot", str(snapshot)]) == 0
    capsys.readouterr()
    api.failure = "retrieval_changed"
    assert smoke.main(["verify", "--snapshot", str(snapshot)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "differs from the exact terminal object" in report["error"]
    assert snapshot.exists() and len(api.saved) == 2
    assert len(api.payloads) == 3
    assert smoke.main(["cleanup", "--snapshot", str(snapshot)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "cleaned" and not snapshot.exists() and not api.saved


def test_existing_snapshot_cannot_be_overwritten_or_start_generation(monkeypatch, tmp_path, capsys):
    api = FakeAPI(monkeypatch)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("already exists")
    assert smoke.main(["capture", "--snapshot", str(snapshot)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "snapshot already exists" in report["error"]
    assert not api.payloads and snapshot.read_text() == "already exists"


@pytest.mark.parametrize("failure", ["missing", "invalid_json", "owner", "base_url", "schema", "root_id", "marker", "child_parent", "path_id"])
def test_invalid_snapshot_never_deletes_or_generates(monkeypatch, tmp_path, capsys, failure):
    api = FakeAPI(monkeypatch)
    snapshot = tmp_path / "snapshot.json"
    state = {"schema": 1, "base_url": "http://example.test", "identity": {"owner_id": "owner", "other_id": "other"},
             "root": _response(), "child": _response(CHILD_ID, MARKER, parent_id=ROOT_ID), "marker": MARKER}
    if failure == "owner":
        state["identity"]["owner_id"] = "foreign"
    elif failure == "base_url":
        state["base_url"] = "http://another.test"
    elif failure == "schema":
        state["schema"] = 2
    elif failure == "root_id":
        state["root"]["id"] = "con_wrong"
    elif failure == "marker":
        del state["marker"]
    elif failure == "child_parent":
        state["child"]["previous_response_id"] = "resp_other"
    elif failure == "path_id":
        state["root"]["id"] = "resp_/../../api/another"
    if failure != "missing":
        snapshot.write_text("broken" if failure == "invalid_json" else json.dumps(state))
    assert smoke.main(["cleanup", "--snapshot", str(snapshot)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert not api.payloads and not api.deletions


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_invalid_budget_stops_before_network(monkeypatch, budget):
    monkeypatch.setattr(smoke.http, "_json", lambda *_a, **_k: pytest.fail("network called"))
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--max-output-tokens", budget])
    assert exc.value.code == 2


def test_snapshot_write_failure_cleans_only_new_records(monkeypatch, tmp_path, capsys):
    api = FakeAPI(monkeypatch)
    snapshot = tmp_path / "snapshot.json"
    def failed_write(*_args):
        raise OSError("disk full")
    monkeypatch.setattr(smoke, "_write_snapshot", failed_write)
    assert smoke.main(["capture", "--snapshot", str(snapshot)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "disk full" in report["error"]
    assert not api.saved and not snapshot.exists()
    assert report["cleanup"]["descendant_cascade"] is True


def test_missing_assertion_stops_before_network(monkeypatch, capsys):
    monkeypatch.setattr(smoke.http, "BASE_URL", "http://example.test")
    monkeypatch.setattr(smoke.http, "USER_TOKEN", USER_ASSERTION)
    monkeypatch.setattr(smoke.http, "ADMIN_TOKEN", "")
    monkeypatch.setattr(smoke.http, "_json", lambda *_a, **_k: pytest.fail("network called"))
    assert smoke.main([]) == 2
    assert "AUDREY_ADMIN_JWT" in capsys.readouterr().err
