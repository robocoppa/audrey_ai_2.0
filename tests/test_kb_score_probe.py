"""The KB score probe authenticates and fails without claiming a measurement."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location(
    "kb_score_probe", Path(__file__).resolve().parents[1] / "scripts/probes/kb_score_probe.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
TOKEN = "probe-service-secret"  # noqa: S105 - synthetic service credential


@pytest.fixture
def query_file(tmp_path, monkeypatch):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"on_domain": ["What is basalt?"], "off_domain": ["Unrelated"]}))
    monkeypatch.setattr(sys, "argv", ["kb_score_probe.py", "--queries", str(path)])
    return path


def test_missing_credential_fails_before_reading_queries_or_sending_requests(monkeypatch, capsys):
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["kb_score_probe.py", "--queries", "/missing/queries.json"])

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("a probe without credentials must not send a request")

    monkeypatch.setattr(probe.httpx, "post", unexpected_request)
    assert probe.main() == 2
    assert "Set KB_SERVICE_TOKEN" in capsys.readouterr().err


def test_authenticated_queries_keep_empty_results_distinct_from_http_failure(
    query_file, monkeypatch, capsys,
):
    monkeypatch.setenv("KB_SERVICE_TOKEN", TOKEN)
    seen = []

    def respond(request):
        seen.append(request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"score": .85, "source": "public/basalt.txt"},
        ] if body["query"] == "What is basalt?" else []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(probe.httpx, "post", client.post)
        assert probe.main() == 0
    assert len(seen) == 2
    assert all(request.headers["X-Audrey-Service-Token"] == TOKEN for request in seen)
    assert all(request.url.path == "/v1/kb/query" for request in seen)
    output = capsys.readouterr()
    assert "0.8500" in output.out and "no hits" in output.out
    assert TOKEN not in output.out + output.err


@pytest.mark.parametrize("failure", ["unauthorized", "unavailable", "transport", "invalid_json"])
def test_failed_query_exits_nonzero_without_a_score_report_or_leaking_credentials(
    query_file, monkeypatch, capsys, failure,
):
    monkeypatch.setenv("KB_SERVICE_TOKEN", TOKEN)

    def respond(request):
        if failure == "transport":
            raise httpx.ConnectError(f"provider error containing {TOKEN}", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, text=TOKEN)
        return httpx.Response(401 if failure == "unauthorized" else 503, text=TOKEN)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(probe.httpx, "post", client.post)
        assert probe.main() == 1
    output = capsys.readouterr()
    assert "KB score probe failed" in output.err
    assert "floor analysis" not in output.out
    assert TOKEN not in output.out + output.err
