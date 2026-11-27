"""Transport and scoring coverage for the six-model cloud bake-off preset."""
import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def bakeoff(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "evals"))
    return importlib.import_module("eval_cloud_models")


def _client(monkeypatch, bakeoff, response):
    client = httpx.Client
    transport = httpx.MockTransport(response)
    monkeypatch.setattr(
        bakeoff.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs)
    )


def _run(monkeypatch, bakeoff, *args):
    monkeypatch.setattr(sys, "argv", ["eval_cloud_models.py", *map(str, args)])
    return bakeoff.main()


def _tags(models):
    return httpx.Response(
        200, json={"models": [{"name": tag, "digest": f"digest-{i}"} for i, tag in enumerate(models)]}
    )


def test_inventory_check_never_generates(monkeypatch, bakeoff, tmp_path, capsys):
    def response(request):
        assert request.method == "GET"
        assert request.url.path == "/api/tags"
        assert "authorization" not in request.headers
        return _tags(bakeoff.MODELS)

    _client(monkeypatch, bakeoff, response)
    capture = tmp_path / "capture.jsonl"
    assert _run(monkeypatch, bakeoff, "collect", "--check", "--capture", capture) == 0
    assert "156 planned requests" in capsys.readouterr().out
    assert not capture.exists()


def test_missing_tag_stops_before_inference(monkeypatch, bakeoff, tmp_path, capsys):
    def response(request):
        assert request.method == "GET"
        return _tags(tag for tag in bakeoff.MODELS if tag != "hy4:cloud")

    _client(monkeypatch, bakeoff, response)
    capture = tmp_path / "capture.jsonl"
    assert _run(monkeypatch, bakeoff, "collect", "--capture", capture) == 2
    assert "missing Ollama tags: hy4:cloud" in capsys.readouterr().err
    assert not capture.exists()


def test_paired_run_scores_code_truncation_and_service_errors(monkeypatch, bakeoff, tmp_path):
    requests = []
    code = r"""import re
def parse_duration(s):
    if not re.fullmatch(r"(\d+[hms])+", s):
        raise ValueError(s)
    pairs = re.findall(r"(\d+)([hms])", s)
    if len({unit for _, unit in pairs}) != len(pairs):
        raise ValueError(s)
    return sum(int(value) * {"h": 3600, "m": 60, "s": 1}[unit] for value, unit in pairs)
"""

    def response(request):
        assert "authorization" not in request.headers
        if request.url.path == "/api/tags":
            return _tags(bakeoff.MODELS)
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert set(body) == {"model", "messages", "stream"}
        assert body["stream"]
        requests.append(body)
        if body["model"] == "deepseek-v4-pro:cloud":
            return httpx.Response(429)
        answer = code.replace("3600", "360") if body["model"] == "glm-5.3:cloud" else code
        frames = [
            {"message": {"thinking": "private thinking"}, "done": False},
            {"message": {"content": f"```python\n{answer}```"}, "done": False},
            {
                "done": True,
                "done_reason": "length" if body["model"] == "hy4:cloud" else "stop",
                "prompt_eval_count": 120,
                "eval_count": 90,
            },
        ]
        if body["model"] == "minimax-m3.1-flash:cloud":
            frames.pop()
        return httpx.Response(200, text="\n".join(map(json.dumps, frames)))

    _client(monkeypatch, bakeoff, response)
    capture = tmp_path / "capture.jsonl"
    assert _run(
        monkeypatch, bakeoff, "collect", "--capture", capture, "--only", "parse-duration",
        "--repeat", 2,
    ) == 0
    assert [body["model"] for body in requests] == [tag for tag in bakeoff.MODELS for _ in range(2)]
    assert all(body["messages"] == requests[0]["messages"] for body in requests)
    assert "private thinking" not in capture.read_text()
    requests.clear()

    answers = tmp_path / "answers.md"
    results = tmp_path / "results.json"
    comparison = tmp_path / "compare.md"
    score_args = (
        "score", "--capture", capture, "--answers", answers,
        "--results", results, "--compare", comparison,
    )
    assert _run(monkeypatch, bakeoff, *score_args) == 1
    assert not requests
    records = json.loads(results.read_text())
    assert len(records) == 12
    assert "| `kimi-k3.1-pro:cloud` | Candidate | 2/2 | 2/2 |" in comparison.read_text()
    assert "ValueError" in answers.read_text()
    assert records[0]["checks"]["code_runs"] is True
    glm = next(record for record in records if record["model"] == "glm-5.3:cloud")
    assert glm["checks"]["code_runs"] is False
    hy4 = next(record for record in records if record["model"] == "hy4:cloud")
    assert hy4["checks"]["not_truncated"] is False
    assert hy4["ollama"]["output_tokens"] == 90
    deepseek = next(record for record in records if record["model"] == "deepseek-v4-pro:cloud")
    assert deepseek["error"] == "HTTP 429 from Ollama"
    minimax = next(record for record in records if record["model"] == "minimax-m3.1-flash:cloud")
    assert minimax["error"] == "Ollama stream ended before its done frame"
    assert "| `deepseek-v4-pro:cloud` | Baseline | 0/2 |" in comparison.read_text()

    capture.write_text("\n".join(capture.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="capture is incomplete"):
        _run(monkeypatch, bakeoff, *score_args)
