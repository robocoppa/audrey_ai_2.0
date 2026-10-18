"""The offline extraction boundary exposes reasons without parser output."""

from __future__ import annotations

import json
import signal
import subprocess

import pytest

from audrey.references import indexing as module
from audrey.references.extract import ERROR_MESSAGES, ERROR_PREFIX
from audrey.references.library import ReferenceReadError


@pytest.mark.parametrize("code", ["memory_limit", "password_required", "text_decode_error", "checksum_mismatch"])
def test_parent_exposes_only_allowlisted_child_reason(monkeypatch, code):
    def child(*args, **kwargs):
        kwargs["stderr"].write(b"private-source-warning https://user:secret@example.test\n")
        kwargs["stderr"].write((ERROR_PREFIX + json.dumps({"code": code}) + "\n").encode())
        return subprocess.CompletedProcess(args[0], 2)

    monkeypatch.setattr(module.subprocess, "run", child)
    with pytest.raises(ReferenceReadError) as caught:
        module._extract({})
    assert str(caught.value) == ERROR_MESSAGES[code]
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("trailer", ["not-json", json.dumps({"code": "secret-detail"}), "[]"])
def test_malformed_or_unknown_diagnostics_cannot_escape(monkeypatch, trailer):
    def child(*args, **kwargs):
        kwargs["stderr"].write(("secret-warning\n" + ERROR_PREFIX + trailer + "\n").encode())
        return subprocess.CompletedProcess(args[0], 2)

    monkeypatch.setattr(module.subprocess, "run", child)
    with pytest.raises(ReferenceReadError, match=r"worker failed \(exit 2\)") as caught:
        module._extract({})
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(("exit_code", "message"), [
    (-signal.SIGXCPU, "CPU time limit"),
    (-signal.SIGXFSZ, "output limit"),
    (-signal.SIGKILL, "worker stopped \\(signal 9\\)"),
])
def test_terminated_worker_has_an_honest_reason(monkeypatch, exit_code, message):
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **_: subprocess.CompletedProcess(args[0], exit_code))
    with pytest.raises(ReferenceReadError, match=message):
        module._extract({})


def test_timeout_remains_distinct_from_parser_failure(monkeypatch):
    def child(*args, **kwargs):
        assert kwargs["timeout"] == module.MAX_WALL_SECONDS
        raise subprocess.TimeoutExpired("must not expose this secret", module.MAX_WALL_SECONDS)

    monkeypatch.setattr(module.subprocess, "run", child)
    with pytest.raises(ReferenceReadError, match=f"{module.MAX_WALL_SECONDS}-second time limit"):
        module._extract({})


@pytest.mark.parametrize("body", [b"", b"not-json", b"[]", b'{"page":2}'])
def test_success_exit_cannot_hide_invalid_page_output(monkeypatch, body):
    def child(*args, **kwargs):
        kwargs["stdout"].write(body)
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(module.subprocess, "run", child)
    with pytest.raises(ReferenceReadError, match="invalid page data"):
        module._extract({})


def test_parent_still_rejects_oversized_output(monkeypatch):
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 32)

    def child(*args, **kwargs):
        kwargs["stdout"].write(b"x" * 33)
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(module.subprocess, "run", child)
    with pytest.raises(ReferenceReadError, match="output limit"):
        module._extract({})
