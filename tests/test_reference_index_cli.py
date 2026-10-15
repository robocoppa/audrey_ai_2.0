"""Operator preflight, safe reports and public-only clients without live calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from audrey.references import index_cli as module
from audrey.references.library import ReferenceUnavailable


@pytest.fixture
def cli(tmp_path, monkeypatch):
    root = tmp_path / "public-index"
    root.mkdir()
    cfg = SimpleNamespace(
        raw={"reference_libraries": {"index_root": str(root)}},
        env=SimpleNamespace(
            ollama_host="http://local-ollama", qdrant_host="local-qdrant", qdrant_port=6333
        ),
    )
    monkeypatch.setattr(module, "get_config", lambda: cfg)
    registry = Mock()
    monkeypatch.setattr(module, "ReferenceLibraryRegistry", lambda _: registry)
    run = AsyncMock(
        return_value={"schema": 1, "status": "published", "coverage": {"subset": False}}
    )
    monkeypatch.setattr(module, "_run", run)
    report = root / "reports" / "reference-test.json"
    return report, registry, run


@pytest.mark.parametrize("status,exit_code", [("published", 0), ("failed", 1)])
def test_cli_reports_publication_result_atomically(cli, capsys, status, exit_code):
    report, _, run = cli
    run.return_value = {"schema": 1, "status": status}
    assert module.main(["--report", str(report)]) == exit_code
    saved = json.loads(report.read_text())
    assert saved["schema"] == 1 and saved["status"] == status and saved["exit_code"] == exit_code
    assert saved["created_at"] and saved["elapsed_seconds"] >= 0
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert not list(report.parent.glob("tmp*"))


def test_source_preflight_failure_makes_no_network_clients(cli):
    report, registry, run = cli
    registry.catalog.side_effect = ReferenceUnavailable("Archive update is in progress")
    assert module.main(["--report", str(report)]) == 2
    run.assert_not_awaited()
    assert json.loads(report.read_text())["status"] == "setup_error"


def test_unexpected_failure_reports_type_without_connection_credentials(cli):
    report, _, run = cli
    run.side_effect = RuntimeError("https://user:secret@private.example")
    assert module.main(["--report", str(report)]) == 3
    assert json.loads(report.read_text())["error"] == "RuntimeError"
    assert "secret" not in report.read_text()


def test_report_refuses_symlink_target(tmp_path):
    target = tmp_path / "target"
    target.write_text("retained")
    link = tmp_path / "report"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        module.write_report(link, {"status": "failed"})
    assert target.read_text() == "retained"
