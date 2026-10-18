"""Operator preflight, safe reports and public-only clients without live calls."""

from __future__ import annotations

import json
import stat
from pathlib import Path
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


def test_unexpected_failure_keeps_traceback_private(cli, capsys):
    report, _, run = cli
    run.side_effect = RuntimeError("https://user:secret@private.example")
    assert module.main(["--report", str(report)]) == 3
    saved = json.loads(report.read_text())
    assert saved["error"] == "RuntimeError"
    assert "secret" not in report.read_text()
    assert "secret" not in capsys.readouterr().out
    diagnostic = report.parent.parent / saved["private_diagnostic"]
    details = diagnostic.read_text()
    assert "Traceback (most recent call last)" in details
    assert "RuntimeError: https://user:secret@private.example" in details
    assert "_run" in details
    assert stat.S_IMODE(diagnostic.stat().st_mode) == 0o600
    assert stat.S_IMODE(diagnostic.parent.stat().st_mode) == 0o700



@pytest.mark.parametrize("component", ["root", "ancestor", "diagnostics", "file"])
def test_private_diagnostic_refuses_symlink_traversal(tmp_path, monkeypatch, component):
    root = tmp_path / "index"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "existing.txt"
    target.write_text("retained")
    if component == "root":
        root.symlink_to(outside, target_is_directory=True)
    elif component == "ancestor":
        ancestor = tmp_path / "linked"
        ancestor.symlink_to(outside, target_is_directory=True)
        root = ancestor / "index"
    else:
        root.mkdir()
        if component == "diagnostics":
            (root / "diagnostics").symlink_to(outside, target_is_directory=True)
        else:
            diagnostics = root / "diagnostics"
            diagnostics.mkdir()
            monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="known"))
            (diagnostics / "failure-known.traceback.txt").symlink_to(target)
    with pytest.raises(OSError):
        module.private_diagnostic(root, RuntimeError("must stay private"))
    assert target.read_text() == "retained"
    assert sorted(path.name for path in outside.iterdir()) == ["existing.txt"]


def test_diagnostic_write_failure_preserves_original_crash(cli, monkeypatch, capsys):
    report, _, run = cli
    run.side_effect = RuntimeError("original-secret")
    monkeypatch.setattr(module, "private_diagnostic", Mock(side_effect=PermissionError("write-secret")))
    assert module.main(["--report", str(report)]) == 3
    saved = json.loads(report.read_text())
    assert saved["error"] == "RuntimeError"
    assert saved["status"] == "crashed"
    assert saved["exit_code"] == 3
    assert saved["private_diagnostic_saved"] is False
    assert "private_diagnostic" not in saved
    assert "secret" not in report.read_text() + capsys.readouterr().out


def test_unexpected_source_preflight_failure_keeps_private_startup_traceback(cli):
    report, registry, run = cli
    registry.catalog.side_effect = RuntimeError("source-startup-secret")
    assert module.main(["--report", str(report)]) == 3
    run.assert_not_awaited()
    saved = json.loads(report.read_text())
    assert "source-startup-secret" not in report.read_text()
    diagnostic = report.parent.parent / saved["private_diagnostic"]
    assert "source-startup-secret" in diagnostic.read_text()


def test_pre_config_crash_does_not_guess_a_diagnostic_root(cli, monkeypatch):
    report, _, run = cli
    monkeypatch.setattr(module, "get_config", Mock(side_effect=RuntimeError("config-secret")))
    diagnostic = Mock()
    monkeypatch.setattr(module, "private_diagnostic", diagnostic)
    assert module.main(["--report", str(report)]) == 3
    run.assert_not_awaited()
    diagnostic.assert_not_called()
    saved = json.loads(report.read_text())
    assert saved["private_diagnostic_saved"] is False
    assert "config-secret" not in report.read_text()


def test_report_refuses_symlink_target(tmp_path):
    target = tmp_path / "target"
    target.write_text("retained")
    link = tmp_path / "report"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        module.write_report(link, {"status": "failed"})
    assert target.read_text() == "retained"


async def test_index_job_uses_the_same_reference_factory_as_runtime(monkeypatch):
    import yaml
    config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    cfg = SimpleNamespace(raw=config,
                          env=SimpleNamespace(ollama_host="http://local", qdrant_host="local", qdrant_port=6333))
    ollama = SimpleNamespace(aclose=AsyncMock())
    qdrant = SimpleNamespace(close=Mock())
    embedder = object()
    factory = Mock(return_value=embedder)
    monkeypatch.setattr(module, "OllamaClient", lambda _: ollama)
    monkeypatch.setattr(module, "QdrantKB", lambda *_, **kw: qdrant)
    monkeypatch.setattr(module, "reference_embedder", factory)
    build = AsyncMock(return_value={"status": "published"})
    monkeypatch.setattr(module, "build_reference_index", build)
    args = SimpleNamespace(library="irs-federal", catalog="current", max_documents=None,
                           product_number=[], release_id=None, rebuild=False, allow_extraction_gap=())
    assert await module._run(args, cfg, object(), object()) == {"status": "published"}
    factory.assert_called_once_with(ollama, config)
    assert build.await_args.args[3] is embedder
    ollama.aclose.assert_awaited_once()
    qdrant.close.assert_called_once()


@pytest.mark.parametrize("flags", [
    ["--allow-extraction-gap", "*"],
    ["--allow-extraction-gap", "a" * 64],
    ["--release-id", "rel_" + "b" * 32, "--rebuild", "--allow-extraction-gap", "a" * 64],
    ["--release-id", "rel_" + "b" * 32, "--allow-extraction-gap", "a" * 64,
     "--allow-extraction-gap", "a" * 64],
])
def test_gap_approvals_are_explicit_and_validated_before_execution(cli, flags):
    report, registry, run = cli
    assert module.main(["--report", str(report), *flags]) == 2
    registry.catalog.assert_not_called()
    run.assert_not_awaited()


def test_cli_passes_exact_approved_gap_ids_to_worker(cli):
    report, _, run = cli
    identity = "a" * 64
    release = "rel_" + "b" * 32
    assert module.main(["--report", str(report), "--release-id", release,
                        "--allow-extraction-gap", identity]) == 0
    args = run.call_args.args[0]
    assert args.release_id == release
    assert args.allow_extraction_gap == (identity,)
