"""The short installed-image smoke detects scheduling regressions and bad setup."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/smoke/smoke_agent_admission.py"


def _module():
    spec = importlib.util.spec_from_file_location("admission_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=ROOT,
        capture_output=True, text=True, check=False, timeout=15,
    )


def test_standalone_smoke_reports_all_cases_and_real_configured_capacities(tmp_path):
    config = tmp_path / "isolated.yaml"
    config.write_text("gpu:\n  concurrency: 2\nfairness:\n  max_inflight_per_user: 2\n  background_aging_seconds: 17\n")
    result = _run("--config", str(config), "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["configured"] == {"gpu_concurrency": 2, "max_inflight_per_user": 2, "background_aging_seconds": 17}
    assert len(receipt["checks"]) == 5
    assert all(row["passed"] for row in receipt["checks"])
    assert "no models/network/accounts" in receipt["scope"]


def test_short_supported_aging_does_not_age_the_unaged_priority_fixture(tmp_path):
    config = tmp_path / "short-aging.yaml"
    config.write_text("fairness:\n  background_aging_seconds: 0.0001\n")
    result = _run("--config", str(config), "--json")
    receipt = json.loads(result.stdout)
    assert result.returncode == 0, receipt
    assert receipt["configured"]["background_aging_seconds"] == 0.0001
    assert receipt["fixtures"]["unaged_priority_aging_seconds"] == 60
    assert receipt["checks"][2]["check"] == "gpu_background_aging_synthetic_timestamp"


@pytest.mark.parametrize("text", ["[]", "fairness: [invalid]", "fairness:\n  background_aging_seconds: 0", "gpu:\n  concurrency: 100"])
def test_invalid_configuration_fails_before_any_smoke_case(tmp_path, text):
    config = tmp_path / "invalid.yaml"
    config.write_text(text)
    result = _run("--config", str(config), "--json")
    receipt = json.loads(result.stdout)
    assert result.returncode == receipt["exit_code"] == 2
    assert receipt["checks"] == []
    assert "Traceback" not in result.stderr


def test_missing_config_is_a_setup_error(tmp_path):
    result = _run("--config", str(tmp_path / "missing.yaml"), "--json")
    assert result.returncode == 2
    assert json.loads(result.stdout)["checks"] == []


@pytest.mark.parametrize("deadline", ["0", "11", "nan", "inf"])
def test_unbounded_or_invalid_deadlines_are_rejected(deadline):
    result = _run("--timeout-seconds", deadline)
    assert result.returncode == 2
    assert "at most 10" in result.stderr


def test_old_backend_support_is_reported_as_setup_error(monkeypatch):
    module = _module()

    def missing(_path):
        raise module.SmokeSetupError("Installed backend lacks admission support; rebuild audrey.")

    monkeypatch.setattr(module, "load_runtime", missing)
    assert module.main(["--json"]) == 2


async def test_disabled_cooperative_handoff_is_a_finding_and_leaves_resources_idle(monkeypatch):
    from audrey.routes.inflight import _PrioritySemaphore

    module = _module()
    runtime = module.load_runtime(ROOT / "config.yaml")
    monkeypatch.setattr(_PrioritySemaphore, "_notify_background_permits", lambda _self: None)
    monkeypatch.setattr(module, "CASES", (("chat_priority_cap_one", module.chat_priority),))
    receipts = await module.run(runtime, 0.05)
    assert receipts == [{"check": "chat_priority_cap_one", "passed": False, "reason": "Timed out."}]
    # Scenario.__aexit__ checks idle resources even on this failure path.
    assert runtime.current() == "interactive" and runtime.queued_at() is None


def test_unexpected_failure_has_a_sanitized_crash_receipt(monkeypatch, capsys):
    module = _module()

    def broken(_path):
        raise RuntimeError("private fixture content must not enter the report")

    monkeypatch.setattr(module, "load_runtime", broken)
    assert module.main(["--json"]) == 3
    assert "private fixture" not in capsys.readouterr().out
