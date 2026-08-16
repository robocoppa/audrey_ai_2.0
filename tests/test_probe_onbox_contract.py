"""`probe-onbox.sh` must keep the properties that make an unattended run survive.

This is a shell script, so the tests are structural rather than behavioural —
but every property asserted here has a specific failure behind it:

- **Self-detaching.** Long runs previously died to laptop-sleep process
  orphaning and looked like broken probes for two days. An incantation you have
  to remember (`nohup … &`) is one you eventually forget, at the cost of the run.
- **`setsid`, not just `nohup`.** nohup only ignores SIGHUP; a closed terminal
  can still take the process group with it.
- **Logs on a bind mount.** A log written inside the container is destroyed by
  the `up -d --build` that a config change requires.
- **Container clock for the stamp.** The box runs three clocks (host PDT,
  container logs MDT, `docker inspect` UTC), so a host-stamped filename would
  not line up with the log lines inside the run it names.
- **The Telegram format is eval-onbox.sh's, unchanged.** It works and is
  trusted; it is not to be "improved".
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "probe-onbox.sh"


@pytest.fixture(scope="module")
def src() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_the_script_exists_and_is_executable():
    assert _SCRIPT.is_file()
    assert _SCRIPT.stat().st_mode & 0o111, "must be chmod +x to run on the box"


class TestItSurvivesTheSessionClosing:
    def test_it_self_detaches_rather_than_documenting_nohup(self, src):
        assert "_PROBE_DETACHED" in src, "no re-exec marker — it would loop forever"
        assert "setsid nohup" in src

    def test_setsid_not_just_nohup(self, src):
        # nohup alone only ignores SIGHUP. Leaving the session entirely is the
        # property that actually survives a dropped SSH connection.
        assert "setsid" in src

    def test_there_is_a_foreground_escape_hatch(self, src):
        # Debugging a wrapper that always detaches is miserable.
        assert "FOREGROUND" in src

    def test_it_tells_you_where_the_log_is_before_detaching(self, src):
        assert "Safe to close this session" in src
        assert "tail -f" in src


class TestItRunsWhereThePythonIs:
    def test_it_execs_inside_the_container(self, src):
        # Unraid has no python3; the probe cannot run on the host at all.
        assert "docker exec" in src
        assert "python3" in src

    def test_it_copies_the_probe_in(self, src):
        # The repo is NOT bind-mounted into audrey-ai — only config.yaml,
        # /data and /datasets — so a `git pull` on the host is invisible
        # inside the container until the copy happens.
        assert "docker cp" in src

    def test_it_copies_every_run_not_once(self, src):
        assert "not mounted" in src or "not bind-mounted" in src, (
            "the reason for copying every run must stay written down"
        )

    def test_it_defaults_to_the_audrey_container(self, src):
        assert 'CONTAINER="${CONTAINER:-audrey-ai}"' in src


class TestLogsSurviveARebuild:
    def test_the_log_dir_is_under_appdata_not_the_container(self, src):
        # `up -d --build` is required after a config.yaml change, and would
        # take a container-local log with it.
        assert 'OUT_DIR="${OUT_DIR:-${APPDATA}/testing-out/probes}"' in src

    def test_appdata_defaults_to_the_real_box_path(self, src):
        # ⚠️ WITH the _2.0 suffix. `/mnt/user/appdata/audrey` does not exist.
        assert "/mnt/user/appdata/audrey_ai_2.0" in src


class TestTheStampComesFromTheContainerClock:
    def test_stamp_is_read_from_the_container(self, src):
        assert 'docker exec "${CONTAINER}" date' in src

    def test_it_falls_back_to_the_host_clock(self, src):
        # A stopped container must not stop you naming a log file.
        assert "|| date +" in src


class TestTelegramFormatIsUnchanged:
    """The user confirmed this format works. It is not to be redesigned."""

    def test_it_matches_eval_onbox_s_message_shape(self, src):
        assert "finished (exit ${rc})" in src
        assert "（summary unavailable）" in src  # full-width parens, house style

    def test_the_full_log_goes_as_a_document_not_inline(self, src):
        # Telegram text messages cap at 4096 chars.
        assert "sendDocument" in src
        assert "sendMessage" in src

    def test_inlining_a_log_tail_is_recorded_as_reverted(self, src):
        assert "reverted" in src, (
            "the note explaining why the format is not to be changed is gone"
        )

    def test_notify_failures_are_non_fatal(self, src):
        # The probe already ran; a failed send must not mask its result.
        assert src.count("WARN: Telegram") >= 2


class TestExitCodes:
    def test_exit_one_is_described_as_a_finding_not_a_failure(self, src):
        # router_probe exits 1 for a DISQUALIFIED candidate;
        # check_model_inventory exits 1 when config names a missing model.
        # Both are the point of running them.
        assert "FINDINGS" in src

    def test_the_probe_s_exit_code_is_propagated(self, src):
        assert 'exit "${rc}"' in src

    def test_usage_and_missing_probe_exit_two(self, src):
        assert src.count("exit 2") >= 2


class TestArgumentHandling:
    def test_key_value_pairs_become_container_env(self, src):
        assert 'ENV_FLAGS+=("-e" "${kv}")' in src

    def test_args_is_special_cased_for_flags(self, src):
        assert "ARGS=*)" in src

    def test_an_unrecognised_argument_warns_rather_than_being_swallowed(self, src):
        assert "WARN: ignoring" in src
