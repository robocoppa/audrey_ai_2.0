"""`check_model_inventory` catches models the config names but Ollama does not have.

Boot validation compares `config.yaml` against ITSELF — `_validate_deep_panel_pools`
checks every pool slot resolves in `model_registry`, and `model_registry` is in the
same file. A model that was never pulled therefore passes boot and fails at request
time, quietly, after taking a GPU-gate slot.

Two things are pinned harder than the rest:

1. **Role keys are found by descent, not by a hardcoded list.** Writing the list by
   hand missed `deep_panel_research.*.factchecker` and `kb.video.summarise_model`.
   A check that silently skips a role is worse than no check, so there is a test
   that invents a role key the tool has never seen.

2. **A cloud model's SIZE column is a bare `-`.** A size-shaped regex would drop
   every cloud entry and report them all as phantoms — a confident, completely
   wrong table of the kind this repo has produced six times.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import check_model_inventory as cmi  # noqa: E402

# Real `ollama list` output, including the bare-`-` cloud rows and a 274 MB entry.
_OLLAMA_LIST = """\
NAME                             ID              SIZE      MODIFIED
qwen3.8:latest                   22130167c4c2    17 GB     5 hours ago
glm-5.2:cloud                    ce8fd6f94793    -         8 weeks ago
nomic-embed-text:latest          0a109f422b47    274 MB    4 months ago
qwen3-vl:32b                     ff2e46876908    20 GB     4 months ago
qwen3.6:35b                      07d35212591f    23 GB     3 months ago
"""


class TestParsingOllamaList:
    def test_cloud_rows_survive_the_bare_dash_size(self):
        inv = cmi.parse_ollama_list(_OLLAMA_LIST)
        assert "glm-5.2:cloud" in inv.names
        assert inv.sizes.get("glm-5.2:cloud") is None

    def test_the_header_row_is_not_a_model(self):
        inv = cmi.parse_ollama_list(_OLLAMA_LIST)
        assert not any(n.upper() == "NAME" for n in inv.names)
        assert len(inv.names) == 5

    def test_sizes_parse_across_units(self):
        inv = cmi.parse_ollama_list(_OLLAMA_LIST)
        assert inv.sizes["qwen3.8:latest"] == 17 * 10**9
        assert inv.sizes["nomic-embed-text:latest"] == 274 * 10**6

    def test_blank_and_ragged_lines_are_skipped(self):
        inv = cmi.parse_ollama_list("\n\nqwen3.8:latest\n   \nfoo:bar x 1 GB now\n")
        assert inv.names == {"foo:bar"}  # the one-column line has nothing to parse


class TestFindingModelNamesInConfig:
    def test_a_role_key_the_tool_has_never_seen_is_still_found(self):
        # The whole point of descending instead of listing roles. If someone
        # adds `deep_panel_research.general.summariser` next month, it must be
        # checked without editing the script.
        cfg = {"deep_panel_research": {"general": {"a_brand_new_role": "ghost:1b"}}}
        found = cmi.config_models(cfg)
        assert "ghost:1b" in found
        assert found["ghost:1b"] == ["deep_panel_research.general.a_brand_new_role"]

    def test_registry_specs_contribute_their_name_not_their_location(self):
        cfg = {"model_registry": {"code": [
            {"name": "m:1b", "priority": 100, "location": "local"},
        ]}}
        found = cmi.config_models(cfg)
        assert "m:1b" in found
        assert "local" not in found  # would otherwise be checked as a model

    def test_worker_index_is_recorded_so_the_primary_can_be_ranked(self):
        cfg = {"deep_panel": {"code": {"workers": ["first:1b", "second:1b"]}}}
        found = cmi.config_models(cfg)
        assert found["first:1b"] == ["deep_panel.code.workers[0]"]
        assert found["second:1b"] == ["deep_panel.code.workers[1]"]

    def test_sections_outside_the_allowlist_are_ignored(self):
        # `version: "7.0.0"` and URLs must not be mistaken for models.
        cfg = {"version": "7.0.0", "tools": {"servers": ["http://custom-tools:8001"]}}
        assert cmi.config_models(cfg) == {}

    def test_the_kb_embedder_counts_as_referenced(self):
        # Reached outside model_registry. Without this it lands on the reclaim
        # list and someone deletes the KB's embedder.
        cfg = {"kb": {"text_embedder": "nomic-embed-text"}}
        assert "nomic-embed-text" in cmi.config_models(cfg)


class TestTheRealConfigIsFullyCovered:
    """Guards against a model-bearing section being added and not allowlisted."""

    def test_every_role_the_real_config_uses_is_reachable(self):
        import yaml
        cfg = yaml.safe_load((_ROOT / "config.yaml").read_text())
        found = cmi.config_models(cfg)
        paths = {p for ps in found.values() for p in ps}
        # The two roles a hand-written list missed on the first attempt.
        assert any("factchecker" in p for p in paths), "research factchecker missed"
        assert any("summarise_model" in p for p in paths), "video summariser missed"
        # And the load-bearing ones.
        assert "router.model" in paths
        assert any("tool_capable_models" in p for p in paths)
        assert any("text_embedder" in p for p in paths)
        assert any(p.startswith("model_registry.vl") for p in paths)

    def test_the_router_model_is_named_and_would_be_checked(self):
        import yaml
        cfg = yaml.safe_load((_ROOT / "config.yaml").read_text())
        found = cmi.config_models(cfg)
        router = cfg["router"]["model"]
        assert router in found
        assert cmi._severity("router.model") == "CRITICAL"


class TestBareNamesMeanLatest:
    """The one false positive the first real run produced.

    `config.yaml` says `kb.text_embedder: "nomic-embed-text"`; `ollama list`
    prints `nomic-embed-text:latest`. Ollama implies `:latest`, so they are the
    same model — but a naive string compare called it BOTH a phantom and
    reclaimable in the same report.
    """

    def test_a_bare_name_gets_latest(self):
        assert cmi.normalise("nomic-embed-text") == "nomic-embed-text:latest"

    def test_a_tagged_name_is_untouched(self):
        assert cmi.normalise("qwen3.6:35b") == "qwen3.6:35b"
        assert cmi.normalise("qwen3.5:397b-cloud") == "qwen3.5:397b-cloud"

    def test_the_embedder_is_neither_missing_nor_reclaimable(self):
        cfg = {"kb": {"text_embedder": "nomic-embed-text"}}
        rep = cmi.build_report(cmi.config_models(cfg), cmi.parse_ollama_list(_OLLAMA_LIST))
        assert [m["model"] for m in rep["missing"]] == []
        assert "nomic-embed-text:latest" not in {u["model"] for u in rep["unreferenced"]}

    def test_it_works_the_other_way_round_too(self):
        # Config tagged `:latest`, box bare — same model, still not a phantom.
        cfg = {"router": {"model": "solo:latest"}}
        inv = cmi.Inventory(names={"solo:latest"})
        assert cmi.build_report(cmi.config_models(cfg), inv)["missing"] == []


class TestSeverity:
    def test_the_router_outranks_everything(self):
        assert cmi._severity("router.model") == "CRITICAL"

    def test_the_first_worker_outranks_later_ones(self):
        assert cmi._severity("deep_panel.code.workers[0]") == "HIGH"
        assert cmi._severity("deep_panel.code.workers[2]") == "MEDIUM"

    def test_tool_capable_is_high_because_it_fails_silently(self):
        # A model that cannot call tools just answers — no error to notice.
        assert cmi._severity("fast_path.tool_capable_models[3]") == "HIGH"

    def test_passthrough_only_is_low(self):
        assert cmi._severity("passthrough.allowed_models[1]") == "LOW"

    def test_an_unrecognised_path_still_gets_a_level(self):
        assert cmi._severity("something.new") == "UNKNOWN"


class TestReport:
    def _inv(self):
        return cmi.parse_ollama_list(_OLLAMA_LIST)

    def test_a_phantom_is_reported_with_its_paths(self):
        cfg = {"deep_panel_local": {"reasoning": {"workers": ["gone:32b", "qwen3.8:latest"]}}}
        rep = cmi.build_report(cmi.config_models(cfg), self._inv())
        assert [m["model"] for m in rep["missing"]] == ["gone:32b"]
        assert rep["missing"][0]["severity"] == "HIGH"  # workers[0]
        assert rep["missing"][0]["paths"] == ["deep_panel_local.reasoning.workers[0]"]

    def test_missing_are_sorted_worst_first(self):
        cfg = {
            "passthrough": {"allowed_models": ["low:1b"]},
            "router": {"model": "crit:1b"},
            "deep_panel": {"code": {"workers": ["high:1b"]}},
        }
        rep = cmi.build_report(cmi.config_models(cfg), self._inv())
        assert [m["severity"] for m in rep["missing"]] == ["CRITICAL", "HIGH", "LOW"]

    def test_a_model_in_two_roles_takes_the_worse_one(self):
        cfg = {
            "passthrough": {"allowed_models": ["gone:1b"]},
            "router": {"model": "gone:1b"},
        }
        rep = cmi.build_report(cmi.config_models(cfg), self._inv())
        assert rep["missing"][0]["severity"] == "CRITICAL"

    def test_unreferenced_weights_are_totalled_for_reclaim(self):
        cfg = {"model_registry": {"general": [{"name": "qwen3.8:latest"}]}}
        rep = cmi.build_report(cmi.config_models(cfg), self._inv())
        names = {u["model"] for u in rep["unreferenced"]}
        assert "qwen3.6:35b" in names
        assert rep["reclaimable_bytes"] >= 23 * 10**9

    def test_a_clean_box_reports_no_missing(self):
        cfg = {"model_registry": {"general": [{"name": "qwen3.8:latest"}]}}
        rep = cmi.build_report(cmi.config_models(cfg), self._inv())
        assert rep["missing"] == []
        assert "✅" in cmi.render(rep)


class TestItRefusesToGuess:
    def test_an_empty_inventory_is_an_error_not_a_report(self, tmp_path, capsys):
        # Reporting every model as missing off an empty parse is exactly the
        # "confident, plausible, completely wrong table" failure mode.
        cfg = tmp_path / "config.yaml"
        cfg.write_text("router:\n  model: qwen3:4b\n")
        empty = tmp_path / "tags.txt"
        empty.write_text("NAME  ID  SIZE  MODIFIED\n")
        rc = cmi.main(["--config", str(cfg), "--ollama-list", str(empty)])
        assert rc == 2
        assert "refusing" in capsys.readouterr().err

    def test_a_missing_config_is_an_error(self, tmp_path, capsys):
        rc = cmi.main(["--config", str(tmp_path / "nope.yaml"),
                       "--ollama-list", str(tmp_path / "also-nope.txt")])
        assert rc == 2
        assert "no such config" in capsys.readouterr().err

    def test_no_inventory_source_is_a_usage_error(self, tmp_path):
        with pytest.raises(SystemExit):
            cmi.main(["--config", str(tmp_path / "c.yaml")])


class TestExitStatusGatesADeploy:
    def test_phantoms_exit_nonzero(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("router:\n  model: not-pulled:1b\n")
        tags = tmp_path / "tags.txt"
        tags.write_text(_OLLAMA_LIST)
        assert cmi.main(["--config", str(cfg), "--ollama-list", str(tags)]) == 1

    def test_unreferenced_alone_exits_zero(self, tmp_path):
        # Reclaimable weights are information, not a failure.
        cfg = tmp_path / "config.yaml"
        cfg.write_text("router:\n  model: qwen3.8:latest\n")
        tags = tmp_path / "tags.txt"
        tags.write_text(_OLLAMA_LIST)
        assert cmi.main(["--config", str(cfg), "--ollama-list", str(tags)]) == 0
