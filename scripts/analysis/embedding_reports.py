#!/usr/bin/env python3
"""Offline rescoring and assembly of embedding measurements, without model calls.

Original reports remain immutable. A label repair is retrospective annotation;
it changes neither saved rankings nor the declared calibration objective. A
baseline continuation may replace failed baseline arms only after the fixture,
retrieval, runtime, and measurement contracts have been checked.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_REPORT_BYTES = 64 * 1024 * 1024
CANDIDATES = ("candidate", "candidate_2", "candidate_3", "candidate_4")
BASELINES = ("prefixed", "raw")
MATCHED_SETTINGS = ("rounds", "batch_size", "cold_samples", "stress_batches64",
                    "timeout_seconds", "keep_alive")


class ReportError(ValueError):
    """An input cannot safely be rescored or combined."""


def _load_probe():
    path = ROOT / "scripts/probes/embedding_document_probe.py"
    spec = importlib.util.spec_from_file_location("embedding_reports_probe", path)
    if spec is None or spec.loader is None:
        raise ReportError("embedding document probe is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def read_report(path: Path | str) -> dict:
    """Read a full plain JSON, gzip JSON, or probe wrapper-log report."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rb") as stream:
            data = stream.read(MAX_REPORT_BYTES + 1)
        if len(data) > MAX_REPORT_BYTES:
            raise ReportError("report exceeds 64 MiB")
        text = data.decode("utf-8")
        start = re.search(r"(?m)^\s*\{", text)
        if start is None:
            raise ReportError("report does not contain a JSON object")
        report, _ = json.JSONDecoder().raw_decode(text[start.start():].lstrip())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportError("report is unreadable or contains invalid JSON") from error
    if (not isinstance(report, dict) or report.get("schema") != 2
            or not isinstance(report.get("results"), list)
            or not isinstance(report.get("fixtures"), dict)):
        raise ReportError("expected a full schema2 embedding document report")
    return report


def fixture_inputs_sha256(path: Path | str) -> str:
    """Hash model/scoring inputs while excluding relevance annotations."""
    return probe.fixture_inputs_sha256(json.loads(Path(path).read_bytes()))


def _sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refresh_metadata(report: dict, name: str, path: Path, *, synthetic: bool):
    if report["fixtures"].get(name, {}).get("sha256") != _sha256(path):
        raise ReportError(f"{name} fixture does not match original report checksum")
    _, metadata = probe.load_fixture(path, synthetic=synthetic)
    metadata["input_sha256"] = fixture_inputs_sha256(path)
    report["fixtures"][name] = metadata


def _rescore_corpus(corpus: dict, queries: dict) -> None:
    seen = set()
    for row in corpus["queries"]:
        query = queries.get(row["id"])
        if query is None:
            raise ReportError("report query IDs do not match revised fixture")
        if (row["split"], row["workload"], row["group"], row["kind"]) != (
                query.split, query.workload, query.group, query.kind):
            raise ReportError("report query context does not match revised fixture")
        key = (row["id"], row["repeat"])
        if key in seen:
            raise ReportError("report contains duplicate query/repeat rows")
        seen.add(key)
        row.update(relevant=list(query.relevant), answerable=bool(query.relevant))
    if {row["id"] for row in corpus["queries"]} != set(queries):
        raise ReportError("report query IDs do not match revised fixture")
    first = [row for row in corpus["queries"] if row["repeat"] == 1]
    if len(first) != len(queries):
        raise ReportError("report is missing first-repeat queries")
    corpus.update(positive_queries=sum(bool(query.relevant) for query in queries.values()),
                  negative_queries=sum(not query.relevant for query in queries.values()),
                  ranking=probe.quality_summary(first))
    current = probe.CURRENT[corpus["workload"]]
    corpus["current_policy"] = {"min_dense_score": current[0], "min_term_overlap": current[1],
                                **probe.admission_metrics(first, *current, diagnostics=True)}


def rescore_reference_labels(report: dict, original_fixture_path: Path | str,
                             revised_fixture_path: Path | str, *,
                             personal_fixture_path: Path | str | None = None) -> dict:
    """Apply annotation-only repairs to recorded results, without reranking.

The original public fixture must be preserved separately. Changes to source,
document, or query inputs are rejected. Optional personal-fixture metadata
refresh adds its input digest without changing any personal labels.
"""
    original, revised = Path(original_fixture_path), Path(revised_fixture_path)
    if report.get("fixtures", {}).get("public", {}).get("sha256") != _sha256(original):
        raise ReportError("original public fixture does not match report checksum")
    old_inputs = fixture_inputs_sha256(original)
    if old_inputs != fixture_inputs_sha256(revised):
        raise ReportError("label repair changes model inputs or scoring contexts")
    probe.load_fixture(original, synthetic=False)
    corpora, metadata = probe.load_fixture(revised, synthetic=False)
    metadata["input_sha256"] = old_inputs
    contexts = {corpus.id: corpus for corpus in corpora}
    result = copy.deepcopy(report)
    for arm in result["results"]:
        if not arm.get("comparable"):
            continue
        found = {row["id"] for row in arm["corpora"] if row["workload"] == "reference"}
        if found != set(contexts):
            raise ReportError("report reference corpora do not match revised fixture")
        for corpus in arm["corpora"]:
            if corpus["workload"] == "reference":
                _rescore_corpus(corpus, {query.id: query for query in contexts[corpus["id"]].queries})
        # The objective remains unchanged; only corrected calibration labels
        # select thresholds. Holdout labels never participate in that search.
        arm["workloads"] = probe.summarize_workloads(arm["corpora"])
    result["fixtures"]["public"] = metadata
    if personal_fixture_path is not None:
        _refresh_metadata(result, "personal", Path(personal_fixture_path), synthetic=True)
    result["label_repair"] = {
        "original_fixture_sha256": _sha256(original), "revised_fixture_sha256": _sha256(revised),
        "input_sha256": old_inputs, "retrospective_annotation_only": True,
        "rankings_unchanged": True, "new_embedding_calls": 0,
        "recalibration_on_corrected_calibration_labels": True,
        "thresholds_tuned_on_holdout": False,
        "meaning": "Retrospective label repair, not a new independent holdout measurement"}
    result["quality_recommendation"] = probe.recommendation(result)
    return result


def _arms(report: dict) -> dict:
    results = report.get("results", [])
    identities = [row.get("arm") for row in results]
    if len(identities) != len(set(identities)):
        raise ReportError("duplicate arm IDs")
    return {row["arm"]: row for row in results}


def _require_complete(arms: dict, expected: tuple[str, ...]) -> None:
    for identity in expected:
        arm = arms.get(identity, {})
        if (arm.get("status") != "measured" or arm.get("comparable") is not True
                or not arm.get("corpora") or not arm.get("workloads")
                or not isinstance(arm.get("dimensions"), int) or arm["dimensions"] < 1):
            raise ReportError(f"arm {identity} is not a complete comparable measurement")


def _query_contract(arm: dict) -> dict:
    contexts = {}
    for corpus in arm["corpora"]:
        identity = corpus["id"]
        if identity in contexts:
            raise ReportError("duplicate corpus IDs")
        rows = [(row["id"], row["repeat"], row["split"], row["workload"],
                 row["group"], row["kind"], row["answerable"], tuple(row["relevant"]))
                for row in corpus["queries"]]
        if len(rows) != len({(row[0], row[1]) for row in rows}):
            raise ReportError("duplicate query/repeat rows")
        contexts[identity] = (corpus["workload"], corpus["split"], corpus["documents"],
                              corpus["unique_queries"], tuple(sorted(rows)))
    return contexts


def _source_summary(report: dict, role: str) -> dict:
    return {"role": role, **{key: copy.deepcopy(report.get(key)) for key in (
        "created_at", "status", "measurement_status", "embedding_calls", "settings")}}


def merge_baselines(candidate_report: dict, baseline_report: dict) -> dict:
    """Combine four completed candidates with a matching two-arm continuation."""
    if candidate_report.get("schema") != 2 or baseline_report.get("schema") != 2:
        raise ReportError("only schema2 document reports can be combined")
    if candidate_report.get("fixtures") != baseline_report.get("fixtures"):
        raise ReportError("fixture metadata differs between reports")
    if candidate_report.get("retrieval") != baseline_report.get("retrieval"):
        raise ReportError("retrieval semantics differ between reports")
    for key in MATCHED_SETTINGS:
        if (key not in candidate_report.get("settings", {})
                or candidate_report["settings"][key] != baseline_report.get("settings", {}).get(key)):
            raise ReportError(f"measurement setting {key} differs between reports")
    version = candidate_report.get("provenance", {}).get("ollama_version")
    if not version or version != baseline_report.get("provenance", {}).get("ollama_version"):
        raise ReportError("Ollama versions differ between reports")
    candidates, baselines = _arms(candidate_report), _arms(baseline_report)
    if set(candidates) != set(CANDIDATES + BASELINES) or set(baselines) != set(BASELINES):
        raise ReportError("expected four candidate arms and a two-arm baseline continuation")
    _require_complete(candidates, CANDIDATES)
    _require_complete(baselines, BASELINES)
    if any(candidates[identity].get("comparable") for identity in BASELINES):
        raise ReportError("continuation may only replace incomplete baseline arms")
    if baseline_report.get("errors") or baseline_report.get("measurement_status") != "completed":
        raise ReportError("baseline continuation has incomplete measurement or errors")
    if any(error.get("arm") not in BASELINES for error in candidate_report.get("errors", [])):
        raise ReportError("candidate report contains errors outside superseded baselines")
    models = [candidates[identity]["model"] for identity in CANDIDATES]
    if (len(set(models)) != 4
            or models != candidate_report["settings"].get("candidate_models")):
        raise ReportError("candidate model tags differ from declared settings")
    nomic_models = {candidates[identity]["model"] for identity in BASELINES}
    if len(nomic_models) != 1 or nomic_models != {baselines[key]["model"] for key in BASELINES}:
        raise ReportError("baseline model tags differ between reports")
    nomic = next(iter(nomic_models))
    installed = candidate_report["provenance"].get("installed_models", {})
    continued = baseline_report["provenance"].get("installed_models", {})
    digest = installed.get(nomic, {}).get("digest")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest != continued.get(nomic, {}).get("digest")):
        raise ReportError("installed Nomic model digest differs between reports")
    if {baselines[key]["dimensions"] for key in BASELINES} != {768}:
        raise ReportError("Nomic baseline dimensions differ from expected native 768")
    for identity in BASELINES:
        expected = ({"document_prefix": "search_document: ", "query_prefix": "search_query: "}
                    if identity == "prefixed" else {"document_prefix": "", "query_prefix": ""})
        if baselines[identity].get("format") != expected:
            raise ReportError("baseline formatting differs from expected Nomic profile")
    contract = _query_contract(candidates[CANDIDATES[0]])
    for arm in [*(candidates[key] for key in CANDIDATES), *baselines.values()]:
        if _query_contract(arm) != contract:
            raise ReportError("query IDs, labels, repeats, or corpus contexts differ")
    calls = [report.get("embedding_calls") for report in (candidate_report, baseline_report)]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in calls):
        raise ReportError("both reports require nonnegative actual embedding call counts")
    result = copy.deepcopy(candidate_report)
    result["results"] = [copy.deepcopy(candidates[key]) for key in CANDIDATES]
    result["results"].extend(copy.deepcopy(baselines[key]) for key in BASELINES)
    result.update(status="measured", measurement_status="completed", errors=[], embedding_calls=sum(calls))
    result["residency_after"] = copy.deepcopy(baseline_report.get("residency_after"))
    result["report_composition"] = {
        "mode": "offline assembly of candidate measurements and baseline-only continuation",
        "source_reports": [_source_summary(candidate_report, "candidate_measurement"),
                           _source_summary(baseline_report, "baseline_continuation")],
        "historical_errors": copy.deepcopy(candidate_report.get("errors", [])),
        "superseded_baseline_results": [copy.deepcopy(candidates[key]) for key in BASELINES],
        "embedding_call_accounting": {"candidate_source_actual": calls[0],
                                      "baseline_source_actual": calls[1], "combined_actual": sum(calls)},
        "new_embedding_calls_during_assembly": 0, "same_gpu_session_proven": False,
        "meaning": "Original full-run estimate is retained; actual calls include failed original baseline attempts"}
    result["quality_recommendation"] = probe.recommendation(result)
    return result
