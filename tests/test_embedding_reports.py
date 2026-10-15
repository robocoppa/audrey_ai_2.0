"""Offline report composition must never turn mismatched runs into evidence."""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "embedding_reports", ROOT / "scripts/analysis/embedding_reports.py")
assert SPEC and SPEC.loader
reports = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reports
SPEC.loader.exec_module(reports)


def fixture(tmp_path):
    body = {"schema": 2, "language": "en", "public": True, "synthetic": False,
            "workload": "reference", "sources": [], "documents": [], "queries": []}
    for index, split in enumerate(("calibration", "holdout")):
        source = f"source-{split}"
        body["sources"].append({"id": source, "url": f"https://example.gov/{split}.pdf",
                                "sha256": str(index) * 64, "split": split,
                                "title": f"Public {split}", "pages": 1,
                                "license": "US government work"})
        for suffix in ("first", "alternate"):
            body["documents"].append({"id": f"{split}-{suffix}", "text": "The preferred color is navy.",
                                      "source_id": source, "page": 1, "offset": 0,
                                      "split": split})
        for positive in (True, False):
            body["queries"].append({"id": f"{split}-{'yes' if positive else 'no'}",
                "text": "What is the preferred color?" if positive else "What is the bank balance?",
                "group": "reference", "split": split, "answerable": positive,
                "relevant": [f"{split}-first"] if positive else [],
                "evidence": [{"source_id": source, "page": 1,
                              "quote": "The preferred color is navy."}] if positive else []})
    original = tmp_path / "original.json"
    original.write_text(json.dumps(body))
    return body, original


def corpus_results(path):
    corpora, metadata = reports.probe.load_fixture(path, synthetic=False)
    results = []
    for corpus in corpora:
        ids = [f"{corpus.split}-alternate", f"{corpus.split}-first"]
        rows = []
        for query in corpus.queries:
            rows.append({"id": query.id, "group": query.group, "kind": query.kind,
                         "split": query.split, "workload": query.workload, "repeat": 1,
                         "answerable": bool(query.relevant), "relevant": list(query.relevant),
                         "dense": ids, "hybrid": ids, "lexical": ids,
                         "candidates": [{"id": identity, "dense_cosine": .9 if query.relevant else .1,
                                         "lexical_overlap": 0., "lexical": False} for identity in ids]})
        results.append({"id": corpus.id, "workload": corpus.workload, "split": corpus.split,
                        "documents": len(corpus.documents), "unique_queries": len(corpus.queries),
                        "positive_queries": 1, "negative_queries": 1,
                        "queries": rows, "ranking": reports.probe.quality_summary(rows),
                        "current_policy": {}})
    return results, metadata


def measured_arm(identity, model, corpora):
    return {"arm": identity, "model": model, "status": "measured", "comparable": True,
            "dimensions": 768, "corpora": copy.deepcopy(corpora),
            "workloads": reports.probe.summarize_workloads(corpora),
            "format": {"document_prefix": "search_document: " if identity == "prefixed" else "",
                       "query_prefix": "search_query: " if identity == "prefixed" else ""}}


def report_pair(path):
    corpora, metadata = corpus_results(path)
    models = ["embeddinggemma-2:latest", "qwen3-embedding:0.6b",
              "qwen3-embedding:4b", "qwen3-embedding:8b"]
    candidate = {"schema": 2, "status": "failed", "measurement_status": "incomplete",
                 "created_at": "2026-10-15T00:00:00Z", "fixtures": {"public": metadata},
                 "retrieval": {"cosine": "exact", "qdrant_contacted": False},
                 "settings": {"rounds": 1, "batch_size": 16, "cold_samples": 3,
                              "stress_batches64": 2, "timeout_seconds": 120, "keep_alive": "10m",
                              "candidate_models": models, "estimated_embedding_requests": 100},
                 "provenance": {"ollama_version": "0.40.4", "installed_models": {
                     "nomic-embed-text:latest": {"digest": "a" * 64}}},
                 "errors": [{"arm": identity, "error": "unload race"} for identity in reports.BASELINES],
                 "embedding_calls": 82, "results": []}
    candidate["results"] = [measured_arm(identity, model, corpora)
                            for identity, model in zip(reports.CANDIDATES, models, strict=True)]
    candidate["results"].extend({"arm": identity, "model": "nomic-embed-text:latest",
                                "status": "incomplete", "comparable": False}
                               for identity in reports.BASELINES)
    baseline = copy.deepcopy(candidate)
    baseline.update(status="measured", measurement_status="completed", errors=[], embedding_calls=40)
    baseline["settings"]["candidate_models"] = []
    baseline["settings"]["estimated_embedding_requests"] = 40
    baseline["results"] = [measured_arm(identity, "nomic-embed-text:latest", corpora)
                           for identity in reports.BASELINES]
    return candidate, baseline


@pytest.mark.parametrize("format_", ("json", "json.gz", "log"))
def test_read_full_report_formats(tmp_path, format_):
    _, path = fixture(tmp_path)
    report, _ = report_pair(path)
    output = tmp_path / f"report.{format_}"
    data = json.dumps(report).encode()
    if format_ == "json.gz":
        output.write_bytes(gzip.compress(data))
    elif format_ == "log":
        output.write_bytes(b"Starting detached probe\n" + data + b"\nfinished: exit 1\n")
    else:
        output.write_bytes(data)
    assert reports.read_report(output) == report


def test_rescore_adds_alternate_without_reranking_or_mutation(tmp_path):
    body, original = fixture(tmp_path)
    candidate, _ = report_pair(original)
    before = copy.deepcopy(candidate)
    for query in body["queries"]:
        if query["answerable"]:
            query["relevant"].append(f"{query['split']}-alternate")
    revised = tmp_path / "revised.json"
    revised.write_text(json.dumps(body))
    rescored = reports.rescore_reference_labels(candidate, original, revised)
    assert candidate == before
    assert reports.fixture_inputs_sha256(original) == reports.fixture_inputs_sha256(revised)
    for old_arm, new_arm in zip(before["results"][:4], rescored["results"][:4], strict=True):
        for old_corpus, new_corpus in zip(old_arm["corpora"], new_arm["corpora"], strict=True):
            assert old_corpus["ranking"]["dense"]["hit_rate_at_1"] == 0
            assert new_corpus["ranking"]["dense"]["hit_rate_at_1"] == 1
            for old_query, new_query in zip(old_corpus["queries"], new_corpus["queries"], strict=True):
                assert all(old_query[key] == new_query[key] for key in ("dense", "hybrid", "candidates"))
    assert rescored["quality_recommendation"]["status"] == "inconclusive"
    assert rescored["label_repair"]["thresholds_tuned_on_holdout"] is False
    assert rescored["label_repair"]["new_embedding_calls"] == 0
    assert rescored["fixtures"]["public"]["sha256"] == hashlib.sha256(revised.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ("document", "query", "source", "missing_query"))
def test_annotation_rejects_changed_model_inputs(tmp_path, change):
    body, original = fixture(tmp_path)
    candidate, _ = report_pair(original)
    if change == "document":
        body["documents"][0]["text"] = "New text"
    elif change == "query":
        body["queries"][0]["text"] = "New question"
    elif change == "source":
        body["sources"][0]["url"] = "https://example.gov/changed.pdf"
    else:
        body["queries"].pop()
    revised = tmp_path / "revised.json"
    revised.write_text(json.dumps(body))
    with pytest.raises(reports.ReportError, match="changes model inputs"):
        reports.rescore_reference_labels(candidate, original, revised)


def test_annotation_rejects_wrong_original_or_report_query(tmp_path):
    _, original = fixture(tmp_path)
    candidate, _ = report_pair(original)
    candidate["fixtures"]["public"]["sha256"] = "0" * 64
    with pytest.raises(reports.ReportError, match="original public fixture"):
        reports.rescore_reference_labels(candidate, original, original)
    candidate, _ = report_pair(original)
    candidate["results"][0]["corpora"][0]["queries"][0]["id"] = "unknown"
    with pytest.raises(reports.ReportError, match="query IDs"):
        reports.rescore_reference_labels(candidate, original, original)


def test_merge_retains_provenance_actual_calls_and_inputs(tmp_path):
    _, path = fixture(tmp_path)
    candidate, baseline = report_pair(path)
    before = copy.deepcopy((candidate, baseline))
    merged = reports.merge_baselines(candidate, baseline)
    assert (candidate, baseline) == before
    assert merged["embedding_calls"] == 122
    assert merged["status"] == "measured"
    assert merged["measurement_status"] == "completed"
    assert merged["errors"] == []
    assert len(merged["results"]) == 6
    assert merged["settings"]["estimated_embedding_requests"] == 100
    composition = merged["report_composition"]
    assert composition["historical_errors"] == candidate["errors"]
    assert composition["superseded_baseline_results"] == candidate["results"][-2:]
    assert composition["new_embedding_calls_during_assembly"] == 0
    assert composition["same_gpu_session_proven"] is False


@pytest.mark.parametrize("change,match", (
    (lambda a, b: b["fixtures"]["public"].update(sha256="other"), "fixture"),
    (lambda a, b: b["retrieval"].update(cosine="ANN"), "retrieval"),
    (lambda a, b: b["settings"].update(rounds=3), "rounds"),
    (lambda a, b: b["settings"].update(batch_size=8), "batch_size"),
    (lambda a, b: b["provenance"].update(ollama_version="other"), "Ollama"),
    (lambda a, b: b["provenance"]["installed_models"]["nomic-embed-text:latest"].update(digest="b" * 64), "digest"),
    (lambda a, b: b["results"][0].update(comparable=False), "complete comparable"),
    (lambda a, b: a["results"][0].update(status="incomplete"), "complete comparable"),
    (lambda a, b: b["results"][0]["corpora"][0]["queries"][0].update(id="different"), "query IDs"),
    (lambda a, b: b["results"][0]["corpora"][0]["queries"][0].update(relevant=[]), "query IDs"),
    (lambda a, b: b["results"][0].update(dimensions=512), "dimensions"),
    (lambda a, b: b["results"][0]["format"].update(query_prefix=""), "formatting"),
    (lambda a, b: a["errors"].append({"arm": "candidate", "error": "bad"}), "outside"),
    (lambda a, b: b["results"].append(copy.deepcopy(b["results"][0])), "duplicate arm"),
))
def test_merge_rejects_incompatible_or_incomplete_runs(tmp_path, change, match):
    _, path = fixture(tmp_path)
    candidate, baseline = report_pair(path)
    change(candidate, baseline)
    with pytest.raises(reports.ReportError, match=match):
        reports.merge_baselines(candidate, baseline)
