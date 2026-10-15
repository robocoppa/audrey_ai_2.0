"""Hermetic boundaries for the deeper, read-only embedding comparison."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "embedding_document_probe", os.environ.get(
        "EMBEDDING_DOCUMENT_PROBE_PATH", ROOT / "scripts/probes/embedding_document_probe.py")
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def fixture_body(*, synthetic=True):
    body = {"schema": 2, "language": "en", "public": True, "synthetic": synthetic,
            "workload": "memory" if synthetic else "reference",
            "sources": [], "documents": [], "queries": []}
    for index, split in enumerate(("calibration", "holdout")):
        source = {"id": f"source-{split}", "split": split, "title": f"Public {split}",
                  "url": f"https://example.gov/{split}.pdf", "sha256": str(index) * 64,
                  "pages": 1, "license": "US government work"}
        doc_id = f"doc-{split}"
        text = f"The {split} preferred color is navy. A preferred color is a personal preference."
        body["sources"].append(source)
        body["documents"].append({"id": doc_id, "text": text, "source_id": source["id"],
                                  "page": 1, "offset": 0, "split": split})
        for positive in (True, False):
            body["queries"].append({
                "id": f"{split}-{'positive' if positive else 'negative'}",
                "text": f"What is the {split} preferred color?" if positive
                        else f"What is the {split} bank balance?",
                "group": "memory" if synthetic else "reference", "split": split,
                "relevant": [doc_id] if positive else [], "answerable": positive,
                "evidence": [{"source_id": source["id"], "page": 1,
                              "quote": f"The {split} preferred color is navy."}] if positive else []})
    return body


def load_body(tmp_path, body=None, *, synthetic=True):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(body or fixture_body(synthetic=synthetic)))
    return probe.load_fixture(path, synthetic=synthetic)


def evidence_row(identity, *, positive=True, split="calibration", score=.8,
                 overlap=0, lexical=False, first="right"):
    return {"id": identity, "split": split, "relevant": ["right"] if positive else [],
            "candidates": [{"id": first, "dense_cosine": score,
                            "lexical_overlap": overlap, "lexical": lexical}]}


def test_real_public_source_and_evidence_provenance_validates(tmp_path):
    corpora, metadata = load_body(tmp_path, synthetic=False)
    assert {corpus.split for corpus in corpora} == {"calibration", "holdout"}
    assert metadata["synthetic"] is False
    assert len(metadata["sha256"]) == 64
    assert all(corpus.workload == "reference" for corpus in corpora)


@pytest.mark.parametrize("change,match", [
    (lambda body: body["documents"][1].update(source_id="source-calibration"), "source leakage"),
    (lambda body: body["queries"][2].update(relevant=["doc-calibration"]), "leaks across"),
    (lambda body: body["sources"].append(body["sources"][0]), "unique"),
    (lambda body: body["documents"][0].update(workload="history"), "leaks across"),
    (lambda body: body["queries"][0]["evidence"][0].update(quote="invented unsupported quote"), "quote"),
    (lambda body: body["queries"][0].update(answerable=False), "answerable"),
    (lambda body: body.update(language="ja"), "English"),
    (lambda body: body.update(public=False), "public"),
    (lambda body: body["queries"].pop(1), "no-answer"),
])
def test_bad_fixture_or_split_leakage_rejected(tmp_path, change, match):
    body = fixture_body()
    change(body)
    with pytest.raises(probe.ProbeError, match=match):
        load_body(tmp_path, body)


def test_missing_real_evidence_and_source_checksum_are_rejected(tmp_path):
    body = fixture_body(synthetic=False)
    body["queries"][0]["evidence"] = []
    with pytest.raises(probe.ProbeError, match="require source evidence"):
        load_body(tmp_path, body, synthetic=False)
    body = fixture_body(synthetic=False)
    body["sources"][0]["sha256"] = "not-a-checksum"
    with pytest.raises(probe.ProbeError, match="SHA256"):
        load_body(tmp_path, body, synthetic=False)


def test_production_or_rule_counts_lexical_false_admission():
    row = evidence_row("missing-year", positive=False, score=.2, overlap=1, lexical=True)
    assert probe.admitted(row, .9, .7) == ["right"]
    assert probe.admitted(row, .9, 1.01) == []
    assert probe.admitted(row, .9, None) == []
    metrics = probe.admission_metrics([row], .9, .7, diagnostics=True)
    assert metrics["negative_admissions"] == 1
    assert metrics["feasible"] is False
    assert metrics["queries"][0]["wrong_admitted"] == ["right"]


def test_wrong_source_first_is_not_a_correct_answer_even_if_relevant_is_second():
    row = evidence_row("wrong-first", first="wrong-source")
    row["candidates"].append({"id": "right", "dense_cosine": .75,
                              "lexical_overlap": 0, "lexical": False})
    metrics = probe.admission_metrics([row], .5, None)
    assert metrics["supported_top3"] == 1
    assert metrics["correct_top1"] == 0
    assert metrics["wrong_positive_top1"] == 1
    assert metrics["wrong_positive_hits_top3"] == 1


def test_frozen_calibration_does_not_fit_holdout():
    calibration = [evidence_row("positive", score=.9),
                   evidence_row("negative", positive=False, score=.7)]
    fitted = probe.calibrate(calibration, hybrid=False)
    assert fitted["calibration_feasible"] is True
    cutoff = fitted["min_dense_score"]
    assert .7 < cutoff <= .9
    holdout = [evidence_row("holdout-positive", split="holdout", score=.6),
               evidence_row("holdout-negative", positive=False, split="holdout", score=.95)]
    result = probe.admission_metrics(holdout, cutoff, None)
    assert result["supported_top3"] == 0
    assert result["negative_admissions"] == 1
    assert probe.calibrate(calibration, hybrid=False)["min_dense_score"] == cutoff
    with pytest.raises(probe.ProbeError, match="only calibration"):
        probe.calibrate(holdout, hybrid=False)


def test_calibration_reports_infeasible_inseparable_scores():
    fitted = probe.calibrate([evidence_row("positive", score=.7),
                              evidence_row("negative", positive=False, score=.8)], hybrid=False)
    assert fitted["calibration_feasible"] is False
    assert fitted["calibration"]["negative_admissions"] == 1


def test_calibration_can_disable_lexical_admission_for_exact_but_absent_details():
    fitted = probe.calibrate([
        evidence_row("positive", score=.9, overlap=1, lexical=True),
        evidence_row("negative", positive=False, score=.7, overlap=1, lexical=True),
    ], hybrid=True)
    assert fitted["calibration_feasible"] is True
    assert fitted["min_term_overlap"] == 1.01


def test_matrix_rank_uses_local_corpus_and_does_not_overflow(tmp_path):
    corpora, _ = load_body(tmp_path)
    corpus = corpora[0]
    row = probe.rank_query(corpus, corpus.queries[0], np.asarray([[1., 0.]]),
                           [1e200, 0.], [])
    assert row["dense"] == [corpus.documents[0].id]
    assert row["candidates"][0]["dense_cosine"] == 1
    assert "doc-holdout" not in row["dense"]


class FakeOllama:
    def __init__(self, corpora, *, malformed=False, missing=False):
        self.calls = []
        self.resident = {"unrelated-worker:latest"}
        self.malformed = malformed
        self.missing = missing
        self.text_indices = {}
        for index, corpus in enumerate(corpora):
            for doc in corpus.documents:
                self.text_indices[doc.text] = index
            for query in corpus.queries:
                self.text_indices[query.text] = index if query.relevant else 767

    def handle(self, request):
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.path, body))
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.4"})
        if path == "/api/tags":
            tags = ["nomic-embed-text:latest"]
            if not self.missing:
                tags.append("embeddinggemma-2:latest")
            return httpx.Response(200, json={"models": [
                {"name": tag, "digest": "f" * 64, "size": 100} for tag in tags]})
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["embedding"], "model_info": {}})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": tag} for tag in self.resident]})
        if path == "/api/generate":
            assert body == {"model": body["model"], "keep_alive": 0, "stream": False}
            self.resident.discard(body["model"])
            return httpx.Response(200, json={"done": True})
        assert path == "/api/embed"
        assert body["truncate"] is False
        self.resident = {body["model"]}
        if self.malformed and body["model"] == "embeddinggemma-2:latest":
            return httpx.Response(200, json={"embeddings": [[0.] * 768]})
        vectors = []
        for text in body["input"]:
            for prefix in ("title: none | text: ", "task: search result | query: ",
                           "search_document: ", "search_query: "):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            vector = [0.] * 768
            vector[self.text_indices.get(text, 0)] = 1.
            vectors.append(vector)
        return httpx.Response(200, json={"embeddings": vectors, "model": body["model"],
                                        "load_duration": 1, "total_duration": 2,
                                        "prompt_eval_count": 3})

    def run(self, corpora, **kwargs):
        return asyncio.run(probe.collect_report(
            probe.Settings(rounds=2, batch_size=16), corpora, {},
            transport=httpx.MockTransport(self.handle),
            stress_inputs=["frozen public stress text"] * 64,
            stress_metadata={"documents": 64}, **kwargs))


def test_full_measurement_counts_unique_labels_and_records_cold_and_batch64(tmp_path):
    corpora, _ = load_body(tmp_path)
    fake = FakeOllama(corpora)
    report = fake.run(corpora)
    assert report["measurement_status"] == "completed"
    assert report["embedding_calls"] == report["settings"]["estimated_embedding_requests"]
    assert report["quality_recommendation"]["status"] == "candidate"
    assert report["retrieval"]["qdrant_contacted"] is False
    assert fake.resident == {"nomic-embed-text:latest"}
    for arm in report["results"]:
        assert arm["workloads"]["memory"]["holdout"]["positive_queries"] == 1
        assert arm["workloads"]["memory"]["holdout"]["negative_queries"] == 1
        assert len(arm["cold_queries"]["samples"]) == 3
        assert all(row["verified_unloaded"] for row in arm["cold_queries"]["samples"])
        assert len(arm["ingestion"]["batch64_stress"]) == 2
        assert arm["ingestion"]["batch64_stress"][0]["documents"] == 64
        assert arm["corpora"][0]["unique_queries"] == 2
        assert len(arm["corpora"][0]["queries"]) == 4
        assert arm["corpora"][0]["repeat_ranking_consistent"] is True
    assert "preferred color" not in json.dumps(report)


def test_vector_failure_omits_partial_accuracy_and_unloads_candidate(tmp_path):
    corpora, _ = load_body(tmp_path)
    fake = FakeOllama(corpora, malformed=True)
    report = fake.run(corpora)
    assert report["measurement_status"] == "incomplete"
    assert report["quality_recommendation"]["status"] == "inconclusive"
    failed = report["results"][0]
    assert failed["comparable"] is False
    assert "workloads" not in failed
    assert "corpora" not in failed
    assert "embeddinggemma-2:latest" not in fake.resident
    assert all(arm["comparable"] for arm in report["results"][1:])


def test_all_models_verified_before_any_unload_or_embedding(tmp_path):
    corpora, _ = load_body(tmp_path)
    fake = FakeOllama(corpora, missing=True)
    with pytest.raises(probe.ProbeError, match="not installed"):
        fake.run(corpora)
    assert not any(path in {"/api/embed", "/api/generate"} for path, _ in fake.calls)


def test_budget_excess_fails_before_client_or_network_creation(tmp_path, monkeypatch):
    corpora, _ = load_body(tmp_path)
    many = [probe.Corpus(str(index), corpus.workload, corpus.split,
                         corpus.documents, corpus.queries * 300)
            for index, corpus in enumerate(corpora)]
    def forbidden(*args, **kwargs):
        pytest.fail("client created before request-budget preflight")
    monkeypatch.setattr(probe.base, "Ollama", forbidden)
    with pytest.raises(probe.ProbeError, match="exceed maximum"):
        asyncio.run(probe.collect_report(probe.Settings(rounds=2), many, {}))


def test_quality_findings_are_distinct_from_measurement_failure(tmp_path):
    corpora, _ = load_body(tmp_path)
    report = FakeOllama(corpora).run(corpora)
    assert report["measurement_status"] == "completed"
    gemma = report["results"][0]
    gemma["workloads"]["memory"]["holdout_passed"] = False
    result = probe.recommendation(report)
    assert result["status"] == "inconclusive"
    assert result["workloads"]["memory"]["status"] == "inconclusive"
    assert report["measurement_status"] == "completed"


def test_all_requested_candidates_use_native_dimensions_and_correct_formats():
    tags = ("embeddinggemma-2:latest", "qwen3-embedding:0.6b",
            "qwen3-embedding:4b", "qwen3-embedding:8b")
    settings = probe.Settings(candidates=tags, rounds=2, batch_size=16)
    arms = settings.selected_arms()
    assert [arm.model for arm in arms] == [*tags, "nomic-embed-text:latest", "nomic-embed-text:latest"]
    assert len({arm.id for arm in arms}) == 6
    assert arms[0].expected_dimensions == 768
    for arm in arms[1:4]:
        assert arm.document_prefix == ""
        assert arm.query_prefix.startswith("Instruct:") and arm.query_prefix.endswith("Query: ")
        assert arm.expected_dimensions is None
    assert arms[-1].document_prefix == arms[-1].query_prefix == ""


@pytest.mark.parametrize("settings", [
    probe.Settings(candidates=("qwen3-embedding:cloud",)),
    probe.Settings(candidates=("unknown:latest",)),
    probe.Settings(candidates=("qwen3-embedding:0.6b",) * 2),
    probe.Settings(candidate="qwen3-embedding:4b"),
])
def test_invalid_candidate_profile_fails_without_network(settings):
    with pytest.raises(probe.ProbeError):
        settings.selected_arms()


def test_frozen_public_fixtures_and_full_run_fit_explicit_budget():
    public, pm = probe.load_fixture(ROOT / "evals/cases/embedding_real_documents.json", synthetic=False)
    personal, mm = probe.load_fixture(ROOT / "evals/cases/embedding_personal_cases.json", synthetic=True)
    regression, rm = probe.load_regression(ROOT / "evals/cases/embedding_retrieval_cases.json")
    corpora = [*public, *personal, regression]
    assert pm["documents"] == 336 and pm["queries"] == 80
    assert mm["documents"] == 48 and mm["queries"] == 48
    assert rm["documents"] == 32 and rm["queries"] == 28
    settings = probe.Settings(candidates=("embeddinggemma-2:latest", "qwen3-embedding:0.6b",
                                        "qwen3-embedding:4b", "qwen3-embedding:8b"),
                              rounds=2, batch_size=16)
    assert probe.call_budget(settings, corpora) == 2070
    stress, metadata = probe.load_stress(ROOT / "evals/cases/embedding_real_documents.json")
    assert len(stress) == 64 and 1000 <= metadata["max_tokens"] <= 1010
    assert metadata["min_tokens"] >= 900


def test_public_text_checksum_tampering_is_rejected(tmp_path):
    body = fixture_body(synthetic=False)
    body["documents"][0]["text_sha256"] = "0" * 64
    with pytest.raises(probe.ProbeError, match="text checksum"):
        load_body(tmp_path, body, synthetic=False)


def test_source_aliases_do_not_evade_split_guard(tmp_path):
    body = fixture_body(synthetic=False)
    body["sources"][1]["url"] = body["sources"][0]["url"]
    with pytest.raises(probe.ProbeError, match="source leakage"):
        load_body(tmp_path, body, synthetic=False)


def test_stress_checksum_tampering_fails_before_model_calls(tmp_path):
    body = json.loads((ROOT / "evals/cases/embedding_real_documents.json").read_text())
    body["stress"]["texts"][0] = "changed text"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(body))
    with pytest.raises(probe.ProbeError, match="batch64 inputs"):
        probe.load_stress(path)


def test_wrong_personal_workload_cannot_replace_regression_gate(tmp_path):
    corpora, _ = load_body(tmp_path)
    report = FakeOllama(corpora).run(corpora)
    for arm in report["results"]:
        ranking = {method: {"mrr": .5 if arm["arm"] == "candidate" else 1,
                             "hit_rate_at_3": 1} for method in ("dense", "hybrid")}
        arm["corpora"].append({"workload": "regression", "ranking": ranking})
    decision = probe.recommendation(report)
    assert decision["regression_nonregression"] is False
    assert decision["status"] == "inconclusive"


def test_all_six_arms_complete_with_different_native_vector_widths(tmp_path):
    corpora, _ = load_body(tmp_path)
    fake = FakeOllama(corpora)
    qwen = {"qwen3-embedding:0.6b": 1024,
            "qwen3-embedding:4b": 2560, "qwen3-embedding:8b": 4096}
    tags = ("embeddinggemma-2:latest", *qwen)
    settings = probe.Settings(candidates=tags, rounds=2, batch_size=16)
    query_prefix = settings.selected_arms()[1].query_prefix

    def handler(request):
        response = None
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/api/embed" and body["model"] in qwen:
            forwarded = dict(body, input=[text.removeprefix(query_prefix) for text in body["input"]])
            request = httpx.Request(request.method, request.url, json=forwarded)
            response = fake.handle(request)
            content = response.json()
            width = qwen[body["model"]]
            content["embeddings"] = [vector + [0.] * (width - len(vector))
                                     for vector in content["embeddings"]]
            return httpx.Response(200, json=content)
        response = fake.handle(request)
        if request.url.path == "/api/tags":
            content = response.json()
            content["models"].extend({"name": tag, "digest": "a" * 64, "size": 123} for tag in qwen)
            return httpx.Response(200, json=content)
        return response

    report = asyncio.run(probe.collect_report(
        settings, corpora, {}, transport=httpx.MockTransport(handler),
        stress_inputs=["frozen public text"] * 64, stress_metadata={"documents": 64}))
    assert report["errors"] == []
    assert report["measurement_status"] == "completed"
    assert [row["dimensions"] for row in report["results"]] == [768, 1024, 2560, 4096, 768, 768]
    assert report["embedding_calls"] == report["settings"]["estimated_embedding_requests"]
    assert len(report["quality_recommendation"]["candidates"]) == 4
    assert fake.resident == {"nomic-embed-text:latest"}
    assert all(path.startswith("/api/") for path, _ in fake.calls)


def test_check_mode_validates_actual_inputs_without_network(monkeypatch, capsys):
    monkeypatch.setenv("CANDIDATES", "embeddinggemma-2:latest,qwen3-embedding:0.6b,qwen3-embedding:4b,qwen3-embedding:8b")
    def forbidden(*args, **kwargs):
        pytest.fail("check mode must not create a client")
    monkeypatch.setattr(probe.base, "Ollama", forbidden)
    assert probe.main(["--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["estimated_embedding_requests"] == 2070
    assert report["model_calls"] == 0
    assert report["documents"] == 416 and report["unique_queries"] == 156


def test_public_hsa_labels_exclude_identical_msa_wording():
    body = json.loads((ROOT / "evals/cases/embedding_real_documents.json").read_text())
    for suffix, page in (("-46", 3), ("-52", 8)):
        query = next(q for q in body["queries"] if q["id"].endswith(suffix))
        assert {e["page"] for e in query["evidence"]} == {page}
        docs = {d["id"]: d for d in body["documents"]}
        assert {docs[identity]["page"] for identity in query["relevant"]} == {page}


def test_real_reference_and_kb_pool_depths_are_explicit():
    assert probe.RETRIEVAL_POOLS["reference"] == 24
    assert probe.RETRIEVAL_POOLS["regression"] == 40
    assert probe.base.RETRIEVAL_POOL == 20
