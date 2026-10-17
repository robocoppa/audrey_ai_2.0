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
    (lambda body: body["documents"][0].update(workload="documents"), "invalid workload"),
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


def test_baseline_only_resume_never_calls_or_requires_a_candidate(tmp_path):
    corpora, _ = load_body(tmp_path)
    fake = FakeOllama(corpora, missing=True)
    settings = probe.Settings(rounds=2, batch_size=16, baselines_only=True)
    assert [arm.id for arm in settings.selected_arms()] == ["prefixed", "raw"]
    report = asyncio.run(probe.collect_report(
        settings, corpora, {}, transport=httpx.MockTransport(fake.handle),
        stress_inputs=["frozen public text"] * 64, stress_metadata={"documents": 64}))
    assert report["measurement_status"] == "completed"
    assert report["quality_recommendation"]["status"] == "baselines_measured"
    assert report["settings"]["candidate_models"] == []
    assert all(body.get("model", "nomic-embed-text:latest") == "nomic-embed-text:latest"
               for _, body in fake.calls)


def test_baseline_only_check_has_only_missing_requests(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not contact a model")
    monkeypatch.setattr(probe.base, "Ollama", forbidden)
    assert probe.main(["--check", "--baselines-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["arms"] == ["nomic-embed-text:latest", "nomic-embed-text:latest"]
    assert report["estimated_embedding_requests"] == 690
    assert report["unique_queries"] == 156


def test_input_digest_changes_for_model_text_but_not_label_corrections():
    body = fixture_body(synthetic=False)
    before = probe.fixture_inputs_sha256(body)
    body["queries"][0]["relevant"] = []
    body["queries"][0]["evidence"] = []
    assert probe.fixture_inputs_sha256(body) == before
    body["queries"][0]["text"] += " changed input"
    assert probe.fixture_inputs_sha256(body) != before


def document_mode_profiles():
    return probe.document_config(ROOT / "config.yaml")


def public_and_upload_corpora(tmp_path):
    public = fixture_body(synthetic=False)
    upload = json.loads(json.dumps(public))
    upload["workload"] = "documents"
    upload["chunk_policy"] = {"tokens": 1000, "overlap_tokens": 100}
    upload["label_policy"] = "Existing reviewed questions; not new independent labels"
    upload["derivation"] = {"questions_reused": True}
    for source in upload["sources"]:
        source["pages"] = 2
    for doc in upload["documents"]:
        doc["end_page"] = 2
        doc["end_offset"] = len(doc["text"])
    for query in upload["queries"]:
        for evidence in query["evidence"]:
            evidence["page"] = 2
    first = tmp_path / "reference.json"
    second = tmp_path / "upload.json"
    first.write_text(json.dumps(public))
    second.write_text(json.dumps(upload))
    reference, reference_meta = probe.load_fixture(first, synthetic=False)
    documents, document_meta = probe.load_fixture(second, synthetic=False)
    return [*reference, *documents], {"public": reference_meta, "upload": document_meta}, first, second


class DocumentModeOllama(FakeOllama):
    """Actual configured profiles/native widths, with no production connections."""
    def __init__(self, corpora, *, wrong_digest=False, wrong_width=False):
        super().__init__(corpora)
        self.profiles, _ = document_mode_profiles()
        self.resident.update(profile.model for profile in self.profiles.values())
        self.wire_calls = []
        self.wrong_digest = wrong_digest
        self.wrong_width = wrong_width

    def handle(self, request):
        body = json.loads(request.content) if request.content else {}
        self.wire_calls.append((request.url.path, body))
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": profile.model, "digest": "f" * 64 if self.wrong_digest else profile.digest, "size": 100}
                for profile in self.profiles.values()
            ]})
        if request.url.path == "/api/embed":
            profile = next(profile for profile in self.profiles.values() if profile.model == body["model"])
            before = set(self.resident)
            inputs = [value.removeprefix(profile.query_prefix) for value in body["input"]]
            response = super().handle(httpx.Request(request.method, request.url, json={**body, "input": inputs}))
            self.resident.update(before)
            value = response.json()
            width = 768 if self.wrong_width else profile.dimensions
            value["embeddings"] = [vector + [0.] * (width - len(vector)) for vector in value["embeddings"]]
            return httpx.Response(200, json=value)
        return super().handle(request)

    def compare(self, corpora, metadata):
        profiles, policies = document_mode_profiles()
        settings = probe.Settings(
            candidate=profiles["text"].model, candidate_profile="qwen3",
            incumbent=profiles["history"].model, documents_only=True,
            rounds=1, batch_size=16,
        )
        return asyncio.run(probe.collect_report(
            settings, corpora, metadata, profiles=profiles, production_policies=policies,
            transport=httpx.MockTransport(self.handle),
        ))


def test_document_mode_compares_layouts_without_unloads_stress_or_private_fixtures(tmp_path):
    corpora, metadata, _, _ = public_and_upload_corpora(tmp_path)
    fake = DocumentModeOllama(corpora)
    report = fake.compare(corpora, metadata)
    assert report["measurement_status"] == "completed"
    assert report["errors"] == []
    assert report["embedding_calls"] == report["settings"]["estimated_embedding_requests"] == 38
    assert report["settings"]["cold_samples"] == report["settings"]["stress_batches64"] == 0
    assert not any(path == "/api/generate" for path, _ in fake.wire_calls)
    assert all(len(body["input"]) != 64 for path, body in fake.wire_calls if path == "/api/embed")
    assert report["restoration"] == {
        "warm_requests_completed": ["qwen3-embedding:4b", "nomic-embed-text:latest"],
        "keep_alive": "24h", "explicit_unloads": 0,
        "resident_models_after": ["nomic-embed-text:latest", "qwen3-embedding:4b", "unrelated-worker:latest"],
        "both_configured_observed": True,
    }
    assert {"qwen3-embedding:4b", "nomic-embed-text:latest"} <= fake.resident
    embeds = [body for path, body in fake.wire_calls if path == "/api/embed"]
    assert {body["model"] for body in embeds[-2:]} == {"qwen3-embedding:4b", "nomic-embed-text:latest"}
    assert all(body["keep_alive"] == "24h" for body in embeds[-2:])
    assert [arm["dimensions"] for arm in report["results"]] == [2560, 768, 768]
    for arm in report["results"]:
        assert len(arm["corpora"]) == 4
        assert arm["cold_queries"]["status"] == "not_requested"
        assert arm["ingestion"]["batch64_stress"] == []
        for corpus in arm["corpora"]:
            assert corpus["ranking"]["dense"]["hits_at_5"] == 1
            assert corpus["ranking"]["hybrid"]["mrr"] == 1
            assert "current_policy" not in corpus
            policy = corpus["configured_qwen_policy_diagnostic"]
            assert policy["not_a_cross_model_calibration"]
            assert (policy["min_dense_score"], policy["min_term_overlap"]) == (
                (.54, None) if corpus["workload"] == "reference" else (.53, .7)
            )
    assert report["quality_recommendation"]["automatic_promotion"] is False


def test_document_mode_checks_pinned_digest_before_any_embedding(tmp_path):
    corpora, metadata, _, _ = public_and_upload_corpora(tmp_path)
    fake = DocumentModeOllama(corpora, wrong_digest=True)
    with pytest.raises(probe.ProbeError, match="pinned digest"):
        fake.compare(corpora, metadata)
    assert not any(path == "/api/embed" for path, _ in fake.wire_calls)


def test_document_mode_rejects_wrong_native_vector_width(tmp_path):
    corpora, metadata, _, _ = public_and_upload_corpora(tmp_path)
    report = DocumentModeOllama(corpora, wrong_width=True).compare(corpora, metadata)
    assert report["measurement_status"] == "incomplete"
    assert report["results"][0]["status"] == "incomplete"
    assert report["results"][0]["comparable"] is False
    assert report["quality_recommendation"]["automatic_promotion"] is False


def test_document_summary_contains_paired_metrics_without_nested_query_dumps(tmp_path):
    corpora, metadata, _, _ = public_and_upload_corpora(tmp_path)
    report = DocumentModeOllama(corpora).compare(corpora, metadata)
    compact = probe.compact_report(report)
    assert compact["fixtures"]["upload"]["layout"]["chunk_policy"] == {"tokens": 1000, "overlap_tokens": 100}
    assert compact["fixtures"]["upload"]["layout"]["derivation"]["questions_reused"]
    assert len(compact["paired_holdout"]) == 8
    assert all(pair["positive_queries"] == 1 and pair["rank_wins"] == pair["rank_losses"] == 0
               for pair in compact["paired_holdout"])
    for arm in compact["results"]:
        assert "corpora" not in arm and "workloads" not in arm
        assert len(arm["holdout_layouts"]) == 2
        for layout in arm["holdout_layouts"]:
            assert "queries" not in layout["own_calibrated_policy"]
            assert "queries" not in layout["configured_qwen_policy_diagnostic"]
            assert layout["ranking_before_admission"]["dense"]["hits_at_5"] == 1


def test_document_pairing_counts_wins_losses_at_each_depth():
    def arm(identity, ranking):
        row = {"id": "same-question", "repeat": 1, "relevant": ["answer"],
               "dense": ranking, "hybrid": ranking}
        return {"arm": identity, "comparable": True, "corpora": [{
            "id": "reference:holdout", "workload": "reference", "split": "holdout", "queries": [row],
        }]}
    pairs = probe.paired_document_results([
        arm("candidate", ["answer", "other"]),
        arm("raw", ["other", "answer"]), arm("prefixed", ["answer", "other"]),
    ])
    raw = next(pair for pair in pairs if pair["baseline"] == "raw" and pair["method"] == "dense")
    assert raw["rank_wins"] == 1 and raw["rank_losses"] == 0
    assert raw["at_depth"]["1"] == {"wins": 1, "losses": 0, "ties": 0}
    assert raw["at_depth"]["3"] == {"wins": 0, "losses": 0, "ties": 1}
    assert raw["rank_changes"] == [{"id": "same-question", "qwen_rank": 1, "baseline_rank": 2}]


def test_document_page_spans_validate_evidence_and_source_bounds(tmp_path):
    corpora, _metadata, _, upload_path = public_and_upload_corpora(tmp_path)
    assert next(c for c in corpora if c.workload == "documents").documents[0].end_page == 2
    body = json.loads(upload_path.read_text())
    body["queries"][0]["evidence"][0]["page"] = 3
    upload_path.write_text(json.dumps(body))
    with pytest.raises(probe.ProbeError, match="quote"):
        probe.load_fixture(upload_path, synthetic=False)
    body["documents"][0]["end_page"] = 0
    upload_path.write_text(json.dumps(body))
    with pytest.raises(probe.ProbeError, match="end page"):
        probe.load_fixture(upload_path, synthetic=False)


def test_document_only_check_uses_real_layouts_and_profiles_without_network(monkeypatch, capsys):
    monkeypatch.delenv("CANDIDATES", raising=False)
    monkeypatch.delenv("CANDIDATE", raising=False)
    monkeypatch.delenv("CANDIDATE_PROFILE", raising=False)
    monkeypatch.delenv("ROUNDS", raising=False)
    monkeypatch.setattr(probe.base, "Ollama", lambda *_args, **_kwargs: pytest.fail("offline check contacted models"))
    assert probe.main([
        "--documents-only", "--check", "--config", str(ROOT / "config.yaml"),
        "--upload-cases", str(ROOT / "evals/cases/embedding_upload_documents.json"),
        "--personal-cases", "/not-needed/personal.json", "--regression-cases", "/not-needed/regression.json",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_calls"] == 0 and report["documents_only"] is True
    assert report["documents"] == 504 and report["unique_queries"] == 80
    assert report["query_layout_cases"] == 160
    assert report["estimated_embedding_requests"] == 581
    assert report["stress"] is None
    assert set(report["fixtures"]) == {"public", "upload"}


def test_document_mode_refuses_model_override_outside_config_before_network(monkeypatch, capsys):
    monkeypatch.setenv("CANDIDATES", "qwen3-embedding:8b")
    monkeypatch.setattr(probe.base, "Ollama", lambda *_args, **_kwargs: pytest.fail("override should fail before models"))
    assert probe.main(["--documents-only", "--check", "--config", str(ROOT / "config.yaml")]) == 2
    assert "configured embedding profile" in json.loads(capsys.readouterr().out)["error"]


def test_document_mode_preserves_quality_findings_without_automatic_promotion(tmp_path):
    corpora, metadata, _, _ = public_and_upload_corpora(tmp_path)
    fake = DocumentModeOllama(corpora)
    for index, corpus in enumerate(corpora):
        for query in corpus.queries:
            fake.text_indices[query.text] = index
    report = fake.compare(corpora, metadata)
    assert report["measurement_status"] == "completed"
    assert report["quality_recommendation"]["quality_findings"] is True
    assert report["quality_recommendation"]["automatic_promotion"] is False


@pytest.mark.parametrize("malformed_yaml", [False, True])
def test_document_configuration_errors_return_setup_json_before_models(tmp_path, monkeypatch, capsys, malformed_yaml):
    path = tmp_path / "bad-config.yaml"
    if malformed_yaml:
        path.write_text("text_embedding: [invalid\n")
    else:
        body = probe.yaml.safe_load((ROOT / "config.yaml").read_text())
        del body["kb"]
        path.write_text(probe.yaml.safe_dump(body))
    monkeypatch.setattr(probe.base, "Ollama", lambda *_args, **_kwargs: pytest.fail("invalid config contacted models"))
    assert probe.main(["--documents-only", "--check", "--config", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["measurement_status"] == "setup_failed"


def test_document_restoration_reports_eviction_instead_of_claiming_residency(tmp_path):
    corpora, metadata, _, _ = public_and_upload_corpora(tmp_path)

    class SingleResident(DocumentModeOllama):
        def handle(self, request):
            response = super().handle(request)
            if request.url.path == "/api/embed":
                self.resident = {json.loads(request.content)["model"]}
            return response

    report = SingleResident(corpora).compare(corpora, metadata)
    assert report["measurement_status"] == "completed"
    assert len(report["restoration"]["warm_requests_completed"]) == 2
    assert report["restoration"]["both_configured_observed"] is False
    assert probe.compact_report(report)["restoration"]["resident_models_after"] == ["nomic-embed-text:latest"]


def test_completed_document_quality_findings_exit_one_without_claiming_measurement_failure(tmp_path, monkeypatch, capsys):
    corpora, metadata, public_path, upload_path = public_and_upload_corpora(tmp_path)
    fake = DocumentModeOllama(corpora)
    for index, corpus in enumerate(corpora):
        for query in corpus.queries:
            fake.text_indices[query.text] = index
    report = fake.compare(corpora, metadata)

    async def measured(*_args, **_kwargs):
        return report

    monkeypatch.setattr(probe, "collect_report", measured)
    assert probe.main([
        "--documents-only", "--summary", "--config", str(ROOT / "config.yaml"),
        "--public-cases", str(public_path), "--upload-cases", str(upload_path),
    ]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["measurement_status"] == "completed"
    assert output["quality_recommendation"]["quality_findings"] is True
    assert output["quality_recommendation"]["automatic_promotion"] is False


def test_document_fixture_summary_preserves_label_revision_limits_without_transcripts():
    fixtures = {}
    for name, filename in (("public", "embedding_real_documents.json"), ("upload", "embedding_upload_documents.json")):
        _, fixtures[name] = probe.load_fixture(ROOT / "evals/cases" / filename, synthetic=False)
    compact = probe.compact_fixture_metadata(fixtures)
    revision = compact["public"]["layout"]["label_policy"]["revision"]
    assert revision["added_passages"] == 33 and revision["affected_queries"] == 27
    assert revision["retrospective"] is True
    assert revision["model_inputs_changed"] is False
    assert revision["thresholds_tuned_on_holdout"] is False
    assert revision["limits"] and "additions" not in revision
    assert compact["upload"]["layout"]["label_policy"]["original_reviewed_policy"] == "Same reviewed policy as public fixture above"
    assert "additions" in fixtures["public"]["layout"]["label_policy"]["revision"]


def test_document_check_survives_actual_shallow_onbox_script_location(monkeypatch, capsys):
    monkeypatch.setattr(probe, "__file__", "/tmp/probe-DOCUMENTS-STAMP.py")  # noqa: S108 - no file is created
    monkeypatch.setattr(probe.base, "Ollama", lambda *_args, **_kwargs: pytest.fail("offline copied check contacted models"))
    assert probe.main([
        "--documents-only", "--check", "--config", str(ROOT / "config.yaml"),
        "--public-cases", str(ROOT / "evals/cases/embedding_real_documents.json"),
        "--upload-cases", str(ROOT / "evals/cases/embedding_upload_documents.json"),
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model_calls"] == 0 and output["unique_queries"] == 80
    assert output["query_layout_cases"] == 160 and output["estimated_embedding_requests"] == 581
