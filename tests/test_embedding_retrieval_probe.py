"""Offline contracts for the isolated, read-only embedding candidate probe."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "embedding_retrieval_probe", ROOT / "scripts/probes/embedding_retrieval_probe.py"
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)
FIXTURE = ROOT / "evals/cases/embedding_retrieval_cases.json"


def test_fixture_is_balanced_public_english_with_difficult_distractors():
    documents, queries, digest = probe.load_cases(FIXTURE)
    assert len(documents) == 32
    assert len(queries) == 28
    assert len(digest) == 64
    assert all(doc.text.isascii() for doc in documents)
    for group in probe.GROUPS:
        assert sum(query.group == group and bool(query.relevant) for query in queries) == 6
        assert sum(query.group == group and not query.relevant for query in queries) == 1
    assert {"x-aster-2024", "x-harbor-with", "x-refund-draft"} <= {doc.id for doc in documents}
    assert all("SYNTHETIC REFERENCE" in doc.text for doc in documents if doc.id.startswith("r-"))


@pytest.mark.parametrize("change,match", [
    (lambda body: body["documents"].append(body["documents"][0]), "duplicate id"),
    (lambda body: body["queries"][0].update(relevant=["missing"]), "missing document"),
    (lambda body: body["queries"][0].update(answerable=False), "answerable"),
    (lambda body: body["queries"][0].update(group="medical"), "known task group"),
    (lambda body: body["documents"][0].update(text="bonjour\u00e9"), "English-only"),
    (lambda body: body.update(language="ja"), "English"),
    (lambda body: body.update(synthetic=False), "synthetic"),
    (lambda body: body["queries"][0].update(relevant=["g-starter", "g-starter"]), "unique"),
])
def test_invalid_or_misleading_fixtures_refused(tmp_path, change, match):
    body = json.loads(FIXTURE.read_text())
    change(body)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(body))
    with pytest.raises(probe.ProbeError, match=match):
        probe.load_cases(path)


def test_metrics_distinguish_recall_from_hit_rate_and_count_missing_positive():
    metrics = probe.ranking_metrics(
        [["a", "c", "b"], ["wrong", "right"], ["miss"]],
        [("a", "b"), ("right",), ("absent",)],
    )
    assert metrics["mrr"] == 0.5
    assert metrics["recall_at_1"] == pytest.approx(1 / 6, abs=0.000001)
    assert metrics["hit_rate_at_1"] == pytest.approx(1 / 3, abs=0.000001)
    assert metrics["recall_at_3"] == pytest.approx(2 / 3, abs=0.000001)
    assert probe.percentile([1, 2, 3, 4], 0.95) == 4
    assert probe.percentile([], 0.5) is None


@pytest.mark.parametrize("rows,count,dimensions,match", [
    ([], 1, 2, "count mismatch"),
    ([[1.0]], 1, 2, "dimensions"),
    ([[0.0, 0.0]], 1, 2, "zero"),
    ([[math.nan, 1.0]], 1, 2, "non-finite"),
    ([[math.inf, 1.0]], 1, 2, "non-finite"),
    ([[True, 1.0]], 1, 2, "nonnumeric"),
    ([["1", 1.0]], 1, 2, "nonnumeric"),
    ([[1.0, 0.0], [1.0]], 2, None, "dimensions"),
])
def test_embedding_contract_rejects_silent_shape_or_numeric_corruption(
    rows, count, dimensions, match,
):
    with pytest.raises(probe.ProbeError, match=match):
        probe.validate_vectors({"embeddings": rows}, count, dimensions)


def test_sparse_ranking_uses_existing_helpers_and_returns_no_zero_score_hits():
    docs = [probe.Document("a", "rarekitten " * 4), probe.Document("b", "unrelated trees")]
    assert [hit.source for hit in probe.lexical_rank(docs, "rarekitten")] == ["a"]
    assert probe.lexical_rank(docs, "missingword") == []


def test_cosine_does_not_overflow_for_finite_nonunit_vectors():
    assert probe.cosine([1e200, 0], [1e200, 0]) == 1


def test_unanswerable_queries_are_diagnostics_not_fabricated_positive_labels():
    rows = [
        {"dense": ["yes"], "relevant": ["yes"], "group": "general"},
        {"dense": ["wrong"], "relevant": [], "group": "reference"},
    ]
    summary = probe.summarize_rows(rows, "dense")
    assert summary["overall"]["queries"] == 1
    assert summary["overall"]["mrr"] == 1
    assert summary["groups"]["reference"]["mrr"] is None


class FakeOllama:
    def __init__(self, *, missing=False, candidate_error=False, malformed=False):
        self.documents, self.queries, self.digest = probe.load_cases(FIXTURE)
        self.calls = []
        self.residents = {"unrelated-generation:32b"}
        self.missing = missing
        self.candidate_error = candidate_error
        self.malformed = malformed

    def handle(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body))
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.4"})
        if path == "/api/tags":
            models = ["nomic-embed-text:latest"]
            if not self.missing:
                models.append("embeddinggemma-2:latest")
            return httpx.Response(200, json={"models": [
                {"name": tag, "digest": "f" * 64, "size": 123, "details": {"family": "test"}}
                for tag in models
            ]})
        if path == "/api/show":
            return httpx.Response(200, json={
                "capabilities": ["embedding"],
                "model_info": {"general.architecture": "test", "test.context_length": 8192,
                               "test.embedding_length": 768},
            })
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": tag} for tag in self.residents]})
        if path == "/api/generate":
            assert body["keep_alive"] == 0
            assert "prompt" not in body
            self.residents.discard(body["model"])
            return httpx.Response(200, json={"done": True})
        assert path == "/api/embed"
        assert body["truncate"] is False
        assert body["keep_alive"] == "10m"
        assert "dimensions" not in body  # Measure native output, no silent reduction.
        self.residents = {body["model"]}
        if self.candidate_error and body["model"] == "embeddinggemma-2:latest":
            return httpx.Response(400, json={"error": "runner cannot load model"})
        if self.malformed:
            return httpx.Response(200, json={"embeddings": [[0.0] * 768]})
        vectors = []
        for text in body["input"]:
            plain = text
            for prefix in ("title: none | text: ", "task: search result | query: ",
                           "search_document: ", "search_query: "):
                if plain.startswith(prefix):
                    plain = plain[len(prefix):]
                    break
            matching_docs = [index for index, doc in enumerate(self.documents) if doc.text == plain]
            if matching_docs:
                index = matching_docs[0]
            else:
                query = next(query for query in self.queries if query.text == plain)
                index = next((i for i, doc in enumerate(self.documents)
                              if doc.id in query.relevant), 767)
            vector = [0.0] * 768
            vector[index] = 1.0
            vectors.append(vector)
        return httpx.Response(200, json={"model": body["model"], "embeddings": vectors,
                                       "load_duration": 5, "total_duration": 10,
                                       "prompt_eval_count": 20})

    def run(self):
        return asyncio.run(probe.collect_report(
            probe.Settings(), self.documents, self.queries, self.digest,
            transport=httpx.MockTransport(self.handle),
        ))


def test_full_collection_is_sequential_same_content_three_arms_and_retains_incumbent():
    fake = FakeOllama()
    report = fake.run()
    assert report["measurement_status"] == "completed"
    assert report["fixture"]["sha256"] == fake.digest
    assert report["embedding_calls"] == 99
    assert report["settings"] == {
        "timeout_seconds": 120, "keep_alive": "10m", "batch_size": 8, "rounds": 1,
        "selected_arms": ["gemma", "prefixed", "raw"],
        "estimated_embedding_requests": 99, "maximum_embedding_requests": 512,
    }
    assert report["provenance"]["ollama_version"] == "0.40.4"
    assert report["provenance"]["installed_models"]["embeddinggemma-2:latest"]["digest"] == "f" * 64
    assert [arm["arm"] for arm in report["results"]] == ["gemma", "prefixed", "raw"]
    assert report["residency_after"] == [{"name": "nomic-embed-text:latest"}]
    for arm in report["results"]:
        assert arm["dimensions"] == 768
        assert arm["vector_norms"] == {"min": 1, "max": 1, "mean": 1}
        assert arm["dense"]["overall"]["mrr"] == 1
        assert arm["dense"]["overall"]["queries"] == 24
        assert arm["unanswerable_diagnostics"]["queries"] == 4
        assert arm["unanswerable_diagnostics"]["rejection_accuracy_measured"] is False
        assert arm["warm_single_query"]["samples"] == 28
        assert arm["cold_single_query"]["controlled_unload"] is True
        assert arm["ingestion"]["documents"] == 32
    unloads = [body["model"] for _, path, body in fake.calls if path == "/api/generate"]
    assert unloads == ["embeddinggemma-2:latest", "embeddinggemma-2:latest",
                       "nomic-embed-text:latest", "nomic-embed-text:latest"]
    assert "unrelated-generation:32b" in report["results"][0]["residency"]["displaced"]
    assert all(path.startswith("/api/") for _, path, _ in fake.calls)
    assert "queries" not in probe.compact_report(report)["results"][0]


def test_missing_installation_fails_before_unloading_or_embedding():
    fake = FakeOllama(missing=True)
    with pytest.raises(probe.ProbeError, match="not installed"):
        fake.run()
    assert not any(path in {"/api/generate", "/api/embed"} for _, path, _ in fake.calls)


def test_candidate_load_error_is_findings_and_does_not_drop_incumbent_evidence():
    fake = FakeOllama(candidate_error=True)
    report = fake.run()
    assert report["status"] == "failed"
    assert report["measurement_status"] == "incomplete"
    assert report["errors"][0]["error"] == "/api/embed: HTTP 400"
    assert report["results"][0]["status"] == "incomplete"
    assert report["results"][0]["stage"] == "cold_single_query"
    assert report["results"][0]["embedding_requests_attempted"] == 1
    assert report["results"][0]["completed_query_samples"] == 0
    assert report["results"][0]["comparable"] is False
    assert report["results"][-1]["dense"]["overall"]["mrr"] == 1
    assert report["residency_after"] == [{"name": "nomic-embed-text:latest"}]


def test_malformed_vectors_cannot_publish_comparative_quality_metrics():
    fake = FakeOllama(malformed=True)
    report = fake.run()
    assert report["status"] == "failed"
    assert len(report["errors"]) == 3
    assert all(result["status"] == "incomplete" for result in report["results"])
    assert all("dense" not in result for result in report["results"])


@pytest.mark.parametrize("settings", [
    probe.Settings(rounds=0), probe.Settings(batch_size=99),
    probe.Settings(arms=("unknown",)), probe.Settings(arms=("raw", "raw")),
    probe.Settings(timeout=math.inf), probe.Settings(base_url="file:///tmp/local"),
    probe.Settings(candidate="remote:cloud"),
])
def test_settings_fail_before_calls(settings):
    with pytest.raises(probe.ProbeError):
        settings.selected_arms()


def test_candidate_tag_is_configurable_and_arm_order_is_not():
    arms = probe.Settings(candidate="embeddinggemma-2:270m", arms=("raw", "gemma")).selected_arms()
    assert [arm.id for arm in arms] == ["gemma", "raw"]
    assert arms[0].model == "embeddinggemma-2:270m"


def test_final_residency_failure_preserves_measurements_as_incomplete():
    fake = FakeOllama()
    original = fake.handle
    ps_calls = 0

    def handler(request):
        nonlocal ps_calls
        if request.url.path == "/api/ps":
            ps_calls += 1
            if ps_calls == 11:
                return httpx.Response(503, json={"error": "busy"})
        return original(request)

    report = asyncio.run(probe.collect_report(
        probe.Settings(), fake.documents, fake.queries, fake.digest,
        transport=httpx.MockTransport(handler),
    ))
    assert report["measurement_status"] == "incomplete"
    assert len(report["results"]) == 3
    assert report["results"][-1]["dense"]["overall"]["mrr"] == 1
    assert report["residency_after"] is None


def test_network_errors_do_not_echo_urls_or_credentials():
    def handler(request):
        raise httpx.ConnectError("secret@private.example", request=request)

    async def run():
        client = probe.Ollama(probe.Settings(), httpx.MockTransport(handler))
        try:
            with pytest.raises(probe.ProbeError) as error:
                await client.request("GET", "/api/version")
            return str(error.value)
        finally:
            await client.client.aclose()

    assert asyncio.run(run()) == "/api/version: ConnectError"


def test_excessive_direct_fixture_is_rejected_without_touching_models():
    with pytest.raises(probe.ProbeError, match="bounded"):
        asyncio.run(probe.collect_report(
            probe.Settings(), [probe.Document(str(i), "text") for i in range(65)],
            [probe.Query("q", "query", "general", ("1",))], "digest",
            transport=httpx.MockTransport(lambda request: pytest.fail("unexpected request")),
        ))
