"""Asynchronous model unload must not discard otherwise valid retrieval evidence."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "embedding_document_unload_probe", os.environ.get(
        "EMBEDDING_DOCUMENT_PROBE_PATH", ROOT / "scripts/probes/embedding_document_probe.py")
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class LifecycleClient:
    def __init__(self, *, polls_until_unloaded=1, successful_unloads=None):
        self.model = "nomic-embed-text:latest"
        self.resident = True
        self.polls_until_unloaded = polls_until_unloaded
        self.successful_unloads = successful_unloads
        self.unload_calls = 0
        self.residency_calls = 0
        self.after_unload_polls = 0
        self.embed_calls = 0
        self.settings = SimpleNamespace(batch_size=16, rounds=1)

    async def unload(self, model):
        assert model == self.model
        self.unload_calls += 1
        self.after_unload_polls = 0

    async def residency(self):
        self.residency_calls += 1
        if self.unload_calls:
            self.after_unload_polls += 1
            if ((self.successful_unloads is None or self.unload_calls <= self.successful_unloads)
                    and self.after_unload_polls >= self.polls_until_unloaded):
                self.resident = False
        return [{"name": self.model}] if self.resident else []

    async def embed(self, arm, texts):
        self.embed_calls += 1
        self.resident = True
        self.after_unload_polls = -1000  # No later disappearance without another unload.
        return [[0., 1.] if "balance" in text else [1., 0.] for text in texts], {
            "wall_seconds": .01}


@pytest.fixture
def no_poll_delay(monkeypatch):
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(probe.asyncio, "sleep", sleep)
    return sleeps


def test_verified_unload_waits_for_asynchronous_residency_removal(no_poll_delay):
    client = LifecycleClient(polls_until_unloaded=3)
    result = asyncio.run(probe.verify_unload(client, client.model))
    assert result == {"verified_unloaded": True, "after_unload": [], "polls": 3}
    assert client.unload_calls == 1
    assert no_poll_delay == [.25, .25]
    assert client.embed_calls == 0


def test_resident_unload_stops_polling_at_the_fixed_bound(no_poll_delay):
    client = LifecycleClient(successful_unloads=0)
    result = asyncio.run(probe.verify_unload(client, client.model))
    assert result["verified_unloaded"] is False
    assert result["polls"] == 21
    assert "resident" in result["reason"]
    assert len(no_poll_delay) == 20
    assert client.embed_calls == 0


def test_unresponsive_unload_is_bounded_by_timeout():
    class HangingClient:
        async def unload(self, model):
            await asyncio.Future()

    result = asyncio.run(probe.verify_unload(HangingClient(), "model", timeout_seconds=.001))
    assert result["verified_unloaded"] is False
    assert result["polls"] == 0
    assert "bounded wait" in result["reason"]


def test_unload_transport_error_is_reported_as_missing_cold_evidence():
    class BrokenClient:
        async def unload(self, model):
            raise probe.ProbeError("/api/generate: HTTP 500")

    result = asyncio.run(probe.verify_unload(BrokenClient(), "model"))
    assert result["verified_unloaded"] is False
    assert result["polls"] == 0
    assert "/api/generate: HTTP 500" in result["reason"]


@pytest.mark.parametrize("verified_cold,status", [(0, "unavailable"), (1, "partial")])
def test_failed_cold_verification_preserves_warm_ranking_and_batches(
        verified_cold, status, no_poll_delay):
    client = LifecycleClient(successful_unloads=verified_cold)
    corpora = []
    for split in ("calibration", "holdout"):
        document = probe.Document(f"doc-{split}", "The preferred color is navy.",
                                  f"source-{split}", 1, 0, split, "memory")
        queries = (
            probe.Query(f"positive-{split}", "What is the preferred color?", "memory",
                        (document.id,), split, "memory"),
            probe.Query(f"negative-{split}", "What is the bank balance?", "memory", (),
                        split, "memory"),
        )
        corpora.append(probe.Corpus(f"memory:{split}", "memory", split, (document,), queries))
    lexical = {(corpus.id, query.id): [] for corpus in corpora for query in corpus.queries}
    progress = {}
    result = asyncio.run(probe.measure_arm(
        client, probe.base.Arm("raw", client.model, "", ""), corpora, lexical,
        stress_inputs=["public text"] * 64, stress_metadata={"documents": 64}, progress=progress))
    assert result["status"] == "measured"
    assert result["comparable"] is True
    assert progress["stage"] == "completed"
    assert result["cold_queries"]["status"] == status
    assert result["cold_queries"]["requested_samples"] == 3
    assert result["cold_queries"]["unavailable_samples"] == 3 - verified_cold
    assert len(result["cold_queries"]["samples"]) == verified_cold
    assert all(row["verified_unloaded"] for row in result["cold_queries"]["samples"])
    assert result["cold_queries"]["issues"][0]["verified_unloaded"] is False
    assert result["cold_queries"]["p50_seconds"] == (.01 if verified_cold else None)
    assert result["warm_queries"]["samples"] == 4
    assert result["ingestion"]["documents"] == 2
    assert len(result["ingestion"]["batch64_stress"]) == 2
    assert result["workloads"]["memory"]["holdout_passed"] is True
    assert len(result["corpora"]) == 2
    assert client.embed_calls == 8 + verified_cold
