#!/usr/bin/env python3
"""Compare public real-document retrieval and invented memory/history records.

Run with the existing probe-onbox.sh wrapper; copy this probe's three fixtures
and embedding_retrieval_probe.py alongside it. --documents-only instead uses
the public reference fixture and optional --upload-cases layout, with no cold
or stress sampling and no explicit model unloads. It warms both configured
serving profiles on completion. This evaluator reads no private collections
and writes no Qdrant index, application setting or model weights.
Calibration and holdout sources are disjoint. Thresholds are fitted only on
calibration records and then frozen before holdout scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from audrey.embedding_profile import load_profile
from audrey.kb.fusion import passes_evidence, reciprocal_rank_fusion
from audrey.kb.qdrant import KBHit


def load_baseline():
    beside = Path(__file__).with_name("embedding_retrieval_probe.py")
    path = beside if beside.is_file() else (
        Path(__file__).resolve().parents[2] / "scripts/probes/embedding_retrieval_probe.py"
    )
    spec = importlib.util.spec_from_file_location("embedding_document_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("copy embedding_retrieval_probe.py alongside this probe")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_baseline()
ProbeError = base.ProbeError
MAX_EMBED_CALLS = 4096
MAX_BYTES = 1_500_000
MAX_DOCUMENTS = 768
MAX_QUERIES = 120
SPLITS = {"calibration", "holdout"}
WORKLOADS = {"reference", "documents", "memory", "history"}
RETRIEVAL_POOLS = {"reference": 24, "documents": 40, "regression": 40, "memory": 20, "history": 20}
CURRENT = {"reference": (0.55, 0.5), "documents": (0.53, 0.7), "regression": (0.53, 0.7),
           "memory": (0.5, None), "history": (0.4, None)}


@dataclass(frozen=True)
class Settings(base.Settings):
    candidate_profile: str = "gemma"
    candidates: tuple[str, ...] = ()
    baselines_only: bool = False
    documents_only: bool = False

    def selected_arms(self):
        original = super().selected_arms()
        if self.baselines_only:
            return [arm for arm in original if arm.id in {"prefixed", "raw"}]
        tags = self.candidates or (self.candidate,)
        if len(tags) > 4 or len(tags) != len(set(tags)):
            raise ProbeError("CANDIDATES must contain one to four unique local model tags")
        candidates = []
        for index, tag in enumerate(tags):
            # Reuse all model-tag, URL, timeout and settings guards.
            base.Settings(candidate=tag, incumbent=self.incumbent).selected_arms()
            profile = ("qwen3" if tag.startswith("qwen3-embedding:") else "gemma") if self.candidates else self.candidate_profile
            identity = "candidate" if index == 0 else f"candidate_{index + 1}"
            if profile == "gemma" and tag.startswith("embeddinggemma"):
                arm = base.Arm(identity, tag, "title: none | text: ", "task: search result | query: ")
            elif profile == "qwen3" and tag.startswith("qwen3-embedding:"):
                arm = base.Arm(identity, tag, "",
                    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                    expected_dimensions=None)
            else:
                raise ProbeError("candidate profile must match an embeddinggemma or qwen3-embedding tag")
            candidates.append(arm)
        return candidates + [arm for arm in original if arm.id != "gemma"]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source_id: str
    page: int
    offset: int
    split: str
    workload: str
    end_page: int | None = None


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    group: str
    relevant: tuple[str, ...]
    split: str
    workload: str
    kind: str = "retrieval"


@dataclass(frozen=True)
class Corpus:
    id: str
    workload: str
    split: str
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]


def default_path(name: str) -> Path:
    copied = Path(__file__).with_name(name)
    if copied.is_file():
        return copied
    parents = Path(__file__).resolve().parents
    # The on-box wrapper copies this script directly to /tmp. argparse still
    # evaluates defaults for skipped fixtures; there may be no repo ancestor.
    return parents[2] / "evals/cases" / name if len(parents) > 2 else copied


def _text(value, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ProbeError(f"{label} must contain 1..{maximum} characters")
    return value


def _positive_int(value, label: str, *, zero=False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ProbeError(f"{label} must be a {'nonnegative' if zero else 'positive'} integer")
    return value



def fixture_inputs_sha256(body: dict) -> str:
    """Fingerprint model inputs and corpus scopes, independently of gold labels."""
    keys = {
        "sources": ("id", "url", "sha256", "split"),
        "documents": ("id", "text", "source_id", "page", "end_page", "offset", "end_offset", "split", "workload"),
        "queries": ("id", "text", "group", "split", "workload", "kind"),
    }
    selected = {name: [{key: row[key] for key in fields if key in row}
                       for row in body.get(name, [])] for name, fields in keys.items()}
    data = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def load_fixture(path: Path, *, synthetic: bool) -> tuple[list[Corpus], dict]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ProbeError("fixture exceeds 1.5MB")
        body = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ProbeError("fixture is missing or invalid JSON") from error
    if (not isinstance(body, dict) or body.get("schema") != 2
            or body.get("language") != "en" or body.get("public") is not True
            or body.get("synthetic") is not synthetic):
        raise ProbeError("fixture must be schema2, English, public and correctly identify synthetic data")
    sources = body.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ProbeError("sources must be a nonempty list")
    source_map = {}
    source_fingerprints = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ProbeError("source must be an object")
        identity = _text(source.get("id"), "source id", 200)
        if identity in source_map or source.get("split") not in SPLITS:
            raise ProbeError("source IDs must be unique and each source belongs to one split")
        if not synthetic:
            if (not str(source.get("url", "")).startswith("https://")
                    or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))
                    or source.get("license") != "US government work"):
                raise ProbeError("public sources require https URL, SHA256 and government-work license")
            _text(source.get("title"), "source title", 400)
            _positive_int(source.get("pages"), "source pages")
        if not synthetic:
            for fingerprint in (source["url"], source["sha256"]):
                if fingerprint in source_fingerprints and source_fingerprints[fingerprint] != source["split"]:
                    raise ProbeError("source leakage: same source URL or digest appears in both splits")
                source_fingerprints[fingerprint] = source["split"]
        source_map[identity] = source
    docs = body.get("documents")
    queries = body.get("queries")
    if not isinstance(docs, list) or not 1 <= len(docs) <= MAX_DOCUMENTS:
        raise ProbeError("document count must be 1..768")
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
        raise ProbeError("query count must be 1..120")
    documents, query_rows, doc_map, query_ids = [], [], {}, set()
    default_workload = body.get("workload", "reference" if not synthetic else None)
    for row in docs:
        if not isinstance(row, dict):
            raise ProbeError("document must be an object")
        identity = _text(row.get("id"), "document id", 200)
        source = source_map.get(row.get("source_id"))
        split = row.get("split")
        workload = row.get("workload", default_workload)
        if identity in doc_map or source is None:
            raise ProbeError("document has duplicate id or missing source")
        if split not in SPLITS or source["split"] != split:
            raise ProbeError("source leakage: document split differs from its source")
        if workload not in WORKLOADS or (synthetic and workload in {"reference", "documents"}):
            raise ProbeError("document has an invalid workload")
        page = _positive_int(row.get("page"), "document page")
        end_page = _positive_int(row.get("end_page", page), "document end page")
        if end_page < page or (not synthetic and end_page > source["pages"]):
            raise ProbeError("document page span exceeds source page count or is reversed")
        doc = Document(identity, _text(row.get("text"), "document text", 10000),
                       row["source_id"], page,
                       _positive_int(row.get("offset"), "document offset", zero=True),
                       split, workload, end_page if "end_page" in row else None)
        if "text_sha256" in row and hashlib.sha256(doc.text.encode()).hexdigest() != row["text_sha256"]:
            raise ProbeError("document text checksum differs from its frozen source excerpt")
        documents.append(doc)
        doc_map[identity] = doc
    for row in queries:
        if not isinstance(row, dict):
            raise ProbeError("query must be an object")
        identity = _text(row.get("id"), "query id", 200)
        if identity in query_ids:
            raise ProbeError("query IDs must be unique")
        query_ids.add(identity)
        split = row.get("split")
        workload = row.get("workload", default_workload)
        if split not in SPLITS or workload not in WORKLOADS:
            raise ProbeError("query has an invalid split or workload")
        relevant = row.get("relevant")
        if (not isinstance(relevant, list) or any(not isinstance(item, str) for item in relevant)
                or len(set(relevant)) != len(relevant)
                or row.get("answerable") is not bool(relevant)):
            raise ProbeError("query answerable must match its unique relevance labels")
        for identity_ in relevant:
            doc = doc_map.get(identity_)
            if doc is None or (doc.split, doc.workload) != (split, workload):
                raise ProbeError("query label references missing document or leaks across splits/workloads")
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list) or (relevant and not synthetic and not evidence):
            raise ProbeError("answerable public queries require source evidence")
        for item in evidence:
            if not isinstance(item, dict):
                raise ProbeError("evidence must be an object")
            quote = " ".join(_text(item.get("quote"), "evidence quote", 5000).split())
            if not any(doc.source_id == item.get("source_id")
                       and doc.page <= _positive_int(item.get("page"), "evidence page") <= (doc.end_page or doc.page)
                       and quote in " ".join(doc.text.split())
                       for doc in (doc_map[key] for key in relevant)):
                raise ProbeError("evidence quote does not match a labeled source/page")
        if relevant and not synthetic:
            for key in relevant:
                doc = doc_map[key]
                if not any(item.get("source_id") == doc.source_id
                           and doc.page <= _positive_int(item.get("page"), "evidence page") <= (doc.end_page or doc.page)
                           and " ".join(item["quote"].split()) in " ".join(doc.text.split())
                           for item in evidence):
                    raise ProbeError("every relevant document requires matching source/page evidence")
        query_rows.append(Query(identity, _text(row.get("text"), "query text", 2000),
                                str(row.get("group", workload)), tuple(relevant), split,
                                workload, str(row.get("kind", "retrieval"))))
    corpora = []
    contexts = {(doc.workload, doc.split) for doc in documents}
    for workload, split in sorted(contexts):
        selected_docs = tuple(doc for doc in documents
                              if (doc.workload, doc.split) == (workload, split))
        selected_queries = tuple(query for query in query_rows
                                 if (query.workload, query.split) == (workload, split))
        if not selected_queries or not any(query.relevant for query in selected_queries):
            raise ProbeError("every corpus requires answerable queries")
        if not any(not query.relevant for query in selected_queries):
            raise ProbeError("every calibration/holdout corpus requires no-answer queries")
        corpora.append(Corpus(f"{workload}:{split}", workload, split,
                              selected_docs, selected_queries))
    if any((query.workload, query.split) not in contexts for query in query_rows):
        raise ProbeError("query context has no documents")
    for workload in {corpus.workload for corpus in corpora}:
        if {corpus.split for corpus in corpora if corpus.workload == workload} != SPLITS:
            raise ProbeError("each workload requires calibration and holdout")
    return corpora, {"sha256": hashlib.sha256(raw).hexdigest(),
                     "input_sha256": fixture_inputs_sha256(body), "schema": 2,
                     "public": True, "synthetic": synthetic, "language": "en",
                     "documents": len(documents), "queries": len(query_rows),
                     "layout": {key: body[key] for key in
                                ("chunking", "chunk_policy", "label_policy", "derivation", "provenance", "scope", "questions_reused_from") if key in body},
                     "sources": [{key: value for key, value in source.items()
                                  if key in {"id", "url", "sha256", "title", "split", "pages", "license"}}
                                 for source in sources]}


def load_regression(path: Path) -> tuple[Corpus, dict]:
    docs, queries, digest = base.load_cases(path)
    return Corpus("regression", "regression", "regression",
                  tuple(Document(doc.id, doc.text, doc.id, 1, 0, "regression", "regression")
                        for doc in docs),
                  tuple(Query(query.id, query.text, query.group, query.relevant,
                              "regression", "regression") for query in queries)), {
                      "sha256": digest, "schema": 1, "synthetic": True,
                      "documents": len(docs), "queries": len(queries)}


def call_budget(settings, corpora: list[Corpus], *, cold_samples=3, stress_batches=2) -> int:
    arms = settings.selected_arms()
    if not corpora or len({corpus.id for corpus in corpora}) != len(corpora):
        raise ProbeError("corpora must be nonempty with unique identifiers")
    if len(corpora) > 7 or sum(len(c.documents) for c in corpora) > 900:
        raise ProbeError("combined corpus exceeds bounded context/document counts")
    if any(not corpus.documents or not corpus.queries for corpus in corpora):
        raise ProbeError("each corpus requires documents and queries")
    if cold_samples != 3 or not 1 <= stress_batches <= 2:
        raise ProbeError("this evaluation requires three cold samples and one or two stress batches")
    lifecycle = 0 if settings.documents_only else cold_samples + stress_batches
    calls = len(arms) * (lifecycle
                        + sum(math.ceil(len(corpus.documents) / settings.batch_size)
                              + len(corpus.queries) * settings.rounds for corpus in corpora))
    if settings.documents_only:
        calls += 2  # Restore the configured text/history profiles with 24-hour residency.
    if calls > MAX_EMBED_CALLS:
        raise ProbeError(f"estimated embedding requests {calls} exceed maximum {MAX_EMBED_CALLS}")
    return calls


def rank_query(corpus: Corpus, query: Query, matrix: np.ndarray, vector: list[float],
               lexical: list[KBHit]) -> dict:
    q = np.asarray(vector, dtype=np.float64)
    q /= math.hypot(*vector)
    scores = np.clip(matrix @ q, -1, 1)
    order = sorted(range(len(corpus.documents)),
                   key=lambda index: (-float(scores[index]), corpus.documents[index].id))
    dense = [KBHit(float(scores[index]), corpus.documents[index].id, "text", 0,
                   corpus.documents[index].text, {}) for index in order]
    fused = reciprocal_rank_fusion(dense[:RETRIEVAL_POOLS[corpus.workload]], lexical, query=query.text)
    hybrid = corpus.workload in {"reference", "documents", "regression"}
    candidates = [{"id": entry.hit.source, "dense_cosine": entry.dense_score,
                   "lexical_overlap": entry.overlap, "lexical": "lexical" in entry.retrievers}
                  for entry in fused] if hybrid else [
                      {"id": hit.source, "dense_cosine": hit.score,
                       "lexical_overlap": 0.0, "lexical": False}
                      for hit in dense[:RETRIEVAL_POOLS[corpus.workload]]]
    return {"id": query.id, "group": query.group, "kind": query.kind,
            "split": query.split, "workload": query.workload,
            "answerable": bool(query.relevant), "relevant": list(query.relevant),
            "dense": [hit.source for hit in dense],
            "hybrid": [entry.hit.source for entry in fused],
            "lexical": [hit.source for hit in lexical], "candidates": candidates}


def admitted(row: dict, min_score: float, min_overlap: float | None) -> list[str]:
    result = []
    for candidate in row["candidates"]:
        # Construct the same minimal evidence object consumed by production.
        entry = SimpleNamespace(dense_score=candidate["dense_cosine"],
                                overlap=candidate["lexical_overlap"],
                                retrievers={"lexical"} if candidate["lexical"] else set())
        if passes_evidence(entry, min_score=min_score,
                           min_overlap=1.01 if min_overlap is None else min_overlap):
            result.append(candidate["id"])
    return result[:3]


def admission_metrics(rows: list[dict], min_score: float,
                      min_overlap: float | None, *, diagnostics=False) -> dict:
    positives = negatives = supported = correct_first = wrong_first = rejected = 0
    negative_admissions = wrong_hits = 0
    per_query = []
    for row in rows:
        ids = admitted(row, min_score, min_overlap)
        gold = set(row["relevant"])
        if gold:
            positives += 1
            supported += bool(gold.intersection(ids))
            correct_first += bool(ids and ids[0] in gold)
            wrong_first += bool(ids and ids[0] not in gold)
            rejected += not ids
            wrong_hits += sum(identity not in gold for identity in ids)
        else:
            negatives += 1
            negative_admissions += bool(ids)
        if diagnostics:
            per_query.append({"id": row["id"], "answerable": bool(gold),
                              "admitted": ids, "supported": bool(gold.intersection(ids)),
                              "wrong_first": bool(ids and ids[0] not in gold),
                              "wrong_admitted": [identity for identity in ids if identity not in gold]})
    rate = supported / positives if positives else 0.0
    result = {"positive_queries": positives, "negative_queries": negatives,
              "supported_top3": supported, "positive_support_rate": round(rate, 6),
              "correct_top1": correct_first, "wrong_positive_top1": wrong_first,
              "positive_rejected": rejected, "wrong_positive_hits_top3": wrong_hits,
              "negative_admissions": negative_admissions,
              "negative_admission_rate": round(negative_admissions / negatives, 6) if negatives else None,
              "feasible": bool(positives and negatives and rate >= .9 and negative_admissions == 0)}
    if diagnostics:
        result["queries"] = per_query
    return result


def calibrate(rows: list[dict], *, hybrid: bool) -> dict:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ProbeError("threshold calibration may use only calibration queries")
    candidates = []
    for cutoff in (value / 50 for value in range(51)):
        for overlap in ([.5, .7, .9, 1.0, 1.01] if hybrid else [None]):
            metrics = admission_metrics(rows, cutoff, overlap)
            candidates.append({"min_dense_score": cutoff, "min_term_overlap": overlap, **metrics})
    feasible = [row for row in candidates if row["feasible"]]
    pool = feasible or candidates
    chosen = max(pool, key=lambda row: (
        row["supported_top3"], -row["negative_admissions"],
        -row["wrong_positive_top1"], -row["wrong_positive_hits_top3"],
        row["min_dense_score"], row["min_term_overlap"] or 0))
    return {"min_dense_score": chosen["min_dense_score"],
            "min_term_overlap": chosen["min_term_overlap"],
            "calibration": admission_metrics(rows, chosen["min_dense_score"],
                                             chosen["min_term_overlap"], diagnostics=True),
            "calibration_feasible": bool(feasible), "grid_candidates": len(candidates),
            "selection": "feasible pool first; maximize supported top3, minimize false admissions and wrong hits",
            "frozen_before_holdout": True}


def quality_summary(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["relevant"]]
    result = {}
    for method in ("dense", "hybrid", "lexical"):
        metrics = base.ranking_metrics([row[method] for row in positive],
                                       [tuple(row["relevant"]) for row in positive])
        for depth in (1, 3, 5):
            count = sum(bool(set(row[method][:depth]) & set(row["relevant"])) for row in positive)
            metrics[f"hits_at_{depth}"] = count
            metrics[f"hit_rate_at_{depth}"] = round(count / len(positive), 6) if positive else None
        result[method] = metrics
    return result


def summarize_workloads(corpora_results: list[dict], policies: dict | None = None) -> dict:
    contexts = {row["id"]: row for row in corpora_results}
    result = {}
    for workload in sorted({row["workload"] for row in corpora_results} - {"regression"}):
        calibration = contexts[f"{workload}:calibration"]["queries"]
        holdout = contexts[f"{workload}:holdout"]["queries"]
        # First repeat only: repeated embeddings do not manufacture new labels.
        fitted = calibrate([row for row in calibration if row["repeat"] == 1],
                           hybrid=workload in {"reference", "documents"})
        holdout_metrics = admission_metrics(
            [row for row in holdout if row["repeat"] == 1], fitted["min_dense_score"],
            fitted["min_term_overlap"], diagnostics=True)
        current = policies[workload] if policies is not None else CURRENT[workload]
        result[workload] = {
            "calibrated_policy": fitted, "holdout": holdout_metrics,
            ("configured_qwen_policy_diagnostic" if policies is not None else "current_policy"):
                              {"min_dense_score": current[0], "min_term_overlap": current[1],
                               "not_a_cross_model_calibration": policies is not None,
                               "holdout": admission_metrics(
                                   [row for row in holdout if row["repeat"] == 1], *current,
                                   diagnostics=True)},
            "holdout_passed": fitted["calibration_feasible"] and holdout_metrics["feasible"]
                              and holdout_metrics["wrong_positive_top1"] == 0}
    return result


def load_stress(path: Path) -> tuple[list[str], dict]:
    """Read frozen public inputs, avoiding runtime downloads by tokenizers."""
    try:
        body = json.loads(path.read_text())
        stress = body["stress"]
        texts, metadata = stress["texts"], stress["metadata"]
        if not isinstance(texts, list) or len(texts) != 64:
            raise ValueError
        if any(not isinstance(text, str) or not 1 <= len(text) <= 12000 for text in texts):
            raise ValueError
        digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        if metadata["sha256"] != digest or metadata["documents"] != 64:
            raise ValueError
        if metadata["tokenizer"] != "cl100k_base" or not 900 <= metadata["min_tokens"] <= metadata["max_tokens"] <= 1100:
            raise ValueError
        return texts, metadata
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ProbeError("frozen public batch64 inputs are missing or invalid") from error


async def verify_unload(client, model: str, *, timeout_seconds=5.0, poll_interval=.25) -> dict:
    """Wait for asynchronous unload; a failed cold check must not discard warm evidence."""
    snapshot, polls = None, 0
    try:
        async with asyncio.timeout(timeout_seconds):
            await client.unload(model)
            attempts = math.ceil(timeout_seconds / poll_interval) + 1
            for attempt in range(attempts):
                snapshot = await client.residency()
                polls += 1
                if model not in base.resident_names(snapshot):
                    return {"verified_unloaded": True, "after_unload": snapshot, "polls": polls}
                if attempt + 1 < attempts:
                    await asyncio.sleep(poll_interval)
        reason = "model remained resident after bounded unload polling"
    except TimeoutError:
        reason = "model unload verification exceeded its bounded wait"
    except ProbeError as error:
        reason = f"model unload could not be verified: {error}"
    return {"verified_unloaded": False, "after_unload": snapshot,
            "polls": polls, "reason": reason}


async def measure_arm(client, arm, corpora: list[Corpus], lexical: dict,
                      *, stress_inputs, stress_metadata, progress: dict,
                      cold_samples=3, stress_batches=2, policies=None,
                      quality_only=False) -> dict:
    progress.update(stage="cold_samples", completed_documents=0, completed_query_samples=0)
    before = await client.residency()
    cold, cold_issues = [], []
    sample_query = corpora[0].queries[0]
    for index in range(0 if quality_only else cold_samples):
        unload = await verify_unload(client, arm.model)
        if not unload["verified_unloaded"]:
            cold_issues.append({"sample": index + 1, **unload})
            break
        _, timing = await client.embed(arm, [arm.query_prefix + sample_query.text])
        cold.append({**unload, **timing})
    progress["stage"] = "corpus_embeddings"
    corpus_results, warm_times, ingest_times, norms = [], [], [], []
    for corpus in corpora:
        vectors, batches = [], []
        for offset in range(0, len(corpus.documents), client.settings.batch_size):
            batch = corpus.documents[offset:offset + client.settings.batch_size]
            got, timing = await client.embed(arm, [arm.document_prefix + doc.text for doc in batch])
            vectors.extend(got)
            norms.extend(math.hypot(*vector) for vector in got)
            batches.append({"documents": len(batch), **timing})
            ingest_times.append(timing["wall_seconds"])
            progress["completed_documents"] += len(batch)
        matrix = np.asarray(vectors, dtype=np.float64)
        matrix /= np.asarray([math.hypot(*vector) for vector in vectors])[:, None]
        progress["stage"] = "warm_queries"
        rows = []
        for repeat in range(client.settings.rounds):
            for query in corpus.queries:
                got, timing = await client.embed(arm, [arm.query_prefix + query.text])
                warm_times.append(timing["wall_seconds"])
                row = rank_query(corpus, query, matrix, got[0], lexical[(corpus.id, query.id)])
                row.update(repeat=repeat + 1, query_timing=timing)
                rows.append(row)
                progress["completed_query_samples"] += 1
        first = [row for row in rows if row["repeat"] == 1]
        corpus_results.append({
            "id": corpus.id, "workload": corpus.workload, "split": corpus.split,
            "documents": len(corpus.documents), "unique_queries": len(corpus.queries),
            "positive_queries": sum(bool(query.relevant) for query in corpus.queries),
            "negative_queries": sum(not query.relevant for query in corpus.queries),
            "ranking": quality_summary(first),
            ("configured_qwen_policy_diagnostic" if policies is not None else "current_policy"):
                              {"min_dense_score": (policies or CURRENT)[corpus.workload][0],
                               "min_term_overlap": (policies or CURRENT)[corpus.workload][1],
                               "not_a_cross_model_calibration": policies is not None,
                               **admission_metrics(first, *(policies or CURRENT)[corpus.workload], diagnostics=True)},
            "ingestion": {"batches": batches, "seconds": round(sum(
                sample["wall_seconds"] for sample in batches), 6)},
            "repeat_ranking_consistent": all(
                row["dense"] == first[index % len(first)]["dense"]
                and row["hybrid"] == first[index % len(first)]["hybrid"]
                for index, row in enumerate(rows)),
            "queries": rows})
        progress["stage"] = "corpus_embeddings"
    progress["stage"] = "batch64_stress"
    stress = []
    for _ in range(0 if quality_only else stress_batches):
        _, timing = await client.embed(arm, [arm.document_prefix + text for text in stress_inputs])
        stress.append({**stress_metadata, **timing})
    after = await client.residency()
    progress["stage"] = "completed"
    return {"arm": arm.id, "model": arm.model, "status": "measured", "comparable": True,
            "format": {"document_prefix": arm.document_prefix, "query_prefix": arm.query_prefix},
            "dimensions": len(vectors[0]),
            "document_vector_norms": {"min": round(min(norms), 8), "max": round(max(norms), 8)},
            "cold_queries": {"status": "not_requested" if quality_only else "completed" if len(cold) == cold_samples else (
                                 "partial" if cold else "unavailable"),
                             "requested_samples": 0 if quality_only else cold_samples, "samples": cold,
                             "unavailable_samples": 0 if quality_only else cold_samples - len(cold), "issues": cold_issues,
                             "within_memory_4s": sum(sample["wall_seconds"] <= 4 for sample in cold),
                             "within_kb_24s": sum(sample["wall_seconds"] <= 24 for sample in cold),
                             "p50_seconds": base.percentile(
                [sample["wall_seconds"] for sample in cold], .5)},
            "warm_queries": {"samples": len(warm_times),
                             "p50_seconds": base.percentile(warm_times, .5),
                             "p95_seconds": base.percentile(warm_times, .95)},
            "ingestion": {"documents": sum(len(corpus.documents) for corpus in corpora),
                          "batch_size": client.settings.batch_size,
                          "seconds": round(sum(ingest_times), 6), "batch64_stress": stress},
            "workloads": summarize_workloads(corpus_results, policies), "corpora": corpus_results,
            "residency": {"before": before, "after": after,
                          "displaced": sorted(base.resident_names(before) - base.resident_names(after))}}


def assess_candidate(measured: dict, candidate_id: str) -> dict:
    decisions = {}
    for workload, assessment in measured[candidate_id]["workloads"].items():
        baseline = measured["prefixed" if workload == "reference" else "raw"]
        method = "hybrid" if workload == "reference" else "dense"
        context = f"{workload}:holdout"
        gemma_quality = next(row for row in measured[candidate_id]["corpora"] if row["id"] == context)
        old_quality = next(row for row in baseline["corpora"] if row["id"] == context)
        old_admission = baseline["workloads"][workload]["holdout"]
        new_admission = assessment["holdout"]
        nonregression = (gemma_quality["ranking"][method]["mrr"]
                         >= old_quality["ranking"][method]["mrr"]
                         and gemma_quality["ranking"][method]["hit_rate_at_3"]
                         >= old_quality["ranking"][method]["hit_rate_at_3"]
                         and new_admission["supported_top3"] >= old_admission["supported_top3"]
                         and new_admission["negative_admissions"] <= old_admission["negative_admissions"])
        decisions[workload] = {"status": "candidate" if assessment["holdout_passed"]
                              and nonregression else "inconclusive",
                              "baseline": baseline["arm"], "ranking_nonregression": nonregression,
                              "frozen_admission_gate": assessment["holdout_passed"]}
    regression = {}
    for arm_id in (candidate_id, "raw"):
        regression[arm_id] = next((row for row in measured[arm_id]["corpora"]
                                   if row["workload"] == "regression"), None)
    regression_ok = True
    if any(regression.values()):
        regression_ok = all(regression.values()) and all(
            regression[candidate_id]["ranking"][method][metric]
            >= regression["raw"]["ranking"][method][metric]
            for method in ("dense", "hybrid") for metric in ("mrr", "hit_rate_at_3"))
    return {"status": "candidate" if regression_ok and all(row["status"] == "candidate"
                                          for row in decisions.values()) else "inconclusive",
            "workloads": decisions, "regression_nonregression": bool(regression_ok),
            "migration_performed": False,
            "meaning": "bounded local evidence for a migration decision; synthetic personal records are not private-corpus acceptance"}


def document_config(path):
    try:
        profiles = {role: load_profile(path, role=role) for role in ("text", "history")}
        if (not profiles["text"].model.startswith("qwen3-embedding:")
                or profiles["history"].model != "nomic-embed-text:latest"):
            raise ProbeError("document comparison requires configured Qwen text and Nomic history profiles")
        body = yaml.safe_load(path.read_text())
        reference = body["reference_libraries"]["search"]
        kb = body["kb"]
        policies = {
            "reference": (reference["min_dense_score"], reference.get("min_term_overlap")),
            "documents": (kb["min_score"], kb["hybrid"]["min_term_overlap"]),
        }
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise ProbeError("document comparison configuration is incomplete or invalid") from error
    for score, overlap in policies.values():
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not -1 <= score <= 1
                or (overlap is not None and (isinstance(overlap, bool)
                    or not isinstance(overlap, (int, float))
                    or not math.isfinite(overlap) or not 0 <= overlap <= 1))):
            raise ProbeError("configured document retrieval policy is invalid")
    return profiles, policies


def configured_arms(settings, profiles):
    arms = settings.selected_arms()
    if not settings.documents_only:
        return arms
    if profiles is None or settings.baselines_only or len(arms) != 3:
        raise ProbeError("documents-only mode requires the configured Qwen and both Nomic formats")
    output = []
    for arm in arms:
        profile = profiles["text" if arm.id == "candidate" else "history"]
        if arm.model != profile.model or (arm.id != "raw" and (
                arm.document_prefix != profile.document_prefix or arm.query_prefix != profile.query_prefix)):
            raise ProbeError("comparison model or input format differs from configured embedding profile")
        output.append(replace(arm, expected_dimensions=profile.dimensions))
    return output


def verify_configured_models(metadata, profiles):
    for profile in profiles.values():
        observed = metadata["installed_models"].get(profile.model, {})
        if observed.get("digest") != profile.digest:
            raise ProbeError("installed comparison weights differ from the configured pinned digest")


def paired_document_results(results):
    measured = {arm["arm"]: arm for arm in results if arm.get("comparable")}
    if not {"candidate", "raw", "prefixed"} <= set(measured):
        return []
    pairs = []
    for baseline in ("raw", "prefixed"):
        for corpus in measured["candidate"]["corpora"]:
            if corpus["split"] != "holdout":
                continue
            other = next(row for row in measured[baseline]["corpora"] if row["id"] == corpus["id"])
            selected = {row["id"]: row for row in corpus["queries"] if row["repeat"] == 1 and row["relevant"]}
            controls = {row["id"]: row for row in other["queries"] if row["repeat"] == 1 and row["relevant"]}
            if set(selected) != set(controls):
                raise ProbeError("paired comparison query scopes differ")
            for method in ("dense", "hybrid"):
                changes, wins, losses, ties = [], 0, 0, 0
                at_depth = {str(depth): {"wins": 0, "losses": 0, "ties": 0} for depth in (1, 3, 5)}
                for identity, row in selected.items():
                    control = controls[identity]
                    if row["relevant"] != control["relevant"]:
                        raise ProbeError("paired comparison relevance labels differ")
                    ranks = [next((index for index, item in enumerate(value[method], 1)
                                   if item in value["relevant"]), None) for value in (row, control)]
                    left, right = [rank if rank is not None else math.inf for rank in ranks]
                    wins += left < right
                    losses += left > right
                    ties += left == right
                    if left != right:
                        changes.append({"id": identity, "qwen_rank": ranks[0], "baseline_rank": ranks[1]})
                    for depth in (1, 3, 5):
                        outcome = "wins" if left <= depth < right else "losses" if right <= depth < left else "ties"
                        at_depth[str(depth)][outcome] += 1
                pairs.append({"layout": corpus["workload"], "baseline": baseline, "method": method,
                              "positive_queries": len(selected), "rank_wins": wins,
                              "rank_losses": losses, "rank_ties": ties,
                              "at_depth": at_depth, "rank_changes": changes})
    return pairs


def recommendation(report: dict) -> dict:
    measured = {row["arm"]: row for row in report["results"] if row.get("comparable")}
    if report.get("settings", {}).get("documents_only"):
        findings = bool(report["errors"]) or any(
            not workload["holdout_passed"] for arm in measured.values()
            for workload in arm["workloads"].values()
        )
        return {"status": "comparison_completed" if not report["errors"] else "incomplete",
                "quality_findings": findings, "automatic_promotion": False,
                "migration_performed": False,
                "meaning": "Compare paired ranks and per-model calibrated admission separately; configured Qwen thresholds are diagnostics, not Nomic calibration"}
    if report.get("settings", {}).get("baselines_only") and not report["errors"] and set(measured) == {"prefixed", "raw"}:
        return {"status": "baselines_measured", "migration_performed": False,
                "meaning": "combine these matching baselines with the retained candidate measurements"}
    if report["errors"] or not {"candidate", "prefixed", "raw"} <= set(measured):
        return {"status": "inconclusive", "reason": "all requested candidate and baseline arms must complete"}
    candidates = {identity: {"model": arm["model"], **assess_candidate(measured, identity)}
                  for identity, arm in measured.items() if identity.startswith("candidate")}
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return {"status": "candidate" if any(row["status"] == "candidate" for row in candidates.values()) else "inconclusive",
            "candidates": candidates, "migration_performed": False,
            "meaning": "compare quality, rejection, latency and residency per workload before choosing; no automatic switch"}



async def collect_report(settings, corpora: list[Corpus], fixtures: dict,
                         *, transport=None, cold_samples=3, stress_batches=2,
                         stress_inputs=None, stress_metadata=None,
                         profiles=None, production_policies=None) -> dict:
    calls = call_budget(settings, corpora, cold_samples=cold_samples, stress_batches=stress_batches)
    arms = configured_arms(settings, profiles)
    # All lexical construction is outside embedding timings and done once.
    lexical = {(corpus.id, query.id): base.lexical_rank(
        [base.Document(doc.id, doc.text) for doc in corpus.documents], query.text,
        limit=RETRIEVAL_POOLS[corpus.workload])
        for corpus in corpora for query in corpus.queries}
    if stress_inputs is None and not settings.documents_only:
        raise ProbeError("frozen public stress inputs must be loaded before model requests")
    client = base.Ollama(settings, transport, max_embed_calls=MAX_EMBED_CALLS)
    report = None
    try:
        metadata = await base.preflight(client, arms)
        if settings.documents_only:
            verify_configured_models(metadata, profiles)
        report = {
            "schema": 2, "status": "measured", "measurement_status": "completed",
            "created_at": datetime.now(UTC).isoformat(), "provenance": metadata,
            "fixtures": fixtures, "results": [], "errors": [],
            "scope": "Matched public English document layouts; reused questions/sources, not independent new evidence" if settings.documents_only else "public English government documents plus synthetic memory/history and regression",
            "settings": {"rounds": settings.rounds, "batch_size": settings.batch_size,
                         "candidate_profile": settings.candidate_profile,
                         "baselines_only": settings.baselines_only, "documents_only": settings.documents_only,
                         "candidate_models": [arm.model for arm in arms if arm.id.startswith("candidate")],
                         "cold_samples": 0 if settings.documents_only else cold_samples,
                         "stress_batches64": 0 if settings.documents_only else stress_batches,
                         "estimated_embedding_requests": calls,
                         "maximum_embedding_requests": MAX_EMBED_CALLS,
                         "timeout_seconds": settings.timeout, "keep_alive": settings.keep_alive},
            "retrieval": {"cosine": "exact normalized NumPy, isolated corpus per workload/split",
                          "lexical": "existing BM25 helpers and corpus IDF emulation",
                          "fusion": "production reciprocal_rank_fusion and passes_evidence",
                          "retrieval_pools": RETRIEVAL_POOLS,
                          "threshold_fit": "calibration only; frozen holdout; unique labels not repeats",
                          "qdrant_contacted": False, "production_changed": False,
                          "private_documents_read": False},
            "limitations": [
                "Source-disjoint holdout is bounded public-document evidence, not a universal benchmark",
                "Memory/history records are invented; no private user text was read or reported",
                "Cold means verified Ollama unload, not filesystem-cache or GPU reset",
                "No worker-concurrency stress or end-to-end answer grading",
                "Exact in-memory ranking emulates retrieval, not Qdrant ANN or access-control execution",
                "Calibration infeasibility is reported; a cosine threshold cannot enforce year/scope policy",
                "Batch64 uses frozen public cl100k1000-token chunks; model-specific token counts can differ",
                "No image/audio embedding validation and no production reindex"],
            "residency_before": await client.residency()}
        for arm in arms:
            progress = {}
            try:
                result = await measure_arm(client, arm, corpora, lexical, progress=progress,
                                           stress_inputs=stress_inputs, stress_metadata=stress_metadata,
                                           cold_samples=cold_samples, stress_batches=stress_batches,
                                           policies=production_policies, quality_only=settings.documents_only)
                report["results"].append(result)
            except (ProbeError, ValueError, ArithmeticError) as error:
                report["errors"].append({"arm": arm.id, "stage": progress.get("stage"),
                                         "error": str(error) if isinstance(error, ProbeError) else type(error).__name__})
                report["results"].append({"arm": arm.id, "model": arm.model,
                                          "status": "incomplete", "comparable": False, **progress})
            finally:
                if not settings.documents_only and arm.model != settings.incumbent:
                    try:
                        await client.unload(arm.model)
                    except ProbeError as error:
                        report["errors"].append({"arm": arm.id, "cleanup_error": str(error)})
        if settings.documents_only:
            report["limitations"] = [
                "Same public sources/questions reused across layouts; not independent new benchmark evidence",
                "Exact in-memory dense/RRF ranking before per-file diversity; not native end-to-end top-k or Qdrant ANN",
                "No cold/stress samples or explicit unloads; timing includes uncontrolled current residency",
                "Fixed configured Qwen policy diagnostics do not calibrate Nomic's score space",
                "No private data, production index writes, model promotion or setting changes",
            ]
            report["configured_profiles"] = {role: profile.metadata["audrey_embedding"]
                                             for role, profile in profiles.items()}
            report["paired_holdout"] = paired_document_results(report["results"])
            # Leave both existing serving profiles warm, without unloading any model.
            previous_settings = client.settings
            client.settings = replace(settings, keep_alive="24h")
            restored = []
            for role, profile in profiles.items():
                try:
                    arm = base.Arm(role, profile.model, profile.document_prefix,
                                   profile.query_prefix, profile.dimensions)
                    await client.embed(arm, [profile.query("document comparison completed")])
                    restored.append(profile.model)
                except ProbeError as error:
                    report["errors"].append({"restoration_error": str(error), "role": role})
            client.settings = previous_settings
            report["restoration"] = {"warm_requests_completed": restored, "keep_alive": "24h",
                                     "explicit_unloads": 0, "resident_models_after": None,
                                     "both_configured_observed": None}
            try:
                verify_configured_models(await base.preflight(client, arms), profiles)
            except ProbeError as error:
                report["errors"].append({"identity_recheck_error": str(error)})
        try:
            report["residency_after"] = await client.residency()
            if settings.documents_only:
                resident = sorted(base.resident_names(report["residency_after"]))
                report["restoration"]["resident_models_after"] = resident
                report["restoration"]["both_configured_observed"] = all(
                    profile.model in resident for profile in profiles.values()
                )
        except ProbeError as error:
            report["errors"].append({"residency_error": str(error)})
        report["embedding_calls"] = client.embed_calls
        if report["errors"]:
            report.update(status="failed", measurement_status="incomplete")
        report["quality_recommendation"] = recommendation(report)
        return report
    finally:
        await client.client.aclose()


def compact_fixture_metadata(fixtures):
    """Keep provenance without repeating the retained per-question label transcript."""
    def without_transcripts(value):
        if isinstance(value, dict):
            return {key: without_transcripts(item) for key, item in value.items() if key != "additions"}
        if isinstance(value, list):
            return [without_transcripts(item) for item in value]
        return value

    result = {
        name: {key: without_transcripts(metadata[key]) for key in
               ("sha256", "input_sha256", "documents", "queries", "layout") if key in metadata}
        for name, metadata in fixtures.items()
    }
    for metadata in result.values():
        policy = metadata.get("layout", {}).get("label_policy")
        if isinstance(policy, dict) and "revision" in policy:
            revision = policy["revision"]
            metadata["layout"]["label_policy"] = {
                key: policy[key] for key in
                ("created_before_model_measurement", "tested_models_generated_gold") if key in policy
            }
            metadata["layout"]["label_policy"]["revision"] = {
                key: revision[key] for key in (
                    "date", "original_fixture_sha256", "retrospective", "model_inputs_changed",
                    "thresholds_tuned_on_holdout", "added_passages", "affected_queries", "limits",
                ) if key in revision
            }
    original = fixtures.get("public", {}).get("layout", {}).get("label_policy")
    for name, metadata in result.items():
        policy = metadata.get("layout", {}).get("label_policy")
        if name != "public" and isinstance(policy, dict) and "original_reviewed_policy" in policy:
            if policy["original_reviewed_policy"] == without_transcripts(original):
                policy["original_reviewed_policy"] = "Same reviewed policy as public fixture above"
    return result


def compact_report(report):
    if report.get("settings", {}).get("documents_only"):
        result = {key: report[key] for key in (
            "schema", "status", "measurement_status", "created_at", "scope", "provenance",
            "settings", "embedding_calls", "errors", "limitations", "configured_profiles",
            "restoration", "quality_recommendation",
        ) if key in report}
        result["fixtures"] = compact_fixture_metadata(report["fixtures"])
        result["paired_holdout"] = []
        for original in report["paired_holdout"]:
            pair = {key: value for key, value in original.items() if key != "rank_ties"}
            pair["at_depth"] = {depth: {key: value for key, value in counts.items() if key != "ties"}
                                for depth, counts in original["at_depth"].items()}
            pair["rank_changes"] = [change for change in original["rank_changes"] if any(
                ((change["qwen_rank"] or math.inf) <= depth)
                != ((change["baseline_rank"] or math.inf) <= depth) for depth in (1, 3, 5)
            )]
            result["paired_holdout"].append(pair)
        result["paired_change_scope"] = "Holdout first repeat; listed IDs change top1/3/5 membership; aggregate rank wins/losses include every positive question"
        admission_keys = (
            "supported_top3", "negative_admissions", "wrong_positive_top1",
            "wrong_positive_hits_top3", "positive_rejected",
        )
        result["configured_qwen_policy_diagnostic"] = {
            "not_a_cross_model_calibration": True,
            "policies": {},
        }
        result["results"] = []
        for arm in report["results"]:
            output = {key: arm[key] for key in (
                "arm", "model", "status", "comparable", "format", "dimensions", "warm_queries",
            ) if key in arm}
            output["holdout_layouts"] = []
            for corpus in arm.get("corpora", []):
                if corpus["split"] != "holdout":
                    continue
                workload = arm["workloads"][corpus["workload"]]
                fitted = workload["calibrated_policy"]
                diagnostic = corpus["configured_qwen_policy_diagnostic"]
                result["configured_qwen_policy_diagnostic"]["policies"][corpus["workload"]] = {
                    key: diagnostic[key] for key in ("min_dense_score", "min_term_overlap")
                }
                output["holdout_layouts"].append({
                    "layout": corpus["workload"], "documents": corpus["documents"],
                    "positive_queries": corpus["positive_queries"], "negative_queries": corpus["negative_queries"],
                    "ranking_before_admission": {method: {
                        key: corpus["ranking"][method][key]
                        for key in ("hits_at_1", "hits_at_3", "hits_at_5", "mrr")
                    } for method in ("dense", "hybrid")},
                    "own_calibrated_policy": {"min_dense_score": fitted["min_dense_score"],
                        "min_term_overlap": fitted["min_term_overlap"],
                        "calibration_feasible": fitted["calibration_feasible"],
                        **{key: workload["holdout"][key] for key in admission_keys}},
                    "configured_qwen_policy_diagnostic": {
                        key: diagnostic[key] for key in admission_keys
                    },
                })
            result["results"].append(output)
        return result
    result = {key: value for key, value in report.items()
              if key not in {"results", "residency_before", "residency_after"}}
    result["results"] = []
    for arm in report["results"]:
        summary = {key: value for key, value in arm.items() if key not in {"corpora", "residency"}}
        summary["corpora"] = [{key: value for key, value in corpus.items() if key != "queries"}
                              for corpus in arm.get("corpora", [])]
        result["results"].append(summary)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--documents-only", action="store_true", help="compare public document layouts only; no cold/stress/unload operations")
    parser.add_argument("--config", type=Path, default=Path("/app/config.yaml"))
    parser.add_argument("--upload-cases", type=Path)
    parser.add_argument("--baselines-only", action="store_true", help="measure only the missing Nomic arms; preserve completed candidates")
    parser.add_argument("--check", action="store_true", help="validate frozen inputs and request budget without contacting any service")
    parser.add_argument("--public-cases", type=Path, default=default_path("embedding_real_documents.json"))
    parser.add_argument("--personal-cases", type=Path, default=default_path("embedding_personal_cases.json"))
    parser.add_argument("--regression-cases", type=Path, default=default_path("embedding_retrieval_cases.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        profiles, policies = document_config(args.config) if args.documents_only else (None, None)
        settings = Settings(
            base_url=os.environ.get("OLLAMA", "http://ollama:11434"),
            candidate=os.environ.get("CANDIDATE", profiles["text"].model if profiles else "embeddinggemma-2:latest"),
            incumbent=os.environ.get("INCUMBENT", profiles["history"].model if profiles else "nomic-embed-text:latest"),
            candidate_profile=os.environ.get("CANDIDATE_PROFILE", "qwen3" if profiles else "gemma"),
            baselines_only=args.baselines_only, documents_only=args.documents_only,
            candidates=tuple(item.strip() for item in os.environ.get("CANDIDATES", "").split(",") if item.strip()),
            rounds=int(os.environ.get("ROUNDS", "1" if profiles else "2")),
            batch_size=int(os.environ.get("BATCH_SIZE", "16")),
            timeout=float(os.environ.get("TIMEOUT", "120")),
            keep_alive=os.environ.get("KEEP_ALIVE", "24h" if profiles else "10m"))
        public, public_meta = load_fixture(args.public_cases, synthetic=False)
        fixture_metadata = {"public": public_meta}
        if args.documents_only:
            corpora = public
            if any(corpus.workload != "reference" for corpus in public):
                raise ProbeError("public document layout must use the reference workload")
            if args.upload_cases is not None:
                uploads, upload_meta = load_fixture(args.upload_cases, synthetic=False)
                if any(corpus.workload != "documents" for corpus in uploads):
                    raise ProbeError("upload layout must use the documents workload")
                original_queries = {(q.id, q.text, q.split, q.kind) for c in public for q in c.queries}
                uploaded_queries = {(q.id, q.text, q.split, q.kind) for c in uploads for q in c.queries}
                if original_queries != uploaded_queries:
                    raise ProbeError("document layouts must reuse identical query inputs")
                if {(row["url"], row["sha256"], row["split"]) for row in public_meta["sources"]} != {
                    (row["url"], row["sha256"], row["split"]) for row in upload_meta["sources"]
                }:
                    raise ProbeError("document layouts must reuse identical source PDFs and splits")
                corpora = [*public, *uploads]
                fixture_metadata["upload"] = upload_meta
            stress_inputs = stress_metadata = None
            configured_arms(settings, profiles)
        else:
            if args.upload_cases is not None:
                raise ProbeError("--upload-cases requires --documents-only")
            personal, personal_meta = load_fixture(args.personal_cases, synthetic=True)
            regression, regression_meta = load_regression(args.regression_cases)
            corpora = [*public, *personal, regression]
            fixture_metadata.update(personal=personal_meta, regression=regression_meta)
            stress_inputs, stress_metadata = load_stress(args.public_cases)
        call_budget(settings, corpora)
        if args.check:
            print(json.dumps({"schema": 2, "status": "ready", "model_calls": 0,
                "documents": sum(len(c.documents) for c in corpora),
                "unique_queries": (len({(q.id, q.text) for c in corpora for q in c.queries})
                                   if settings.documents_only else sum(len(c.queries) for c in corpora)),
                "query_layout_cases": sum(len(c.queries) for c in corpora),
                "arms": [arm.model for arm in settings.selected_arms()],
                "estimated_embedding_requests": call_budget(settings, corpora),
                "documents_only": settings.documents_only,
                "fixtures": fixture_metadata,
                "configured_profiles": {role: profile.metadata["audrey_embedding"]
                                         for role, profile in (profiles or {}).items()},
                "stress": stress_metadata}, indent=2))
            return 0
        report = asyncio.run(collect_report(settings, corpora, fixture_metadata,
            stress_inputs=stress_inputs, stress_metadata=stress_metadata,
            profiles=profiles, production_policies=policies))
    except (ProbeError, ValueError, OSError) as error:
        print(json.dumps({"schema": 2, "status": "failed", "measurement_status": "setup_failed",
                          "error": str(error)}, indent=2))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"schema": 2, "status": "interrupted", "measurement_status": "interrupted"}))
        return 130
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(compact_report(report) if args.summary else report, indent=2))
    if report["measurement_status"] != "completed":
        return 1
    if settings.documents_only:
        return 1 if report["quality_recommendation"]["quality_findings"] else 0
    return 0 if report["quality_recommendation"]["status"] in {"candidate", "baselines_measured"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
