#!/usr/bin/env python3
"""Compare installed English text embedders without writing an index.

Run through the existing probe-onbox.sh wrapper with
COPY=embedding_retrieval_cases.json. Defaults compare EmbeddingGemma 2 with
Nomic's recommended retrieval prefixes and the deployed raw-text KB baseline.
All passages and questions are synthetic, public, and short. No source files,
Qdrant collections, production settings, or model weights are changed.

Run while the box is otherwise idle: each tested model is unloaded before its
loading sample. Candidate arms run first; Nomic raw runs last and stays warm.
Only tested models are explicitly unloaded. Other resident models may still
be displaced by Ollama; residency snapshots record that fact.

Environment: OLLAMA (default http://ollama:11434), CASE_FILE, CANDIDATE
(default embeddinggemma-2:latest), INCUMBENT (default nomic-embed-text:latest),
ARMS (comma-separated gemma,prefixed,raw; default all three), ROUNDS (1..10),
BATCH_SIZE (1..16, default 8), TIMEOUT (seconds, default 120), KEEP_ALIVE
(default 10m). --summary prints compact metrics; full output is the default
for the existing Telegram log handoff. --report optionally saves full JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from audrey.kb.bm25 import document_vector, query_vector
from audrey.kb.fusion import reciprocal_rank_fusion
from audrey.kb.qdrant import KBHit

GROUPS = frozenset({"general", "document", "code", "reference"})
MAX_DOCUMENTS = 64
MAX_QUERIES = 48
MAX_EMBED_CALLS = 1024
RETRIEVAL_POOL = 20


class ProbeError(RuntimeError):
    """Measurement cannot be trusted; no production fallback is attempted."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    group: str
    relevant: tuple[str, ...]


@dataclass(frozen=True)
class Arm:
    id: str
    model: str
    document_prefix: str
    query_prefix: str
    expected_dimensions: int | None = 768


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://ollama:11434"
    candidate: str = "embeddinggemma-2:latest"
    incumbent: str = "nomic-embed-text:latest"
    arms: tuple[str, ...] = ("gemma", "prefixed", "raw")
    rounds: int = 1
    batch_size: int = 8
    timeout: float = 120
    keep_alive: str = "10m"

    def selected_arms(self) -> list[Arm]:
        if not self.base_url.startswith(("http://", "https://")):
            raise ProbeError("OLLAMA must use http:// or https://")
        if not 1 <= self.rounds <= 10 or not 1 <= self.batch_size <= 16:
            raise ProbeError("ROUNDS must be 1..10 and BATCH_SIZE must be 1..16")
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 600:
            raise ProbeError("TIMEOUT must be finite and between 1 and 600 seconds")
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ProbeError("ARMS must be nonempty and contain no duplicates")
        if set(self.arms) - {"gemma", "prefixed", "raw"}:
            raise ProbeError("ARMS allows only gemma,prefixed,raw")
        for tag in (self.candidate, self.incumbent):
            if not tag.strip() or len(tag) > 200 or "cloud" in tag.lower():
                raise ProbeError("model tags must be explicit, bounded local model names")
        if self.candidate == self.incumbent:
            raise ProbeError("candidate and incumbent must be different tags")
        definitions = [
            Arm("gemma", self.candidate, "title: none | text: ", "task: search result | query: "),
            Arm("prefixed", self.incumbent, "search_document: ", "search_query: "),
            Arm("raw", self.incumbent, "", ""),
        ]
        # Ignore requested order: the deployed Nomic baseline must be last.
        return [arm for arm in definitions if arm.id in self.arms]


def default_case_path() -> Path:
    copied = Path(__file__).with_name("embedding_retrieval_cases.json")
    if copied.is_file():
        return copied
    return Path(__file__).resolve().parents[2] / "evals/cases/embedding_retrieval_cases.json"


def load_cases(path: Path) -> tuple[list[Document], list[Query], str]:
    try:
        raw = path.read_bytes()
        if len(raw) > 256_000:
            raise ProbeError("fixture exceeds 256KB")
        body = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ProbeError("fixture is missing or invalid JSON") from exc
    if not isinstance(body, dict) or body.get("schema") != 1:
        raise ProbeError("fixture must be a schema-1 JSON object")
    if body.get("language") != "en" or body.get("synthetic") is not True:
        raise ProbeError("fixture must identify itself as English and synthetic")
    documents, queries = [], []
    for key, maximum in (("documents", MAX_DOCUMENTS), ("queries", MAX_QUERIES)):
        rows = body.get(key)
        if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
            raise ProbeError(f"{key} count must be 1..{maximum}")
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ProbeError(f"{key} must contain objects")
            identity, text = row.get("id"), row.get("text")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ProbeError(f"{key} has missing or duplicate id")
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                raise ProbeError(f"{key} text must contain 1..4000 characters")
            if not text.isascii():
                raise ProbeError("this fixture is English-only ASCII text")
            seen.add(identity)
            if key == "documents":
                documents.append(Document(identity, text))
            else:
                relevant = row.get("relevant")
                if row.get("group") not in GROUPS:
                    raise ProbeError("queries must have a known task group")
                if (not isinstance(relevant, list)
                        or any(not isinstance(item, str) for item in relevant)
                        or len(relevant) != len(set(relevant))):
                    raise ProbeError("relevant must contain unique document IDs")
                if row.get("answerable") is not bool(relevant):
                    raise ProbeError("answerable must match whether relevant documents exist")
                queries.append(Query(identity, text, row["group"], tuple(relevant)))
    document_ids = {doc.id for doc in documents}
    if any(set(query.relevant) - document_ids for query in queries):
        raise ProbeError("query references a missing document")
    return documents, queries, hashlib.sha256(raw).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; a single observation remains one observation."""
    if not values:
        return None
    return round(sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)], 6)


def ranking_metrics(rankings: list[list[str]], relevant: list[tuple[str, ...]]) -> dict:
    if len(rankings) != len(relevant):
        raise ProbeError("rankings and labels have different counts")
    if not rankings:
        return {"queries": 0, "mrr": None, "recall_at_1": None, "recall_at_3": None,
                "hit_rate_at_1": None, "hit_rate_at_3": None}
    reciprocal, recall_one, recall_three, hits_one, hits_three = [], [], [], [], []
    for ranking, labels in zip(rankings, relevant, strict=True):
        if len(ranking) != len(set(ranking)) or not labels:
            raise ProbeError("rankings must be unique and labels nonempty")
        expected = set(labels)
        ranks = [rank for rank, doc_id in enumerate(ranking, 1) if doc_id in expected]
        reciprocal.append(1 / ranks[0] if ranks else 0)
        one, three = len(set(ranking[:1]) & expected), len(set(ranking[:3]) & expected)
        recall_one.append(one / len(expected))
        recall_three.append(three / len(expected))
        hits_one.append(bool(one))
        hits_three.append(bool(three))
    return {
        "queries": len(rankings), "mrr": round(statistics.mean(reciprocal), 6),
        "recall_at_1": round(statistics.mean(recall_one), 6),
        "recall_at_3": round(statistics.mean(recall_three), 6),
        "hit_rate_at_1": round(statistics.mean(hits_one), 6),
        "hit_rate_at_3": round(statistics.mean(hits_three), 6),
    }


def validate_vectors(body: dict, count: int, dimensions: int | None) -> list[list[float]]:
    rows = body.get("embeddings")
    if not isinstance(rows, list) or len(rows) != count:
        raise ProbeError("embedding response count mismatch")
    result = []
    for row in rows:
        if not isinstance(row, list) or not row or len(row) > 8192:
            raise ProbeError("embedding vector has invalid dimensions")
        if dimensions is None:
            dimensions = len(row)
        if len(row) != dimensions:
            raise ProbeError(f"embedding dimensions must be {dimensions}, got {len(row)}")
        if any(isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
               for v in row):
            raise ProbeError("embedding contains non-finite or nonnumeric values")
        vector = [float(v) for v in row]
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ProbeError("embedding vector has zero or invalid norm")
        result.append(vector)
    return result


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ProbeError("cosine dimensions mismatch")
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    if not left_norm or not right_norm:
        raise ProbeError("cosine requires nonzero vectors")
    score = sum((a / left_norm) * (b / right_norm)
                for a, b in zip(left, right, strict=True))
    return min(1.0, max(-1.0, score))


def lexical_rank(
    documents: list[Document], query: str, *, limit: int = RETRIEVAL_POOL,
) -> list[KBHit]:
    """Existing BM25 token/TF helpers plus Qdrant's corpus-dependent IDF.

    This emulates ranking only; it is not a Qdrant endpoint or a score-floor
    calibration. Lexical zero-score documents are absent, as in sparse search.
    """
    sparse = [dict(zip(*document_vector(doc.text), strict=True)) for doc in documents]
    frequencies: Counter[int] = Counter()
    for vector in sparse:
        frequencies.update(vector.keys())
    terms, _ = query_vector(query)
    total = len(documents)
    hits = []
    for doc, vector in zip(documents, sparse, strict=True):
        score = sum(vector.get(term, 0) * math.log(
            1 + (total - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
        ) for term in terms)
        if score > 0:
            hits.append(KBHit(score, doc.id, "text", 0, doc.text, {}))
    return sorted(hits, key=lambda hit: (-hit.score, hit.source))[:limit]


class Ollama:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None,
        *, max_embed_calls: int = MAX_EMBED_CALLS,
    ):
        if (isinstance(max_embed_calls, bool) or not isinstance(max_embed_calls, int)
                or not 1 <= max_embed_calls <= 4096):
            raise ProbeError("embedding call budget must be an integer between 1 and 4096")
        self.max_embed_calls = max_embed_calls
        self.client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"), timeout=settings.timeout,
            transport=transport, follow_redirects=False, trust_env=False,
        )
        self.settings = settings
        self.embed_calls = 0

    async def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            response = await self.client.request(method, path, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProbeError(f"{path}: HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProbeError(f"{path}: {type(exc).__name__}") from exc
        if not isinstance(body, dict) or body.get("error"):
            raise ProbeError(f"{path}: invalid response object")
        return body

    async def embed(self, arm: Arm, texts: list[str]) -> tuple[list[list[float]], dict]:
        self.embed_calls += 1
        if self.embed_calls > self.max_embed_calls:
            raise ProbeError("bounded embedding call budget exhausted")
        started = time.monotonic()
        body = await self.request("POST", "/api/embed", {
            "model": arm.model, "input": texts, "truncate": False,
            "keep_alive": self.settings.keep_alive,
        })
        elapsed = time.monotonic() - started
        vectors = validate_vectors(body, len(texts), arm.expected_dimensions)
        timing = {"wall_seconds": round(elapsed, 6)}
        for key in ("total_duration", "load_duration", "prompt_eval_count"):
            value = body.get(key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ProbeError(f"embedding response has invalid {key}")
                timing[key] = value
        if body.get("model") not in (None, arm.model):
            raise ProbeError("embedding response names a different model")
        return vectors, timing

    async def residency(self) -> list[dict]:
        body = await self.request("GET", "/api/ps")
        if not isinstance(body.get("models"), list):
            raise ProbeError("/api/ps: models must be a list")
        keys = ("name", "model", "digest", "size", "size_vram", "expires_at", "context_length")
        return [{key: model[key] for key in keys if key in model}
                for model in body["models"] if isinstance(model, dict)]

    async def unload(self, model: str) -> None:
        # Empty generate requests implement Ollama's model lifecycle control;
        # this performs no token generation and is valid for embedding tags.
        await self.request("POST", "/api/generate", {
            "model": model, "keep_alive": 0, "stream": False,
        })


def resident_names(rows: list[dict]) -> set[str]:
    return {row.get("name", row.get("model", "")) for row in rows}


async def preflight(client: Ollama, arms: list[Arm]) -> dict:
    version = await client.request("GET", "/api/version")
    if not isinstance(version.get("version"), str) or not version["version"]:
        raise ProbeError("Ollama version is missing")
    tags = await client.request("GET", "/api/tags")
    if not isinstance(tags.get("models"), list):
        raise ProbeError("/api/tags: models must be a list")
    inventory = {row.get("name", row.get("model")): row for row in tags["models"]
                 if isinstance(row, dict)}
    models = {}
    for tag in dict.fromkeys(arm.model for arm in arms):
        if tag not in inventory:
            raise ProbeError(f"required model is not installed: {tag}; this probe never pulls weights")
        show = await client.request("POST", "/api/show", {"model": tag})
        if "embedding" not in show.get("capabilities", []):
            raise ProbeError(f"{tag}: installed model does not advertise embedding capability")
        row, info = inventory[tag], show.get("model_info", {})
        if not isinstance(info, dict):
            raise ProbeError(f"{tag}: invalid model metadata")
        models[tag] = {
            "digest": row.get("digest"), "bytes": row.get("size"),
            "details": row.get("details", {}), "capabilities": show["capabilities"],
            "model_info": {key: value for key, value in info.items()
                           if key == "general.architecture"
                           or key.endswith((".embedding_length", ".context_length"))},
        }
    return {"ollama_version": version["version"], "installed_models": models}


def summarize_rows(rows: list[dict], method: str) -> dict:
    def metrics(selected: list[dict]) -> dict:
        positive = [row for row in selected if row["relevant"]]
        return ranking_metrics([row[method] for row in positive],
                               [tuple(row["relevant"]) for row in positive])
    return {"overall": metrics(rows), "groups": {
        group: metrics([row for row in rows if row["group"] == group])
        for group in sorted(GROUPS)
    }}


async def measure_arm(
    client: Ollama, arm: Arm, documents: list[Document], queries: list[Query],
    *, progress: dict | None = None,
) -> dict:
    if progress is None:
        progress = {}
    progress.update(stage="residency_before", completed_documents=0, completed_query_samples=0)
    before = await client.residency()
    progress["stage"] = "cold_unload"
    await client.unload(arm.model)
    unloaded = await client.residency()
    if arm.model in resident_names(unloaded):
        raise ProbeError(f"{arm.model}: unload did not complete; cold sample unavailable")
    progress["stage"] = "cold_single_query"
    cold_vectors, cold = await client.embed(arm, [arm.query_prefix + queries[0].text])
    norms = [math.hypot(*cold_vectors[0])]
    corpus_vectors, ingest_samples = [], []
    progress["stage"] = "corpus_embeddings"
    ingest_started = time.monotonic()
    for offset in range(0, len(documents), client.settings.batch_size):
        batch = documents[offset:offset + client.settings.batch_size]
        vectors, timing = await client.embed(arm, [arm.document_prefix + doc.text for doc in batch])
        corpus_vectors.extend(vectors)
        progress["completed_documents"] += len(batch)
        ingest_samples.append({"documents": len(batch), **timing})
        norms.extend(math.hypot(*vector) for vector in vectors)
    ingest_seconds = time.monotonic() - ingest_started
    rows, timings = [], []
    progress["stage"] = "warm_queries"
    for repeat in range(client.settings.rounds):
        for query in queries:
            vectors, timing = await client.embed(arm, [arm.query_prefix + query.text])
            query_vector_ = vectors[0]
            norms.append(math.hypot(*query_vector_))
            timings.append(timing["wall_seconds"])
            progress["completed_query_samples"] += 1
            dense = sorted([
                KBHit(cosine(query_vector_, vector), doc.id, "text", 0, doc.text, {})
                for doc, vector in zip(documents, corpus_vectors, strict=True)
            ], key=lambda hit: (-hit.score, hit.source))
            lexical = lexical_rank(documents, query.text)
            hybrid = reciprocal_rank_fusion(dense[:RETRIEVAL_POOL], lexical, query=query.text)
            correct_scores = [hit.score for hit in dense if hit.source in query.relevant]
            wrong_scores = [hit.score for hit in dense if hit.source not in query.relevant]
            rows.append({
                "id": query.id, "repeat": repeat + 1, "group": query.group,
                "relevant": list(query.relevant), "dense": [hit.source for hit in dense],
                "hybrid": [entry.hit.source for entry in hybrid],
                "lexical": [hit.source for hit in lexical],
                "answerable": bool(query.relevant),
                "hybrid_top_evidence": [{"id": entry.hit.source,
                                         "dense_cosine": entry.dense_score,
                                         "lexical_overlap": entry.overlap}
                                        for entry in hybrid[:3]],
                "dense_top_scores": [{"id": hit.source, "cosine": round(hit.score, 6)}
                                     for hit in dense[:3]],
                "dense_relevant_margin": round(max(correct_scores) - max(wrong_scores), 6)
                if correct_scores and wrong_scores else None,
                "query_timing": timing,
            })
    progress["stage"] = "residency_after"
    after = await client.residency()
    progress["stage"] = "completed"
    return {
        "arm": arm.id, "model": arm.model, "status": "measured", "comparable": True,
        "format": {"document_prefix": arm.document_prefix, "query_prefix": arm.query_prefix,
                   "actual_document_titles_embedded": False},
        "dimensions": len(corpus_vectors[0]),
        "vector_norms": {"min": round(min(norms), 8), "max": round(max(norms), 8),
                         "mean": round(statistics.mean(norms), 8)},
        "cold_single_query": {"controlled_unload": True, **cold},
        "warm_single_query": {"samples": len(timings),
                              "p50_seconds": percentile(timings, 0.5),
                              "p95_seconds": percentile(timings, 0.95)},
        "ingestion": {"documents": len(documents), "batch_size": client.settings.batch_size,
                      "seconds": round(ingest_seconds, 6),
                      "documents_per_second": round(len(documents) / ingest_seconds, 4),
                      "batches": ingest_samples, "excludes_loading_sample": True},
        "dense": summarize_rows(rows, "dense"), "hybrid": summarize_rows(rows, "hybrid"),
        "lexical": summarize_rows(rows, "lexical"), "queries": rows,
        "unanswerable_diagnostics": {
            "queries": sum(not row["relevant"] for row in rows),
            "maximum_nearest_dense_cosine": max(
                (row["dense_top_scores"][0]["cosine"] for row in rows if not row["relevant"]),
                default=None,
            ),
            "rejection_accuracy_measured": False,
            "note": "Inspect per-query nearest hits; no model-specific rejection threshold is assumed",
        },
        "residency": {"before": before, "after_unload": unloaded, "after": after,
                      "displaced": sorted(resident_names(before) - resident_names(after))},
    }


async def collect_report(
    settings: Settings, documents: list[Document], queries: list[Query], fixture_sha256: str,
    *, transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    arms = settings.selected_arms()
    calls = len(arms) * (1 + math.ceil(len(documents) / settings.batch_size)
                        + len(queries) * settings.rounds)
    if (not 1 <= len(documents) <= MAX_DOCUMENTS
            or not 1 <= len(queries) <= MAX_QUERIES):
        raise ProbeError(
            f"fixture exceeds bounded counts: documents must be 1..{MAX_DOCUMENTS} "
            f"and queries must be 1..{MAX_QUERIES}"
        )
    if calls > MAX_EMBED_CALLS:
        raise ProbeError(
            f"estimated embedding requests {calls} exceed maximum {MAX_EMBED_CALLS}; "
            "reduce ROUNDS, select fewer ARMS, or use fewer fixture queries"
        )
    client = Ollama(settings, transport)
    try:
        metadata = await preflight(client, arms)
        report = {
            "schema": 1, "status": "measured", "measurement_status": "completed",
            "created_at": datetime.now(UTC).isoformat(), "provenance": metadata,
            "fixture": {"sha256": fixture_sha256, "language": "en", "synthetic": True,
                        "documents": len(documents), "queries": len(queries),
                        "rounds": settings.rounds,
                        "answerable_queries": sum(bool(query.relevant) for query in queries),
                        "unanswerable_queries": sum(not query.relevant for query in queries)},
            "scope": "Synthetic English short-passage retrieval; no production acceptance",
            "settings": {"timeout_seconds": settings.timeout,
                         "keep_alive": settings.keep_alive, "batch_size": settings.batch_size,
                         "rounds": settings.rounds, "selected_arms": [arm.id for arm in arms],
                         "estimated_embedding_requests": calls,
                         "maximum_embedding_requests": MAX_EMBED_CALLS},
            "retrieval": {"cosine": "exact in memory", "hybrid_pool_per_retriever": RETRIEVAL_POOL,
                          "lexical": "existing BM25 TF/tokenizer plus corpus IDF emulation",
                          "fusion": "existing reciprocal_rank_fusion, k=60",
                          "score_thresholds_applied": False, "qdrant_contacted": False},
            "limitations": [
                "Small synthetic fixture: useful screening evidence, not a production quality verdict",
                "English short passages only; no long-context, images, OCR, or private uploads",
                "No concurrency benchmark; run on an otherwise idle box",
                "Cold means unloaded from Ollama, not a cleared filesystem cache or GPU reset",
                "Cosine floors are model-specific; existing admission thresholds are not validated",
                "Synthetic edition/negation distractors test ranking, not year or access-policy enforcement",
                "Unanswerable queries are nearest-hit diagnostics, excluded from positive-label MRR and recall",
                "Switching models requires rebuilding every affected text vector index, including memories",
            ],
            "residency_before": await client.residency(), "results": [], "errors": [],
        }
        for arm in arms:
            progress = {}
            attempted_before = client.embed_calls
            try:
                result = await measure_arm(client, arm, documents, queries, progress=progress)
                report["results"].append(result)
            except ProbeError as exc:
                report["errors"].append({"arm": arm.id, "error": str(exc),
                                         "stage": progress.get("stage", "unknown")})
                report["results"].append({
                    "arm": arm.id, "model": arm.model, "status": "incomplete",
                    "comparable": False, **progress,
                    "embedding_requests_attempted": client.embed_calls - attempted_before,
                    "note": "Partial accuracy omitted; this arm is not comparable",
                })
            finally:
                if arm.model != settings.incumbent:
                    try:
                        await client.unload(arm.model)
                    except ProbeError as exc:
                        report["errors"].append({"arm": arm.id, "cleanup_error": str(exc)})
        try:
            report["residency_after"] = await client.residency()
        except ProbeError as exc:
            report["residency_after"] = None
            report["errors"].append({"residency_error": str(exc)})
        report["embedding_calls"] = client.embed_calls
        if report["errors"]:
            report.update(status="failed", measurement_status="incomplete")
        return report
    finally:
        await client.client.aclose()


def compact_report(report: dict) -> dict:
    result = {key: value for key, value in report.items()
              if key not in {"results", "residency_before", "residency_after"}}
    result["results"] = [{key: value for key, value in arm.items()
                          if key not in {"queries", "residency"}} for arm in report["results"]]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--cases", type=Path, default=default_case_path())
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        settings = Settings(
            base_url=os.environ.get("OLLAMA", "http://ollama:11434"),
            candidate=os.environ.get("CANDIDATE", "embeddinggemma-2:latest"),
            incumbent=os.environ.get("INCUMBENT", "nomic-embed-text:latest"),
            arms=tuple(item.strip() for item in os.environ.get("ARMS", "gemma,prefixed,raw").split(",")),
            rounds=int(os.environ.get("ROUNDS", "1")),
            batch_size=int(os.environ.get("BATCH_SIZE", "8")),
            timeout=float(os.environ.get("TIMEOUT", "120")),
            keep_alive=os.environ.get("KEEP_ALIVE", "10m"),
        )
        path = Path(os.environ["CASE_FILE"]) if "CASE_FILE" in os.environ else args.cases
        documents, queries, digest = load_cases(path)
        report = asyncio.run(collect_report(settings, documents, queries, digest))
    except (ProbeError, ValueError) as exc:
        print(json.dumps({"schema": 1, "status": "setup_error", "error": str(exc)}, indent=2))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"schema": 1, "status": "interrupted"}))
        return 130
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact_report(report) if args.summary else report, indent=2))
    return 0 if report["measurement_status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
