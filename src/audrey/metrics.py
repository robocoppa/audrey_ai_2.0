"""Prometheus metrics for Audrey (Phase 17).

Eight metrics, each tied to an actual question:

  audrey_pipeline_seconds        — fast-path vs deep latency by task type
  audrey_pipeline_total          — fast/deep ratio + outcome counts
  audrey_dispatch_total          — which model is actually being picked
  audrey_model_seconds           — per-model latency (ollama call timing)
  audrey_gpu_gate_wait_seconds   — local-model queue wait time
  audrey_kb_search_seconds       — KB query latency (text/image; merged or not)
  audrey_kb_search_hits          — hits returned per query (zero = retrieval miss)
  audrey_auth_cache_size         — current Phase 14 token cache size

Cardinality is bounded by design:
  - `model` labels come from the registry (a few dozen at most)
  - `task_type` is one of {code, reasoning, general, vl}
  - `mode`, `path`, `outcome`, `kind`, `had_user_collection` are tiny enums
  - No per-user labels — those would explode cardinality and leak emails

`render()` is the only public surface. It serializes the default
registry to Prometheus text exposition format. The /metrics route hands
the bytes back as `text/plain; version=0.0.4`.

The metrics module is imported once from `main.py`; instrumentation
sites do `from audrey.metrics import pipeline_seconds` etc. and call the
counter/histogram methods inline. If you ever see a duplicated-timeseries
error, that means something imported this twice through a different
package path — find it and fix the import, don't add a unregister hack.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ─── Pipeline ─────────────────────────────────────────────────────────

# Buckets chosen from observed Audrey behavior: fast-path is sub-second
# to ~3s; deep panel is 5-90s. The 0.05 bucket catches caching wins.
_PIPELINE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 45.0, 90.0)

pipeline_seconds = Histogram(
    "audrey_pipeline_seconds",
    "Wall-clock time spent in a pipeline run, end-to-end.",
    labelnames=("mode", "task_type"),
    buckets=_PIPELINE_BUCKETS,
)

pipeline_total = Counter(
    "audrey_pipeline_total",
    "Pipeline runs by mode, task type, and outcome.",
    labelnames=("mode", "task_type", "outcome"),
)

# ─── Dispatch ─────────────────────────────────────────────────────────

dispatch_total = Counter(
    "audrey_dispatch_total",
    "Model dispatches by model name, task type, and dispatch path.",
    labelnames=("model", "task_type", "path"),
)

# ─── Model calls (Ollama) ─────────────────────────────────────────────

# Wider tail than pipeline because cloud cold starts can take ~30s.
_MODEL_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 45.0, 90.0, 180.0)

model_seconds = Histogram(
    "audrey_model_seconds",
    "Wall-clock time for a single model generation call.",
    labelnames=("model", "outcome"),
    buckets=_MODEL_BUCKETS,
)

# ─── GPU gate ─────────────────────────────────────────────────────────

# Sub-second buckets for the common case; longer tail catches contention.
_GATE_BUCKETS = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

gpu_gate_wait_seconds = Histogram(
    "audrey_gpu_gate_wait_seconds",
    "Time spent waiting to acquire the GPU concurrency gate.",
    buckets=_GATE_BUCKETS,
)

# ─── KB search ────────────────────────────────────────────────────────

_KB_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

kb_search_seconds = Histogram(
    "audrey_kb_search_seconds",
    "Wall-clock time for a single KB search (embed + qdrant search + merge).",
    labelnames=("kind", "had_user_collection"),
    buckets=_KB_BUCKETS,
)

# Bucketing on hit counts is unusual but lets us see "how many queries
# returned 0 hits?" alongside "what's the typical hit count?".
_HIT_COUNT_BUCKETS = (0, 1, 2, 5, 10, 20)

kb_search_hits = Histogram(
    "audrey_kb_search_hits",
    "Number of hits returned per KB search.",
    labelnames=("kind",),
    buckets=_HIT_COUNT_BUCKETS,
)

# ─── Auth cache ───────────────────────────────────────────────────────

auth_cache_size = Gauge(
    "audrey_auth_cache_size",
    "Number of OWUI bearer tokens currently cached.",
)


def render() -> tuple[bytes, str]:
    """Serialize the default registry. Returns (body, content_type)."""
    return generate_latest(), CONTENT_TYPE_LATEST


__all__ = [
    "render",
    "pipeline_seconds",
    "pipeline_total",
    "dispatch_total",
    "model_seconds",
    "gpu_gate_wait_seconds",
    "kb_search_seconds",
    "kb_search_hits",
    "auth_cache_size",
]
