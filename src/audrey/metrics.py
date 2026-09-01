"""Prometheus metrics for Audrey.

Each metric is tied to a specific operational question:

  audrey_pipeline_seconds          — fast-path vs deep latency by task type
  audrey_pipeline_total            — fast/deep ratio + outcome counts
  audrey_dispatch_total            — which model is actually being picked
  audrey_model_seconds             — per-model latency (ollama call timing)
  audrey_gpu_gate_wait_seconds     — local-model queue wait time
  audrey_kb_search_seconds         — KB query latency (text/image; merged or not)
  audrey_kb_search_hits            — hits returned per query (zero = retrieval miss)
  audrey_video_describe_seconds    — per-keyframe vision latency (phase 38 input)
  audrey_vision_stage_seconds      — where that latency actually goes, by stage
  audrey_vision_eval_tokens        — description length in tokens, not characters
  audrey_auth_cache_size           — OWUI token cache occupancy
  audrey_user_inflight_blocked_seconds — wait at the per-user concurrency cap
  audrey_inflight_cap_breached_total — soft cap on tracked users exceeded
  audrey_tool_calls_total          — tool dispatches inside ReAct, by outcome
  audrey_tool_call_seconds         — per-tool dispatch latency
  audrey_file_deletion_events_total — durable deletion lifecycle outcomes
  audrey_file_deletion_pending     — tombstones still awaiting full cleanup

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

from typing import Any

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

# ─── Video visual pass (Phase 36) ─────────────────────────────────────

# Buckets run long on purpose. `vision.timeout_s` is 120s because a dense
# screenshot is a slow decode, so a histogram topping out at 10s would report
# every real frame as +Inf and answer nothing.
#
# Unlabelled deliberately. A `user` or `model` label would grow with the user
# base, and the question this exists to answer — "what does a frame actually
# cost, and can keyframes_max go up?" — is about the deployment rather than
# about who uploaded. Phase 38 is the consumer; without this it is guesswork.
_DESCRIBE_BUCKETS = (1, 2, 5, 10, 20, 30, 60, 120, 300)

video_describe_seconds = Histogram(
    "audrey_video_describe_seconds",
    "Wall-clock time to describe one video keyframe through the vl pool.",
    buckets=_DESCRIBE_BUCKETS,
)

# ─── Vision cost attribution (Phase 38) ───────────────────────────────

# Phase 36 measured the wall clock and stopped there: 62.3s per frame with a
# 4x spread. That number sizes the problem but does not point at a fix,
# because every lever phase 38 lists is a bet on which *part* of it is large:
#
#   queue        — waiting on FairLocalGate behind chat. Fixed by scheduling,
#                  or by moving description off the local GPU entirely.
#   load         — the model being evicted between frames. Fixed by keep_alive.
#                  Batching and downscaling do nothing for it.
#   prompt_eval  — the image itself, as tokens. This is the only stage a
#                  smaller frame makes cheaper, and the only one where
#                  batching several frames per call could amortise anything.
#   eval         — the model writing prose. Bounded by how much we asked for,
#                  so the lever is the prompt and num_predict, neither of
#                  which appears in the phase-38 plan's ranking at all.
#
# The stages are disjoint and sum to the wall clock, so one dashboard answers
# "which lever is worth building" instead of four A-B deployments answering it
# one guess at a time.
_STAGE_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)

vision_stage_seconds = Histogram(
    "audrey_vision_stage_seconds",
    "Where a vision call's wall clock goes, by stage.",
    labelnames=("stage",),  # stage ∈ {queue, load, prompt_eval, eval}
    buckets=_STAGE_BUCKETS,
)

# Characters are what the log line reports and what the chunker sees, but
# tokens are what the model is billed in time for. A description that is long
# because the prompt asked for verbatim transcription of a photo with no text
# in it shows up here as tokens spent, and nowhere else.
_EVAL_TOKEN_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)

vision_eval_tokens = Histogram(
    "audrey_vision_eval_tokens",
    "Tokens generated per vision call.",
    buckets=_EVAL_TOKEN_BUCKETS,
)

# ─── Auth cache ───────────────────────────────────────────────────────

auth_cache_size = Gauge(
    "audrey_auth_cache_size",
    "Number of OWUI bearer tokens currently cached.",
)

# ─── Per-user in-flight cap ───────────────────────────────────────────

# Most slot acquires are immediate (0 bucket). Long tail catches users
# parked behind their own concurrent requests.
_INFLIGHT_BUCKETS = (0.0, 0.05, 0.5, 2.0, 10.0, 30.0, 120.0)

user_inflight_blocked_seconds = Histogram(
    "audrey_user_inflight_blocked_seconds",
    "Time waited at the per-user in-flight cap before the request started running.",
    buckets=_INFLIGHT_BUCKETS,
)

inflight_cap_breached_total = Counter(
    "audrey_inflight_cap_breached_total",
    "Times the in-flight registry admitted a new user while at its soft cap "
    "(every tracked user was busy, so no eviction was possible).",
)

# ─── Per-tool dispatch ────────────────────────────────────────────────

# Tool cardinality is bounded (currently 6: kb_search, kb_image_search,
# memory_recall, memory_search, memory_store, web_search). Adding tool
# as a label is safe — no per-user labels here, no risk of explosion.
# Outcome bucketed three ways so timeouts are distinguishable from
# other errors: matters for "is the tool slow?" vs "is the tool broken?"
_TOOL_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

tool_calls_total = Counter(
    "audrey_tool_calls_total",
    "Tool dispatches inside the ReAct loop.",
    labelnames=("tool", "outcome"),  # outcome ∈ {ok, error, timeout}
)

tool_call_seconds = Histogram(
    "audrey_tool_call_seconds",
    "Per-tool dispatch latency (success and failure).",
    labelnames=("tool",),
    buckets=_TOOL_BUCKETS,
)

# ─── Chat archive ────────────────────────────────────────────────────
# Best-effort archive writes must not be silent failures. Outcome
# enumerates the cases the operator actually cares about: ok, partial
# (stream cut short but persisted), fail (HTTP/transport failure),
# deferred (host discovery unavailable), skipped (no usable content).

chat_archive_writes_total = Counter(
    "audrey_chat_archive_writes_total",
    "Chat archive write attempts from Audrey to custom-tools.",
    labelnames=("result",),  # result ∈ {ok, partial, fail, deferred, skipped}
)

chat_archive_write_seconds = Histogram(
    "audrey_chat_archive_write_seconds",
    "Latency of the archive-write call from Audrey's side.",
    buckets=_TOOL_BUCKETS,
)

chat_archive_queue_events_total = Counter(
    "audrey_chat_archive_queue_events_total",
    "Durable chat-archive queue events inside Audrey.",
    labelnames=("result",),
)

chat_archive_queue_depth = Gauge(
    "audrey_chat_archive_queue_depth",
    "Durable chat-archive source rows awaiting delivery.",
)

chat_archive_enqueue_seconds = Histogram(
    "audrey_chat_archive_enqueue_seconds",
    "Latency of the local durable chat-archive enqueue.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ─── Per-user file deletion ─────────────────────────────────────

file_deletion_events_total = Counter(
    "audrey_file_deletion_events_total",
    "Durable per-user file deletion lifecycle events.",
    labelnames=("result",),
)

file_deletion_pending = Gauge(
    "audrey_file_deletion_pending",
    "Durable file-deletion tombstones awaiting complete cleanup.",
)


# ─── Operational readiness ───────────────────────────────────────────
# Every label value comes from a fixed component/capability/queue catalogue.
# No URL, model response, error string, or user identity enters these series.

readiness_state = Gauge(
    "audrey_readiness_state",
    "Current readiness state as a one-hot gauge.",
    labelnames=("state",),
)

readiness_component_available = Gauge(
    "audrey_readiness_component_available",
    "Whether an operational component is currently available.",
    labelnames=("component",),
)

readiness_component_required = Gauge(
    "audrey_readiness_component_required",
    "Whether an operational component is required by deployment policy.",
    labelnames=("component",),
)

readiness_component_enabled = Gauge(
    "audrey_readiness_component_enabled",
    "Whether an operational component is enabled by deployment policy.",
    labelnames=("component",),
)

readiness_tools = Gauge(
    "audrey_readiness_tools",
    "Tool policy, discovery, and availability counts.",
    labelnames=("kind",),
)

readiness_capability_available = Gauge(
    "audrey_readiness_capability_available",
    "Whether a declared tool capability is currently available.",
    labelnames=("capability",),
)

readiness_queue_depth = Gauge(
    "audrey_readiness_queue_depth",
    "Current durable or worker queue depth.",
    labelnames=("queue",),
)

readiness_queue_active = Gauge(
    "audrey_readiness_queue_active",
    "Current active leases in a worker queue.",
    labelnames=("queue",),
)

readiness_queue_available = Gauge(
    "audrey_readiness_queue_available",
    "Whether a queue source is currently readable.",
    labelnames=("queue",),
)

readiness_queue_attempts = Gauge(
    "audrey_readiness_queue_attempts",
    "Accumulated attempts for pending repair work.",
    labelnames=("queue",),
)

readiness_queue_with_error = Gauge(
    "audrey_readiness_queue_with_error",
    "Pending queue rows carrying a sanitized error state.",
    labelnames=("queue",),
)

readiness_queue_exhausted = Gauge(
    "audrey_readiness_queue_exhausted",
    "Pending queue rows requiring operator attention.",
    labelnames=("queue",),
)

readiness_queue_oldest_age_seconds = Gauge(
    "audrey_readiness_queue_oldest_age_seconds",
    "Age of the oldest waiting item, or zero when the queue is empty.",
    labelnames=("queue",),
)

readiness_worker_running = Gauge(
    "audrey_readiness_worker_running",
    "Whether a configured background worker task is running.",
    labelnames=("worker",),
)

readiness_worker_enabled = Gauge(
    "audrey_readiness_worker_enabled",
    "Whether a background worker is enabled by deployment policy.",
    labelnames=("worker",),
)

readiness_worker_last_success_age_seconds = Gauge(
    "audrey_readiness_worker_last_success_age_seconds",
    "Age of a worker last successful activity, or zero when unobserved.",
    labelnames=("worker",),
)

readiness_worker_last_failure_age_seconds = Gauge(
    "audrey_readiness_worker_last_failure_age_seconds",
    "Age of a worker last failed activity, or zero when unobserved.",
    labelnames=("worker",),
)

readiness_pressure = Gauge(
    "audrey_readiness_pressure",
    "Current aggregate scheduler pressure without per-user labels.",
    labelnames=("scheduler", "kind"),
)


def publish_readiness(snapshot: Any) -> None:
    """Mirror one readiness snapshot into bounded Prometheus gauges."""
    for state in ("ready", "degraded", "unready"):
        readiness_state.labels(state=state).set(int(snapshot.status == state))
    for name, component in snapshot.components.items():
        readiness_component_available.labels(component=name).set(
            int(component.status == "available")
        )
        readiness_component_required.labels(component=name).set(
            int(component.required)
        )
        readiness_component_enabled.labels(component=name).set(
            int(component.status != "disabled")
        )
    readiness_tools.labels(kind="policy").set(snapshot.tools.policy_count)
    readiness_tools.labels(kind="discovered").set(snapshot.tools.discovered_count)
    readiness_tools.labels(kind="available").set(snapshot.tools.available_count)
    for capability in snapshot.tools.capabilities:
        readiness_capability_available.labels(capability=capability.name).set(
            int(capability.available)
        )
    for name, queue in snapshot.queues.items():
        readiness_queue_depth.labels(queue=name).set(queue.depth)
        readiness_queue_active.labels(queue=name).set(queue.active)
        readiness_queue_available.labels(queue=name).set(int(queue.available))
        readiness_queue_attempts.labels(queue=name).set(queue.attempts)
        readiness_queue_with_error.labels(queue=name).set(queue.with_error)
        readiness_queue_exhausted.labels(queue=name).set(queue.exhausted)
        readiness_queue_oldest_age_seconds.labels(queue=name).set(
            queue.oldest_age_seconds
        )
    for name, worker in snapshot.workers.items():
        readiness_worker_running.labels(worker=name).set(int(worker.running))
        readiness_worker_enabled.labels(worker=name).set(int(worker.enabled))
        readiness_worker_last_failure_age_seconds.labels(worker=name).set(
            worker.last_failure_age_seconds
        )
        readiness_worker_last_success_age_seconds.labels(worker=name).set(
            worker.last_success_age_seconds
        )
    for kind, value in snapshot.pressure.gpu_gate.model_dump().items():
        readiness_pressure.labels(scheduler="gpu_gate", kind=kind).set(value)
    for kind, value in snapshot.pressure.user_inflight.model_dump().items():
        readiness_pressure.labels(scheduler="user_inflight", kind=kind).set(value)


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
    "video_describe_seconds",
    "vision_stage_seconds",
    "vision_eval_tokens",
    "auth_cache_size",
    "user_inflight_blocked_seconds",
    "inflight_cap_breached_total",
    "tool_calls_total",
    "tool_call_seconds",
    "chat_archive_writes_total",
    "chat_archive_write_seconds",
    "chat_archive_queue_events_total",
    "chat_archive_queue_depth",
    "chat_archive_enqueue_seconds",
    "file_deletion_events_total",
    "file_deletion_pending",
    "publish_readiness",
    "readiness_state",
    "readiness_component_available",
    "readiness_component_required",
    "readiness_component_enabled",
    "readiness_tools",
    "readiness_capability_available",
    "readiness_queue_depth",
    "readiness_queue_active",
    "readiness_queue_available",
    "readiness_queue_attempts",
    "readiness_queue_with_error",
    "readiness_queue_exhausted",
    "readiness_queue_oldest_age_seconds",
    "readiness_worker_running",
    "readiness_worker_enabled",
    "readiness_worker_last_success_age_seconds",
    "readiness_worker_last_failure_age_seconds",
    "readiness_pressure",
]
