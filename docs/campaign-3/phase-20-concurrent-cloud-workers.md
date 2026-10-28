# Campaign 3 Phase 20 — Concurrent cloud drafts

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Default deep text pools can gather one local draft and two independent cloud
drafts concurrently. Cloud calls bypass the local GPU gate but honor their own
global concurrency cap. Planner subtask capacity and pool worker capacity agree.

The intended benefit is draft diversity while latency remains bounded by the
slowest worker. Actual vendors, models and caps belong to `config.yaml`; historical
pool names here are not a current model inventory.
