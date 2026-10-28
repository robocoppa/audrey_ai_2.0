# Campaign 3 Phase 23 — Pool-aware synthesis deadline

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Panel workers and the synthesizer resolve their deadline through the same
pool-aware helper. Cloud-only pools use the cloud budget; local-holding pools
use the deep-worker budget. Both graph and streaming call sites honor it.

Primary/fallback/degraded synthesis still follows the existing bounded failure
path. Historical “23a” refers to this phase; unrelated lesson review records
remain in the gitignored course audit.
