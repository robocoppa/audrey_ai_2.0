# Campaign 3 Phase 33 — Leased media jobs

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Audrey owns metadata writes. Workers poll for the oldest pending job and receive
a lease; they return complete/fail results over service-authenticated HTTP rather
than writing the application database. Completion must still hold the same lease.

Expired leases become eligible for retry, with a bounded attempt count turning
poison jobs into terminal failures. Failure reasons are visible. Requeue clears
old derived search artifacts before replacement work and refuses to claim a
successful reset if deletion failed. Source availability constrains requeue.
