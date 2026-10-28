# Campaign 2 Phase 10 — Global KB reconciliation

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Global reconciliation runs at startup and periodically, and has an authenticated
administrative trigger. It sweeps global text/image collections; per-user
upload collections are managed separately. The admin route returns the
reconciliation result and records who triggered it.

The recurring reconciliation contract began in Campaign 1; this phase completed
its documentation and route coverage.
