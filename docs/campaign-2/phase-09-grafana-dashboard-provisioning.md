# Campaign 2 Phase 09 — Grafana provisioning

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Datasources, dashboard providers and dashboard JSON are versioned under
[`monitoring/grafana`](../../monitoring/grafana/). Stable datasource UIDs keep
dashboard references deterministic; provisioning prevents silent UI-only drift.

The tools dashboard covers dispatch rate, error rate, latency and archive-write
health. Prometheus queries run through the server-side datasource rather than
requiring browser access to the metrics network.
