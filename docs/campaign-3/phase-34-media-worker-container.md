# Campaign 3 Phase 34 — Isolated media worker

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The worker isolates long CPU media processing from chat serving. It reads source
files through a read-only mount and reaches models only through Audrey's narrow
service route, preserving fair scheduling and authentication. It has no direct
Ollama address or general egress path.

ffmpeg extracts artifacts without rewriting the original. Shutdown waits are
interruptible and allow an active job to drain within the configured grace period.
An inherited environment file must not reintroduce forbidden model connectivity.
