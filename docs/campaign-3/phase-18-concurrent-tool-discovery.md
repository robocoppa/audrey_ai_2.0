# Campaign 3 Phase 18 — Concurrent tool discovery

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Configured tool servers are discovered concurrently. Results are folded in
configuration order so completion timing cannot change duplicate-name precedence.
One unreachable server must not discard other servers' successful discovery.

Campaign 4 adds recovery, serialized refresh and explicit dependency states;
those later contracts govern registry replacement during partial outages.
