# Campaign 2 Phase 14 — Fleet watchdog boundary

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Fleet watchdog is a separate project:
[robocoppa/fleet-watchdog](https://github.com/robocoppa/fleet-watchdog).
Its own README and deployment guide are authoritative; Audrey does not own its
registry, secrets, sender setup or deployment lifecycle.

The design uses pushed heartbeats and edge-triggered alerts for stale senders,
token/send failures and optional backend reachability. Transient machines have
explicit expected-up policy. A real send probe proves more than token validation.
Sender scripts always exit zero for scheduler compatibility and log failures to
stderr. The duplicated host-by-host setup log has been removed.
