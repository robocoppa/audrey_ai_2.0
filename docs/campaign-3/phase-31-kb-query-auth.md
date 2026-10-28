# Campaign 3 Phase 31 — KB query authentication

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

KB text/image queries accept either an internal service token or authenticated
user credentials. The internal `X-Audrey-Service-Token` path may act for the
explicitly supplied user; ordinary bearer callers are pinned to their own
authenticated identity and cannot override it in the body.

A blank/mismatched service token fails closed. The tool sidecar supplies the
service credential for internal calls, which do not carry browser credentials.
Adding browser-only authentication to that path would break pipeline retrieval.
The tools server remains an internal service rather than a public API.
