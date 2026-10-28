# Campaign 3 Phase 27 — On-box evaluation runner

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The standalone evaluation image and
[`eval-onbox.sh`](../../evals/eval-onbox.sh) run the existing harness on Tower's
Docker network so a long run can survive laptop disconnection. Credentials
remain runtime-only and results are persisted outside the disposable container.

Current authentication uses Audrey's direct API with an Audrey personal token.
The old external UI key, hostname and deployment recipes are retired. Use
[`evals/README.md`](../../evals/README.md) and
[the testing protocol](../reference/testing-protocol.md) for current commands,
notification requirements and report transfer to the laptop.
