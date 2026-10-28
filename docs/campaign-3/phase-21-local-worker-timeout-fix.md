# Campaign 3 Phase 21 — Local worker deadlines

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Local workers serialize through the fair GPU gate and may need to load different
models. Their deadline must allow the intended cold-load/generation behavior;
adding workers is not parallel GPU capacity. Pool ordering affects which local
model loads first.

The phase adjusted local pools and worker headroom. Further timeout growth is
not a substitute for measuring capacity or reducing a serial pool.
