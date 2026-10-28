# Campaign 3 Phase 38 — Measured video optimization

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Video optimization followed measurement of queue, load, prefill and generation
instead of assuming that frame batching or model warmth would dominate. The
measured visual pass spent most time generating prose. A keyframe-specific prompt
with transcript context reduced unnecessary scene inventory and improved useful
on-screen detail.

On the historical comparison, visual wall time fell from 291.1s to 135.7s and
generation from 254.5s to 98.6s. These are one workload's measurements, not a
general performance guarantee. Output caps must use measured tokens rather than
a guessed character/token ratio; an earlier cap lost frames.

Source retention remains explicit. Reclamation may occur only after dependent
artifacts complete, must preserve quota accounting and must report reclaimed
source state honestly. The current Campaign 4 file contract retains downloaded
originals for playback; this history does not authorize another retention change.
