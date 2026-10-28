# Campaign 2 Phase 03 — Lesson citation checking

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

[`check-lesson-links.py`](../../scripts/lessons/check-lesson-links.py) checks
source citations against neighboring code snippets or declaration landmarks.
It reports missing paths, out-of-range lines and likely drift, and proposes
nearby replacements where an anchor is recognizable. It does not rewrite files.

Filtered and list-only modes support targeted maintenance. Broad lesson-link
sweeps remain user-directed; the historical automatic sweep rule is retired.
