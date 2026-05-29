"""Shared scheduling constants.

Both the fair GPU gate (`pipeline/fair_gate.py`) and the per-user
in-flight cap (`routes/inflight.py`) bucket anonymous traffic under
the same synthetic key. Defining it once here avoids two sources of
truth.
"""

from __future__ import annotations

ANON_USER_BUCKET = "__anon__"

__all__ = ["ANON_USER_BUCKET"]
