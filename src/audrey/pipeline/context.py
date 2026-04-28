"""Context injection — current date/time as a system message.

Every pipeline run gets a system message at the top with the current
server-side date and time in ISO-8601 format, so models always know
what "today" means. Without this, a model trained on data from a year
ago will hedge dates ("as of 2024-2025") even when the user is asking
about right now.

Coverage:
- The graph runs `node_datetime` first, before memory_recall — every
  non-streaming request gets the message.
- The streaming deep route bypasses the graph for orchestration, so
  it calls `datetime_system_message()` directly inside `_phase_thinking`.
- The streaming fast path goes through the graph too (via the
  tool-capable branch), so it's covered.

Server-side only. The OWUI frontend can also inject `{{CURRENT_DATETIME}}`
via its template-variable system; that adds the *user's local* time,
which is complementary. Both pieces matter:
  - Server time covers programmatic clients (curl, scripts, the
    prompt-suggestion JSON) that don't go through OWUI.
  - User-local time covers timezone-sensitive questions ("what's open
    near me right now?") that need the browser's wall clock.

The system message itself is intentionally short and machine-readable.
Models do better with `2026-04-27T14:32:00-07:00` than
"It is the afternoon of Monday, April twenty-seventh, in the year
twenty twenty-six."
"""

from __future__ import annotations

import datetime as _dt
from typing import Any


def iso_now() -> str:
    """Return current local time as a timezone-aware ISO-8601 string.

    Uses `astimezone()` with no arg, which picks up the system's local
    timezone (whatever `TZ=` resolves to inside the container, falling
    back to the host's tzdata). On Unraid, this is typically the array's
    configured timezone — same as what the user sees in the UI.
    """
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def datetime_system_message() -> dict[str, Any]:
    """Build the system message that gets prepended to every request.

    Single-line content so it doesn't crowd out the user's actual prompt
    in the model's context window. Phrasing tells the model what the
    timestamp *means* so it doesn't try to reason about it as data —
    e.g. "this is when the request arrived, use it as the present."
    """
    return {
        "role": "system",
        "content": (
            f"Current server date and time: {iso_now()}. "
            "Treat this as the present moment when reasoning about "
            "dates, recency, or time-sensitive facts."
        ),
    }


__all__ = ["iso_now", "datetime_system_message"]
