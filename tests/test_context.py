"""Hermetic tests for `pipeline/context.py`.

The module is two functions — `iso_now()` and `datetime_system_message()`.
The first formats current local time; the second wraps it in a system
message. These tests pin the *shape* (return types, ISO-8601 format,
timezone offset present, system-message role + content phrasing) rather
than the value, which depends on wall-clock and the host timezone.
"""

from __future__ import annotations

import datetime as _dt
import re

from audrey.pipeline.context import datetime_system_message, iso_now


def test_iso_now_returns_string_with_seconds_precision():
    s = iso_now()
    assert isinstance(s, str)
    # Seconds precision: HH:MM:SS followed by tz offset, no fractional
    # seconds. Pinning the whole tail rules out future drift to
    # microsecond precision (which would bloat every log line).
    assert re.search(r"T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2})$", s) is not None


def test_iso_now_parses_as_aware_datetime():
    parsed = _dt.datetime.fromisoformat(iso_now())
    assert parsed.tzinfo is not None


def test_datetime_system_message_shape():
    msg = datetime_system_message()
    assert msg["role"] == "system"
    assert "Current server date and time:" in msg["content"]
    # The timestamp from iso_now() should be substring of the content.
    # We can't pin the value (it ticks), but we can pin that there's
    # an ISO-shaped substring after the colon.
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", msg["content"])


def test_datetime_system_message_includes_treat_as_present_phrasing():
    # The phrasing tells the model what the timestamp *means*, which
    # is what makes it reduce hedging about "today." Pin the load-bearing
    # phrase so a future refactor doesn't silently weaken it.
    msg = datetime_system_message()
    assert "present moment" in msg["content"]
