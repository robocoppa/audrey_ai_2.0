"""Bounded recurring schedules with explicit UTC and local-clock semantics."""

from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def normalize_schedule(value: dict) -> dict:
    """Validate without I/O; return only the fields understood by the scheduler."""
    if not isinstance(value, dict):
        raise ValueError("Invalid watch schedule.")
    kind, timezone = value.get("kind"), value.get("timezone")
    if not isinstance(kind, str) or kind not in {"interval", "daily", "weekly"}:
        raise ValueError("Invalid watch cadence.")
    if not isinstance(timezone, str) or not 1 <= len(timezone) <= 100:
        raise ValueError("Invalid watch timezone.")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("Invalid watch timezone.") from None
    result = {"kind": kind, "timezone": timezone}
    allowed = {"kind", "timezone"}
    if kind == "interval":
        minutes = value.get("interval_minutes")
        if type(minutes) is not int or not 60 <= minutes <= 10080:
            raise ValueError("Watch intervals must be between 60 and 10080 minutes.")
        result["interval_minutes"] = minutes
        allowed.add("interval_minutes")
    else:
        time = value.get("time")
        if not isinstance(time, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", time):
            raise ValueError("Watch time must use HH:MM.")
        result["time"] = time
        allowed.add("time")
        if kind == "weekly":
            weekday = value.get("weekday")
            if type(weekday) is not int or not 0 <= weekday <= 6:
                raise ValueError("Invalid watch weekday.")
            result["weekday"] = weekday
            allowed.add("weekday")
    if value.keys() - allowed:
        raise ValueError("Unexpected watch schedule fields.")
    return result


def _utc(value: dt.datetime) -> dt.datetime:
    if not isinstance(value, dt.datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Watch scheduling requires an aware timestamp.")
    return value.astimezone(dt.UTC)


def _wall_clock(date: dt.date, hour: int, minute: int, zone: ZoneInfo) -> dt.datetime:
    local = dt.datetime.combine(date, dt.time(hour, minute))
    # Round-tripping distinguishes nonexistent wall clocks from repeated ones.
    # Gaps (including historical date changes) move to the first valid minute.
    for offset in range(2881):
        candidate = local + dt.timedelta(minutes=offset)
        valid = []
        for fold in (0, 1):
            instant = candidate.replace(tzinfo=zone, fold=fold).astimezone(dt.UTC)
            if instant.astimezone(zone).replace(tzinfo=None) == candidate:
                valid.append(instant)
        if valid:
            return min(valid)
    raise ValueError("Could not resolve the scheduled local time.")


def next_occurrence(
    schedule: dict, after: dt.datetime, *, anchor: dt.datetime | None = None,
) -> dt.datetime:
    """Return the first occurrence strictly after ``after``.

    Intervals use elapsed UTC time, retaining cadence when ``anchor`` is supplied.
    Daily/weekly schedules choose the first fold only, even when ``after`` falls
    between two repeated wall clocks. Spring gaps move to the first valid minute.
    """
    schedule, after = normalize_schedule(schedule), _utc(after)
    if schedule["kind"] == "interval":
        period = dt.timedelta(minutes=schedule["interval_minutes"])
        if anchor is None:
            return after + period
        anchor = _utc(anchor)
        return anchor if anchor > after else anchor + ((after - anchor) // period + 1) * period
    zone = ZoneInfo(schedule["timezone"])
    hour, minute = map(int, schedule["time"].split(":"))
    start = after.astimezone(zone).date()
    for offset in range(16):
        date = start + dt.timedelta(days=offset)
        if schedule["kind"] == "weekly" and date.weekday() != schedule["weekday"]:
            continue
        candidate = _wall_clock(date, hour, minute, zone)
        if candidate > after:
            return candidate
    raise ValueError("Could not resolve the next scheduled occurrence.")


def preview_schedule(schedule: dict, after: dt.datetime, count: int = 3) -> list[str]:
    if type(count) is not int or not 1 <= count <= 10:
        raise ValueError("Invalid watch preview count.")
    result = []
    for _ in range(count):
        after = next_occurrence(schedule, after)
        result.append(after.isoformat())
    return result
