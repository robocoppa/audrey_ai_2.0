"""UTC intervals and timezone wall clocks, including DST gaps and folds."""

import datetime as dt

import pytest

from audrey.watch_schedule import next_occurrence, normalize_schedule, preview_schedule


def _time(value):
    return dt.datetime.fromisoformat(value)


def test_interval_uses_elapsed_time_and_preserves_anchor_after_downtime():
    schedule = {"kind": "interval", "timezone": "America/Denver", "interval_minutes": 60}
    assert next_occurrence(schedule, _time("2026-03-08T08:30:00+00:00")) == _time("2026-03-08T09:30:00+00:00")
    assert next_occurrence(schedule, _time("2026-03-08T12:45:00+00:00"), anchor=_time(
        "2026-03-08T08:30:00+00:00",
    )) == _time("2026-03-08T13:30:00+00:00")


def test_nonexistent_daily_time_moves_to_first_valid_minute():
    schedule = {"kind": "daily", "timezone": "America/Denver", "time": "02:30"}
    assert next_occurrence(schedule, _time("2026-03-08T00:00:00+00:00")) == _time("2026-03-08T09:00:00+00:00")
    assert next_occurrence(schedule, _time("2026-03-08T09:00:00+00:00")) == _time("2026-03-09T08:30:00+00:00")


def test_repeated_daily_time_runs_only_first_fold():
    schedule = {"kind": "daily", "timezone": "America/Denver", "time": "01:30"}
    assert next_occurrence(schedule, _time("2026-11-01T00:00:00+00:00")) == _time("2026-11-01T07:30:00+00:00")
    # Between the first and second 01:30, the next run is tomorrow.
    assert next_occurrence(schedule, _time("2026-11-01T08:00:00+00:00")) == _time("2026-11-02T08:30:00+00:00")


def test_weekly_local_day_and_half_hour_dst_gap():
    weekly = {"kind": "weekly", "timezone": "America/Denver", "time": "09:00", "weekday": 0}
    assert next_occurrence(weekly, _time("2026-10-30T12:00:00+00:00")) == _time("2026-11-02T16:00:00+00:00")
    half_hour = {"kind": "daily", "timezone": "Australia/Lord_Howe", "time": "02:10"}
    assert next_occurrence(half_hour, _time("2026-10-03T12:00:00+00:00")) == _time("2026-10-03T15:30:00+00:00")


def test_preview_is_future_bounded_and_does_not_mutate_input():
    schedule = {"kind": "daily", "timezone": "UTC", "time": "12:00"}
    assert preview_schedule(schedule, _time("2026-10-30T12:00:00+00:00"), 2) == [
        "2026-10-31T12:00:00+00:00", "2026-11-01T12:00:00+00:00",
    ]
    assert schedule == {"kind": "daily", "timezone": "UTC", "time": "12:00"}
    with pytest.raises(ValueError):
        preview_schedule(schedule, dt.datetime(2026, 10, 30))
    with pytest.raises(ValueError):
        preview_schedule(schedule, dt.datetime.now(dt.UTC), True)


@pytest.mark.parametrize("value", [
    None, [], {}, {"kind": []}, {"kind": {}},
    {"kind": "daily", "timezone": "../UTC", "time": "12:00"},
    {"kind": "daily", "timezone": "UTC\x00", "time": "12:00"},
    {"kind": "daily", "timezone": "UTC", "time": "24:00"},
    {"kind": "daily", "timezone": "UTC", "time": "1:00"},
    {"kind": "daily", "timezone": "UTC", "time": "12:00", "extra": 1},
    {"kind": "weekly", "timezone": "UTC", "time": "12:00", "weekday": True},
    {"kind": "weekly", "timezone": "UTC", "time": "12:00", "weekday": 7},
    {"kind": "interval", "timezone": "UTC", "interval_minutes": True},
    {"kind": "interval", "timezone": "UTC", "interval_minutes": 59},
    {"kind": "interval", "timezone": "UTC", "interval_minutes": 10081},
])
def test_hostile_or_unbounded_schedules_fail_as_value_errors(value):
    with pytest.raises(ValueError):
        normalize_schedule(value)
